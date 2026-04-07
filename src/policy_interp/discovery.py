"""Bottom up module discovery and validity evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score, roc_auc_score

from policy_interp.io import read_jsonl, read_parquet, write_parquet
from policy_interp.schemas import ExperimentConfig
from policy_interp.utils import set_seed

try:
    import igraph as ig
    import leidenalg
except ImportError:
    ig = None
    leidenalg = None


@dataclass(slots=True)
class DiscoveryArtifacts:
    module_candidates_path: Path
    module_stability_path: Path
    module_membership_path: Path
    module_proxy_alignment_path: Path


def run_module_discovery(config: ExperimentConfig) -> DiscoveryArtifacts:
    set_seed(config.splits.seed)
    extraction_root = config.run_root / "extraction"
    prepared_segments = read_parquet(config.run_root / config.dataset.prepared_segments_name)
    candidate_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []

    for layer in config.extract.layers:
        feature_summary = read_parquet(extraction_root / f"feature_summary_layer_{layer}.parquet")
        top_features = read_parquet(extraction_root / f"segment_top_features_layer_{layer}.parquet")
        context_records = read_jsonl(extraction_root / f"feature_top_contexts_layer_{layer}.jsonl")
        pair_weights, feature_activity = _compute_pairwise_jaccard(top_features)
        candidates = _discover_candidates_for_layer(
            layer=layer,
            pair_weights=pair_weights,
            feature_activity=feature_activity,
            feature_summary=feature_summary,
            context_records=context_records,
            config=config,
        )
        candidate_rows.extend(candidates["candidates"])
        membership_rows.extend(candidates["memberships"])
        stable_modules = _compute_stability_for_layer(candidates["candidates"], config)
        stability_rows.extend(stable_modules)

        stable_df = pd.DataFrame(stable_modules)
        if not stable_df.empty:
            layer_alignment = _align_modules_to_proxies(
                stable_modules=stable_df,
                top_features=top_features,
                segments=prepared_segments,
                config=config,
            )
            alignment_rows.extend(layer_alignment.to_dict(orient="records"))

    candidates_frame = pd.DataFrame(candidate_rows)
    stability_frame = pd.DataFrame(stability_rows)
    membership_frame = pd.DataFrame(membership_rows)
    alignment_frame = pd.DataFrame(alignment_rows)

    discovery_root = config.run_root / "discovery"
    discovery_root.mkdir(parents=True, exist_ok=True)
    candidates_path = discovery_root / "module_candidates.parquet"
    stability_path = discovery_root / "module_stability.parquet"
    membership_path = discovery_root / "module_membership.parquet"
    alignment_path = discovery_root / "module_proxy_alignment.parquet"
    write_parquet(candidates_frame, candidates_path)
    write_parquet(stability_frame, stability_path)
    write_parquet(membership_frame, membership_path)
    write_parquet(alignment_frame, alignment_path)

    return DiscoveryArtifacts(
        module_candidates_path=candidates_path,
        module_stability_path=stability_path,
        module_membership_path=membership_path,
        module_proxy_alignment_path=alignment_path,
    )


def _compute_pairwise_jaccard(top_features: pd.DataFrame) -> tuple[dict[tuple[int, int], float], dict[int, int]]:
    grouped = top_features.groupby("segment_id")["feature_id"].apply(lambda col: sorted(set(col.tolist())))
    feature_activity: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()

    for features in grouped.tolist():
        for feature in features:
            feature_activity[feature] += 1
        for left, right in combinations(features, 2):
            pair_counts[(left, right)] += 1

    weights: dict[tuple[int, int], float] = {}
    for (left, right), overlap in pair_counts.items():
        union = feature_activity[left] + feature_activity[right] - overlap
        if union <= 0:
            continue
        weights[(left, right)] = overlap / union
    return weights, dict(feature_activity)


def _discover_candidates_for_layer(
    layer: int,
    pair_weights: dict[tuple[int, int], float],
    feature_activity: dict[int, int],
    feature_summary: pd.DataFrame,
    context_records: list[dict[str, object]],
    config: ExperimentConfig,
) -> dict[str, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    memberships: list[dict[str, object]] = []
    feature_bins = _build_frequency_bins(feature_summary, config.validity.enrichment_match_bins)
    raw_p_values: list[float] = []

    for threshold in config.validity.edge_thresholds:
        graph = nx.Graph()
        for feature_id, count in feature_activity.items():
            if count > 0:
                graph.add_node(feature_id)
        for (left, right), weight in pair_weights.items():
            if weight >= threshold:
                graph.add_edge(left, right, weight=weight)

        for resolution in config.validity.resolutions:
            communities = _run_community_detection(
                graph=graph,
                resolution=resolution,
                seed=config.splits.seed,
                method=config.validity.community_method,
            )
            module_index = 0
            for members in communities:
                member_list = sorted(int(item) for item in members)
                if not (config.validity.min_module_size <= len(member_list) <= config.validity.max_module_size):
                    continue
                module_index += 1
                candidate_id = f"layer_{layer}_thr_{threshold:.2f}_res_{resolution:.2f}_m_{module_index:03d}"
                internal_mean = _internal_pairwise_mean(member_list, pair_weights)
                p_value = _permutation_p_value(
                    module_members=member_list,
                    pair_weights=pair_weights,
                    feature_bins=feature_bins,
                    random_trials=config.validity.enrichment_random_trials,
                    seed=config.splits.seed + layer + module_index,
                )
                raw_p_values.append(p_value)
                contexts = [record for record in context_records if int(record["feature_id"]) in member_list]
                top_ngrams = _top_ngrams_from_contexts(
                    [str(record["context_text"]) for record in contexts],
                    top_k=config.discovery.top_ngrams_per_module,
                    ngram_range=config.discovery.ngram_range,
                )
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "layer": layer,
                        "edge_threshold": threshold,
                        "resolution": resolution,
                        "module_size": len(member_list),
                        "feature_ids": member_list,
                        "internal_pairwise_mean": internal_mean,
                        "raw_p_value": p_value,
                        "top_ngrams": top_ngrams,
                    }
                )
                for feature_id in member_list:
                    memberships.append(
                        {
                            "candidate_id": candidate_id,
                            "layer": layer,
                            "feature_id": feature_id,
                        }
                    )

    q_values = _benjamini_hochberg(raw_p_values)
    for row, q_value in zip(rows, q_values):
        row["q_value"] = q_value
        row["passes_fdr"] = q_value <= config.validity.enrichment_alpha
    return {"candidates": rows, "memberships": memberships}


def _build_frequency_bins(feature_summary: pd.DataFrame, num_bins: int) -> dict[int, list[int]]:
    frequencies = feature_summary["activation_frequency"].to_numpy(dtype=float)
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(frequencies, quantiles)
    feature_bins: dict[int, list[int]] = defaultdict(list)
    for feature_id, frequency in zip(feature_summary["feature_id"].tolist(), frequencies.tolist()):
        bin_index = int(np.searchsorted(edges, frequency, side="right") - 1)
        bin_index = max(0, min(num_bins - 1, bin_index))
        feature_bins[bin_index].append(int(feature_id))
    return feature_bins


def _internal_pairwise_mean(module_members: list[int], pair_weights: dict[tuple[int, int], float]) -> float:
    values: list[float] = []
    for left, right in combinations(sorted(module_members), 2):
        values.append(pair_weights.get((left, right), 0.0))
    if not values:
        return 0.0
    return float(np.mean(values))


def _permutation_p_value(
    module_members: list[int],
    pair_weights: dict[tuple[int, int], float],
    feature_bins: dict[int, list[int]],
    random_trials: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    observed = _internal_pairwise_mean(module_members, pair_weights)
    histogram = Counter(_feature_bin_for_member(member, feature_bins) for member in module_members)
    null_values: list[float] = []
    for _ in range(random_trials):
        sampled: list[int] = []
        for bin_index, count in histogram.items():
            population = feature_bins[bin_index]
            replace = len(population) < count
            sampled.extend(rng.choice(population, size=count, replace=replace).tolist())
        sampled = sampled[: len(module_members)]
        null_values.append(_internal_pairwise_mean(sorted(set(sampled)), pair_weights))
    null_array = np.asarray(null_values, dtype=float)
    return float((1.0 + np.sum(null_array >= observed)) / (1.0 + null_array.size))


def _feature_bin_for_member(member: int, feature_bins: dict[int, list[int]]) -> int:
    for bin_index, members in feature_bins.items():
        if member in members:
            return bin_index
    return 0


def _top_ngrams_from_contexts(contexts: list[str], top_k: int, ngram_range: tuple[int, int]) -> list[str]:
    if not contexts:
        return []
    vectorizer = CountVectorizer(stop_words="english", ngram_range=ngram_range, max_features=500)
    try:
        matrix = vectorizer.fit_transform(contexts)
    except ValueError:
        return []
    if matrix.shape[1] == 0:
        return []
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    top_indices = counts.argsort()[::-1][:top_k]
    return vocabulary[top_indices].tolist()


def _run_community_detection(
    graph: nx.Graph,
    resolution: float,
    seed: int,
    method: str,
) -> list[set[int]]:
    if graph.number_of_nodes() == 0:
        return []
    if method == "leiden" and ig is not None and leidenalg is not None:
        return _run_leiden_with_igraph(graph, resolution, seed)
    if method == "leiden":
        warnings.warn(
            "Leiden backend not available; falling back to NetworkX Louvain communities.",
            RuntimeWarning,
            stacklevel=2,
        )
    return list(
        nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=seed,
        )
    )


def _run_leiden_with_igraph(graph: nx.Graph, resolution: float, seed: int) -> list[set[int]]:
    assert ig is not None
    assert leidenalg is not None
    nodes = sorted(int(node) for node in graph.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes)}
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(nodes))
    ig_graph.vs["feature_id"] = nodes
    edges = [(node_index[int(left)], node_index[int(right)]) for left, right in graph.edges()]
    ig_graph.add_edges(edges)
    ig_graph.es["weight"] = [float(graph.edges[left, right].get("weight", 1.0)) for left, right in graph.edges()]
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=ig_graph.es["weight"],
        resolution_parameter=resolution,
        seed=seed,
    )
    communities: list[set[int]] = []
    for membership in partition:
        feature_ids = {int(ig_graph.vs[index]["feature_id"]) for index in membership}
        communities.append(feature_ids)
    return communities


def _compute_stability_for_layer(candidate_rows: list[dict[str, object]], config: ExperimentConfig) -> list[dict[str, object]]:
    if not candidate_rows:
        return []
    frame = pd.DataFrame(candidate_rows)
    frame = frame.loc[frame["passes_fdr"]].copy()
    canonical = frame.loc[
        (frame["edge_threshold"] == config.validity.canonical_threshold)
        & (frame["resolution"] == config.validity.canonical_resolution)
    ].copy()
    if canonical.empty:
        return []
    stable_rows: list[dict[str, object]] = []
    layer = int(canonical["layer"].iloc[0])
    candidate_lookup = {
        row["candidate_id"]: set(row["feature_ids"])
        for row in frame.to_dict(orient="records")
    }
    all_rows = frame.to_dict(orient="records")
    for stable_index, canonical_row in enumerate(all_rows_for_setting(canonical), start=1):
        canonical_features = candidate_lookup[canonical_row["candidate_id"]]
        presence = 1
        matches: list[dict[str, object]] = []
        for other_row in all_rows:
            if other_row["candidate_id"] == canonical_row["candidate_id"]:
                continue
            if other_row["edge_threshold"] == canonical_row["edge_threshold"] and other_row["resolution"] == canonical_row["resolution"]:
                continue
            other_features = candidate_lookup[other_row["candidate_id"]]
            jaccard = _set_jaccard(canonical_features, other_features)
            matches.append({"candidate_id": other_row["candidate_id"], "jaccard": jaccard})
        grouped_best = _best_match_per_setting(matches, frame)
        for match in grouped_best:
            if match["jaccard"] >= config.validity.min_best_match_jaccard:
                presence += 1
        stable_rows.append(
            {
                "stable_module_id": f"layer_{layer}_stable_{stable_index:03d}",
                "layer": layer,
                "canonical_candidate_id": canonical_row["candidate_id"],
                "feature_ids": sorted(canonical_features),
                "module_size": canonical_row["module_size"],
                "internal_pairwise_mean": canonical_row["internal_pairwise_mean"],
                "q_value": canonical_row["q_value"],
                "presence_count": presence,
                "stable": presence >= config.validity.min_presence_configs,
                "top_ngrams": canonical_row["top_ngrams"],
            }
        )
    return stable_rows


def all_rows_for_setting(frame: pd.DataFrame) -> list[dict[str, object]]:
    return frame.to_dict(orient="records")


def _best_match_per_setting(matches: list[dict[str, object]], frame: pd.DataFrame) -> list[dict[str, object]]:
    settings: dict[tuple[float, float], dict[str, object]] = {}
    setting_lookup = frame.set_index("candidate_id")[["edge_threshold", "resolution"]].to_dict(orient="index")
    for match in matches:
        key = (
            float(setting_lookup[match["candidate_id"]]["edge_threshold"]),
            float(setting_lookup[match["candidate_id"]]["resolution"]),
        )
        if key not in settings or match["jaccard"] > settings[key]["jaccard"]:
            settings[key] = match
    return list(settings.values())


def _set_jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _align_modules_to_proxies(
    stable_modules: pd.DataFrame,
    top_features: pd.DataFrame,
    segments: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    feature_scores = top_features.groupby(["segment_id", "feature_id"])["pooled_activation"].max().reset_index()
    proxy_rows: list[dict[str, object]] = []
    metadata = segments[["segment_id", "split", "segment_validated", *config.dataset.proxy_columns.keys()]].drop_duplicates()
    for module in stable_modules.itertuples(index=False):
        if not bool(module.stable):
            continue
        members = set(module.feature_ids)
        module_scores = (
            feature_scores.loc[feature_scores["feature_id"].isin(members)]
            .groupby("segment_id")["pooled_activation"]
            .mean()
            .rename("module_score")
            .reset_index()
        )
        scored = metadata.merge(module_scores, on="segment_id", how="left").fillna({"module_score": 0.0})
        for proxy_key in config.dataset.proxy_columns.keys():
            train_frame = scored.loc[scored["split"] == "train"].copy()
            dev_frame = scored.loc[scored["split"] == "dev"].copy()
            test_frame = scored.loc[scored["split"] == "test"].copy()
            validated_frame = scored.loc[scored["split"] == "validated"].copy()
            if train_frame[proxy_key].nunique() < 2:
                train_auc = np.nan
            else:
                train_auc = roc_auc_score(train_frame[proxy_key].astype(int), train_frame["module_score"])
            if dev_frame[proxy_key].nunique() < 2:
                dev_auc = np.nan
            else:
                dev_auc = roc_auc_score(dev_frame[proxy_key].astype(int), dev_frame["module_score"])
            if test_frame[proxy_key].nunique() < 2:
                test_auc = np.nan
            else:
                test_auc = roc_auc_score(test_frame[proxy_key].astype(int), test_frame["module_score"])
            mi = mutual_info_score(
                scored[proxy_key].astype(int),
                (scored["module_score"] > 0).astype(int),
            )
            if validated_frame[proxy_key].nunique() < 2:
                validated_auc = np.nan
            else:
                validated_auc = roc_auc_score(
                    validated_frame[proxy_key].astype(int),
                    validated_frame["module_score"],
                )
            proxy_rows.append(
                {
                    "stable_module_id": module.stable_module_id,
                    "layer": module.layer,
                    "proxy": proxy_key,
                    "train_auc": train_auc,
                    "dev_auc": dev_auc,
                    "test_auc": test_auc,
                    "mutual_information": mi,
                    "validated_auc": validated_auc,
                }
            )
    return pd.DataFrame(proxy_rows)


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    pvals = np.asarray(p_values, dtype=float)[order]
    n = len(p_values)
    qvals = np.empty(n, dtype=float)
    min_coeff = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        coeff = pvals[idx] * n / rank
        min_coeff = min(min_coeff, coeff)
        qvals[idx] = min_coeff
    output = np.empty(n, dtype=float)
    output[order] = np.clip(qvals, 0.0, 1.0)
    return output.tolist()
