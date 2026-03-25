from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from _common import save_with_metadata, setup_logging
from data.io import read_jsonl, write_jsonl
from data.matching import MatchConfig, greedy_match_rows, pairwise_text_similarity, try_encode_texts_with_fallback


def _load_proxy_name(proxy_dir: Path, split: str) -> str:
    rows = read_jsonl(proxy_dir / f"{split}.jsonl")
    for row in rows:
        name = row.get("proxy_name")
        if name:
            return str(name)
    raise ValueError(f"Unable to resolve proxy_name for {proxy_dir}")


def _eligible_candidates(
    eligible_rows: list[dict],
    *,
    proxy_name: str,
    positive_segment_ids: set[str],
) -> list[dict]:
    candidates: list[dict] = []
    for row in eligible_rows:
        if row["segment_id"] in positive_segment_ids:
            continue
        row_tags = set(row.get("all_tags", []) or row.get("tags", []))
        if proxy_name in row_tags:
            continue
        candidates.append(row)
    return candidates


def _match_split(
    positives: list[dict],
    candidates: list[dict],
    *,
    config: MatchConfig,
    device: str,
) -> tuple[list[dict], dict]:
    if not positives:
        return [], {"n_positives": 0, "n_candidates": len(candidates), "n_matched": 0}
    all_texts = [row["text"] for row in positives] + [row["text"] for row in candidates]
    vectors, method_used = try_encode_texts_with_fallback(all_texts, config=config, device=device)
    pos_vectors = vectors[: len(positives)]
    cand_vectors = vectors[len(positives) :]
    sim = pairwise_text_similarity(pos_vectors, cand_vectors)
    matched_rows, diag = greedy_match_rows(positives, candidates, similarity_matrix=sim, config=config)
    diag["encoding_method_used"] = method_used
    return matched_rows, diag


def _materialize_split_set(
    *,
    proxy_dir: Path,
    eligible_dir: Path,
    out_dir: Path,
    proxy_name: str,
    config: MatchConfig,
    device: str,
) -> dict[str, dict]:
    split_diagnostics: dict[str, dict] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        positives = read_jsonl(proxy_dir / f"{split}.jsonl")
        eligible_rows = read_jsonl(eligible_dir / f"{split}.jsonl")
        positive_ids = {row["segment_id"] for row in positives}
        candidates = _eligible_candidates(
            eligible_rows,
            proxy_name=proxy_name,
            positive_segment_ids=positive_ids,
        )
        matched_rows, diag = _match_split(positives, candidates, config=config, device=device)
        for row in matched_rows:
            row["negative_for_proxy_name"] = proxy_name
            row["negative_for_proxy_slug"] = proxy_dir.name
        write_jsonl(out_dir / f"{split}.jsonl", matched_rows)
        split_diagnostics[split] = diag
        logging.info(
            "%s | split=%s positives=%d candidates=%d matched=%d",
            proxy_dir,
            split,
            len(positives),
            len(candidates),
            len(matched_rows),
        )
    return split_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_root", default="data/processed/public_values")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--method", default="sentence", choices=["sentence", "tfidf"])
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow_replacement", action="store_true")
    args = parser.parse_args()

    setup_logging("build_matched_negatives")
    manifest_root = Path(args.manifest_root)
    eligible_dir = manifest_root / "eligible"
    validated_eligible_dir = manifest_root / "validated"
    if not eligible_dir.exists():
        raise FileNotFoundError(f"Missing eligible manifests: {eligible_dir}")

    config = MatchConfig(
        method=args.method,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        allow_replacement=args.allow_replacement,
    )

    selected_families = set(args.families) if args.families else None
    summary: dict[str, dict] = {}

    for family_dir in sorted(path for path in manifest_root.iterdir() if path.is_dir() and (path / "proxies").exists()):
        family_name = family_dir.name
        if selected_families is not None and family_name not in selected_families:
            continue
        summary[family_name] = {"proxies": {}, "validated_proxies": {}}
        for proxy_dir in sorted((family_dir / "proxies").iterdir()):
            if not proxy_dir.is_dir():
                continue
            proxy_name = _load_proxy_name(proxy_dir, "train")
            diagnostics = _materialize_split_set(
                proxy_dir=proxy_dir,
                eligible_dir=eligible_dir,
                out_dir=family_dir / "negatives" / proxy_dir.name,
                proxy_name=proxy_name,
                config=config,
                device=args.device,
            )
            (family_dir / "negatives" / proxy_dir.name / "diagnostics.json").write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary[family_name]["proxies"][proxy_dir.name] = diagnostics

        validated_root = family_dir / "validated"
        if validated_root.exists():
            for proxy_dir in sorted(validated_root.iterdir()):
                if not proxy_dir.is_dir():
                    continue
                proxy_name = _load_proxy_name(proxy_dir, "train")
                diagnostics = _materialize_split_set(
                    proxy_dir=proxy_dir,
                    eligible_dir=validated_eligible_dir,
                    out_dir=family_dir / "validated_negatives" / proxy_dir.name,
                    proxy_name=proxy_name,
                    config=config,
                    device=args.device,
                )
                (family_dir / "validated_negatives" / proxy_dir.name / "diagnostics.json").write_text(
                    json.dumps(diagnostics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                summary[family_name]["validated_proxies"][proxy_dir.name] = diagnostics

    save_with_metadata(
        output_path=manifest_root / "negative_matching_summary.json",
        payload={"summary": summary},
        config={"args": vars(args)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
