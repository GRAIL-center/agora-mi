from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from _common import save_with_metadata, setup_logging
from data.io import load_agora_records, write_jsonl
from data.public_values import (
    all_family_proxy_rows,
    assign_governance_strategy_categories,
    assign_public_value_families,
    load_public_value_spec,
    proxy_slug,
)
from data.split import assert_no_overlap, split_doc_ids


def _passes_quality_filters(
    row: dict[str, Any],
    *,
    require_ai_related: bool,
    require_operative: bool,
    require_annotated: bool,
    validated_only: bool,
    min_text_chars: int,
    max_text_chars: int,
) -> bool:
    text = str(row.get("text", ""))
    if len(text) < min_text_chars or len(text) > max_text_chars:
        return False
    if require_ai_related and row.get("not_ai_related") is True:
        return False
    if require_operative and row.get("non_operative") is True:
        return False
    if require_annotated and row.get("segment_annotated") is not True:
        return False
    if validated_only and row.get("segment_validated") is not True:
        return False
    return True


def _quality_flags(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "non_operative": row.get("non_operative"),
        "not_ai_related": row.get("not_ai_related"),
        "segment_annotated": row.get("segment_annotated"),
        "segment_validated": row.get("segment_validated"),
        "document_annotated": row.get("document_annotated"),
        "document_validated": row.get("document_validated"),
        "segment_unreviewed_machine_output": row.get("segment_unreviewed_machine_output"),
    }


def _metadata_blob(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "official_name": row.get("official_name"),
        "authority": row.get("authority"),
        "collections": row.get("collection_values") or ([] if not row.get("collections") else [row["collections"]]),
        "collection_domains": row.get("collection_domains", []),
        "jurisdiction": row.get("jurisdiction"),
        "document_form": row.get("document_form"),
        "year": row.get("year"),
        "application_tags": row.get("application_tags", []),
        "risk_tags": row.get("risk_tags", []),
        "harm_tags": row.get("harm_tags", []),
        "incentive_tags": row.get("incentive_tags", []),
        "strategy_tags": row.get("strategy_tags", []),
        "source": row.get("source"),
        "official_plaintext_source": row.get("official_plaintext_source"),
        "official_pdf_source": row.get("official_pdf_source"),
        "most_recent_activity": row.get("most_recent_activity"),
        "document_most_recent_activity_date": row.get("document_most_recent_activity_date"),
        "document_proposed_date": row.get("document_proposed_date"),
        "document_applies_government": row.get("document_applies_government"),
        "document_applies_private_sector": row.get("document_applies_private_sector"),
    }


def _base_manifest_row(
    row: dict[str, Any],
    *,
    split: str,
    family_name: str | None,
    family_label: str | None,
    family_status: str | None,
    family_info: dict[str, Any] | None,
    strategy_hits: dict[str, list[str]],
    proxy_name: str | None = None,
    proxy_tier: str | None = None,
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "segment_id": row["segment_id"],
        "doc_id": row["doc_id"],
        "document_id": row["document_id"],
        "text": row["text"],
        "summary": row.get("summary"),
        "split": split,
        "family_name": family_name,
        "family_label": family_label,
        "family_status": family_status,
        "proxy_name": proxy_name,
        "proxy_slug": proxy_slug(proxy_name) if proxy_name else None,
        "proxy_tier": proxy_tier,
        "tags": list(row.get("tags", [])),
        "all_tags": list(row.get("all_tags", [])),
        "main_proxy_hits": list((family_info or {}).get("main_hits", [])),
        "secondary_proxy_hits": list((family_info or {}).get("secondary_hits", [])),
        "excluded_proxy_hits": list((family_info or {}).get("excluded_hits", [])),
        "strategy_category_hits": {k: list(v) for k, v in strategy_hits.items()},
        "strategy_categories": sorted(strategy_hits.keys()),
        "quality_flags": _quality_flags(row),
        "metadata": _metadata_blob(row),
    }


def _save_split_rows(base_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        rows = rows_by_split.get(split, [])
        write_jsonl(base_dir / f"{split}.jsonl", rows)
        counts[split] = len(rows)
    return counts


def _materialize_transfer_manifest(
    *,
    out_path: Path,
    source_family: str,
    source_proxy_name: str,
    target_family: str,
    target_proxy_name: str,
    family_relation: str,
) -> None:
    payload = {
        "manifest_version": 1,
        "source_family": source_family,
        "source_proxy": source_proxy_name,
        "source_proxy_slug": proxy_slug(source_proxy_name),
        "target_family": target_family,
        "target_proxy": target_proxy_name,
        "target_proxy_slug": proxy_slug(target_proxy_name),
        "family_relation": family_relation,
        "evaluation_split": "test",
        "feature_bank_path": None,
        "expected_feature_bank_glob": (
            f"results/policy_discovery/{source_family}/{proxy_slug(source_proxy_name)}/**/feature_bank.json"
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/public_value_families.yaml")
    parser.add_argument("--input_dir", default="data/raw/agora")
    parser.add_argument("--out_dir", default="data/processed/public_values")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--validated_only", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--dev_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()

    setup_logging("build_public_value_corpus")
    spec = load_public_value_spec(args.config)
    records = load_agora_records(args.input_dir)
    logging.info("Loaded AGORA records: %d", len(records))

    require_ai_related = bool(spec.quality_filters.get("require_ai_related", True))
    require_operative = bool(spec.quality_filters.get("require_operative", True))
    require_annotated = bool(spec.quality_filters.get("prefer_annotated", True))
    min_text_chars = int(spec.quality_filters.get("min_text_chars", 0))
    max_text_chars = int(spec.quality_filters.get("max_text_chars", 10_000_000))

    selected_families = set(args.families) if args.families else {
        key for key, family in spec.families.items() if family.status == "main"
    }
    selected_proxy_rows = all_family_proxy_rows(
        spec,
        family_keys=selected_families,
        include_secondary=True,
        statuses={"main", "exploratory"},
    )

    eligible_rows: list[dict[str, Any]] = []
    eligible_validated_rows: list[dict[str, Any]] = []
    for row in records:
        base_ok = _passes_quality_filters(
            row,
            require_ai_related=require_ai_related,
            require_operative=require_operative,
            require_annotated=require_annotated,
            validated_only=args.validated_only,
            min_text_chars=min_text_chars,
            max_text_chars=max_text_chars,
        )
        if not base_ok:
            continue
        eligible_rows.append(row)
        if row.get("segment_validated") is True:
            eligible_validated_rows.append(row)

    logging.info("Eligible rows after quality filters: %d", len(eligible_rows))
    logging.info("Validated eligible rows: %d", len(eligible_validated_rows))

    eligible_doc_ids = sorted({str(row["doc_id"]) for row in eligible_rows})
    doc_split = split_doc_ids(
        eligible_doc_ids,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    assert_no_overlap(doc_split)
    split_lookup = {
        **{doc_id: "train" for doc_id in doc_split.train},
        **{doc_id: "dev" for doc_id in doc_split.dev},
        **{doc_id: "test" for doc_id in doc_split.test},
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eligible_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    validated_eligible_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)

    family_pool_rows: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        key: {
            "all": defaultdict(list),
            "main": defaultdict(list),
        }
        for key in selected_families
    }
    proxy_rows: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        row["family_key"]: defaultdict(lambda: defaultdict(list))
        for row in selected_proxy_rows
    }
    validated_proxy_rows: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        row["family_key"]: defaultdict(lambda: defaultdict(list))
        for row in selected_proxy_rows
    }
    family_proxy_counts: dict[str, dict[str, int]] = {key: defaultdict(int) for key in selected_families}
    strategy_counts: dict[str, dict[str, int]] = {key: defaultdict(int) for key in selected_families}

    for row in eligible_rows:
        split = split_lookup[str(row["doc_id"])]
        tag_names = row.get("all_tags", []) or row.get("tags", [])
        strategy_hits = assign_governance_strategy_categories(tag_names, spec)
        family_hits = assign_public_value_families(
            tag_names,
            spec,
            include_secondary=True,
            require_main=False,
            statuses={"main", "exploratory"},
        )

        eligible_by_split[split].append(
            _base_manifest_row(
                row,
                split=split,
                family_name=None,
                family_label=None,
                family_status=None,
                family_info=None,
                strategy_hits=strategy_hits,
            )
        )
        if row.get("segment_validated") is True:
            validated_eligible_by_split[split].append(
                _base_manifest_row(
                    row,
                    split=split,
                    family_name=None,
                    family_label=None,
                    family_status=None,
                    family_info=None,
                    strategy_hits=strategy_hits,
                )
            )

        for family_key, info in family_hits.items():
            if family_key not in selected_families:
                continue
            family = spec.families[family_key]
            family_row = _base_manifest_row(
                row,
                split=split,
                family_name=family_key,
                family_label=family.label,
                family_status=family.status,
                family_info=info,
                strategy_hits=strategy_hits,
            )
            family_pool_rows[family_key]["all"][split].append(family_row)
            if info["main_hits"]:
                family_pool_rows[family_key]["main"][split].append(family_row)
            for strategy_key in family_row["strategy_categories"]:
                strategy_counts[family_key][strategy_key] += 1

            for proxy_name in info["main_hits"]:
                slug = proxy_slug(proxy_name)
                proxy_row = dict(family_row)
                proxy_row["proxy_name"] = proxy_name
                proxy_row["proxy_slug"] = slug
                proxy_row["proxy_tier"] = "main"
                proxy_rows[family_key][slug][split].append(proxy_row)
                family_proxy_counts[family_key][proxy_name] += 1
                if row.get("segment_validated") is True:
                    validated_proxy_rows[family_key][slug][split].append(proxy_row)

            for proxy_name in info["secondary_hits"]:
                slug = proxy_slug(proxy_name)
                proxy_row = dict(family_row)
                proxy_row["proxy_name"] = proxy_name
                proxy_row["proxy_slug"] = slug
                proxy_row["proxy_tier"] = "secondary"
                proxy_rows[family_key][slug][split].append(proxy_row)
                family_proxy_counts[family_key][proxy_name] += 1
                if row.get("segment_validated") is True:
                    validated_proxy_rows[family_key][slug][split].append(proxy_row)

    eligible_dir = out_dir / "eligible"
    validated_eligible_dir = out_dir / "validated"
    eligible_counts = _save_split_rows(eligible_dir, eligible_by_split)
    validated_eligible_counts = _save_split_rows(validated_eligible_dir, validated_eligible_by_split)

    root_summary: dict[str, Any] = {
        "manifest_version": 1,
        "config_path": args.config,
        "seed": args.seed,
        "split_counts": eligible_counts,
        "validated_split_counts": validated_eligible_counts,
        "families": {},
        "selected_proxies": selected_proxy_rows,
    }

    for family_key in sorted(selected_families):
        family = spec.families[family_key]
        family_dir = out_dir / family_key
        family_dir.mkdir(parents=True, exist_ok=True)

        pool_counts = {
            tier: _save_split_rows(family_dir / "family_pools" / tier, family_pool_rows[family_key][tier])
            for tier in ("main", "all")
        }

        proxy_summary: dict[str, Any] = {}
        for proxy_name, proxy_tier in ((proxy, tier) for proxy, tier in [(p["proxy_name"], p["proxy_tier"]) for p in selected_proxy_rows if p["family_key"] == family_key]):
            slug = proxy_slug(proxy_name)
            proxy_dir = family_dir / "proxies" / slug
            validated_dir = family_dir / "validated" / slug
            proxy_counts = _save_split_rows(proxy_dir, proxy_rows[family_key][slug])
            validated_counts = _save_split_rows(validated_dir, validated_proxy_rows[family_key][slug])
            proxy_summary[slug] = {
                "proxy_name": proxy_name,
                "proxy_slug": slug,
                "proxy_tier": proxy_tier,
                "counts": proxy_counts,
                "validated_counts": validated_counts,
            }

        family_summary = {
            "manifest_version": 1,
            "family_name": family_key,
            "family_label": family.label,
            "family_status": family.status,
            "family_description": family.description,
            "pool_counts": pool_counts,
            "proxy_counts": dict(family_proxy_counts[family_key]),
            "strategy_counts": dict(strategy_counts[family_key]),
            "proxies": proxy_summary,
        }
        (family_dir / "summary.json").write_text(
            __import__("json").dumps(family_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        root_summary["families"][family_key] = family_summary

    all_selected_proxies = [row for row in selected_proxy_rows if row["family_key"] in selected_families]
    for source in all_selected_proxies:
        source_family_dir = out_dir / source["family_key"] / "transfers"
        for target in all_selected_proxies:
            if source["family_key"] == target["family_key"] and source["proxy_slug"] == target["proxy_slug"]:
                continue
            relation = "within_family" if source["family_key"] == target["family_key"] else "cross_family"
            _materialize_transfer_manifest(
                out_path=source_family_dir / f"{source['proxy_slug']}__to__{target['proxy_slug']}.json",
                source_family=source["family_key"],
                source_proxy_name=source["proxy_name"],
                target_family=target["family_key"],
                target_proxy_name=target["proxy_name"],
                family_relation=relation,
            )

    save_with_metadata(
        output_path=out_dir / "summary.json",
        payload={"summary": root_summary},
        config={
            "config_path": args.config,
            "input_dir": args.input_dir,
            "out_dir": args.out_dir,
            "seed": args.seed,
            "families": sorted(selected_families),
            "validated_only": args.validated_only,
            "train_ratio": args.train_ratio,
            "dev_ratio": args.dev_ratio,
            "test_ratio": args.test_ratio,
        },
    )
    save_with_metadata(
        output_path=Path("logs") / "build_public_value_corpus_summary.json",
        payload={"summary": root_summary},
        config={
            "config_path": args.config,
            "input_dir": args.input_dir,
            "out_dir": args.out_dir,
            "seed": args.seed,
            "families": sorted(selected_families),
            "validated_only": args.validated_only,
            "train_ratio": args.train_ratio,
            "dev_ratio": args.dev_ratio,
            "test_ratio": args.test_ratio,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
