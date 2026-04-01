from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_title(card: dict[str, Any]) -> str:
    doc_id = str(card.get("document_id") or "")
    return f"Document {doc_id}"


def _method_label(method_name: str) -> str:
    return {
        "sparse_sae_feature_bank": "dossier_backed",
        "finetuned_encoder_multilabel": "black_box_supervised",
        "semantic_sentence_embed_logreg": "black_box_embedding",
    }.get(method_name, method_name)


def _presentation_text(card: dict[str, Any], *, include_sparse_details: bool) -> str:
    lines = [
        f"Family: {card.get('family_display_name')}",
        f"Anchor: {card.get('proxy_anchor_display_name')}",
        f"Priority score: {float(card.get('priority_score', 0.0)):.3f}",
        f"Reliability score: {float(card.get('reliability_score', 0.0)):.3f}",
        f"Note: {card.get('natural_language_note', '')}",
    ]
    if include_sparse_details:
        top_ids = ", ".join(str(value) for value in card.get("top_feature_ids") or [])
        lines.extend(
            [
                f"Selected layer: {card.get('selected_layer')}",
                f"Top sparse features: {top_ids}",
                f"Mean feature stability: {float(card.get('mean_feature_stability', 0.0)):.3f}",
                f"Causal badge: {card.get('causal_badge')}",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dossier_method", default="sparse_sae_feature_bank")
    parser.add_argument("--baseline_method", default="finetuned_encoder_multilabel")
    parser.add_argument("--items_per_family", type=int, default=6)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    output_root = _ensure_dir(Path(args.output_root).resolve())
    method_results_dir = bundle_root / "results" / "policy_analysis_assistant_locked_2b" / "method_results"

    dossier_payload = _load_json(method_results_dir / f"{args.dossier_method}.json")
    baseline_payload = _load_json(method_results_dir / f"{args.baseline_method}.json")

    dossier_cards = list(dossier_payload.get("segment_cards") or [])
    baseline_cards = list(baseline_payload.get("segment_cards") or [])
    baseline_by_segment = {str(card["segment_id"]): card for card in baseline_cards}

    selected: list[dict[str, Any]] = []
    for family in ["equality_neutrality", "individual_rights", "transparency_accountability"]:
        family_cards = [card for card in dossier_cards if str(card.get("family")) == family]
        family_cards.sort(key=lambda card: float(card.get("priority_score", 0.0)), reverse=True)
        selected.extend(family_cards[: args.items_per_family])

    rng = random.Random(args.seed)
    tasks: list[dict[str, Any]] = []
    sheet_rows: list[dict[str, Any]] = []

    for index, dossier_card in enumerate(selected, start=1):
        segment_id = str(dossier_card["segment_id"])
        baseline_card = baseline_by_segment.get(segment_id)
        if baseline_card is None:
            continue

        dossier_text = _presentation_text(dossier_card, include_sparse_details=True)
        baseline_text = _presentation_text(baseline_card, include_sparse_details=False)
        if rng.random() < 0.5:
            variant_a, variant_b = dossier_text, baseline_text
            condition_a, condition_b = _method_label(args.dossier_method), _method_label(args.baseline_method)
        else:
            variant_a, variant_b = baseline_text, dossier_text
            condition_a, condition_b = _method_label(args.baseline_method), _method_label(args.dossier_method)

        item_id = f"assistant_eval_{index:03d}"
        title = _normalize_title(dossier_card)
        task = {
            "item_id": item_id,
            "family": dossier_card.get("family"),
            "family_display_name": dossier_card.get("family_display_name"),
            "segment_id": segment_id,
            "document_id": dossier_card.get("document_id"),
            "document_title": title,
            "segment_text": dossier_card.get("segment_text", ""),
            "variant_a": variant_a,
            "variant_b": variant_b,
            "hidden_condition_a": condition_a,
            "hidden_condition_b": condition_b,
            "recommended_questions": [
                "Which card would help you triage this segment faster?",
                "Which card would you trust more when justifying a prioritization decision?",
                "Which card gives a clearer basis for follow-up review?",
            ],
        }
        tasks.append(task)
        sheet_rows.append(
            {
                "item_id": item_id,
                "family": dossier_card.get("family_display_name"),
                "document_title": title,
                "segment_id": segment_id,
                "speed_winner": "",
                "trust_winner": "",
                "justification_winner": "",
                "notes": "",
            }
        )

    (output_root / "tasks.jsonl").write_text(
        "\n".join(json.dumps(task, ensure_ascii=False) for task in tasks) + "\n",
        encoding="utf-8",
    )
    with (output_root / "annotation_sheet.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet_rows[0].keys()) if sheet_rows else ["item_id"])
        writer.writeheader()
        for row in sheet_rows:
            writer.writerow(row)

    readme = "\n".join(
        [
            "# Assistant Human Evaluation Pack",
            "",
            "This package supports a small blinded comparison between dossier-backed sparse cards and a black-box baseline.",
            "",
            "## Suggested protocol",
            "",
            "1. Give each evaluator `tasks.jsonl` and `annotation_sheet.csv`.",
            "2. Hide `hidden_condition_a` and `hidden_condition_b` during annotation.",
            "3. Ask evaluators to choose a winner for speed, trust, and justification quality.",
            "4. Aggregate wins after revealing the hidden conditions.",
            "",
            "## Notes",
            "",
            "1. Variant A and Variant B are randomized with a fixed seed.",
            "2. The sparse dossier condition exposes layer, feature identifiers, stability, and causal badge.",
            "3. The baseline condition exposes only the model note and ranking style summary.",
            "",
        ]
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
