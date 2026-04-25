"""Build clearly labeled activation explanation surrogate evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_interp.audit_evidence_suite import extract_json_object  # noqa: E402
from policy_interp.io import read_jsonl, write_jsonl  # noqa: E402
from run_local_audit_reports import LocalCausalLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packages",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval" / "packages" / "C2_sae_only.jsonl",
        help="SAE package JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval" / "tool_evidence" / "activation_oracle_surrogate_evidence.jsonl",
        help="Output evidence JSONL.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic surrogate explanations.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Local Hugging Face instruction model id.")
    parser.add_argument("--max-items-per-case", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def render_prompt(package: dict[str, Any], item: dict[str, Any]) -> str:
    return (
        "You are writing an activation explanation surrogate for an audit experiment.\n"
        "The input is a manually reviewed SAE activation item, not a true activation oracle vector readout.\n"
        "Explain what the activation may be tracking in natural language and state limitations.\n"
        "Return one JSON object with keys activation_meaning, useful_audit_hint, limitations, confidence.\n\n"
        f"Passage:\n{package.get('passage', '')}\n\n"
        f"SAE activation item:\n{item}\n"
    )


def dry_surrogate(item: dict[str, Any]) -> dict[str, Any]:
    label = str(item.get("label", "the activated concept"))
    span = str(item.get("activated_span", ""))[:300]
    return {
        "activation_meaning": f"The activation appears related to {label}.",
        "useful_audit_hint": f"Check whether the passage span supports this concept: {span}",
        "limitations": "This is a surrogate explanation derived from SAE labels, not a true activation oracle.",
        "confidence": 0.35,
    }


def main() -> None:
    args = parse_args()
    packages = read_jsonl(args.packages)
    generator = None
    if not args.dry_run:
        generator = LocalCausalLM(
            args.model,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    records: list[dict[str, Any]] = []
    for package in packages:
        items: list[dict[str, Any]] = []
        for rank, item in enumerate(package.get("evidence_items", [])[: args.max_items_per_case], start=1):
            if args.dry_run:
                parsed = dry_surrogate(item)
                raw_output = ""
                parse_status = "parsed"
            else:
                raw_output = generator.generate(render_prompt(package, item))
                try:
                    parsed = extract_json_object(raw_output)
                    parse_status = "parsed"
                except ValueError as exc:
                    parsed = {"limitations": str(exc), "confidence": 0.0}
                    parse_status = "parse_failed"
            items.append(
                {
                    "evidence_id": f"AO_SURROGATE_{rank}_{item.get('evidence_id', '')}",
                    "tool": "activation_oracle_surrogate",
                    "evidence_type": "natural_language_activation_explanation_surrogate",
                    "label_source": "local_activation_explanation_surrogate",
                    "source_evidence_id": item.get("evidence_id"),
                    "activation_meaning": parsed.get("activation_meaning", ""),
                    "useful_audit_hint": parsed.get("useful_audit_hint", ""),
                    "limitations": parsed.get("limitations", ""),
                    "confidence": parsed.get("confidence"),
                    "parse_status": parse_status,
                    "raw_output": raw_output,
                    "caveat": "This is not a true Activation Oracle result unless replaced by a compatible oracle backend.",
                }
            )
        records.append({"case_id": package["case_id"], "evidence_items": items})
    write_jsonl(records, args.output)
    print(f"Wrote activation explanation surrogate evidence for {len(records)} cases to {args.output}")


if __name__ == "__main__":
    main()
