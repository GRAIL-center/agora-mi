"""Run a local model or deterministic dry run over audit evidence packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_interp.audit_evidence_suite import (  # noqa: E402
    build_blackbox_surface_items,
    dry_run_audit_report,
    extract_json_object,
    normalize_audit_report,
    render_auditor_prompt,
    render_blackbox_generator_prompt,
)
from policy_interp.io import read_jsonl, write_jsonl  # noqa: E402
from policy_interp.utils import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packages",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval" / "packages",
        help="Package JSONL file or directory containing package JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval" / "reports" / "audit_reports_dry_run.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--condition", action="append", default=None, help="Condition id to run. Repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum records to process. Use 0 for all.")
    parser.add_argument(
        "--task",
        choices=("audit_report", "blackbox_observations"),
        default="audit_report",
        help="Generate final audit reports or black box evidence observations.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic local scaffolds instead of an LLM.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Local Hugging Face model id.")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output and skip completed keys.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N generated records.")
    return parser.parse_args()


def load_packages(path: Path, conditions: set[str] | None) -> list[dict[str, Any]]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("C*.jsonl"))
    packages: list[dict[str, Any]] = []
    for file_path in files:
        for package in read_jsonl(file_path):
            if conditions and package.get("condition_id") not in conditions:
                continue
            packages.append(package)
    return packages


class LocalCausalLM:
    """Small wrapper around transformers generation."""

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str = "auto",
        trust_remote_code: bool = False,
        max_new_tokens: int = 900,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str) -> str:
        import torch

        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            inputs = {key: value.to(device) for key, value in inputs.items()}
        do_sample = self.temperature > 0
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        input_length = inputs["input_ids"].shape[-1]
        generated = outputs[0][input_length:]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return decoded


def dry_run_blackbox_observations(package: dict[str, Any]) -> dict[str, Any]:
    case_record = {
        "case_id": package["case_id"],
        "passage": package["passage"],
        "visible_metadata": package.get("visible_metadata", {}),
    }
    return {"evidence_items": build_blackbox_surface_items(case_record)}


def run_one(
    package: dict[str, Any],
    *,
    task: str,
    generator: LocalCausalLM | None,
    model_name: str,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        payload = dry_run_audit_report(package) if task == "audit_report" else dry_run_blackbox_observations(package)
        repair_notes = []
        if task == "audit_report":
            payload, repair_notes = normalize_audit_report(payload, package)
        return {
            "case_id": package["case_id"],
            "condition_id": package.get("condition_id"),
            "task": task,
            "model": "dry_run",
            "report": payload if task == "audit_report" else None,
            "evidence_items": payload.get("evidence_items") if task == "blackbox_observations" else None,
            "raw_output": "",
            "parse_status": "parsed",
            "repair_notes": repair_notes,
        }

    if generator is None:
        raise ValueError("A generator is required when dry_run is false.")
    prompt = render_auditor_prompt(package) if task == "audit_report" else render_blackbox_generator_prompt(package)
    raw_output = generator.generate(prompt)
    try:
        parsed = extract_json_object(raw_output)
        parse_status = "parsed"
    except ValueError as exc:
        parsed = {"parse_error": str(exc)}
        parse_status = "parse_failed"
    repair_notes: list[str] = []
    if task == "audit_report" and parse_status == "parsed":
        parsed, repair_notes = normalize_audit_report(parsed, package)
    return {
        "case_id": package["case_id"],
        "condition_id": package.get("condition_id"),
        "task": task,
        "model": model_name,
        "report": parsed if task == "audit_report" else None,
        "evidence_items": parsed.get("evidence_items") if task == "blackbox_observations" else None,
        "raw_output": raw_output,
        "parse_status": parse_status,
        "repair_notes": repair_notes,
    }


def main() -> None:
    args = parse_args()
    conditions = set(args.condition) if args.condition else None
    packages = load_packages(args.packages, conditions)
    if args.limit and args.limit > 0:
        packages = packages[: args.limit]
    completed: set[tuple[str, str, str]] = set()
    if args.resume and args.output.exists():
        for record in read_jsonl(args.output):
            completed.add((str(record.get("case_id")), str(record.get("condition_id")), str(record.get("task"))))
    packages = [
        package
        for package in packages
        if (str(package.get("case_id")), str(package.get("condition_id")), args.task) not in completed
    ]
    generator = None
    if not args.dry_run:
        generator = LocalCausalLM(
            args.model,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    ensure_dir(args.output.parent)
    mode = "a" if args.resume and args.output.exists() else "w"
    written = 0
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, package in enumerate(packages, start=1):
            record = run_one(
                package,
                task=args.task,
                generator=generator,
                model_name=args.model,
                dry_run=args.dry_run,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            if args.progress_every and (index % args.progress_every == 0 or index == len(packages)):
                print(f"generated {index}/{len(packages)} records", flush=True)
    print(f"Wrote {written} records to {args.output}")


if __name__ == "__main__":
    main()
