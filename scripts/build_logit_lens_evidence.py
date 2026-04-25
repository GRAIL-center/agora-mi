"""Build local logit lens evidence from audit packages."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_interp.audit_evidence_suite import AUDIT_SURFACE_TERMS  # noqa: E402
from policy_interp.io import read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packages",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval" / "packages" / "C0_passage_only.jsonl",
        help="Passage only package JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval" / "tool_evidence" / "logit_lens_evidence.jsonl",
        help="Output evidence JSONL.",
    )
    parser.add_argument("--model", default="google/gemma-2-2b", help="Local Hugging Face target model id.")
    parser.add_argument("--layers", default="6,12,18,24", help="Comma separated transformer layer indices.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=0, help="Use 0 for all cases.")
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            layers.append(int(part))
    return layers


def choose_positions(
    passage: str,
    tokenizer: Any,
    encoded: dict[str, Any],
    *,
    max_positions: int,
) -> list[int]:
    offsets = encoded.get("offset_mapping")
    attention = encoded["attention_mask"][0].tolist()
    valid_positions = [index for index, value in enumerate(attention) if value]
    if not valid_positions:
        return []
    chosen: list[int] = []
    if offsets is not None:
        offset_rows = offsets[0].tolist()
        lower = passage.lower()
        char_offsets = []
        for term in AUDIT_SURFACE_TERMS:
            found = lower.find(term)
            if found >= 0:
                char_offsets.append(found)
        for char_offset in char_offsets:
            for token_index, (start, end) in enumerate(offset_rows):
                if start <= char_offset < end and token_index in valid_positions:
                    if token_index not in chosen:
                        chosen.append(token_index)
                    break
                if start > char_offset:
                    break
            if len(chosen) >= max_positions:
                break
    if not chosen:
        last = valid_positions[-2] if len(valid_positions) > 1 else valid_positions[-1]
        chosen.append(last)
    return chosen[:max_positions]


def safe_token_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return "<empty>"
    try:
        cleaned.encode("ascii")
    except UnicodeEncodeError:
        return cleaned.encode("unicode_escape").decode("ascii")
    return cleaned


def normalize_for_unembedding(model: Any, hidden: Any) -> Any:
    norm = getattr(getattr(model, "model", None), "norm", None)
    if norm is None:
        return hidden
    return norm(hidden)


def informative_token(token: str) -> bool:
    if token in {"<empty>", "<bos>", "<eos>", "<pad>"}:
        return False
    noisy_tokens = {
        "jefus",
        "majefty",
        "monfieur",
        "themfelves",
        "myfelf",
        "itfelf",
        "whofe",
        "felf",
        "efq",
        "thofe",
        "thefe",
        "fuch",
        "fome",
        "fhall",
        "muft",
        "perfons",
        "berfhka",
        "bershka",
        "cdti",
        "fubject",
        "berdayakan",
        "bekasi",
        "neceff",
        "vectorielle",
        "fallu",
        "exigences",
        "fhew",
        "fevere",
        "fafety",
        "fecurity",
    }
    if token.lower() in noisy_tokens:
        return False
    if "\\u" in token or "\\x" in token:
        return False
    if len(token) > 40:
        return False
    if re.fullmatch(r"[\W_]+", token):
        return False
    return True


def top_tokens_from_hidden(model: Any, tokenizer: Any, hidden: Any, top_k: int) -> tuple[list[str], list[float]]:
    import torch

    output_embeddings = model.get_output_embeddings()
    weight_dtype = output_embeddings.weight.dtype
    normal_hidden = hidden.detach().clone().to(dtype=weight_dtype)
    candidate_count = max(top_k * 8, top_k)
    with torch.no_grad():
        normalized_hidden = normalize_for_unembedding(model, normal_hidden)
        logits = output_embeddings(normalized_hidden)
        values, indices = torch.topk(logits.float(), k=candidate_count)
    tokens: list[str] = []
    scores: list[float] = []
    fallback_tokens: list[str] = []
    fallback_scores: list[float] = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=False):
        token = safe_token_text(tokenizer.decode([int(index)]))
        fallback_tokens.append(token)
        fallback_scores.append(float(value))
        if informative_token(token):
            tokens.append(token)
            scores.append(float(value))
        if len(tokens) >= top_k:
            break
    if not tokens:
        return fallback_tokens[:top_k], fallback_scores[:top_k]
    return tokens, scores


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    packages = read_jsonl(args.packages)
    if args.max_cases and args.max_cases > 0:
        packages = packages[: args.max_cases]
    layers = parse_layers(args.layers)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    records: list[dict[str, Any]] = []
    for package in packages:
        passage = str(package.get("passage", ""))
        encoded = tokenizer(
            passage,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=args.max_length,
        )
        positions = choose_positions(passage, tokenizer, encoded, max_positions=args.max_positions)
        model_inputs = {key: value for key, value in encoded.items() if key != "offset_mapping"}
        device = getattr(model, "device", None)
        if device is not None and str(device) != "meta":
            model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
        with torch.inference_mode():
            outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
        evidence_items: list[dict[str, Any]] = []
        for layer in layers:
            hidden_index = min(layer + 1, len(outputs.hidden_states) - 1)
            layer_hidden = outputs.hidden_states[hidden_index][0]
            for position in positions:
                tokens, scores = top_tokens_from_hidden(model, tokenizer, layer_hidden[position], args.top_k)
                trigger_token = safe_token_text(tokenizer.decode([int(model_inputs["input_ids"][0, position].item())]))
                evidence_items.append(
                    {
                        "evidence_id": f"LOGIT_L{layer}_P{position}",
                        "tool": "logit_lens",
                        "evidence_type": "intermediate_token_direction",
                        "label_source": "local_logit_lens",
                        "layer": layer,
                        "token_position": position,
                        "trigger_token": trigger_token,
                        "top_tokens": tokens,
                        "top_token_scores": scores,
                        "caveat": "Logit lens shows intermediate token directions, not a final audit finding.",
                    }
                )
        records.append({"case_id": package["case_id"], "evidence_items": evidence_items})
    write_jsonl(records, args.output)
    print(f"Wrote logit lens evidence for {len(records)} cases to {args.output}")


if __name__ == "__main__":
    main()
