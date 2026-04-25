"""Build proxy free steering vector evidence from local hidden state directions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_logit_lens_evidence import (  # noqa: E402
    choose_positions,
    informative_token,
    normalize_for_unembedding,
    safe_token_text,
)
from policy_interp.io import read_jsonl, write_jsonl  # noqa: E402


POSITIVE_STEERING_PROMPTS = (
    "The policy imposes mandatory obligations, risk controls, compliance duties, and safety requirements.",
    "Providers must test, monitor, document, and mitigate high risk AI systems before deployment.",
    "The regulation requires transparency, accountability, enforcement, and protection from harms.",
)

NEGATIVE_STEERING_PROMPTS = (
    "The passage is descriptive background without binding obligations or risk controls.",
    "The document gives general context and does not require testing, monitoring, or compliance action.",
    "The text is an informational note with no mandatory provider duties or enforcement requirements.",
)


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
        default=ROOT / "artifacts" / "audit_evidence_eval" / "tool_evidence" / "steering_vector_evidence_gemma2_2b.jsonl",
        help="Output evidence JSONL.",
    )
    parser.add_argument("--model", default="google/gemma-2-2b", help="Local Hugging Face target model id.")
    parser.add_argument("--layers", default="12,18,24", help="Comma separated transformer layer indices.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Direction strength in hidden norm units.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=0, help="Use 0 for all cases.")
    parser.add_argument("--max-positions", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def final_token_index(attention_mask: Any) -> int:
    values = attention_mask[0].tolist()
    active = [index for index, item in enumerate(values) if item]
    return active[-1] if active else 0


def top_tokens_from_logits(tokenizer: Any, logits: Any, top_k: int) -> tuple[list[str], list[float]]:
    import torch

    candidate_count = max(top_k * 8, top_k)
    values, indices = torch.topk(logits.float(), k=candidate_count)
    tokens: list[str] = []
    scores: list[float] = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=False):
        token = safe_token_text(tokenizer.decode([int(index)]))
        if not informative_token(token):
            continue
        tokens.append(token)
        scores.append(float(value))
        if len(tokens) >= top_k:
            break
    return tokens, scores


def logits_from_hidden(model: Any, hidden: Any) -> Any:
    import torch

    output_embeddings = model.get_output_embeddings()
    weight_dtype = output_embeddings.weight.dtype
    with torch.no_grad():
        normalized_hidden = normalize_for_unembedding(model, hidden.detach().clone().to(dtype=weight_dtype))
        return output_embeddings(normalized_hidden).float()


def kl_divergence(base_logits: Any, steered_logits: Any) -> float:
    import torch

    with torch.no_grad():
        base_log_probs = torch.log_softmax(base_logits.float(), dim=-1)
        steered_log_probs = torch.log_softmax(steered_logits.float(), dim=-1)
        base_probs = base_log_probs.exp()
        return float((base_probs * (base_log_probs - steered_log_probs)).sum().item())


def mean_prompt_hidden(model: Any, tokenizer: Any, prompts: tuple[str, ...], layer: int, max_length: int) -> Any:
    import torch

    vectors = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        device = getattr(model, "device", None)
        if device is not None and str(device) != "meta":
            encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden_index = min(layer + 1, len(outputs.hidden_states) - 1)
        position = final_token_index(encoded["attention_mask"])
        vectors.append(outputs.hidden_states[hidden_index][0, position].detach().clone())
    return torch.stack(vectors, dim=0).mean(dim=0)


def build_directions(model: Any, tokenizer: Any, layers: list[int], max_length: int) -> dict[int, Any]:
    import torch

    directions = {}
    for layer in layers:
        positive = mean_prompt_hidden(model, tokenizer, POSITIVE_STEERING_PROMPTS, layer, max_length)
        negative = mean_prompt_hidden(model, tokenizer, NEGATIVE_STEERING_PROMPTS, layer, max_length)
        direction = positive - negative
        direction = direction / torch.clamp(direction.norm(), min=1e-6)
        directions[layer] = direction
    return directions


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
    directions = build_directions(model, tokenizer, layers, args.max_length)

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
        items: list[dict[str, Any]] = []
        for layer in layers:
            hidden_index = min(layer + 1, len(outputs.hidden_states) - 1)
            layer_hidden = outputs.hidden_states[hidden_index][0]
            direction = directions[layer].to(layer_hidden.device)
            for position in positions:
                hidden = layer_hidden[position].detach().clone()
                hidden_norm = torch.clamp(hidden.norm(), min=1e-6)
                steered_hidden = hidden + args.alpha * hidden_norm * direction
                base_logits = logits_from_hidden(model, hidden)
                steered_logits = logits_from_hidden(model, steered_hidden)
                base_tokens, base_scores = top_tokens_from_logits(tokenizer, base_logits, args.top_k)
                steered_tokens, steered_scores = top_tokens_from_logits(tokenizer, steered_logits, args.top_k)
                trigger_token = safe_token_text(tokenizer.decode([int(model_inputs["input_ids"][0, position].item())]))
                items.append(
                    {
                        "evidence_id": f"STEER_L{layer}_P{position}",
                        "tool": "steering_vector",
                        "evidence_type": "hidden_state_direction_probe",
                        "label": "obligation and risk control steering direction",
                        "label_source": "local_contrastive_direction",
                        "layer": layer,
                        "token_position": position,
                        "trigger_token": trigger_token,
                        "alpha": args.alpha,
                        "base_top_tokens": base_tokens,
                        "base_top_token_scores": base_scores,
                        "steered_top_tokens": steered_tokens,
                        "steered_top_token_scores": steered_scores,
                        "kl_divergence": kl_divergence(base_logits, steered_logits),
                        "top_token_changed": bool(base_tokens[:1] != steered_tokens[:1]),
                        "caveat": (
                            "This is a local hidden state direction probe measured at the unembedding. "
                            "It is not a full causal rollout through all later layers."
                        ),
                    }
                )
        records.append({"case_id": package["case_id"], "evidence_items": items})
    write_jsonl(records, args.output)
    print(f"Wrote steering vector evidence for {len(records)} cases to {args.output}")


if __name__ == "__main__":
    main()
