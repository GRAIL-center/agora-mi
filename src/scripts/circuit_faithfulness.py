from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from _common import ensure_dir, read_config, setup_logging
from data.io import read_jsonl
from model.prompt import build_prompts, load_prompt_config
from scripts.interference_clamp import _sequence_logprob_from_logits, _resolve_token_ids

_CT_AVAILABLE = True
try:
    from circuit_tracer import ReplacementModel
except ImportError:
    _CT_AVAILABLE = False


def _parse_nodes(graph_path: str | Path) -> list[tuple[int, int]]:
    """Parse node IDs like 'L12_F345' into (layer, feature_idx)."""
    with Path(graph_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = []
    for node_id in data.get("nodes", {}).keys():
        # Expect formatted like "L{layer}_F{feature_idx}"
        parts = node_id.split("_")
        if len(parts) == 2 and parts[0].startswith("L") and parts[1].startswith("F"):
            layer = int(parts[0][1:])
            f_idx = int(parts[1][1:])
            nodes.append((layer, f_idx))
    return nodes


def _get_target_prob(model, tokens, target_ids) -> float:
    with torch.no_grad():
        logits = model(tokens)
    prompt_pos = len(tokens) - 1
    logp = _sequence_logprob_from_logits(
        logits.unsqueeze(0),
        prompt_pos=prompt_pos, 
        target_token_ids=target_ids
    )
    return float(logp.item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Circuit Faithfulness (CPR & CMD)")
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, required=True, help="Layer of the extracted graph (e.g. 24)")
    parser.add_argument("--safe_jsonl", default="data/processed/dsafe_train.jsonl")
    parser.add_argument("--innov_jsonl", default="data/processed/dinnov_train.jsonl")
    parser.add_argument("--transcoder_set", default="mntss/gemma-scope-transcoders")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    setup_logging("circuit_faithfulness")
    if not _CT_AVAILABLE:
        logging.error("circuit-tracer is not installed.")
        return 1

    cfg = read_config(args.config)
    results_dir = Path(cfg.get("results_dir", "results"))
    graph_path = results_dir / "graphs" / f"circuit_innov_layer{args.layer}.json"
    
    if not graph_path.exists():
        logging.error(f"Circuit graph not found: {graph_path}")
        return 1

    circuit_nodes = _parse_nodes(graph_path)
    logging.info(f"Loaded {len(circuit_nodes)} nodes from {graph_path.name}")

    safe_rows = read_jsonl(args.safe_jsonl)[:args.max_samples]
    innov_rows = read_jsonl(args.innov_jsonl)[:args.max_samples]
    
    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/prompt_templates.yaml"))
    template = str(prompt_cfg["template_v1"])
    use_chat_template = cfg.get("use_chat_template", False)
    
    model_id = cfg.get("model_id", "google/gemma-2-2b")
    logging.info(f"Loading ReplacementModel ({model_id}) with transcoders...")
    model = ReplacementModel.from_pretrained(
        model_name=model_id,
        transcoder_set=args.transcoder_set,
        backend="transformerlens",
    )
    
    safe_prompts = build_prompts(safe_rows, template, tokenizer=model.tokenizer, use_chat_template=use_chat_template)
    innov_prompts = build_prompts(innov_rows, template, tokenizer=model.tokenizer, use_chat_template=use_chat_template)

    target_tokens = prompt_cfg.get("target_tokens", {})
    inc_label = str(target_tokens.get("INCENTIVE", "INCENTIVE"))
    res_label = str(target_tokens.get("RESTRICTION", "RESTRICTION"))
    inc_ids = _resolve_token_ids(model.tokenizer, inc_label)
    res_ids = _resolve_token_ids(model.tokenizer, res_label)

    # We evaluate P(Safe | ...) which corresponds roughly to the RESTRICTION prediction 
    # for a safe behavior (refusal). Wait, Agora dataset SAFE = RESTRICTION? 
    # In Agora, target is mapping safe behaviors to restriction blocks.
    # Let's track logprobs for BOTH and compute normalized probability just like Dose-Response.
    def get_prob(logits, tokens) -> float:
        prompt_pos = len(tokens) - 1
        lp_inc = _sequence_logprob_from_logits(logits.unsqueeze(0), prompt_pos=prompt_pos, target_token_ids=inc_ids).item()
        lp_res = _sequence_logprob_from_logits(logits.unsqueeze(0), prompt_pos=prompt_pos, target_token_ids=res_ids).item()
        m = max(lp_inc, lp_res)
        return float(np.exp(lp_res - m) / (np.exp(lp_inc - m) + np.exp(lp_res - m)))

    results = []
    
    logging.info("Evaluating CPR (Sufficiency) and CMD (Necessity)...")
    for i in tqdm(range(len(safe_prompts))):
        p_safe = safe_prompts[i]
        p_innov = innov_prompts[i]
        
        t_safe = model.ensure_tokenized(p_safe)
        t_innov = model.ensure_tokenized(p_innov)
        
        # 1. Base Probs
        logits_safe, acts_safe = model.get_activations(t_safe)
        prob_safe_base = get_prob(logits_safe, t_safe)
        
        logits_innov, _ = model.get_activations(t_innov)
        prob_innov_base = get_prob(logits_innov, t_innov)

        # 2. Sufficiency Patching (CPR)
        # We run on Corrupted (Innov), but patch circuit nodes to Clean (Safe) activations.
        suff_interventions = []
        for (L, F) in circuit_nodes:
            # Match pos shape. Since tokens might differ in length, we patch the last token (or all if lengths match)
            # To be safe and align with standard seq-len differences, we patch the *last* token.
            # Transcoder feature shape is [seq, d_transcoder]. We intervene on pos=-2 or -1.
            # In autoregressive models, the final decision happens at the last token.
            pos_safe = len(t_safe) - 1
            pos_innov = len(t_innov) - 1
            val = acts_safe[L, pos_safe, F]
            suff_interventions.append((L, pos_innov, F, val))
            
        logits_suff, _ = model.feature_intervention(
            t_innov, suff_interventions, apply_activation_function=True
        )
        prob_suff = get_prob(logits_suff, t_innov)

        # 3. Necessity Patching (CMD)
        # We run on Clean (Safe), but zero-ablate the circuit nodes.
        nece_interventions = []
        for (L, F) in circuit_nodes:
            pos_safe = len(t_safe) - 1
            nece_interventions.append((L, pos_safe, F, 0.0))
            
        logits_nece, _ = model.feature_intervention(
            t_safe, nece_interventions, apply_activation_function=True
        )
        prob_nece = get_prob(logits_nece, t_safe)
        
        results.append({
            "idx": i,
            "prob_clean": prob_safe_base,
            "prob_corrupt": prob_innov_base,
            "prob_suff": prob_suff,
            "prob_nece": prob_nece
        })
        
    df = pd.DataFrame(results)
    
    # CPR = (Suff - Corrupt) / (Clean - Corrupt)
    # CMD = (Clean - Nece) / (Clean - Corrupt)
    denom = (df["prob_clean"] - df["prob_corrupt"]).replace(0, np.nan)
    df["CPR"] = (df["prob_suff"] - df["prob_corrupt"]) / denom
    df["CMD"] = (df["prob_clean"] - df["prob_nece"]) / denom
    
    out_csv = results_dir / f"circuit_faithfulness_layer{args.layer}.csv"
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    
    logging.info(f"Faithfulness evaluation complete.")
    logging.info(f"Mean CPR (Sufficiency): {df['CPR'].mean():.4f}")
    logging.info(f"Mean CMD (Necessity):   {df['CMD'].mean():.4f}")
    logging.info(f"Saved results to {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
