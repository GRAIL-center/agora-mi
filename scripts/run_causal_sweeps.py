from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from _common import ensure_dir, read_config, setup_logging


def _run(cmd: list[str]) -> None:
    logging.info("RUN: %s", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _write_temp_run_config(base_cfg: dict, out_path: Path, *, topk: int) -> Path:
    cfg = dict(base_cfg)
    cfg["topk"] = int(topk)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def _summary_from_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("effect_summary", {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layers", default="12,16,20,24")
    parser.add_argument("--ks", default="8,16,32,64")
    parser.add_argument("--selection_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--sae_npz", default=None)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--bootstrap_B", type=int, default=None)
    parser.add_argument("--perm_N", type=int, default=None)
    args = parser.parse_args()

    setup_logging("run_causal_sweeps")
    cfg = read_config(args.config)
    layers = _parse_int_list(args.layers)
    ks = _parse_int_list(args.ks)
    if not layers or not ks:
        raise ValueError("layers and ks must be non-empty.")

    max_k = max(ks)
    tmp_cfg_path = Path("logs") / "run_sweeps_temp_config.yaml"
    _write_temp_run_config(cfg, tmp_cfg_path, topk=max_k)

    out_rows = []
    for layer in layers:
        safe_sel = f"data/processed/dsafe_{args.selection_split}.jsonl"
        innov_sel = f"data/processed/dinnov_{args.selection_split}.jsonl"
        tag = args.selection_split
        cmd = [
            sys.executable,
            "scripts/compute_polarization.py",
            "--config",
            str(tmp_cfg_path),
            "--layer",
            str(layer),
            "--safe_jsonl",
            safe_sel,
            "--innov_jsonl",
            innov_sel,
            "--tag",
            tag,
        ]
        if args.sae_npz:
            cmd += ["--sae_npz", args.sae_npz]
        _run(cmd)

        # Sanity check on the same selection split features.
        sanity_cmd = [
            sys.executable,
            "scripts/sanity_checks.py",
            "--config",
            str(tmp_cfg_path),
            "--layer",
            str(layer),
            "--safe_jsonl",
            safe_sel,
            "--innov_jsonl",
            innov_sel,
            "--features_npz",
            str(Path(cfg.get("results_dir", "results")) / "polarization" / f"layer{layer}_{tag}_features.npz"),
            "--topk_safe_json",
            str(Path(cfg.get("results_dir", "results")) / "polarization" / f"layer{layer}_{tag}_topk_safe.json"),
            "--topk_innov_json",
            str(Path(cfg.get("results_dir", "results")) / "polarization" / f"layer{layer}_{tag}_topk_innov.json"),
        ]
        _run(sanity_cmd)

        topk_safe_json = (
            Path(cfg.get("results_dir", "results"))
            / "polarization"
            / f"layer{layer}_{tag}_topk_safe.json"
        )

        for k in ks:
            summary_out = (
                Path(cfg.get("results_dir", "results"))
                / "pilot_ablation"
                / f"effect_summary_layer{layer}_k{k}_{args.eval_split}.json"
            )
            sample_out = (
                Path(cfg.get("results_dir", "results"))
                / "pilot_ablation"
                / f"effects_layer{layer}_k{k}_{args.eval_split}.csv"
            )
            cmd = [
                sys.executable,
                "scripts/pilot_ablation.py",
                "--config",
                str(tmp_cfg_path),
                "--layer",
                str(layer),
                "--safe_jsonl",
                f"data/processed/dsafe_{args.eval_split}.jsonl",
                "--topk_safe_json",
                str(topk_safe_json),
                "--k",
                str(k),
                "--max_samples",
                str(args.max_samples),
                "--summary_out",
                str(summary_out),
                "--sample_out",
                str(sample_out),
            ]
            if args.sae_npz:
                cmd += ["--sae_npz", args.sae_npz]
            if args.bootstrap_B is not None:
                cmd += ["--bootstrap_B", str(args.bootstrap_B)]
            if args.perm_N is not None:
                cmd += ["--perm_N", str(args.perm_N)]
            _run(cmd)

            s = _summary_from_json(summary_out)
            out_rows.append(
                {
                    "layer": layer,
                    "k": k,
                    "selection_split": args.selection_split,
                    "eval_split": args.eval_split,
                    "n_samples": s.get("n_samples"),
                    "mean_effect_safe": s.get("mean_effect"),
                    "mean_effect_random": s.get("random_control_mean_effect"),
                    "mean_delta_safe_minus_random": s.get("paired_delta_mean"),
                    "delta_bootstrap_95_ci": s.get("paired_delta_bootstrap_95_ci"),
                    "delta_permutation_p": s.get("paired_delta_permutation_p"),
                    "summary_json": str(summary_out),
                    "sample_csv": str(sample_out),
                }
            )

    sweep_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "stats")
    table_path = sweep_dir / "sweep_summary.csv"
    md_path = sweep_dir / "sweep_summary.md"
    df = pd.DataFrame(out_rows).sort_values(["layer", "k"]).reset_index(drop=True)
    df.to_csv(table_path, index=False)

    lines = [
        "# Sweep Summary",
        "",
        f"- Layers: {layers}",
        f"- k values: {ks}",
        f"- Selection split: {args.selection_split}",
        f"- Evaluation split: {args.eval_split}",
        "",
        "## Best by |delta safe-random| (descending)",
        "",
    ]
    ranked = df.copy()
    ranked["abs_delta"] = ranked["mean_delta_safe_minus_random"].abs()
    ranked = ranked.sort_values("abs_delta", ascending=False).head(10)
    for _, r in ranked.iterrows():
        lines.append(
            "- "
            f"L={int(r['layer'])}, k={int(r['k'])}, "
            f"delta={r['mean_delta_safe_minus_random']:.6f}, "
            f"p={r['delta_permutation_p']}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Saved sweep CSV: %s", table_path)
    logging.info("Saved sweep markdown: %s", md_path)
    print(table_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
