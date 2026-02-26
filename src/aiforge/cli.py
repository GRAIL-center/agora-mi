from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aiforge.ablation import SAEFeatureEditor, run_pilot_ablation
from aiforge.data import build_contrastive_corpora, parse_tag_list
from aiforge.io_utils import read_jsonl
from aiforge.modeling import extract_last_token_residuals, load_causal_lm
from aiforge.polarization import compute_polarization_table
from aiforge.sae import load_sae_backend, save_features_npz


def _read_texts(path: str, text_field: str, max_samples: int | None) -> list[str]:
    rows = read_jsonl(path)
    texts = [
        str(row[text_field]).strip()
        for row in rows
        if isinstance(row.get(text_field), str) and str(row[text_field]).strip()
    ]
    if max_samples is not None:
        return texts[:max_samples]
    return texts


def cmd_build_contrastive(args: argparse.Namespace) -> int:
    stats = build_contrastive_corpora(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        safe_tags=parse_tag_list(args.safe_tags),
        innov_tags=parse_tag_list(args.innov_tags),
        id_field=args.id_field,
        text_field=args.text_field,
        tag_field=args.tag_field,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    print("Contrastive corpora built.")
    for k in sorted(stats):
        print(f"{k}: {stats[k]}")
    return 0


def cmd_extract_residuals(args: argparse.Namespace) -> int:
    safe_texts = _read_texts(args.safe_jsonl, args.text_field, args.max_per_class)
    innov_texts = _read_texts(args.innov_jsonl, args.text_field, args.max_per_class)
    prompts = safe_texts + innov_texts
    labels = np.array(["safe"] * len(safe_texts) + ["innov"] * len(innov_texts))
    if not prompts:
        raise ValueError("No prompts loaded from safe/innov JSONL files.")

    loaded = load_causal_lm(args.model_name, device=args.device, dtype=args.dtype)
    residuals, lengths = extract_last_token_residuals(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        device=loaded.device,
        texts=prompts,
        layer=args.layer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        residuals=residuals,
        lengths=lengths,
        labels=labels,
        prompts=np.array(prompts, dtype=object),
        layer=np.array([args.layer], dtype=np.int64),
        model_name=np.array([args.model_name], dtype=object),
    )
    print(f"Saved residuals: {out_path}")
    print(f"Residual shape: {residuals.shape}")
    return 0


def cmd_encode_sae(args: argparse.Namespace) -> int:
    arr = np.load(args.residuals_npz, allow_pickle=True)
    if "residuals" not in arr.files:
        raise ValueError(f"{args.residuals_npz} must include `residuals` array.")
    residuals = np.asarray(arr["residuals"], dtype=np.float32)
    labels = np.asarray(arr["labels"]).astype(str) if "labels" in arr.files else np.array([])
    prompts = np.asarray(arr["prompts"], dtype=object) if "prompts" in arr.files else None

    sae = load_sae_backend(
        sae_npz=args.sae_npz,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
    )
    if residuals.shape[1] != sae.d_model:
        raise ValueError(
            f"Residual dimension {residuals.shape[1]} != SAE d_model {sae.d_model}"
        )

    features = sae.encode(residuals)
    save_features_npz(output_path=args.output_npz, features=features, labels=labels, prompts=prompts)

    mean_activation = features.mean(axis=0)
    top_idx = np.argsort(mean_activation)[::-1][: args.top_k]
    print(f"Saved SAE features: {args.output_npz}")
    print(f"Feature shape: {features.shape}")
    print("Top active features by mean activation:")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"{rank:>2}. feature={idx} mean={mean_activation[idx]:.6f}")
    return 0


def cmd_polarization(args: argparse.Namespace) -> int:
    arr = np.load(args.features_npz, allow_pickle=True)
    if "features" not in arr.files or "labels" not in arr.files:
        raise ValueError(f"{args.features_npz} must include `features` and `labels` arrays.")
    features = np.asarray(arr["features"], dtype=np.float32)
    labels = np.asarray(arr["labels"]).astype(str)

    df = compute_polarization_table(features, labels)
    df_top = df.head(args.top_k).copy()

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_top.to_csv(out_path, index=False)
    print(f"Saved top-k polarization table: {out_path}")
    print(df_top.head(10).to_string(index=False))

    if args.full_csv:
        full_path = Path(args.full_csv)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(full_path, index=False)
        print(f"Saved full polarization table: {full_path}")
    return 0


def cmd_pilot_ablation(args: argparse.Namespace) -> int:
    safe_rows = read_jsonl(args.safe_jsonl)
    safe_prompts = [
        str(row[args.text_field]).strip()
        for row in safe_rows
        if isinstance(row.get(args.text_field), str) and str(row[args.text_field]).strip()
    ]
    if args.max_samples is not None:
        safe_prompts = safe_prompts[: args.max_samples]
    if not safe_prompts:
        raise ValueError("No safe prompts found.")

    pol = pd.read_csv(args.polarization_csv)
    if "feature_index" not in pol.columns or "delta" not in pol.columns:
        raise ValueError("polarization CSV must contain `feature_index` and `delta` columns.")
    safety = pol[pol["delta"] > 0].sort_values("delta", ascending=False)
    feature_idx = safety["feature_index"].astype(int).head(args.k).to_numpy()
    if feature_idx.size == 0:
        raise ValueError("No positive-delta safety features found in polarization CSV.")

    loaded = load_causal_lm(args.model_name, device=args.device, dtype=args.dtype)
    sae = load_sae_backend(
        sae_npz=args.sae_npz,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
    )
    editor = SAEFeatureEditor(sae=sae, feature_idx=feature_idx)
    df = run_pilot_ablation(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        device=loaded.device,
        prompts=safe_prompts,
        target_token=args.target_token,
        layer=args.layer,
        editor=editor,
        max_length=args.max_length,
    )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved pilot ablation results: {out_path}")
    print(
        "Delta logprob summary: "
        f"mean={df['delta_logprob'].mean():.6f} std={df['delta_logprob'].std(ddof=1):.6f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiforge",
        description="Safety vs innovation circuit analysis pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-contrastive", help="Build Dsafe/Dinnov train-dev-test splits.")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--safe-tags", required=True, help="Comma-separated safe tags")
    p.add_argument("--innov-tags", required=True, help="Comma-separated innovation tags")
    p.add_argument("--id-field", default="document_id")
    p.add_argument("--text-field", default="text")
    p.add_argument("--tag-field", default="tags")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--dev-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_build_contrastive)

    p = sub.add_parser(
        "extract-residuals",
        help="Run a model forward pass and save residual stream vectors.",
    )
    p.add_argument("--safe-jsonl", required=True)
    p.add_argument("--innov-jsonl", required=True)
    p.add_argument("--text-field", default="text")
    p.add_argument("--model-name", default="google/gemma-2-2b")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--output-npz", required=True)
    p.set_defaults(func=cmd_extract_residuals)

    p = sub.add_parser(
        "encode-sae",
        help="Encode residual vectors with SAE and save feature activations.",
    )
    p.add_argument("--residuals-npz", required=True)
    p.add_argument("--sae-npz", default=None)
    p.add_argument("--sae-release", default=None)
    p.add_argument("--sae-id", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output-npz", required=True)
    p.set_defaults(func=cmd_encode_sae)

    p = sub.add_parser(
        "polarization",
        help="Compute feature polarization delta and save top-k features.",
    )
    p.add_argument("--features-npz", required=True)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--full-csv", default=None)
    p.set_defaults(func=cmd_polarization)

    p = sub.add_parser(
        "pilot-ablation",
        help="Ablate top safety features and measure target-token logprob shift.",
    )
    p.add_argument("--safe-jsonl", required=True)
    p.add_argument("--text-field", default="text")
    p.add_argument("--model-name", default="google/gemma-2-2b")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--sae-npz", default=None)
    p.add_argument("--sae-release", default=None)
    p.add_argument("--sae-id", default=None)
    p.add_argument("--polarization-csv", required=True)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--target-token", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--output-csv", required=True)
    p.set_defaults(func=cmd_pilot_ablation)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
