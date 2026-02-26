"""Verify Global FDR survivorship across all evaluated layers.

The reviewer noted that FDR (Benjamini-Hochberg) was applied per-layer,
which inflates the false discovery rate when testing multiple layers.
This script aggregates all p-values from all evaluated layers,
applies BH globally over the pooled hypotheses, and reports
which features survive at q < 0.05.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from analysis.fdr import benjamini_hochberg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/polarization", type=str)
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 12, 16, 20, 24])
    parser.add_argument("--q", type=float, default=0.05, help="FDR threshold")
    parser.add_argument("--tag", default="dev")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    
    all_dfs = []
    
    print(f"Aggregating p-values across layers: {args.layers}")
    for layer in args.layers:
        suffix = f"_{args.tag}" if args.tag else ""
        csv_path = results_dir / f"layer{layer}{suffix}_delta.csv"
        
        if not csv_path.exists():
            print(f"Warning: Missing CSV for layer {layer} at {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        if "p_value" not in df.columns:
            print(f"Warning: 'p_value' column missing in layer {layer}")
            continue
            
        df["layer"] = layer
        all_dfs.append(df)
        
    if not all_dfs:
        print("No valid CSVs found with p-values.")
        return
        
    global_df = pd.concat(all_dfs, ignore_index=True)
    total_hypotheses = len(global_df)
    
    print(f"\nTotal hypotheses pooled: {total_hypotheses:,}")
    
    # Apply Benjamini-Hochberg globally
    p_values = global_df["p_value"].values
    fdr_result = benjamini_hochberg(p_values, q=args.q)
    
    global_df["global_q_value"] = fdr_result["q_values"]
    global_df["global_fdr_reject"] = fdr_result["reject"]
    
    # Filter survivors
    survivors = global_df[global_df["global_fdr_reject"]]
    total_survivors = len(survivors)
    print(f"Total global survivors (q < {args.q}): {total_survivors}")
    
    print("\nSurvivors per layer:")
    layer_counts = survivors["layer"].value_counts().sort_index()
    for layer in args.layers:
        count = layer_counts.get(layer, 0)
        print(f"  Layer {layer}: {count} features")
        
    if 24 in args.layers:
        l24_orig_count = 42
        l24_new_count = layer_counts.get(24, 0)
        print(f"\nLayer 24 comparison:")
        print(f"  Original (per-layer) FDR: {l24_orig_count}")
        print(f"  Global FDR: {l24_new_count}")
        if l24_new_count == l24_orig_count:
            print("  ✅ All 42 original Layer 24 features SURVIVED global FDR correction!")
        else:
            print(f"  ⚠️ Lost {l24_orig_count - l24_new_count} features under global correction.")
            
    # Save the global results
    out_path = results_dir / f"global_fdr{f'_{args.tag}' if args.tag else ''}_results.csv"
    global_df.to_csv(out_path, index=False)
    print(f"\nSaved global FDR results to: {out_path}")
    

if __name__ == "__main__":
    main()
