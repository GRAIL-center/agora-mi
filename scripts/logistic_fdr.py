import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from transformers import AutoTokenizer

def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05):
    """Simple BH FDR."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    thresholds = (np.arange(1, n + 1) / n) * q
    reject = sorted_p_values <= thresholds
    
    max_reject_idx = np.where(reject)[0]
    if len(max_reject_idx) > 0:
        max_idx = max_reject_idx[-1]
        reject_mask = np.zeros(n, dtype=bool)
        reject_mask[sorted_indices[:max_idx + 1]] = True
    else:
        reject_mask = np.zeros(n, dtype=bool)
        
    return reject_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--topk_json", default="results/polarization/layer24_train_topk_safe.json")
    parser.add_argument("--features_npz", default="results/polarization/layer24_train_features.npz")
    args = parser.parse_args()
    
    npz = np.load(args.features_npz, allow_pickle=True)
    f_safe = npz["safe_features"]
    f_innov = npz["innov_features"]
    text_safe = npz["safe_text"]
    text_innov = npz["innov_text"]
    
    # Get sequence lengths
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    len_safe = [len(tokenizer.encode(t)) for t in text_safe]
    len_innov = [len(tokenizer.encode(t)) for t in text_innov]
    
    X_len = np.concatenate([len_safe, len_innov])
    y_class = np.concatenate([np.ones(len(f_safe)), np.zeros(len(f_innov))])
    
    f_all = np.concatenate([f_safe, f_innov], axis=0)
    
    n_features = f_all.shape[1]
    
    print(f"Total features to evaluate: {n_features}")
    
    # Load the targeted 42 features to make sure they survive
    with open(args.topk_json) as f:
        target_ids = json.load(f)["feature_ids"]
        
    p_values = np.ones(n_features)
    
    for i in tqdm(range(n_features)):
        # If feature is completely dead (all exact 0), skip
        if np.max(f_all[:, i]) == 0:
            continue
            
        X = sm.add_constant(np.column_stack((f_all[:, i], X_len)))
        try:
            model = sm.Logit(y_class, X)
            result = model.fit(disp=False, method='bfgs')
            # Variable 1 is the feature activation
            p_values[i] = result.pvalues[1]
        except Exception:
            # Convergence failure or separation
            p_values[i] = 1.0

    reject = benjamini_hochberg(p_values, q=0.05)
    
    target_reject = reject[target_ids]
    
    print(f"\n--- Covariate-Adjusted FDR Results (Layer {args.layer}) ---")
    print(f"Total features surviving FDR: {np.sum(reject)}")
    print(f"Target 42 features surviving FDR: {np.sum(target_reject)} / {len(target_ids)}")
    
    # Output to a JSON
    out_dict = {
        "sequence_length_adjusted": True,
        "total_survivors": int(np.sum(reject)),
        "target_survivors": int(np.sum(target_reject)),
        "target_ids": target_ids,
        "target_p_values": p_values[target_ids].tolist()
    }
    
    out_path = f"results/polarization/layer{args.layer}_logistic_fdr.json"
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
