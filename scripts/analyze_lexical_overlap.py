"""Compare surviving SAE features between original and lexical control datasets."""

import json
from pathlib import Path

def main():
    results_dir = Path("results/polarization")
    
    # Paths for original vs lexical control top-K lists
    orig_safe = results_dir / "layer24_train_topk_safe.json"
    lex_safe = results_dir / "layer24_lexical_control_topk_safe.json"
    
    orig_innov = results_dir / "layer24_train_topk_innov.json"
    lex_innov = results_dir / "layer24_lexical_control_topk_innov.json"
    
    if not all(p.exists() for p in [orig_safe, lex_safe, orig_innov, lex_innov]):
        print("Missing JSON files for comparison. Ensure compute_polarization.py has finished for both.")
        return
        
    def get_feats(path: Path) -> set[int]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data["feature_ids"])
            
    # Load feature sets
    os_feats = get_feats(orig_safe)
    ls_feats = get_feats(lex_safe)
    
    oi_feats = get_feats(orig_innov)
    li_feats = get_feats(lex_innov)
    
    # Compute Safe overlaps (the critical 42 features)
    safe_intersect = os_feats.intersection(ls_feats)
    safe_percent = len(safe_intersect) / len(os_feats) * 100 if len(os_feats) > 0 else 0
    
    # Compute Innov overlaps
    innov_intersect = oi_feats.intersection(li_feats)
    innov_percent = len(innov_intersect) / len(oi_feats) * 100 if len(oi_feats) > 0 else 0
    
    print("=== Lexical Control Overlap Analysis ===")
    print("\n[Safety/Brake Features]")
    print(f"Original Count: {len(os_feats)}")
    print(f"Controlled Count: {len(ls_feats)}")
    print(f"Overlap: {len(safe_intersect)} features ({safe_percent:.1f}% retention)")
    if safe_percent > 80:
        print("✅ SUCCESS: Strong abstract concept representation. Features are robust to lexical confounds.")
    else:
        print("⚠️ WARNING: High sensitivity to exact structural words identified.")
        
    print("\n[Innovation/Accelerator Features]")
    print(f"Original Count: {len(oi_feats)}")
    print(f"Controlled Count: {len(li_feats)}")
    print(f"Overlap: {len(innov_intersect)} features ({innov_percent:.1f}% retention)")
    if innov_percent > 80:
        print("✅ SUCCESS: Strong abstract concept representation. Features are robust to lexical confounds.")
    elif innov_percent > 0:
        print("⚠️ WARNING: High sensitivity to exact structural words identified.")
        
if __name__ == "__main__":
    main()
