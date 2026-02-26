import json
from pathlib import Path
from analysis.circuit_graph import CircuitGraph
from analysis.topology import compute_topology_metrics
import numpy as np

def independent_bootstrap_ci(data1, data2, B=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n1, n2 = len(data1), len(data2)
    s1 = rng.choice(data1, size=(B, n1), replace=True)
    s2 = rng.choice(data2, size=(B, n2), replace=True)
    means1 = s1.mean(axis=1)
    means2 = s2.mean(axis=1)
    diffs = means1 - means2
    lower = np.percentile(diffs, 100 * (alpha / 2))
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return float(lower), float(upper)

def independent_permutation_test(data1, data2, N=10000, seed=42):
    rng = np.random.default_rng(seed)
    pool = np.concatenate([data1, data2])
    n1 = len(data1)
    actual_diff = np.abs(np.mean(data1) - np.mean(data2))
    count = 0
    for _ in range(N):
        rng.shuffle(pool)
        perm_mean1 = pool[:n1].mean()
        perm_mean2 = pool[n1:].mean()
        perm_diff = np.abs(perm_mean1 - perm_mean2)
        if perm_diff >= actual_diff:
            count += 1
    return count / N

def test_metric(name, safe_vals, innov_vals):
    s_arr = np.array(safe_vals)
    i_arr = np.array(innov_vals)
    mean_safe = s_arr.mean()
    mean_innov = i_arr.mean()
    delta = mean_safe - mean_innov
    ci_low, ci_high = independent_bootstrap_ci(s_arr, i_arr)
    p_val = independent_permutation_test(s_arr, i_arr)
    print(f"\n--- {name} ---")
    print(f"Safe : {s_arr.round(3)} (mean={mean_safe:.3f})")
    print(f"Innov: {i_arr.round(3)} (mean={mean_innov:.3f})")
    print(f"Mean Delta (Safe - Innov): {delta:.3f}")
    print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"P-value: {p_val:.4f}")

def main():
    base = Path("results/graphs")
    metrics = {"depth": ([], []), "branching": ([], []), "density": ([], [])}

    for i in range(20):
        s_path = base / f"circuit_safe_layer24_off{i}.json"
        if s_path.exists():
            s_graph = CircuitGraph.load_json(s_path)
            s_m = compute_topology_metrics(s_graph.to_dict())
            metrics["depth"][0].append(s_m["depth"])
            metrics["branching"][0].append(s_m["branching_factor"])
            metrics["density"][0].append(s_m["density"])
            
        i_path = base / f"circuit_innov_layer24_off{i}.json"
        if i_path.exists():
            i_graph = CircuitGraph.load_json(i_path)
            i_m = compute_topology_metrics(i_graph.to_dict())
            metrics["depth"][1].append(i_m["depth"])
            metrics["branching"][1].append(i_m["branching_factor"])
            metrics["density"][1].append(i_m["density"])
            
    test_metric("Depth", metrics["depth"][0], metrics["depth"][1])
    test_metric("Branching Factor", metrics["branching"][0], metrics["branching"][1])
    test_metric("Density", metrics["density"][0], metrics["density"][1])

if __name__ == "__main__":
    main()
