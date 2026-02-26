from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: np.ndarray | list[float], q: float = 0.05) -> dict[str, np.ndarray]:
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if n == 0:
        return {"q_values": np.array([]), "reject": np.array([], dtype=bool)}

    order = np.argsort(p)
    ranked = p[order]
    q_vals_ranked = np.empty(n, dtype=np.float64)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        running = min(running, val)
        q_vals_ranked[i] = running
    q_vals = np.empty(n, dtype=np.float64)
    q_vals[order] = np.clip(q_vals_ranked, 0.0, 1.0)
    reject = q_vals <= q
    return {"q_values": q_vals, "reject": reject}
