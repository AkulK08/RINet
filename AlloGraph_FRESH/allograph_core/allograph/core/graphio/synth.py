from __future__ import annotations
import numpy as np
from .types import GraphBundle

def synthetic_rin(
    n: int = 60,
    k: int = 4,
    seed: int = 0,
    ring_weight: float = 1.0,
    long_range_prob: float = 0.05,
    long_range_weight: float = 0.5,
) -> GraphBundle:
    """
    Deterministic synthetic residue interaction network (RIN).
    - ring + local k-neighborhood + sparse long-range edges (seeded)
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=float)

    # ring edges
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = ring_weight
        A[j, i] = ring_weight

    # local neighborhood
    for i in range(n):
        for d in range(2, k + 1):
            j = (i + d) % n
            A[i, j] = ring_weight
            A[j, i] = ring_weight

    # long-range sparse edges (seeded so deterministic)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < long_range_prob:
                A[i, j] = max(A[i, j], long_range_weight)
                A[j, i] = max(A[j, i], long_range_weight)

    meta = {
        "source": "synthetic_rin",
        "n": n,
        "k": k,
        "seed": seed,
    }
    return GraphBundle(A=A, meta=meta, name=f"synthetic_rin(n={n},seed={seed})")
