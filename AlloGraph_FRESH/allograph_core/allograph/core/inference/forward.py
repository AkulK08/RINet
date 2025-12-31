from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..graphio.types import GraphBundle
from ..math.laplacian import graph_laplacian
from scipy.linalg import eigh


@dataclass
class InverseResult:
    source_scores: np.ndarray
    meta: Dict[str, Any]


def run_inverse_landweber(
    bundle: GraphBundle,
    observed: np.ndarray,
    dt: float = 1.0,
    steps: int = 100,
    tau: float = 1e-2,
    method: str = "heat_kernel",
    laplacian: str = "normalized",
) -> InverseResult:
    """
    Deterministic inverse using Landweber iteration (iterative regularization).
    This solver attempts to solve s ≈ argmin ||K s − y||^2 via s_{k+1} = s_k + τ K^T(y - K s_k).

    Args
    ----
    bundle
        GraphBundle with adjacency A.
    observed
        Observed propagated state y.
    dt, steps
        Define total diffusion time for forward kernel (if used).
    tau
        Step size for Landweber iteration; should be < 2 / ||K||^2 for stability.
    method
        "heat_kernel" uses exp(-tL) as forward operator.
    laplacian
        "normalized" or "combinatorial" graph Laplacian for diffusion.

    Returns
    -------
    InverseResult with source_scores and metadata.
    """
    bundle.validate()
    A = bundle.A.astype(float)
    n = A.shape[0]

    # Build forward operator K
    t = float(dt) * int(steps)
    if method == "heat_kernel":
        if laplacian == "normalized":
            L = _normalized_laplacian(A)
        elif laplacian == "combinatorial":
            L = graph_laplacian(A)
        else:
            raise ValueError(f"Unknown laplacian {laplacian}")
        evals, evecs = eigh(L)
        K = (evecs * np.exp(-t * evals)) @ evecs.T
    else:
        raise ValueError(f"Unknown inverse method {method}")

    # Landweber iteration
    y = np.asarray(observed, dtype=float).flatten()
    s = np.zeros(n, dtype=float)
    KT = K.T

    for k in range(int(steps)):
        r = y - (K @ s)  # residual
        s = s + tau * (KT @ r)

    meta: Dict[str, Any] = dict(bundle.meta)
    meta.update({
        "inverse_method": "landweber",
        "method": method,
        "laplacian": laplacian,
        "steps": int(steps),
        "dt": float(dt),
        "tau": float(tau),
        "time": float(t),
    })
    return InverseResult(source_scores=s, meta=meta)


def _normalized_laplacian(A: np.ndarray) -> np.ndarray:
    deg = np.sum(A, axis=1)
    inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    D_inv_sqrt = np.diag(inv_sqrt)
    I = np.eye(A.shape[0], dtype=float)
    return I - D_inv_sqrt @ A @ D_inv_sqrt
