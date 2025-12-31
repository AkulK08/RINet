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


def run_inverse_tikhonov(
    bundle: GraphBundle,
    observed: np.ndarray,
    dt: float = 1.0,
    steps: int = 60,
    method: str = "heat_kernel",
    lam: float = 1e-3,
    laplacian: str = "normalized",
) -> InverseResult:
    """
    Deterministic inverse source solver using Tikhonov regularization.

    We model observed ≈ K(t) s, where K(t) is the forward diffusion operator
    (e.g., exp(-t L)). The regularized solution is:
        s = argmin ||K s - y||^2 + λ ||s||^2
    with closed-form:
        s = (K^T K + λ I)^(-1) K^T y

    Args
    ----
    bundle
      GraphBundle with adjacency.
    observed
      Observed state vector (length n).
    dt, steps
      Total diffusion time t = dt*steps.
    method
      "heat_kernel" (uses exp(-tL)) or "prop_matrix" (uses power propagation).
    lam
      Regularization strength λ >= 0.
    laplacian
      "normalized" (recommended) or "combinatorial" Laplacian.

    Returns
    -------
    InverseResult with deterministic source_scores and metadata.
    """

    bundle.validate()
    A = bundle.A.astype(float)
    n = A.shape[0]

    # Total diffusion time
    t = float(dt) * int(steps)
    if t < 0:
        raise ValueError("dt*steps must be non-negative.")

    # Build diffusion kernel K
    if method == "heat_kernel":
        # Normalized Laplacian
        if laplacian == "normalized":
            L = _normalized_laplacian(A)
        elif laplacian == "combinatorial":
            L = graph_laplacian(A)
        else:
            raise ValueError("Unknown laplacian option.")

        # Spectral expm: K = exp(-t L)
        evals, evecs = eigh(L)
        exp_evals = np.exp(-t * evals)
        K = (evecs * exp_evals) @ evecs.T
    else:
        raise ValueError(f"Unknown method {method}")

    # Regularized inversion: s = (K^T K + λ I)^(-1) K^T y
    y = np.asarray(observed, dtype=float).flatten()
    KTK = K.T @ K
    reg = lam * np.eye(n)
    M = KTK + reg
    rhs = K.T @ y

    # Solve linear system (deterministic, no randomization)
    s = np.linalg.solve(M, rhs)

    meta: Dict[str, Any] = dict(bundle.meta)
    meta.update({
        "inverse_method": "tikhonov",
        "lam": float(lam),
        "t": float(t),
        "method": method,
        "laplacian": laplacian,
        "n": n,
    })

    return InverseResult(source_scores=s, meta=meta)


def _normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Symmetric normalized Laplacian: I - D^{-1/2} A D^{-1/2}.
    """
    deg = np.sum(A, axis=1)
    inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    D_inv_sqrt = np.diag(inv_sqrt)
    I = np.eye(A.shape[0], dtype=float)
    return I - D_inv_sqrt @ A @ D_inv_sqrt
