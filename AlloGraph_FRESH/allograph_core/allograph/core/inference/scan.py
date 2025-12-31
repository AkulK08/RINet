from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..graphio.types import GraphBundle
from ..math.laplacian import graph_laplacian
from .forward import run_forward


@dataclass
class ScanResult:
    scores: np.ndarray
    meta: Dict[str, Any]


def run_scan(
    bundle: GraphBundle,
    model: str = "diffusion",
    steps: int = 60,
    dt: float = 0.10,
) -> ScanResult:
    """
    Baseline scan (your original): for each node i, seed at i, run forward, and
    record sum(abs(state)). Deterministic but O(N) forward runs.
    """
    bundle.validate()
    n = bundle.n
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        fr = run_forward(bundle, model=model, seed_nodes=[i], steps=steps, dt=dt)
        scores[i] = float(np.sum(np.abs(fr.state)))
    meta = {**bundle.meta, "model": model, "steps": int(steps), "dt": float(dt), "kind": "scan_sumabs"}
    return ScanResult(scores=scores, meta=meta)


# -------------------------
# Advanced deterministic scan
# -------------------------

def _normalized_laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Symmetric normalized Laplacian:
        L_norm = I - D^{-1/2} A D^{-1/2}

    Deterministic. Requires no symmetry strictly, but if A is symmetric,
    L_norm is symmetric PSD (nicer numerics).
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square; got shape {A.shape}")

    n = A.shape[0]
    deg = np.sum(A, axis=1)
    inv_sqrt_deg = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv_sqrt_deg[mask] = 1.0 / np.sqrt(deg[mask])

    D_inv_sqrt = np.diag(inv_sqrt_deg)
    I = np.eye(n, dtype=float)
    return I - D_inv_sqrt @ A @ D_inv_sqrt


def _expm_symmetric(M: np.ndarray) -> np.ndarray:
    """
    Compute exp(M) for a *symmetric* matrix M via eigendecomposition:
        M = Q Λ Q^T  =>  exp(M) = Q exp(Λ) Q^T

    Deterministic (no randomness). Numerical differences are limited to
    floating-point/eigendecomposition tolerance.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Matrix must be square; got shape {M.shape}")

    # If M is not symmetric due to small numerical asymmetry, symmetrize it.
    # This keeps determinism and makes eigh appropriate.
    M = 0.5 * (M + M.T)

    evals, evecs = np.linalg.eigh(M)
    # exp(Λ) applied to columns of evecs: (evecs * exp(evals)) @ evecs.T
    exp_evals = np.exp(evals)
    return (evecs * exp_evals) @ evecs.T


def run_scan_spectral_diffusion(
    bundle: GraphBundle,
    steps: int = 60,
    dt: float = 0.10,
    laplacian: str = "normalized",
    score: str = "sumabs",
) -> ScanResult:
    """
    Advanced deterministic scan using the CLOSED-FORM diffusion operator (heat kernel).

    We solve the continuous-time diffusion/heat equation on the graph:
        dx/dt = -L x
    in closed form:
        x(t) = exp(-t L) x(0)

    If seed is e_i (unit vector), the propagated state is column i of K(t):
        K(t) = exp(-t L)

    So scanning all seeds becomes a *column-wise summary* of K(t), computed once
    (O(n^3) eigendecomposition), rather than n separate forward runs.

    Args:
      bundle: GraphBundle
      steps, dt: define total diffusion time t = steps*dt
      laplacian: "normalized" (recommended) or "combinatorial"
      score: one of:
        - "sumabs": sum_j |K_{j,i}|
        - "l2": ||K[:, i]||_2
        - "mass": sum_j K_{j,i}  (can be informative if kernel preserves mass)

    Determinism:
      - 100% deterministic as implemented (no RNG, no sampling).
      - Only floating-point / eigen numerical tolerance exists (tiny, not stochastic).
    """
    bundle.validate()
    A = np.asarray(bundle.A, dtype=float)
    n = A.shape[0]

    t = float(dt) * int(steps)
    if t < 0:
        raise ValueError("Total diffusion time t = steps*dt must be >= 0.")

    if laplacian == "normalized":
        L = _normalized_laplacian_from_adjacency(A)
    elif laplacian == "combinatorial":
        # Uses your project’s laplacian function for consistency.
        L = graph_laplacian(A).astype(float)
        # For numerical stability with expm via eigh, symmetrize:
        L = 0.5 * (L + L.T)
    else:
        raise ValueError("laplacian must be 'normalized' or 'combinatorial'.")

    # Heat kernel
    K = _expm_symmetric(-t * L)

    if score == "sumabs":
        scores = np.sum(np.abs(K), axis=0)  # per-column
        kind = "scan_heatkernel_sumabs"
    elif score == "l2":
        scores = np.linalg.norm(K, ord=2, axis=0)  # per-column l2
        kind = "scan_heatkernel_l2"
    elif score == "mass":
        scores = np.sum(K, axis=0)
        kind = "scan_heatkernel_mass"
    else:
        raise ValueError("score must be one of: 'sumabs', 'l2', 'mass'.")

    meta: Dict[str, Any] = dict(bundle.meta)
    meta.update({
        "model": "exact_heat_kernel",
        "laplacian": laplacian,
        "steps": int(steps),
        "dt": float(dt),
        "time": float(t),
        "score": score,
        "kind": kind,
        "notes": (
            "Deterministic spectral scan via K(t)=exp(-tL). "
            "No RNG/sampling; only floating-point numerical tolerance."
        ),
    })

    return ScanResult(scores=np.asarray(scores, dtype=float), meta=meta)
