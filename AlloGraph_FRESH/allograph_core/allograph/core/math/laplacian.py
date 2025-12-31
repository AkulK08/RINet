from __future__ import annotations
import numpy as np
from typing import Literal

LaplacianType = Literal["unnormalized", "normalized_sym", "normalized_rw"]

def graph_laplacian(
    A: np.ndarray,
    kind: LaplacianType = "unnormalized",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute a graph Laplacian matrix.
    
    Args
    ----
    A
        Adjacency matrix (n×n). Should be symmetric for undirected graphs.
    kind
        Type of Laplacian to compute:
        - "unnormalized": L = D - A
        - "normalized_sym": L_sym = I - D^{-1/2} A D^{-1/2}
        - "normalized_rw": L_rw = I - D^{-1} A
    eps
        Small value to avoid divide-by-zero for isolated nodes.

    Returns
    -------
    Laplacian matrix (n×n).
    
    Notes
    -----
    - The unnormalized Laplacian is symmetric and positive semidefinite.
    - The normalized symmetric Laplacian has eigenvalues in [0, 2], which
      can be beneficial for spectral methods. :contentReference[oaicite:6]{index=6}
    - The random walk Laplacian is stochastic in form and can be useful
      for walk-based dynamics. :contentReference[oaicite:7]{index=7}
    """
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError(f"Adjacency must be square (got shape {A.shape}).")

    deg = np.sum(A, axis=1)
    
    if kind == "unnormalized":
        D = np.diag(deg)
        return D - A

    # degrees for normalization
    inv_sqrt_deg = np.zeros_like(deg)
    inv_deg = np.zeros_like(deg)
    
    mask_pos = deg > 0
    inv_sqrt_deg[mask_pos] = 1.0 / np.sqrt(deg[mask_pos] + eps)
    inv_deg[mask_pos] = 1.0 / (deg[mask_pos] + eps)

    if kind == "normalized_sym":
        D_inv_sqrt = np.diag(inv_sqrt_deg)
        I = np.eye(n, dtype=float)
        # L_sym = I - D^{-1/2} A D^{-1/2}
        return I - D_inv_sqrt @ A @ D_inv_sqrt

    if kind == "normalized_rw":
        D_inv = np.diag(inv_deg)
        I = np.eye(n, dtype=float)
        # L_rw = I - D^{-1} A
        return I - D_inv @ A

    # fallback: should not happen due to type annotation
    raise ValueError(f"Unknown Laplacian kind '{kind}'.")
