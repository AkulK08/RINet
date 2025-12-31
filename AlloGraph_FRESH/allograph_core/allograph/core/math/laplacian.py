from __future__ import annotations
import numpy as np

def graph_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Unnormalized Laplacian L = D - A
    """
    deg = np.sum(A, axis=1)
    D = np.diag(deg)
    return D - A
