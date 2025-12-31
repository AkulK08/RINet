from __future__ import annotations
import numpy as np
from .base import BaseDynamicsModel, DynamicsResult

class MarkovAllostericDynamics(BaseDynamicsModel):
    """
    Deterministic network propagation model for allosteric dynamics.
    
    This model propagates a signal vector x0 over a weighted network A
    using a normalized transition matrix and a damping factor alpha,
    while remaining fully deterministic.

    The propagation rule:
        x_{t+1} = alpha * (P @ x_t) + (1 - alpha) * x_t
    where P is a row-normalized transition matrix.
    """

    name = "markov_allosteric"

    def __init__(self, return_history: bool = False):
        """
        Parameters
        ----------
        return_history : bool
            If True, return the full history of the state vector over time.
        """
        self.return_history = return_history

    def _normalize_matrix(self, A: np.ndarray) -> np.ndarray:
        """
        Normalize adjacency matrix A row-wise to form a transition matrix.
        Avoids rows of all zeros by adding a tiny epsilon.
        """
        A = np.asarray(A, dtype=float)
        row_sums = A.sum(axis=1, keepdims=True) + 1e-12
        return A / row_sums

    def run(
        self,
        A: np.ndarray,
        x0: np.ndarray,
        steps: int = 50,
        dt: float = 1.0,
        alpha: float = 0.85,
    ) -> DynamicsResult:
        """
        Propagate the signal over the network deterministically.

        Parameters
        ----------
        A : np.ndarray
            Weighted adjacency matrix (n×n).
        x0 : np.ndarray
            Initial state vector (length n).
        steps : int
            Number of propagation steps.
        dt : float
            Time scaling factor (informational).
        alpha : float
            Weighting factor controlling diffusion vs retention.
            0 < alpha <= 1. (alpha=1 is pure diffusion).

        Returns
        -------
        DynamicsResult
            Contains final state (and optional history) plus metadata.
        """
        # Ensure correct array shapes
        A = np.asarray(A, dtype=float)
        x0 = np.asarray(x0, dtype=float).flatten()

        # Build row-normalized transition matrix
        P = self._normalize_matrix(A)

        # Initialize state
        x = x0.copy()
        history = [x.copy()] if self.return_history else None

        # Iterative propagation
        for _ in range(int(steps)):
            x_next = alpha * (P @ x) + (1.0 - alpha) * x
            # Re-normalize to maintain scale
            norm = np.linalg.norm(x_next, ord=1) + 1e-12
            x = x_next / norm

            if self.return_history:
                history.append(x.copy())

        meta = {
            "steps": int(steps),
            "dt": float(dt),
            "alpha": float(alpha),
            "model": self.name,
        }

        return DynamicsResult(state=x, history=history, meta=meta)
