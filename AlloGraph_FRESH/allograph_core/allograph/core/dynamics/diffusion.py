from __future__ import annotations
import numpy as np
from .base import BaseDynamicsModel, DynamicsResult
from ..math.laplacian import graph_laplacian

class ExactDiffusionModel(BaseDynamicsModel):
    """
    Exact graph diffusion dynamics using the matrix exponential of the (normalized)
    Laplacian. This corresponds to the analytical solution of the continuous graph
    heat equation x(t) = exp(-t * L) x0, which is widely used in spectral graph
    methods and deterministic graph signal processing. 100% deterministic.
    """

    name = "exact_diffusion"

    def _normalized_laplacian(self, A: np.ndarray) -> np.ndarray:
        """
        Compute a symmetric normalized Laplacian:
            L_norm = I - D^{-1/2} A D^{-1/2}
        where A is the adjacency matrix and D the degree matrix.
        This ensures eigenvalues are in a bounded range [0, 2] and
        gives a well-behaved diffusion operator.
        """
        A = np.asarray(A, dtype=float)
        deg = np.sum(A, axis=1)
        inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = np.diag(inv_sqrt_deg)
        I = np.eye(A.shape[0], dtype=float)
        # Symmetric normalized Laplacian
        return I - D_inv_sqrt @ A @ D_inv_sqrt

    def _matrix_exponential(self, M: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential exp(M) by eigendecomposition:
            M = Q Λ Q^T => exp(M) = Q exp(Λ) Q^T
        This is deterministic, exact up to floating-point precision,
        and avoids iterative integration error.
        """
        # Eigendecompose M (symmetric => real eigenvalues)
        eigvals, eigvecs = np.linalg.eigh(M)
        exp_eig = np.exp(eigvals)  # element-wise exp
        return (eigvecs * exp_eig) @ eigvecs.T

    def run(
        self,
        A: np.ndarray,
        x0: np.ndarray,
        steps: int = 1,
        dt: float = 1.0
    ) -> DynamicsResult:
        """
        A: adjacency matrix (n×n)
        x0: initial state (length n or shape (n,))
        steps: number of diffusion time increments
        dt: diffusion time step size
        """
        # Convert to arrays
        A = np.asarray(A, dtype=float)
        x = np.asarray(x0, dtype=float).flatten()

        # Get normalized Laplacian
        L = graph_laplacian(A)
        L_norm = self._normalized_laplacian(A)

        # Combine time factor
        t = float(dt) * int(steps)

        # Compute exact diffusion kernel: exp(-t * L_norm)
        diffusion_operator = self._matrix_exponential(-t * L_norm)

        # Apply the diffusion operator
        x_final = diffusion_operator @ x

        meta = {
            "steps": int(steps),
            "dt": float(dt),
            "model": self.name,
            "time": t,
        }
        return DynamicsResult(state=x_final, meta=meta)
