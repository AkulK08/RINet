from __future__ import annotations
import numpy as np
from .base import BaseDynamicsModel, DynamicsResult
from ..math.laplacian import graph_laplacian

class DiffusionModel(BaseDynamicsModel):
    name = "diffusion"

    def run(self, A: np.ndarray, x0: np.ndarray, steps: int = 50, dt: float = 0.10) -> DynamicsResult:
        A = np.asarray(A, dtype=float)
        x = np.asarray(x0, dtype=float).copy()
        L = graph_laplacian(A)

        # deterministic forward Euler diffusion
        for _ in range(int(steps)):
            x = x + dt * (-(L @ x))

        meta = {"steps": int(steps), "dt": float(dt), "model": self.name}
        return DynamicsResult(state=x, meta=meta)
