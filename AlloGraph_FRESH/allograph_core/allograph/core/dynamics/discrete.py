from __future__ import annotations
import numpy as np
from .base import BaseDynamicsModel, DynamicsResult

class DiscreteHopModel(BaseDynamicsModel):
    name = "discrete"

    def run(self, A: np.ndarray, x0: np.ndarray, steps: int = 25, dt: float = 1.0) -> DynamicsResult:
        """
        Simple discrete propagation: x <- normalize(Ax) each step (deterministic)
        """
        A = np.asarray(A, dtype=float)
        x = np.asarray(x0, dtype=float).copy()
        for _ in range(int(steps)):
            x = A @ x
            s = float(np.sum(np.abs(x))) + 1e-12
            x = x / s
        meta = {"steps": int(steps), "dt": float(dt), "model": self.name}
        return DynamicsResult(state=x, meta=meta)
