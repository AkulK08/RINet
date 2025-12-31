from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
@dataclass
class DynamicsResult:
    state: np.ndarray
    meta: Dict[str, Any]
class BaseDynamicsModel:
    name: str = "base"
def run(self, A: np.ndarray, x0: np.ndarray, steps: int, dt: float) -> DynamicsResult:
    raise NotImplementedError
