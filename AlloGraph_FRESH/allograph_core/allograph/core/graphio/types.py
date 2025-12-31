from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, Optional

@dataclass
class GraphBundle:
    A: np.ndarray              # adjacency matrix (N x N), float
    meta: Dict[str, Any]       # optional metadata (residue ids, chain, etc.)
    name: str = "GraphBundle"

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    def validate(self) -> None:
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be a square 2D matrix")
        if not np.isfinite(self.A).all():
            raise ValueError("A contains non-finite values")
