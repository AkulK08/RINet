from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict
from ..graphio.types import GraphBundle

@dataclass
class InverseResult:
    source_scores: np.ndarray
    meta: Dict[str, Any]

def run_inverse(bundle: GraphBundle, observed: np.ndarray) -> InverseResult:
    """
    Stub-level callable inverse:
    returns correlation-like score between each node basis vector and observed.
    (This is NOT the final science, but it is deterministic and gives a result.)
    """
    bundle.validate()
    y = np.asarray(observed, dtype=float)
    y = y / (np.linalg.norm(y) + 1e-12)
    n = bundle.n
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        e = np.zeros(n, dtype=float); e[i] = 1.0
        scores[i] = float(np.dot(e, y))
    meta = {**bundle.meta, "kind": "inverse_stub"}
    return InverseResult(source_scores=scores, meta=meta)
