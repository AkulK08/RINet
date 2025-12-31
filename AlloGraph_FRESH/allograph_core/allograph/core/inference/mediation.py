from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..graphio.types import GraphBundle


@dataclass
class MediationResult:
    centrality: np.ndarray
    meta: Dict[str, Any]


def run_mediation(bundle: GraphBundle) -> MediationResult:
    """
    Simple degree centrality as a mediator proxy (deterministic MVP).
    """
    bundle.validate()

    deg = np.sum(bundle.A > 0, axis=1).astype(float)

    meta = {
        **bundle.meta,
        "kind": "degree_centrality",
    }

    return MediationResult(centrality=deg, meta=meta)
