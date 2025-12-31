from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..graphio.types import GraphBundle
from .forward import run_forward


@dataclass
class ScanResult:
    scores: np.ndarray
    meta: Dict[str, Any]


def run_scan(
    bundle: GraphBundle,
    model: str = "diffusion",
    steps: int = 60,
    dt: float = 0.10,
) -> ScanResult:
    """
    Sensitivity scan: for each residue i, seed at i and record total signal sum.

    Deterministic O(N) forward runs.
    For MVP correctness, not speed.
    """
    bundle.validate()

    n = bundle.n
    scores = np.zeros(n, dtype=float)

    for i in range(n):
        fr = run_forward(
            bundle,
            model=model,
            seed_nodes=[i],
            steps=steps,
            dt=dt,
        )
        scores[i] = float(np.sum(np.abs(fr.state)))

    meta = {
        **bundle.meta,
        "model": model,
        "steps": int(steps),
        "dt": float(dt),
        "kind": "scan_sumabs",
    }

    return ScanResult(scores=scores, meta=meta)
