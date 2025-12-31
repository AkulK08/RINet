from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from ..graphio.types import GraphBundle
from ..dynamics import make_model

@dataclass
class ForwardResult:
    state: np.ndarray
    meta: Dict[str, Any]

def run_forward(
    bundle: GraphBundle,
    model: str = "diffusion",
    seed_nodes: Optional[List[int]] = None,
    steps: int = 60,
    dt: float = 0.10,
    seed_strength: float = 1.0,
) -> ForwardResult:
    bundle.validate()
    n = bundle.n
    seed_nodes = seed_nodes or [0]
    x0 = np.zeros(n, dtype=float)
    for s in seed_nodes:
        if 0 <= int(s) < n:
            x0[int(s)] = float(seed_strength)

    dyn = make_model(model)
    out = dyn.run(bundle.A, x0=x0, steps=steps, dt=dt)
    meta = {
        **bundle.meta,
        **out.meta,
        "seed_nodes": list(map(int, seed_nodes)),
        "seed_strength": float(seed_strength),
        "bundle_name": bundle.name,
    }
    return ForwardResult(state=out.state, meta=meta)
