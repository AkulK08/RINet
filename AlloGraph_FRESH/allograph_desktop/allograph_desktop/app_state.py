from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import numpy as np
from allograph.core.graphio.types import GraphBundle

@dataclass
class AppState:
    # data
    bundle: Optional[GraphBundle] = None
    pdb_path: Optional[str] = None
    chain: Optional[str] = None
    cutoff: float = 8.0
    weight_mode: str = "binary"

    # last results
    last_forward: Optional[np.ndarray] = None
    last_scan: Optional[np.ndarray] = None
    last_inverse: Optional[np.ndarray] = None
    last_mediators: Optional[np.ndarray] = None
    last_sweep: Optional[np.ndarray] = None

    # meta bucket
    meta: Dict[str, Any] = field(default_factory=dict)

    # run params
    model: str = "diffusion"
    steps: int = 60
    dt: float = 0.10
    seed_node: int = 5
    demo_n: int = 60
    demo_seed: int = 0

    # UI
    recent_projects: List[str] = field(default_factory=list)
