from __future__ import annotations
import json
import numpy as np
from typing import Any, Dict, Optional, Tuple
from .types import GraphBundle

def bundle_to_npz(bundle: GraphBundle, path: str) -> None:
    meta_json = json.dumps(bundle.meta)
    np.savez_compressed(path, A=bundle.A.astype(float), meta=meta_json, name=bundle.name)

def bundle_from_npz(path: str) -> GraphBundle:
    z = np.load(path, allow_pickle=False)
    A = z["A"].astype(float)
    meta_json = str(z["meta"])
    name = str(z.get("name", "bundle"))
    meta = json.loads(meta_json) if meta_json else {}
    return GraphBundle(A=A, meta=meta, name=name)

def adjacency_from_csv(path: str) -> np.ndarray:
    A = np.loadtxt(path, delimiter=",")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("CSV adjacency must be an NxN matrix.")
    return A.astype(float)

def bundle_from_adjacency(A: np.ndarray, name: str="imported_csv", meta: Optional[Dict[str, Any]] = None) -> GraphBundle:
    meta = dict(meta or {})
    meta["source"] = meta.get("source", "csv_adjacency")
    return GraphBundle(A=A.astype(float), meta=meta, name=name)
