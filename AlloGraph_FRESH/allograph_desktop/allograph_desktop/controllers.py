from __future__ import annotations
import os, json
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

from allograph.core.graphio import (
    synthetic_rin, pdb_to_rin, available_chains,
    bundle_to_npz, bundle_from_npz, adjacency_from_csv, bundle_from_adjacency
)
from allograph.core.inference.forward import run_forward
from allograph.core.inference.scan import run_scan
from allograph.core.inference.inverse import run_inverse
from allograph.core.inference.mediation import run_mediation
from allograph_desktop.app_state import AppState

# ---------- Graph creation / import ----------
def ctrl_make_demo_rin(state: AppState) -> None:
    state.bundle = synthetic_rin(n=int(state.demo_n), seed=int(state.demo_seed))

def ctrl_set_pdb(state: AppState, pdb_path: str) -> None:
    state.pdb_path = pdb_path

def ctrl_get_chains(pdb_path: str) -> List[str]:
    return available_chains(pdb_path)

def ctrl_build_rin_from_pdb(state: AppState) -> None:
    if not state.pdb_path:
        raise ValueError("No PDB path set.")
    contact_mode = getattr(state, "contact_mode", "ca")
    state.bundle = pdb_to_rin(
        pdb_path=state.pdb_path,
        cutoff=float(state.cutoff),
        chain=state.chain,
        weight_mode=state.weight_mode,
        contact_mode=contact_mode,
    )

def ctrl_import_csv_adjacency(state: AppState, csv_path: str) -> None:
    A = adjacency_from_csv(csv_path)
    meta = {"source": "csv_adjacency", "csv_path": csv_path}
    state.bundle = bundle_from_adjacency(A, name=os.path.basename(csv_path), meta=meta)

def ctrl_import_npz_bundle(state: AppState, npz_path: str) -> None:
    state.bundle = bundle_from_npz(npz_path)

# ---------- Runs ----------
def ctrl_run_forward(state: AppState) -> Tuple[np.ndarray, dict]:
    if state.bundle is None:
        ctrl_make_demo_rin(state)
    out = run_forward(
        state.bundle,
        model=state.model,
        seed_nodes=[int(state.seed_node)],
        steps=int(state.steps),
        dt=float(state.dt),
    )
    state.last_forward = out.state
    state.meta["forward"] = out.meta
    return out.state, out.meta

def ctrl_run_scan(state: AppState) -> Tuple[np.ndarray, dict]:
    if state.bundle is None:
        ctrl_make_demo_rin(state)
    out = run_scan(state.bundle, model=state.model, steps=int(state.steps), dt=float(state.dt))
    state.last_scan = out.scores
    state.meta["scan"] = out.meta
    return out.scores, out.meta

def ctrl_run_inverse(state: AppState) -> Tuple[np.ndarray, dict]:
    if state.bundle is None:
        ctrl_make_demo_rin(state)
    if state.last_forward is None:
        ctrl_run_forward(state)
    inv = run_inverse(state.bundle, observed=state.last_forward)
    state.last_inverse = inv.source_scores
    state.meta["inverse"] = inv.meta
    return inv.source_scores, inv.meta

def ctrl_run_mediators(state: AppState) -> Tuple[np.ndarray, dict]:
    if state.bundle is None:
        ctrl_make_demo_rin(state)
    med = run_mediation(state.bundle)
    state.last_mediators = med.centrality
    state.meta["mediation"] = med.meta
    return med.centrality, med.meta

def ctrl_sweep_seednode_sumabs(state: AppState) -> Tuple[np.ndarray, dict]:
    if state.bundle is None:
        ctrl_make_demo_rin(state)
    N = state.bundle.n
    scores = np.zeros(N, dtype=float)
    for s in range(N):
        out = run_forward(state.bundle, model=state.model, seed_nodes=[int(s)], steps=int(state.steps), dt=float(state.dt))
        scores[s] = float(np.sum(np.abs(out.state)))
    meta = {"kind": "sweep_seednode_sumabs", "N": int(N), "steps": int(state.steps), "dt": float(state.dt), "model": state.model}
    state.last_sweep = scores
    state.meta["sweep"] = meta
    return scores, meta

# ---------- Export ----------
def ctrl_export_bundle_npz(state: AppState, out_path: str) -> None:
    if state.bundle is None:
        raise ValueError("No graph loaded.")
    bundle_to_npz(state.bundle, out_path)

# ---------- Project save/load ----------
def project_dump(state: AppState) -> Dict[str, Any]:
    def arr(a):
        if a is None: return None
        return a.tolist()
    return {
        "pdb_path": state.pdb_path,
        "chain": state.chain,
        "cutoff": state.cutoff,
        "weight_mode": state.weight_mode,
        "contact_mode": getattr(state, "contact_mode", "ca"),
        "model": state.model,
        "steps": state.steps,
        "dt": state.dt,
        "seed_node": state.seed_node,
        "demo_n": state.demo_n,
        "demo_seed": state.demo_seed,
        "bundle_meta": (state.bundle.meta if state.bundle is not None else None),
        "bundle_name": (state.bundle.name if state.bundle is not None else None),
        "bundle_A": (state.bundle.A.tolist() if state.bundle is not None else None),
        "last_forward": arr(state.last_forward),
        "last_scan": arr(state.last_scan),
        "last_inverse": arr(state.last_inverse),
        "last_mediators": arr(state.last_mediators),
        "last_sweep": arr(state.last_sweep),
        "meta": state.meta,
    }

def project_load(state: AppState, payload: Dict[str, Any]) -> None:
    state.pdb_path = payload.get("pdb_path", None)
    state.chain = payload.get("chain", None)
    state.cutoff = float(payload.get("cutoff", 8.0))
    state.weight_mode = payload.get("weight_mode", "binary")
    state.model = payload.get("model", "diffusion")
    state.steps = int(payload.get("steps", 60))
    state.dt = float(payload.get("dt", 0.10))
    state.seed_node = int(payload.get("seed_node", 5))
    state.demo_n = int(payload.get("demo_n", 60))
    state.demo_seed = int(payload.get("demo_seed", 0))
    state.contact_mode = payload.get("contact_mode", "ca")

    A = payload.get("bundle_A", None)
    if A is not None:
        A = np.asarray(A, dtype=float)
        meta = payload.get("bundle_meta", {}) or {}
        name = payload.get("bundle_name", "restored_bundle")
        from allograph.core.graphio.types import GraphBundle
        state.bundle = GraphBundle(A=A, meta=meta, name=name)
    else:
        state.bundle = None

    def arr(x):
        if x is None: return None
        return np.asarray(x, dtype=float)

    state.last_forward = arr(payload.get("last_forward", None))
    state.last_scan = arr(payload.get("last_scan", None))
    state.last_inverse = arr(payload.get("last_inverse", None))
    state.last_mediators = arr(payload.get("last_mediators", None))
    state.last_sweep = arr(payload.get("last_sweep", None))
    state.meta = payload.get("meta", {}) or {}
