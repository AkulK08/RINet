from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from .types import GraphBundle

def _res_id(chain_id: str, resseq: int, icode: str) -> str:
    icode = (icode or "").strip()
    return f"{chain_id}:{resseq}{icode}"

def pdb_to_rin(
    pdb_path: str,
    cutoff: float = 8.0,
    chain: Optional[str] = None,
    weight_mode: str = "binary",
) -> GraphBundle:
    """
    Build a residue interaction network (RIN) from a PDB file.
    Nodes: residues with CA atoms.
    Edges: CA-CA distance <= cutoff.
    weight_mode:
      - "binary": A_ij = 1 if contact else 0
      - "inv_distance": A_ij = 1 / (d + eps) for contacts
    """
    try:
        from Bio.PDB import PDBParser
    except Exception as e:
        raise ImportError("Biopython is required for pdb_to_rin. Install with: pip install biopython") from e

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    coords: List[np.ndarray] = []
    resid_labels: List[str] = []

    # Use first model
    model = next(structure.get_models())

    for ch in model:
        cid = ch.id
        if chain is not None and cid != chain:
            continue
        for res in ch:
            # Skip hetero/water residues: Bio.PDB uses id tuple (" ", resseq, icode) for standard residues
            if res.id[0] != " ":
                continue
            if "CA" not in res:
                continue
            ca = res["CA"].get_coord()
            coords.append(np.asarray(ca, dtype=float))
            resid_labels.append(_res_id(cid, int(res.id[1]), str(res.id[2])))

    if len(coords) == 0:
        raise ValueError("No residues with CA found (check chain filter or PDB).")

    X = np.vstack(coords)  # (N,3)
    N = X.shape[0]

    # Pairwise distances (O(N^2) but fine for MVP)
    A = np.zeros((N, N), dtype=float)
    eps = 1e-9
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(X[i] - X[j]))
            if d <= float(cutoff):
                if weight_mode == "inv_distance":
                    w = 1.0 / (d + eps)
                else:
                    w = 1.0
                A[i, j] = w
                A[j, i] = w

    meta: Dict[str, Any] = {
        "source": "pdb_to_rin",
        "pdb_path": str(pdb_path),
        "cutoff": float(cutoff),
        "chain": chain,
        "weight_mode": str(weight_mode),
        "residue_ids": resid_labels,
    }
    return GraphBundle(A=A, meta=meta, name=f"rin({chain or all};cutoff={cutoff})")

def available_chains(pdb_path: str) -> List[str]:
    try:
        from Bio.PDB import PDBParser
    except Exception as e:
        raise ImportError("Biopython is required for available_chains. Install with: pip install biopython") from e
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = next(structure.get_models())
    return [ch.id for ch in model]
