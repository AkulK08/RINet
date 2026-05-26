# spectral_geometry_engine.py
"""
Spectral geometry tools for residue-interaction networks.

This module is intentionally self-contained and NumPy-first so it can live inside
AlloGraph/RINet without forcing heavy scientific dependencies.  If SciPy is
available, a few sparse/eigensolver paths are used opportunistically; otherwise
all algorithms fall back to dense NumPy implementations.

The main entry point is :class:`SpectralGeometryEngine`.  It accepts either the
project's ``GraphBundle`` type or a raw square adjacency matrix and exposes a
research-style API for spectral graph geometry on protein residue networks:

* unnormalized, normalized, and random-walk Laplacians
* eigendecompositions with stable sorting and normalization
* Fiedler vector and algebraic connectivity
* Cheeger sweep cuts
* effective resistance and commute-time distances
* resistance centrality
* spectral clustering and spectral embeddings
* graph heat kernels, heat-kernel signatures, and diffusion distance
* spectral entropy and eigenvector localization statistics
* structured report objects with JSON/CSV export helpers

The implementation tries to be honest about numerical assumptions.  RINs are
normally undirected weighted graphs, but the code can symmetrize noisy matrices,
clip negative contacts when requested, and regularize near-isolated residues.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # Optional.  The repository only requires NumPy/Matplotlib.
    import scipy.linalg as _scipy_linalg  # type: ignore
    import scipy.sparse as _scipy_sparse  # type: ignore
    import scipy.sparse.linalg as _scipy_sparse_linalg  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - depends on environment
    _scipy_linalg = None
    _scipy_sparse = None
    _scipy_sparse_linalg = None
    _HAVE_SCIPY = False

try:
    from .graphio.types import GraphBundle
except Exception:  # pragma: no cover - allows standalone static analysis
    GraphBundle = Any  # type: ignore

try:
    from .math.laplacian import graph_laplacian
except Exception:  # pragma: no cover
    def graph_laplacian(A: np.ndarray) -> np.ndarray:
        degree = np.sum(A, axis=1)
        return np.diag(degree) - A


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]
IndexSequence = Union[Sequence[int], np.ndarray]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SpectralGeometryError(RuntimeError):
    """Base class for spectral geometry errors."""


class InvalidGraphError(SpectralGeometryError):
    """Raised when an adjacency matrix cannot be interpreted as a graph."""


class DecompositionError(SpectralGeometryError):
    """Raised when an eigendecomposition cannot be computed or reused."""


class ClusteringError(SpectralGeometryError):
    """Raised when spectral clustering receives invalid parameters."""


class ExportError(SpectralGeometryError):
    """Raised when a report or matrix cannot be exported."""


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------


def _as_float_array(A: ArrayLike) -> np.ndarray:
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise InvalidGraphError(f"expected a square 2D adjacency matrix, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise InvalidGraphError("adjacency contains NaN or infinite values")
    return arr.copy()


def _safe_name(obj: Any, fallback: str = "graph") -> str:
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return fallback


def adjacency_from_graph(graph: Union[GraphBundle, ArrayLike]) -> np.ndarray:
    """Return a defensive float copy of an adjacency matrix or GraphBundle."""
    if hasattr(graph, "A"):
        return _as_float_array(getattr(graph, "A"))
    return _as_float_array(graph)  # type: ignore[arg-type]


def metadata_from_graph(graph: Union[GraphBundle, ArrayLike]) -> Dict[str, Any]:
    meta = getattr(graph, "meta", {}) if hasattr(graph, "meta") else {}
    return dict(meta or {})


def residue_labels_from_graph(graph: Union[GraphBundle, ArrayLike], n: Optional[int] = None) -> List[str]:
    """Best-effort residue labels from GraphBundle metadata.

    Existing RINet graph bundles may or may not include residue identifiers.  We
    accept several likely keys so the engine remains compatible with future PDB
    readers and with hand-built synthetic graphs.
    """
    if n is None:
        n = adjacency_from_graph(graph).shape[0]
    meta = metadata_from_graph(graph)
    candidates = [
        "residue_ids",
        "residue_labels",
        "node_labels",
        "labels",
        "residues",
    ]
    for key in candidates:
        values = meta.get(key)
        if isinstance(values, (list, tuple)) and len(values) == n:
            return [str(v) for v in values]
    return [str(i) for i in range(n)]


def symmetrize_matrix(A: np.ndarray, mode: str = "average") -> np.ndarray:
    """Symmetrize an adjacency matrix using a named rule.

    Parameters
    ----------
    A:
        Square weighted adjacency matrix.
    mode:
        ``"average"`` uses ``(A + A.T) / 2``; ``"max"`` keeps the stronger
        contact; ``"min"`` keeps only contacts present in both directions;
        ``"none"`` returns a copy unchanged.
    """
    mode = mode.lower()
    if mode == "none":
        return A.copy()
    if mode == "average":
        return 0.5 * (A + A.T)
    if mode == "max":
        return np.maximum(A, A.T)
    if mode == "min":
        return np.minimum(A, A.T)
    raise ValueError(f"unknown symmetrization mode: {mode!r}")


def sanitize_adjacency(
    A: ArrayLike,
    *,
    symmetrize: str = "average",
    remove_self_loops: bool = True,
    clip_negative: bool = True,
    negative_tolerance: float = 1e-12,
) -> np.ndarray:
    """Return a spectral-analysis-ready adjacency matrix."""
    B = _as_float_array(A)
    if remove_self_loops:
        np.fill_diagonal(B, 0.0)
    if clip_negative:
        if np.min(B) < -abs(negative_tolerance):
            warnings.warn(
                "adjacency contains negative weights; clipping them to zero for spectral geometry",
                RuntimeWarning,
                stacklevel=2,
            )
        B = np.where(B < 0.0, 0.0, B)
    B = symmetrize_matrix(B, symmetrize)
    if remove_self_loops:
        np.fill_diagonal(B, 0.0)
    return B.astype(float, copy=False)


def degree_vector(A: np.ndarray) -> np.ndarray:
    """Weighted degree of each residue/node."""
    return np.sum(A, axis=1).astype(float)


def volume(A: np.ndarray) -> float:
    """Weighted graph volume, equal to sum of degrees."""
    return float(np.sum(degree_vector(A)))


def edge_count(A: np.ndarray, *, directed: bool = False, tolerance: float = 0.0) -> int:
    mask = A > tolerance
    if directed:
        return int(np.count_nonzero(mask))
    upper = np.triu(mask, 1)
    return int(np.count_nonzero(upper))


def density(A: np.ndarray, *, tolerance: float = 0.0) -> float:
    n = A.shape[0]
    if n <= 1:
        return 0.0
    return float(edge_count(A, tolerance=tolerance) / (n * (n - 1) / 2.0))


def connected_components(A: np.ndarray, *, tolerance: float = 0.0) -> List[List[int]]:
    """Return connected components of the symmetrized support graph."""
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []
    support = A > tolerance
    support = np.logical_or(support, support.T)
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            nbrs = np.flatnonzero(support[u])
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def largest_component_indices(A: np.ndarray, *, tolerance: float = 0.0) -> np.ndarray:
    comps = connected_components(A, tolerance=tolerance)
    if not comps:
        return np.array([], dtype=int)
    return np.array(comps[0], dtype=int)


def induced_subgraph(A: np.ndarray, indices: IndexSequence) -> np.ndarray:
    idx = np.asarray(indices, dtype=int)
    return A[np.ix_(idx, idx)]


def normalize_rows(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def safe_inverse(x: np.ndarray, *, eps: float = 1e-12, zero_for_small: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = np.abs(x) > eps
    out[mask] = 1.0 / x[mask]
    if not zero_for_small:
        out[~mask] = 1.0 / eps
    return out


def safe_inverse_sqrt(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = x > eps
    out[mask] = 1.0 / np.sqrt(x[mask])
    return out


def stable_argsort(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.asarray(values), kind="mergesort")


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonify_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_value(v) for v in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _write_json(data: Mapping[str, Any], path: Union[str, os.PathLike[str]], *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_jsonify_value(dict(data)), f, indent=indent, sort_keys=True)
        f.write("\n")


def _write_csv_rows(rows: Sequence[Mapping[str, Any]], path: Union[str, os.PathLike[str]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonify_value(row.get(k, "")) for k in keys})


def _top_indices(values: np.ndarray, k: int, *, reverse: bool = True) -> List[int]:
    values = np.asarray(values, dtype=float)
    if k <= 0:
        return []
    k = min(k, values.size)
    order = np.argsort(values, kind="mergesort")
    if reverse:
        order = order[::-1]
    return [int(i) for i in order[:k]]


def _matrix_stats(M: np.ndarray) -> Dict[str, float]:
    if M.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "frobenius": 0.0}
    return {
        "min": float(np.min(M)),
        "max": float(np.max(M)),
        "mean": float(np.mean(M)),
        "std": float(np.std(M)),
        "frobenius": float(np.linalg.norm(M, ord="fro")),
    }


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass
class SpectralOptions:
    """Configuration controlling graph sanitization and decomposition."""

    laplacian_kind: str = "normalized"
    symmetrize: str = "average"
    remove_self_loops: bool = True
    clip_negative: bool = True
    regularization: float = 1e-12
    prefer_scipy: bool = True
    dense_threshold: int = 450
    eigen_tolerance: float = 1e-10
    random_seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EigenDecomposition:
    """Container for sorted graph eigenpairs."""

    values: np.ndarray
    vectors: np.ndarray
    laplacian_kind: str
    matrix_shape: Tuple[int, int]
    solver: str = "numpy.linalg.eigh"
    elapsed_seconds: float = 0.0
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.vectors = np.asarray(self.vectors, dtype=float)
        if self.vectors.ndim != 2:
            raise DecompositionError("eigenvectors must be a 2D matrix")
        if self.values.ndim != 1:
            raise DecompositionError("eigenvalues must be a 1D array")
        if self.vectors.shape[1] != self.values.size:
            raise DecompositionError("number of eigenvectors does not match number of eigenvalues")

    @property
    def n(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def rank(self) -> int:
        return int(self.values.size)

    def first(self, k: int) -> "EigenDecomposition":
        k = max(0, min(int(k), self.rank))
        return EigenDecomposition(
            values=self.values[:k].copy(),
            vectors=self.vectors[:, :k].copy(),
            laplacian_kind=self.laplacian_kind,
            matrix_shape=self.matrix_shape,
            solver=self.solver,
            elapsed_seconds=self.elapsed_seconds,
            options=dict(self.options),
        )

    def nontrivial(self, k: int, *, zero_tolerance: float = 1e-8) -> "EigenDecomposition":
        mask = self.values > zero_tolerance
        idx = np.flatnonzero(mask)[: max(0, int(k))]
        return EigenDecomposition(
            values=self.values[idx].copy(),
            vectors=self.vectors[:, idx].copy(),
            laplacian_kind=self.laplian_kind if hasattr(self, "laplian_kind") else self.laplacian_kind,
            matrix_shape=self.matrix_shape,
            solver=self.solver,
            elapsed_seconds=self.elapsed_seconds,
            options=dict(self.options),
        )

    def to_dict(self, *, include_vectors: bool = False, max_vectors: Optional[int] = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "laplacian_kind": self.laplacian_kind,
            "matrix_shape": list(self.matrix_shape),
            "solver": self.solver,
            "elapsed_seconds": float(self.elapsed_seconds),
            "eigenvalues": self.values.tolist(),
            "options": dict(self.options),
        }
        if include_vectors:
            V = self.vectors
            if max_vectors is not None:
                V = V[:, : max(0, int(max_vectors))]
            data["eigenvectors"] = V.tolist()
        return data

    def save_npz(self, path: Union[str, os.PathLike[str]]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            values=self.values,
            vectors=self.vectors,
            laplacian_kind=np.array(self.laplacian_kind),
            matrix_shape=np.array(self.matrix_shape, dtype=int),
            solver=np.array(self.solver),
            elapsed_seconds=np.array(self.elapsed_seconds, dtype=float),
            options=np.array(json.dumps(_jsonify_value(self.options))),
        )

    @staticmethod
    def load_npz(path: Union[str, os.PathLike[str]]) -> "EigenDecomposition":
        z = np.load(path, allow_pickle=False)
        options_raw = str(z.get("options", "{}"))
        try:
            options = json.loads(options_raw)
        except Exception:
            options = {}
        shape_arr = np.asarray(z["matrix_shape"], dtype=int)
        return EigenDecomposition(
            values=np.asarray(z["values"], dtype=float),
            vectors=np.asarray(z["vectors"], dtype=float),
            laplacian_kind=str(z["laplacian_kind"]),
            matrix_shape=(int(shape_arr[0]), int(shape_arr[1])),
            solver=str(z.get("solver", "loaded")),
            elapsed_seconds=float(z.get("elapsed_seconds", 0.0)),
            options=options,
        )


@dataclass
class SweepCutResult:
    """Result of a Cheeger sweep over an ordering vector."""

    conductance: float
    cut_weight: float
    set_volume: float
    complement_volume: float
    indices: List[int]
    ordering: List[int]
    threshold_position: int
    vector_name: str = "fiedler"

    @property
    def size(self) -> int:
        return len(self.indices)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpectralEmbeddingResult:
    coordinates: np.ndarray
    eigenvalues: np.ndarray
    node_labels: List[str]
    method: str
    normalized_rows: bool = False

    def to_dict(self, *, max_nodes: Optional[int] = None) -> Dict[str, Any]:
        coords = self.coordinates
        labels = self.node_labels
        if max_nodes is not None:
            coords = coords[:max_nodes]
            labels = labels[:max_nodes]
        return {
            "method": self.method,
            "normalized_rows": self.normalized_rows,
            "eigenvalues": self.eigenvalues.tolist(),
            "node_labels": list(labels),
            "coordinates": coords.tolist(),
        }

    def rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for i, label in enumerate(self.node_labels):
            row: Dict[str, Any] = {"node": i, "label": label}
            for j in range(self.coordinates.shape[1]):
                row[f"x{j + 1}"] = float(self.coordinates[i, j])
            rows.append(row)
        return rows

    def to_csv(self, path: Union[str, os.PathLike[str]]) -> None:
        _write_csv_rows(self.rows(), path)


@dataclass
class SpectralClusterResult:
    labels: np.ndarray
    centroids: np.ndarray
    embedding: SpectralEmbeddingResult
    inertia: float
    iterations: int
    converged: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": self.labels.astype(int).tolist(),
            "centroids": self.centroids.tolist(),
            "embedding": self.embedding.to_dict(),
            "inertia": float(self.inertia),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
        }

    def rows(self) -> List[Dict[str, Any]]:
        rows = self.embedding.rows()
        for i, row in enumerate(rows):
            row["cluster"] = int(self.labels[i])
        return rows

    def to_csv(self, path: Union[str, os.PathLike[str]]) -> None:
        _write_csv_rows(self.rows(), path)


@dataclass
class EffectiveResistanceResult:
    resistance: np.ndarray
    commute_time: np.ndarray
    pseudoinverse_diagonal: np.ndarray
    volume: float
    component_count: int
    labels: List[str]

    def to_dict(self, *, include_matrices: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "volume": float(self.volume),
            "component_count": int(self.component_count),
            "labels": self.labels,
            "resistance_stats": _matrix_stats(self.resistance),
            "commute_time_stats": _matrix_stats(self.commute_time),
            "pseudoinverse_diagonal": self.pseudoinverse_diagonal.tolist(),
        }
        if include_matrices:
            data["resistance"] = self.resistance.tolist()
            data["commute_time"] = self.commute_time.tolist()
        return data

    def resistance_centrality(self, *, mode: str = "harmonic", eps: float = 1e-12) -> np.ndarray:
        R = self.resistance.copy()
        np.fill_diagonal(R, np.inf)
        mode = mode.lower()
        if mode == "harmonic":
            inv = np.where(np.isfinite(R) & (R > eps), 1.0 / R, 0.0)
            return np.sum(inv, axis=1)
        if mode == "mean_inverse":
            inv = np.where(np.isfinite(R) & (R > eps), 1.0 / R, 0.0)
            denom = np.maximum(np.count_nonzero(inv, axis=1), 1)
            return np.sum(inv, axis=1) / denom
        if mode == "closeness":
            finite = np.where(np.isfinite(R), R, 0.0)
            counts = np.maximum(np.count_nonzero(np.isfinite(R), axis=1) - 1, 1)
            sums = np.sum(finite, axis=1)
            return np.where(sums > eps, counts / sums, 0.0)
        raise ValueError(f"unknown resistance centrality mode {mode!r}")

    def top_resistance_central_nodes(self, k: int = 10, *, mode: str = "harmonic") -> List[Dict[str, Any]]:
        scores = self.resistance_centrality(mode=mode)
        rows = []
        for i in _top_indices(scores, k):
            rows.append({"node": i, "label": self.labels[i], "score": float(scores[i])})
        return rows


@dataclass
class HeatKernelResult:
    t: float
    kernel: np.ndarray
    hks: np.ndarray
    trace: float
    entropy: float

    def to_dict(self, *, include_kernel: bool = False) -> Dict[str, Any]:
        data = {
            "t": float(self.t),
            "hks": self.hks.tolist(),
            "trace": float(self.trace),
            "entropy": float(self.entropy),
            "kernel_stats": _matrix_stats(self.kernel),
        }
        if include_kernel:
            data["kernel"] = self.kernel.tolist()
        return data


@dataclass
class DiffusionDistanceResult:
    t: float
    distances: np.ndarray
    kernel_diagonal: np.ndarray

    def to_dict(self, *, include_matrix: bool = False) -> Dict[str, Any]:
        data = {
            "t": float(self.t),
            "distance_stats": _matrix_stats(self.distances),
            "kernel_diagonal": self.kernel_diagonal.tolist(),
        }
        if include_matrix:
            data["distances"] = self.distances.tolist()
        return data


@dataclass
class EigenvectorLocalizationResult:
    mode_index: int
    eigenvalue: float
    inverse_participation_ratio: float
    participation_ratio: float
    entropy: float
    max_abs_entry: float
    top_nodes: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpectralReport:
    """Structured report summarizing the spectral geometry of a RIN."""

    graph_name: str
    n_nodes: int
    n_edges: int
    density: float
    volume: float
    component_count: int
    component_sizes: List[int]
    laplacian_kind: str
    eigenvalues: List[float]
    algebraic_connectivity: float
    spectral_gap: float
    spectral_radius: float
    cheeger_cut: Optional[Dict[str, Any]]
    entropy: Dict[str, float]
    localization: List[Dict[str, Any]]
    heat_kernel_summaries: List[Dict[str, Any]]
    top_resistance_central_nodes: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Union[str, os.PathLike[str]]) -> None:
        _write_json(self.to_dict(), path)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# Spectral Geometry Report: {self.graph_name}")
        lines.append("")
        lines.append("## Graph summary")
        lines.append("")
        lines.append(f"- Nodes: {self.n_nodes}")
        lines.append(f"- Edges: {self.n_edges}")
        lines.append(f"- Density: {self.density:.6g}")
        lines.append(f"- Volume: {self.volume:.6g}")
        lines.append(f"- Connected components: {self.component_count} ({self.component_sizes})")
        lines.append("")
        lines.append("## Core spectral statistics")
        lines.append("")
        lines.append(f"- Laplacian: `{self.laplacian_kind}`")
        lines.append(f"- Algebraic connectivity: {self.algebraic_connectivity:.8g}")
        lines.append(f"- Spectral gap: {self.spectral_gap:.8g}")
        lines.append(f"- Spectral radius: {self.spectral_radius:.8g}")
        lines.append("")
        lines.append("First eigenvalues:")
        lines.append("")
        for i, lam in enumerate(self.eigenvalues[:20]):
            lines.append(f"{i}. `{lam:.10g}`")
        if self.cheeger_cut:
            lines.append("")
            lines.append("## Cheeger sweep cut")
            lines.append("")
            lines.append(f"- Conductance: {self.cheeger_cut.get('conductance', 0.0):.8g}")
            lines.append(f"- Size: {self.cheeger_cut.get('size', len(self.cheeger_cut.get('indices', [])))}")
            lines.append(f"- Cut weight: {self.cheeger_cut.get('cut_weight', 0.0):.8g}")
        lines.append("")
        lines.append("## Spectral entropy")
        lines.append("")
        for k, v in self.entropy.items():
            lines.append(f"- {k}: {v:.8g}")
        if self.top_resistance_central_nodes:
            lines.append("")
            lines.append("## Top resistance-central residues")
            lines.append("")
            lines.append("| rank | node | label | score |")
            lines.append("|---:|---:|:---|---:|")
            for r, row in enumerate(self.top_resistance_central_nodes, start=1):
                lines.append(f"| {r} | {row['node']} | {row['label']} | {row['score']:.8g} |")
        return "\n".join(lines) + "\n"

    def save_markdown(self, path: Union[str, os.PathLike[str]]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Laplacians and linear algebra
# ---------------------------------------------------------------------------


def unnormalized_laplacian(A: np.ndarray) -> np.ndarray:
    return graph_laplacian(A)


def normalized_laplacian(A: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    d = degree_vector(A)
    invsqrt = safe_inverse_sqrt(d, eps=eps)
    S = invsqrt[:, None] * A * invsqrt[None, :]
    L = np.eye(A.shape[0], dtype=float) - S
    isolated = d <= eps
    if np.any(isolated):
        # Standard convention: isolated nodes contribute zero rows/columns.
        L[isolated, :] = 0.0
        L[:, isolated] = 0.0
    return 0.5 * (L + L.T)


def random_walk_laplacian(A: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    d = degree_vector(A)
    inv = safe_inverse(d, eps=eps)
    return np.eye(A.shape[0], dtype=float) - inv[:, None] * A


def transition_matrix(A: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    d = degree_vector(A)
    P = safe_inverse(d, eps=eps)[:, None] * A
    return P


def laplacian_matrix(A: np.ndarray, kind: str = "normalized", *, eps: float = 1e-12) -> np.ndarray:
    kind = kind.lower().replace("_", "-")
    if kind in {"unnormalized", "combinatorial", "standard"}:
        return unnormalized_laplacian(A)
    if kind in {"normalized", "symmetric-normalized", "sym-normalized", "sym"}:
        return normalized_laplacian(A, eps=eps)
    if kind in {"random-walk", "randomwalk", "rw"}:
        return random_walk_laplacian(A, eps=eps)
    raise ValueError(f"unknown Laplacian kind {kind!r}")


def is_symmetric(M: np.ndarray, *, atol: float = 1e-10) -> bool:
    return bool(np.allclose(M, M.T, atol=atol, rtol=0.0))


def eigh_sorted(M: np.ndarray, *, prefer_scipy: bool = True) -> Tuple[np.ndarray, np.ndarray, str]:
    """Dense symmetric eigendecomposition sorted from low to high."""
    if prefer_scipy and _HAVE_SCIPY and _scipy_linalg is not None:
        vals, vecs = _scipy_linalg.eigh(M, check_finite=False)
        solver = "scipy.linalg.eigh"
    else:
        vals, vecs = np.linalg.eigh(M)
        solver = "numpy.linalg.eigh"
    order = stable_argsort(vals)
    vals = np.real(vals[order]).astype(float)
    vecs = np.real(vecs[:, order]).astype(float)
    vals[np.abs(vals) < 1e-14] = 0.0
    return vals, vecs, solver


def eig_sorted(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    vals, vecs = np.linalg.eig(M)
    vals = np.real_if_close(vals, tol=1000)
    vecs = np.real_if_close(vecs, tol=1000)
    vals = np.asarray(np.real(vals), dtype=float)
    vecs = np.asarray(np.real(vecs), dtype=float)
    order = stable_argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    norms = np.linalg.norm(vecs, axis=0)
    vecs[:, norms > 0] /= norms[norms > 0]
    vals[np.abs(vals) < 1e-14] = 0.0
    return vals, vecs, "numpy.linalg.eig"


def low_rank_eigh(
    M: np.ndarray,
    k: int,
    *,
    which: str = "SM",
    prefer_scipy: bool = True,
    dense_threshold: int = 450,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Compute k eigenpairs, using SciPy sparse eigsh when useful.

    For small matrices, the dense path is often more stable and simpler.  For
    larger matrices with SciPy installed, eigsh can save time when only a few
    modes are requested.
    """
    n = M.shape[0]
    k = int(k)
    if k <= 0:
        return np.array([], dtype=float), np.zeros((n, 0), dtype=float), "none"
    if k >= n - 1 or n <= dense_threshold or not (prefer_scipy and _HAVE_SCIPY):
        vals, vecs, solver = eigh_sorted(0.5 * (M + M.T), prefer_scipy=prefer_scipy)
        return vals[:k], vecs[:, :k], solver
    try:  # pragma: no cover - SciPy not guaranteed
        assert _scipy_sparse is not None and _scipy_sparse_linalg is not None
        S = _scipy_sparse.csr_matrix(0.5 * (M + M.T))
        vals, vecs = _scipy_sparse_linalg.eigsh(S, k=k, which=which)
        order = stable_argsort(vals)
        vals = np.real(vals[order]).astype(float)
        vecs = np.real(vecs[:, order]).astype(float)
        vals[np.abs(vals) < 1e-14] = 0.0
        return vals, vecs, "scipy.sparse.linalg.eigsh"
    except Exception:
        vals, vecs, solver = eigh_sorted(0.5 * (M + M.T), prefer_scipy=prefer_scipy)
        return vals[:k], vecs[:, :k], solver + " fallback"


def pseudoinverse_from_eigendecomposition(
    values: np.ndarray,
    vectors: np.ndarray,
    *,
    zero_tolerance: float = 1e-10,
) -> np.ndarray:
    inv = np.zeros_like(values, dtype=float)
    mask = values > zero_tolerance
    inv[mask] = 1.0 / values[mask]
    return (vectors * inv[None, :]) @ vectors.T


# ---------------------------------------------------------------------------
# K-means; avoids scikit-learn dependency.
# ---------------------------------------------------------------------------


@dataclass
class KMeansResult:
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    iterations: int
    converged: bool


def _kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=float)
    first = int(rng.integers(0, n))
    centroids[0] = X[first]
    closest_sq = np.sum((X - centroids[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 0:
            centroids[c] = X[int(rng.integers(0, n))]
            continue
        probs = closest_sq / total
        idx = int(rng.choice(n, p=probs))
        centroids[c] = X[idx]
        closest_sq = np.minimum(closest_sq, np.sum((X - centroids[c]) ** 2, axis=1))
    return centroids


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_init: int = 8,
) -> KMeansResult:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ClusteringError("kmeans expects a 2D feature matrix")
    n, d = X.shape
    if k <= 0 or k > n:
        raise ClusteringError(f"k must be between 1 and n={n}, got {k}")
    rng = np.random.default_rng(seed)
    best: Optional[KMeansResult] = None
    for init_idx in range(max(1, int(n_init))):
        centroids = _kmeans_plus_plus_init(X, k, rng)
        labels = np.zeros(n, dtype=int)
        converged = False
        prev_inertia = math.inf
        it = 0
        for it in range(1, max_iter + 1):
            dist_sq = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist_sq, axis=1).astype(int)
            inertia = float(np.sum(dist_sq[np.arange(n), labels]))
            new_centroids = centroids.copy()
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    new_centroids[c] = np.mean(X[mask], axis=0)
                else:
                    # Re-seed empty clusters with the point currently furthest
                    # from its assigned centroid.
                    far = int(np.argmax(dist_sq[np.arange(n), labels]))
                    new_centroids[c] = X[far]
            shift = float(np.linalg.norm(new_centroids - centroids))
            centroids = new_centroids
            if abs(prev_inertia - inertia) <= tol * max(1.0, prev_inertia) or shift <= tol:
                converged = True
                break
            prev_inertia = inertia
        dist_sq = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dist_sq, axis=1).astype(int)
        inertia = float(np.sum(dist_sq[np.arange(n), labels]))
        candidate = KMeansResult(labels=labels, centroids=centroids, inertia=inertia, iterations=it, converged=converged)
        if best is None or candidate.inertia < best.inertia:
            best = candidate
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class SpectralGeometryEngine:
    """Spectral graph geometry engine for residue-interaction networks.

    Parameters
    ----------
    graph:
        Either a ``GraphBundle`` or a raw square adjacency matrix.
    options:
        Optional :class:`SpectralOptions`.  Defaults are conservative for RINs:
        average symmetrization, self-loop removal, negative-weight clipping, and
        normalized Laplacian analysis.
    name:
        Optional display name for reports.
    labels:
        Optional residue/node labels.  If omitted, metadata is inspected.
    """

    def __init__(
        self,
        graph: Union[GraphBundle, ArrayLike],
        options: Optional[SpectralOptions] = None,
        *,
        name: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        self.original_graph = graph
        self.options = options or SpectralOptions()
        raw = adjacency_from_graph(graph)
        self.A = sanitize_adjacency(
            raw,
            symmetrize=self.options.symmetrize,
            remove_self_loops=self.options.remove_self_loops,
            clip_negative=self.options.clip_negative,
        )
        self.n = int(self.A.shape[0])
        self.name = name or _safe_name(graph, "residue_interaction_network")
        self.meta = metadata_from_graph(graph)
        self.labels = list(labels) if labels is not None else residue_labels_from_graph(graph, self.n)
        if len(self.labels) != self.n:
            raise InvalidGraphError(f"labels length {len(self.labels)} does not match graph size {self.n}")
        self._laplacian_cache: Dict[str, np.ndarray] = {}
        self._decomp_cache: Dict[Tuple[str, Optional[int]], EigenDecomposition] = {}
        self._resistance_cache: Optional[EffectiveResistanceResult] = None

    # ------------------------------ construction -------------------------

    @classmethod
    def from_adjacency(
        cls,
        A: ArrayLike,
        *,
        name: str = "adjacency_graph",
        labels: Optional[Sequence[str]] = None,
        options: Optional[SpectralOptions] = None,
    ) -> "SpectralGeometryEngine":
        return cls(A, options=options, name=name, labels=labels)

    @classmethod
    def from_bundle(
        cls,
        bundle: GraphBundle,
        *,
        options: Optional[SpectralOptions] = None,
    ) -> "SpectralGeometryEngine":
        return cls(bundle, options=options)

    def copy_with_adjacency(
        self,
        A: ArrayLike,
        *,
        name_suffix: str = "copy",
        labels: Optional[Sequence[str]] = None,
    ) -> "SpectralGeometryEngine":
        return SpectralGeometryEngine(
            A,
            options=SpectralOptions(**self.options.to_dict()),
            name=f"{self.name}_{name_suffix}",
            labels=labels or self.labels,
        )

    # ------------------------------ graph summaries ----------------------

    @property
    def degrees(self) -> np.ndarray:
        return degree_vector(self.A)

    @property
    def volume(self) -> float:
        return volume(self.A)

    @property
    def edge_count(self) -> int:
        return edge_count(self.A)

    @property
    def density(self) -> float:
        return density(self.A)

    @property
    def components(self) -> List[List[int]]:
        return connected_components(self.A)

    def graph_summary(self) -> Dict[str, Any]:
        comps = self.components
        deg = self.degrees
        return {
            "name": self.name,
            "n_nodes": self.n,
            "n_edges": self.edge_count,
            "density": self.density,
            "volume": self.volume,
            "component_count": len(comps),
            "component_sizes": [len(c) for c in comps],
            "degree_min": float(np.min(deg)) if deg.size else 0.0,
            "degree_max": float(np.max(deg)) if deg.size else 0.0,
            "degree_mean": float(np.mean(deg)) if deg.size else 0.0,
            "degree_std": float(np.std(deg)) if deg.size else 0.0,
            "metadata": dict(self.meta),
        }

    # ------------------------------ matrices -----------------------------

    def laplacian(self, kind: Optional[str] = None) -> np.ndarray:
        kind = (kind or self.options.laplacian_kind).lower()
        if kind not in self._laplacian_cache:
            self._laplacian_cache[kind] = laplacian_matrix(self.A, kind, eps=self.options.regularization)
        return self._laplacian_cache[kind].copy()

    def unnormalized_laplacian(self) -> np.ndarray:
        return self.laplacian("unnormalized")

    def normalized_laplacian(self) -> np.ndarray:
        return self.laplacian("normalized")

    def random_walk_laplacian(self) -> np.ndarray:
        return self.laplacian("random-walk")

    def transition_matrix(self) -> np.ndarray:
        return transition_matrix(self.A, eps=self.options.regularization)

    def modularity_matrix(self) -> np.ndarray:
        d = self.degrees
        vol = float(np.sum(d))
        if vol <= self.options.regularization:
            return self.A.copy()
        return self.A - np.outer(d, d) / vol

    # ------------------------------ eigensystems -------------------------

    def decompose(
        self,
        *,
        kind: Optional[str] = None,
        k: Optional[int] = None,
        use_cache: bool = True,
    ) -> EigenDecomposition:
        kind = (kind or self.options.laplacian_kind).lower()
        cache_key = (kind, int(k) if k is not None else None)
        if use_cache and cache_key in self._decomp_cache:
            return self._decomp_cache[cache_key]
        L = self.laplacian(kind)
        start = time.perf_counter()
        if kind in {"random-walk", "randomwalk", "rw"} and not is_symmetric(L):
            vals, vecs, solver = eig_sorted(L)
            if k is not None:
                vals = vals[: int(k)]
                vecs = vecs[:, : int(k)]
        else:
            if k is not None and int(k) < self.n:
                vals, vecs, solver = low_rank_eigh(
                    L,
                    int(k),
                    prefer_scipy=self.options.prefer_scipy,
                    dense_threshold=self.options.dense_threshold,
                )
            else:
                vals, vecs, solver = eigh_sorted(
                    0.5 * (L + L.T),
                    prefer_scipy=self.options.prefer_scipy,
                )
        elapsed = time.perf_counter() - start
        dec = EigenDecomposition(
            values=vals,
            vectors=vecs,
            laplacian_kind=kind,
            matrix_shape=L.shape,
            solver=solver,
            elapsed_seconds=elapsed,
            options=self.options.to_dict(),
        )
        if use_cache:
            self._decomp_cache[cache_key] = dec
        return dec

    def eigenvalues(self, *, kind: Optional[str] = None, k: Optional[int] = None) -> np.ndarray:
        return self.decompose(kind=kind, k=k).values.copy()

    def eigenvectors(self, *, kind: Optional[str] = None, k: Optional[int] = None) -> np.ndarray:
        return self.decompose(kind=kind, k=k).vectors.copy()

    def fiedler_index(self, *, kind: str = "unnormalized", zero_tolerance: float = 1e-8) -> int:
        vals = self.decompose(kind=kind).values
        positive = np.flatnonzero(vals > zero_tolerance)
        if positive.size == 0:
            return min(1, len(vals) - 1)
        return int(positive[0])

    def fiedler_vector(self, *, kind: str = "unnormalized", zero_tolerance: float = 1e-8) -> np.ndarray:
        dec = self.decompose(kind=kind)
        idx = self.fiedler_index(kind=kind, zero_tolerance=zero_tolerance)
        return dec.vectors[:, idx].copy()

    def algebraic_connectivity(self, *, kind: str = "unnormalized", zero_tolerance: float = 1e-8) -> float:
        dec = self.decompose(kind=kind)
        idx = self.fiedler_index(kind=kind, zero_tolerance=zero_tolerance)
        if dec.values.size == 0:
            return 0.0
        return float(dec.values[idx])

    def spectral_gap(self, *, kind: Optional[str] = None) -> float:
        vals = self.decompose(kind=kind).values
        if vals.size < 2:
            return 0.0
        return float(vals[1] - vals[0])

    def spectral_radius(self, *, kind: Optional[str] = None) -> float:
        vals = self.decompose(kind=kind).values
        if vals.size == 0:
            return 0.0
        return float(np.max(np.abs(vals)))

    def mode_table(self, *, kind: Optional[str] = None, k: int = 20) -> List[Dict[str, Any]]:
        dec = self.decompose(kind=kind)
        rows: List[Dict[str, Any]] = []
        for i, lam in enumerate(dec.values[: max(0, int(k))]):
            vec = dec.vectors[:, i]
            rows.append(
                {
                    "mode": i,
                    "eigenvalue": float(lam),
                    "vector_norm": float(np.linalg.norm(vec)),
                    "mean": float(np.mean(vec)),
                    "std": float(np.std(vec)),
                    "max_abs": float(np.max(np.abs(vec))),
                    "ipr": float(np.sum(vec**4) / max(np.sum(vec**2) ** 2, self.options.regularization)),
                }
            )
        return rows

    # ------------------------------ Cheeger/sweep cuts -------------------

    def conductance_of_set(self, indices: IndexSequence) -> Tuple[float, float, float, float]:
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0 or idx.size >= self.n:
            return math.inf, 0.0, 0.0, self.volume
        mask = np.zeros(self.n, dtype=bool)
        mask[idx] = True
        cut = float(np.sum(self.A[np.ix_(mask, ~mask)]))
        deg = self.degrees
        vol_s = float(np.sum(deg[mask]))
        vol_t = float(np.sum(deg[~mask]))
        denom = min(vol_s, vol_t)
        phi = math.inf if denom <= self.options.regularization else cut / denom
        return float(phi), cut, vol_s, vol_t

    def cheeger_sweep(
        self,
        vector: Optional[np.ndarray] = None,
        *,
        kind: str = "unnormalized",
        vector_name: str = "fiedler",
    ) -> SweepCutResult:
        if vector is None:
            vector = self.fiedler_vector(kind=kind)
        vector = np.asarray(vector, dtype=float).reshape(-1)
        if vector.size != self.n:
            raise ValueError(f"sweep vector length {vector.size} does not match n={self.n}")
        order = np.argsort(vector, kind="mergesort")
        best_phi = math.inf
        best_indices: List[int] = []
        best_cut = 0.0
        best_vol_s = 0.0
        best_vol_t = self.volume
        best_pos = -1
        # Skip empty set and full set.
        for pos in range(1, self.n):
            idx = order[:pos]
            phi, cut, vol_s, vol_t = self.conductance_of_set(idx)
            if phi < best_phi:
                best_phi = phi
                best_indices = [int(i) for i in idx]
                best_cut = cut
                best_vol_s = vol_s
                best_vol_t = vol_t
                best_pos = pos
        return SweepCutResult(
            conductance=float(best_phi),
            cut_weight=float(best_cut),
            set_volume=float(best_vol_s),
            complement_volume=float(best_vol_t),
            indices=best_indices,
            ordering=[int(i) for i in order],
            threshold_position=int(best_pos),
            vector_name=vector_name,
        )

    def bipartition_from_fiedler(self, *, kind: str = "unnormalized", threshold: Optional[float] = None) -> Tuple[List[int], List[int]]:
        v = self.fiedler_vector(kind=kind)
        if threshold is None:
            threshold = float(np.median(v))
        left = np.flatnonzero(v <= threshold).astype(int).tolist()
        right = np.flatnonzero(v > threshold).astype(int).tolist()
        return left, right

    # ------------------------------ resistance geometry ------------------

    def laplacian_pseudoinverse(self, *, zero_tolerance: float = 1e-10) -> np.ndarray:
        dec = self.decompose(kind="unnormalized")
        return pseudoinverse_from_eigendecomposition(dec.values, dec.vectors, zero_tolerance=zero_tolerance)

    def effective_resistance(self, *, use_cache: bool = True) -> EffectiveResistanceResult:
        if use_cache and self._resistance_cache is not None:
            return self._resistance_cache
        Lp = self.laplacian_pseudoinverse()
        diag = np.diag(Lp).copy()
        R = diag[:, None] + diag[None, :] - 2.0 * Lp
        R = np.maximum(0.0, 0.5 * (R + R.T))
        comps = connected_components(self.A)
        if len(comps) > 1:
            # Effective resistance between disconnected components is infinite.
            comp_id = np.empty(self.n, dtype=int)
            for cid, comp in enumerate(comps):
                comp_id[np.asarray(comp, dtype=int)] = cid
            mismatch = comp_id[:, None] != comp_id[None, :]
            R[mismatch] = np.inf
        C = R * self.volume
        result = EffectiveResistanceResult(
            resistance=R,
            commute_time=C,
            pseudoinverse_diagonal=diag,
            volume=self.volume,
            component_count=len(comps),
            labels=list(self.labels),
        )
        if use_cache:
            self._resistance_cache = result
        return result

    def commute_time_distance(self) -> np.ndarray:
        return self.effective_resistance().commute_time.copy()

    def resistance_centrality(self, *, mode: str = "harmonic") -> np.ndarray:
        return self.effective_resistance().resistance_centrality(mode=mode)

    def top_resistance_central_nodes(self, k: int = 10, *, mode: str = "harmonic") -> List[Dict[str, Any]]:
        return self.effective_resistance().top_resistance_central_nodes(k=k, mode=mode)

    # ------------------------------ embeddings and clustering ------------

    def spectral_embedding(
        self,
        n_components: int = 2,
        *,
        kind: str = "normalized",
        skip_trivial: bool = True,
        row_normalize: bool = False,
        scale: Optional[str] = None,
    ) -> SpectralEmbeddingResult:
        n_components = int(n_components)
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        request = min(self.n, n_components + (1 if skip_trivial else 0) + 4)
        dec = self.decompose(kind=kind, k=request)
        start = 1 if skip_trivial and dec.values.size > 1 else 0
        stop = min(dec.values.size, start + n_components)
        coords = dec.vectors[:, start:stop].copy()
        vals = dec.values[start:stop].copy()
        if scale is not None:
            scale_l = scale.lower()
            if scale_l in {"eigenvalue", "lambda"}:
                coords = coords * vals[None, :]
            elif scale_l in {"inverse-sqrt", "resistance"}:
                coords = coords * safe_inverse_sqrt(vals, eps=self.options.regularization)[None, :]
            elif scale_l in {"heat", "diffusion"}:
                coords = coords * np.exp(-vals)[None, :]
            else:
                raise ValueError(f"unknown spectral embedding scale {scale!r}")
        if row_normalize:
            coords = normalize_rows(coords)
        return SpectralEmbeddingResult(
            coordinates=coords,
            eigenvalues=vals,
            node_labels=list(self.labels),
            method=f"{kind} spectral embedding",
            normalized_rows=row_normalize,
        )

    def spectral_clustering(
        self,
        n_clusters: int,
        *,
        n_components: Optional[int] = None,
        kind: str = "normalized",
        row_normalize: bool = True,
        seed: Optional[int] = None,
        max_iter: int = 200,
        n_init: int = 8,
    ) -> SpectralClusterResult:
        n_clusters = int(n_clusters)
        if n_clusters <= 0 or n_clusters > self.n:
            raise ClusteringError(f"n_clusters must be in [1, {self.n}], got {n_clusters}")
        dims = int(n_components or n_clusters)
        embedding = self.spectral_embedding(
            dims,
            kind=kind,
            skip_trivial=True,
            row_normalize=row_normalize,
        )
        km = kmeans(
            embedding.coordinates,
            n_clusters,
            seed=self.options.random_seed if seed is None else int(seed),
            max_iter=max_iter,
            n_init=n_init,
        )
        return SpectralClusterResult(
            labels=km.labels,
            centroids=km.centroids,
            embedding=embedding,
            inertia=km.inertia,
            iterations=km.iterations,
            converged=km.converged,
        )

    # ------------------------------ heat kernels -------------------------

    def heat_kernel(
        self,
        t: float,
        *,
        kind: str = "normalized",
        use_eigendecomposition: bool = True,
    ) -> HeatKernelResult:
        t = float(t)
        if t < 0:
            raise ValueError("heat-kernel time t must be nonnegative")
        if use_eigendecomposition:
            dec = self.decompose(kind=kind)
            weights = np.exp(-t * dec.values)
            K = (dec.vectors * weights[None, :]) @ dec.vectors.T
        else:
            L = self.laplacian(kind)
            if _HAVE_SCIPY and _scipy_linalg is not None:
                K = _scipy_linalg.expm(-t * L)
            else:
                vals, vecs, _ = eigh_sorted(0.5 * (L + L.T), prefer_scipy=False)
                weights = np.exp(-t * vals)
                K = (vecs * weights[None, :]) @ vecs.T
        K = np.real_if_close(K, tol=1000).astype(float)
        K = 0.5 * (K + K.T)
        hks = np.diag(K).copy()
        trace = float(np.trace(K))
        entropy = spectral_entropy(np.linalg.eigvalsh(K), normalize=True)
        return HeatKernelResult(t=t, kernel=K, hks=hks, trace=trace, entropy=entropy)

    def heat_kernel_signature(self, times: Sequence[float], *, kind: str = "normalized") -> np.ndarray:
        cols = []
        for t in times:
            cols.append(self.heat_kernel(float(t), kind=kind).hks)
        if not cols:
            return np.zeros((self.n, 0), dtype=float)
        return np.column_stack(cols)

    def heat_trace(self, times: Sequence[float], *, kind: str = "normalized") -> np.ndarray:
        return np.array([self.heat_kernel(float(t), kind=kind).trace for t in times], dtype=float)

    def diffusion_distance(self, t: float, *, kind: str = "normalized") -> DiffusionDistanceResult:
        hk = self.heat_kernel(t, kind=kind)
        K = hk.kernel
        # Diffusion distance at time t can be written as squared Euclidean
        # distance between heat-kernel rows.  This is dense but clear.
        row_norm = np.sum(K * K, axis=1)
        D2 = row_norm[:, None] + row_norm[None, :] - 2.0 * (K @ K.T)
        D2 = np.maximum(D2, 0.0)
        return DiffusionDistanceResult(t=float(t), distances=np.sqrt(D2), kernel_diagonal=np.diag(K).copy())

    def diffusion_distance_embedding(
        self,
        t: float,
        n_components: int = 3,
        *,
        kind: str = "normalized",
    ) -> SpectralEmbeddingResult:
        dec = self.decompose(kind=kind)
        start = 1 if dec.values.size > 1 else 0
        stop = min(dec.values.size, start + int(n_components))
        vals = dec.values[start:stop]
        coords = dec.vectors[:, start:stop] * np.exp(-float(t) * vals)[None, :]
        return SpectralEmbeddingResult(
            coordinates=coords,
            eigenvalues=vals.copy(),
            node_labels=list(self.labels),
            method=f"diffusion distance embedding t={float(t):.4g}",
            normalized_rows=False,
        )

    # ------------------------------ entropy/localization -----------------

    def spectral_entropy(
        self,
        *,
        kind: Optional[str] = None,
        beta: Optional[float] = None,
        normalized: bool = True,
        skip_zero: bool = True,
    ) -> float:
        vals = self.decompose(kind=kind).values
        if skip_zero:
            vals = vals[vals > self.options.eigen_tolerance]
        if vals.size == 0:
            return 0.0
        if beta is None:
            weights = np.maximum(vals, 0.0)
        else:
            weights = np.exp(-float(beta) * vals)
        return spectral_entropy(weights, normalize=normalized)

    def von_neumann_entropy(self, *, kind: str = "unnormalized") -> float:
        vals = self.decompose(kind=kind).values
        total = float(np.sum(vals))
        if total <= self.options.regularization:
            return 0.0
        probs = vals[vals > self.options.eigen_tolerance] / total
        return float(-np.sum(probs * np.log(probs)))

    def eigenvector_localization(
        self,
        mode_index: int,
        *,
        kind: Optional[str] = None,
        top_k: int = 10,
    ) -> EigenvectorLocalizationResult:
        dec = self.decompose(kind=kind)
        m = int(mode_index)
        if m < 0 or m >= dec.values.size:
            raise IndexError(f"mode_index must be in [0, {dec.values.size - 1}], got {m}")
        v = dec.vectors[:, m].astype(float)
        norm2 = float(np.sum(v * v))
        if norm2 <= self.options.regularization:
            probs = np.zeros_like(v)
        else:
            probs = (v * v) / norm2
        ipr = float(np.sum(probs * probs))
        pr = 0.0 if ipr <= self.options.regularization else float(1.0 / ipr)
        ent = spectral_entropy(probs, normalize=False)
        order = np.argsort(np.abs(v), kind="mergesort")[::-1]
        top: List[Dict[str, Any]] = []
        for i in order[: max(0, int(top_k))]:
            top.append(
                {
                    "node": int(i),
                    "label": self.labels[int(i)],
                    "value": float(v[int(i)]),
                    "abs_value": float(abs(v[int(i)])),
                    "mass": float(probs[int(i)]),
                }
            )
        return EigenvectorLocalizationResult(
            mode_index=m,
            eigenvalue=float(dec.values[m]),
            inverse_participation_ratio=ipr,
            participation_ratio=pr,
            entropy=ent,
            max_abs_entry=float(np.max(np.abs(v))) if v.size else 0.0,
            top_nodes=top,
        )

    def localization_table(self, *, kind: Optional[str] = None, modes: int = 20, top_k: int = 5) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        dec = self.decompose(kind=kind)
        for m in range(min(int(modes), dec.values.size)):
            rows.append(self.eigenvector_localization(m, kind=kind, top_k=top_k).to_dict())
        return rows

    # ------------------------------ residue-level summaries --------------

    def node_spectral_features(
        self,
        *,
        modes: int = 10,
        heat_times: Sequence[float] = (0.1, 1.0, 5.0),
        include_resistance: bool = True,
    ) -> List[Dict[str, Any]]:
        dec = self.decompose(kind="normalized", k=min(self.n, max(2, modes + 1)))
        features: List[Dict[str, Any]] = []
        hks = self.heat_kernel_signature(heat_times, kind="normalized") if heat_times else np.zeros((self.n, 0))
        resistance_scores = self.resistance_centrality() if include_resistance else np.zeros(self.n)
        deg = self.degrees
        for i in range(self.n):
            row: Dict[str, Any] = {
                "node": i,
                "label": self.labels[i],
                "degree": float(deg[i]),
                "resistance_centrality": float(resistance_scores[i]) if include_resistance else 0.0,
            }
            for m in range(min(modes, dec.vectors.shape[1])):
                row[f"eigvec_{m}"] = float(dec.vectors[i, m])
            for j, t in enumerate(heat_times):
                row[f"hks_t{j}_{float(t):.4g}"] = float(hks[i, j])
            features.append(row)
        return features

    def save_node_spectral_features(
        self,
        path: Union[str, os.PathLike[str]],
        *,
        modes: int = 10,
        heat_times: Sequence[float] = (0.1, 1.0, 5.0),
    ) -> None:
        _write_csv_rows(self.node_spectral_features(modes=modes, heat_times=heat_times), path)

    def pairwise_geometry_features(
        self,
        pairs: Optional[Sequence[Tuple[int, int]]] = None,
        *,
        heat_time: float = 1.0,
        include_all_edges_if_none: bool = True,
    ) -> List[Dict[str, Any]]:
        R = self.effective_resistance().resistance
        C = self.effective_resistance().commute_time
        D = self.diffusion_distance(heat_time).distances
        rows: List[Dict[str, Any]] = []
        if pairs is None:
            if include_all_edges_if_none:
                ii, jj = np.where(np.triu(self.A > 0, 1))
                pairs_iter = list(zip(ii.astype(int).tolist(), jj.astype(int).tolist()))
            else:
                pairs_iter = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        else:
            pairs_iter = [(int(i), int(j)) for i, j in pairs]
        for i, j in pairs_iter:
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "label_i": self.labels[i],
                    "label_j": self.labels[j],
                    "weight": float(self.A[i, j]),
                    "effective_resistance": float(R[i, j]),
                    "commute_time": float(C[i, j]),
                    "diffusion_distance": float(D[i, j]),
                }
            )
        return rows

    def save_pairwise_geometry_features(
        self,
        path: Union[str, os.PathLike[str]],
        *,
        heat_time: float = 1.0,
        include_all_edges_if_none: bool = True,
    ) -> None:
        rows = self.pairwise_geometry_features(
            heat_time=heat_time,
            include_all_edges_if_none=include_all_edges_if_none,
        )
        _write_csv_rows(rows, path)

    # ------------------------------ exports/reports ----------------------

    def report(
        self,
        *,
        modes: int = 30,
        heat_times: Sequence[float] = (0.1, 1.0, 5.0),
        include_resistance: bool = True,
    ) -> SpectralReport:
        dec = self.decompose(kind=self.options.laplacian_kind)
        comps = self.components
        try:
            cut = self.cheeger_sweep(kind="unnormalized").to_dict()
            cut["size"] = len(cut.get("indices", []))
        except Exception:
            cut = None
        heat_summaries = [self.heat_kernel(t, kind=self.options.laplacian_kind).to_dict(include_kernel=False) for t in heat_times]
        localization = self.localization_table(kind=self.options.laplacian_kind, modes=min(modes, dec.values.size), top_k=5)
        if include_resistance:
            top_res = self.top_resistance_central_nodes(k=15)
        else:
            top_res = []
        return SpectralReport(
            graph_name=self.name,
            n_nodes=self.n,
            n_edges=self.edge_count,
            density=self.density,
            volume=self.volume,
            component_count=len(comps),
            component_sizes=[len(c) for c in comps],
            laplacian_kind=self.options.laplacian_kind,
            eigenvalues=[float(x) for x in dec.values[: max(0, int(modes))]],
            algebraic_connectivity=self.algebraic_connectivity(kind="unnormalized"),
            spectral_gap=self.spectral_gap(kind=self.options.laplacian_kind),
            spectral_radius=self.spectral_radius(kind=self.options.laplacian_kind),
            cheeger_cut=cut,
            entropy={
                "spectral_entropy": self.spectral_entropy(kind=self.options.laplacian_kind),
                "von_neumann_entropy": self.von_neumann_entropy(kind="unnormalized"),
            },
            localization=localization,
            heat_kernel_summaries=heat_summaries,
            top_resistance_central_nodes=top_res,
            metadata={"graph_summary": self.graph_summary(), "options": self.options.to_dict()},
        )

    def save_report_bundle(
        self,
        directory: Union[str, os.PathLike[str]],
        *,
        modes: int = 30,
        heat_times: Sequence[float] = (0.1, 1.0, 5.0),
    ) -> Dict[str, str]:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        report = self.report(modes=modes, heat_times=heat_times)
        paths = {
            "report_json": str(out / "spectral_report.json"),
            "report_md": str(out / "spectral_report.md"),
            "eigen_npz": str(out / "eigendecomposition.npz"),
            "node_features_csv": str(out / "node_spectral_features.csv"),
            "edge_geometry_csv": str(out / "edge_pair_geometry.csv"),
        }
        report.to_json(paths["report_json"])
        report.save_markdown(paths["report_md"])
        self.decompose(kind=self.options.laplacian_kind).save_npz(paths["eigen_npz"])
        self.save_node_spectral_features(paths["node_features_csv"], modes=min(12, modes), heat_times=heat_times)
        self.save_pairwise_geometry_features(paths["edge_geometry_csv"], heat_time=float(heat_times[0]) if heat_times else 1.0)
        return paths


# ---------------------------------------------------------------------------
# Standalone algorithms useful outside the class
# ---------------------------------------------------------------------------


def spectral_entropy(weights: np.ndarray, *, normalize: bool = True, eps: float = 1e-15) -> float:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w[np.isfinite(w)]
    w = np.maximum(w, 0.0)
    total = float(np.sum(w))
    if total <= eps:
        return 0.0
    p = w / total
    p = p[p > eps]
    H = float(-np.sum(p * np.log(p)))
    if normalize and p.size > 1:
        H /= math.log(p.size)
    return H


def heat_kernel_from_eigenpairs(values: np.ndarray, vectors: np.ndarray, t: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    weights = np.exp(-float(t) * values)
    K = (vectors * weights[None, :]) @ vectors.T
    return 0.5 * (K + K.T)


def diffusion_distance_from_kernel(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    row_norm = np.sum(K * K, axis=1)
    D2 = row_norm[:, None] + row_norm[None, :] - 2.0 * (K @ K.T)
    return np.sqrt(np.maximum(D2, 0.0))


def spectral_gap_from_values(values: Sequence[float]) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size < 2:
        return 0.0
    return float(arr[1] - arr[0])


def algebraic_connectivity_from_adjacency(A: ArrayLike) -> float:
    B = sanitize_adjacency(A)
    vals, _, _ = eigh_sorted(unnormalized_laplacian(B))
    if vals.size < 2:
        return 0.0
    return float(vals[1])


def fiedler_vector_from_adjacency(A: ArrayLike) -> np.ndarray:
    B = sanitize_adjacency(A)
    vals, vecs, _ = eigh_sorted(unnormalized_laplacian(B))
    if vals.size < 2:
        return np.zeros(B.shape[0], dtype=float)
    return vecs[:, 1].copy()


def cheeger_cut_from_adjacency(A: ArrayLike) -> SweepCutResult:
    return SpectralGeometryEngine.from_adjacency(A).cheeger_sweep()


def effective_resistance_matrix(A: ArrayLike) -> np.ndarray:
    return SpectralGeometryEngine.from_adjacency(A).effective_resistance().resistance


def commute_time_matrix(A: ArrayLike) -> np.ndarray:
    return SpectralGeometryEngine.from_adjacency(A).commute_time_distance()


def spectral_cluster_labels(A: ArrayLike, n_clusters: int, *, seed: int = 0) -> np.ndarray:
    return SpectralGeometryEngine.from_adjacency(A, options=SpectralOptions(random_seed=seed)).spectral_clustering(n_clusters).labels


def heat_kernel_signature(A: ArrayLike, times: Sequence[float]) -> np.ndarray:
    return SpectralGeometryEngine.from_adjacency(A).heat_kernel_signature(times)


def graph_spectral_summary(A: ArrayLike, *, name: str = "graph") -> Dict[str, Any]:
    return SpectralGeometryEngine.from_adjacency(A, name=name).report().to_dict()


def compare_spectral_geometries(
    before: Union[GraphBundle, ArrayLike],
    after: Union[GraphBundle, ArrayLike],
    *,
    modes: int = 20,
    heat_time: float = 1.0,
) -> Dict[str, Any]:
    """Compare two RINs through spectral geometry diagnostics.

    This is useful for future counterfactual modules: remove a residue/contact,
    run this function, and obtain a compact before/after spectral signature.
    """
    e0 = SpectralGeometryEngine(before, name="before")
    e1 = SpectralGeometryEngine(after, name="after")
    k0 = min(modes, e0.n)
    k1 = min(modes, e1.n)
    vals0 = e0.eigenvalues(k=k0)
    vals1 = e1.eigenvalues(k=k1)
    k = min(vals0.size, vals1.size)
    if k:
        eig_delta = vals1[:k] - vals0[:k]
    else:
        eig_delta = np.array([], dtype=float)
    h0 = e0.heat_kernel(heat_time).trace
    h1 = e1.heat_kernel(heat_time).trace
    return {
        "before": e0.graph_summary(),
        "after": e1.graph_summary(),
        "modes_compared": int(k),
        "eigenvalue_delta": eig_delta.tolist(),
        "eigenvalue_delta_l2": float(np.linalg.norm(eig_delta)) if eig_delta.size else 0.0,
        "algebraic_connectivity_before": e0.algebraic_connectivity(),
        "algebraic_connectivity_after": e1.algebraic_connectivity(),
        "algebraic_connectivity_delta": e1.algebraic_connectivity() - e0.algebraic_connectivity(),
        "heat_trace_before": float(h0),
        "heat_trace_after": float(h1),
        "heat_trace_delta": float(h1 - h0),
    }


# ---------------------------------------------------------------------------
# Lightweight command helpers.  This module is not wired into allograph.cli yet,
# but these helpers make it immediately usable by the later orchestrator/DSL.
# ---------------------------------------------------------------------------


def load_adjacency_csv(path: Union[str, os.PathLike[str]]) -> np.ndarray:
    A = np.loadtxt(path, delimiter=",")
    return _as_float_array(A)


def run_spectral_report_from_csv(
    csv_path: Union[str, os.PathLike[str]],
    out_dir: Union[str, os.PathLike[str]],
    *,
    name: Optional[str] = None,
    laplacian_kind: str = "normalized",
) -> Dict[str, str]:
    A = load_adjacency_csv(csv_path)
    engine = SpectralGeometryEngine.from_adjacency(
        A,
        name=name or Path(csv_path).stem,
        options=SpectralOptions(laplacian_kind=laplacian_kind),
    )
    return engine.save_report_bundle(out_dir)


def _demo_adjacency(n: int = 24, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(n, 3))
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    A = np.exp(-(dist**2) / 1.5)
    A[dist > 1.8] = 0.0
    np.fill_diagonal(A, 0.0)
    A = 0.5 * (A + A.T)
    # Ensure a weak backbone so the demo is connected like a protein chain.
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = max(A[i, i + 1], 0.25)
    return A


def demo_report() -> SpectralReport:
    engine = SpectralGeometryEngine.from_adjacency(_demo_adjacency(), name="spectral_geometry_demo")
    return engine.report()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Tiny module-level CLI for ad-hoc use.

    The repository's canonical CLI lives in ``allograph.cli``.  This entry point
    intentionally stays optional so adding the file cannot break existing CLI
    behavior.  Example:

    ``python -m allograph.core.spectral_geometry_engine --demo --out spectral_out``
    """
    import argparse

    parser = argparse.ArgumentParser(description="Spectral geometry analysis for RINet adjacency matrices")
    parser.add_argument("--csv", type=str, help="CSV adjacency matrix")
    parser.add_argument("--demo", action="store_true", help="Use an internal synthetic demo graph")
    parser.add_argument("--out", type=str, default="spectral_geometry_out", help="Output directory")
    parser.add_argument("--laplacian", type=str, default="normalized", help="Laplacian kind")
    parser.add_argument("--name", type=str, default=None, help="Graph/report name")
    args = parser.parse_args(argv)

    if args.demo:
        A = _demo_adjacency()
        engine = SpectralGeometryEngine.from_adjacency(
            A,
            name=args.name or "spectral_geometry_demo",
            options=SpectralOptions(laplacian_kind=args.laplacian),
        )
        paths = engine.save_report_bundle(args.out)
    elif args.csv:
        paths = run_spectral_report_from_csv(args.csv, args.out, name=args.name, laplacian_kind=args.laplacian)
    else:
        parser.error("provide --csv or --demo")
        return 2
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


__all__ = [
    "SpectralGeometryError",
    "InvalidGraphError",
    "DecompositionError",
    "ClusteringError",
    "ExportError",
    "SpectralOptions",
    "EigenDecomposition",
    "SweepCutResult",
    "SpectralEmbeddingResult",
    "SpectralClusterResult",
    "EffectiveResistanceResult",
    "HeatKernelResult",
    "DiffusionDistanceResult",
    "EigenvectorLocalizationResult",
    "SpectralReport",
    "SpectralGeometryEngine",
    "adjacency_from_graph",
    "metadata_from_graph",
    "residue_labels_from_graph",
    "sanitize_adjacency",
    "symmetrize_matrix",
    "degree_vector",
    "volume",
    "edge_count",
    "density",
    "connected_components",
    "largest_component_indices",
    "induced_subgraph",
    "unnormalized_laplacian",
    "normalized_laplacian",
    "random_walk_laplacian",
    "transition_matrix",
    "laplacian_matrix",
    "spectral_entropy",
    "heat_kernel_from_eigenpairs",
    "diffusion_distance_from_kernel",
    "algebraic_connectivity_from_adjacency",
    "fiedler_vector_from_adjacency",
    "cheeger_cut_from_adjacency",
    "effective_resistance_matrix",
    "commute_time_matrix",
    "spectral_cluster_labels",
    "heat_kernel_signature",
    "graph_spectral_summary",
    "compare_spectral_geometries",
    "run_spectral_report_from_csv",
    "demo_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
