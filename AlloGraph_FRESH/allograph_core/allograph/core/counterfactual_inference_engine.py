
# counterfactual_inference_engine.py
"""
Counterfactual intervention and sensitivity analysis for residue-interaction networks.

This module extends the lightweight RINet/AlloGraph core with a serious, reusable
counterfactual engine.  It is designed for questions of the form:

    * What changes if residue 88 is removed from the contact graph?
    * Which contacts most control diffusion into an active-site residue?
    * Can a path be blocked without globally destroying the network?
    * Which mutations are predicted to amplify or suppress signal transfer?
    * Are expensive exact ablations consistent with first-order sensitivity?

The implementation intentionally combines two computational regimes:

1. Exact interventions
   These clone the graph, apply an explicit edit, rerun diffusion, and compare
   before/after states.  This is simple, faithful to the model, and useful for
   final reporting.

2. First-order / adjoint sensitivity
   For the Euler heat-diffusion model used by the current project,

       x_{t+1} = (I - dt L) x_t,

   the derivative of a scalar objective J = c^T x_T with respect to an edge
   weight w_ij is computed by a discrete adjoint:

       dJ/dw_ij = sum_t y_{t+1}^T (-dt dL_ij) x_t,
       dL_ij = (e_i - e_j)(e_i - e_j)^T.

   This is not cosmetic mathematics: it avoids rerunning the diffusion model for
   every edge when ranking thousands of possible perturbations.  Exact scans are
   still available when correctness matters more than speed.

Only NumPy is required.  SciPy is used opportunistically if available for sparse
linear algebra, but the module remains compatible with the repository's current
minimal dependencies.
"""

from __future__ import annotations

import abc
import csv
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union

import numpy as np

try:  # Optional dependency; the package currently does not require SciPy.
    import scipy.sparse as _scipy_sparse  # type: ignore
    import scipy.sparse.linalg as _scipy_sparse_linalg  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - environment dependent
    _scipy_sparse = None
    _scipy_sparse_linalg = None
    _HAVE_SCIPY = False

try:
    from .graphio.types import GraphBundle
except Exception:  # pragma: no cover - permits standalone static analysis
    GraphBundle = Any  # type: ignore

try:
    from .graphio.io import adjacency_from_csv, bundle_from_adjacency
except Exception:  # pragma: no cover
    adjacency_from_csv = None  # type: ignore
    bundle_from_adjacency = None  # type: ignore

try:
    from .math.laplacian import graph_laplian as _typo_guard  # noqa: F401  # pragma: no cover
except Exception:
    pass

try:
    from .math.laplacian import graph_laplacian
except Exception:  # pragma: no cover
    def graph_laplacian(A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        return np.diag(np.sum(A, axis=1)) - A


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]
VectorLike = Union[np.ndarray, Sequence[float]]
NodeLike = Union[int, str]
Edge = Tuple[int, int]
JsonDict = Dict[str, Any]


# =============================================================================
# Errors
# =============================================================================


class CounterfactualError(RuntimeError):
    """Base class for all counterfactual engine errors."""


class InvalidGraphError(CounterfactualError):
    """Raised when an adjacency matrix is not usable as a graph."""


class InterventionError(CounterfactualError):
    """Raised when an intervention cannot be applied."""


class DiffusionError(CounterfactualError):
    """Raised when diffusion cannot be run or compared."""


class SensitivityError(CounterfactualError):
    """Raised when derivative-based approximations are invalid."""


class ExportError(CounterfactualError):
    """Raised when a report cannot be exported."""


# =============================================================================
# Core graph utilities
# =============================================================================


def _as_square_float_matrix(A: ArrayLike, *, name: str = "adjacency") -> np.ndarray:
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise InvalidGraphError(f"{name} must be a square matrix, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise InvalidGraphError(f"{name} contains NaN or infinite values")
    return arr.copy()


def _as_float_vector(x: VectorLike, n: int, *, name: str = "vector") -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got length {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr.copy()


def _stable_argsort_desc(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return np.lexsort((np.arange(values.size), -values))


def _stable_argsort_abs_desc(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return np.lexsort((np.arange(values.size), -np.abs(values)))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    return value


def _jsonify(value: Any) -> Any:
    """Convert NumPy/dataclass/path values to JSON-serializable objects."""
    if dataclasses.is_dataclass(value):
        return _jsonify(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _utc_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def symmetrize(A: np.ndarray, mode: str = "average") -> np.ndarray:
    mode = mode.lower().strip()
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
    symmetrize_mode: str = "average",
    remove_self_loops: bool = True,
    clip_negative: bool = True,
    negative_tolerance: float = 1e-12,
) -> np.ndarray:
    B = _as_square_float_matrix(A)
    if remove_self_loops:
        np.fill_diagonal(B, 0.0)
    if clip_negative:
        if float(np.min(B)) < -abs(float(negative_tolerance)):
            warnings.warn(
                "negative edge weights were clipped to zero for counterfactual analysis",
                RuntimeWarning,
                stacklevel=2,
            )
        B = np.where(B < 0.0, 0.0, B)
    B = symmetrize(B, symmetrize_mode)
    if remove_self_loops:
        np.fill_diagonal(B, 0.0)
    return B.astype(float, copy=False)


def degree_vector(A: np.ndarray) -> np.ndarray:
    return np.sum(A, axis=1).astype(float)


def edge_list(A: np.ndarray, *, tolerance: float = 0.0, upper_only: bool = True) -> List[Edge]:
    A = np.asarray(A, dtype=float)
    edges: List[Edge] = []
    n = A.shape[0]
    if upper_only:
        for i in range(n):
            js = np.flatnonzero(A[i, i + 1 :] > tolerance) + i + 1
            for j in js:
                edges.append((int(i), int(j)))
    else:
        rows, cols = np.nonzero(A > tolerance)
        for i, j in zip(rows, cols):
            if i != j:
                edges.append((int(i), int(j)))
    return edges


def graph_density(A: np.ndarray) -> float:
    n = A.shape[0]
    if n <= 1:
        return 0.0
    return 2.0 * len(edge_list(A)) / float(n * (n - 1))


def graph_hash(A: np.ndarray, *, precision: int = 12) -> str:
    B = np.round(np.asarray(A, dtype=float), precision)
    h = hashlib.sha256()
    h.update(str(B.shape).encode("utf-8"))
    h.update(B.tobytes())
    return h.hexdigest()


def normalize_signal(x: VectorLike, *, norm: str = "l1", eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1).copy()
    norm = norm.lower().strip()
    if norm in ("none", "raw"):
        return arr
    if norm == "l1":
        scale = float(np.sum(np.abs(arr)))
    elif norm == "l2":
        scale = float(np.linalg.norm(arr))
    elif norm == "max":
        scale = float(np.max(np.abs(arr))) if arr.size else 0.0
    elif norm == "sum":
        scale = float(np.sum(arr))
    else:
        raise ValueError(f"unknown normalization {norm!r}")
    if abs(scale) <= eps:
        return arr
    return arr / scale


def seed_signal(
    n: int,
    seed_nodes: Sequence[int],
    *,
    strengths: Optional[Sequence[float]] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    x = np.zeros(int(n), dtype=float)
    if strengths is None:
        strengths = [1.0] * len(seed_nodes)
    if len(seed_nodes) != len(strengths):
        raise ValueError("seed_nodes and strengths must have the same length")
    for node, strength in zip(seed_nodes, strengths):
        i = int(node)
        if i < 0 or i >= n:
            raise ValueError(f"seed node {i} outside graph with {n} nodes")
        x[i] += float(strength)
    if normalize:
        x = normalize_signal(x, norm=normalize)
    return x


def target_vector(
    n: int,
    targets: Optional[Sequence[int]] = None,
    *,
    weights: Optional[Sequence[float]] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    if targets is None:
        c = np.ones(n, dtype=float)
    else:
        c = np.zeros(n, dtype=float)
        if weights is None:
            weights = [1.0] * len(targets)
        if len(targets) != len(weights):
            raise ValueError("targets and weights must have the same length")
        for node, weight in zip(targets, weights):
            i = int(node)
            if i < 0 or i >= n:
                raise ValueError(f"target node {i} outside graph with {n} nodes")
            c[i] += float(weight)
    if normalize:
        c = normalize_signal(c, norm=normalize)
    return c


@dataclass
class GraphView:
    """Validated view of a GraphBundle or adjacency matrix.

    The class stores labels and metadata next to a sanitized adjacency matrix so
    that intervention reports can talk about residue IDs rather than only raw
    integer positions.
    """

    A: np.ndarray
    labels: List[str]
    name: str = "graph"
    meta: Dict[str, Any] = field(default_factory=dict)
    source_type: str = "array"

    @classmethod
    def from_graph(
        cls,
        graph: Union[GraphBundle, ArrayLike, "GraphView"],
        *,
        name: Optional[str] = None,
        symmetrize_mode: str = "average",
        remove_self_loops: bool = True,
        clip_negative: bool = True,
    ) -> "GraphView":
        if isinstance(graph, GraphView):
            return cls(
                A=sanitize_adjacency(
                    graph.A,
                    symmetrize_mode=symmetrize_mode,
                    remove_self_loops=remove_self_loops,
                    clip_negative=clip_negative,
                ),
                labels=list(graph.labels),
                name=name or graph.name,
                meta=dict(graph.meta),
                source_type=graph.source_type,
            )
        if hasattr(graph, "A"):
            A_raw = getattr(graph, "A")
            meta = dict(getattr(graph, "meta", {}) or {})
            graph_name = name or str(getattr(graph, "name", "GraphBundle"))
            labels = cls._labels_from_meta(meta, np.asarray(A_raw).shape[0])
            return cls(
                A=sanitize_adjacency(
                    A_raw,
                    symmetrize_mode=symmetrize_mode,
                    remove_self_loops=remove_self_loops,
                    clip_negative=clip_negative,
                ),
                labels=labels,
                name=graph_name,
                meta=meta,
                source_type="GraphBundle",
            )
        A = sanitize_adjacency(
            graph,  # type: ignore[arg-type]
            symmetrize_mode=symmetrize_mode,
            remove_self_loops=remove_self_loops,
            clip_negative=clip_negative,
        )
        labels = [str(i) for i in range(A.shape[0])]
        return cls(A=A, labels=labels, name=name or "array_graph", meta={}, source_type="array")

    @staticmethod
    def _labels_from_meta(meta: Mapping[str, Any], n: int) -> List[str]:
        for key in ("residue_ids", "residue_labels", "node_labels", "labels", "residues"):
            value = meta.get(key)
            if isinstance(value, (list, tuple)) and len(value) == n:
                return [str(v) for v in value]
        return [str(i) for i in range(n)]

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    @property
    def degrees(self) -> np.ndarray:
        return degree_vector(self.A)

    @property
    def edges(self) -> List[Edge]:
        return edge_list(self.A)

    def resolve_node(self, node: NodeLike) -> int:
        if isinstance(node, str) and not node.lstrip("+-").isdigit():
            try:
                return self.labels.index(node)
            except ValueError as exc:
                raise InterventionError(f"unknown residue label {node!r}") from exc
        idx = int(node)
        if idx < 0 or idx >= self.n:
            raise InterventionError(f"node {idx} outside graph with {self.n} nodes")
        return idx

    def resolve_edge(self, edge: Tuple[NodeLike, NodeLike]) -> Edge:
        i = self.resolve_node(edge[0])
        j = self.resolve_node(edge[1])
        if i == j:
            raise InterventionError("self-edge interventions are not supported")
        return (i, j) if i < j else (j, i)

    def label(self, node: int) -> str:
        i = int(node)
        if 0 <= i < self.n:
            return self.labels[i]
        return str(i)

    def summary(self) -> JsonDict:
        deg = self.degrees
        return {
            "name": self.name,
            "source_type": self.source_type,
            "nodes": self.n,
            "edges": len(self.edges),
            "density": graph_density(self.A),
            "degree_min": float(np.min(deg)) if deg.size else 0.0,
            "degree_max": float(np.max(deg)) if deg.size else 0.0,
            "degree_mean": float(np.mean(deg)) if deg.size else 0.0,
            "weight_sum": float(np.sum(self.A) / 2.0),
            "hash": graph_hash(self.A),
        }

    def to_bundle(self) -> GraphBundle:
        if GraphBundle is Any:  # pragma: no cover
            raise RuntimeError("GraphBundle is not importable in this environment")
        meta = dict(self.meta)
        meta.setdefault("residue_ids", list(self.labels))
        return GraphBundle(A=self.A.copy(), meta=meta, name=self.name)


# =============================================================================
# Diffusion and comparison objects
# =============================================================================


@dataclass(frozen=True)
class DiffusionConfig:
    """Configuration for deterministic diffusion comparisons."""

    steps: int = 50
    dt: float = 0.10
    model: str = "euler_heat"
    normalize_initial: Optional[str] = None
    clamp_negative: bool = False
    renormalize_each_step: Optional[str] = None
    stability_check: bool = True

    def validate(self) -> None:
        if int(self.steps) < 0:
            raise DiffusionError("steps must be nonnegative")
        if float(self.dt) < 0:
            raise DiffusionError("dt must be nonnegative")
        if self.model.lower() not in ("euler_heat", "heat", "diffusion", "discrete_hop"):
            raise DiffusionError(f"unknown diffusion model {self.model!r}")

    def to_json(self) -> JsonDict:
        return dataclasses.asdict(self)


@dataclass
class DiffusionTrace:
    """Full or partial trajectory of a diffusion run."""

    initial: np.ndarray
    final: np.ndarray
    steps: int
    dt: float
    model: str
    trajectory: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def state_at(self, t: int) -> np.ndarray:
        if self.trajectory is None:
            if int(t) == self.steps:
                return self.final.copy()
            if int(t) == 0:
                return self.initial.copy()
            raise ValueError("trajectory was not stored")
        return self.trajectory[int(t)].copy()

    def to_json(self, *, include_trajectory: bool = False) -> JsonDict:
        data: JsonDict = {
            "initial": self.initial.tolist(),
            "final": self.final.tolist(),
            "steps": int(self.steps),
            "dt": float(self.dt),
            "model": self.model,
            "meta": dict(self.meta),
        }
        if include_trajectory and self.trajectory is not None:
            data["trajectory"] = self.trajectory.tolist()
        return data


@dataclass(frozen=True)
class ComparisonMetrics:
    """Numerical summary of before/after diffusion differences."""

    l1_delta: float
    l2_delta: float
    linf_delta: float
    mean_abs_delta: float
    signed_sum_delta: float
    cosine_similarity: float
    pearson_correlation: float
    spearman_correlation: float
    top1_before: int
    top1_after: int
    top1_changed: bool
    topk_overlap: float
    target_delta: Optional[float] = None
    target_abs_delta: Optional[float] = None
    target_relative_delta: Optional[float] = None

    def to_json(self) -> JsonDict:
        return dataclasses.asdict(self)


@dataclass
class DiffusionComparison:
    """Before/after diffusion result for one intervention."""

    intervention_id: str
    intervention_name: str
    before: np.ndarray
    after: np.ndarray
    delta: np.ndarray
    metrics: ComparisonMetrics
    intervention_meta: Dict[str, Any] = field(default_factory=dict)
    graph_meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        return {
            "intervention_id": self.intervention_id,
            "intervention_name": self.intervention_name,
            "before": self.before.tolist(),
            "after": self.after.tolist(),
            "delta": self.delta.tolist(),
            "metrics": self.metrics.to_json(),
            "intervention_meta": _jsonify(self.intervention_meta),
            "graph_meta": _jsonify(self.graph_meta),
        }

    def compact_row(self) -> JsonDict:
        row = {
            "intervention_id": self.intervention_id,
            "intervention_name": self.intervention_name,
        }
        row.update(self.metrics.to_json())
        for key, value in self.intervention_meta.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"intervention_{key}"] = value
        return row


def diffusion_operator(A: np.ndarray, dt: float) -> np.ndarray:
    """Euler heat diffusion matrix M = I - dt L."""
    L = graph_laplacian(A)
    return np.eye(A.shape[0], dtype=float) - float(dt) * L


def _check_euler_stability(A: np.ndarray, dt: float) -> None:
    """Warn when explicit Euler is likely unstable for the Laplacian."""
    if A.shape[0] == 0 or dt <= 0:
        return
    max_degree = float(np.max(degree_vector(A))) if A.size else 0.0
    # Gershgorin bound lambda_max(L) <= 2 d_max. Euler heat is stable roughly
    # for dt * lambda_max <= 2.
    if max_degree > 0 and dt * 2.0 * max_degree > 2.1:
        warnings.warn(
            "explicit Euler diffusion may be unstable: dt * 2 * max_degree is large",
            RuntimeWarning,
            stacklevel=2,
        )


def run_diffusion(
    A: np.ndarray,
    x0: VectorLike,
    config: DiffusionConfig = DiffusionConfig(),
    *,
    store_trajectory: bool = False,
) -> DiffusionTrace:
    """Run a deterministic diffusion model on an adjacency matrix.

    The default model matches the repository's existing ``DiffusionModel``:
    repeated explicit Euler updates by ``x <- x - dt L x``.
    """
    config.validate()
    A = _as_square_float_matrix(A)
    n = A.shape[0]
    x = _as_float_vector(x0, n, name="x0")
    if config.normalize_initial:
        x = normalize_signal(x, norm=config.normalize_initial)
    initial = x.copy()
    trajectory = np.zeros((int(config.steps) + 1, n), dtype=float) if store_trajectory else None
    if trajectory is not None:
        trajectory[0] = x
    model = config.model.lower().strip()
    if config.stability_check and model in ("euler_heat", "heat", "diffusion"):
        _check_euler_stability(A, float(config.dt))
    if model in ("euler_heat", "heat", "diffusion"):
        L = graph_laplacian(A)
        for t in range(int(config.steps)):
            x = x - float(config.dt) * (L @ x)
            if config.clamp_negative:
                x = np.maximum(x, 0.0)
            if config.renormalize_each_step:
                x = normalize_signal(x, norm=config.renormalize_each_step)
            if trajectory is not None:
                trajectory[t + 1] = x
    elif model == "discrete_hop":
        for t in range(int(config.steps)):
            x = A @ x
            s = float(np.sum(np.abs(x))) + 1e-12
            x = x / s
            if trajectory is not None:
                trajectory[t + 1] = x
    else:  # config.validate should prevent this
        raise DiffusionError(f"unsupported model {config.model!r}")
    return DiffusionTrace(
        initial=initial,
        final=x.copy(),
        steps=int(config.steps),
        dt=float(config.dt),
        model=config.model,
        trajectory=trajectory,
        meta={"stored_trajectory": store_trajectory},
    )


def rankdata(values: VectorLike) -> np.ndarray:
    """Average ranks for ties, compatible with Spearman correlation."""
    x = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def pearson_corr(a: VectorLike, b: VectorLike, *, eps: float = 1e-12) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("correlation inputs must have same shape")
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= eps:
        return 0.0
    return float(np.dot(x, y) / denom)


def spearman_corr(a: VectorLike, b: VectorLike) -> float:
    return pearson_corr(rankdata(a), rankdata(b))


def cosine_similarity(a: VectorLike, b: VectorLike, *, eps: float = 1e-12) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= eps:
        return 0.0
    return float(np.dot(x, y) / denom)


def topk_overlap(a: VectorLike, b: VectorLike, k: int) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    k = max(1, min(int(k), x.size))
    ia = set(int(i) for i in _stable_argsort_desc(x)[:k])
    ib = set(int(i) for i in _stable_argsort_desc(y)[:k])
    return float(len(ia.intersection(ib)) / k)


def compare_states(
    before: VectorLike,
    after: VectorLike,
    *,
    target: Optional[VectorLike] = None,
    topk: int = 10,
    eps: float = 1e-12,
) -> ComparisonMetrics:
    b = np.asarray(before, dtype=float).reshape(-1)
    a = np.asarray(after, dtype=float).reshape(-1)
    if b.shape != a.shape:
        raise DiffusionError(f"cannot compare states with shapes {b.shape} and {a.shape}")
    delta = a - b
    if b.size == 0:
        top1_before = top1_after = -1
    else:
        top1_before = int(np.argmax(b))
        top1_after = int(np.argmax(a))
    target_delta = None
    target_abs_delta = None
    target_relative_delta = None
    if target is not None:
        c = _as_float_vector(target, b.size, name="target")
        base = float(np.dot(c, b))
        shifted = float(np.dot(c, a))
        target_delta = shifted - base
        target_abs_delta = abs(target_delta)
        target_relative_delta = float(target_delta / (abs(base) + eps))
    return ComparisonMetrics(
        l1_delta=float(np.sum(np.abs(delta))),
        l2_delta=float(np.linalg.norm(delta)),
        linf_delta=float(np.max(np.abs(delta))) if delta.size else 0.0,
        mean_abs_delta=float(np.mean(np.abs(delta))) if delta.size else 0.0,
        signed_sum_delta=float(np.sum(delta)),
        cosine_similarity=cosine_similarity(b, a),
        pearson_correlation=pearson_corr(b, a),
        spearman_correlation=spearman_corr(b, a),
        top1_before=top1_before,
        top1_after=top1_after,
        top1_changed=bool(top1_before != top1_after),
        topk_overlap=topk_overlap(b, a, topk),
        target_delta=target_delta,
        target_abs_delta=target_abs_delta,
        target_relative_delta=target_relative_delta,
    )


# =============================================================================
# Intervention hierarchy
# =============================================================================


@dataclass(frozen=True)
class Intervention(abc.ABC):
    """Abstract base class for graph counterfactual interventions."""

    name: str = "intervention"
    preserve_size: bool = True

    @abc.abstractmethod
    def apply(self, graph: GraphView) -> np.ndarray:
        """Return a new adjacency matrix after applying the intervention."""

    @abc.abstractmethod
    def affected_nodes(self, graph: GraphView) -> List[int]:
        """Return node indices directly touched by the intervention."""

    def affected_edges(self, graph: GraphView) -> List[Edge]:
        return []

    def metadata(self, graph: GraphView) -> JsonDict:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "preserve_size": bool(self.preserve_size),
            "affected_nodes": self.affected_nodes(graph),
            "affected_labels": [graph.label(i) for i in self.affected_nodes(graph)],
            "affected_edges": self.affected_edges(graph),
        }

    def intervention_id(self, graph: Optional[GraphView] = None) -> str:
        payload: JsonDict = {"class": type(self).__name__, "data": dataclasses.asdict(self)}
        if graph is not None:
            payload["graph_hash"] = graph_hash(graph.A)
        text = json.dumps(_jsonify(payload), sort_keys=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def short_name(self, graph: Optional[GraphView] = None) -> str:
        return self.name or type(self).__name__


@dataclass(frozen=True)
class NodeRemovalIntervention(Intervention):
    """Remove one node either by isolation or by deletion.

    ``mode='isolate'`` preserves the state dimension by zeroing all incident
    contacts.  This is usually the right choice for before/after residue-score
    comparisons.  ``mode='delete'`` physically removes the row and column and is
    mainly useful for graph-theoretic experiments where dimension changes are
    acceptable.
    """

    node: NodeLike = 0
    mode: str = "isolate"
    name: str = "remove_node"
    preserve_size: bool = True

    def apply(self, graph: GraphView) -> np.ndarray:
        i = graph.resolve_node(self.node)
        mode = self.mode.lower().strip()
        if mode == "isolate":
            B = graph.A.copy()
            B[i, :] = 0.0
            B[:, i] = 0.0
            return B
        if mode == "delete":
            if graph.n <= 1:
                raise InterventionError("cannot delete the only node")
            return np.delete(np.delete(graph.A, i, axis=0), i, axis=1)
        raise InterventionError(f"unknown node removal mode {self.mode!r}")

    def affected_nodes(self, graph: GraphView) -> List[int]:
        return [graph.resolve_node(self.node)]

    def affected_edges(self, graph: GraphView) -> List[Edge]:
        i = graph.resolve_node(self.node)
        return [(min(i, j), max(i, j)) for j in np.flatnonzero(graph.A[i] > 0) if int(j) != i]

    def metadata(self, graph: GraphView) -> JsonDict:
        data = super().metadata(graph)
        i = graph.resolve_node(self.node)
        data.update({"node": i, "node_label": graph.label(i), "mode": self.mode})
        return data


@dataclass(frozen=True)
class EdgeRemovalIntervention(Intervention):
    """Remove a single undirected contact edge."""

    i: NodeLike = 0
    j: NodeLike = 1
    name: str = "remove_edge"
    preserve_size: bool = True

    def apply(self, graph: GraphView) -> np.ndarray:
        i, j = graph.resolve_edge((self.i, self.j))
        B = graph.A.copy()
        B[i, j] = 0.0
        B[j, i] = 0.0
        return B

    def affected_nodes(self, graph: GraphView) -> List[int]:
        i, j = graph.resolve_edge((self.i, self.j))
        return [i, j]

    def affected_edges(self, graph: GraphView) -> List[Edge]:
        return [graph.resolve_edge((self.i, self.j))]

    def metadata(self, graph: GraphView) -> JsonDict:
        data = super().metadata(graph)
        i, j = graph.resolve_edge((self.i, self.j))
        data.update({
            "i": i,
            "j": j,
            "i_label": graph.label(i),
            "j_label": graph.label(j),
            "original_weight": float(graph.A[i, j]),
        })
        return data


@dataclass(frozen=True)
class EdgeReweightIntervention(Intervention):
    """Multiply or set one contact edge weight."""

    i: NodeLike = 0
    j: NodeLike = 1
    factor: Optional[float] = 0.5
    new_weight: Optional[float] = None
    name: str = "reweight_edge"
    preserve_size: bool = True

    def apply(self, graph: GraphView) -> np.ndarray:
        i, j = graph.resolve_edge((self.i, self.j))
        B = graph.A.copy()
        old = float(B[i, j])
        if self.new_weight is not None:
            w = float(self.new_weight)
        else:
            w = old * float(self.factor if self.factor is not None else 1.0)
        if w < 0:
            raise InterventionError("edge weights must remain nonnegative")
        B[i, j] = w
        B[j, i] = w
        return B

    def affected_nodes(self, graph: GraphView) -> List[int]:
        i, j = graph.resolve_edge((self.i, self.j))
        return [i, j]

    def affected_edges(self, graph: GraphView) -> List[Edge]:
        return [graph.resolve_edge((self.i, self.j))]

    def metadata(self, graph: GraphView) -> JsonDict:
        i, j = graph.resolve_edge((self.i, self.j))
        old = float(graph.A[i, j])
        new = float(self.new_weight) if self.new_weight is not None else old * float(self.factor if self.factor is not None else 1.0)
        data = super().metadata(graph)
        data.update({
            "i": i,
            "j": j,
            "i_label": graph.label(i),
            "j_label": graph.label(j),
            "factor": self.factor,
            "new_weight": new,
            "original_weight": old,
            "delta_weight": new - old,
        })
        return data


@dataclass(frozen=True)
class LocalMutationIntervention(Intervention):
    """Mutation-style local perturbation around a residue.

    This is not a biochemical force field.  It is a graph-level proxy for a
    mutation that weakens or strengthens the contact neighborhood of one residue,
    optionally with distance decay across one- or two-hop shells.
    """

    node: NodeLike = 0
    factor: float = 0.5
    radius: int = 1
    decay: float = 0.5
    include_node_edges: bool = True
    name: str = "local_mutation"
    preserve_size: bool = True

    def _shell_distances(self, graph: GraphView, source: int) -> Dict[int, int]:
        radius = max(0, int(self.radius))
        distances: Dict[int, int] = {source: 0}
        frontier = [source]
        for depth in range(1, radius + 1):
            next_frontier: List[int] = []
            for u in frontier:
                for v in np.flatnonzero(graph.A[u] > 0):
                    v_int = int(v)
                    if v_int not in distances:
                        distances[v_int] = depth
                        next_frontier.append(v_int)
            frontier = next_frontier
            if not frontier:
                break
        return distances

    def apply(self, graph: GraphView) -> np.ndarray:
        i = graph.resolve_node(self.node)
        B = graph.A.copy()
        shell = self._shell_distances(graph, i)
        if self.radius <= 0:
            return B
        for u, v in edge_list(graph.A):
            du = shell.get(u)
            dv = shell.get(v)
            if du is None and dv is None:
                continue
            if not self.include_node_edges and (u == i or v == i):
                continue
            d = min([d for d in (du, dv) if d is not None])
            local_strength = float(self.factor) + (1.0 - float(self.factor)) * (1.0 - float(self.decay) ** max(0, self.radius - d + 1))
            # factor is strongest near the mutation site; farther shells approach 1.
            shell_factor = float(self.factor) if d <= 1 else 1.0 - (1.0 - float(self.factor)) * (float(self.decay) ** (d - 1))
            B[u, v] = max(0.0, B[u, v] * shell_factor)
            B[v, u] = B[u, v]
        return B

    def affected_nodes(self, graph: GraphView) -> List[int]:
        i = graph.resolve_node(self.node)
        return sorted(self._shell_distances(graph, i))

    def affected_edges(self, graph: GraphView) -> List[Edge]:
        touched = set(self.affected_nodes(graph))
        return [(u, v) for u, v in edge_list(graph.A) if u in touched or v in touched]

    def metadata(self, graph: GraphView) -> JsonDict:
        i = graph.resolve_node(self.node)
        data = super().metadata(graph)
        data.update({
            "node": i,
            "node_label": graph.label(i),
            "factor": float(self.factor),
            "radius": int(self.radius),
            "decay": float(self.decay),
            "include_node_edges": bool(self.include_node_edges),
        })
        return data


@dataclass(frozen=True)
class PathBlockingIntervention(Intervention):
    """Weaken or remove all contacts along a specified residue path."""

    path: Sequence[NodeLike] = field(default_factory=list)
    factor: float = 0.0
    name: str = "block_path"
    preserve_size: bool = True

    def _resolved_path(self, graph: GraphView) -> List[int]:
        nodes = [graph.resolve_node(v) for v in self.path]
        if len(nodes) < 2:
            raise InterventionError("path blocking requires at least two nodes")
        return nodes

    def apply(self, graph: GraphView) -> np.ndarray:
        nodes = self._resolved_path(graph)
        B = graph.A.copy()
        factor = float(self.factor)
        if factor < 0:
            raise InterventionError("path blocking factor must be nonnegative")
        for u, v in zip(nodes[:-1], nodes[1:]):
            i, j = (u, v) if u < v else (v, u)
            B[i, j] = B[i, j] * factor
            B[j, i] = B[i, j]
        return B

    def affected_nodes(self, graph: GraphView) -> List[int]:
        return self._resolved_path(graph)

    def affected_edges(self, graph: GraphView) -> List[Edge]:
        nodes = self._resolved_path(graph)
        return [(min(u, v), max(u, v)) for u, v in zip(nodes[:-1], nodes[1:])]

    def metadata(self, graph: GraphView) -> JsonDict:
        nodes = self._resolved_path(graph)
        data = super().metadata(graph)
        data.update({
            "path": nodes,
            "path_labels": [graph.label(i) for i in nodes],
            "factor": float(self.factor),
            "original_path_weights": [float(graph.A[u, v]) for u, v in zip(nodes[:-1], nodes[1:])],
        })
        return data


@dataclass(frozen=True)
class ContactThresholdIntervention(Intervention):
    """Remove or keep contacts according to weight threshold rules."""

    threshold: float = 0.0
    mode: str = "remove_below"
    name: str = "threshold_contacts"
    preserve_size: bool = True

    def apply(self, graph: GraphView) -> np.ndarray:
        B = graph.A.copy()
        threshold = float(self.threshold)
        mode = self.mode.lower().strip()
        if mode == "remove_below":
            B[B < threshold] = 0.0
        elif mode == "remove_above":
            B[B > threshold] = 0.0
        elif mode == "keep_top_fraction":
            edges = edge_list(B)
            if not edges:
                return B
            weights = np.array([B[i, j] for i, j in edges], dtype=float)
            frac = min(max(threshold, 0.0), 1.0)
            keep = max(0, int(math.ceil(frac * len(edges))))
            order = _stable_argsort_desc(weights)
            keep_edges = {edges[int(k)] for k in order[:keep]}
            C = np.zeros_like(B)
            for i, j in keep_edges:
                C[i, j] = B[i, j]
                C[j, i] = B[j, i]
            B = C
        else:
            raise InterventionError(f"unknown threshold intervention mode {self.mode!r}")
        np.fill_diagonal(B, 0.0)
        return B

    def affected_nodes(self, graph: GraphView) -> List[int]:
        before = graph.A
        after = self.apply(graph)
        changed = np.flatnonzero(np.any(np.abs(after - before) > 1e-15, axis=1))
        return [int(i) for i in changed]

    def affected_edges(self, graph: GraphView) -> List[Edge]:
        before = graph.A
        after = self.apply(graph)
        changed: List[Edge] = []
        for i, j in edge_list(np.maximum(before, after)):
            if abs(float(after[i, j] - before[i, j])) > 1e-15:
                changed.append((i, j))
        return changed

    def metadata(self, graph: GraphView) -> JsonDict:
        data = super().metadata(graph)
        data.update({"threshold": float(self.threshold), "mode": self.mode})
        return data


@dataclass(frozen=True)
class CompositeIntervention(Intervention):
    """Apply several interventions in sequence."""

    interventions: Sequence[Intervention] = field(default_factory=list)
    name: str = "composite"
    preserve_size: bool = True

    def apply(self, graph: GraphView) -> np.ndarray:
        current = graph
        A = graph.A.copy()
        for intervention in self.interventions:
            local_view = GraphView(A=A, labels=graph.labels, name=graph.name, meta=graph.meta, source_type=graph.source_type)
            A = intervention.apply(local_view)
            if A.shape != graph.A.shape:
                raise InterventionError("composite interventions currently require size-preserving edits")
        return A

    def affected_nodes(self, graph: GraphView) -> List[int]:
        nodes: set[int] = set()
        for intervention in self.interventions:
            nodes.update(intervention.affected_nodes(graph))
        return sorted(nodes)

    def affected_edges(self, graph: GraphView) -> List[Edge]:
        edges: set[Edge] = set()
        for intervention in self.interventions:
            edges.update(intervention.affected_edges(graph))
        return sorted(edges)

    def metadata(self, graph: GraphView) -> JsonDict:
        data = super().metadata(graph)
        data["children"] = [child.metadata(graph) for child in self.interventions]
        return data


# =============================================================================
# Exact counterfactual evaluation
# =============================================================================


@dataclass
class CounterfactualEvaluation:
    """Detailed evaluation of a single intervention."""

    graph_name: str
    graph_hash_before: str
    graph_hash_after: str
    intervention: Intervention
    intervention_id: str
    comparison: DiffusionComparison
    before_graph_summary: JsonDict
    after_graph_summary: JsonDict
    runtime_seconds: float
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> JsonDict:
        return {
            "graph_name": self.graph_name,
            "graph_hash_before": self.graph_hash_before,
            "graph_hash_after": self.graph_hash_after,
            "intervention_id": self.intervention_id,
            "intervention": self.comparison.intervention_meta,
            "comparison": self.comparison.to_json(),
            "before_graph_summary": _jsonify(self.before_graph_summary),
            "after_graph_summary": _jsonify(self.after_graph_summary),
            "runtime_seconds": float(self.runtime_seconds),
            "notes": list(self.notes),
        }

    def compact_row(self) -> JsonDict:
        row = self.comparison.compact_row()
        row["runtime_seconds"] = float(self.runtime_seconds)
        row["graph_hash_after"] = self.graph_hash_after
        return row


class CounterfactualEngine:
    """Main exact intervention runner.

    Parameters
    ----------
    graph:
        ``GraphBundle`` or square adjacency matrix.
    diffusion_config:
        Diffusion settings.  Defaults to the same Euler heat model as the
        existing core diffusion implementation.
    seed_state:
        Optional default initial signal.  If omitted, each evaluation must pass
        one explicitly.
    """

    def __init__(
        self,
        graph: Union[GraphBundle, ArrayLike, GraphView],
        *,
        diffusion_config: DiffusionConfig = DiffusionConfig(),
        seed_state: Optional[VectorLike] = None,
        target: Optional[VectorLike] = None,
        symmetrize_mode: str = "average",
    ) -> None:
        self.graph = GraphView.from_graph(graph, symmetrize_mode=symmetrize_mode)
        self.config = diffusion_config
        self.config.validate()
        self.seed_state = None if seed_state is None else _as_float_vector(seed_state, self.graph.n, name="seed_state")
        self.target = None if target is None else _as_float_vector(target, self.graph.n, name="target")
        self._baseline_cache: Dict[str, DiffusionTrace] = {}

    @property
    def A(self) -> np.ndarray:
        return self.graph.A

    @property
    def n(self) -> int:
        return self.graph.n

    def baseline(self, seed_state: Optional[VectorLike] = None, *, store_trajectory: bool = False) -> DiffusionTrace:
        x0 = self._resolve_seed(seed_state)
        key = hashlib.sha1(
            json.dumps({
                "seed": np.round(x0, 12).tolist(),
                "config": self.config.to_json(),
                "trajectory": bool(store_trajectory),
                "graph": graph_hash(self.A),
            }, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if key not in self._baseline_cache:
            self._baseline_cache[key] = run_diffusion(self.A, x0, self.config, store_trajectory=store_trajectory)
        return self._baseline_cache[key]

    def _resolve_seed(self, seed_state: Optional[VectorLike]) -> np.ndarray:
        if seed_state is not None:
            return _as_float_vector(seed_state, self.n, name="seed_state")
        if self.seed_state is not None:
            return self.seed_state.copy()
        raise DiffusionError("no seed_state supplied")

    def _resolve_target(self, target: Optional[VectorLike]) -> Optional[np.ndarray]:
        if target is not None:
            return _as_float_vector(target, self.n, name="target")
        if self.target is not None:
            return self.target.copy()
        return None

    def evaluate(
        self,
        intervention: Intervention,
        *,
        seed_state: Optional[VectorLike] = None,
        target: Optional[VectorLike] = None,
        topk: int = 10,
        store_trajectory: bool = False,
    ) -> CounterfactualEvaluation:
        start = time.perf_counter()
        x0 = self._resolve_seed(seed_state)
        c = self._resolve_target(target)
        before_trace = self.baseline(x0, store_trajectory=store_trajectory)
        A_after = intervention.apply(self.graph)
        A_after = _as_square_float_matrix(A_after, name="intervened adjacency")
        if A_after.shape != self.A.shape:
            raise InterventionError(
                "this engine compares diffusion vectors of equal dimension; use preserve_size interventions"
            )
        after_trace = run_diffusion(A_after, x0, self.config, store_trajectory=store_trajectory)
        metrics = compare_states(before_trace.final, after_trace.final, target=c, topk=topk)
        delta = after_trace.final - before_trace.final
        after_view = GraphView(A=A_after, labels=self.graph.labels, name=self.graph.name, meta=self.graph.meta, source_type=self.graph.source_type)
        iid = intervention.intervention_id(self.graph)
        comparison = DiffusionComparison(
            intervention_id=iid,
            intervention_name=intervention.short_name(self.graph),
            before=before_trace.final.copy(),
            after=after_trace.final.copy(),
            delta=delta,
            metrics=metrics,
            intervention_meta=intervention.metadata(self.graph),
            graph_meta={"before": self.graph.summary(), "after": after_view.summary()},
        )
        elapsed = time.perf_counter() - start
        return CounterfactualEvaluation(
            graph_name=self.graph.name,
            graph_hash_before=graph_hash(self.A),
            graph_hash_after=graph_hash(A_after),
            intervention=intervention,
            intervention_id=iid,
            comparison=comparison,
            before_graph_summary=self.graph.summary(),
            after_graph_summary=after_view.summary(),
            runtime_seconds=float(elapsed),
        )

    def evaluate_many(
        self,
        interventions: Iterable[Intervention],
        *,
        seed_state: Optional[VectorLike] = None,
        target: Optional[VectorLike] = None,
        topk: int = 10,
        limit: Optional[int] = None,
    ) -> List[CounterfactualEvaluation]:
        results: List[CounterfactualEvaluation] = []
        for k, intervention in enumerate(interventions):
            if limit is not None and k >= int(limit):
                break
            results.append(self.evaluate(intervention, seed_state=seed_state, target=target, topk=topk))
        return results

    def compare_intervention_set(
        self,
        interventions: Iterable[Intervention],
        *,
        objective: str = "l2_delta",
        descending: bool = True,
        seed_state: Optional[VectorLike] = None,
        target: Optional[VectorLike] = None,
        topk: int = 10,
    ) -> List[CounterfactualEvaluation]:
        results = self.evaluate_many(interventions, seed_state=seed_state, target=target, topk=topk)
        def key(ev: CounterfactualEvaluation) -> float:
            return _safe_float(getattr(ev.comparison.metrics, objective, None), 0.0)
        return sorted(results, key=key, reverse=descending)


# =============================================================================
# Exact ablation scanner
# =============================================================================


@dataclass
class AblationScanConfig:
    scan_nodes: bool = True
    scan_edges: bool = True
    node_mode: str = "isolate"
    edge_factor: float = 0.0
    max_edges: Optional[int] = None
    edge_weight_threshold: float = 0.0
    rank_by: str = "l2_delta"
    top_k: int = 25
    include_local_mutations: bool = False
    mutation_factors: Tuple[float, ...] = (0.25, 0.5, 1.5)
    mutation_radius: int = 1


@dataclass
class AblationScanReport:
    graph_summary: JsonDict
    config: AblationScanConfig
    ranked: List[CounterfactualEvaluation]
    node_results: List[CounterfactualEvaluation] = field(default_factory=list)
    edge_results: List[CounterfactualEvaluation] = field(default_factory=list)
    mutation_results: List[CounterfactualEvaluation] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)

    def to_json(self, *, compact: bool = False) -> JsonDict:
        if compact:
            ranked = [ev.compact_row() for ev in self.ranked]
            nodes = [ev.compact_row() for ev in self.node_results]
            edges = [ev.compact_row() for ev in self.edge_results]
            mutations = [ev.compact_row() for ev in self.mutation_results]
        else:
            ranked = [ev.to_json() for ev in self.ranked]
            nodes = [ev.to_json() for ev in self.node_results]
            edges = [ev.to_json() for ev in self.edge_results]
            mutations = [ev.to_json() for ev in self.mutation_results]
        return {
            "created_at": self.created_at,
            "graph_summary": _jsonify(self.graph_summary),
            "config": _jsonify(self.config),
            "ranked": ranked,
            "node_results": nodes,
            "edge_results": edges,
            "mutation_results": mutations,
        }

    def to_csv(self, path: Union[str, os.PathLike[str]]) -> None:
        rows = [ev.compact_row() for ev in self.ranked]
        write_csv_rows(path, rows)

    def to_json_file(self, path: Union[str, os.PathLike[str]], *, compact: bool = False, indent: int = 2) -> None:
        write_json(path, self.to_json(compact=compact), indent=indent)


class ExactAblationScanner:
    """Generate and rank exact node/edge/local-mutation ablations."""

    def __init__(self, engine: CounterfactualEngine, config: AblationScanConfig = AblationScanConfig()) -> None:
        self.engine = engine
        self.config = config

    def node_interventions(self) -> Iterator[Intervention]:
        for i in range(self.engine.n):
            yield NodeRemovalIntervention(node=i, mode=self.config.node_mode)

    def edge_interventions(self) -> Iterator[Intervention]:
        edges = [e for e in edge_list(self.engine.A, tolerance=self.config.edge_weight_threshold)]
        if self.config.max_edges is not None:
            weights = np.array([self.engine.A[i, j] for i, j in edges], dtype=float)
            order = _stable_argsort_desc(weights)[: int(self.config.max_edges)]
            edges = [edges[int(k)] for k in order]
        for i, j in edges:
            if self.config.edge_factor == 0.0:
                yield EdgeRemovalIntervention(i=i, j=j)
            else:
                yield EdgeReweightIntervention(i=i, j=j, factor=self.config.edge_factor)

    def mutation_interventions(self) -> Iterator[Intervention]:
        for i in range(self.engine.n):
            for factor in self.config.mutation_factors:
                yield LocalMutationIntervention(node=i, factor=float(factor), radius=self.config.mutation_radius)

    def run(
        self,
        *,
        seed_state: Optional[VectorLike] = None,
        target: Optional[VectorLike] = None,
        topk_compare: int = 10,
    ) -> AblationScanReport:
        node_results: List[CounterfactualEvaluation] = []
        edge_results: List[CounterfactualEvaluation] = []
        mutation_results: List[CounterfactualEvaluation] = []
        if self.config.scan_nodes:
            node_results = self.engine.evaluate_many(self.node_interventions(), seed_state=seed_state, target=target, topk=topk_compare)
        if self.config.scan_edges:
            edge_results = self.engine.evaluate_many(self.edge_interventions(), seed_state=seed_state, target=target, topk=topk_compare)
        if self.config.include_local_mutations:
            mutation_results = self.engine.evaluate_many(self.mutation_interventions(), seed_state=seed_state, target=target, topk=topk_compare)
        all_results = node_results + edge_results + mutation_results
        rank_by = self.config.rank_by
        ranked = sorted(
            all_results,
            key=lambda ev: _safe_float(getattr(ev.comparison.metrics, rank_by, None), 0.0),
            reverse=True,
        )[: int(self.config.top_k)]
        return AblationScanReport(
            graph_summary=self.engine.graph.summary(),
            config=self.config,
            ranked=ranked,
            node_results=node_results,
            edge_results=edge_results,
            mutation_results=mutation_results,
        )


# =============================================================================
# First-order and adjoint sensitivity
# =============================================================================


@dataclass(frozen=True)
class EdgeSensitivity:
    i: int
    j: int
    weight: float
    derivative: float
    absolute_derivative: float
    predicted_delta_for_removal: float
    score: float
    label_i: str = ""
    label_j: str = ""

    def to_json(self) -> JsonDict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class NodeSensitivity:
    node: int
    label: str
    degree: float
    incident_abs_derivative_sum: float
    incident_signed_derivative_sum: float
    predicted_removal_delta: float
    max_incident_abs_derivative: float
    incident_edges: int

    def to_json(self) -> JsonDict:
        return dataclasses.asdict(self)


@dataclass
class SensitivityReport:
    graph_summary: JsonDict
    diffusion_config: DiffusionConfig
    objective_description: str
    edge_sensitivities: List[EdgeSensitivity]
    node_sensitivities: List[NodeSensitivity]
    baseline_objective: float
    created_at: str = field(default_factory=_utc_now)
    meta: Dict[str, Any] = field(default_factory=dict)

    def top_edges(self, k: int = 20, *, by: str = "score") -> List[EdgeSensitivity]:
        return sorted(self.edge_sensitivities, key=lambda e: _safe_float(getattr(e, by), 0.0), reverse=True)[: int(k)]

    def top_nodes(self, k: int = 20, *, by: str = "incident_abs_derivative_sum") -> List[NodeSensitivity]:
        return sorted(self.node_sensitivities, key=lambda e: _safe_float(getattr(e, by), 0.0), reverse=True)[: int(k)]

    def to_json(self) -> JsonDict:
        return {
            "created_at": self.created_at,
            "graph_summary": _jsonify(self.graph_summary),
            "diffusion_config": self.diffusion_config.to_json(),
            "objective_description": self.objective_description,
            "baseline_objective": float(self.baseline_objective),
            "edge_sensitivities": [e.to_json() for e in self.edge_sensitivities],
            "node_sensitivities": [n.to_json() for n in self.node_sensitivities],
            "meta": _jsonify(self.meta),
        }

    def to_edge_csv(self, path: Union[str, os.PathLike[str]]) -> None:
        write_csv_rows(path, [e.to_json() for e in self.edge_sensitivities])

    def to_node_csv(self, path: Union[str, os.PathLike[str]]) -> None:
        write_csv_rows(path, [n.to_json() for n in self.node_sensitivities])

    def to_json_file(self, path: Union[str, os.PathLike[str]], *, indent: int = 2) -> None:
        write_json(path, self.to_json(), indent=indent)


class FirstOrderSensitivityApproximator:
    """Adjoint sensitivity engine for Euler heat diffusion.

    For a scalar objective ``J = c^T x_T``, this computes derivatives with
    respect to all undirected edge weights using one forward trajectory and one
    backward adjoint trajectory, rather than one full diffusion per edge.

    This is the central non-brute-force component of the module.
    """

    def __init__(
        self,
        graph: Union[GraphBundle, ArrayLike, GraphView],
        *,
        diffusion_config: DiffusionConfig = DiffusionConfig(),
        symmetrize_mode: str = "average",
    ) -> None:
        self.graph = GraphView.from_graph(graph, symmetrize_mode=symmetrize_mode)
        self.config = diffusion_config
        self.config.validate()
        if self.config.model.lower() not in ("euler_heat", "heat", "diffusion"):
            raise SensitivityError("first-order sensitivity currently supports Euler heat diffusion only")

    @property
    def A(self) -> np.ndarray:
        return self.graph.A

    @property
    def n(self) -> int:
        return self.graph.n

    def forward_trajectory(self, x0: VectorLike) -> np.ndarray:
        trace = run_diffusion(self.A, x0, self.config, store_trajectory=True)
        if trace.trajectory is None:
            raise SensitivityError("trajectory was not stored")
        return trace.trajectory

    def adjoint_trajectory(self, objective_vector: VectorLike) -> np.ndarray:
        """Return y_t values where y_T = c and y_t = M.T y_{t+1}."""
        c = _as_float_vector(objective_vector, self.n, name="objective_vector")
        M = diffusion_operator(self.A, self.config.dt)
        y = np.zeros((int(self.config.steps) + 1, self.n), dtype=float)
        y[-1] = c
        for t in range(int(self.config.steps) - 1, -1, -1):
            y[t] = M.T @ y[t + 1]
        return y

    def baseline_objective(self, x0: VectorLike, objective_vector: VectorLike) -> float:
        trace = run_diffusion(self.A, x0, self.config, store_trajectory=False)
        c = _as_float_vector(objective_vector, self.n, name="objective_vector")
        return float(np.dot(c, trace.final))

    def edge_derivatives(
        self,
        x0: VectorLike,
        objective_vector: VectorLike,
        *,
        edges: Optional[Sequence[Edge]] = None,
    ) -> List[EdgeSensitivity]:
        x = self.forward_trajectory(x0)
        y = self.adjoint_trajectory(objective_vector)
        if edges is None:
            edges = edge_list(self.A)
        rows: List[EdgeSensitivity] = []
        dt = float(self.config.dt)
        for i, j in edges:
            i = int(i)
            j = int(j)
            if i == j:
                continue
            # dL/dw = (e_i-e_j)(e_i-e_j)^T.  Since M = I - dt L,
            # y^T dM x = -dt * (y_i-y_j)*(x_i-x_j).
            derivative = 0.0
            for t in range(int(self.config.steps)):
                derivative += -dt * float((y[t + 1, i] - y[t + 1, j]) * (x[t, i] - x[t, j]))
            w = float(self.A[i, j])
            predicted_remove = derivative * (-w)
            rows.append(EdgeSensitivity(
                i=i,
                j=j,
                weight=w,
                derivative=float(derivative),
                absolute_derivative=abs(float(derivative)),
                predicted_delta_for_removal=float(predicted_remove),
                score=abs(float(predicted_remove)),
                label_i=self.graph.label(i),
                label_j=self.graph.label(j),
            ))
        return sorted(rows, key=lambda e: e.score, reverse=True)

    def node_derivatives(self, edge_sensitivities: Sequence[EdgeSensitivity]) -> List[NodeSensitivity]:
        signed = np.zeros(self.n, dtype=float)
        absolute = np.zeros(self.n, dtype=float)
        predicted_remove = np.zeros(self.n, dtype=float)
        max_abs = np.zeros(self.n, dtype=float)
        counts = np.zeros(self.n, dtype=int)
        for e in edge_sensitivities:
            for node in (e.i, e.j):
                signed[node] += e.derivative
                absolute[node] += e.absolute_derivative
                predicted_remove[node] += e.predicted_delta_for_removal
                max_abs[node] = max(max_abs[node], e.absolute_derivative)
                counts[node] += 1
        degrees = degree_vector(self.A)
        rows = [
            NodeSensitivity(
                node=i,
                label=self.graph.label(i),
                degree=float(degrees[i]),
                incident_abs_derivative_sum=float(absolute[i]),
                incident_signed_derivative_sum=float(signed[i]),
                predicted_removal_delta=float(predicted_remove[i]),
                max_incident_abs_derivative=float(max_abs[i]),
                incident_edges=int(counts[i]),
            )
            for i in range(self.n)
        ]
        return sorted(rows, key=lambda r: abs(r.predicted_removal_delta), reverse=True)

    def report(
        self,
        x0: VectorLike,
        objective_vector: VectorLike,
        *,
        objective_description: str = "linear objective c^T x_T",
        edges: Optional[Sequence[Edge]] = None,
    ) -> SensitivityReport:
        c = _as_float_vector(objective_vector, self.n, name="objective_vector")
        edge_rows = self.edge_derivatives(x0, c, edges=edges)
        node_rows = self.node_derivatives(edge_rows)
        return SensitivityReport(
            graph_summary=self.graph.summary(),
            diffusion_config=self.config,
            objective_description=objective_description,
            edge_sensitivities=edge_rows,
            node_sensitivities=node_rows,
            baseline_objective=self.baseline_objective(x0, c),
            meta={
                "method": "discrete adjoint first-order edge-weight sensitivity",
                "edge_count": len(edge_rows),
                "node_count": len(node_rows),
            },
        )

    def global_sensitivity_ranking(
        self,
        x0: VectorLike,
        *,
        norm: str = "l2",
        top_k: Optional[int] = None,
    ) -> List[EdgeSensitivity]:
        """Rank edges by approximate impact on the whole final state.

        Unlike a target-specific objective, this estimates the Jacobian column
        norm for each edge by propagating the perturbation equation.  It is more
        expensive than the scalar adjoint method but still avoids modifying the
        graph and rerunning from scratch for every intervention.
        """
        x = self.forward_trajectory(x0)
        M = diffusion_operator(self.A, self.config.dt)
        edges = edge_list(self.A)
        rows: List[EdgeSensitivity] = []
        for i, j in edges:
            dx = np.zeros(self.n, dtype=float)
            for t in range(int(self.config.steps)):
                forcing = np.zeros(self.n, dtype=float)
                diff = float(x[t, i] - x[t, j])
                # dM x_t for a unit increase in w_ij
                forcing[i] += -float(self.config.dt) * diff
                forcing[j] += float(self.config.dt) * diff
                dx = M @ dx + forcing
            if norm == "l1":
                impact = float(np.sum(np.abs(dx)))
            elif norm == "linf":
                impact = float(np.max(np.abs(dx))) if dx.size else 0.0
            else:
                impact = float(np.linalg.norm(dx))
            w = float(self.A[i, j])
            rows.append(EdgeSensitivity(
                i=i,
                j=j,
                weight=w,
                derivative=float(np.sum(dx)),
                absolute_derivative=impact,
                predicted_delta_for_removal=float(-w * np.sum(dx)),
                score=float(w * impact),
                label_i=self.graph.label(i),
                label_j=self.graph.label(j),
            ))
        rows = sorted(rows, key=lambda e: e.score, reverse=True)
        if top_k is not None:
            rows = rows[: int(top_k)]
        return rows


# =============================================================================
# Hybrid exact/approximate ranking and validation
# =============================================================================


@dataclass
class ApproximationValidationRow:
    intervention_id: str
    intervention_name: str
    predicted_delta: float
    exact_delta: float
    absolute_error: float
    relative_error: float
    rank_predicted: int
    rank_exact: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        return dataclasses.asdict(self)


@dataclass
class ApproximationValidationReport:
    rows: List[ApproximationValidationRow]
    pearson: float
    spearman: float
    mean_absolute_error: float
    median_absolute_error: float
    created_at: str = field(default_factory=_utc_now)

    def to_json(self) -> JsonDict:
        return {
            "created_at": self.created_at,
            "pearson": float(self.pearson),
            "spearman": float(self.spearman),
            "mean_absolute_error": float(self.mean_absolute_error),
            "median_absolute_error": float(self.median_absolute_error),
            "rows": [r.to_json() for r in self.rows],
        }

    def to_csv(self, path: Union[str, os.PathLike[str]]) -> None:
        write_csv_rows(path, [r.to_json() for r in self.rows])


class HybridCounterfactualRanker:
    """Use adjoint rankings to choose candidates, then validate exactly."""

    def __init__(
        self,
        engine: CounterfactualEngine,
        approximator: Optional[FirstOrderSensitivityApproximator] = None,
    ) -> None:
        self.engine = engine
        self.approximator = approximator or FirstOrderSensitivityApproximator(engine.graph, diffusion_config=engine.config)

    def top_edge_removal_candidates(
        self,
        x0: VectorLike,
        objective_vector: VectorLike,
        *,
        top_k: int = 25,
    ) -> Tuple[List[EdgeSensitivity], List[Intervention]]:
        report = self.approximator.report(x0, objective_vector, objective_description="target-specific edge removal ranking")
        top_edges = report.top_edges(top_k, by="score")
        interventions = [EdgeRemovalIntervention(i=e.i, j=e.j) for e in top_edges]
        return top_edges, interventions

    def validate_top_edges(
        self,
        x0: VectorLike,
        objective_vector: VectorLike,
        *,
        top_k: int = 25,
        target: Optional[VectorLike] = None,
    ) -> ApproximationValidationReport:
        top_edges, interventions = self.top_edge_removal_candidates(x0, objective_vector, top_k=top_k)
        exact = self.engine.evaluate_many(interventions, seed_state=x0, target=target if target is not None else objective_vector)
        predicted = np.array([e.predicted_delta_for_removal for e in top_edges], dtype=float)
        exact_delta = np.array([
            ev.comparison.metrics.target_delta
            if ev.comparison.metrics.target_delta is not None
            else ev.comparison.metrics.signed_sum_delta
            for ev in exact
        ], dtype=float)
        pred_rank_order = _stable_argsort_abs_desc(predicted)
        exact_rank_order = _stable_argsort_abs_desc(exact_delta)
        pred_ranks = np.empty(predicted.size, dtype=int)
        exact_ranks = np.empty(predicted.size, dtype=int)
        for rank, idx in enumerate(pred_rank_order, start=1):
            pred_ranks[int(idx)] = rank
        for rank, idx in enumerate(exact_rank_order, start=1):
            exact_ranks[int(idx)] = rank
        rows: List[ApproximationValidationRow] = []
        for k, (sens, ev) in enumerate(zip(top_edges, exact)):
            error = float(abs(exact_delta[k] - predicted[k]))
            rel = float(error / (abs(exact_delta[k]) + 1e-12))
            rows.append(ApproximationValidationRow(
                intervention_id=ev.intervention_id,
                intervention_name=ev.comparison.intervention_name,
                predicted_delta=float(predicted[k]),
                exact_delta=float(exact_delta[k]),
                absolute_error=error,
                relative_error=rel,
                rank_predicted=int(pred_ranks[k]),
                rank_exact=int(exact_ranks[k]),
                meta={"i": sens.i, "j": sens.j, "label_i": sens.label_i, "label_j": sens.label_j},
            ))
        return ApproximationValidationReport(
            rows=rows,
            pearson=pearson_corr(predicted, exact_delta) if predicted.size else 0.0,
            spearman=spearman_corr(predicted, exact_delta) if predicted.size else 0.0,
            mean_absolute_error=float(np.mean(np.abs(predicted - exact_delta))) if predicted.size else 0.0,
            median_absolute_error=float(np.median(np.abs(predicted - exact_delta))) if predicted.size else 0.0,
        )


# =============================================================================
# Path and target-specific analysis helpers
# =============================================================================


def shortest_path_by_hops(A: np.ndarray, source: int, target: int) -> List[int]:
    """Unweighted BFS shortest path for contact graphs."""
    n = A.shape[0]
    source = int(source)
    target = int(target)
    if source == target:
        return [source]
    visited = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=int)
    queue = [source]
    visited[source] = True
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in np.flatnonzero(A[u] > 0):
            v = int(v)
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                if v == target:
                    path = [target]
                    cur = target
                    while parent[cur] != -1:
                        cur = int(parent[cur])
                        path.append(cur)
                    return list(reversed(path))
                queue.append(v)
    return []


def widest_path(A: np.ndarray, source: int, target: int) -> List[int]:
    """Maximum bottleneck path using a Dijkstra-like relaxation."""
    n = A.shape[0]
    source = int(source)
    target = int(target)
    capacity = np.full(n, -np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    used = np.zeros(n, dtype=bool)
    capacity[source] = np.inf
    for _ in range(n):
        candidates = np.where(~used, capacity, -np.inf)
        u = int(np.argmax(candidates))
        if not math.isfinite(candidates[u]) and candidates[u] < 0:
            break
        if u == target:
            break
        used[u] = True
        for v in np.flatnonzero(A[u] > 0):
            v = int(v)
            cap = min(capacity[u], float(A[u, v]))
            if cap > capacity[v]:
                capacity[v] = cap
                parent[v] = u
    if parent[target] == -1 and source != target:
        return []
    path = [target]
    cur = target
    while cur != source:
        cur = int(parent[cur])
        if cur < 0:
            return []
        path.append(cur)
    return list(reversed(path))


def target_specific_sensitivity(
    graph: Union[GraphBundle, ArrayLike, GraphView],
    seed_nodes: Sequence[int],
    target_nodes: Sequence[int],
    *,
    seed_strengths: Optional[Sequence[float]] = None,
    target_weights: Optional[Sequence[float]] = None,
    diffusion_config: DiffusionConfig = DiffusionConfig(),
    top_k_edges: int = 20,
    top_k_nodes: int = 20,
) -> SensitivityReport:
    view = GraphView.from_graph(graph)
    x0 = seed_signal(view.n, seed_nodes, strengths=seed_strengths)
    c = target_vector(view.n, target_nodes, weights=target_weights)
    approximator = FirstOrderSensitivityApproximator(view, diffusion_config=diffusion_config)
    report = approximator.report(
        x0,
        c,
        objective_description=f"signal arriving at targets {list(target_nodes)} from seeds {list(seed_nodes)}",
    )
    report.edge_sensitivities[:] = report.top_edges(max(top_k_edges, len(report.edge_sensitivities)))
    report.node_sensitivities[:] = report.top_nodes(max(top_k_nodes, len(report.node_sensitivities)))
    return report


# =============================================================================
# Reports and export helpers
# =============================================================================


@dataclass
class CounterfactualReport:
    title: str
    graph_summary: JsonDict
    diffusion_config: DiffusionConfig
    evaluations: List[CounterfactualEvaluation] = field(default_factory=list)
    sensitivity_report: Optional[SensitivityReport] = None
    ablation_report: Optional[AblationScanReport] = None
    validation_report: Optional[ApproximationValidationReport] = None
    created_at: str = field(default_factory=_utc_now)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, *, compact: bool = False) -> JsonDict:
        data: JsonDict = {
            "title": self.title,
            "created_at": self.created_at,
            "graph_summary": _jsonify(self.graph_summary),
            "diffusion_config": self.diffusion_config.to_json(),
            "evaluations": [ev.compact_row() if compact else ev.to_json() for ev in self.evaluations],
            "meta": _jsonify(self.meta),
        }
        if self.sensitivity_report is not None:
            data["sensitivity_report"] = self.sensitivity_report.to_json()
        if self.ablation_report is not None:
            data["ablation_report"] = self.ablation_report.to_json(compact=compact)
        if self.validation_report is not None:
            data["validation_report"] = self.validation_report.to_json()
        return data

    def to_json_file(self, path: Union[str, os.PathLike[str]], *, compact: bool = False, indent: int = 2) -> None:
        write_json(path, self.to_json(compact=compact), indent=indent)

    def to_evaluations_csv(self, path: Union[str, os.PathLike[str]]) -> None:
        write_csv_rows(path, [ev.compact_row() for ev in self.evaluations])

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"Created: `{self.created_at}`")
        lines.append("")
        lines.append("## Graph")
        lines.append("")
        for key, value in self.graph_summary.items():
            lines.append(f"- **{key}**: `{value}`")
        lines.append("")
        lines.append("## Diffusion configuration")
        lines.append("")
        for key, value in self.diffusion_config.to_json().items():
            lines.append(f"- **{key}**: `{value}`")
        if self.evaluations:
            lines.append("")
            lines.append("## Exact counterfactual evaluations")
            lines.append("")
            lines.append("| Intervention | L2 Δ | L∞ Δ | Target Δ | Top-1 changed |")
            lines.append("|---|---:|---:|---:|---:|")
            for ev in self.evaluations:
                m = ev.comparison.metrics
                target_delta = "" if m.target_delta is None else f"{m.target_delta:.6g}"
                lines.append(
                    f"| {ev.comparison.intervention_name} | {m.l2_delta:.6g} | {m.linf_delta:.6g} | {target_delta} | {m.top1_changed} |"
                )
        if self.sensitivity_report is not None:
            lines.append("")
            lines.append("## First-order sensitivity")
            lines.append("")
            lines.append(f"Objective: {self.sensitivity_report.objective_description}")
            lines.append("")
            lines.append("### Top sensitive contacts")
            lines.append("")
            lines.append("| Edge | Labels | Weight | dJ/dw | Predicted removal Δ | Score |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for e in self.sensitivity_report.top_edges(15):
                lines.append(
                    f"| ({e.i}, {e.j}) | {e.label_i} - {e.label_j} | {e.weight:.6g} | {e.derivative:.6g} | {e.predicted_delta_for_removal:.6g} | {e.score:.6g} |"
                )
            lines.append("")
            lines.append("### Top sensitive residues")
            lines.append("")
            lines.append("| Node | Label | Degree | Incident sensitivity | Predicted removal Δ |")
            lines.append("|---:|---|---:|---:|---:|")
            for n in self.sensitivity_report.top_nodes(15):
                lines.append(
                    f"| {n.node} | {n.label} | {n.degree:.6g} | {n.incident_abs_derivative_sum:.6g} | {n.predicted_removal_delta:.6g} |"
                )
        if self.validation_report is not None:
            lines.append("")
            lines.append("## Approximation validation")
            lines.append("")
            lines.append(f"- Pearson: `{self.validation_report.pearson:.6g}`")
            lines.append(f"- Spearman: `{self.validation_report.spearman:.6g}`")
            lines.append(f"- Mean absolute error: `{self.validation_report.mean_absolute_error:.6g}`")
        return "\n".join(lines) + "\n"

    def to_markdown_file(self, path: Union[str, os.PathLike[str]]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")


def write_json(path: Union[str, os.PathLike[str]], obj: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(obj), f, indent=indent, sort_keys=True)
        f.write("\n")


def write_csv_rows(path: Union[str, os.PathLike[str]], rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonify(row.get(key, "")) for key in keys})


def export_state_delta_csv(
    path: Union[str, os.PathLike[str]],
    comparison: DiffusionComparison,
    labels: Optional[Sequence[str]] = None,
) -> None:
    rows = []
    n = comparison.before.size
    labels = list(labels) if labels is not None else [str(i) for i in range(n)]
    for i in range(n):
        rows.append({
            "node": i,
            "label": labels[i] if i < len(labels) else str(i),
            "before": float(comparison.before[i]),
            "after": float(comparison.after[i]),
            "delta": float(comparison.delta[i]),
            "abs_delta": float(abs(comparison.delta[i])),
        })
    write_csv_rows(path, rows)


# =============================================================================
# Convenience constructors and high-level workflows
# =============================================================================


def make_intervention(kind: str, **kwargs: Any) -> Intervention:
    """Factory used by CLIs, DSLs, and config files."""
    key = kind.lower().strip().replace("-", "_")
    if key in ("node_removal", "remove_node", "node"):
        return NodeRemovalIntervention(**kwargs)
    if key in ("edge_removal", "remove_edge", "edge"):
        return EdgeRemovalIntervention(**kwargs)
    if key in ("edge_reweight", "reweight_edge", "weaken_edge", "strengthen_edge"):
        return EdgeReweightIntervention(**kwargs)
    if key in ("local_mutation", "mutate_node", "mutation"):
        return LocalMutationIntervention(**kwargs)
    if key in ("path_block", "block_path", "path_blocking"):
        return PathBlockingIntervention(**kwargs)
    if key in ("threshold", "threshold_contacts"):
        return ContactThresholdIntervention(**kwargs)
    if key in ("composite", "multi"):
        children = kwargs.pop("interventions", [])
        if children and isinstance(children[0], Mapping):
            children = [make_intervention(**child) for child in children]
        return CompositeIntervention(interventions=children, **kwargs)
    raise InterventionError(f"unknown intervention kind {kind!r}")


def interventions_from_config(config: Sequence[Mapping[str, Any]]) -> List[Intervention]:
    interventions: List[Intervention] = []
    for spec in config:
        data = dict(spec)
        kind = str(data.pop("kind", data.pop("type", "")))
        if not kind:
            raise InterventionError(f"intervention config missing kind/type: {spec}")
        interventions.append(make_intervention(kind, **data))
    return interventions


def run_counterfactual_suite(
    graph: Union[GraphBundle, ArrayLike, GraphView],
    seed_state: VectorLike,
    *,
    target: Optional[VectorLike] = None,
    interventions: Optional[Sequence[Intervention]] = None,
    diffusion_config: DiffusionConfig = DiffusionConfig(),
    run_ablation_scan: bool = False,
    ablation_config: AblationScanConfig = AblationScanConfig(),
    run_sensitivity: bool = True,
    objective_vector: Optional[VectorLike] = None,
    validate_top_k: int = 0,
    title: str = "RINet counterfactual inference report",
) -> CounterfactualReport:
    engine = CounterfactualEngine(graph, diffusion_config=diffusion_config, seed_state=seed_state, target=target)
    evaluations: List[CounterfactualEvaluation] = []
    if interventions:
        evaluations = engine.evaluate_many(interventions, seed_state=seed_state, target=target)
    sensitivity_report = None
    validation_report = None
    if run_sensitivity:
        c = objective_vector if objective_vector is not None else target
        if c is None:
            c = np.ones(engine.n, dtype=float)
        approximator = FirstOrderSensitivityApproximator(engine.graph, diffusion_config=diffusion_config)
        sensitivity_report = approximator.report(seed_state, c, objective_description="counterfactual suite objective")
        if validate_top_k and validate_top_k > 0:
            ranker = HybridCounterfactualRanker(engine, approximator)
            validation_report = ranker.validate_top_edges(seed_state, c, top_k=int(validate_top_k), target=target if target is not None else c)
    ablation_report = None
    if run_ablation_scan:
        scanner = ExactAblationScanner(engine, ablation_config)
        ablation_report = scanner.run(seed_state=seed_state, target=target)
    return CounterfactualReport(
        title=title,
        graph_summary=engine.graph.summary(),
        diffusion_config=diffusion_config,
        evaluations=evaluations,
        sensitivity_report=sensitivity_report,
        ablation_report=ablation_report,
        validation_report=validation_report,
        meta={"engine": "counterfactual_inference_engine"},
    )


def run_counterfactual_report_from_csv(
    adjacency_csv: Union[str, os.PathLike[str]],
    output_dir: Union[str, os.PathLike[str]],
    *,
    seed_nodes: Sequence[int],
    target_nodes: Optional[Sequence[int]] = None,
    steps: int = 50,
    dt: float = 0.10,
    interventions: Optional[Sequence[Intervention]] = None,
    run_ablation_scan: bool = True,
    run_sensitivity: bool = True,
) -> CounterfactualReport:
    if adjacency_from_csv is None:
        A = np.loadtxt(adjacency_csv, delimiter=",")
    else:
        A = adjacency_from_csv(str(adjacency_csv))
    view = GraphView.from_graph(A, name=Path(adjacency_csv).stem)
    x0 = seed_signal(view.n, seed_nodes)
    target = target_vector(view.n, target_nodes) if target_nodes is not None else None
    config = DiffusionConfig(steps=int(steps), dt=float(dt))
    report = run_counterfactual_suite(
        view,
        x0,
        target=target,
        interventions=interventions,
        diffusion_config=config,
        run_ablation_scan=run_ablation_scan,
        run_sensitivity=run_sensitivity,
        validate_top_k=10 if run_sensitivity else 0,
        title=f"Counterfactual analysis for {view.name}",
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.to_json_file(out / "counterfactual_report.json", compact=False)
    report.to_json_file(out / "counterfactual_report_compact.json", compact=True)
    report.to_markdown_file(out / "counterfactual_report.md")
    if report.evaluations:
        report.to_evaluations_csv(out / "counterfactual_evaluations.csv")
    if report.sensitivity_report is not None:
        report.sensitivity_report.to_edge_csv(out / "edge_sensitivity.csv")
        report.sensitivity_report.to_node_csv(out / "node_sensitivity.csv")
    if report.ablation_report is not None:
        report.ablation_report.to_csv(out / "ablation_ranked.csv")
    return report


# =============================================================================
# Minimal demo / smoke-test helper
# =============================================================================


def _demo_graph(n: int = 24) -> GraphView:
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for d in (1, 2):
            j = (i + d) % n
            A[i, j] = 1.0 / d
            A[j, i] = A[i, j]
    # two communities with a few long-range contacts
    for i, j, w in [(2, 13, 0.25), (5, 17, 0.3), (8, 20, 0.2), (11, 23, 0.4)]:
        A[i, j] = w
        A[j, i] = w
    labels = [f"R{i+1}" for i in range(n)]
    return GraphView(A=A, labels=labels, name="counterfactual_demo", meta={"source": "demo"})


def demo_counterfactual_report(output_dir: Optional[Union[str, os.PathLike[str]]] = None) -> CounterfactualReport:
    graph = _demo_graph()
    x0 = seed_signal(graph.n, [0], strengths=[1.0])
    target = target_vector(graph.n, [12, 13, 14])
    interventions: List[Intervention] = [
        NodeRemovalIntervention(node=5),
        EdgeRemovalIntervention(i=2, j=13),
        EdgeReweightIntervention(i=5, j=17, factor=0.1),
        LocalMutationIntervention(node=10, factor=0.4, radius=2),
        PathBlockingIntervention(path=shortest_path_by_hops(graph.A, 0, 13), factor=0.0),
    ]
    report = run_counterfactual_suite(
        graph,
        x0,
        target=target,
        interventions=interventions,
        diffusion_config=DiffusionConfig(steps=60, dt=0.03),
        run_ablation_scan=True,
        ablation_config=AblationScanConfig(scan_nodes=True, scan_edges=True, max_edges=20, top_k=15),
        run_sensitivity=True,
        objective_vector=target,
        validate_top_k=8,
        title="Demo RINet counterfactual report",
    )
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report.to_json_file(out / "counterfactual_report.json")
        report.to_json_file(out / "counterfactual_report_compact.json", compact=True)
        report.to_markdown_file(out / "counterfactual_report.md")
        if report.sensitivity_report is not None:
            report.sensitivity_report.to_edge_csv(out / "edge_sensitivity.csv")
            report.sensitivity_report.to_node_csv(out / "node_sensitivity.csv")
        if report.ablation_report is not None:
            report.ablation_report.to_csv(out / "ablation_ranked.csv")
    return report


__all__ = [
    "AblationScanConfig",
    "AblationScanReport",
    "ApproximationValidationReport",
    "ApproximationValidationRow",
    "ComparisonMetrics",
    "CompositeIntervention",
    "ContactThresholdIntervention",
    "CounterfactualEngine",
    "CounterfactualEvaluation",
    "CounterfactualReport",
    "DiffusionComparison",
    "DiffusionConfig",
    "DiffusionTrace",
    "EdgeRemovalIntervention",
    "EdgeReweightIntervention",
    "EdgeSensitivity",
    "ExactAblationScanner",
    "FirstOrderSensitivityApproximator",
    "GraphView",
    "HybridCounterfactualRanker",
    "Intervention",
    "LocalMutationIntervention",
    "NodeRemovalIntervention",
    "NodeSensitivity",
    "PathBlockingIntervention",
    "SensitivityReport",
    "compare_states",
    "demo_counterfactual_report",
    "diffusion_operator",
    "edge_list",
    "graph_hash",
    "interventions_from_config",
    "make_intervention",
    "normalize_signal",
    "run_counterfactual_report_from_csv",
    "run_counterfactual_suite",
    "run_diffusion",
    "seed_signal",
    "shortest_path_by_hops",
    "target_specific_sensitivity",
    "target_vector",
    "widest_path",
]


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import argparse

    parser = argparse.ArgumentParser(description="RINet counterfactual inference demo")
    parser.add_argument("--out", type=str, default="counterfactual_demo_out", help="Output directory")
    args = parser.parse_args()
    demo = demo_counterfactual_report(args.out)
    print(demo.to_markdown())
