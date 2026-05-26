
"""
Graph signal processing tools for RINet / AlloGraph.

This module treats residue-level quantities as signals on a residue interaction
network.  The emphasis is on reusable scientific-computing machinery rather
than one-off formulas: spectral transforms, polynomial filters, Krylov heat
approximations, graph differential operators, wavelets, denoising, anomaly
scoring, and structured reports.

The implementation intentionally depends only on NumPy and the existing
AlloGraph package.  When ``spectral_geometry_engine`` is available, selected
basis construction routines can reuse it, but this module also contains its own
validated fallbacks so it does not increase project dependencies.

Typical use
-----------

>>> from allograph.core.graphio.synth import synthetic_rin
>>> from allograph.core.graph_signal_processing import GraphSignal, GraphSignalProcessor
>>> bundle = synthetic_rin(n=50, seed=3)
>>> signal = GraphSignal.from_seeds(bundle, seeds=[0, 4], strength=1.0)
>>> gsp = GraphSignalProcessor(bundle)
>>> smoothed = gsp.heat_smooth(signal, time=2.0)
>>> report = gsp.analyze_multiscale(signal)

Design notes
------------

The core object is ``GraphSignalProcessor``.  It caches a graph Laplacian and,
when requested, a Fourier basis.  Exact spectral methods are useful for small to
medium residue graphs; polynomial and Krylov methods are included for larger
networks where dense eigendecomposition is too expensive.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .graphio.types import GraphBundle
from .math.laplacian import graph_laplacian

try:  # optional internal acceleration / reuse; never required
    from .spectral_geometry_engine import SpectralGeometryEngine  # type: ignore
except Exception:  # pragma: no cover - optional import
    SpectralGeometryEngine = None  # type: ignore

ArrayLike = Union[Sequence[float], np.ndarray]
TransferFunction = Callable[[np.ndarray], np.ndarray]

_EPS = 1e-12


class GSPError(RuntimeError):
    """Base class for graph-signal-processing errors."""


class SignalShapeError(GSPError):
    """Raised when a signal length does not match the graph."""


class GraphValidationError(GSPError):
    """Raised when an adjacency matrix cannot be interpreted as a graph."""


class BasisError(GSPError):
    """Raised when a Fourier basis cannot be built or used safely."""


class FilteringError(GSPError):
    """Raised when a graph filter specification is invalid."""


class DenoisingError(GSPError):
    """Raised when a denoising method fails to converge or is misconfigured."""


def _as_float_matrix(A: np.ndarray, *, copy: bool = True) -> np.ndarray:
    arr = np.asarray(A, dtype=float)
    if copy:
        arr = arr.copy()
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise GraphValidationError(f"adjacency must be square; got shape {arr.shape!r}")
    if not np.isfinite(arr).all():
        raise GraphValidationError("adjacency contains NaN or infinite values")
    return arr


def _clean_adjacency(A: np.ndarray, *, symmetrize: bool = True, remove_self_loops: bool = True) -> np.ndarray:
    B = _as_float_matrix(A, copy=True)
    if remove_self_loops:
        np.fill_diagonal(B, 0.0)
    B[B < 0] = 0.0
    if symmetrize:
        B = 0.5 * (B + B.T)
    return B


def _validate_signal(x: ArrayLike, n: int, *, name: str = "signal") -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape[0] != n:
        raise SignalShapeError(f"{name} has length {arr.shape[0]}, expected {n}")
    if not np.isfinite(arr).all():
        raise SignalShapeError(f"{name} contains NaN or infinite values")
    return arr


def _degree(A: np.ndarray) -> np.ndarray:
    return np.asarray(A.sum(axis=1), dtype=float)


def _safe_inverse_sqrt_degree(deg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(deg, dtype=float)
    mask = deg > _EPS
    out[mask] = 1.0 / np.sqrt(deg[mask])
    return out


def _safe_inverse_degree(deg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(deg, dtype=float)
    mask = deg > _EPS
    out[mask] = 1.0 / deg[mask]
    return out


def _normalized_laplacian(A: np.ndarray) -> np.ndarray:
    deg = _degree(A)
    inv_sqrt = _safe_inverse_sqrt_degree(deg)
    return np.eye(A.shape[0]) - (inv_sqrt[:, None] * A * inv_sqrt[None, :])


def _random_walk_laplacian(A: np.ndarray) -> np.ndarray:
    deg = _degree(A)
    inv = _safe_inverse_degree(deg)
    return np.eye(A.shape[0]) - inv[:, None] * A


def _laplacian(A: np.ndarray, kind: str = "combinatorial") -> np.ndarray:
    kind = str(kind).lower().replace("-", "_")
    if kind in {"combinatorial", "unnormalized", "plain"}:
        return graph_laplacian(A)
    if kind in {"normalized", "symmetric_normalized", "sym"}:
        return _normalized_laplacian(A)
    if kind in {"random_walk", "rw"}:
        return _random_walk_laplacian(A)
    raise GraphValidationError(f"unknown Laplacian kind: {kind!r}")


def _edge_list(A: np.ndarray, *, include_weights: bool = True) -> List[Tuple[int, int, float]]:
    rows, cols = np.where(np.triu(A, k=1) > _EPS)
    edges: List[Tuple[int, int, float]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        w = float(A[i, j]) if include_weights else 1.0
        edges.append((int(i), int(j), w))
    return edges


def _stable_eigh(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    S = 0.5 * (np.asarray(L, dtype=float) + np.asarray(L, dtype=float).T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.real(vals)
    vecs = np.real(vecs)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    vals[np.abs(vals) < 1e-11] = 0.0
    for k in range(vecs.shape[1]):
        pivot = int(np.argmax(np.abs(vecs[:, k])))
        if vecs[pivot, k] < 0:
            vecs[:, k] *= -1.0
    return vals, vecs


def _matrix_power_solve(M: np.ndarray, b: np.ndarray, *, ridge: float = 1e-10) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M + ridge * np.eye(M.shape[0])) @ b


def _safe_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size or a.size == 0:
        return float("nan")
    aa = a - float(a.mean())
    bb = b - float(b.mean())
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom <= _EPS:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, implemented without SciPy."""
    values = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg
        i = j
    return ranks


def _soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - float(tau), 0.0)


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _local_mean(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    deg = _degree(A)
    out = np.zeros_like(x, dtype=float)
    mask = deg > _EPS
    out[mask] = (A @ x)[mask] / deg[mask]
    out[~mask] = x[~mask]
    return out


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return _json_ready(asdict(obj))
    return obj


@dataclass(frozen=True)
class NodeScore:
    """A named score attached to one graph node / residue."""

    node: int
    score: float
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": int(self.node),
            "score": float(self.score),
            "label": self.label,
            "metadata": _json_ready(self.metadata),
        }


@dataclass(frozen=True)
class EdgeScore:
    """A named score attached to one undirected graph edge."""

    i: int
    j: int
    score: float
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "i": int(self.i),
            "j": int(self.j),
            "score": float(self.score),
            "weight": float(self.weight),
            "metadata": _json_ready(self.metadata),
        }


@dataclass
class FourierBasis:
    """Eigenbasis for a graph Laplacian."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    laplacian_kind: str
    graph_name: str = "GraphBundle"
    built_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.eigenvalues = np.asarray(self.eigenvalues, dtype=float).reshape(-1)
        self.eigenvectors = np.asarray(self.eigenvectors, dtype=float)
        if self.eigenvectors.ndim != 2:
            raise BasisError("eigenvectors must be a 2D array")
        if self.eigenvectors.shape[1] != self.eigenvalues.shape[0]:
            raise BasisError("basis dimension mismatch")

    @property
    def n(self) -> int:
        return int(self.eigenvectors.shape[0])

    @property
    def modes(self) -> int:
        return int(self.eigenvectors.shape[1])

    @property
    def lmax(self) -> float:
        if self.eigenvalues.size == 0:
            return 0.0
        return float(np.max(self.eigenvalues))

    def truncate(self, modes: int) -> "FourierBasis":
        m = max(1, min(int(modes), self.modes))
        return FourierBasis(
            eigenvalues=self.eigenvalues[:m].copy(),
            eigenvectors=self.eigenvectors[:, :m].copy(),
            laplacian_kind=self.laplacian_kind,
            graph_name=self.graph_name,
            built_at=self.built_at,
        )

    def coefficients(self, x: ArrayLike) -> np.ndarray:
        signal = _validate_signal(x, self.n)
        return self.eigenvectors.T @ signal

    def synthesize(self, coeffs: ArrayLike) -> np.ndarray:
        c = np.asarray(coeffs, dtype=float).reshape(-1)
        if c.shape[0] != self.modes:
            raise SignalShapeError(f"coefficient vector has length {c.shape[0]}, expected {self.modes}")
        return self.eigenvectors @ c

    def energy_density(self, x: ArrayLike) -> np.ndarray:
        c = self.coefficients(x)
        return c * c

    def cumulative_energy(self, x: ArrayLike) -> np.ndarray:
        density = self.energy_density(x)
        total = float(np.sum(density))
        if total <= _EPS:
            return np.zeros_like(density)
        return np.cumsum(density) / total

    def spectral_entropy(self, x: ArrayLike, *, normalized: bool = True) -> float:
        density = self.energy_density(x)
        total = float(np.sum(density))
        if total <= _EPS:
            return 0.0
        p = density / total
        p = p[p > _EPS]
        ent = -float(np.sum(p * np.log(p)))
        if normalized and self.modes > 1:
            ent /= math.log(self.modes)
        return ent

    def to_dict(self, *, include_vectors: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "laplacian_kind": self.laplacian_kind,
            "graph_name": self.graph_name,
            "n": self.n,
            "modes": self.modes,
            "lmax": self.lmax,
            "eigenvalues": self.eigenvalues.tolist(),
            "built_at": self.built_at,
        }
        if include_vectors:
            out["eigenvectors"] = self.eigenvectors.tolist()
        return out


@dataclass
class GraphSignal:
    """Residue-level scalar field on a graph.

    ``values[i]`` is the quantity attached to residue/node ``i``.  Metadata is
    carried through transformations so downstream exports can still describe
    how a signal was generated.
    """

    values: np.ndarray
    bundle: Optional[GraphBundle] = None
    name: str = "signal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float).reshape(-1)
        if not np.isfinite(self.values).all():
            raise SignalShapeError("GraphSignal values contain NaN or infinite values")
        if self.bundle is not None:
            _validate_signal(self.values, self.bundle.n, name=self.name)

    @classmethod
    def zeros(cls, bundle: GraphBundle, *, name: str = "zero_signal") -> "GraphSignal":
        return cls(np.zeros(bundle.n, dtype=float), bundle=bundle, name=name)

    @classmethod
    def from_seeds(
        cls,
        bundle: GraphBundle,
        seeds: Sequence[int],
        *,
        strength: float = 1.0,
        strengths: Optional[Sequence[float]] = None,
        name: str = "seed_signal",
    ) -> "GraphSignal":
        x = np.zeros(bundle.n, dtype=float)
        if strengths is None:
            strengths = [float(strength)] * len(seeds)
        if len(strengths) != len(seeds):
            raise SignalShapeError("strengths must be the same length as seeds")
        for node, val in zip(seeds, strengths):
            idx = int(node)
            if idx < 0 or idx >= bundle.n:
                raise SignalShapeError(f"seed node {idx} out of range for graph with {bundle.n} nodes")
            x[idx] += float(val)
        return cls(x, bundle=bundle, name=name, metadata={"seeds": list(map(int, seeds)), "strengths": list(map(float, strengths))})

    @classmethod
    def from_mapping(
        cls,
        bundle: GraphBundle,
        mapping: Mapping[int, float],
        *,
        default: float = 0.0,
        name: str = "mapped_signal",
    ) -> "GraphSignal":
        x = np.full(bundle.n, float(default), dtype=float)
        for node, value in mapping.items():
            idx = int(node)
            if idx < 0 or idx >= bundle.n:
                raise SignalShapeError(f"node {idx} out of range for graph with {bundle.n} nodes")
            x[idx] = float(value)
        return cls(x, bundle=bundle, name=name, metadata={"source": "mapping"})

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        *,
        bundle: Optional[GraphBundle] = None,
        node_column: str = "node",
        value_column: str = "value",
        name: Optional[str] = None,
    ) -> "GraphSignal":
        p = Path(path)
        if bundle is None:
            rows: List[Tuple[int, float]] = []
            with p.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append((int(row[node_column]), float(row[value_column])))
            if not rows:
                raise SignalShapeError(f"no signal rows found in {p}")
            n = max(i for i, _ in rows) + 1
            x = np.zeros(n, dtype=float)
            for i, v in rows:
                x[i] = v
            return cls(x, bundle=None, name=name or p.stem, metadata={"path": str(p)})
        x = np.zeros(bundle.n, dtype=float)
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row[node_column])
                if 0 <= idx < bundle.n:
                    x[idx] = float(row[value_column])
        return cls(x, bundle=bundle, name=name or p.stem, metadata={"path": str(p)})

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    def copy(self, *, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "GraphSignal":
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return GraphSignal(self.values.copy(), bundle=self.bundle, name=name or self.name, metadata=merged)

    def with_values(self, values: ArrayLike, *, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "GraphSignal":
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return GraphSignal(np.asarray(values, dtype=float), bundle=self.bundle, name=name or self.name, metadata=merged)

    def centered(self, *, weighted_by_degree: bool = False) -> "GraphSignal":
        if weighted_by_degree and self.bundle is not None:
            deg = _degree(_clean_adjacency(self.bundle.A))
            denom = float(np.sum(deg))
            mu = float(np.dot(deg, self.values) / denom) if denom > _EPS else float(np.mean(self.values))
        else:
            mu = float(np.mean(self.values))
        return self.with_values(self.values - mu, name=f"{self.name}_centered", metadata={"center": mu})

    def normalized(self, *, p: float = 2.0) -> "GraphSignal":
        if p == 1:
            norm = float(np.sum(np.abs(self.values)))
        elif p == 2:
            norm = float(np.linalg.norm(self.values))
        elif math.isinf(p):
            norm = float(np.max(np.abs(self.values)))
        else:
            norm = float(np.sum(np.abs(self.values) ** p) ** (1.0 / p))
        if norm <= _EPS:
            return self.copy(name=f"{self.name}_normalized", metadata={"norm": norm})
        return self.with_values(self.values / norm, name=f"{self.name}_normalized", metadata={"norm": norm, "p": p})

    def zscored(self) -> "GraphSignal":
        mu = float(np.mean(self.values))
        sigma = float(np.std(self.values))
        if sigma <= _EPS:
            z = np.zeros_like(self.values)
        else:
            z = (self.values - mu) / sigma
        return self.with_values(z, name=f"{self.name}_zscore", metadata={"mean": mu, "std": sigma})

    def robust_zscored(self) -> "GraphSignal":
        med = float(np.median(self.values))
        mad = _mad(self.values)
        scale = 1.4826 * mad
        if scale <= _EPS:
            z = np.zeros_like(self.values)
        else:
            z = (self.values - med) / scale
        return self.with_values(z, name=f"{self.name}_robust_zscore", metadata={"median": med, "mad": mad})

    def clipped(self, low: Optional[float] = None, high: Optional[float] = None) -> "GraphSignal":
        lo = -np.inf if low is None else float(low)
        hi = np.inf if high is None else float(high)
        return self.with_values(np.clip(self.values, lo, hi), name=f"{self.name}_clipped", metadata={"low": low, "high": high})

    def thresholded(self, threshold: float, *, mode: str = "absolute") -> "GraphSignal":
        t = float(threshold)
        if mode == "absolute":
            y = np.where(np.abs(self.values) >= t, self.values, 0.0)
        elif mode == "positive":
            y = np.where(self.values >= t, self.values, 0.0)
        elif mode == "negative":
            y = np.where(self.values <= -t, self.values, 0.0)
        else:
            raise SignalShapeError(f"unknown threshold mode: {mode!r}")
        return self.with_values(y, name=f"{self.name}_thresholded", metadata={"threshold": t, "mode": mode})

    def top_nodes(self, k: int = 10, *, absolute: bool = True, labels: Optional[Sequence[str]] = None) -> List[NodeScore]:
        scores = np.abs(self.values) if absolute else self.values
        order = np.argsort(-scores)[: max(0, int(k))]
        out: List[NodeScore] = []
        for idx in order.tolist():
            label = labels[idx] if labels is not None and idx < len(labels) else None
            out.append(NodeScore(node=int(idx), score=float(self.values[idx]), label=label, metadata={"rank_score": float(scores[idx])}))
        return out

    def similarity(self, other: "GraphSignal", *, mode: str = "cosine") -> float:
        y = _validate_signal(other.values, self.n, name=other.name)
        x = self.values
        mode = mode.lower()
        if mode == "cosine":
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            if denom <= _EPS:
                return 0.0
            return float(np.dot(x, y) / denom)
        if mode == "pearson":
            return _safe_correlation(x, y)
        if mode == "spearman":
            return _safe_correlation(_rankdata(x), _rankdata(y))
        if mode == "l2_negative":
            return -float(np.linalg.norm(x - y))
        raise SignalShapeError(f"unknown similarity mode: {mode!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "values": self.values.tolist(),
            "bundle_name": self.bundle.name if self.bundle is not None else None,
            "metadata": _json_ready(self.metadata),
        }

    def to_csv(self, path: Union[str, Path], *, labels: Optional[Sequence[str]] = None) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            fieldnames = ["node", "value"]
            if labels is not None:
                fieldnames.append("label")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, v in enumerate(self.values.tolist()):
                row: Dict[str, Any] = {"node": i, "value": float(v)}
                if labels is not None:
                    row["label"] = labels[i] if i < len(labels) else ""
                writer.writerow(row)
        return p

    def to_json(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


@dataclass(frozen=True)
class FilterSpec:
    """Serializable description of a graph spectral filter."""

    name: str
    kind: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def transfer(self) -> TransferFunction:
        kind = self.kind.lower().replace("-", "_")
        params = dict(self.parameters)
        if kind == "low_pass":
            cutoff = float(params.get("cutoff", 1.0))
            rolloff = float(params.get("rolloff", 0.0))
            return lambda lam: smooth_low_pass(lam, cutoff=cutoff, rolloff=rolloff)
        if kind == "high_pass":
            cutoff = float(params.get("cutoff", 1.0))
            rolloff = float(params.get("rolloff", 0.0))
            return lambda lam: smooth_high_pass(lam, cutoff=cutoff, rolloff=rolloff)
        if kind == "band_pass":
            low = float(params.get("low", 0.0))
            high = float(params.get("high", 1.0))
            rolloff = float(params.get("rolloff", 0.0))
            return lambda lam: smooth_band_pass(lam, low=low, high=high, rolloff=rolloff)
        if kind == "heat":
            t = float(params.get("time", 1.0))
            return lambda lam: np.exp(-t * lam)
        if kind == "tikhonov":
            alpha = float(params.get("alpha", 1.0))
            return lambda lam: 1.0 / (1.0 + alpha * lam)
        if kind == "power":
            exponent = float(params.get("exponent", 1.0))
            scale = float(params.get("scale", 1.0))
            return lambda lam: 1.0 / (1.0 + scale * np.power(np.maximum(lam, 0.0), exponent))
        raise FilteringError(f"unknown filter kind: {self.kind!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "parameters": _json_ready(self.parameters)}


def smooth_low_pass(lam: np.ndarray, *, cutoff: float, rolloff: float = 0.0) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    c = float(cutoff)
    r = float(rolloff)
    if r <= _EPS:
        return (lam <= c).astype(float)
    return 1.0 / (1.0 + np.exp((lam - c) / r))


def smooth_high_pass(lam: np.ndarray, *, cutoff: float, rolloff: float = 0.0) -> np.ndarray:
    return 1.0 - smooth_low_pass(lam, cutoff=cutoff, rolloff=rolloff)


def smooth_band_pass(lam: np.ndarray, *, low: float, high: float, rolloff: float = 0.0) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    if high < low:
        raise FilteringError("band-pass high cutoff must be >= low cutoff")
    return smooth_high_pass(lam, cutoff=low, rolloff=rolloff) * smooth_low_pass(lam, cutoff=high, rolloff=rolloff)


@dataclass
class FilterResult:
    original: GraphSignal
    filtered: GraphSignal
    spec: FilterSpec
    spectral_energy_before: Optional[np.ndarray] = None
    spectral_energy_after: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": {"name": self.original.name, "n": self.original.n},
            "filtered": self.filtered.to_dict(),
            "spec": self.spec.to_dict(),
            "spectral_energy_before": None if self.spectral_energy_before is None else self.spectral_energy_before.tolist(),
            "spectral_energy_after": None if self.spectral_energy_after is None else self.spectral_energy_after.tolist(),
            "metadata": _json_ready(self.metadata),
        }


@dataclass
class EnergyReport:
    signal_name: str
    l2_norm: float
    l1_norm: float
    linf_norm: float
    mean: float
    std: float
    total_variation: float
    dirichlet_energy: float
    laplacian_kind: str
    spectral_entropy: Optional[float] = None
    spectral_centroid: Optional[float] = None
    high_frequency_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass
class WaveletScaleReport:
    scale: float
    coefficient_norm: float
    coefficient_l1: float
    coefficient_max_abs: float
    localization_entropy: float
    top_nodes: List[NodeScore]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": float(self.scale),
            "coefficient_norm": float(self.coefficient_norm),
            "coefficient_l1": float(self.coefficient_l1),
            "coefficient_max_abs": float(self.coefficient_max_abs),
            "localization_entropy": float(self.localization_entropy),
            "top_nodes": [x.to_dict() for x in self.top_nodes],
        }


@dataclass
class MultiscaleReport:
    signal_name: str
    scales: List[WaveletScaleReport]
    cross_scale_correlations: List[List[float]]
    dominant_scale_by_node: List[int]
    scale_energy_distribution: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "scales": [s.to_dict() for s in self.scales],
            "cross_scale_correlations": self.cross_scale_correlations,
            "dominant_scale_by_node": self.dominant_scale_by_node,
            "scale_energy_distribution": self.scale_energy_distribution,
            "metadata": _json_ready(self.metadata),
        }

    def to_json(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    def to_markdown(self, path: Union[str, Path], *, top_k: int = 8) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append(f"# Multiscale graph-signal report: `{self.signal_name}`")
        lines.append("")
        lines.append("| scale | L2 norm | L1 norm | max | localization entropy |")
        lines.append("|---:|---:|---:|---:|---:|")
        for s in self.scales:
            lines.append(f"| {s.scale:.6g} | {s.coefficient_norm:.6g} | {s.coefficient_l1:.6g} | {s.coefficient_max_abs:.6g} | {s.localization_entropy:.6g} |")
        lines.append("")
        lines.append("## Top residues by scale")
        for s in self.scales:
            lines.append("")
            lines.append(f"### Scale {s.scale:.6g}")
            for ns in s.top_nodes[:top_k]:
                label = f" ({ns.label})" if ns.label else ""
                lines.append(f"- node {ns.node}{label}: {ns.score:.6g}")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p


@dataclass
class AnomalyReport:
    signal_name: str
    method: str
    scores: List[NodeScore]
    threshold: float
    flagged_nodes: List[int]
    residual_signal: Optional[GraphSignal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def top(self, k: int = 10) -> List[NodeScore]:
        return sorted(self.scores, key=lambda x: abs(x.score), reverse=True)[: int(k)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "method": self.method,
            "scores": [s.to_dict() for s in self.scores],
            "threshold": float(self.threshold),
            "flagged_nodes": [int(i) for i in self.flagged_nodes],
            "residual_signal": None if self.residual_signal is None else self.residual_signal.to_dict(),
            "metadata": _json_ready(self.metadata),
        }

    def to_csv(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["node", "score", "flagged", "label"])
            writer.writeheader()
            flagged = set(self.flagged_nodes)
            for score in self.scores:
                writer.writerow({"node": score.node, "score": score.score, "flagged": score.node in flagged, "label": score.label or ""})
        return p

    def to_json(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


@dataclass
class DenoisingReport:
    original: GraphSignal
    denoised: GraphSignal
    residual: GraphSignal
    method: str
    objective_trace: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.residual.values))

    @property
    def signal_norm(self) -> float:
        return float(np.linalg.norm(self.denoised.values))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": {"name": self.original.name, "n": self.original.n},
            "denoised": self.denoised.to_dict(),
            "residual": self.residual.to_dict(),
            "method": self.method,
            "objective_trace": [float(x) for x in self.objective_trace],
            "residual_norm": self.residual_norm,
            "signal_norm": self.signal_norm,
            "metadata": _json_ready(self.metadata),
        }

    def to_json(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


class GraphDifferentialOperators:
    """Discrete calculus on an undirected weighted residue graph."""

    def __init__(self, A: np.ndarray):
        self.A = _clean_adjacency(A)
        self.n = self.A.shape[0]
        self.edges = _edge_list(self.A)
        self._incidence: Optional[np.ndarray] = None
        self._sqrt_weights: Optional[np.ndarray] = None

    @property
    def incidence(self) -> np.ndarray:
        if self._incidence is None:
            B = np.zeros((len(self.edges), self.n), dtype=float)
            for row, (i, j, _w) in enumerate(self.edges):
                B[row, i] = -1.0
                B[row, j] = 1.0
            self._incidence = B
        return self._incidence

    @property
    def sqrt_weights(self) -> np.ndarray:
        if self._sqrt_weights is None:
            self._sqrt_weights = np.sqrt(np.asarray([w for _i, _j, w in self.edges], dtype=float))
        return self._sqrt_weights

    def gradient(self, signal: ArrayLike, *, weighted: bool = True) -> np.ndarray:
        x = _validate_signal(signal, self.n)
        g = self.incidence @ x
        if weighted:
            g = self.sqrt_weights * g
        return g

    def edge_gradient_table(self, signal: ArrayLike, *, weighted: bool = True) -> List[EdgeScore]:
        g = self.gradient(signal, weighted=weighted)
        out: List[EdgeScore] = []
        for val, (i, j, w) in zip(g.tolist(), self.edges):
            out.append(EdgeScore(i=i, j=j, score=float(val), weight=float(w)))
        return out

    def divergence(self, edge_field: ArrayLike, *, weighted: bool = True) -> np.ndarray:
        y = np.asarray(edge_field, dtype=float).reshape(-1)
        if y.shape[0] != len(self.edges):
            raise SignalShapeError(f"edge field has length {y.shape[0]}, expected {len(self.edges)}")
        if weighted:
            y = self.sqrt_weights * y
        return -(self.incidence.T @ y)

    def laplacian_apply(self, signal: ArrayLike) -> np.ndarray:
        return graph_laplacian(self.A) @ _validate_signal(signal, self.n)

    def dirichlet_energy(self, signal: ArrayLike) -> float:
        g = self.gradient(signal, weighted=True)
        return 0.5 * float(np.dot(g, g))

    def total_variation(self, signal: ArrayLike, *, p: float = 1.0, weighted: bool = True) -> float:
        g = np.abs(self.gradient(signal, weighted=weighted))
        if p == 1:
            return float(np.sum(g))
        if p == 2:
            return float(np.sqrt(np.sum(g * g)))
        if math.isinf(p):
            return float(np.max(g)) if g.size else 0.0
        return float(np.sum(g ** p) ** (1.0 / p))

    def p_laplacian_apply(self, signal: ArrayLike, *, p: float = 2.0, epsilon: float = 1e-8) -> np.ndarray:
        x = _validate_signal(signal, self.n)
        raw = self.incidence @ x
        if abs(p - 2.0) <= 1e-12:
            return self.laplacian_apply(x)
        weights = np.asarray([w for _i, _j, w in self.edges], dtype=float)
        coeff = weights * np.power(raw * raw + epsilon * epsilon, 0.5 * (p - 2.0))
        field = coeff * raw
        return self.incidence.T @ field

    def local_disagreement(self, signal: ArrayLike) -> np.ndarray:
        x = _validate_signal(signal, self.n)
        deg = _degree(self.A)
        disagreement = np.zeros(self.n, dtype=float)
        for i, j, w in self.edges:
            d = abs(x[i] - x[j]) * w
            disagreement[i] += d
            disagreement[j] += d
        mask = deg > _EPS
        disagreement[mask] /= deg[mask]
        return disagreement


class ChebyshevFilterBank:
    """Chebyshev polynomial approximation to spectral graph filters.

    Given a transfer function ``g(lambda)``, the filter approximates ``g(L)x``
    without forming eigenvectors.  This is the appropriate scalable method for
    residue graphs large enough that dense eigendecomposition is undesirable.
    """

    def __init__(self, L: np.ndarray, *, lmax: Optional[float] = None):
        self.L = np.asarray(L, dtype=float)
        self.n = self.L.shape[0]
        if self.L.ndim != 2 or self.L.shape[0] != self.L.shape[1]:
            raise FilteringError("Chebyshev filter requires a square Laplacian")
        if lmax is None:
            # Gershgorin / exact hybrid: exact is OK for current project scale;
            # fall back to diagonal bound if eigh fails.
            try:
                lmax = float(np.max(np.linalg.eigvalsh(0.5 * (self.L + self.L.T))))
            except np.linalg.LinAlgError:
                lmax = float(np.max(np.sum(np.abs(self.L), axis=1)))
        self.lmax = max(float(lmax), _EPS)
        self.L_scaled = (2.0 / self.lmax) * self.L - np.eye(self.n)

    @staticmethod
    def coefficients(transfer: TransferFunction, order: int, *, lmax: float, grid_size: int = 2048) -> np.ndarray:
        K = int(order)
        if K < 0:
            raise FilteringError("Chebyshev order must be nonnegative")
        m = max(int(grid_size), 4 * (K + 1))
        j = np.arange(m)
        theta = math.pi * (j + 0.5) / m
        x = np.cos(theta)
        lam = 0.5 * float(lmax) * (x + 1.0)
        values = np.asarray(transfer(lam), dtype=float)
        coeffs = np.zeros(K + 1, dtype=float)
        for k in range(K + 1):
            coeffs[k] = (2.0 / m) * float(np.sum(values * np.cos(k * theta)))
        coeffs[0] *= 0.5
        return coeffs

    def apply_coefficients(self, signal: ArrayLike, coeffs: ArrayLike) -> np.ndarray:
        x = _validate_signal(signal, self.n)
        c = np.asarray(coeffs, dtype=float).reshape(-1)
        if c.size == 0:
            return np.zeros_like(x)
        if c.size == 1:
            return c[0] * x
        T0 = x
        T1 = self.L_scaled @ x
        y = c[0] * T0 + c[1] * T1
        for k in range(2, c.size):
            T2 = 2.0 * (self.L_scaled @ T1) - T0
            y = y + c[k] * T2
            T0, T1 = T1, T2
        return y

    def apply(self, signal: ArrayLike, transfer: TransferFunction, *, order: int = 30, grid_size: int = 2048) -> np.ndarray:
        coeffs = self.coefficients(transfer, order, lmax=self.lmax, grid_size=grid_size)
        return self.apply_coefficients(signal, coeffs)


class LanczosHeatKernel:
    """Krylov approximation for ``exp(-t L) x``.

    This class implements a short-recurrence Lanczos projection.  For each input
    vector, it constructs a Krylov basis ``Q`` and tridiagonal matrix ``T`` such
    that ``exp(-tL)x ≈ ||x|| Q exp(-tT) e_1``.  It is substantially more
    algorithmic than brute-force diffusion stepping and is valuable when one
    needs heat smoothing without a full eigenbasis.
    """

    def __init__(self, L: np.ndarray):
        self.L = np.asarray(L, dtype=float)
        if self.L.ndim != 2 or self.L.shape[0] != self.L.shape[1]:
            raise FilteringError("Lanczos heat kernel requires a square Laplacian")
        self.n = self.L.shape[0]

    def project(self, signal: ArrayLike, *, dimension: int = 30, reorthogonalize: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        x = _validate_signal(signal, self.n)
        beta0 = float(np.linalg.norm(x))
        if beta0 <= _EPS:
            return np.zeros((self.n, 1)), np.zeros((1, 1)), 0.0
        m = max(1, min(int(dimension), self.n))
        Q = np.zeros((self.n, m), dtype=float)
        alpha = np.zeros(m, dtype=float)
        beta = np.zeros(max(m - 1, 0), dtype=float)
        q_prev = np.zeros(self.n, dtype=float)
        q = x / beta0
        actual = m
        for k in range(m):
            Q[:, k] = q
            z = self.L @ q
            if k > 0:
                z -= beta[k - 1] * q_prev
            alpha[k] = float(np.dot(q, z))
            z -= alpha[k] * q
            if reorthogonalize:
                # Modified Gram-Schmidt against all prior Lanczos vectors.  This
                # trades a little cost for numerical reliability in dense NumPy.
                for j in range(k + 1):
                    corr = float(np.dot(Q[:, j], z))
                    z -= corr * Q[:, j]
            if k == m - 1:
                break
            b = float(np.linalg.norm(z))
            if b <= 1e-12:
                actual = k + 1
                break
            beta[k] = b
            q_prev = q
            q = z / b
        Q = Q[:, :actual]
        T = np.diag(alpha[:actual])
        for k in range(actual - 1):
            T[k, k + 1] = beta[k]
            T[k + 1, k] = beta[k]
        return Q, T, beta0

    def apply(self, signal: ArrayLike, *, time_value: float = 1.0, dimension: int = 30) -> np.ndarray:
        Q, T, norm = self.project(signal, dimension=dimension)
        if norm <= _EPS:
            return np.zeros(self.n, dtype=float)
        vals, vecs = _stable_eigh(T)
        e1 = np.zeros(T.shape[0], dtype=float)
        e1[0] = 1.0
        small = vecs @ (np.exp(-float(time_value) * vals) * (vecs.T @ e1))
        return norm * (Q @ small)


class GraphWaveletTransform:
    """Spectral graph wavelets for residue-level signals."""

    def __init__(self, basis: FourierBasis):
        self.basis = basis

    @staticmethod
    def mexican_hat_kernel(lam: np.ndarray, scale: float) -> np.ndarray:
        z = np.maximum(float(scale) * np.asarray(lam, dtype=float), 0.0)
        return z * np.exp(-z)

    @staticmethod
    def heat_kernel(lam: np.ndarray, scale: float) -> np.ndarray:
        return np.exp(-float(scale) * np.asarray(lam, dtype=float))

    @staticmethod
    def band_kernel(lam: np.ndarray, scale: float) -> np.ndarray:
        z = float(scale) * np.asarray(lam, dtype=float)
        return np.power(z, 2.0) * np.exp(-z)

    @staticmethod
    def diffusion_difference_kernel(lam: np.ndarray, scale: float) -> np.ndarray:
        s = float(scale)
        lam = np.asarray(lam, dtype=float)
        return np.exp(-s * lam) - np.exp(-2.0 * s * lam)

    def kernel(self, name: str) -> Callable[[np.ndarray, float], np.ndarray]:
        key = name.lower().replace("-", "_")
        if key in {"mexican_hat", "mexican", "laplacian_of_heat"}:
            return self.mexican_hat_kernel
        if key in {"heat", "heat_kernel"}:
            return self.heat_kernel
        if key in {"band", "band_kernel"}:
            return self.band_kernel
        if key in {"diffusion_difference", "dog", "difference_of_heat"}:
            return self.diffusion_difference_kernel
        raise FilteringError(f"unknown wavelet kernel: {name!r}")

    def transform(
        self,
        signal: ArrayLike,
        scales: Sequence[float],
        *,
        kernel: str = "mexican_hat",
        include_scaling: bool = True,
    ) -> Dict[float, np.ndarray]:
        x = _validate_signal(signal, self.basis.n)
        coeffs = self.basis.coefficients(x)
        fun = self.kernel(kernel)
        out: Dict[float, np.ndarray] = {}
        for scale in scales:
            s = float(scale)
            response = np.asarray(fun(self.basis.eigenvalues, s), dtype=float)
            out[s] = self.basis.synthesize(response * coeffs)
        if include_scaling:
            # A low-frequency scaling channel stabilizes reconstruction and
            # exposes coarse residue-domain trends.
            max_scale = max([float(s) for s in scales], default=1.0)
            response = np.exp(-max_scale * self.basis.eigenvalues)
            out[0.0] = self.basis.synthesize(response * coeffs)
        return out

    def frame_energy(self, coeffs_by_scale: Mapping[float, np.ndarray]) -> Dict[float, float]:
        return {float(s): float(np.dot(c, c)) for s, c in coeffs_by_scale.items()}

    def approximate_inverse(self, coeffs_by_scale: Mapping[float, np.ndarray], *, weights: Optional[Mapping[float, float]] = None) -> np.ndarray:
        if not coeffs_by_scale:
            return np.zeros(self.basis.n, dtype=float)
        y = np.zeros(self.basis.n, dtype=float)
        total = 0.0
        for scale, coeffs in coeffs_by_scale.items():
            w = 1.0 if weights is None else float(weights.get(scale, 1.0))
            y += w * _validate_signal(coeffs, self.basis.n, name="wavelet_coefficients")
            total += abs(w)
        if total <= _EPS:
            return y
        return y / total

    def localization_entropy(self, coeffs: ArrayLike) -> float:
        c = np.abs(_validate_signal(coeffs, self.basis.n))
        total = float(np.sum(c))
        if total <= _EPS:
            return 0.0
        p = c / total
        p = p[p > _EPS]
        return -float(np.sum(p * np.log(p))) / math.log(self.basis.n) if self.basis.n > 1 else 0.0

    def report(
        self,
        signal: GraphSignal,
        scales: Sequence[float],
        *,
        kernel: str = "mexican_hat",
        top_k: int = 10,
        labels: Optional[Sequence[str]] = None,
    ) -> MultiscaleReport:
        coeffs = self.transform(signal.values, scales, kernel=kernel, include_scaling=False)
        scale_reports: List[WaveletScaleReport] = []
        matrices: List[np.ndarray] = []
        energies: List[float] = []
        for scale in scales:
            s = float(scale)
            c = coeffs[s]
            matrices.append(c)
            e = float(np.dot(c, c))
            energies.append(e)
            tmp = GraphSignal(c, bundle=signal.bundle, name=f"{signal.name}_wavelet_{s:g}")
            scale_reports.append(
                WaveletScaleReport(
                    scale=s,
                    coefficient_norm=float(np.linalg.norm(c)),
                    coefficient_l1=float(np.sum(np.abs(c))),
                    coefficient_max_abs=float(np.max(np.abs(c))) if c.size else 0.0,
                    localization_entropy=self.localization_entropy(c),
                    top_nodes=tmp.top_nodes(top_k, labels=labels),
                )
            )
        corr = []
        for a in matrices:
            row = []
            for b in matrices:
                row.append(_safe_correlation(a, b))
            corr.append(row)
        if matrices:
            stack = np.vstack([np.abs(v) for v in matrices])
            dominant = np.argmax(stack, axis=0).astype(int).tolist()
        else:
            dominant = []
        total = float(sum(energies))
        distribution = [float(e / total) if total > _EPS else 0.0 for e in energies]
        return MultiscaleReport(
            signal_name=signal.name,
            scales=scale_reports,
            cross_scale_correlations=corr,
            dominant_scale_by_node=dominant,
            scale_energy_distribution=distribution,
            metadata={"kernel": kernel, "basis": self.basis.to_dict(include_vectors=False)},
        )


class GraphSignalProcessor:
    """High-level GSP engine for one residue interaction graph."""

    def __init__(
        self,
        bundle: GraphBundle,
        *,
        laplacian_kind: str = "combinatorial",
        symmetrize: bool = True,
        remove_self_loops: bool = True,
    ):
        bundle.validate()
        self.bundle = bundle
        self.A = _clean_adjacency(bundle.A, symmetrize=symmetrize, remove_self_loops=remove_self_loops)
        self.n = bundle.n
        self.laplacian_kind = laplacian_kind
        self.L = _laplacian(self.A, kind=laplacian_kind)
        self.degree = _degree(self.A)
        self.operators = GraphDifferentialOperators(self.A)
        self._basis_cache: Dict[str, FourierBasis] = {}

    def _wrap(self, values: ArrayLike, *, source: GraphSignal, name: str, metadata: Optional[Dict[str, Any]] = None) -> GraphSignal:
        merged = dict(source.metadata)
        if metadata:
            merged.update(metadata)
        return GraphSignal(np.asarray(values, dtype=float), bundle=self.bundle, name=name, metadata=merged)

    def signal(self, values: ArrayLike, *, name: str = "signal", metadata: Optional[Dict[str, Any]] = None) -> GraphSignal:
        return GraphSignal(_validate_signal(values, self.n), bundle=self.bundle, name=name, metadata=metadata or {})

    def basis(self, *, modes: Optional[int] = None, force_recompute: bool = False) -> FourierBasis:
        key = f"{self.laplacian_kind}:full"
        if force_recompute or key not in self._basis_cache:
            if SpectralGeometryEngine is not None:
                try:
                    engine = SpectralGeometryEngine(self.bundle, laplacian_kind=self.laplacian_kind)  # type: ignore[operator]
                    dec = engine.eigendecomposition()  # accepts default full basis in generated engine
                    eigenvalues = getattr(dec, "eigenvalues", None)
                    eigenvectors = getattr(dec, "eigenvectors", None)
                    if eigenvalues is not None and eigenvectors is not None:
                        self._basis_cache[key] = FourierBasis(
                            eigenvalues=np.asarray(eigenvalues, dtype=float),
                            eigenvectors=np.asarray(eigenvectors, dtype=float),
                            laplacian_kind=self.laplacian_kind,
                            graph_name=self.bundle.name,
                        )
                    else:
                        vals, vecs = _stable_eigh(self.L)
                        self._basis_cache[key] = FourierBasis(vals, vecs, self.laplacian_kind, self.bundle.name)
                except Exception:
                    vals, vecs = _stable_eigh(self.L)
                    self._basis_cache[key] = FourierBasis(vals, vecs, self.laplacian_kind, self.bundle.name)
            else:
                vals, vecs = _stable_eigh(self.L)
                self._basis_cache[key] = FourierBasis(vals, vecs, self.laplacian_kind, self.bundle.name)
        b = self._basis_cache[key]
        return b.truncate(modes) if modes is not None else b

    def graph_fourier_transform(self, signal: Union[GraphSignal, ArrayLike], *, modes: Optional[int] = None) -> np.ndarray:
        x = signal.values if isinstance(signal, GraphSignal) else signal
        return self.basis(modes=modes).coefficients(x)

    def inverse_graph_fourier_transform(self, coeffs: ArrayLike, *, modes: Optional[int] = None) -> GraphSignal:
        b = self.basis(modes=modes)
        y = b.synthesize(coeffs)
        return GraphSignal(y, bundle=self.bundle, name="inverse_gft", metadata={"modes": b.modes})

    def apply_spectral_transfer(
        self,
        signal: GraphSignal,
        transfer: TransferFunction,
        *,
        name: str = "spectral_filtered",
        modes: Optional[int] = None,
        spec: Optional[FilterSpec] = None,
    ) -> FilterResult:
        b = self.basis(modes=modes)
        coeff = b.coefficients(signal.values)
        response = np.asarray(transfer(b.eigenvalues), dtype=float).reshape(-1)
        if response.shape[0] != b.modes:
            raise FilteringError(f"transfer returned {response.shape[0]} values for {b.modes} modes")
        y = b.synthesize(response * coeff)
        out = GraphSignal(y, bundle=self.bundle, name=name, metadata={"filter_response_min": float(np.min(response)), "filter_response_max": float(np.max(response))})
        return FilterResult(
            original=signal,
            filtered=out,
            spec=spec or FilterSpec(name=name, kind="custom", parameters={}),
            spectral_energy_before=coeff * coeff,
            spectral_energy_after=(response * coeff) ** 2,
            metadata={"modes": b.modes, "laplacian_kind": self.laplacian_kind},
        )

    def low_pass(self, signal: GraphSignal, *, cutoff: float, rolloff: float = 0.0, modes: Optional[int] = None) -> FilterResult:
        spec = FilterSpec("low_pass", "low_pass", {"cutoff": cutoff, "rolloff": rolloff})
        return self.apply_spectral_transfer(signal, spec.transfer(), name=f"{signal.name}_lowpass", modes=modes, spec=spec)

    def high_pass(self, signal: GraphSignal, *, cutoff: float, rolloff: float = 0.0, modes: Optional[int] = None) -> FilterResult:
        spec = FilterSpec("high_pass", "high_pass", {"cutoff": cutoff, "rolloff": rolloff})
        return self.apply_spectral_transfer(signal, spec.transfer(), name=f"{signal.name}_highpass", modes=modes, spec=spec)

    def band_pass(self, signal: GraphSignal, *, low: float, high: float, rolloff: float = 0.0, modes: Optional[int] = None) -> FilterResult:
        spec = FilterSpec("band_pass", "band_pass", {"low": low, "high": high, "rolloff": rolloff})
        return self.apply_spectral_transfer(signal, spec.transfer(), name=f"{signal.name}_bandpass", modes=modes, spec=spec)

    def heat_smooth(self, signal: GraphSignal, *, time: float = 1.0, modes: Optional[int] = None, method: str = "spectral") -> GraphSignal:
        method = method.lower()
        if method == "spectral":
            spec = FilterSpec("heat", "heat", {"time": time})
            return self.apply_spectral_transfer(signal, spec.transfer(), name=f"{signal.name}_heat_t{time:g}", modes=modes, spec=spec).filtered
        if method == "chebyshev":
            return self.chebyshev_filter(signal, FilterSpec("heat", "heat", {"time": time}), order=40).filtered
        if method == "lanczos":
            y = LanczosHeatKernel(self.L).apply(signal.values, time_value=time, dimension=min(40, self.n))
            return self._wrap(y, source=signal, name=f"{signal.name}_lanczos_heat", metadata={"time": time, "method": "lanczos"})
        raise FilteringError(f"unknown heat smoothing method: {method!r}")

    def tikhonov_smooth(self, signal: GraphSignal, *, alpha: float = 1.0) -> GraphSignal:
        M = np.eye(self.n) + float(alpha) * self.L
        y = _matrix_power_solve(M, signal.values)
        return self._wrap(y, source=signal, name=f"{signal.name}_tikhonov", metadata={"alpha": float(alpha)})

    def polynomial_filter(self, signal: GraphSignal, coefficients: Sequence[float], *, name: str = "polynomial_filtered") -> FilterResult:
        c = np.asarray(coefficients, dtype=float).reshape(-1)
        if c.size == 0:
            raise FilteringError("polynomial filter needs at least one coefficient")
        y = np.zeros(self.n, dtype=float)
        current = signal.values.copy()
        for k, coeff in enumerate(c):
            if k == 0:
                current = signal.values.copy()
            elif k == 1:
                current = self.L @ signal.values
            else:
                current = self.L @ current
            y += coeff * current
        out = self._wrap(y, source=signal, name=name, metadata={"coefficients": c.tolist(), "basis": "monomial_laplacian"})
        return FilterResult(signal, out, FilterSpec(name, "polynomial", {"coefficients": c.tolist()}), metadata={"laplacian_kind": self.laplacian_kind})

    def chebyshev_filter(
        self,
        signal: GraphSignal,
        spec: FilterSpec,
        *,
        order: int = 30,
        grid_size: int = 2048,
        name: Optional[str] = None,
    ) -> FilterResult:
        bank = ChebyshevFilterBank(self.L)
        y = bank.apply(signal.values, spec.transfer(), order=order, grid_size=grid_size)
        out = self._wrap(y, source=signal, name=name or f"{signal.name}_{spec.name}_cheb", metadata={"chebyshev_order": int(order), "spec": spec.to_dict()})
        return FilterResult(signal, out, spec, metadata={"method": "chebyshev", "order": int(order), "lmax": bank.lmax})

    def total_variation(self, signal: GraphSignal, *, p: float = 1.0) -> float:
        return self.operators.total_variation(signal.values, p=p)

    def dirichlet_energy(self, signal: GraphSignal) -> float:
        return self.operators.dirichlet_energy(signal.values)

    def gradient(self, signal: GraphSignal, *, weighted: bool = True) -> np.ndarray:
        return self.operators.gradient(signal.values, weighted=weighted)

    def divergence(self, edge_field: ArrayLike, *, weighted: bool = True) -> np.ndarray:
        return self.operators.divergence(edge_field, weighted=weighted)

    def wavelets(self) -> GraphWaveletTransform:
        return GraphWaveletTransform(self.basis())

    def wavelet_transform(
        self,
        signal: GraphSignal,
        scales: Sequence[float],
        *,
        kernel: str = "mexican_hat",
        include_scaling: bool = True,
    ) -> Dict[float, GraphSignal]:
        raw = self.wavelets().transform(signal.values, scales, kernel=kernel, include_scaling=include_scaling)
        return {
            float(s): self._wrap(c, source=signal, name=f"{signal.name}_wavelet_{float(s):g}", metadata={"scale": float(s), "kernel": kernel})
            for s, c in raw.items()
        }

    def analyze_multiscale(
        self,
        signal: GraphSignal,
        *,
        scales: Optional[Sequence[float]] = None,
        kernel: str = "mexican_hat",
        top_k: int = 10,
        labels: Optional[Sequence[str]] = None,
    ) -> MultiscaleReport:
        if scales is None:
            scales = default_wavelet_scales(self.basis().eigenvalues, count=8)
        return self.wavelets().report(signal, scales, kernel=kernel, top_k=top_k, labels=labels)

    def energy_report(self, signal: GraphSignal, *, high_frequency_cutoff: Optional[float] = None) -> EnergyReport:
        x = _validate_signal(signal.values, self.n)
        b = self.basis()
        density = b.energy_density(x)
        total_energy = float(np.sum(density))
        if total_energy <= _EPS:
            centroid = 0.0
            hfr = 0.0
        else:
            centroid = float(np.dot(b.eigenvalues, density) / total_energy)
            cutoff = high_frequency_cutoff
            if cutoff is None:
                cutoff = float(np.median(b.eigenvalues))
            hfr = float(np.sum(density[b.eigenvalues >= cutoff]) / total_energy)
        return EnergyReport(
            signal_name=signal.name,
            l2_norm=float(np.linalg.norm(x)),
            l1_norm=float(np.sum(np.abs(x))),
            linf_norm=float(np.max(np.abs(x))) if x.size else 0.0,
            mean=float(np.mean(x)) if x.size else 0.0,
            std=float(np.std(x)) if x.size else 0.0,
            total_variation=self.total_variation(signal),
            dirichlet_energy=self.dirichlet_energy(signal),
            laplacian_kind=self.laplacian_kind,
            spectral_entropy=b.spectral_entropy(x),
            spectral_centroid=centroid,
            high_frequency_ratio=hfr,
        )

    def denoise_tikhonov(self, signal: GraphSignal, *, alpha: float = 1.0) -> DenoisingReport:
        den = self.tikhonov_smooth(signal, alpha=alpha)
        residual = self._wrap(signal.values - den.values, source=signal, name=f"{signal.name}_residual", metadata={"method": "tikhonov"})
        obj = 0.5 * float(np.dot(den.values - signal.values, den.values - signal.values)) + 0.5 * float(alpha) * self.dirichlet_energy(den)
        return DenoisingReport(signal, den, residual, "tikhonov", [obj], metadata={"alpha": float(alpha)})

    def denoise_heat(self, signal: GraphSignal, *, time: float = 1.0, method: str = "lanczos") -> DenoisingReport:
        den = self.heat_smooth(signal, time=time, method=method)
        residual = self._wrap(signal.values - den.values, source=signal, name=f"{signal.name}_heat_residual", metadata={"method": "heat", "time": time})
        obj = 0.5 * float(np.dot(residual.values, residual.values))
        return DenoisingReport(signal, den, residual, f"heat_{method}", [obj], metadata={"time": float(time)})

    def denoise_wavelet_shrinkage(
        self,
        signal: GraphSignal,
        *,
        scales: Optional[Sequence[float]] = None,
        kernel: str = "mexican_hat",
        threshold: Optional[float] = None,
        threshold_mode: str = "soft",
    ) -> DenoisingReport:
        if scales is None:
            scales = default_wavelet_scales(self.basis().eigenvalues, count=8)
        wt = self.wavelets()
        coeffs = wt.transform(signal.values, scales, kernel=kernel, include_scaling=True)
        all_detail = np.concatenate([v.reshape(-1) for s, v in coeffs.items() if float(s) != 0.0]) if coeffs else np.array([])
        if threshold is None:
            sigma = 1.4826 * _mad(all_detail) if all_detail.size else 0.0
            threshold = sigma * math.sqrt(2.0 * math.log(max(self.n, 2)))
        shrunk: Dict[float, np.ndarray] = {}
        for s, c in coeffs.items():
            if float(s) == 0.0:
                shrunk[s] = c
            elif threshold_mode == "soft":
                shrunk[s] = _soft_threshold(c, float(threshold))
            elif threshold_mode == "hard":
                shrunk[s] = np.where(np.abs(c) >= float(threshold), c, 0.0)
            else:
                raise DenoisingError(f"unknown threshold mode: {threshold_mode!r}")
        y = wt.approximate_inverse(shrunk)
        den = self._wrap(y, source=signal, name=f"{signal.name}_wavelet_denoised", metadata={"threshold": threshold, "kernel": kernel})
        residual = self._wrap(signal.values - den.values, source=signal, name=f"{signal.name}_wavelet_residual", metadata={"threshold": threshold})
        obj = [0.5 * float(np.dot(residual.values, residual.values)) + float(threshold) * float(np.sum(np.abs(np.concatenate(list(shrunk.values())))))]
        return DenoisingReport(signal, den, residual, "wavelet_shrinkage", obj, metadata={"scales": list(map(float, scales)), "kernel": kernel, "threshold": float(threshold)})

    def denoise_tv_ista(
        self,
        signal: GraphSignal,
        *,
        lam: float = 0.1,
        iterations: int = 250,
        step: Optional[float] = None,
        tolerance: float = 1e-7,
    ) -> DenoisingReport:
        """Approximate graph total-variation denoising with primal smoothing.

        The objective is ``0.5||x-y||^2 + lam * ||B x||_1``.  A full production
        solver would use Chambolle-Pock or ADMM.  Here we implement a compact
        proximal-gradient-like majorization using a smoothed TV penalty.  It is
        deterministic, dependency-free, and gives a useful denoised signal for
        moderate graphs.
        """
        y = signal.values.copy()
        x = y.copy()
        if step is None:
            lip = 1.0 + float(lam) * max(float(np.max(self.degree)) * 2.0, 1.0)
            step = 1.0 / lip
        trace: List[float] = []
        eps = 1e-5
        for it in range(int(iterations)):
            grad_data = x - y
            edge_diff = self.operators.incidence @ x
            edge_weight = np.asarray([w for _i, _j, w in self.operators.edges], dtype=float)
            smooth_abs_grad = edge_weight * edge_diff / np.sqrt(edge_diff * edge_diff + eps * eps)
            grad_tv = self.operators.incidence.T @ smooth_abs_grad
            x_new = x - float(step) * (grad_data + float(lam) * grad_tv)
            residual = x_new - y
            tv = self.total_variation(GraphSignal(x_new, bundle=self.bundle, name="tmp"))
            obj = 0.5 * float(np.dot(residual, residual)) + float(lam) * tv
            trace.append(obj)
            if np.linalg.norm(x_new - x) <= float(tolerance) * (1.0 + np.linalg.norm(x)):
                x = x_new
                break
            x = x_new
        den = self._wrap(x, source=signal, name=f"{signal.name}_tv_denoised", metadata={"lambda": float(lam), "iterations": len(trace)})
        residual = self._wrap(signal.values - den.values, source=signal, name=f"{signal.name}_tv_residual", metadata={"lambda": float(lam)})
        return DenoisingReport(signal, den, residual, "smoothed_tv_ista", trace, metadata={"lambda": float(lam), "step": float(step), "tolerance": float(tolerance)})

    def anomaly_residual(
        self,
        signal: GraphSignal,
        *,
        smoother: str = "tikhonov",
        alpha: float = 1.0,
        time_value: float = 1.0,
        z_threshold: float = 3.0,
        labels: Optional[Sequence[str]] = None,
    ) -> AnomalyReport:
        if smoother == "tikhonov":
            den = self.denoise_tikhonov(signal, alpha=alpha)
        elif smoother == "heat":
            den = self.denoise_heat(signal, time=time_value)
        elif smoother == "wavelet":
            den = self.denoise_wavelet_shrinkage(signal)
        else:
            raise FilteringError(f"unknown smoother: {smoother!r}")
        residual = den.residual
        rz = residual.robust_zscored()
        scores = [NodeScore(i, float(rz.values[i]), labels[i] if labels is not None and i < len(labels) else None, {"raw_residual": float(residual.values[i])}) for i in range(self.n)]
        flagged = [int(s.node) for s in scores if abs(s.score) >= float(z_threshold)]
        return AnomalyReport(signal.name, "residual_robust_zscore", scores, float(z_threshold), flagged, residual_signal=residual, metadata={"smoother": smoother, "alpha": alpha, "time": time_value})

    def anomaly_local_disagreement(
        self,
        signal: GraphSignal,
        *,
        z_threshold: float = 3.0,
        labels: Optional[Sequence[str]] = None,
    ) -> AnomalyReport:
        disagreement = self.operators.local_disagreement(signal.values)
        tmp = GraphSignal(disagreement, bundle=self.bundle, name=f"{signal.name}_local_disagreement")
        z = tmp.robust_zscored()
        scores = [NodeScore(i, float(z.values[i]), labels[i] if labels is not None and i < len(labels) else None, {"raw_disagreement": float(disagreement[i])}) for i in range(self.n)]
        flagged = [int(s.node) for s in scores if s.score >= float(z_threshold)]
        return AnomalyReport(signal.name, "local_disagreement", scores, float(z_threshold), flagged, residual_signal=tmp, metadata={})

    def anomaly_spectral_leverage(
        self,
        signal: GraphSignal,
        *,
        high_frequency_quantile: float = 0.6,
        z_threshold: float = 3.0,
        labels: Optional[Sequence[str]] = None,
    ) -> AnomalyReport:
        b = self.basis()
        cutoff = float(np.quantile(b.eigenvalues, float(high_frequency_quantile)))
        coeffs = b.coefficients(signal.values)
        mask = b.eigenvalues >= cutoff
        high = b.eigenvectors[:, mask] @ coeffs[mask] if np.any(mask) else np.zeros(self.n)
        tmp = GraphSignal(high, bundle=self.bundle, name=f"{signal.name}_high_frequency_component")
        z = tmp.robust_zscored()
        scores = [NodeScore(i, float(z.values[i]), labels[i] if labels is not None and i < len(labels) else None, {"component": float(high[i])}) for i in range(self.n)]
        flagged = [int(s.node) for s in scores if abs(s.score) >= float(z_threshold)]
        return AnomalyReport(signal.name, "spectral_leverage", scores, float(z_threshold), flagged, residual_signal=tmp, metadata={"cutoff": cutoff, "quantile": high_frequency_quantile})

    def pairwise_signal_distance(self, signal_a: GraphSignal, signal_b: GraphSignal, *, metric: str = "l2") -> float:
        a = _validate_signal(signal_a.values, self.n, name=signal_a.name)
        b = _validate_signal(signal_b.values, self.n, name=signal_b.name)
        metric = metric.lower()
        if metric == "l2":
            return float(np.linalg.norm(a - b))
        if metric == "l1":
            return float(np.sum(np.abs(a - b)))
        if metric == "linf":
            return float(np.max(np.abs(a - b)))
        if metric == "dirichlet":
            return self.dirichlet_energy(GraphSignal(a - b, bundle=self.bundle, name="difference"))
        if metric == "spectral_angle":
            ca = self.graph_fourier_transform(signal_a)
            cb = self.graph_fourier_transform(signal_b)
            denom = float(np.linalg.norm(ca) * np.linalg.norm(cb))
            if denom <= _EPS:
                return 0.0
            cos = float(np.clip(np.dot(ca, cb) / denom, -1.0, 1.0))
            return float(math.acos(cos))
        raise FilteringError(f"unknown distance metric: {metric!r}")

    def export_signal_table(self, signals: Sequence[GraphSignal], path: Union[str, Path], *, labels: Optional[Sequence[str]] = None) -> Path:
        if not signals:
            raise SignalShapeError("no signals to export")
        for s in signals:
            _validate_signal(s.values, self.n, name=s.name)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["node"] + (["label"] if labels is not None else []) + [s.name for s in signals]
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(self.n):
                row: Dict[str, Any] = {"node": i}
                if labels is not None:
                    row["label"] = labels[i] if i < len(labels) else ""
                for s in signals:
                    row[s.name] = float(s.values[i])
                writer.writerow(row)
        return p

    def full_report(
        self,
        signal: GraphSignal,
        *,
        scales: Optional[Sequence[float]] = None,
        anomaly_threshold: float = 3.0,
        labels: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        energy = self.energy_report(signal)
        multiscale = self.analyze_multiscale(signal, scales=scales, labels=labels)
        residual_anom = self.anomaly_residual(signal, z_threshold=anomaly_threshold, labels=labels)
        local_anom = self.anomaly_local_disagreement(signal, z_threshold=anomaly_threshold, labels=labels)
        return {
            "graph": {"name": self.bundle.name, "n": self.n, "edges": len(self.operators.edges), "laplacian_kind": self.laplacian_kind},
            "signal": {"name": signal.name, "metadata": _json_ready(signal.metadata)},
            "energy": energy.to_dict(),
            "multiscale": multiscale.to_dict(),
            "anomalies": {
                "residual": residual_anom.to_dict(),
                "local_disagreement": local_anom.to_dict(),
            },
        }

    def write_full_report(self, signal: GraphSignal, outdir: Union[str, Path], *, labels: Optional[Sequence[str]] = None) -> Path:
        p = Path(outdir)
        p.mkdir(parents=True, exist_ok=True)
        report = self.full_report(signal, labels=labels)
        (p / "gsp_report.json").write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
        signal.to_csv(p / "signal.csv", labels=labels)
        self.analyze_multiscale(signal, labels=labels).to_markdown(p / "multiscale_report.md")
        lines = [
            f"# Graph signal processing report: `{signal.name}`",
            "",
            f"Graph: `{self.bundle.name}` with {self.n} nodes and {len(self.operators.edges)} undirected edges.",
            "",
            "## Energy summary",
            "",
        ]
        er = report["energy"]
        for key in ["l2_norm", "l1_norm", "linf_norm", "total_variation", "dirichlet_energy", "spectral_entropy", "spectral_centroid", "high_frequency_ratio"]:
            lines.append(f"- **{key}**: {er.get(key)}")
        lines.append("")
        lines.append("## Top residual anomalies")
        for ns in self.anomaly_residual(signal, labels=labels).top(10):
            lines.append(f"- node {ns.node}: {ns.score:.6g}")
        (p / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p


def default_wavelet_scales(eigenvalues: ArrayLike, *, count: int = 8) -> List[float]:
    vals = np.asarray(eigenvalues, dtype=float).reshape(-1)
    positive = vals[vals > 1e-10]
    if positive.size == 0:
        return [1.0]
    lo = 1.0 / float(np.max(positive))
    hi = 1.0 / float(np.min(positive))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.linspace(0.25, 4.0, int(count)).tolist()
    lo = max(lo, 1e-3)
    hi = min(hi, 1e3)
    return np.geomspace(lo, hi, int(count)).astype(float).tolist()


def graph_fourier_transform(bundle: GraphBundle, signal: ArrayLike, *, laplacian_kind: str = "combinatorial") -> np.ndarray:
    return GraphSignalProcessor(bundle, laplacian_kind=laplacian_kind).graph_fourier_transform(GraphSignal(signal, bundle=bundle))


def inverse_graph_fourier_transform(bundle: GraphBundle, coeffs: ArrayLike, *, laplacian_kind: str = "combinatorial") -> GraphSignal:
    return GraphSignalProcessor(bundle, laplacian_kind=laplacian_kind).inverse_graph_fourier_transform(coeffs)


def heat_kernel_smooth(bundle: GraphBundle, signal: ArrayLike, *, time_value: float = 1.0, method: str = "lanczos") -> GraphSignal:
    proc = GraphSignalProcessor(bundle)
    s = GraphSignal(signal, bundle=bundle, name="input_signal")
    return proc.heat_smooth(s, time=time_value, method=method)


def low_pass_filter(bundle: GraphBundle, signal: ArrayLike, *, cutoff: float, rolloff: float = 0.0) -> GraphSignal:
    proc = GraphSignalProcessor(bundle)
    s = GraphSignal(signal, bundle=bundle, name="input_signal")
    return proc.low_pass(s, cutoff=cutoff, rolloff=rolloff).filtered


def band_pass_filter(bundle: GraphBundle, signal: ArrayLike, *, low: float, high: float, rolloff: float = 0.0) -> GraphSignal:
    proc = GraphSignalProcessor(bundle)
    s = GraphSignal(signal, bundle=bundle, name="input_signal")
    return proc.band_pass(s, low=low, high=high, rolloff=rolloff).filtered


def graph_dirichlet_energy(bundle: GraphBundle, signal: ArrayLike) -> float:
    return GraphSignalProcessor(bundle).dirichlet_energy(GraphSignal(signal, bundle=bundle))


def graph_total_variation(bundle: GraphBundle, signal: ArrayLike, *, p: float = 1.0) -> float:
    return GraphSignalProcessor(bundle).total_variation(GraphSignal(signal, bundle=bundle), p=p)


def detect_signal_anomalies(
    bundle: GraphBundle,
    signal: ArrayLike,
    *,
    method: str = "residual",
    threshold: float = 3.0,
) -> AnomalyReport:
    proc = GraphSignalProcessor(bundle)
    s = GraphSignal(signal, bundle=bundle, name="input_signal")
    if method == "residual":
        return proc.anomaly_residual(s, z_threshold=threshold)
    if method == "local_disagreement":
        return proc.anomaly_local_disagreement(s, z_threshold=threshold)
    if method == "spectral_leverage":
        return proc.anomaly_spectral_leverage(s, z_threshold=threshold)
    raise FilteringError(f"unknown anomaly method: {method!r}")


def run_gsp_report_from_arrays(
    adjacency: np.ndarray,
    signal_values: ArrayLike,
    outdir: Union[str, Path],
    *,
    graph_name: str = "array_graph",
    signal_name: str = "signal",
) -> Path:
    bundle = GraphBundle(A=_clean_adjacency(adjacency), meta={}, name=graph_name)
    proc = GraphSignalProcessor(bundle)
    signal = GraphSignal(signal_values, bundle=bundle, name=signal_name)
    return proc.write_full_report(signal, outdir)


def run_gsp_report_from_csv(
    adjacency_csv: Union[str, Path],
    signal_csv: Union[str, Path],
    outdir: Union[str, Path],
    *,
    delimiter: str = ",",
    node_column: str = "node",
    value_column: str = "value",
) -> Path:
    A = np.loadtxt(str(adjacency_csv), delimiter=delimiter)
    bundle = GraphBundle(A=_clean_adjacency(A), meta={"adjacency_csv": str(adjacency_csv)}, name=Path(adjacency_csv).stem)
    signal = GraphSignal.from_csv(signal_csv, bundle=bundle, node_column=node_column, value_column=value_column)
    proc = GraphSignalProcessor(bundle)
    return proc.write_full_report(signal, outdir)


class GraphSignalBatch:
    """Utility for comparing and processing many residue signals on one graph."""

    def __init__(self, processor: GraphSignalProcessor, signals: Sequence[GraphSignal]):
        self.processor = processor
        self.signals = list(signals)
        for s in self.signals:
            _validate_signal(s.values, processor.n, name=s.name)

    def names(self) -> List[str]:
        return [s.name for s in self.signals]

    def matrix(self) -> np.ndarray:
        if not self.signals:
            return np.zeros((self.processor.n, 0), dtype=float)
        return np.column_stack([s.values for s in self.signals])

    def correlation_matrix(self, *, mode: str = "pearson") -> np.ndarray:
        m = len(self.signals)
        C = np.eye(m, dtype=float)
        for i in range(m):
            for j in range(i + 1, m):
                C[i, j] = C[j, i] = self.signals[i].similarity(self.signals[j], mode=mode)
        return C

    def spectral_energy_matrix(self, *, modes: Optional[int] = None) -> np.ndarray:
        b = self.processor.basis(modes=modes)
        if not self.signals:
            return np.zeros((b.modes, 0), dtype=float)
        return np.column_stack([b.energy_density(s.values) for s in self.signals])

    def denoise_all(self, *, method: str = "tikhonov", alpha: float = 1.0, time_value: float = 1.0) -> List[DenoisingReport]:
        out: List[DenoisingReport] = []
        for s in self.signals:
            if method == "tikhonov":
                out.append(self.processor.denoise_tikhonov(s, alpha=alpha))
            elif method == "heat":
                out.append(self.processor.denoise_heat(s, time=time_value))
            elif method == "wavelet":
                out.append(self.processor.denoise_wavelet_shrinkage(s))
            else:
                raise DenoisingError(f"unknown denoising method: {method!r}")
        return out

    def consensus_signal(self, *, weights: Optional[Sequence[float]] = None, name: str = "consensus_signal") -> GraphSignal:
        M = self.matrix()
        if M.shape[1] == 0:
            return GraphSignal.zeros(self.processor.bundle, name=name)
        if weights is None:
            y = np.mean(M, axis=1)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] != M.shape[1]:
                raise SignalShapeError("weights must match number of signals")
            denom = float(np.sum(w))
            abs_denom = float(np.sum(np.abs(w)))
            if abs_denom <= _EPS or abs(denom) <= _EPS:
                y = np.mean(M, axis=1)
            else:
                y = (M @ w) / denom
        return GraphSignal(y, bundle=self.processor.bundle, name=name, metadata={"source_signals": self.names()})

    def export_summary(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for s in self.signals:
            er = self.processor.energy_report(s).to_dict()
            er["signal_name"] = s.name
            rows.append(er)
        if not rows:
            p.write_text("", encoding="utf-8")
            return p
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return p


def _demo() -> None:
    from .graphio.synth import synthetic_rin

    bundle = synthetic_rin(n=40, seed=7)
    signal = GraphSignal.from_seeds(bundle, [0, 5, 9], strengths=[1.0, 0.7, -0.4], name="demo_seed")
    proc = GraphSignalProcessor(bundle)
    print("Graph:", bundle.name, "nodes=", bundle.n)
    print("Energy:", proc.energy_report(signal).to_dict())
    print("Heat top nodes:", [s.to_dict() for s in proc.heat_smooth(signal, time=1.5, method="lanczos").top_nodes(5)])
    print("Anomalies:", [s.to_dict() for s in proc.anomaly_local_disagreement(signal).top(5)])
    report = proc.analyze_multiscale(signal, top_k=3)
    print("Scales:", [s.scale for s in report.scales])


if __name__ == "__main__":  # pragma: no cover
    _demo()
