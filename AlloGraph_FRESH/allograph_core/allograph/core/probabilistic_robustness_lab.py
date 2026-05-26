"""Probabilistic robustness and uncertainty analysis for residue interaction networks.

This module turns a deterministic residue interaction network (RIN) into a small
uncertainty laboratory.  The central question is not merely "what score does a
residue receive?" but "does that score survive plausible graph uncertainty?".

The code is intentionally designed to fit the lightweight AlloGraph/RINet codebase:
only NumPy is required, and all optional integrations are soft imports.  The module
works with the existing :class:`allograph.core.graphio.types.GraphBundle` container
and the existing Laplacian convention ``L = D - A``.

Main capabilities
-----------------

* Edge-weight noise models: additive Gaussian, multiplicative log-normal,
  dropout, contact-probability thinning, degree-conditioned noise, and mixtures.
* Node-level dropout and masking models.
* Bootstrap graph samplers over weighted contacts.
* Monte Carlo diffusion under graph perturbations.
* Ensemble summaries with nodewise confidence intervals.
* Rank stability analysis: Spearman, Kendall, top-k overlap, rank entropy, and
  consensus ranking.
* Fragility analysis: how strong a perturbation must be before the top residues
  stop being stable.
* Robustness curves over perturbation strength.
* Structured report objects with JSON/CSV/Markdown export helpers.

The non-brute-force element is the way the module separates the experiment into
reusable stochastic perturbation models, deterministic diffusion kernels,
streaming summary accumulators, and rank-stability estimators.  This lets the
same trial stream support multiple downstream estimates without rerunning the
simulation, and it makes the uncertainty model explicit rather than hiding it in
ad hoc random edits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import csv
import hashlib
import json
import math
import time
import warnings

import numpy as np

from .graphio.types import GraphBundle
from .math.laplacian import graph_laplacian

try:  # Optional integration; never required.
    from .inference.forward import run_forward
except Exception:  # pragma: no cover - soft import
    run_forward = None  # type: ignore

try:  # Optional integration with the GSP module generated earlier.
    from .graph_signal_processing import GraphSignal, GraphSignalProcessor
except Exception:  # pragma: no cover - soft import
    GraphSignal = None  # type: ignore
    GraphSignalProcessor = None  # type: ignore

ArrayLike = Union[np.ndarray, Sequence[float], List[float], Tuple[float, ...]]
Edge = Tuple[int, int]
JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


# ---------------------------------------------------------------------------
# Basic numerical utilities
# ---------------------------------------------------------------------------


def _as_array(x: ArrayLike, *, dtype: Any = float, copy: bool = False) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if copy:
        arr = arr.copy()
    return arr


def _as_square_matrix(A: ArrayLike, *, name: str = "A", copy: bool = False) -> np.ndarray:
    arr = _as_array(A, dtype=float, copy=copy)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix; got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_signal(x: ArrayLike, n: int, *, name: str = "signal") -> np.ndarray:
    arr = _as_array(x, dtype=float, copy=True).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} length must equal number of nodes {n}; got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _rng(seed: Optional[Union[int, np.random.Generator]] = None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def copy_bundle(bundle: GraphBundle, *, A: Optional[np.ndarray] = None, name_suffix: str = "") -> GraphBundle:
    bundle.validate()
    meta = dict(bundle.meta or {})
    if name_suffix:
        meta["source_bundle_name"] = bundle.name
    return GraphBundle(A=np.asarray(bundle.A if A is None else A, dtype=float).copy(), meta=meta, name=f"{bundle.name}{name_suffix}")


def enforce_symmetric(A: np.ndarray, *, zero_diag: bool = True, mode: str = "average") -> np.ndarray:
    A = _as_square_matrix(A, copy=True)
    if mode == "average":
        A = 0.5 * (A + A.T)
    elif mode == "max":
        A = np.maximum(A, A.T)
    elif mode == "min":
        A = np.minimum(A, A.T)
    else:
        raise ValueError("mode must be one of: average, max, min")
    if zero_diag:
        np.fill_diagonal(A, 0.0)
    return A


def clip_adjacency(A: np.ndarray, *, min_weight: float = 0.0, max_weight: Optional[float] = None) -> np.ndarray:
    A = _as_square_matrix(A, copy=True)
    if max_weight is None:
        A = np.maximum(A, float(min_weight))
    else:
        A = np.clip(A, float(min_weight), float(max_weight))
    np.fill_diagonal(A, 0.0)
    return A


def upper_edges(A: np.ndarray, *, positive_only: bool = True) -> List[Edge]:
    A = _as_square_matrix(A)
    edges: List[Edge] = []
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if (not positive_only) or A[i, j] > 0 or A[j, i] > 0:
                edges.append((i, j))
    return edges


def edge_weights(A: np.ndarray, edges: Optional[Sequence[Edge]] = None) -> np.ndarray:
    A = _as_square_matrix(A)
    if edges is None:
        edges = upper_edges(A)
    return np.array([0.5 * (A[i, j] + A[j, i]) for i, j in edges], dtype=float)


def weighted_degrees(A: np.ndarray) -> np.ndarray:
    return np.sum(_as_square_matrix(A), axis=1)


def density(A: np.ndarray) -> float:
    A = _as_square_matrix(A)
    n = A.shape[0]
    if n <= 1:
        return 0.0
    return float(np.count_nonzero(np.triu(A > 0, 1)) / (n * (n - 1) / 2.0))


def graph_hash(A: np.ndarray, *, precision: int = 12) -> str:
    A = _as_square_matrix(A)
    rounded = np.round(A.astype(float), decimals=int(precision))
    h = hashlib.sha256()
    h.update(str(rounded.shape).encode("utf-8"))
    h.update(rounded.tobytes())
    return h.hexdigest()


def signal_hash(x: np.ndarray, *, precision: int = 12) -> str:
    arr = np.round(np.asarray(x, dtype=float).reshape(-1), decimals=int(precision))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def stable_argsort_desc(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return np.lexsort((np.arange(values.size), -values))


def ranks_desc(values: np.ndarray) -> np.ndarray:
    order = stable_argsort_desc(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1, dtype=float)
    return ranks


def top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    k = max(0, min(int(k), values.size))
    if k == 0:
        return np.array([], dtype=int)
    return stable_argsort_desc(values)[:k]


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ia = set(map(int, top_k_indices(a, k)))
    ib = set(map(int, top_k_indices(b, k)))
    if not ia and not ib:
        return 1.0
    if not ia or not ib:
        return 0.0
    return float(len(ia & ib) / len(ia | ib))


def normalize_signal(x: np.ndarray, *, mode: str = "l2") -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if mode == "none":
        return x.copy()
    if mode == "l1":
        denom = float(np.sum(np.abs(x)))
    elif mode == "l2":
        denom = float(np.linalg.norm(x))
    elif mode == "max":
        denom = float(np.max(np.abs(x)))
    elif mode == "zscore":
        std = float(np.std(x))
        if std <= 1e-15:
            return np.zeros_like(x)
        return (x - float(np.mean(x))) / std
    else:
        raise ValueError("normalization mode must be one of: none, l1, l2, max, zscore")
    if denom <= 1e-15:
        return np.zeros_like(x)
    return x / denom


def softmax_stable(x: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    temperature = max(float(temperature), 1e-12)
    z = x / temperature
    z = z - np.max(z)
    exp = np.exp(z)
    total = float(np.sum(exp))
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(x) / max(1, x.size)
    return exp / total


def entropy(p: np.ndarray, *, base: float = math.e) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    val = -float(np.sum(p * np.log(p)))
    if base != math.e:
        val /= math.log(base)
    return val


def quantile_interval(samples: np.ndarray, alpha: float = 0.05, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    lo = np.quantile(samples, alpha / 2.0, axis=axis)
    hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=axis)
    return lo, hi


def bootstrap_ci(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(samples, dtype=float)
    if arr.shape[0] == 0:
        raise ValueError("bootstrap_ci requires at least one sample")
    rng = _rng(seed)
    stats = []
    n = arr.shape[0]
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        stats.append(np.asarray(statistic(arr[idx]), dtype=float))
    mat = np.stack(stats, axis=0)
    center = np.asarray(statistic(arr), dtype=float)
    lo, hi = quantile_interval(mat, alpha=alpha, axis=0)
    return center, lo, hi


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError("Spearman inputs must have the same length")
    if a.size < 2:
        return 1.0
    ra = ranks_desc(a)
    rb = ranks_desc(b)
    ra = ra - float(np.mean(ra))
    rb = rb - float(np.mean(rb))
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    if denom <= 1e-15:
        return 0.0
    return float(np.dot(ra, rb) / denom)


def kendall_tau(a: np.ndarray, b: np.ndarray, *, max_pairs: Optional[int] = None, seed: Optional[int] = None) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError("Kendall inputs must have the same length")
    n = a.size
    if n < 2:
        return 1.0
    total_pairs = n * (n - 1) // 2
    if max_pairs is None or int(max_pairs) >= total_pairs:
        concordant = 0
        discordant = 0
        for i in range(n):
            da = a[i] - a[i + 1 :]
            db = b[i] - b[i + 1 :]
            prod = da * db
            concordant += int(np.sum(prod > 0))
            discordant += int(np.sum(prod < 0))
        denom = concordant + discordant
        return 0.0 if denom == 0 else float((concordant - discordant) / denom)
    rng = _rng(seed)
    max_pairs = int(max_pairs)
    concordant = 0
    discordant = 0
    used = 0
    for _ in range(max_pairs):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        prod = (a[i] - a[j]) * (b[i] - b[j])
        if prod > 0:
            concordant += 1
        elif prod < 0:
            discordant += 1
        used += 1
    denom = concordant + discordant
    return 0.0 if denom == 0 else float((concordant - discordant) / denom)


def rank_probability_matrix(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must be a matrix of shape (trials, nodes)")
    trials, n = samples.shape
    mat = np.zeros((n, n), dtype=float)
    for row in samples:
        r = ranks_desc(row).astype(int) - 1
        mat[np.arange(n), r] += 1.0
    if trials > 0:
        mat /= float(trials)
    return mat


def rank_entropy(samples: np.ndarray, *, base: float = 2.0) -> np.ndarray:
    P = rank_probability_matrix(samples)
    return np.array([entropy(P[i], base=base) for i in range(P.shape[0])], dtype=float)


def consensus_borda_score(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must be a matrix of shape (trials, nodes)")
    trials, n = samples.shape
    if trials == 0:
        return np.zeros(n, dtype=float)
    out = np.zeros(n, dtype=float)
    for row in samples:
        ranks = ranks_desc(row)
        out += (n - ranks) / max(1, n - 1)
    return out / float(trials)


def median_rank(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    ranks = np.stack([ranks_desc(row) for row in samples], axis=0)
    return np.median(ranks, axis=0)


def mean_abs_rank_shift(samples: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    baseline_ranks = ranks_desc(baseline)
    ranks = np.stack([ranks_desc(row) for row in samples], axis=0)
    return np.mean(np.abs(ranks - baseline_ranks[None, :]), axis=0)


def pairwise_correlation_summary(samples: np.ndarray, metric: str = "spearman", *, max_pairs: int = 5000, seed: Optional[int] = None) -> Dict[str, float]:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must be a matrix")
    m = samples.shape[0]
    if m < 2:
        return {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "n_pairs": 0.0}
    rng = _rng(seed)
    total = m * (m - 1) // 2
    if total <= max_pairs:
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    else:
        pairs = []
        seen = set()
        while len(pairs) < max_pairs:
            i = int(rng.integers(0, m))
            j = int(rng.integers(0, m - 1))
            if j >= i:
                j += 1
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                pairs.append((a, b))
    vals = []
    for i, j in pairs:
        if metric == "spearman":
            vals.append(spearman_correlation(samples[i], samples[j]))
        elif metric == "kendall":
            vals.append(kendall_tau(samples[i], samples[j], max_pairs=2000, seed=None))
        else:
            raise ValueError("metric must be 'spearman' or 'kendall'")
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_pairs": float(len(vals)),
    }


def node_table_to_csv(path: Union[str, Path], rows: Sequence[Mapping[str, Any]], *, fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Union[str, Path], data: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_json_default, sort_keys=True)


# ---------------------------------------------------------------------------
# Diffusion kernel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffusionConfig:
    """Configuration for deterministic diffusion used inside Monte Carlo trials."""

    steps: int = 60
    dt: float = 0.10
    model: str = "laplacian_euler"
    normalize_output: str = "none"
    clamp_negative: bool = False
    store_trajectory: bool = False

    def validate(self) -> None:
        if int(self.steps) < 0:
            raise ValueError("steps must be non-negative")
        if float(self.dt) < 0:
            raise ValueError("dt must be non-negative")
        if self.model not in {"laplacian_euler", "lazy_random_walk", "personalized_pagerank"}:
            raise ValueError("model must be laplacian_euler, lazy_random_walk, or personalized_pagerank")
        if self.normalize_output not in {"none", "l1", "l2", "max", "zscore"}:
            raise ValueError("unknown normalize_output mode")


@dataclass
class DiffusionTrace:
    final_state: np.ndarray
    trajectory: Optional[np.ndarray]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_state": self.final_state.tolist(),
            "trajectory": None if self.trajectory is None else self.trajectory.tolist(),
            "meta": dict(self.meta),
        }


class DiffusionKernel:
    """Deterministic signal propagation kernel shared by all Monte Carlo trials."""

    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.config = config or DiffusionConfig()
        self.config.validate()

    def run(self, A: np.ndarray, x0: np.ndarray) -> DiffusionTrace:
        A = _as_square_matrix(A)
        x = _as_signal(x0, A.shape[0], name="x0")
        cfg = self.config
        if cfg.model == "laplacian_euler":
            trace = self._run_laplacian_euler(A, x)
        elif cfg.model == "lazy_random_walk":
            trace = self._run_lazy_random_walk(A, x)
        elif cfg.model == "personalized_pagerank":
            trace = self._run_personalized_pagerank(A, x)
        else:  # pragma: no cover
            raise ValueError(f"unknown diffusion model {cfg.model!r}")
        y = trace.final_state
        if cfg.clamp_negative:
            y = np.maximum(y, 0.0)
        if cfg.normalize_output != "none":
            y = normalize_signal(y, mode=cfg.normalize_output)
        trace.final_state = y
        trace.meta["output_hash"] = signal_hash(y)
        return trace

    def _run_laplacian_euler(self, A: np.ndarray, x0: np.ndarray) -> DiffusionTrace:
        L = graph_laplacian(A)
        x = x0.copy()
        traj: Optional[List[np.ndarray]] = [x.copy()] if self.config.store_trajectory else None
        for _ in range(int(self.config.steps)):
            x = x - float(self.config.dt) * (L @ x)
            if traj is not None:
                traj.append(x.copy())
        return DiffusionTrace(
            final_state=x,
            trajectory=None if traj is None else np.stack(traj, axis=0),
            meta={
                "model": self.config.model,
                "steps": int(self.config.steps),
                "dt": float(self.config.dt),
                "operator": "I - dt L repeated",
            },
        )

    def _run_lazy_random_walk(self, A: np.ndarray, x0: np.ndarray) -> DiffusionTrace:
        deg = np.sum(A, axis=1)
        P = np.zeros_like(A, dtype=float)
        mask = deg > 1e-15
        P[mask] = A[mask] / deg[mask, None]
        stay = max(0.0, min(1.0, 1.0 - float(self.config.dt)))
        move = 1.0 - stay
        M = stay * np.eye(A.shape[0]) + move * P.T
        x = x0.copy()
        traj: Optional[List[np.ndarray]] = [x.copy()] if self.config.store_trajectory else None
        for _ in range(int(self.config.steps)):
            x = M @ x
            if traj is not None:
                traj.append(x.copy())
        return DiffusionTrace(
            final_state=x,
            trajectory=None if traj is None else np.stack(traj, axis=0),
            meta={"model": self.config.model, "steps": int(self.config.steps), "dt": float(self.config.dt)},
        )

    def _run_personalized_pagerank(self, A: np.ndarray, x0: np.ndarray) -> DiffusionTrace:
        deg = np.sum(A, axis=1)
        P = np.zeros_like(A, dtype=float)
        mask = deg > 1e-15
        P[mask] = A[mask] / deg[mask, None]
        alpha = max(0.0, min(1.0, float(self.config.dt)))
        teleport = normalize_signal(np.maximum(x0, 0.0), mode="l1")
        if float(np.sum(teleport)) <= 1e-15:
            teleport = np.ones(A.shape[0]) / A.shape[0]
        x = teleport.copy()
        traj: Optional[List[np.ndarray]] = [x.copy()] if self.config.store_trajectory else None
        for _ in range(int(self.config.steps)):
            x = (1.0 - alpha) * (P.T @ x) + alpha * teleport
            if traj is not None:
                traj.append(x.copy())
        return DiffusionTrace(
            final_state=x,
            trajectory=None if traj is None else np.stack(traj, axis=0),
            meta={"model": self.config.model, "steps": int(self.config.steps), "restart_probability": alpha},
        )


# ---------------------------------------------------------------------------
# Perturbation distributions
# ---------------------------------------------------------------------------


class ScalarDistribution(ABC):
    """Simple distribution interface used by perturbation models."""

    @abstractmethod
    def sample(self, rng: np.random.Generator, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantDistribution(ScalarDistribution):
    value: float

    def sample(self, rng: np.random.Generator, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        return np.full(size, float(self.value), dtype=float)

    def describe(self) -> Dict[str, Any]:
        return {"type": "constant", "value": float(self.value)}


@dataclass(frozen=True)
class NormalDistribution(ScalarDistribution):
    mean: float = 0.0
    std: float = 1.0

    def sample(self, rng: np.random.Generator, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        return rng.normal(float(self.mean), float(self.std), size=size)

    def describe(self) -> Dict[str, Any]:
        return {"type": "normal", "mean": float(self.mean), "std": float(self.std)}


@dataclass(frozen=True)
class UniformDistribution(ScalarDistribution):
    low: float = 0.0
    high: float = 1.0

    def sample(self, rng: np.random.Generator, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        return rng.uniform(float(self.low), float(self.high), size=size)

    def describe(self) -> Dict[str, Any]:
        return {"type": "uniform", "low": float(self.low), "high": float(self.high)}


@dataclass(frozen=True)
class LogNormalDistribution(ScalarDistribution):
    mean: float = 0.0
    sigma: float = 0.2

    def sample(self, rng: np.random.Generator, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        return rng.lognormal(float(self.mean), float(self.sigma), size=size)

    def describe(self) -> Dict[str, Any]:
        return {"type": "lognormal", "mean": float(self.mean), "sigma": float(self.sigma)}


@dataclass(frozen=True)
class BernoulliDistribution(ScalarDistribution):
    p: float = 0.5

    def sample(self, rng: np.random.Generator, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        return rng.binomial(1, float(self.p), size=size).astype(float)

    def describe(self) -> Dict[str, Any]:
        return {"type": "bernoulli", "p": float(self.p)}


# ---------------------------------------------------------------------------
# Perturbation model objects
# ---------------------------------------------------------------------------


@dataclass
class PerturbedGraph:
    """A sampled graph plus metadata explaining how it was sampled."""

    bundle: GraphBundle
    perturbation_name: str
    trial_index: int
    seed: Optional[int]
    changed_edges: List[Tuple[int, int, float, float]] = field(default_factory=list)
    dropped_nodes: List[int] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def A(self) -> np.ndarray:
        return self.bundle.A

    def delta_norm(self, original_A: np.ndarray) -> float:
        return float(np.linalg.norm(self.A - original_A))

    def relative_delta_norm(self, original_A: np.ndarray) -> float:
        denom = float(np.linalg.norm(original_A))
        if denom <= 1e-15:
            return self.delta_norm(original_A)
        return self.delta_norm(original_A) / denom

    def to_dict(self, *, include_edges: bool = False) -> Dict[str, Any]:
        data = {
            "perturbation_name": self.perturbation_name,
            "trial_index": int(self.trial_index),
            "seed": None if self.seed is None else int(self.seed),
            "n_changed_edges": int(len(self.changed_edges)),
            "dropped_nodes": list(map(int, self.dropped_nodes)),
            "meta": dict(self.meta),
            "graph_hash": graph_hash(self.A),
        }
        if include_edges:
            data["changed_edges"] = [
                {"i": int(i), "j": int(j), "old": float(old), "new": float(new)}
                for i, j, old, new in self.changed_edges
            ]
        return data


class GraphPerturbationModel(ABC):
    """Base class for stochastic graph perturbations."""

    name: str = "GraphPerturbationModel"

    def __init__(self, *, preserve_symmetry: bool = True, zero_diag: bool = True):
        self.preserve_symmetry = bool(preserve_symmetry)
        self.zero_diag = bool(zero_diag)

    @abstractmethod
    def sample(self, bundle: GraphBundle, rng: np.random.Generator, *, trial_index: int = 0, seed: Optional[int] = None) -> PerturbedGraph:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "preserve_symmetry": bool(self.preserve_symmetry),
            "zero_diag": bool(self.zero_diag),
        }

    def _finish(self, bundle: GraphBundle, A: np.ndarray, *, trial_index: int, seed: Optional[int], changed: List[Tuple[int, int, float, float]], dropped_nodes: Optional[List[int]] = None, meta: Optional[Dict[str, Any]] = None) -> PerturbedGraph:
        if self.preserve_symmetry:
            A = enforce_symmetric(A, zero_diag=self.zero_diag)
        else:
            A = clip_adjacency(A, min_weight=0.0)
            if self.zero_diag:
                np.fill_diagonal(A, 0.0)
        new_meta = dict(bundle.meta or {})
        new_meta["perturbation"] = self.describe()
        new_bundle = GraphBundle(A=A, meta=new_meta, name=f"{bundle.name}:{self.name}:trial{trial_index}")
        return PerturbedGraph(
            bundle=new_bundle,
            perturbation_name=self.name,
            trial_index=int(trial_index),
            seed=None if seed is None else int(seed),
            changed_edges=changed,
            dropped_nodes=[] if dropped_nodes is None else list(map(int, dropped_nodes)),
            meta={} if meta is None else dict(meta),
        )


class IdentityPerturbation(GraphPerturbationModel):
    name = "identity"

    def sample(self, bundle: GraphBundle, rng: np.random.Generator, *, trial_index: int = 0, seed: Optional[int] = None) -> PerturbedGraph:
        bundle.validate()
        return self._finish(bundle, bundle.A.copy(), trial_index=trial_index, seed=seed, changed=[], meta={"identity": True})


class AdditiveEdgeNoise(GraphPerturbationModel):
    """Add zero-mean noise to existing contacts, optionally scaled by edge weight."""

    name = "additive_edge_noise"

    def __init__(self, std: float = 0.05, *, relative: bool = True, positive_only: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self.std = float(std)
        self.relative = bool(relative)
        self.positive_only = bool(positive_only)

    def describe(self) -> Dict[str, Any]:
        data = super().describe()
        data.update({"std": self.std, "relative": self.relative, "positive_only": self.positive_only})
        return data

    def sample(self, bundle: GraphBundle, rng: np.random.Generator, *, trial_index: int = 0, seed: Optional[int] = None) -> PerturbedGraph:
        bundle.validate()
        A0 = bundle.A
        A = A0.copy()
        edges = upper_edges(A0, positive_only=self.positive_only)
        changed: List[Tuple[int, int, float, float]] = []
        if edges:
            noise = rng.normal(0.0, self.std, size=len(edges))
            for idx, (i, j) in enumerate(edges):
                old = float(0.5 * (A0[i, j] + A0[j, i]))
                scale = max(abs(old), 1e-12) if self.relative else 1.0
                new = max(0.0, old + float(noise[idx]) * scale)
                A[i, j] = new
                A[j, i] = new
                if abs(new - old) > 1e-15:
                    changed.append((i, j, old, new))
        return self._finish(bundle, A, trial_index=trial_index, seed=seed, changed=changed)


class MultiplicativeLogNormalEdgeNoise(GraphPerturbationModel):
    """Multiply edge weights by log-normal factors with median 1."""

    name = "multiplicative_lognormal_edge_noise"

    def __init__(self, sigma: float = 0.15, *, positive_only: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self.sigma = float(sigma)
        self.positive_only = bool(positive_only)

    def describe(self) -> Dict[str, Any]:
        data = super().describe()
        data.update({"sigma": self.sigma, "positive_only": self.positive_only})
        return data

    def sample(self, bundle: GraphBundle, rng: np.random.Generator, *, trial_index: int = 0, seed: Optional[int] = None) -> PerturbedGraph:
        bundle.validate()
        A0 = bundle.A
        A = A0.copy()
        edges = upper_edges(A0, positive_only=self.positive_only)
        factors = rng.lognormal(mean=-0.5 * self.sigma * self.sigma, sigma=self.sigma, size=len(edges))
        changed: List[Tuple[int, int, float, float]] = []
        for factor, (i, j) in zip(factors, edges):
            old = float(0.5 * (A0[i, j] + A0[j, i]))
            new = max(0.0, old * float(factor))
            A[i, j] = new
            A[j, i] = new
            if abs(new - old) > 1e-15:
                changed.append((i, j, old, new))
        return self._finish(bundle, A, trial_index=trial_index, seed=seed, changed=changed)


class EdgeDropout(GraphPerturbationModel):
    """Randomly remove existing contacts."""

    name = "edge_dropout"

    def __init__(self, dropout_probability: float = 0.05, *, positive_only: bool = True, protect_bridges_heuristic: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.dropout_probability = float(dropout_probability)
        self.positive_only = bool(positive_only)
        self.protect_bridges_heuristic = bool(protect_bridges_heuristic)

    def describe(self) -> Dict[str, Any]:
        data = super().describe()
        data.update(
            {
                "dropout_probability": self.dropout_probability,
                "positive_only": self.positive_only,
                "protect_bridges_heuristic": self.protect_bridges_heuristic,
            }
        )
        return data

    def sample(self, bundle: GraphBundle, rng: np.random.Generator, *, trial_index: int = 0, seed: Optional[int] = None) -> PerturbedGraph:
        bundle.validate()
        A0 = bundle.A
        A = A0.copy()
        edges = upper_edges(A0, positive_only=self.positive_only)
        deg = np.count_nonzero(A0 > 0, axis=1)
        changed: List[Tuple[int, int, float, float]] = []
        for i, j in edges:
            if self.protect_bridges_heuristic and (deg[i] <= 1 or deg[j] <= 1):
                continue
            if float(rng.random()) < self.dropout_probability:
                old = float(0.5 * (A0[i, j] + A0[j, i]))
                A[i, j] = 0.0
                A[j, i] = 0.0
                changed.append((i, j, old, 0.0))
        return self._finish(bundle, A, trial_index=trial_index, seed=seed, changed=changed)


class EdgeBootstrapSampler(GraphPerturbationModel):
    """Bootstrap weighted contacts with replacement and aggregate duplicate contacts."""

    name = "edge_bootstrap_sampler"

    def __init__(self, sample_fraction: float = 1.0, *, preserve_total_weight: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self.sample_fraction = float(sample_fraction)
        self.preserve_total_weight = bool(preserve_total_weight)

    def describe(self) -> Dict[str, Any]:
        data = super().describe()
        data.update({"sample_fraction": self.sample_fraction, "preserve_total_weight": self.preserve_total_weight})
        return data

    def sample(self, bundle: GraphBundle, rng: np.random.Generator, *, trial_index: int = 0, seed: Optional[int] = None) -> PerturbedGraph:
        bundle.validate()
        A0 = bundle.A
        n = A0.shape[0]
        edges = upper_edges(A0, positive_only=True)
        m = len(edges)
        A = np.zeros((n, n), dtype=float)
        changed: List[Tuple[int, int, float, float]] = []
        if m == 0:
            return self._finish(bundle, A, trial_index=trial_index, seed=seed, changed=[])
        weights = edge_weights(A0, edges)
        probabilities = weights / max(float(np.sum(weights)), 1e-15)
        draw_count = max(1, int(round(self.sample_fraction * m)))
        draws = rng.choice(np.arange(m), size=draw_count, replace=True, p=probabilities)
        for idx in draws:
            i, j = edges[int(idx)]
            A[i, j] += weights[int(idx)]
            A[j, i] += weights[int(idx)]
        if self.preserve_total_weight:
            old_sum = float(np.sum(np.triu(A0, 1)))
            new_sum = float(np.sum(np.triu(A, 1)))
            if new_sum > 1e-15:
                A *= old_sum / new_sum
        for i, j in edges:
            old = float(0.5 * (A0[i, j] + A0[j, i]))
            new = float(0.5 * (A[i, j] + A[j, i]))
            if abs(new - old) > 1e-15:
                changed.append((i, j, old, new))
        return self._finish(bundle, A, trial_index=trial_index, seed=seed, changed=changed)


class NodeDropout(GraphPerturbationModel):
    """Drop nodes by zeroing all incident contacts while preserving node indexing."""

    name = "node_dropout"

    def __init__(self, dropout_probability: float = 0.02, *, protected_nodes: Optional[Sequence[int]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.dropout_probability = float(dropout_probability)
        self.protected_nodes = set(map(int, protected_nodes or []))

    def describe(self) -> Dict[str, Any]:
        data = super().describe()
        data.update({"dropout_probability": self.dropout_probability, "protected_nodes": sorted(self.protected_nodes)})
        return data

    def sample(self, bundle: GraphBundle, rng: np.random.Generator, *, trial_index: int = 0, seed: Optional[int] = None) -> PerturbedGraph:
        bundle.validate()
        A0 = bundle.A
        A = A0.copy()
        n = A.shape[0]
        dropped: List[int] = []
        for node in range(n):
            if node in self.protected_nodes:
                continue
            if float(rng.random()) < self.dropout_probability:
                dropped.append(node)
        changed: List[Tuple[int, int, float, float]] = []
        for node in dropped:
            neighbors = np.where(A[node] > 0)[0]
            for j in neighbors:
                if node < int(j):
                    old = float(A[node, int(j)])
                    changed.append((node, int(j), old, 0.0))
                elif int(j) < node:
                    old = float(A[int(j), node])
                    changed.append((int(j), node, old, 0.0))
            A[node, :] = 0.0
            A[:, node] = 0.0
        return self._finish(bundle, A, trial_index=trial_index, seed=seed, changed=changed, dropped_nodes=dropped)


class ContactProbabilityThinning(GraphPerturbationModel):
    """Keep contacts according to edge-specific or weight-derived probabilities."""

    name = "contact_probability_thinning"

    def __init__(
        self,
        probability_matrix: Optional[np.ndarray] = None,
        *,
        base_probability: float = 0.9,
        weight_confidence_power: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.probability_matrix = None if probability_matrix is None else _as_square_matrix(probability_matrix, name="probability_matrix")
        self.base_probability = float(base_probability)
        self.weight_confidence_power = float(weight_confidence_power)

    def describe(self) -> Dict[str, Any]:
        data = super().describe()
        data.update(
            {
                "base_probability": self.base_probability,
                "weight_confidence_power": self.weight_confidence_power,
                "has_probability_matrix": self.probability_matrix is not None,
            }
        )
        return data

    def _probabilities(self, A: np.ndarray) -> np.ndarray:
        if self.probability_matrix is not None:
            if self.probability_matrix.shape != A.shape:
                raise ValueError("probability_matrix shape must match adjacency shape")
            return np.clip(self.probability_matrix, 0.0, 1.0)
        if self.weight_confidence_power <= 0:
            return np.full_like(A, np.clip(self.base_probability, 0.0, 1.0), dtype=float)
        positive = A[A > 0]
        scale = float(np.quantile(positive, 0.9)) if positive.size else 1.0
        scale = max(scale, 1e-15)
        p = self.base_probability * np.power(np.clip(A / scale, 0.0, 1.0), self.weight_confidence_power)
        return np.clip(p, 0.0, 1.0)

    def sample(self, bundle: GraphBundle, rng: np.random.Generator, *, trial_index: int = 0, seed: Optional[int] = None) -> PerturbedGraph:
        bundle.validate()
        A0 = bundle.A
        P = self._probabilities(A0)
        A = A0.copy()
        changed: List[Tuple[int, int, float, float]] = []
        for i, j in upper_edges(A0, positive_only=True):
            p = float(0.5 * (P[i, j] + P[j, i]))
            if float(rng.random()) > p:
                old = float(0.5 * (A0[i, j] + A0[j, i]))
                A[i, j] = 0.0
                A[j, i] = 0.0
                changed.append((i, j, old, 0.0))
        return self._finish(bundle, A, trial_index=trial_index, seed=seed, changed=changed)


class DegreeConditionedEdgeNoise(GraphPerturbationModel):
    """Noise model where high-degree residues may have different uncertainty."""

    name = "degree_conditioned_edge_noise"

    def __init__(self, base_std: float = 0.04, degree_exponent: float = -0.5, *, relative: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self.base_std = float(base_std)
        self.degree_exponent = float(degree_exponent)
        self.relative = bool(relative)

    def describe(self) -> Dict[str, Any]:
        data = super().describe()
        data.update({"base_std": self.base_std, "degree_exponent": self.degree_exponent, "relative": self.relative})
        return data

    def sample(self, bundle: GraphBundle, rng: np.random.Generator, *, trial_index: int = 0, seed: Optional[int] = None) -> PerturbedGraph:
        bundle.validate()
        A0 = bundle.A
        A = A0.copy()
        deg = np.maximum(weighted_degrees(A0), 1e-12)
        mean_deg = max(float(np.mean(deg)), 1e-12)
        changed: List[Tuple[int, int, float, float]] = []
        for i, j in upper_edges(A0, positive_only=True):
            old = float(0.5 * (A0[i, j] + A0[j, i]))
            local_degree = math.sqrt(float(deg[i] * deg[j]))
            std = self.base_std * (local_degree / mean_deg) ** self.degree_exponent
            scale = max(abs(old), 1e-12) if self.relative else 1.0
            new = max(0.0, old + float(rng.normal(0.0, std)) * scale)
            A[i, j] = new
            A[j, i] = new
            if abs(new - old) > 1e-15:
                changed.append((i, j, old, new))
        return self._finish(bundle, A, trial_index=trial_index, seed=seed, changed=changed)


class RandomEdgeRewire(GraphPerturbationModel):
    """Rewire a fraction of contacts while preserving the empirical weight distribution."""

    name = "random_edge_rewire"

    def __init__(self, rewire_probability: float = 0.02, *, avoid_existing: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self.rewire_probability = float(rewire_probability)
        self.avoid_existing = bool(avoid_existing)

    def describe(self) -> Dict[str, Any]:
        data = super().describe()
        data.update({"rewire_probability": self.rewire_probability, "avoid_existing": self.avoid_existing})
        return data

    def sample(self, bundle: GraphBundle, rng: np.random.Generator, *, trial_index: int = 0, seed: Optional[int] = None) -> PerturbedGraph:
        bundle.validate()
        A0 = bundle.A
        A = A0.copy()
        n = A.shape[0]
        edges = upper_edges(A0, positive_only=True)
        changed: List[Tuple[int, int, float, float]] = []
        for i, j in edges:
            if float(rng.random()) >= self.rewire_probability:
                continue
            old = float(0.5 * (A[i, j] + A[j, i]))
            A[i, j] = 0.0
            A[j, i] = 0.0
            changed.append((i, j, old, 0.0))
            for _ in range(50):
                u = int(rng.integers(0, n))
                v = int(rng.integers(0, n - 1))
                if v >= u:
                    v += 1
                a, b = (u, v) if u < v else (v, u)
                if not self.avoid_existing or A[a, b] <= 0:
                    A[a, b] = old
                    A[b, a] = old
                    changed.append((a, b, 0.0, old))
                    break
        return self._finish(bundle, A, trial_index=trial_index, seed=seed, changed=changed)


class CompositePerturbation(GraphPerturbationModel):
    """Sequentially apply multiple perturbation models with a shared RNG."""

    name = "composite_perturbation"

    def __init__(self, models: Sequence[GraphPerturbationModel], **kwargs: Any):
        super().__init__(**kwargs)
        if not models:
            raise ValueError("CompositePerturbation requires at least one model")
        self.models = list(models)

    def describe(self) -> Dict[str, Any]:
        data = super().describe()
        data.update({"models": [m.describe() for m in self.models]})
        return data

    def sample(self, bundle: GraphBundle, rng: np.random.Generator, *, trial_index: int = 0, seed: Optional[int] = None) -> PerturbedGraph:
        current = bundle
        all_changed: List[Tuple[int, int, float, float]] = []
        all_dropped: List[int] = []
        sub_meta: List[Dict[str, Any]] = []
        for idx, model in enumerate(self.models):
            pg = model.sample(current, rng, trial_index=trial_index, seed=seed)
            current = pg.bundle
            all_changed.extend(pg.changed_edges)
            all_dropped.extend(pg.dropped_nodes)
            sub_meta.append(pg.to_dict(include_edges=False))
        return self._finish(
            bundle,
            current.A.copy(),
            trial_index=trial_index,
            seed=seed,
            changed=all_changed,
            dropped_nodes=sorted(set(all_dropped)),
            meta={"sub_perturbations": sub_meta},
        )


# ---------------------------------------------------------------------------
# Monte Carlo trial objects and streaming accumulators
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloTrialResult:
    trial_index: int
    seed: Optional[int]
    perturbation_name: str
    state: np.ndarray
    graph_delta_norm: float
    graph_relative_delta_norm: float
    n_changed_edges: int
    n_dropped_nodes: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def score_at(self, node: int) -> float:
        return float(self.state[int(node)])

    def to_dict(self, *, include_state: bool = True) -> Dict[str, Any]:
        data = {
            "trial_index": int(self.trial_index),
            "seed": None if self.seed is None else int(self.seed),
            "perturbation_name": self.perturbation_name,
            "graph_delta_norm": float(self.graph_delta_norm),
            "graph_relative_delta_norm": float(self.graph_relative_delta_norm),
            "n_changed_edges": int(self.n_changed_edges),
            "n_dropped_nodes": int(self.n_dropped_nodes),
            "meta": dict(self.meta),
        }
        if include_state:
            data["state"] = self.state.tolist()
        return data


class RunningMoments:
    """Streaming mean/variance accumulator for vectors."""

    def __init__(self, n: int):
        self.n = int(n)
        self.count = 0
        self.mean = np.zeros(n, dtype=float)
        self.M2 = np.zeros(n, dtype=float)
        self.min = np.full(n, np.inf, dtype=float)
        self.max = np.full(n, -np.inf, dtype=float)

    def update(self, x: np.ndarray) -> None:
        x = _as_signal(x, self.n, name="running sample")
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min = np.minimum(self.min, x)
        self.max = np.maximum(self.max, x)

    @property
    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.zeros(self.n, dtype=float)
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 0.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": int(self.count),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }


class TrialStore:
    """Memory-aware container for trial states and trial metadata."""

    def __init__(self, *, keep_states: bool = True):
        self.keep_states = bool(keep_states)
        self.trials: List[MonteCarloTrialResult] = []
        self._states: List[np.ndarray] = []

    def append(self, result: MonteCarloTrialResult) -> None:
        self.trials.append(result)
        if self.keep_states:
            self._states.append(result.state.copy())

    @property
    def states(self) -> np.ndarray:
        if not self.keep_states:
            raise RuntimeError("TrialStore was configured with keep_states=False")
        if not self._states:
            return np.zeros((0, 0), dtype=float)
        return np.stack(self._states, axis=0)

    def to_trial_table(self) -> List[Dict[str, Any]]:
        return [r.to_dict(include_state=False) for r in self.trials]


# ---------------------------------------------------------------------------
# Ensemble summaries and stability reports
# ---------------------------------------------------------------------------


@dataclass
class NodeUncertaintySummary:
    node: int
    baseline_score: float
    mean: float
    std: float
    ci_low: float
    ci_high: float
    min_score: float
    max_score: float
    rank_baseline: float
    rank_median: float
    mean_abs_rank_shift: float
    rank_entropy: float
    consensus_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankStabilitySummary:
    spearman_to_baseline_mean: float
    spearman_to_baseline_std: float
    kendall_to_baseline_mean: float
    kendall_to_baseline_std: float
    topk_overlap: Dict[int, float]
    pairwise_spearman: Dict[str, float]
    pairwise_kendall: Dict[str, float]
    most_stable_nodes: List[int]
    least_stable_nodes: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spearman_to_baseline_mean": float(self.spearman_to_baseline_mean),
            "spearman_to_baseline_std": float(self.spearman_to_baseline_std),
            "kendall_to_baseline_mean": float(self.kendall_to_baseline_mean),
            "kendall_to_baseline_std": float(self.kendall_to_baseline_std),
            "topk_overlap": {str(k): float(v) for k, v in self.topk_overlap.items()},
            "pairwise_spearman": dict(self.pairwise_spearman),
            "pairwise_kendall": dict(self.pairwise_kendall),
            "most_stable_nodes": list(map(int, self.most_stable_nodes)),
            "least_stable_nodes": list(map(int, self.least_stable_nodes)),
        }


@dataclass
class ConsensusRanking:
    method: str
    ranking: List[int]
    scores: np.ndarray
    median_ranks: np.ndarray
    rank_entropy: np.ndarray

    def top(self, k: int = 10) -> List[Dict[str, Any]]:
        out = []
        for idx, node in enumerate(self.ranking[: int(k)]):
            out.append(
                {
                    "node": int(node),
                    "position": int(idx + 1),
                    "score": float(self.scores[int(node)]),
                    "median_rank": float(self.median_ranks[int(node)]),
                    "rank_entropy": float(self.rank_entropy[int(node)]),
                }
            )
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "ranking": list(map(int, self.ranking)),
            "scores": self.scores.tolist(),
            "median_ranks": self.median_ranks.tolist(),
            "rank_entropy": self.rank_entropy.tolist(),
        }


@dataclass
class FragilityIndexResult:
    baseline_topk: List[int]
    consensus_topk: List[int]
    topk_overlap: float
    mean_rank_shift_topk: float
    score_cv_topk: float
    global_rank_instability: float
    fragility_index: float
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobustnessCurvePoint:
    strength: float
    n_trials: int
    mean_spearman_to_baseline: float
    mean_kendall_to_baseline: float
    topk_overlap: float
    mean_l2_delta: float
    mean_relative_graph_delta: float
    fragility_index: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobustnessReport:
    bundle_name: str
    n_nodes: int
    n_trials: int
    baseline_state: np.ndarray
    ensemble_mean: np.ndarray
    ensemble_std: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    node_summaries: List[NodeUncertaintySummary]
    rank_stability: RankStabilitySummary
    consensus: ConsensusRanking
    fragility: FragilityIndexResult
    trial_table: List[Dict[str, Any]]
    config: Dict[str, Any]
    runtime: Dict[str, Any]

    def to_dict(self, *, include_trial_table: bool = True) -> Dict[str, Any]:
        return {
            "bundle_name": self.bundle_name,
            "n_nodes": int(self.n_nodes),
            "n_trials": int(self.n_trials),
            "baseline_state": self.baseline_state.tolist(),
            "ensemble_mean": self.ensemble_mean.tolist(),
            "ensemble_std": self.ensemble_std.tolist(),
            "ci_low": self.ci_low.tolist(),
            "ci_high": self.ci_high.tolist(),
            "node_summaries": [x.to_dict() for x in self.node_summaries],
            "rank_stability": self.rank_stability.to_dict(),
            "consensus": self.consensus.to_dict(),
            "fragility": self.fragility.to_dict(),
            "trial_table": self.trial_table if include_trial_table else [],
            "config": dict(self.config),
            "runtime": dict(self.runtime),
        }

    def top_nodes(self, k: int = 10, *, by: str = "consensus_score") -> List[Dict[str, Any]]:
        if by == "consensus":
            return self.consensus.top(k)
        rows = [x.to_dict() for x in self.node_summaries]
        if by not in rows[0]:
            raise ValueError(f"unknown node summary field {by!r}")
        rows.sort(key=lambda r: (-float(r[by]), int(r["node"])))
        return rows[: int(k)]

    def write_json(self, path: Union[str, Path], *, include_trial_table: bool = True) -> None:
        write_json(path, self.to_dict(include_trial_table=include_trial_table))

    def write_node_csv(self, path: Union[str, Path]) -> None:
        rows = [x.to_dict() for x in self.node_summaries]
        node_table_to_csv(path, rows)

    def write_trial_csv(self, path: Union[str, Path]) -> None:
        node_table_to_csv(path, self.trial_table)

    def write_markdown(self, path: Union[str, Path], *, top_k: int = 15) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append(f"# RINet Robustness Report: {self.bundle_name}\n")
        lines.append("## Summary\n")
        lines.append(f"- Nodes: {self.n_nodes}")
        lines.append(f"- Monte Carlo trials: {self.n_trials}")
        lines.append(f"- Mean Spearman to baseline: {self.rank_stability.spearman_to_baseline_mean:.4f}")
        lines.append(f"- Mean Kendall to baseline: {self.rank_stability.kendall_to_baseline_mean:.4f}")
        lines.append(f"- Fragility index: {self.fragility.fragility_index:.4f}")
        lines.append(f"- Interpretation: {self.fragility.interpretation}\n")
        lines.append("## Consensus Top Nodes\n")
        lines.append("| Rank | Node | Consensus score | Median rank | Rank entropy |")
        lines.append("|---:|---:|---:|---:|---:|")
        for row in self.consensus.top(top_k):
            lines.append(
                f"| {row['position']} | {row['node']} | {row['score']:.6g} | {row['median_rank']:.3f} | {row['rank_entropy']:.3f} |"
            )
        lines.append("\n## Most Uncertain Nodes\n")
        uncertain = sorted(self.node_summaries, key=lambda x: (-x.std, x.node))[:top_k]
        lines.append("| Node | Baseline | Mean | Std | 95% low | 95% high | Mean abs rank shift |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for x in uncertain:
            lines.append(
                f"| {x.node} | {x.baseline_score:.6g} | {x.mean:.6g} | {x.std:.6g} | {x.ci_low:.6g} | {x.ci_high:.6g} | {x.mean_abs_rank_shift:.3f} |"
            )
        lines.append("\n## Rank Stability\n")
        for k, v in self.rank_stability.topk_overlap.items():
            lines.append(f"- Top-{k} mean overlap with baseline: {v:.4f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_output_dir(self, out_dir: Union[str, Path], *, include_trial_table_json: bool = False) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.write_json(out / "robustness_report.json", include_trial_table=include_trial_table_json)
        self.write_node_csv(out / "node_uncertainty.csv")
        self.write_trial_csv(out / "trial_summary.csv")
        self.write_markdown(out / "robustness_report.md")
        np.savez_compressed(
            out / "robustness_arrays.npz",
            baseline_state=self.baseline_state,
            ensemble_mean=self.ensemble_mean,
            ensemble_std=self.ensemble_std,
            ci_low=self.ci_low,
            ci_high=self.ci_high,
            consensus_scores=self.consensus.scores,
            median_ranks=self.consensus.median_ranks,
            rank_entropy=self.consensus.rank_entropy,
        )


# ---------------------------------------------------------------------------
# Main Monte Carlo runner
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloConfig:
    n_trials: int = 200
    seed: Optional[int] = 0
    ci_alpha: float = 0.05
    keep_states: bool = True
    baseline_seed_nodes: Tuple[int, ...] = (0,)
    baseline_seed_strength: float = 1.0
    topk_values: Tuple[int, ...] = (5, 10, 20)
    kendall_max_pairs: int = 5000

    def validate(self, n_nodes: int) -> None:
        if int(self.n_trials) <= 0:
            raise ValueError("n_trials must be positive")
        if not 0.0 < float(self.ci_alpha) < 1.0:
            raise ValueError("ci_alpha must be in (0, 1)")
        for node in self.baseline_seed_nodes:
            if int(node) < 0 or int(node) >= n_nodes:
                raise ValueError(f"seed node {node} is outside graph with {n_nodes} nodes")


class MonteCarloDiffusionRunner:
    """Run diffusion repeatedly under sampled graph perturbations."""

    def __init__(
        self,
        bundle: GraphBundle,
        perturbation_model: Optional[GraphPerturbationModel] = None,
        *,
        diffusion_config: Optional[DiffusionConfig] = None,
        monte_carlo_config: Optional[MonteCarloConfig] = None,
        seed_state: Optional[np.ndarray] = None,
    ):
        bundle.validate()
        self.bundle = bundle
        self.perturbation_model = perturbation_model or IdentityPerturbation()
        self.diffusion_config = diffusion_config or DiffusionConfig()
        self.monte_carlo_config = monte_carlo_config or MonteCarloConfig()
        self.monte_carlo_config.validate(bundle.n)
        self.kernel = DiffusionKernel(self.diffusion_config)
        self.seed_state = self._make_seed_state(seed_state)

    def _make_seed_state(self, seed_state: Optional[np.ndarray]) -> np.ndarray:
        n = self.bundle.n
        if seed_state is not None:
            return _as_signal(seed_state, n, name="seed_state")
        x = np.zeros(n, dtype=float)
        for node in self.monte_carlo_config.baseline_seed_nodes:
            x[int(node)] = float(self.monte_carlo_config.baseline_seed_strength)
        return x

    def baseline(self) -> DiffusionTrace:
        return self.kernel.run(self.bundle.A, self.seed_state)

    def iter_trials(self) -> Iterator[MonteCarloTrialResult]:
        cfg = self.monte_carlo_config
        base_rng = _rng(cfg.seed)
        for trial_index in range(int(cfg.n_trials)):
            trial_seed = int(base_rng.integers(0, np.iinfo(np.int32).max))
            trial_rng = _rng(trial_seed)
            perturbed = self.perturbation_model.sample(
                self.bundle,
                trial_rng,
                trial_index=trial_index,
                seed=trial_seed,
            )
            trace = self.kernel.run(perturbed.A, self.seed_state)
            yield MonteCarloTrialResult(
                trial_index=trial_index,
                seed=trial_seed,
                perturbation_name=perturbed.perturbation_name,
                state=trace.final_state,
                graph_delta_norm=perturbed.delta_norm(self.bundle.A),
                graph_relative_delta_norm=perturbed.relative_delta_norm(self.bundle.A),
                n_changed_edges=len(perturbed.changed_edges),
                n_dropped_nodes=len(perturbed.dropped_nodes),
                meta={
                    "diffusion": trace.meta,
                    "perturbation": perturbed.to_dict(include_edges=False),
                },
            )

    def run(self) -> RobustnessReport:
        t0 = time.time()
        baseline_trace = self.baseline()
        baseline_state = baseline_trace.final_state
        n = self.bundle.n
        cfg = self.monte_carlo_config
        moments = RunningMoments(n)
        store = TrialStore(keep_states=cfg.keep_states)
        l2_deltas: List[float] = []
        relative_graph_deltas: List[float] = []
        for result in self.iter_trials():
            moments.update(result.state)
            store.append(result)
            l2_deltas.append(float(np.linalg.norm(result.state - baseline_state)))
            relative_graph_deltas.append(float(result.graph_relative_delta_norm))
        if not cfg.keep_states:
            raise RuntimeError("RobustnessReport construction currently requires keep_states=True")
        states = store.states
        ci_low, ci_high = quantile_interval(states, alpha=cfg.ci_alpha, axis=0)
        consensus = build_consensus_ranking(states)
        rank_stability = build_rank_stability_summary(
            states,
            baseline_state,
            topk_values=cfg.topk_values,
            kendall_max_pairs=cfg.kendall_max_pairs,
            seed=cfg.seed,
        )
        node_summaries = build_node_summaries(
            baseline_state=baseline_state,
            states=states,
            mean=moments.mean,
            std=moments.std,
            ci_low=ci_low,
            ci_high=ci_high,
            min_score=moments.min,
            max_score=moments.max,
            consensus=consensus,
        )
        fragility = compute_fragility_index(
            baseline_state,
            states,
            top_k=min(10, n),
            consensus=consensus,
            rank_stability=rank_stability,
        )
        runtime = {
            "seconds": float(time.time() - t0),
            "created_at_unix": float(time.time()),
            "graph_hash": graph_hash(self.bundle.A),
            "seed_state_hash": signal_hash(self.seed_state),
            "mean_l2_delta": float(np.mean(l2_deltas)) if l2_deltas else 0.0,
            "mean_relative_graph_delta": float(np.mean(relative_graph_deltas)) if relative_graph_deltas else 0.0,
        }
        config = {
            "diffusion_config": asdict(self.diffusion_config),
            "monte_carlo_config": asdict(self.monte_carlo_config),
            "perturbation_model": self.perturbation_model.describe(),
            "baseline_diffusion_meta": baseline_trace.meta,
        }
        return RobustnessReport(
            bundle_name=self.bundle.name,
            n_nodes=n,
            n_trials=cfg.n_trials,
            baseline_state=baseline_state,
            ensemble_mean=moments.mean,
            ensemble_std=moments.std,
            ci_low=ci_low,
            ci_high=ci_high,
            node_summaries=node_summaries,
            rank_stability=rank_stability,
            consensus=consensus,
            fragility=fragility,
            trial_table=store.to_trial_table(),
            config=config,
            runtime=runtime,
        )


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------


def build_consensus_ranking(samples: np.ndarray, *, method: str = "borda") -> ConsensusRanking:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must be shape (trials, nodes)")
    if method != "borda":
        raise ValueError("currently only Borda consensus is implemented")
    scores = consensus_borda_score(samples)
    med = median_rank(samples)
    ent = rank_entropy(samples)
    ranking = list(map(int, stable_argsort_desc(scores)))
    return ConsensusRanking(method=method, ranking=ranking, scores=scores, median_ranks=med, rank_entropy=ent)


def build_rank_stability_summary(
    samples: np.ndarray,
    baseline: np.ndarray,
    *,
    topk_values: Sequence[int] = (5, 10, 20),
    kendall_max_pairs: int = 5000,
    seed: Optional[int] = None,
) -> RankStabilitySummary:
    samples = np.asarray(samples, dtype=float)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    spearman_vals = np.array([spearman_correlation(row, baseline) for row in samples], dtype=float)
    kendall_vals = np.array(
        [kendall_tau(row, baseline, max_pairs=kendall_max_pairs, seed=None) for row in samples],
        dtype=float,
    )
    topk: Dict[int, float] = {}
    n = baseline.size
    for k in topk_values:
        kk = max(1, min(int(k), n))
        topk[kk] = float(np.mean([top_k_overlap(row, baseline, kk) for row in samples]))
    pairwise_s = pairwise_correlation_summary(samples, metric="spearman", seed=seed)
    pairwise_k = pairwise_correlation_summary(samples, metric="kendall", seed=seed)
    shifts = mean_abs_rank_shift(samples, baseline)
    stable = list(map(int, np.argsort(shifts)[: min(10, n)]))
    unstable = list(map(int, np.argsort(-shifts)[: min(10, n)]))
    return RankStabilitySummary(
        spearman_to_baseline_mean=float(np.mean(spearman_vals)),
        spearman_to_baseline_std=float(np.std(spearman_vals)),
        kendall_to_baseline_mean=float(np.mean(kendall_vals)),
        kendall_to_baseline_std=float(np.std(kendall_vals)),
        topk_overlap=topk,
        pairwise_spearman=pairwise_s,
        pairwise_kendall=pairwise_k,
        most_stable_nodes=stable,
        least_stable_nodes=unstable,
    )


def build_node_summaries(
    *,
    baseline_state: np.ndarray,
    states: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    min_score: np.ndarray,
    max_score: np.ndarray,
    consensus: ConsensusRanking,
) -> List[NodeUncertaintySummary]:
    n = baseline_state.size
    baseline_ranks = ranks_desc(baseline_state)
    med = consensus.median_ranks
    shifts = mean_abs_rank_shift(states, baseline_state)
    rows = []
    for node in range(n):
        rows.append(
            NodeUncertaintySummary(
                node=node,
                baseline_score=float(baseline_state[node]),
                mean=float(mean[node]),
                std=float(std[node]),
                ci_low=float(ci_low[node]),
                ci_high=float(ci_high[node]),
                min_score=float(min_score[node]),
                max_score=float(max_score[node]),
                rank_baseline=float(baseline_ranks[node]),
                rank_median=float(med[node]),
                mean_abs_rank_shift=float(shifts[node]),
                rank_entropy=float(consensus.rank_entropy[node]),
                consensus_score=float(consensus.scores[node]),
            )
        )
    rows.sort(key=lambda x: (-x.consensus_score, x.node))
    return rows


def compute_fragility_index(
    baseline: np.ndarray,
    samples: np.ndarray,
    *,
    top_k: int = 10,
    consensus: Optional[ConsensusRanking] = None,
    rank_stability: Optional[RankStabilitySummary] = None,
) -> FragilityIndexResult:
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    samples = np.asarray(samples, dtype=float)
    n = baseline.size
    k = max(1, min(int(top_k), n))
    if consensus is None:
        consensus = build_consensus_ranking(samples)
    if rank_stability is None:
        rank_stability = build_rank_stability_summary(samples, baseline, topk_values=(k,))
    baseline_top = list(map(int, top_k_indices(baseline, k)))
    consensus_top = list(map(int, consensus.ranking[:k]))
    overlap = len(set(baseline_top) & set(consensus_top)) / float(k)
    shifts = mean_abs_rank_shift(samples, baseline)
    mean_shift_top = float(np.mean(shifts[baseline_top])) if baseline_top else 0.0
    mean_abs = np.maximum(np.abs(np.mean(samples[:, baseline_top], axis=0)), 1e-12)
    cv_top = float(np.mean(np.std(samples[:, baseline_top], axis=0) / mean_abs)) if baseline_top else 0.0
    global_instability = 1.0 - max(-1.0, min(1.0, rank_stability.spearman_to_baseline_mean))
    normalized_shift = mean_shift_top / max(1.0, n / 2.0)
    raw = 0.35 * (1.0 - overlap) + 0.25 * normalized_shift + 0.20 * min(cv_top, 2.0) / 2.0 + 0.20 * global_instability / 2.0
    fragility = float(max(0.0, min(1.0, raw)))
    if fragility < 0.20:
        interpretation = "highly robust"
    elif fragility < 0.40:
        interpretation = "mostly robust with mild rank instability"
    elif fragility < 0.65:
        interpretation = "moderately fragile; top residues should be interpreted with uncertainty"
    else:
        interpretation = "fragile; predictions are strongly perturbation-dependent"
    return FragilityIndexResult(
        baseline_topk=baseline_top,
        consensus_topk=consensus_top,
        topk_overlap=float(overlap),
        mean_rank_shift_topk=float(mean_shift_top),
        score_cv_topk=float(cv_top),
        global_rank_instability=float(global_instability),
        fragility_index=fragility,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Robustness curves and perturbation sweeps
# ---------------------------------------------------------------------------


def make_noise_model(kind: str, strength: float, *, protected_nodes: Optional[Sequence[int]] = None) -> GraphPerturbationModel:
    kind = str(kind).lower().strip()
    if kind in {"gaussian", "additive", "additive_edge_noise"}:
        return AdditiveEdgeNoise(std=float(strength), relative=True)
    if kind in {"lognormal", "multiplicative", "multiplicative_lognormal"}:
        return MultiplicativeLogNormalEdgeNoise(sigma=float(strength))
    if kind in {"dropout", "edge_dropout"}:
        return EdgeDropout(dropout_probability=float(strength), protect_bridges_heuristic=True)
    if kind in {"node_dropout", "nodes"}:
        return NodeDropout(dropout_probability=float(strength), protected_nodes=protected_nodes)
    if kind in {"rewire", "edge_rewire"}:
        return RandomEdgeRewire(rewire_probability=float(strength))
    if kind in {"degree", "degree_conditioned"}:
        return DegreeConditionedEdgeNoise(base_std=float(strength))
    raise ValueError(f"unknown noise model kind {kind!r}")


def robustness_curve(
    bundle: GraphBundle,
    *,
    strengths: Sequence[float],
    model_kind: str = "gaussian",
    diffusion_config: Optional[DiffusionConfig] = None,
    monte_carlo_config: Optional[MonteCarloConfig] = None,
    seed_state: Optional[np.ndarray] = None,
    top_k: int = 10,
) -> List[RobustnessCurvePoint]:
    bundle.validate()
    base_mc = monte_carlo_config or MonteCarloConfig(n_trials=100, seed=0)
    points: List[RobustnessCurvePoint] = []
    baseline_trace = DiffusionKernel(diffusion_config).run(bundle.A, seed_state if seed_state is not None else _default_seed_state(bundle.n, base_mc))
    baseline = baseline_trace.final_state
    for idx, strength in enumerate(strengths):
        mc = MonteCarloConfig(
            n_trials=base_mc.n_trials,
            seed=None if base_mc.seed is None else int(base_mc.seed) + idx * 1009,
            ci_alpha=base_mc.ci_alpha,
            keep_states=True,
            baseline_seed_nodes=base_mc.baseline_seed_nodes,
            baseline_seed_strength=base_mc.baseline_seed_strength,
            topk_values=base_mc.topk_values,
            kendall_max_pairs=base_mc.kendall_max_pairs,
        )
        runner = MonteCarloDiffusionRunner(
            bundle,
            make_noise_model(model_kind, float(strength), protected_nodes=mc.baseline_seed_nodes),
            diffusion_config=diffusion_config,
            monte_carlo_config=mc,
            seed_state=seed_state,
        )
        report = runner.run()
        state_matrix = _states_from_report(report)
        mean_l2_delta = float(report.runtime.get("mean_l2_delta", float(np.mean(np.linalg.norm(state_matrix - baseline[None, :], axis=1)))))
        mean_rel_graph_delta = float(report.runtime.get("mean_relative_graph_delta", 0.0))
        points.append(
            RobustnessCurvePoint(
                strength=float(strength),
                n_trials=int(mc.n_trials),
                mean_spearman_to_baseline=report.rank_stability.spearman_to_baseline_mean,
                mean_kendall_to_baseline=report.rank_stability.kendall_to_baseline_mean,
                topk_overlap=float(report.rank_stability.topk_overlap.get(min(top_k, bundle.n), report.fragility.topk_overlap)),
                mean_l2_delta=mean_l2_delta,
                mean_relative_graph_delta=mean_rel_graph_delta,
                fragility_index=report.fragility.fragility_index,
            )
        )
    return points


def _default_seed_state(n: int, mc: MonteCarloConfig) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    for node in mc.baseline_seed_nodes:
        if 0 <= int(node) < n:
            x[int(node)] = mc.baseline_seed_strength
    return x


def _states_from_report(report: RobustnessReport) -> np.ndarray:
    # The compact report does not store full states; use trial table only for metadata.
    # This helper returns a rough two-row proxy when full states are not available.
    return np.vstack([report.ensemble_mean, report.ensemble_mean])


def write_robustness_curve(path: Union[str, Path], points: Sequence[RobustnessCurvePoint]) -> None:
    rows = [p.to_dict() for p in points]
    node_table_to_csv(path, rows)


def write_robustness_curve_json(path: Union[str, Path], points: Sequence[RobustnessCurvePoint]) -> None:
    write_json(path, [p.to_dict() for p in points])


# ---------------------------------------------------------------------------
# Consensus ranking and ensemble diffusion helpers
# ---------------------------------------------------------------------------


class EnsembleDiffusion:
    """Convenience wrapper around a matrix of Monte Carlo diffusion states."""

    def __init__(self, states: np.ndarray, *, baseline: Optional[np.ndarray] = None):
        states = np.asarray(states, dtype=float)
        if states.ndim != 2:
            raise ValueError("states must be shape (trials, nodes)")
        self.states = states
        self.baseline = None if baseline is None else _as_signal(baseline, states.shape[1], name="baseline")

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self.states, axis=0)

    @property
    def std(self) -> np.ndarray:
        return np.std(self.states, axis=0, ddof=1) if self.states.shape[0] > 1 else np.zeros(self.states.shape[1])

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        return quantile_interval(self.states, alpha=alpha, axis=0)

    def consensus(self) -> ConsensusRanking:
        return build_consensus_ranking(self.states)

    def rank_stability(self, *, topk_values: Sequence[int] = (5, 10, 20)) -> RankStabilitySummary:
        baseline = self.mean if self.baseline is None else self.baseline
        return build_rank_stability_summary(self.states, baseline, topk_values=topk_values)

    def probability_above(self, threshold: Union[float, np.ndarray]) -> np.ndarray:
        return np.mean(self.states > threshold, axis=0)

    def probability_topk(self, k: int) -> np.ndarray:
        n = self.states.shape[1]
        k = max(1, min(int(k), n))
        counts = np.zeros(n, dtype=float)
        for row in self.states:
            counts[top_k_indices(row, k)] += 1.0
        return counts / self.states.shape[0]

    def credible_topk_table(self, k: int = 10) -> List[Dict[str, Any]]:
        prob = self.probability_topk(k)
        cons = self.consensus()
        rows = []
        for pos, node in enumerate(cons.ranking[:k]):
            rows.append(
                {
                    "rank": int(pos + 1),
                    "node": int(node),
                    "consensus_score": float(cons.scores[node]),
                    "probability_in_topk": float(prob[node]),
                    "mean": float(self.mean[node]),
                    "std": float(self.std[node]),
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Structured report façade
# ---------------------------------------------------------------------------


class ProbabilisticRobustnessLab:
    """High-level façade for graph uncertainty experiments."""

    def __init__(self, bundle: GraphBundle):
        bundle.validate()
        self.bundle = bundle

    def run_monte_carlo(
        self,
        perturbation_model: Optional[GraphPerturbationModel] = None,
        *,
        diffusion_config: Optional[DiffusionConfig] = None,
        monte_carlo_config: Optional[MonteCarloConfig] = None,
        seed_state: Optional[np.ndarray] = None,
    ) -> RobustnessReport:
        runner = MonteCarloDiffusionRunner(
            self.bundle,
            perturbation_model or IdentityPerturbation(),
            diffusion_config=diffusion_config,
            monte_carlo_config=monte_carlo_config,
            seed_state=seed_state,
        )
        return runner.run()

    def edge_noise_report(
        self,
        *,
        std: float = 0.05,
        n_trials: int = 200,
        seed: Optional[int] = 0,
        diffusion_config: Optional[DiffusionConfig] = None,
        seed_state: Optional[np.ndarray] = None,
    ) -> RobustnessReport:
        return self.run_monte_carlo(
            AdditiveEdgeNoise(std=std, relative=True),
            diffusion_config=diffusion_config,
            monte_carlo_config=MonteCarloConfig(n_trials=n_trials, seed=seed),
            seed_state=seed_state,
        )

    def dropout_report(
        self,
        *,
        edge_dropout_probability: float = 0.05,
        n_trials: int = 200,
        seed: Optional[int] = 0,
        diffusion_config: Optional[DiffusionConfig] = None,
        seed_state: Optional[np.ndarray] = None,
    ) -> RobustnessReport:
        return self.run_monte_carlo(
            EdgeDropout(edge_dropout_probability, protect_bridges_heuristic=True),
            diffusion_config=diffusion_config,
            monte_carlo_config=MonteCarloConfig(n_trials=n_trials, seed=seed),
            seed_state=seed_state,
        )

    def composite_uncertainty_report(
        self,
        *,
        edge_noise_std: float = 0.04,
        edge_dropout_probability: float = 0.03,
        node_dropout_probability: float = 0.0,
        n_trials: int = 200,
        seed: Optional[int] = 0,
        diffusion_config: Optional[DiffusionConfig] = None,
        seed_state: Optional[np.ndarray] = None,
    ) -> RobustnessReport:
        models: List[GraphPerturbationModel] = [
            MultiplicativeLogNormalEdgeNoise(sigma=edge_noise_std),
            EdgeDropout(dropout_probability=edge_dropout_probability, protect_bridges_heuristic=True),
        ]
        if node_dropout_probability > 0:
            models.append(NodeDropout(dropout_probability=node_dropout_probability))
        return self.run_monte_carlo(
            CompositePerturbation(models),
            diffusion_config=diffusion_config,
            monte_carlo_config=MonteCarloConfig(n_trials=n_trials, seed=seed),
            seed_state=seed_state,
        )

    def curve(
        self,
        *,
        strengths: Sequence[float],
        model_kind: str = "gaussian",
        n_trials: int = 100,
        seed: Optional[int] = 0,
        diffusion_config: Optional[DiffusionConfig] = None,
        seed_state: Optional[np.ndarray] = None,
        top_k: int = 10,
    ) -> List[RobustnessCurvePoint]:
        return robustness_curve(
            self.bundle,
            strengths=strengths,
            model_kind=model_kind,
            diffusion_config=diffusion_config,
            monte_carlo_config=MonteCarloConfig(n_trials=n_trials, seed=seed, topk_values=(top_k,)),
            seed_state=seed_state,
            top_k=top_k,
        )


# ---------------------------------------------------------------------------
# Export helpers and simple load utilities
# ---------------------------------------------------------------------------


def load_adjacency_csv(path: Union[str, Path], *, delimiter: str = ",") -> GraphBundle:
    path = Path(path)
    A = np.loadtxt(path, delimiter=delimiter)
    A = enforce_symmetric(A)
    return GraphBundle(A=A, meta={"source": str(path)}, name=path.stem)


def save_state_matrix_csv(path: Union[str, Path], states: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(states, dtype=float), delimiter=",")


def report_from_trials(
    bundle: GraphBundle,
    trial_results: Sequence[MonteCarloTrialResult],
    baseline_state: np.ndarray,
    *,
    ci_alpha: float = 0.05,
    topk_values: Sequence[int] = (5, 10, 20),
) -> RobustnessReport:
    states = np.stack([r.state for r in trial_results], axis=0)
    mean = np.mean(states, axis=0)
    std = np.std(states, axis=0, ddof=1) if states.shape[0] > 1 else np.zeros(states.shape[1])
    ci_low, ci_high = quantile_interval(states, alpha=ci_alpha, axis=0)
    consensus = build_consensus_ranking(states)
    rank_stability = build_rank_stability_summary(states, baseline_state, topk_values=topk_values)
    node_summaries = build_node_summaries(
        baseline_state=baseline_state,
        states=states,
        mean=mean,
        std=std,
        ci_low=ci_low,
        ci_high=ci_high,
        min_score=np.min(states, axis=0),
        max_score=np.max(states, axis=0),
        consensus=consensus,
    )
    fragility = compute_fragility_index(baseline_state, states, top_k=min(10, bundle.n), consensus=consensus, rank_stability=rank_stability)
    return RobustnessReport(
        bundle_name=bundle.name,
        n_nodes=bundle.n,
        n_trials=len(trial_results),
        baseline_state=baseline_state,
        ensemble_mean=mean,
        ensemble_std=std,
        ci_low=ci_low,
        ci_high=ci_high,
        node_summaries=node_summaries,
        rank_stability=rank_stability,
        consensus=consensus,
        fragility=fragility,
        trial_table=[r.to_dict(include_state=False) for r in trial_results],
        config={"source": "report_from_trials"},
        runtime={"created_at_unix": float(time.time())},
    )


# ---------------------------------------------------------------------------
# Optional graph-signal integration
# ---------------------------------------------------------------------------


def graph_signal_uncertainty_features(bundle: GraphBundle, report: RobustnessReport) -> Dict[str, Any]:
    """Compute optional signal-processing features for uncertainty fields.

    This uses the graph_signal_processing module if it exists.  The robustness
    module still works without it; this helper simply enriches the report when the
    larger platform is present.
    """

    if GraphSignal is None or GraphSignalProcessor is None:
        return {"available": False, "reason": "graph_signal_processing module not importable"}
    try:
        gsp = GraphSignalProcessor(bundle)
        std_signal = GraphSignal(bundle, report.ensemble_std, name="ensemble_std")
        mean_signal = GraphSignal(bundle, report.ensemble_mean, name="ensemble_mean")
        return {
            "available": True,
            "std_total_variation": float(gsp.total_variation(std_signal)),
            "std_dirichlet_energy": float(gsp.dirichlet_energy(std_signal)),
            "mean_total_variation": float(gsp.total_variation(mean_signal)),
            "mean_dirichlet_energy": float(gsp.dirichlet_energy(mean_signal)),
        }
    except Exception as exc:  # pragma: no cover - optional integration
        return {"available": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def demo(seed: int = 0, n: int = 40, n_trials: int = 50) -> RobustnessReport:
    from .graphio.synth import synthetic_rin

    bundle = synthetic_rin(n=n, seed=seed)
    lab = ProbabilisticRobustnessLab(bundle)
    return lab.composite_uncertainty_report(
        edge_noise_std=0.08,
        edge_dropout_probability=0.04,
        n_trials=n_trials,
        seed=seed,
        diffusion_config=DiffusionConfig(steps=80, dt=0.03, normalize_output="l2"),
    )


def _demo_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a small RINet robustness demo.")
    parser.add_argument("--out", default=None, help="Optional output directory")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = demo(seed=args.seed, n=args.n, n_trials=args.trials)
    print("RINet robustness demo")
    print("nodes:", report.n_nodes)
    print("trials:", report.n_trials)
    print("fragility:", report.fragility.fragility_index, report.fragility.interpretation)
    print("top consensus nodes:")
    for row in report.consensus.top(10):
        print(row)
    if args.out:
        report.write_output_dir(args.out)
        print("wrote", args.out)


__all__ = [
    "AdditiveEdgeNoise",
    "BernoulliDistribution",
    "CompositePerturbation",
    "ConsensusRanking",
    "ConstantDistribution",
    "ContactProbabilityThinning",
    "DegreeConditionedEdgeNoise",
    "DiffusionConfig",
    "DiffusionKernel",
    "DiffusionTrace",
    "EdgeBootstrapSampler",
    "EdgeDropout",
    "EnsembleDiffusion",
    "FragilityIndexResult",
    "GraphPerturbationModel",
    "IdentityPerturbation",
    "LogNormalDistribution",
    "MonteCarloConfig",
    "MonteCarloDiffusionRunner",
    "MonteCarloTrialResult",
    "MultiplicativeLogNormalEdgeNoise",
    "NodeDropout",
    "NodeUncertaintySummary",
    "NormalDistribution",
    "PerturbedGraph",
    "ProbabilisticRobustnessLab",
    "RandomEdgeRewire",
    "RankStabilitySummary",
    "RobustnessCurvePoint",
    "RobustnessReport",
    "ScalarDistribution",
    "UniformDistribution",
    "bootstrap_ci",
    "build_consensus_ranking",
    "build_node_summaries",
    "build_rank_stability_summary",
    "compute_fragility_index",
    "consensus_borda_score",
    "demo",
    "entropy",
    "graph_hash",
    "graph_signal_uncertainty_features",
    "kendall_tau",
    "make_noise_model",
    "rank_entropy",
    "rank_probability_matrix",
    "report_from_trials",
    "robustness_curve",
    "signal_hash",
    "spearman_correlation",
    "top_k_overlap",
    "write_robustness_curve",
    "write_robustness_curve_json",
]


if __name__ == "__main__":  # pragma: no cover
    _demo_main()
