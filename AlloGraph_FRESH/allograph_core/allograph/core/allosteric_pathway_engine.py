"""
Allosteric pathway inference for residue interaction networks.

This module adds a communication-pathway layer to the AlloGraph/RINet core.  It is
intended to sit between low-level graph construction and higher-level biological
interpretation: given source residues, target residues, and a residue interaction
network, it estimates the residues and contacts most likely to mediate allosteric
communication.

The implementation deliberately combines several complementary mathematical
views instead of relying on a single brute-force enumeration of paths:

1. Shortest-path geometry
   Converts contact weights into traversal costs and computes source-target
   geodesics.  This gives an interpretable backbone but can be brittle.

2. Electrical current-flow
   Treats contacts as conductances and solves a Laplacian boundary-value problem.
   Edge currents and node through-currents identify distributed communication
   channels, not only one shortest route.

3. Absorbing Markov bridge flow
   Builds a conditioned random-walk ensemble that starts near sources and reaches
   targets.  This is a Doob-transform style path ensemble: it estimates expected
   edge traffic under target-reaching paths, avoiding naive random walks that
   wander away from the biological endpoint.

4. Heat-kernel diffusion corridors
   Propagates source and target heat profiles and scores residues by the temporal
   overlap of forward and reverse diffusion.  This captures multi-scale corridors.

5. Robust consensus scoring
   Normalizes and aggregates pathway evidence into node/edge scores, a compact
   pathway subgraph, and exportable reports.

The code uses NumPy only by default, with soft hooks to optional modules already
added to the repo, such as `spectral_geometry_engine.py` and
`counterfactual_inference_engine.py`.  No existing file is modified.
"""

from __future__ import annotations

import csv
import heapq
import html
import json
import math
import os
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .graphio.types import GraphBundle
from .math.laplacian import graph_laplacian

try:  # optional: use the spectral module if it has already been added
    from .spectral_geometry_engine import SpectralGeometryEngine  # type: ignore
except Exception:  # pragma: no cover - soft integration hook
    SpectralGeometryEngine = None  # type: ignore


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]
Node = int
Edge = Tuple[int, int]


class AllostericPathwayError(RuntimeError):
    """Base error for pathway analysis failures."""


class GraphValidationError(AllostericPathwayError):
    """Raised when an input graph cannot support pathway analysis."""


class PathwayConfigurationError(AllostericPathwayError):
    """Raised when source/target/pathway configuration is invalid."""


class NumericalPathwayError(AllostericPathwayError):
    """Raised when a numerical solve fails or becomes unstable."""


@dataclass(frozen=True)
class PathwayConfig:
    """Configuration for a full allosteric pathway run.

    Parameters
    ----------
    weight_to_cost:
        Strategy converting adjacency weights into path lengths.  Supported:
        ``"inverse"`` uses ``1 / weight``; ``"negative_log"`` uses
        ``-log(weight/max_weight)``; ``"affinity"`` uses ``1/(eps+weight)`` but
        rescales by median contact strength; ``"unit"`` ignores weights.
    max_paths:
        Number of approximate source-target paths to keep.
    path_diversity_penalty:
        Penalty applied to edges already used by earlier paths during greedy
        path ensemble extraction.  This creates a diverse set of interpretable
        paths without requiring exponential path enumeration.
    current_regularization:
        Small diagonal regularization used when solving Laplacian systems.
    bridge_beta:
        Inverse temperature for the Markov bridge kernel.  Higher values
        concentrate paths around low-cost routes; lower values make corridors
        broader.
    heat_times:
        Diffusion times used for corridor overlap.
    top_nodes:
        Number of residues retained in summary tables.
    top_edges:
        Number of contacts retained in summary tables.
    subgraph_node_fraction:
        Fraction of nodes retained in the recommended pathway subgraph.
    subgraph_min_nodes:
        Minimum number of nodes retained in the recommended pathway subgraph.
    """

    weight_to_cost: str = "inverse"
    max_paths: int = 12
    path_diversity_penalty: float = 0.35
    current_regularization: float = 1e-9
    bridge_beta: float = 1.0
    heat_times: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 8.0)
    top_nodes: int = 25
    top_edges: int = 40
    subgraph_node_fraction: float = 0.20
    subgraph_min_nodes: int = 8
    normalize_scores: bool = True
    include_sources_targets_in_subgraph: bool = True


@dataclass(frozen=True)
class NodeScore:
    """A residue-level pathway score."""

    node: int
    score: float
    current_flow: float = 0.0
    bridge_flow: float = 0.0
    heat_corridor: float = 0.0
    geodesic_frequency: float = 0.0
    degree: float = 0.0
    residue_id: Optional[str] = None
    chain_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EdgeScore:
    """A contact-level pathway score."""

    i: int
    j: int
    score: float
    weight: float
    cost: float
    current: float = 0.0
    bridge_flow: float = 0.0
    geodesic_frequency: float = 0.0

    def edge(self) -> Edge:
        return (int(self.i), int(self.j))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PathwayPath:
    """An interpretable source-target path candidate."""

    nodes: Tuple[int, ...]
    cost: float
    weight_sum: float
    min_weight: float
    mean_weight: float
    source: int
    target: int
    rank: int

    @property
    def length(self) -> int:
        return max(0, len(self.nodes) - 1)

    def edges(self) -> List[Edge]:
        return [(int(a), int(b)) for a, b in zip(self.nodes[:-1], self.nodes[1:])]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["nodes"] = list(self.nodes)
        d["edges"] = [list(e) for e in self.edges()]
        d["length"] = self.length
        return d


@dataclass(frozen=True)
class PathwaySubgraph:
    """Compact subgraph recommended as the source-target communication corridor."""

    nodes: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]
    node_scores: Dict[int, float]
    edge_scores: Dict[str, float]
    source_nodes: Tuple[int, ...]
    target_nodes: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [list(e) for e in self.edges],
            "node_scores": {str(k): float(v) for k, v in self.node_scores.items()},
            "edge_scores": {str(k): float(v) for k, v in self.edge_scores.items()},
            "source_nodes": list(self.source_nodes),
            "target_nodes": list(self.target_nodes),
        }


@dataclass(frozen=True)
class CurrentFlowResult:
    """Electrical current-flow pathway result."""

    potentials: np.ndarray
    edge_current: np.ndarray
    node_through_current: np.ndarray
    injected_current: np.ndarray
    dissipated_power: float
    effective_resistance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "potentials": self.potentials.tolist(),
            "edge_current": self.edge_current.tolist(),
            "node_through_current": self.node_through_current.tolist(),
            "injected_current": self.injected_current.tolist(),
            "dissipated_power": float(self.dissipated_power),
            "effective_resistance": float(self.effective_resistance),
        }


@dataclass(frozen=True)
class MarkovBridgeResult:
    """Conditioned random-walk path-ensemble result."""

    hitting_probability: np.ndarray
    bridge_transition: np.ndarray
    expected_node_visits: np.ndarray
    expected_edge_visits: np.ndarray
    absorption_probability: float
    entropy_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hitting_probability": self.hitting_probability.tolist(),
            "bridge_transition": self.bridge_transition.tolist(),
            "expected_node_visits": self.expected_node_visits.tolist(),
            "expected_edge_visits": self.expected_edge_visits.tolist(),
            "absorption_probability": float(self.absorption_probability),
            "entropy_rate": float(self.entropy_rate),
        }


@dataclass(frozen=True)
class HeatCorridorResult:
    """Multi-scale diffusion-corridor result."""

    times: Tuple[float, ...]
    source_profiles: np.ndarray
    target_profiles: np.ndarray
    node_overlap: np.ndarray
    edge_overlap: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "times": list(self.times),
            "source_profiles": self.source_profiles.tolist(),
            "target_profiles": self.target_profiles.tolist(),
            "node_overlap": self.node_overlap.tolist(),
            "edge_overlap": self.edge_overlap.tolist(),
        }


@dataclass(frozen=True)
class PathwayReport:
    """Structured output from a full allosteric pathway analysis."""

    graph_name: str
    n_nodes: int
    n_edges: int
    source_nodes: Tuple[int, ...]
    target_nodes: Tuple[int, ...]
    config: PathwayConfig
    node_scores: Tuple[NodeScore, ...]
    edge_scores: Tuple[EdgeScore, ...]
    paths: Tuple[PathwayPath, ...]
    subgraph: PathwaySubgraph
    current_flow: Optional[CurrentFlowResult]
    markov_bridge: Optional[MarkovBridgeResult]
    heat_corridor: Optional[HeatCorridorResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def top_nodes(self, k: int = 10) -> Tuple[NodeScore, ...]:
        return self.node_scores[: max(0, int(k))]

    def top_edges(self, k: int = 10) -> Tuple[EdgeScore, ...]:
        return self.edge_scores[: max(0, int(k))]

    def to_dict(self, include_dense_arrays: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "graph_name": self.graph_name,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "source_nodes": list(self.source_nodes),
            "target_nodes": list(self.target_nodes),
            "config": asdict(self.config),
            "node_scores": [x.to_dict() for x in self.node_scores],
            "edge_scores": [x.to_dict() for x in self.edge_scores],
            "paths": [p.to_dict() for p in self.paths],
            "subgraph": self.subgraph.to_dict(),
            "metadata": dict(self.metadata),
        }
        if include_dense_arrays:
            if self.current_flow is not None:
                data["current_flow"] = self.current_flow.to_dict()
            if self.markov_bridge is not None:
                data["markov_bridge"] = self.markov_bridge.to_dict()
            if self.heat_corridor is not None:
                data["heat_corridor"] = self.heat_corridor.to_dict()
        else:
            data["dense_arrays_omitted"] = True
            if self.current_flow is not None:
                data["current_flow_summary"] = {
                    "dissipated_power": self.current_flow.dissipated_power,
                    "effective_resistance": self.current_flow.effective_resistance,
                }
            if self.markov_bridge is not None:
                data["markov_bridge_summary"] = {
                    "absorption_probability": self.markov_bridge.absorption_probability,
                    "entropy_rate": self.markov_bridge.entropy_rate,
                }
            if self.heat_corridor is not None:
                data["heat_corridor_summary"] = {
                    "times": list(self.heat_corridor.times),
                    "max_node_overlap": float(np.max(self.heat_corridor.node_overlap)) if self.heat_corridor.node_overlap.size else 0.0,
                }
        return data

    def write_json(self, path: Union[str, os.PathLike[str]], include_dense_arrays: bool = False) -> pathlib.Path:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(include_dense_arrays=include_dense_arrays), indent=2), encoding="utf-8")
        return p

    def write_node_csv(self, path: Union[str, os.PathLike[str]]) -> pathlib.Path:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "node",
                    "score",
                    "current_flow",
                    "bridge_flow",
                    "heat_corridor",
                    "geodesic_frequency",
                    "degree",
                    "residue_id",
                    "chain_id",
                ],
            )
            writer.writeheader()
            for row in self.node_scores:
                writer.writerow(row.to_dict())
        return p

    def write_edge_csv(self, path: Union[str, os.PathLike[str]]) -> pathlib.Path:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["i", "j", "score", "weight", "cost", "current", "bridge_flow", "geodesic_frequency"],
            )
            writer.writeheader()
            for row in self.edge_scores:
                writer.writerow(row.to_dict())
        return p

    def write_paths_csv(self, path: Union[str, os.PathLike[str]]) -> pathlib.Path:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["rank", "source", "target", "length", "cost", "weight_sum", "min_weight", "mean_weight", "nodes"],
            )
            writer.writeheader()
            for path_obj in self.paths:
                writer.writerow(
                    {
                        "rank": path_obj.rank,
                        "source": path_obj.source,
                        "target": path_obj.target,
                        "length": path_obj.length,
                        "cost": path_obj.cost,
                        "weight_sum": path_obj.weight_sum,
                        "min_weight": path_obj.min_weight,
                        "mean_weight": path_obj.mean_weight,
                        "nodes": " ".join(map(str, path_obj.nodes)),
                    }
                )
        return p

    def write_markdown(self, path: Union[str, os.PathLike[str]]) -> pathlib.Path:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append(f"# Allosteric Pathway Report: {self.graph_name}")
        lines.append("")
        lines.append(f"- Nodes: **{self.n_nodes}**")
        lines.append(f"- Edges: **{self.n_edges}**")
        lines.append(f"- Sources: `{list(self.source_nodes)}`")
        lines.append(f"- Targets: `{list(self.target_nodes)}`")
        if self.current_flow is not None:
            lines.append(f"- Effective resistance: **{self.current_flow.effective_resistance:.6g}**")
            lines.append(f"- Dissipated power: **{self.current_flow.dissipated_power:.6g}**")
        if self.markov_bridge is not None:
            lines.append(f"- Markov bridge entropy rate: **{self.markov_bridge.entropy_rate:.6g}**")
        lines.append("")
        lines.append("## Top residues")
        lines.append("")
        lines.append("| rank | node | score | current | bridge | heat | path freq |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for idx, row in enumerate(self.top_nodes(15), start=1):
            lines.append(
                f"| {idx} | {row.node} | {row.score:.6g} | {row.current_flow:.6g} | "
                f"{row.bridge_flow:.6g} | {row.heat_corridor:.6g} | {row.geodesic_frequency:.6g} |"
            )
        lines.append("")
        lines.append("## Top contacts")
        lines.append("")
        lines.append("| rank | edge | score | weight | current | bridge | path freq |")
        lines.append("|---:|:---|---:|---:|---:|---:|---:|")
        for idx, row in enumerate(self.top_edges(15), start=1):
            lines.append(
                f"| {idx} | ({row.i}, {row.j}) | {row.score:.6g} | {row.weight:.6g} | "
                f"{row.current:.6g} | {row.bridge_flow:.6g} | {row.geodesic_frequency:.6g} |"
            )
        lines.append("")
        lines.append("## Representative paths")
        lines.append("")
        for path_obj in self.paths[:10]:
            node_string = " → ".join(map(str, path_obj.nodes))
            lines.append(f"{path_obj.rank}. `{node_string}`  ")
            lines.append(f"   cost={path_obj.cost:.6g}, min_contact={path_obj.min_weight:.6g}, length={path_obj.length}")
        lines.append("")
        lines.append("## Recommended pathway subgraph")
        lines.append("")
        lines.append(f"Nodes: `{list(self.subgraph.nodes)}`")
        lines.append("")
        lines.append(f"Edges: `{[tuple(e) for e in self.subgraph.edges]}`")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def write_html(self, path: Union[str, os.PathLike[str]]) -> pathlib.Path:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows_nodes = "\n".join(
            f"<tr><td>{idx}</td><td>{r.node}</td><td>{r.score:.6g}</td><td>{r.current_flow:.6g}</td>"
            f"<td>{r.bridge_flow:.6g}</td><td>{r.heat_corridor:.6g}</td></tr>"
            for idx, r in enumerate(self.top_nodes(20), start=1)
        )
        rows_edges = "\n".join(
            f"<tr><td>{idx}</td><td>({r.i}, {r.j})</td><td>{r.score:.6g}</td><td>{r.weight:.6g}</td>"
            f"<td>{r.current:.6g}</td><td>{r.bridge_flow:.6g}</td></tr>"
            for idx, r in enumerate(self.top_edges(20), start=1)
        )
        path_items = "\n".join(
            f"<li><code>{html.escape(' → '.join(map(str, pth.nodes)))}</code> "
            f"<span>cost={pth.cost:.6g}, length={pth.length}</span></li>"
            for pth in self.paths[:12]
        )
        content = f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>Allosteric Pathway Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 2rem; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.45rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: right; }}
h1, h2 {{ margin-top: 1.8rem; }}
code {{ background: #f4f4f4; padding: 0.1rem 0.25rem; border-radius: 0.25rem; }}
.card {{ border: 1px solid #ddd; padding: 1rem; border-radius: 0.75rem; background: #fafafa; }}
</style>
</head>
<body>
<h1>Allosteric Pathway Report: {html.escape(self.graph_name)}</h1>
<div class=\"card\">
<p><b>Nodes:</b> {self.n_nodes} &nbsp; <b>Edges:</b> {self.n_edges}</p>
<p><b>Sources:</b> <code>{html.escape(str(list(self.source_nodes)))}</code></p>
<p><b>Targets:</b> <code>{html.escape(str(list(self.target_nodes)))}</code></p>
</div>
<h2>Top residues</h2>
<table><thead><tr><th>rank</th><th>node</th><th>score</th><th>current</th><th>bridge</th><th>heat</th></tr></thead><tbody>
{rows_nodes}
</tbody></table>
<h2>Top contacts</h2>
<table><thead><tr><th>rank</th><th>edge</th><th>score</th><th>weight</th><th>current</th><th>bridge</th></tr></thead><tbody>
{rows_edges}
</tbody></table>
<h2>Representative paths</h2>
<ol>{path_items}</ol>
<h2>Pathway subgraph</h2>
<p><b>Nodes:</b> <code>{html.escape(str(list(self.subgraph.nodes)))}</code></p>
<p><b>Edges:</b> <code>{html.escape(str([tuple(e) for e in self.subgraph.edges]))}</code></p>
</body>
</html>
"""
        p.write_text(content, encoding="utf-8")
        return p

    def write_output_dir(self, directory: Union[str, os.PathLike[str]], include_dense_arrays: bool = False) -> pathlib.Path:
        out = pathlib.Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.write_json(out / "pathway_report.json", include_dense_arrays=include_dense_arrays)
        self.write_node_csv(out / "node_scores.csv")
        self.write_edge_csv(out / "edge_scores.csv")
        self.write_paths_csv(out / "paths.csv")
        self.write_markdown(out / "pathway_report.md")
        self.write_html(out / "pathway_report.html")
        np.savez_compressed(
            out / "pathway_arrays.npz",
            node_scores=np.array([r.score for r in self.node_scores], dtype=float),
            edge_scores=np.array([[r.i, r.j, r.score] for r in self.edge_scores], dtype=float),
            subgraph_nodes=np.array(self.subgraph.nodes, dtype=int),
            subgraph_edges=np.array(self.subgraph.edges, dtype=int) if self.subgraph.edges else np.zeros((0, 2), dtype=int),
        )
        return out


# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------


def _as_float_matrix(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise GraphValidationError("adjacency matrix must be square")
    if not np.isfinite(A).all():
        raise GraphValidationError("adjacency matrix contains non-finite values")
    A = np.maximum(A, 0.0)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return A


def _edge_count(A: np.ndarray) -> int:
    return int(np.count_nonzero(np.triu(A > 0, 1)))


def _edge_list(A: np.ndarray) -> List[Edge]:
    ii, jj = np.where(np.triu(A > 0, 1))
    return [(int(i), int(j)) for i, j in zip(ii, jj)]


def _node_vector(n: int, nodes: Sequence[int], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    v = np.zeros(n, dtype=float)
    if weights is None:
        weights = [1.0] * len(nodes)
    if len(weights) != len(nodes):
        raise PathwayConfigurationError("weights must match nodes")
    for node, weight in zip(nodes, weights):
        idx = int(node)
        if idx < 0 or idx >= n:
            raise PathwayConfigurationError(f"node {idx} is out of bounds for graph with {n} nodes")
        v[idx] += float(weight)
    return v


def _normalize_distribution(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 0.0)
    s = float(np.sum(v))
    if s <= eps:
        if v.size == 0:
            return v.copy()
        return np.ones_like(v) / float(v.size)
    return v / s


def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def _minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-15:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _sum01(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    s = float(np.sum(x))
    if s <= 1e-15:
        return np.zeros_like(x)
    return x / s


def _rank_desc(values: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(values, dtype=float), kind="mergesort")


def _stable_pinv_solve(L: np.ndarray, b: np.ndarray, regularization: float = 1e-9) -> np.ndarray:
    """Solve a graph Laplacian system with gauge fixing.

    The Laplacian is singular on each connected component.  For pathway current
    flow we only need potential differences, so we fix the last coordinate as a
    reference and solve a reduced system.  If that fails, fall back to a Moore-
    Penrose pseudoinverse.  The right-hand side is centered to enforce zero net
    injection.
    """

    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    n = L.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    b = b - np.mean(b)
    if n == 1:
        return np.zeros(1, dtype=float)
    idx = np.arange(n - 1)
    reduced = L[np.ix_(idx, idx)] + float(regularization) * np.eye(n - 1)
    rhs = b[idx]
    try:
        x_reduced = np.linalg.solve(reduced, rhs)
        x = np.zeros(n, dtype=float)
        x[idx] = x_reduced
        x -= np.mean(x)
        return x
    except np.linalg.LinAlgError:
        try:
            x = np.linalg.pinv(L + float(regularization) * np.eye(n)) @ b
            x -= np.mean(x)
            return np.asarray(x, dtype=float)
        except np.linalg.LinAlgError as exc:
            raise NumericalPathwayError("could not solve Laplacian system") from exc


def _matrix_exponential_symmetric(M: np.ndarray, t: float, x: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.asarray(M, dtype=float))
    coeff = vecs.T @ np.asarray(x, dtype=float)
    return vecs @ (np.exp(float(t) * vals) * coeff)


def _heat_apply_laplacian(L: np.ndarray, t: float, x: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.asarray(L, dtype=float))
    coeff = vecs.T @ np.asarray(x, dtype=float)
    return vecs @ (np.exp(-float(t) * np.maximum(vals, 0.0)) * coeff)


def _row_stochastic(W: np.ndarray) -> np.ndarray:
    W = np.maximum(np.asarray(W, dtype=float), 0.0)
    row_sum = W.sum(axis=1)
    P = np.zeros_like(W, dtype=float)
    good = row_sum > 1e-15
    P[good] = W[good] / row_sum[good, None]
    for i in np.where(~good)[0]:
        P[i, i] = 1.0
    return P


def _connected_components(A: np.ndarray) -> List[List[int]]:
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []
    neighbors = [np.where(A[i] > 0)[0].astype(int).tolist() for i in range(n)]
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in neighbors[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def _ensure_connected_to_targets(A: np.ndarray, sources: Sequence[int], targets: Sequence[int]) -> None:
    comps = _connected_components(A)
    comp_id: Dict[int, int] = {}
    for idx, comp in enumerate(comps):
        for node in comp:
            comp_id[node] = idx
    target_comps = {comp_id[int(t)] for t in targets}
    missing = [int(s) for s in sources if comp_id[int(s)] not in target_comps]
    if missing:
        raise PathwayConfigurationError(
            "some source nodes are not connected to any target node: " + ", ".join(map(str, missing))
        )


def _residue_ids(meta: Mapping[str, Any], n: int) -> List[Optional[str]]:
    for key in ("residue_ids", "residue_id", "resids", "node_ids", "labels"):
        value = meta.get(key)
        if isinstance(value, (list, tuple)) and len(value) == n:
            return [str(x) for x in value]
        if isinstance(value, np.ndarray) and value.shape[0] == n:
            return [str(x) for x in value.tolist()]
    return [None] * n


def _chain_ids(meta: Mapping[str, Any], n: int) -> List[Optional[str]]:
    for key in ("chain_ids", "chains", "chain"):
        value = meta.get(key)
        if isinstance(value, (list, tuple)) and len(value) == n:
            return [str(x) for x in value]
        if isinstance(value, np.ndarray) and value.shape[0] == n:
            return [str(x) for x in value.tolist()]
    return [None] * n


# -----------------------------------------------------------------------------
# Dijkstra and path extraction
# -----------------------------------------------------------------------------


def _dijkstra(cost: np.ndarray, start: int, forbidden_edges: Optional[Set[Edge]] = None, forbidden_nodes: Optional[Set[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    n = cost.shape[0]
    forbidden_edges = forbidden_edges or set()
    forbidden_nodes = forbidden_nodes or set()
    dist = np.full(n, np.inf, dtype=float)
    prev = np.full(n, -1, dtype=int)
    if start in forbidden_nodes:
        return dist, prev
    dist[start] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, int(start))]
    while heap:
        du, u = heapq.heappop(heap)
        if du != dist[u]:
            continue
        neighbors = np.where(np.isfinite(cost[u]) & (cost[u] > 0))[0]
        for v_raw in neighbors:
            v = int(v_raw)
            if v in forbidden_nodes:
                continue
            e = (min(u, v), max(u, v))
            if e in forbidden_edges:
                continue
            nd = du + float(cost[u, v])
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def _reconstruct_path(prev: np.ndarray, start: int, target: int) -> List[int]:
    if start == target:
        return [int(start)]
    out: List[int] = []
    cur = int(target)
    seen: Set[int] = set()
    while cur != -1 and cur not in seen:
        out.append(cur)
        if cur == start:
            return list(reversed(out))
        seen.add(cur)
        cur = int(prev[cur])
    return []


def _path_metrics(A: np.ndarray, cost: np.ndarray, nodes: Sequence[int], rank: int) -> PathwayPath:
    if len(nodes) == 0:
        raise ValueError("empty path")
    weights: List[float] = []
    costs: List[float] = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        weights.append(float(A[int(u), int(v)]))
        costs.append(float(cost[int(u), int(v)]))
    weight_sum = float(np.sum(weights)) if weights else 0.0
    min_weight = float(np.min(weights)) if weights else 0.0
    mean_weight = float(np.mean(weights)) if weights else 0.0
    return PathwayPath(
        nodes=tuple(int(x) for x in nodes),
        cost=float(np.sum(costs)),
        weight_sum=weight_sum,
        min_weight=min_weight,
        mean_weight=mean_weight,
        source=int(nodes[0]),
        target=int(nodes[-1]),
        rank=int(rank),
    )


# -----------------------------------------------------------------------------
# Main engine
# -----------------------------------------------------------------------------


class AllostericPathwayEngine:
    """Infer source-target communication pathways on a residue interaction graph.

    The engine is built to be reusable from CLI tasks, notebooks, the custom
    RINet DSL, or the experiment orchestrator.  All methods return plain
    dataclasses and NumPy arrays so downstream code can serialize or extend them
    without depending on a heavy graph library.
    """

    def __init__(self, bundle: GraphBundle, config: Optional[PathwayConfig] = None):
        self.bundle = bundle
        self.config = config or PathwayConfig()
        self.A = _as_float_matrix(bundle.A)
        self.n = int(self.A.shape[0])
        self.meta: Mapping[str, Any] = bundle.meta or {}
        if self.n == 0:
            raise GraphValidationError("empty graph")
        self.degrees = self.A.sum(axis=1)
        self.edges = _edge_list(self.A)
        self.cost = self.weight_to_cost(self.A, strategy=self.config.weight_to_cost)
        self._residue_ids = _residue_ids(self.meta, self.n)
        self._chain_ids = _chain_ids(self.meta, self.n)

    # ------------------------------------------------------------------
    # Basic graph transformations
    # ------------------------------------------------------------------

    @staticmethod
    def weight_to_cost(A: np.ndarray, strategy: str = "inverse", eps: float = 1e-12) -> np.ndarray:
        A = _as_float_matrix(A)
        cost = np.full_like(A, np.inf, dtype=float)
        mask = A > 0
        if strategy == "inverse":
            cost[mask] = 1.0 / np.maximum(A[mask], eps)
        elif strategy == "negative_log":
            max_w = float(np.max(A[mask])) if np.any(mask) else 1.0
            normalized = np.clip(A[mask] / max(max_w, eps), eps, 1.0)
            cost[mask] = -np.log(normalized) + eps
        elif strategy == "affinity":
            med = float(np.median(A[mask])) if np.any(mask) else 1.0
            cost[mask] = med / np.maximum(A[mask], eps)
        elif strategy == "unit":
            cost[mask] = 1.0
        else:
            raise PathwayConfigurationError(f"unknown weight_to_cost strategy: {strategy!r}")
        np.fill_diagonal(cost, 0.0)
        return cost

    def transition_matrix(self, beta: Optional[float] = None) -> np.ndarray:
        beta_value = self.config.bridge_beta if beta is None else float(beta)
        W = np.zeros_like(self.A, dtype=float)
        mask = self.A > 0
        finite_cost = np.where(np.isfinite(self.cost), self.cost, 0.0)
        W[mask] = self.A[mask] * np.exp(-beta_value * finite_cost[mask])
        return _row_stochastic(W)

    def validate_terminals(self, sources: Sequence[int], targets: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if len(sources) == 0:
            raise PathwayConfigurationError("at least one source node is required")
        if len(targets) == 0:
            raise PathwayConfigurationError("at least one target node is required")
        src = tuple(sorted({int(x) for x in sources}))
        tgt = tuple(sorted({int(x) for x in targets}))
        for node in src + tgt:
            if node < 0 or node >= self.n:
                raise PathwayConfigurationError(f"node {node} is out of bounds for graph with {self.n} nodes")
        overlap = set(src).intersection(tgt)
        if overlap:
            raise PathwayConfigurationError(f"source and target sets overlap: {sorted(overlap)}")
        _ensure_connected_to_targets(self.A, src, tgt)
        return src, tgt

    # ------------------------------------------------------------------
    # Geodesic and path-ensemble methods
    # ------------------------------------------------------------------

    def shortest_path(self, source: int, target: int, cost: Optional[np.ndarray] = None) -> PathwayPath:
        C = self.cost if cost is None else np.asarray(cost, dtype=float)
        dist, prev = _dijkstra(C, int(source))
        nodes = _reconstruct_path(prev, int(source), int(target))
        if not nodes:
            raise PathwayConfigurationError(f"no path from {source} to {target}")
        return _path_metrics(self.A, C, nodes, rank=1)

    def greedy_diverse_paths(
        self,
        sources: Sequence[int],
        targets: Sequence[int],
        max_paths: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
    ) -> Tuple[PathwayPath, ...]:
        """Extract a compact, diverse set of source-target paths.

        This is not an exhaustive k-shortest-path algorithm.  Instead it performs
        repeated shortest-path solves while increasing the cost of already used
        edges.  For pathway interpretation this often gives a better summary:
        paths remain low-cost but do not all collapse onto the same backbone.
        """

        src, tgt = self.validate_terminals(sources, targets)
        k = self.config.max_paths if max_paths is None else int(max_paths)
        penalty = self.config.path_diversity_penalty if diversity_penalty is None else float(diversity_penalty)
        base = self.cost.copy()
        adaptive = base.copy()
        paths: List[PathwayPath] = []
        signatures: Set[Tuple[int, ...]] = set()
        for attempt in range(max(1, k * 4)):
            candidates: List[PathwayPath] = []
            for s in src:
                dist, prev = _dijkstra(adaptive, s)
                for t in tgt:
                    nodes = _reconstruct_path(prev, s, t)
                    if not nodes:
                        continue
                    candidate = _path_metrics(self.A, adaptive, nodes, rank=len(paths) + 1)
                    candidates.append(candidate)
            if not candidates:
                break
            candidates.sort(key=lambda p: (p.cost, p.length, p.source, p.target))
            chosen: Optional[PathwayPath] = None
            for cand in candidates:
                sig = tuple(cand.nodes)
                if sig not in signatures:
                    chosen = cand
                    break
            if chosen is None:
                break
            signatures.add(tuple(chosen.nodes))
            true_path = _path_metrics(self.A, base, chosen.nodes, rank=len(paths) + 1)
            paths.append(true_path)
            for u, v in chosen.edges():
                adaptive[u, v] *= 1.0 + penalty
                adaptive[v, u] *= 1.0 + penalty
            if len(paths) >= k:
                break
        return tuple(paths)

    def geodesic_frequency_scores(self, paths: Sequence[PathwayPath]) -> Tuple[np.ndarray, np.ndarray]:
        node_freq = np.zeros(self.n, dtype=float)
        edge_freq = np.zeros((self.n, self.n), dtype=float)
        if not paths:
            return node_freq, edge_freq
        for p in paths:
            for node in p.nodes:
                node_freq[int(node)] += 1.0
            for u, v in p.edges():
                edge_freq[u, v] += 1.0
                edge_freq[v, u] += 1.0
        node_freq /= float(len(paths))
        edge_freq /= float(len(paths))
        return node_freq, edge_freq

    # ------------------------------------------------------------------
    # Electrical current-flow method
    # ------------------------------------------------------------------

    def current_flow(self, sources: Sequence[int], targets: Sequence[int], source_weights: Optional[Sequence[float]] = None, target_weights: Optional[Sequence[float]] = None) -> CurrentFlowResult:
        src, tgt = self.validate_terminals(sources, targets)
        sdist = _normalize_distribution(_node_vector(self.n, src, source_weights))
        tdist = _normalize_distribution(_node_vector(self.n, tgt, target_weights))
        b = sdist - tdist
        L = graph_laplacian(self.A)
        phi = _stable_pinv_solve(L, b, regularization=self.config.current_regularization)
        raw_current = self.A * (phi[:, None] - phi[None, :])
        abs_current = np.abs(raw_current)
        np.fill_diagonal(abs_current, 0.0)
        node_through = 0.5 * abs_current.sum(axis=1)
        power = float(np.sum(self.A * (phi[:, None] - phi[None, :]) ** 2) / 2.0)
        voltage_drop = float(np.dot(b, phi))
        if abs(voltage_drop) < 1e-15:
            eff_r = 0.0
        else:
            # Unit current injection convention: b sums to zero and L phi = b.
            eff_r = float(voltage_drop)
        return CurrentFlowResult(
            potentials=np.asarray(phi, dtype=float),
            edge_current=abs_current,
            node_through_current=node_through,
            injected_current=b,
            dissipated_power=power,
            effective_resistance=eff_r,
        )

    # ------------------------------------------------------------------
    # Absorbing Markov bridge method
    # ------------------------------------------------------------------

    def markov_bridge_flow(
        self,
        sources: Sequence[int],
        targets: Sequence[int],
        beta: Optional[float] = None,
        source_weights: Optional[Sequence[float]] = None,
    ) -> MarkovBridgeResult:
        """Compute target-conditioned expected traffic using a Doob transform.

        Let ``h_i`` be the probability that a walk starting at node ``i`` reaches
        a target before being killed by numerical leakage.  Conditioning on target
        absorption gives transition probabilities

            P^h_ij = P_ij h_j / h_i.

        The fundamental matrix of the transient conditioned chain gives expected
        visits to nodes and edges before absorption.
        """

        src, tgt = self.validate_terminals(sources, targets)
        target_set = set(tgt)
        source_dist = _normalize_distribution(_node_vector(self.n, src, source_weights))
        P = self.transition_matrix(beta=beta)
        h = np.zeros(self.n, dtype=float)
        h[list(tgt)] = 1.0
        transient = [i for i in range(self.n) if i not in target_set]
        if transient:
            T = np.array(transient, dtype=int)
            Q = P[np.ix_(T, T)]
            R = P[np.ix_(T, np.array(list(tgt), dtype=int))]
            rhs = R @ np.ones(len(tgt), dtype=float)
            try:
                h_T = np.linalg.solve(np.eye(len(T)) - Q + 1e-12 * np.eye(len(T)), rhs)
            except np.linalg.LinAlgError:
                h_T = np.linalg.pinv(np.eye(len(T)) - Q + 1e-12 * np.eye(len(T))) @ rhs
            h[T] = np.clip(h_T, 0.0, 1.0)
        absorption_prob = float(np.dot(source_dist, h))
        eps = 1e-15
        Pb = np.zeros_like(P)
        for i in range(self.n):
            if i in target_set:
                continue
            if h[i] <= eps:
                continue
            Pb[i, :] = P[i, :] * h / h[i]
            row_sum = float(Pb[i, :].sum())
            if row_sum > eps:
                Pb[i, :] /= row_sum
        for t in tgt:
            Pb[t, t] = 1.0
        transient = [i for i in range(self.n) if i not in target_set and h[i] > eps]
        visits = np.zeros(self.n, dtype=float)
        if transient:
            T = np.array(transient, dtype=int)
            Qb = Pb[np.ix_(T, T)]
            alpha = source_dist[T]
            if float(np.sum(alpha)) <= eps:
                alpha = np.zeros_like(alpha)
            try:
                # alpha @ N, where N=(I-Q)^-1, computed as solve on transpose.
                visits_T = np.linalg.solve((np.eye(len(T)) - Qb + 1e-12 * np.eye(len(T))).T, alpha)
            except np.linalg.LinAlgError:
                visits_T = np.linalg.pinv((np.eye(len(T)) - Qb + 1e-12 * np.eye(len(T))).T) @ alpha
            visits[T] = np.maximum(visits_T, 0.0)
        edge_visits = visits[:, None] * Pb
        edge_visits = 0.5 * (edge_visits + edge_visits.T)
        np.fill_diagonal(edge_visits, 0.0)
        entropy_terms = []
        for i in range(self.n):
            probs = Pb[i, Pb[i] > eps]
            if probs.size:
                entropy_terms.append(float(visits[i]) * float(-np.sum(probs * np.log(probs))))
        entropy_rate = float(np.sum(entropy_terms) / max(float(np.sum(visits)), eps))
        return MarkovBridgeResult(
            hitting_probability=h,
            bridge_transition=Pb,
            expected_node_visits=visits,
            expected_edge_visits=edge_visits,
            absorption_probability=absorption_prob,
            entropy_rate=entropy_rate,
        )

    # ------------------------------------------------------------------
    # Heat-kernel corridor method
    # ------------------------------------------------------------------

    def heat_corridor(
        self,
        sources: Sequence[int],
        targets: Sequence[int],
        times: Optional[Sequence[float]] = None,
        source_weights: Optional[Sequence[float]] = None,
        target_weights: Optional[Sequence[float]] = None,
    ) -> HeatCorridorResult:
        src, tgt = self.validate_terminals(sources, targets)
        time_values = tuple(float(t) for t in (times if times is not None else self.config.heat_times))
        if any(t < 0 for t in time_values):
            raise PathwayConfigurationError("heat times must be nonnegative")
        L = graph_laplacian(self.A)
        x0 = _normalize_distribution(_node_vector(self.n, src, source_weights))
        y0 = _normalize_distribution(_node_vector(self.n, tgt, target_weights))
        source_profiles: List[np.ndarray] = []
        target_profiles: List[np.ndarray] = []
        node_overlap = np.zeros(self.n, dtype=float)
        edge_overlap = np.zeros((self.n, self.n), dtype=float)
        for t in time_values:
            xs = _heat_apply_laplacian(L, t, x0)
            yt = _heat_apply_laplacian(L, t, y0)
            xs = np.maximum(xs, 0.0)
            yt = np.maximum(yt, 0.0)
            source_profiles.append(xs)
            target_profiles.append(yt)
            overlap = np.sqrt(np.maximum(xs, 0.0) * np.maximum(yt, 0.0))
            node_overlap += overlap
            edge_overlap += self.A * np.sqrt(np.outer(overlap, overlap))
        if time_values:
            node_overlap /= float(len(time_values))
            edge_overlap /= float(len(time_values))
        np.fill_diagonal(edge_overlap, 0.0)
        return HeatCorridorResult(
            times=time_values,
            source_profiles=np.vstack(source_profiles) if source_profiles else np.zeros((0, self.n), dtype=float),
            target_profiles=np.vstack(target_profiles) if target_profiles else np.zeros((0, self.n), dtype=float),
            node_overlap=node_overlap,
            edge_overlap=edge_overlap,
        )

    # ------------------------------------------------------------------
    # Consensus reports
    # ------------------------------------------------------------------

    def combine_scores(
        self,
        current: Optional[CurrentFlowResult],
        bridge: Optional[MarkovBridgeResult],
        heat: Optional[HeatCorridorResult],
        paths: Sequence[PathwayPath],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        path_node, path_edge = self.geodesic_frequency_scores(paths)
        current_node = current.node_through_current if current is not None else np.zeros(self.n)
        current_edge = current.edge_current if current is not None else np.zeros((self.n, self.n))
        bridge_node = bridge.expected_node_visits if bridge is not None else np.zeros(self.n)
        bridge_edge = bridge.expected_edge_visits if bridge is not None else np.zeros((self.n, self.n))
        heat_node = heat.node_overlap if heat is not None else np.zeros(self.n)
        heat_edge = heat.edge_overlap if heat is not None else np.zeros((self.n, self.n))
        node_parts = {
            "current": _minmax01(current_node),
            "bridge": _minmax01(bridge_node),
            "heat": _minmax01(heat_node),
            "geodesic": _minmax01(path_node),
        }
        edge_parts = {
            "current": _minmax01(current_edge),
            "bridge": _minmax01(bridge_edge),
            "heat": _minmax01(heat_edge),
            "geodesic": _minmax01(path_edge),
        }
        node_score = 0.34 * node_parts["current"] + 0.28 * node_parts["bridge"] + 0.24 * node_parts["heat"] + 0.14 * node_parts["geodesic"]
        edge_score = 0.36 * edge_parts["current"] + 0.30 * edge_parts["bridge"] + 0.20 * edge_parts["heat"] + 0.14 * edge_parts["geodesic"]
        if self.config.normalize_scores:
            node_score = _minmax01(node_score)
            edge_score = _minmax01(edge_score)
        dense_parts = {
            "path_node": path_node,
            "path_edge": path_edge,
            "current_node": current_node,
            "current_edge": current_edge,
            "bridge_node": bridge_node,
            "bridge_edge": bridge_edge,
            "heat_node": heat_node,
            "heat_edge": heat_edge,
        }
        return node_score, edge_score, dense_parts

    def build_node_scores(self, node_score: np.ndarray, dense_parts: Mapping[str, np.ndarray], top: Optional[int] = None) -> Tuple[NodeScore, ...]:
        limit = self.config.top_nodes if top is None else int(top)
        order = _rank_desc(node_score)[: max(0, limit)]
        out: List[NodeScore] = []
        for node in order:
            i = int(node)
            out.append(
                NodeScore(
                    node=i,
                    score=float(node_score[i]),
                    current_flow=float(dense_parts["current_node"][i]),
                    bridge_flow=float(dense_parts["bridge_node"][i]),
                    heat_corridor=float(dense_parts["heat_node"][i]),
                    geodesic_frequency=float(dense_parts["path_node"][i]),
                    degree=float(self.degrees[i]),
                    residue_id=self._residue_ids[i],
                    chain_id=self._chain_ids[i],
                )
            )
        return tuple(out)

    def build_edge_scores(self, edge_score: np.ndarray, dense_parts: Mapping[str, np.ndarray], top: Optional[int] = None) -> Tuple[EdgeScore, ...]:
        limit = self.config.top_edges if top is None else int(top)
        rows: List[EdgeScore] = []
        for i, j in self.edges:
            rows.append(
                EdgeScore(
                    i=i,
                    j=j,
                    score=float(edge_score[i, j]),
                    weight=float(self.A[i, j]),
                    cost=float(self.cost[i, j]),
                    current=float(dense_parts["current_edge"][i, j]),
                    bridge_flow=float(dense_parts["bridge_edge"][i, j]),
                    geodesic_frequency=float(dense_parts["path_edge"][i, j]),
                )
            )
        rows.sort(key=lambda r: (-r.score, -r.weight, r.i, r.j))
        return tuple(rows[: max(0, limit)])

    def build_subgraph(
        self,
        sources: Sequence[int],
        targets: Sequence[int],
        node_score: np.ndarray,
        edge_score: np.ndarray,
        paths: Sequence[PathwayPath],
    ) -> PathwaySubgraph:
        src, tgt = self.validate_terminals(sources, targets)
        k = max(self.config.subgraph_min_nodes, int(math.ceil(self.config.subgraph_node_fraction * self.n)))
        k = min(self.n, max(k, len(src) + len(tgt)))
        selected: Set[int] = set(int(x) for x in _rank_desc(node_score)[:k])
        if self.config.include_sources_targets_in_subgraph:
            selected.update(src)
            selected.update(tgt)
        for p in paths[: min(5, len(paths))]:
            selected.update(p.nodes)
        selected_tuple = tuple(sorted(selected))
        selected_set = set(selected_tuple)
        candidate_edges: List[Tuple[int, int, float]] = []
        for i, j in self.edges:
            if i in selected_set and j in selected_set:
                candidate_edges.append((i, j, float(edge_score[i, j])))
        candidate_edges.sort(key=lambda x: (-x[2], x[0], x[1]))
        max_edges = max(len(selected_tuple) - 1, min(len(candidate_edges), 2 * len(selected_tuple)))
        kept_edges = tuple((int(i), int(j)) for i, j, _ in candidate_edges[:max_edges])
        edge_scores_dict = {f"{i}-{j}": float(edge_score[i, j]) for i, j in kept_edges}
        node_scores_dict = {int(i): float(node_score[i]) for i in selected_tuple}
        return PathwaySubgraph(
            nodes=selected_tuple,
            edges=kept_edges,
            node_scores=node_scores_dict,
            edge_scores=edge_scores_dict,
            source_nodes=src,
            target_nodes=tgt,
        )

    def report(
        self,
        sources: Sequence[int],
        targets: Sequence[int],
        source_weights: Optional[Sequence[float]] = None,
        target_weights: Optional[Sequence[float]] = None,
        run_current: bool = True,
        run_bridge: bool = True,
        run_heat: bool = True,
    ) -> PathwayReport:
        start = time.time()
        src, tgt = self.validate_terminals(sources, targets)
        paths = self.greedy_diverse_paths(src, tgt, max_paths=self.config.max_paths)
        current = self.current_flow(src, tgt, source_weights=source_weights, target_weights=target_weights) if run_current else None
        bridge = self.markov_bridge_flow(src, tgt, source_weights=source_weights) if run_bridge else None
        heat = self.heat_corridor(src, tgt, source_weights=source_weights, target_weights=target_weights) if run_heat else None
        node_score, edge_score, dense_parts = self.combine_scores(current, bridge, heat, paths)
        node_rows = self.build_node_scores(node_score, dense_parts)
        edge_rows = self.build_edge_scores(edge_score, dense_parts)
        subgraph = self.build_subgraph(src, tgt, node_score, edge_score, paths)
        metadata = {
            "created_at_unix": time.time(),
            "runtime_seconds": float(time.time() - start),
            "method": "current_flow + markov_bridge + heat_corridor + greedy_diverse_paths",
            "graph_meta_keys": sorted(str(k) for k in self.meta.keys()),
        }
        return PathwayReport(
            graph_name=str(self.bundle.name),
            n_nodes=self.n,
            n_edges=_edge_count(self.A),
            source_nodes=src,
            target_nodes=tgt,
            config=self.config,
            node_scores=node_rows,
            edge_scores=edge_rows,
            paths=tuple(paths),
            subgraph=subgraph,
            current_flow=current,
            markov_bridge=bridge,
            heat_corridor=heat,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Advanced diagnostics
    # ------------------------------------------------------------------

    def bottleneck_residue_scan(
        self,
        sources: Sequence[int],
        targets: Sequence[int],
        candidate_nodes: Optional[Sequence[int]] = None,
        metric: str = "effective_resistance",
    ) -> List[Dict[str, Any]]:
        """Exact node-ablation scan over candidate pathway residues.

        The scan removes one candidate residue at a time and recomputes a compact
        current-flow metric.  It is intentionally limited to a candidate set; for
        full-graph exhaustive scans the counterfactual module is the better tool.
        """

        src, tgt = self.validate_terminals(sources, targets)
        protected = set(src).union(tgt)
        if candidate_nodes is None:
            base_report = self.report(src, tgt, run_bridge=False, run_heat=False)
            candidate_nodes = [r.node for r in base_report.node_scores if r.node not in protected]
        base_current = self.current_flow(src, tgt)
        if metric == "effective_resistance":
            base_value = base_current.effective_resistance
        elif metric == "dissipated_power":
            base_value = base_current.dissipated_power
        else:
            raise PathwayConfigurationError(f"unknown bottleneck metric: {metric}")
        rows: List[Dict[str, Any]] = []
        for node_raw in candidate_nodes:
            node = int(node_raw)
            if node in protected or node < 0 or node >= self.n:
                continue
            A2 = self.A.copy()
            A2[node, :] = 0.0
            A2[:, node] = 0.0
            bundle = GraphBundle(A=A2, meta=dict(self.meta), name=f"{self.bundle.name}:remove_node_{node}")
            try:
                eng = AllostericPathwayEngine(bundle, self.config)
                cf = eng.current_flow(src, tgt)
                value = cf.effective_resistance if metric == "effective_resistance" else cf.dissipated_power
                delta = float(value - base_value)
                finite = bool(np.isfinite(value))
            except Exception as exc:
                value = float("inf")
                delta = float("inf")
                finite = False
                rows.append({"node": node, "metric": metric, "value": value, "delta": delta, "finite": finite, "error": str(exc)})
                continue
            rows.append({"node": node, "metric": metric, "value": float(value), "delta": delta, "finite": finite})
        rows.sort(key=lambda r: (-float(r["delta"]), int(r["node"])))
        return rows

    def edge_cut_suggestion(
        self,
        sources: Sequence[int],
        targets: Sequence[int],
        max_edges: int = 10,
    ) -> List[Dict[str, Any]]:
        """Suggest contacts whose weakening would most disrupt source-target flow.

        This uses the exact current-flow result rather than brute-force edge
        deletion.  In an electrical network, high-current edges on low-redundancy
        corridors are natural cut candidates.  We combine current magnitude with
        local alternative path redundancy estimated by triangle support.
        """

        current = self.current_flow(sources, targets)
        rows: List[Dict[str, Any]] = []
        A_bool = self.A > 0
        for i, j in self.edges:
            support = int(np.sum(A_bool[i] & A_bool[j]))
            redundancy_penalty = 1.0 / (1.0 + support)
            score = float(current.edge_current[i, j] * redundancy_penalty)
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "score": score,
                    "current": float(current.edge_current[i, j]),
                    "weight": float(self.A[i, j]),
                    "triangle_support": support,
                }
            )
        rows.sort(key=lambda r: (-float(r["score"]), int(r["i"]), int(r["j"])))
        return rows[: max(0, int(max_edges))]

    def pathway_distance_profile(self, sources: Sequence[int], targets: Sequence[int]) -> Dict[str, Any]:
        """Return shortest-distance profiles from source and target sets."""

        src, tgt = self.validate_terminals(sources, targets)
        src_dist = np.full(self.n, np.inf, dtype=float)
        tgt_dist = np.full(self.n, np.inf, dtype=float)
        for s in src:
            d, _ = _dijkstra(self.cost, s)
            src_dist = np.minimum(src_dist, d)
        for t in tgt:
            d, _ = _dijkstra(self.cost, t)
            tgt_dist = np.minimum(tgt_dist, d)
        corridor_distance = src_dist + tgt_dist
        best_total = float(np.nanmin(corridor_distance[np.isfinite(corridor_distance)])) if np.any(np.isfinite(corridor_distance)) else float("inf")
        excess = corridor_distance - best_total
        return {
            "source_distance": src_dist.tolist(),
            "target_distance": tgt_dist.tolist(),
            "corridor_distance": corridor_distance.tolist(),
            "excess_over_geodesic": excess.tolist(),
            "best_source_target_cost": best_total,
        }


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------


def infer_allosteric_pathway(
    bundle: GraphBundle,
    sources: Sequence[int],
    targets: Sequence[int],
    config: Optional[PathwayConfig] = None,
    output_dir: Optional[Union[str, os.PathLike[str]]] = None,
) -> PathwayReport:
    engine = AllostericPathwayEngine(bundle, config=config)
    report = engine.report(sources, targets)
    if output_dir is not None:
        report.write_output_dir(output_dir)
    return report


def pathway_report_from_adjacency(
    A: np.ndarray,
    sources: Sequence[int],
    targets: Sequence[int],
    name: str = "adjacency_graph",
    meta: Optional[Dict[str, Any]] = None,
    config: Optional[PathwayConfig] = None,
) -> PathwayReport:
    bundle = GraphBundle(A=np.asarray(A, dtype=float), meta=meta or {}, name=name)
    return infer_allosteric_pathway(bundle, sources, targets, config=config)


def load_adjacency_csv(path: Union[str, os.PathLike[str]]) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def run_pathway_from_csv(
    csv_path: Union[str, os.PathLike[str]],
    sources: Sequence[int],
    targets: Sequence[int],
    output_dir: Union[str, os.PathLike[str]],
    config: Optional[PathwayConfig] = None,
) -> PathwayReport:
    A = load_adjacency_csv(csv_path)
    bundle = GraphBundle(A=A, meta={"source_csv": str(csv_path)}, name=pathlib.Path(csv_path).stem)
    return infer_allosteric_pathway(bundle, sources, targets, config=config, output_dir=output_dir)


def report_to_network_json(report: PathwayReport) -> Dict[str, Any]:
    """Return a compact node-link JSON object for front-end visualization."""

    node_score_map = {r.node: r for r in report.node_scores}
    edge_score_map = {tuple(sorted((r.i, r.j))): r for r in report.edge_scores}
    nodes = []
    for node in report.subgraph.nodes:
        row = node_score_map.get(int(node))
        nodes.append(
            {
                "id": int(node),
                "score": float(report.subgraph.node_scores.get(int(node), row.score if row else 0.0)),
                "is_source": int(node) in set(report.source_nodes),
                "is_target": int(node) in set(report.target_nodes),
                "residue_id": row.residue_id if row else None,
                "chain_id": row.chain_id if row else None,
            }
        )
    edges = []
    for i, j in report.subgraph.edges:
        row = edge_score_map.get(tuple(sorted((int(i), int(j)))))
        edges.append(
            {
                "source": int(i),
                "target": int(j),
                "score": float(report.subgraph.edge_scores.get(f"{i}-{j}", row.score if row else 0.0)),
                "weight": float(row.weight) if row else 0.0,
            }
        )
    return {"nodes": nodes, "links": edges, "graph_name": report.graph_name}


# -----------------------------------------------------------------------------
# Small CLI for direct module usage
# -----------------------------------------------------------------------------


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _demo() -> None:
    from .graphio.synth import synthetic_rin

    bundle = synthetic_rin(n=70, k=4, seed=4, long_range_prob=0.035)
    engine = AllostericPathwayEngine(bundle)
    report = engine.report([0, 2], [35, 40])
    print("Top residues:")
    for row in report.top_nodes(8):
        print(row)
    print("\nTop contacts:")
    for row in report.top_edges(8):
        print(row)
    print("\nRepresentative paths:")
    for p in report.paths[:5]:
        print(p)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Infer allosteric source-target pathways on a residue interaction network.")
    parser.add_argument("--csv", help="CSV adjacency matrix. If omitted, a synthetic demo graph is used.")
    parser.add_argument("--sources", default="0", help="Comma-separated source nodes, e.g. 0,12")
    parser.add_argument("--targets", default="30", help="Comma-separated target nodes, e.g. 30,41")
    parser.add_argument("--out", default="pathway_output", help="Output directory")
    parser.add_argument("--max-paths", type=int, default=12)
    parser.add_argument("--bridge-beta", type=float, default=1.0)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)

    if args.demo:
        _demo()
        return 0

    config = PathwayConfig(max_paths=args.max_paths, bridge_beta=args.bridge_beta)
    sources = _parse_int_list(args.sources)
    targets = _parse_int_list(args.targets)
    if args.csv:
        report = run_pathway_from_csv(args.csv, sources, targets, args.out, config=config)
    else:
        from .graphio.synth import synthetic_rin

        bundle = synthetic_rin(n=60, k=4, seed=0)
        report = infer_allosteric_pathway(bundle, sources, targets, config=config, output_dir=args.out)
    print(f"Wrote pathway report to {args.out}")
    print("Top residues:", [r.node for r in report.top_nodes(10)])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
