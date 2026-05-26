# experiment_orchestrator.py
"""
Reproducible experiment orchestration for RINet / AlloGraph.

This module turns the lightweight RINet core plus the advanced analysis modules
into a coherent scientific workflow system.  It deliberately avoids heavyweight
workflow engines so it can live inside the existing package, but it borrows the
ideas that make those engines useful:

* declarative JSON/YAML experiment configs
* explicit graph loading and validation
* typed task specifications
* task dependency checks
* artifact registry with SHA-256 hashes
* provenance events and runtime metadata
* structured per-task output directories
* JSON manifest generation
* Markdown and HTML report builders
* graceful fallback when optional advanced modules are missing

The orchestrator is intentionally practical.  It does not pretend to be a full
Snakemake/Airflow replacement; instead, it provides a compact scientific runner
that can execute the analyses that this repository actually exposes: diffusion,
spectral geometry, counterfactual interventions, graph signal processing,
probabilistic robustness, and RINetScript programs.

Example JSON config
-------------------

{
  "name": "demo_allosteric_scan",
  "description": "Full synthetic RINet platform demonstration.",
  "output_dir": "runs/demo_allosteric_scan",
  "graph": {
    "kind": "synthetic",
    "params": {"n": 80, "k": 4, "seed": 0}
  },
  "tasks": [
    {
      "name": "forward_seed_0",
      "kind": "diffusion",
      "params": {"seed_nodes": [0], "steps": 80, "dt": 0.05}
    },
    {
      "name": "spectral_geometry",
      "kind": "spectral",
      "params": {"modes": 30, "heat_times": [0.1, 1.0, 5.0]}
    },
    {
      "name": "counterfactual_node_20",
      "kind": "counterfactual",
      "params": {
        "seed_nodes": [0],
        "target_nodes": [40],
        "interventions": [{"type": "remove_node", "node": 20}]
      }
    },
    {
      "name": "gsp_seed_signal",
      "kind": "gsp",
      "params": {"seed_nodes": [0, 12], "strengths": [1.0, 0.5]}
    },
    {
      "name": "robustness_noise",
      "kind": "robustness",
      "params": {"model": "edge_noise", "std": 0.05, "n_trials": 100}
    }
  ]
}

A minimal command-line entry point is provided at the bottom:

    python -m allograph.core.experiment_orchestrator config.json

Only NumPy is required by the existing package.  YAML support is activated only
when PyYAML is installed.  Optional advanced modules are imported lazily by the
tasks that need them.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import getpass
import hashlib
import html
import inspect
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import time
import traceback
import uuid
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union

import numpy as np

try:  # Optional YAML support.
    import yaml as _yaml  # type: ignore
    _HAVE_YAML = True
except Exception:  # pragma: no cover - environment dependent
    _yaml = None
    _HAVE_YAML = False

try:
    from ..version import __version__ as _ALLOGRAPH_VERSION
except Exception:  # pragma: no cover
    _ALLOGRAPH_VERSION = "unknown"

from .graphio.types import GraphBundle
from .graphio.synth import synthetic_rin
from .graphio.pdb import pdb_to_rin
from .graphio.io import adjacency_from_csv, bundle_from_adjacency, bundle_from_npz, bundle_to_npz
from .inference.forward import run_forward
from .inference.scan import run_scan
from .inference.inverse import run_inverse
from .math.laplacian import graph_laplacian


JsonDict = Dict[str, Any]
PathLike = Union[str, os.PathLike[str]]


# =============================================================================
# Errors
# =============================================================================


class OrchestratorError(RuntimeError):
    """Base class for experiment orchestration errors."""


class ConfigError(OrchestratorError):
    """Raised when an experiment config is invalid."""


class GraphLoadError(OrchestratorError):
    """Raised when the graph declared by a config cannot be loaded."""


class TaskError(OrchestratorError):
    """Raised when a task fails in a controlled way."""


class ArtifactError(OrchestratorError):
    """Raised when an artifact cannot be registered or written."""


class DependencyUnavailableError(TaskError):
    """Raised when a task requires an optional module that is missing."""


# =============================================================================
# General utilities
# =============================================================================


def utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def slugify(value: str, *, fallback: str = "item") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    return text or fallback


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_text(path: PathLike, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def read_text(path: PathLike) -> str:
    return Path(path).read_text(encoding="utf-8")


def stable_json_dumps(obj: Any, *, indent: Optional[int] = None) -> str:
    return json.dumps(json_ready(obj), indent=indent, sort_keys=True, ensure_ascii=False)


def write_json(path: PathLike, obj: Any, *, indent: int = 2) -> Path:
    return write_text(path, stable_json_dumps(obj, indent=indent) + "\n")


def read_json(path: PathLike) -> JsonDict:
    try:
        data = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a JSON object")
    return data


def read_yaml(path: PathLike) -> JsonDict:
    if not _HAVE_YAML:
        raise ConfigError("YAML config requested, but PyYAML is not installed")
    try:
        data = _yaml.safe_load(read_text(path))  # type: ignore[union-attr]
    except Exception as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping/object")
    return data


def load_config_mapping(path: PathLike) -> JsonDict:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return read_yaml(p)
    if suffix == ".json" or suffix == "":
        return read_json(p)
    raise ConfigError(f"unsupported config extension {suffix!r}; use .json, .yaml, or .yml")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_size(path: PathLike) -> int:
    return int(Path(path).stat().st_size)


def relpath(path: PathLike, base: PathLike) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except Exception:
        return str(Path(path))


def graph_hash(bundle_or_A: Union[GraphBundle, np.ndarray]) -> str:
    A = np.asarray(bundle_or_A.A if hasattr(bundle_or_A, "A") else bundle_or_A, dtype=float)
    A = np.ascontiguousarray(A)
    meta = {
        "shape": list(A.shape),
        "dtype": str(A.dtype),
        "sum": float(np.sum(A)) if A.size else 0.0,
    }
    h = hashlib.sha256()
    h.update(stable_json_dumps(meta).encode("utf-8"))
    h.update(A.tobytes())
    return h.hexdigest()


def matrix_summary(A: np.ndarray) -> JsonDict:
    arr = np.asarray(A, dtype=float)
    if arr.size == 0:
        return {"shape": list(arr.shape), "size": 0}
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "nnz": int(np.count_nonzero(arr)),
        "density": float(np.count_nonzero(arr) / arr.size),
    }


def vector_summary(x: np.ndarray, *, top_k: int = 10) -> JsonDict:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return {"length": 0}
    order = np.lexsort((np.arange(arr.size), -np.abs(arr)))[: min(top_k, arr.size)]
    return {
        "length": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "l1_norm": float(np.sum(np.abs(arr))),
        "l2_norm": float(np.linalg.norm(arr)),
        "top_abs": [{"index": int(i), "value": float(arr[i])} for i in order],
    }


def json_ready(obj: Any) -> Any:
    """Convert common scientific objects into JSON-safe structures."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, np.generic):
        return json_ready(obj.item())
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return json_ready(obj.item())
        if obj.size <= 10000:
            return obj.tolist()
        return {"array_summary": matrix_summary(obj) if obj.ndim == 2 else vector_summary(obj)}
    if dataclasses.is_dataclass(obj):
        return json_ready(dataclasses.asdict(obj))
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return json_ready(obj.to_dict())
        except TypeError:
            return json_ready(obj.to_dict(include_matrix=False))
    if isinstance(obj, Mapping):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def write_csv_rows(path: PathLike, rows: Sequence[Mapping[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k)) for k in fieldnames})
    return p


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return stable_json_dumps(json_ready(value))


def save_vector_csv(path: PathLike, values: Sequence[float], *, labels: Optional[Sequence[str]] = None, value_name: str = "value") -> Path:
    arr = np.asarray(values, dtype=float).reshape(-1)
    rows = []
    for i, v in enumerate(arr):
        row = {"node": int(i), value_name: float(v)}
        if labels is not None and i < len(labels):
            row["label"] = labels[i]
        rows.append(row)
    return write_csv_rows(path, rows)


def residue_labels(bundle: GraphBundle) -> List[str]:
    meta = dict(getattr(bundle, "meta", {}) or {})
    for key in ["residue_ids", "residue_labels", "node_labels", "labels", "residues"]:
        vals = meta.get(key)
        if isinstance(vals, (list, tuple)) and len(vals) == bundle.n:
            return [str(v) for v in vals]
    return [str(i) for i in range(bundle.n)]


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def get_git_metadata(cwd: Optional[PathLike] = None) -> JsonDict:
    root = Path(cwd or os.getcwd())
    def run_git(args: Sequence[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(["git", *args], cwd=str(root), stderr=subprocess.DEVNULL, text=True, timeout=3)
            return out.strip()
        except Exception:
            return None
    commit = run_git(["rev-parse", "HEAD"])
    if not commit:
        return {"available": False}
    return {
        "available": True,
        "commit": commit,
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run_git(["status", "--porcelain"])),
        "remote": run_git(["config", "--get", "remote.origin.url"]),
    }


def runtime_metadata() -> JsonDict:
    return {
        "timestamp_utc": utc_now_iso(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
        },
        "packages": {
            "allograph": _ALLOGRAPH_VERSION,
            "numpy": np.__version__,
            "pyyaml": bool(_HAVE_YAML),
        },
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
    }


def make_seed_signal(n: int, seed_nodes: Sequence[int], strengths: Optional[Sequence[float]] = None, *, default_strength: float = 1.0) -> np.ndarray:
    x = np.zeros(int(n), dtype=float)
    if strengths is None:
        strengths = [default_strength for _ in seed_nodes]
    if len(strengths) != len(seed_nodes):
        raise ConfigError("strengths must have the same length as seed_nodes")
    for node, strength in zip(seed_nodes, strengths):
        idx = int(node)
        if idx < 0 or idx >= n:
            raise ConfigError(f"seed node {idx} is outside graph with {n} nodes")
        x[idx] += float(strength)
    return x


def make_target_vector(n: int, target_nodes: Sequence[int], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    y = np.zeros(int(n), dtype=float)
    if weights is None:
        weights = [1.0 for _ in target_nodes]
    if len(weights) != len(target_nodes):
        raise ConfigError("target weights must have the same length as target_nodes")
    for node, weight in zip(target_nodes, weights):
        idx = int(node)
        if idx < 0 or idx >= n:
            raise ConfigError(f"target node {idx} is outside graph with {n} nodes")
        y[idx] += float(weight)
    return y


# =============================================================================
# Config model and validation
# =============================================================================


@dataclass
class GraphSpec:
    kind: str
    path: Optional[str] = None
    params: JsonDict = field(default_factory=dict)
    name: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GraphSpec":
        if not isinstance(data, Mapping):
            raise ConfigError("graph spec must be an object")
        kind = str(data.get("kind", data.get("type", ""))).strip().lower()
        if not kind:
            raise ConfigError("graph.kind is required")
        params = dict(data.get("params", {}) or {})
        path = data.get("path")
        if path is not None:
            path = str(path)
        name = data.get("name")
        return cls(kind=kind, path=path, params=params, name=None if name is None else str(name))

    def to_dict(self) -> JsonDict:
        return {"kind": self.kind, "path": self.path, "params": self.params, "name": self.name}


@dataclass
class TaskSpec:
    name: str
    kind: str
    params: JsonDict = field(default_factory=dict)
    enabled: bool = True
    depends_on: List[str] = field(default_factory=list)
    allow_failure: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], index: int) -> "TaskSpec":
        if not isinstance(data, Mapping):
            raise ConfigError(f"tasks[{index}] must be an object")
        kind = str(data.get("kind", data.get("type", ""))).strip().lower()
        if not kind:
            raise ConfigError(f"tasks[{index}].kind is required")
        name = str(data.get("name", f"{kind}_{index:03d}"))
        params = dict(data.get("params", {}) or {})
        depends = data.get("depends_on", data.get("requires", [])) or []
        if isinstance(depends, str):
            depends = [depends]
        if not isinstance(depends, list):
            raise ConfigError(f"tasks[{index}].depends_on must be a list or string")
        return cls(
            name=name,
            kind=kind,
            params=params,
            enabled=parse_bool(data.get("enabled", True), default=True),
            depends_on=[str(x) for x in depends],
            allow_failure=parse_bool(data.get("allow_failure", False), default=False),
        )

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "kind": self.kind,
            "params": self.params,
            "enabled": self.enabled,
            "depends_on": list(self.depends_on),
            "allow_failure": self.allow_failure,
        }


@dataclass
class ExperimentConfig:
    name: str
    graph: GraphSpec
    tasks: List[TaskSpec]
    output_dir: str = "runs/rinet_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    variables: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)
    strict: bool = True
    fail_fast: bool = True
    created_from: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: Optional[str] = None) -> "ExperimentConfig":
        if not isinstance(data, Mapping):
            raise ConfigError("experiment config must be an object")
        name = str(data.get("name", "rinet_experiment"))
        if "graph" not in data:
            raise ConfigError("experiment config requires a graph object")
        graph = GraphSpec.from_mapping(data["graph"])
        task_data = data.get("tasks", [])
        if not isinstance(task_data, list):
            raise ConfigError("tasks must be a list")
        tasks = [TaskSpec.from_mapping(x, i) for i, x in enumerate(task_data)]
        tags = data.get("tags", []) or []
        if isinstance(tags, str):
            tags = [tags]
        return cls(
            name=name,
            graph=graph,
            tasks=tasks,
            output_dir=str(data.get("output_dir", f"runs/{slugify(name)}")),
            description=str(data.get("description", "")),
            tags=[str(x) for x in tags],
            variables=dict(data.get("variables", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
            strict=parse_bool(data.get("strict", True), default=True),
            fail_fast=parse_bool(data.get("fail_fast", True), default=True),
            created_from=source,
        )

    @classmethod
    def load(cls, path: PathLike) -> "ExperimentConfig":
        p = Path(path)
        return cls.from_mapping(load_config_mapping(p), source=str(p))

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "description": self.description,
            "output_dir": self.output_dir,
            "graph": self.graph.to_dict(),
            "tasks": [t.to_dict() for t in self.tasks],
            "tags": list(self.tags),
            "variables": dict(self.variables),
            "metadata": dict(self.metadata),
            "strict": self.strict,
            "fail_fast": self.fail_fast,
            "created_from": self.created_from,
        }

    def content_hash(self) -> str:
        return sha256_text(stable_json_dumps(self.to_dict()))


class ConfigValidator:
    """Static validation for experiment configs."""

    VALID_GRAPH_KINDS = {"synthetic", "demo", "pdb", "csv", "npz"}
    VALID_TASK_KINDS = {
        "diffusion", "forward", "scan", "inverse",
        "spectral", "spectral_geometry",
        "counterfactual", "intervention",
        "gsp", "graph_signal", "graph_signal_processing",
        "robustness", "monte_carlo",
        "script", "rinet_script", "language",
    }

    def validate(self, config: ExperimentConfig) -> None:
        errors: List[str] = []
        if not config.name.strip():
            errors.append("name cannot be empty")
        if config.graph.kind not in self.VALID_GRAPH_KINDS:
            errors.append(f"unsupported graph kind {config.graph.kind!r}; valid: {sorted(self.VALID_GRAPH_KINDS)}")
        if config.graph.kind in {"pdb", "csv", "npz"} and not config.graph.path:
            errors.append(f"graph.path is required for graph kind {config.graph.kind!r}")
        seen: set[str] = set()
        for task in config.tasks:
            if not task.name.strip():
                errors.append("task name cannot be empty")
            if task.name in seen:
                errors.append(f"duplicate task name {task.name!r}")
            seen.add(task.name)
            if task.kind not in self.VALID_TASK_KINDS:
                errors.append(f"task {task.name!r} has unsupported kind {task.kind!r}")
            for dep in task.depends_on:
                if dep not in seen and dep not in {t.name for t in config.tasks}:
                    errors.append(f"task {task.name!r} depends on unknown task {dep!r}")
        if self._has_dependency_cycle(config.tasks):
            errors.append("task dependency graph contains a cycle")
        if errors:
            raise ConfigError("invalid experiment config:\n" + "\n".join(f"- {e}" for e in errors))

    def _has_dependency_cycle(self, tasks: Sequence[TaskSpec]) -> bool:
        by_name = {t.name: t for t in tasks}
        state: Dict[str, int] = {}

        def visit(name: str) -> bool:
            state[name] = 1
            for dep in by_name[name].depends_on:
                if dep not in by_name:
                    continue
                if state.get(dep) == 1:
                    return True
                if state.get(dep, 0) == 0 and visit(dep):
                    return True
            state[name] = 2
            return False

        for name in by_name:
            if state.get(name, 0) == 0 and visit(name):
                return True
        return False


# =============================================================================
# Artifact and provenance tracking
# =============================================================================


@dataclass
class Artifact:
    id: str
    name: str
    kind: str
    path: Optional[str] = None
    task: Optional[str] = None
    media_type: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: str = field(default_factory=utc_now_iso)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return dataclasses.asdict(self)


class ArtifactRegistry:
    """Append-only artifact registry with content hashes."""

    def __init__(self, root: PathLike):
        self.root = ensure_dir(root)
        self._artifacts: List[Artifact] = []

    @property
    def artifacts(self) -> List[Artifact]:
        return list(self._artifacts)

    def register_file(
        self,
        path: PathLike,
        *,
        name: Optional[str] = None,
        kind: str = "file",
        task: Optional[str] = None,
        media_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Artifact:
        p = Path(path)
        if not p.exists():
            raise ArtifactError(f"cannot register missing file: {p}")
        artifact = Artifact(
            id=f"art_{len(self._artifacts):05d}_{uuid.uuid4().hex[:8]}",
            name=name or p.name,
            kind=kind,
            path=relpath(p, self.root),
            task=task,
            media_type=media_type or guess_media_type(p),
            sha256=sha256_file(p),
            size_bytes=file_size(p),
            metadata=dict(metadata or {}),
        )
        self._artifacts.append(artifact)
        return artifact

    def register_json(self, path: PathLike, obj: Any, *, name: Optional[str] = None, kind: str = "json", task: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> Artifact:
        p = write_json(path, obj)
        return self.register_file(p, name=name, kind=kind, task=task, media_type="application/json", metadata=metadata)

    def register_text(self, path: PathLike, text: str, *, name: Optional[str] = None, kind: str = "text", task: Optional[str] = None, media_type: str = "text/plain", metadata: Optional[Mapping[str, Any]] = None) -> Artifact:
        p = write_text(path, text)
        return self.register_file(p, name=name, kind=kind, task=task, media_type=media_type, metadata=metadata)

    def register_existing_many(self, paths: Iterable[PathLike], *, task: Optional[str] = None, kind: str = "file") -> List[Artifact]:
        out = []
        for p in paths:
            path = Path(p)
            if path.exists() and path.is_file():
                out.append(self.register_file(path, task=task, kind=kind))
        return out

    def to_dict(self) -> JsonDict:
        return {"root": str(self.root), "artifacts": [a.to_dict() for a in self._artifacts]}

    def write_manifest(self, path: PathLike) -> Path:
        return write_json(path, self.to_dict())


def guess_media_type(path: PathLike) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".json": "application/json",
        ".csv": "text/csv",
        ".md": "text/markdown",
        ".html": "text/html",
        ".txt": "text/plain",
        ".npz": "application/octet-stream",
        ".npy": "application/octet-stream",
        ".png": "image/png",
        ".svg": "image/svg+xml",
    }.get(suffix, "application/octet-stream")


@dataclass
class ProvenanceEvent:
    event: str
    timestamp: str = field(default_factory=utc_now_iso)
    task: Optional[str] = None
    status: Optional[str] = None
    duration_seconds: Optional[float] = None
    message: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return dataclasses.asdict(self)


class ProvenanceTracker:
    """Runtime event log for reproducibility and debugging."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.events: List[ProvenanceEvent] = []
        self.runtime = runtime_metadata()
        self.git = get_git_metadata(Path(config.created_from).parent if config.created_from else None)

    def add(self, event: str, *, task: Optional[str] = None, status: Optional[str] = None, duration_seconds: Optional[float] = None, message: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> ProvenanceEvent:
        ev = ProvenanceEvent(
            event=event,
            task=task,
            status=status,
            duration_seconds=None if duration_seconds is None else float(duration_seconds),
            message=message,
            metadata=dict(metadata or {}),
        )
        self.events.append(ev)
        return ev

    def to_dict(self) -> JsonDict:
        return {
            "experiment": {"name": self.config.name, "config_hash": self.config.content_hash()},
            "runtime": self.runtime,
            "git": self.git,
            "events": [e.to_dict() for e in self.events],
        }

    def write(self, path: PathLike) -> Path:
        return write_json(path, self.to_dict())


class ExperimentLogger:
    """Small logger that writes both stdout-friendly messages and a file."""

    def __init__(self, path: PathLike, *, verbose: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._fh = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def log(self, level: str, message: str, *, task: Optional[str] = None) -> None:
        prefix = f"[{utc_now_iso()}] {level.upper():5s}"
        if task:
            prefix += f" [{task}]"
        line = f"{prefix} {message}"
        self._fh.write(line + "\n")
        self._fh.flush()
        if self.verbose:
            print(line)

    def info(self, message: str, *, task: Optional[str] = None) -> None:
        self.log("info", message, task=task)

    def warning(self, message: str, *, task: Optional[str] = None) -> None:
        self.log("warn", message, task=task)

    def error(self, message: str, *, task: Optional[str] = None) -> None:
        self.log("error", message, task=task)


# =============================================================================
# Context and result model
# =============================================================================


@dataclass
class TaskResult:
    name: str
    kind: str
    status: str
    started_at: str
    ended_at: str
    duration_seconds: float
    output_dir: str
    artifacts: List[str] = field(default_factory=list)
    metrics: JsonDict = field(default_factory=dict)
    summary: JsonDict = field(default_factory=dict)
    error: Optional[str] = None
    traceback: Optional[str] = None

    def to_dict(self) -> JsonDict:
        return dataclasses.asdict(self)


@dataclass
class RuntimeContext:
    config: ExperimentConfig
    root_dir: Path
    graph: GraphBundle
    registry: ArtifactRegistry
    provenance: ProvenanceTracker
    logger: ExperimentLogger
    variables: JsonDict = field(default_factory=dict)
    task_results: Dict[str, TaskResult] = field(default_factory=dict)

    def task_dir(self, task: TaskSpec) -> Path:
        return ensure_dir(self.root_dir / "tasks" / slugify(task.name))

    def resolve_path(self, value: PathLike) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p
        if self.config.created_from:
            base = Path(self.config.created_from).resolve().parent
        else:
            base = Path.cwd()
        return (base / p).resolve()

    def record_artifact(self, artifact: Artifact, task_name: Optional[str] = None) -> None:
        if task_name and task_name in self.task_results:
            self.task_results[task_name].artifacts.append(artifact.id)


# =============================================================================
# Graph loading
# =============================================================================


class GraphLoader:
    """Load a graph according to a GraphSpec."""

    def __init__(self, context_path: Optional[PathLike] = None):
        self.context_path = Path(context_path).resolve().parent if context_path else Path.cwd()

    def load(self, spec: GraphSpec) -> GraphBundle:
        kind = spec.kind.lower()
        params = dict(spec.params or {})
        try:
            if kind in {"synthetic", "demo"}:
                bundle = synthetic_rin(**params)
            elif kind == "pdb":
                if not spec.path:
                    raise GraphLoadError("PDB graph requires path")
                path = self._resolve(spec.path)
                bundle = pdb_to_rin(str(path), **params)
            elif kind == "csv":
                if not spec.path:
                    raise GraphLoadError("CSV graph requires path")
                path = self._resolve(spec.path)
                A = adjacency_from_csv(str(path))
                bundle = bundle_from_adjacency(A, name=spec.name or path.stem, meta={"source_path": str(path), **params})
            elif kind == "npz":
                if not spec.path:
                    raise GraphLoadError("NPZ graph requires path")
                path = self._resolve(spec.path)
                bundle = bundle_from_npz(str(path))
            else:
                raise GraphLoadError(f"unsupported graph kind {kind!r}")
        except Exception as exc:
            if isinstance(exc, GraphLoadError):
                raise
            raise GraphLoadError(f"failed loading graph {kind!r}: {exc}") from exc
        if spec.name:
            bundle.name = spec.name
        bundle.validate()
        return bundle

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.context_path / p).resolve()


# =============================================================================
# Task base and registry
# =============================================================================


class ExperimentTask(ABC):
    """Base class for all orchestrated tasks."""

    kind_aliases: Tuple[str, ...] = ()

    def __init__(self, spec: TaskSpec):
        self.spec = spec

    @abstractmethod
    def run(self, context: RuntimeContext, out_dir: Path) -> TaskResult:
        raise NotImplementedError

    def param(self, name: str, default: Any = None) -> Any:
        return self.spec.params.get(name, default)

    def require(self, name: str) -> Any:
        if name not in self.spec.params:
            raise TaskError(f"task {self.spec.name!r} requires parameter {name!r}")
        return self.spec.params[name]

    def seed_vector_from_params(self, bundle: GraphBundle) -> np.ndarray:
        if "seed_state" in self.spec.params:
            arr = np.asarray(self.spec.params["seed_state"], dtype=float).reshape(-1)
            if arr.size != bundle.n:
                raise TaskError(f"seed_state length {arr.size} does not match graph size {bundle.n}")
            return arr
        nodes = self.param("seed_nodes", self.param("seeds", [0]))
        if isinstance(nodes, int):
            nodes = [nodes]
        strengths = self.param("strengths", None)
        return make_seed_signal(bundle.n, [int(x) for x in nodes], strengths=strengths, default_strength=float(self.param("seed_strength", 1.0)))

    def target_vector_from_params(self, bundle: GraphBundle) -> Optional[np.ndarray]:
        nodes = self.param("target_nodes", self.param("targets", None))
        if nodes is None:
            return None
        if isinstance(nodes, int):
            nodes = [nodes]
        weights = self.param("target_weights", None)
        return make_target_vector(bundle.n, [int(x) for x in nodes], weights=weights)

    def result(
        self,
        *,
        status: str,
        started_at: str,
        start_time: float,
        out_dir: Path,
        artifacts: Sequence[Artifact] = (),
        metrics: Optional[Mapping[str, Any]] = None,
        summary: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
        tb: Optional[str] = None,
    ) -> TaskResult:
        ended = utc_now_iso()
        return TaskResult(
            name=self.spec.name,
            kind=self.spec.kind,
            status=status,
            started_at=started_at,
            ended_at=ended,
            duration_seconds=float(time.perf_counter() - start_time),
            output_dir=str(out_dir),
            artifacts=[a.id for a in artifacts],
            metrics=json_ready(metrics or {}),
            summary=json_ready(summary or {}),
            error=error,
            traceback=tb,
        )


class TaskRegistry:
    def __init__(self):
        self._classes: Dict[str, Type[ExperimentTask]] = {}

    def register(self, cls: Type[ExperimentTask]) -> Type[ExperimentTask]:
        aliases = set(getattr(cls, "kind_aliases", ()) or ())
        if not aliases:
            raise ValueError(f"task class {cls.__name__} has no kind aliases")
        for alias in aliases:
            self._classes[alias] = cls
        return cls

    def create(self, spec: TaskSpec) -> ExperimentTask:
        cls = self._classes.get(spec.kind)
        if cls is None:
            raise TaskError(f"unknown task kind {spec.kind!r}")
        return cls(spec)

    def aliases(self) -> List[str]:
        return sorted(self._classes)


TASKS = TaskRegistry()


def register_task(cls: Type[ExperimentTask]) -> Type[ExperimentTask]:
    return TASKS.register(cls)


# =============================================================================
# Concrete tasks
# =============================================================================


@register_task
class DiffusionTask(ExperimentTask):
    kind_aliases = ("diffusion", "forward")

    def run(self, context: RuntimeContext, out_dir: Path) -> TaskResult:
        started, start = utc_now_iso(), time.perf_counter()
        seed_nodes = self.param("seed_nodes", self.param("seeds", [0]))
        if isinstance(seed_nodes, int):
            seed_nodes = [seed_nodes]
        result = run_forward(
            context.graph,
            model=str(self.param("model", "diffusion")),
            seed_nodes=[int(x) for x in seed_nodes],
            steps=int(self.param("steps", 60)),
            dt=float(self.param("dt", 0.10)),
            seed_strength=float(self.param("seed_strength", 1.0)),
        )
        labels = residue_labels(context.graph)
        artifacts: List[Artifact] = []
        artifacts.append(context.registry.register_file(save_vector_csv(out_dir / "state.csv", result.state, labels=labels, value_name="diffusion_score"), task=self.spec.name, kind="diffusion_state"))
        artifacts.append(context.registry.register_json(out_dir / "summary.json", {"meta": result.meta, "state_summary": vector_summary(result.state)}, task=self.spec.name, kind="task_summary"))
        npz_path = out_dir / "state.npz"
        np.savez_compressed(npz_path, state=np.asarray(result.state, dtype=float), labels=np.asarray(labels, dtype=object))
        artifacts.append(context.registry.register_file(npz_path, task=self.spec.name, kind="numpy_archive"))
        return self.result(status="success", started_at=started, start_time=start, out_dir=out_dir, artifacts=artifacts, metrics={"l2_norm": float(np.linalg.norm(result.state)), "sum_abs": float(np.sum(np.abs(result.state)))}, summary={"top_nodes": vector_summary(result.state)["top_abs"]})


@register_task
class ScanTask(ExperimentTask):
    kind_aliases = ("scan",)

    def run(self, context: RuntimeContext, out_dir: Path) -> TaskResult:
        started, start = utc_now_iso(), time.perf_counter()
        result = run_scan(
            context.graph,
            model=str(self.param("model", "diffusion")),
            steps=int(self.param("steps", 60)),
            dt=float(self.param("dt", 0.10)),
        )
        labels = residue_labels(context.graph)
        artifacts = [
            context.registry.register_file(save_vector_csv(out_dir / "scan_scores.csv", result.scores, labels=labels, value_name="scan_score"), task=self.spec.name, kind="scan_scores"),
            context.registry.register_json(out_dir / "summary.json", {"meta": result.meta, "score_summary": vector_summary(result.scores)}, task=self.spec.name, kind="task_summary"),
        ]
        return self.result(status="success", started_at=started, start_time=start, out_dir=out_dir, artifacts=artifacts, metrics={"max_score": float(np.max(result.scores)), "mean_score": float(np.mean(result.scores))}, summary={"top_nodes": vector_summary(result.scores)["top_abs"]})


@register_task
class InverseTask(ExperimentTask):
    kind_aliases = ("inverse",)

    def run(self, context: RuntimeContext, out_dir: Path) -> TaskResult:
        started, start = utc_now_iso(), time.perf_counter()
        observed = self.param("observed", None)
        if observed is None:
            observed = self.seed_vector_from_params(context.graph).tolist()
        result = run_inverse(context.graph, np.asarray(observed, dtype=float))
        labels = residue_labels(context.graph)
        artifacts = [
            context.registry.register_file(save_vector_csv(out_dir / "source_scores.csv", result.source_scores, labels=labels, value_name="source_score"), task=self.spec.name, kind="inverse_scores"),
            context.registry.register_json(out_dir / "summary.json", {"meta": result.meta, "score_summary": vector_summary(result.source_scores)}, task=self.spec.name, kind="task_summary"),
        ]
        return self.result(status="success", started_at=started, start_time=start, out_dir=out_dir, artifacts=artifacts, metrics={"max_score": float(np.max(result.source_scores))}, summary={"top_nodes": vector_summary(result.source_scores)["top_abs"]})


@register_task
class SpectralTask(ExperimentTask):
    kind_aliases = ("spectral", "spectral_geometry")

    def run(self, context: RuntimeContext, out_dir: Path) -> TaskResult:
        started, start = utc_now_iso(), time.perf_counter()
        try:
            from .spectral_geometry_engine import SpectralGeometryEngine
        except Exception as exc:
            raise DependencyUnavailableError("spectral task requires allograph.core.spectral_geometry_engine") from exc
        engine_kwargs = {}
        if "laplacian_kind" in self.spec.params:
            engine_kwargs["laplacian_kind"] = self.param("laplacian_kind")
        engine = SpectralGeometryEngine(context.graph, **engine_kwargs)
        modes = int(self.param("modes", 30))
        heat_times = self.param("heat_times", [0.1, 1.0, 5.0])
        artifacts: List[Artifact] = []
        if hasattr(engine, "save_report_bundle"):
            paths = engine.save_report_bundle(out_dir, modes=modes, heat_times=heat_times)
            artifacts.extend(context.registry.register_existing_many(paths.values(), task=self.spec.name, kind="spectral_artifact"))
            report = engine.report(modes=modes, heat_times=heat_times)
        else:
            report = engine.report(modes=modes, heat_times=heat_times)
            artifacts.append(context.registry.register_json(out_dir / "spectral_report.json", report, task=self.spec.name, kind="spectral_report"))
        summary = json_ready(report)
        metrics = {
            "algebraic_connectivity": summary.get("algebraic_connectivity"),
            "spectral_gap": summary.get("spectral_gap"),
            "spectral_radius": summary.get("spectral_radius"),
        } if isinstance(summary, dict) else {}
        return self.result(status="success", started_at=started, start_time=start, out_dir=out_dir, artifacts=artifacts, metrics=metrics, summary=summary if isinstance(summary, dict) else {"report": summary})


@register_task
class CounterfactualTask(ExperimentTask):
    kind_aliases = ("counterfactual", "intervention")

    def run(self, context: RuntimeContext, out_dir: Path) -> TaskResult:
        started, start = utc_now_iso(), time.perf_counter()
        try:
            from .counterfactual_inference_engine import (
                CounterfactualEngine,
                DiffusionConfig,
                NodeRemovalIntervention,
                EdgeRemovalIntervention,
                EdgeReweightIntervention,
                LocalMutationIntervention,
                PathBlockingIntervention,
                CompositeIntervention,
                FirstOrderSensitivityApproximator,
            )
        except Exception as exc:
            raise DependencyUnavailableError("counterfactual task requires allograph.core.counterfactual_inference_engine") from exc
        x0 = self.seed_vector_from_params(context.graph)
        target = self.target_vector_from_params(context.graph)
        cfg = DiffusionConfig(steps=int(self.param("steps", 80)), dt=float(self.param("dt", 0.05)))
        engine = CounterfactualEngine(context.graph, diffusion_config=cfg, seed_state=x0, target=target)
        specs = self.param("interventions", None)
        artifacts: List[Artifact] = []
        evaluations = []
        if specs:
            interventions = [self._make_intervention(s, NodeRemovalIntervention, EdgeRemovalIntervention, EdgeReweightIntervention, LocalMutationIntervention, PathBlockingIntervention, CompositeIntervention) for s in specs]
            for i, intervention in enumerate(interventions):
                ev = engine.evaluate(intervention, topk=int(self.param("topk", 10)))
                evaluations.append(ev)
                artifacts.append(context.registry.register_json(out_dir / f"evaluation_{i:03d}.json", ev, task=self.spec.name, kind="counterfactual_evaluation"))
            rows = []
            for i, ev in enumerate(evaluations):
                d = json_ready(ev)
                comp = d.get("comparison", {}) if isinstance(d, dict) else {}
                metrics = comp.get("metrics", {}) if isinstance(comp, dict) else {}
                rows.append({"index": i, "intervention_id": d.get("intervention_id") if isinstance(d, dict) else i, **(metrics if isinstance(metrics, dict) else {})})
            artifacts.append(context.registry.register_file(write_csv_rows(out_dir / "evaluation_metrics.csv", rows), task=self.spec.name, kind="counterfactual_metrics"))
        else:
            approximator = FirstOrderSensitivityApproximator(context.graph, diffusion_config=cfg)
            report = approximator.report(x0, target if target is not None else np.ones(context.graph.n), objective_description=str(self.param("objective", "default target response")))
            artifacts.append(context.registry.register_json(out_dir / "sensitivity_report.json", report, task=self.spec.name, kind="sensitivity_report"))
        summary = {"n_evaluations": len(evaluations), "artifacts": [a.name for a in artifacts]}
        return self.result(status="success", started_at=started, start_time=start, out_dir=out_dir, artifacts=artifacts, metrics={"n_evaluations": len(evaluations)}, summary=summary)

    def _make_intervention(self, spec: Mapping[str, Any], NodeRemovalIntervention: Any, EdgeRemovalIntervention: Any, EdgeReweightIntervention: Any, LocalMutationIntervention: Any, PathBlockingIntervention: Any, CompositeIntervention: Any) -> Any:
        if not isinstance(spec, Mapping):
            raise TaskError("each intervention spec must be an object")
        typ = str(spec.get("type", spec.get("kind", ""))).lower()
        if typ in {"remove_node", "node_removal"}:
            return NodeRemovalIntervention(node=int(spec["node"]))
        if typ in {"remove_edge", "edge_removal"}:
            return EdgeRemovalIntervention(i=int(spec.get("i", spec.get("source"))), j=int(spec.get("j", spec.get("target"))))
        if typ in {"reweight_edge", "edge_reweighting", "edge_reweight"}:
            return EdgeReweightIntervention(i=int(spec.get("i", spec.get("source"))), j=int(spec.get("j", spec.get("target"))), new_weight=float(spec.get("new_weight", spec.get("weight"))))
        if typ in {"local_mutation", "mutation"}:
            kwargs = {k: v for k, v in spec.items() if k not in {"type", "kind"}}
            return LocalMutationIntervention(**kwargs)
        if typ in {"path_block", "path_blocking", "block_path"}:
            return PathBlockingIntervention(path=[int(x) for x in spec.get("path", [])], block_strength=float(spec.get("block_strength", spec.get("strength", 0.0))))
        if typ in {"composite", "multi"}:
            children = [self._make_intervention(s, NodeRemovalIntervention, EdgeRemovalIntervention, EdgeReweightIntervention, LocalMutationIntervention, PathBlockingIntervention, CompositeIntervention) for s in spec.get("interventions", [])]
            return CompositeIntervention(children)
        raise TaskError(f"unknown intervention type {typ!r}")


@register_task
class GraphSignalProcessingTask(ExperimentTask):
    kind_aliases = ("gsp", "graph_signal", "graph_signal_processing")

    def run(self, context: RuntimeContext, out_dir: Path) -> TaskResult:
        started, start = utc_now_iso(), time.perf_counter()
        try:
            from .graph_signal_processing import GraphSignal, GraphSignalProcessor
        except Exception as exc:
            raise DependencyUnavailableError("gsp task requires allograph.core.graph_signal_processing") from exc
        x = self.seed_vector_from_params(context.graph)
        processor = GraphSignalProcessor(context.graph, laplacian_kind=str(self.param("laplacian_kind", "combinatorial")))
        signal = GraphSignal(x, bundle=context.graph, name=str(self.param("signal_name", self.spec.name)), metadata={"task": self.spec.name})
        labels = residue_labels(context.graph)
        artifacts: List[Artifact] = []
        if hasattr(processor, "write_full_report"):
            p = processor.write_full_report(signal, out_dir, labels=labels)
            artifacts.extend(context.registry.register_existing_many(Path(p).glob("*"), task=self.spec.name, kind="gsp_artifact"))
        else:
            artifacts.append(context.registry.register_file(save_vector_csv(out_dir / "signal.csv", x, labels=labels, value_name="signal"), task=self.spec.name, kind="graph_signal"))
            summary = {"signal_summary": vector_summary(x), "energy": float(x @ graph_laplacian(context.graph.A) @ x)}
            artifacts.append(context.registry.register_json(out_dir / "gsp_report.json", summary, task=self.spec.name, kind="gsp_report"))
        metrics = {"dirichlet_energy": float(x @ graph_laplacian(context.graph.A) @ x), "signal_l2": float(np.linalg.norm(x))}
        return self.result(status="success", started_at=started, start_time=start, out_dir=out_dir, artifacts=artifacts, metrics=metrics, summary={"signal": vector_summary(x)})


@register_task
class RobustnessTask(ExperimentTask):
    kind_aliases = ("robustness", "monte_carlo")

    def run(self, context: RuntimeContext, out_dir: Path) -> TaskResult:
        started, start = utc_now_iso(), time.perf_counter()
        try:
            from .probabilistic_robustness_lab import (
                ProbabilisticRobustnessLab,
                AdditiveEdgeNoise,
                EdgeDropout,
                NodeDropout,
                EdgeBootstrapSampler,
                DiffusionConfig,
                MonteCarloConfig,
            )
        except Exception as exc:
            raise DependencyUnavailableError("robustness task requires allograph.core.probabilistic_robustness_lab") from exc
        lab = ProbabilisticRobustnessLab(context.graph)
        model_name = str(self.param("model", self.param("perturbation", "edge_noise"))).lower()
        if model_name in {"edge_noise", "noise", "additive"}:
            model = AdditiveEdgeNoise(std=float(self.param("std", 0.05)), relative=parse_bool(self.param("relative", True), default=True))
        elif model_name in {"edge_dropout", "dropout"}:
            model = EdgeDropout(float(self.param("p", self.param("probability", 0.05))))
        elif model_name in {"node_dropout"}:
            model = NodeDropout(float(self.param("p", self.param("probability", 0.05))))
        elif model_name in {"bootstrap", "bootstrap_edges"}:
            model = EdgeBootstrapSampler()
        else:
            raise TaskError(f"unknown robustness perturbation model {model_name!r}")
        diffusion_config = DiffusionConfig(steps=int(self.param("steps", 80)), dt=float(self.param("dt", 0.03)), normalize_output=str(self.param("normalize_output", "none")))
        mc_config = MonteCarloConfig(n_trials=int(self.param("n_trials", 100)), seed=self.param("seed", 0))
        seed_state = self.seed_vector_from_params(context.graph)
        report = lab.run_monte_carlo(model, diffusion_config=diffusion_config, monte_carlo_config=mc_config, seed_state=seed_state)
        artifacts: List[Artifact] = []
        if hasattr(report, "write_output_dir"):
            report.write_output_dir(out_dir)
            artifacts.extend(context.registry.register_existing_many(Path(out_dir).glob("*"), task=self.spec.name, kind="robustness_artifact"))
        else:
            artifacts.append(context.registry.register_json(out_dir / "robustness_report.json", report, task=self.spec.name, kind="robustness_report"))
        ready = json_ready(report)
        metrics: JsonDict = {}
        if isinstance(ready, dict):
            frag = ready.get("fragility", {})
            if isinstance(frag, dict):
                metrics["fragility_index"] = frag.get("fragility_index")
        return self.result(status="success", started_at=started, start_time=start, out_dir=out_dir, artifacts=artifacts, metrics=metrics, summary=ready if isinstance(ready, dict) else {"report": ready})


@register_task
class ScriptTask(ExperimentTask):
    kind_aliases = ("script", "rinet_script", "language")

    def run(self, context: RuntimeContext, out_dir: Path) -> TaskResult:
        started, start = utc_now_iso(), time.perf_counter()
        try:
            from .rinet_language import RINetRuntimeHooks, RINetInterpreter, RINetParser, RINetLexer
        except Exception as exc:
            raise DependencyUnavailableError("script task requires allograph.core.rinet_language") from exc
        source = self.param("source", None)
        if source is None and self.param("path", None):
            source = read_text(context.resolve_path(str(self.param("path"))))
        if source is None:
            raise TaskError("script task requires source or path")
        script_path = write_text(out_dir / "script.rinet", str(source))
        hooks = RINetRuntimeHooks()
        interpreter = RINetInterpreter(hooks=hooks)
        lexer = RINetLexer(str(source))
        tokens = lexer.lex()
        parser = RINetParser(tokens, str(source))
        program = parser.parse()
        value = interpreter.execute(program)
        artifacts = [
            context.registry.register_file(script_path, task=self.spec.name, kind="rinet_script"),
            context.registry.register_json(out_dir / "script_result.json", {"result": json_ready(value), "symbols": json_ready(interpreter.environment.snapshot())}, task=self.spec.name, kind="script_result"),
        ]
        return self.result(status="success", started_at=started, start_time=start, out_dir=out_dir, artifacts=artifacts, metrics={"token_count": len(tokens)}, summary={"result": json_ready(value)})


# =============================================================================
# Report builders
# =============================================================================


class MarkdownReportBuilder:
    def build(self, config: ExperimentConfig, graph: GraphBundle, results: Sequence[TaskResult], registry: ArtifactRegistry, provenance: ProvenanceTracker) -> str:
        lines: List[str] = []
        lines.append(f"# RINet Experiment Report: {config.name}")
        lines.append("")
        if config.description:
            lines.append(config.description)
            lines.append("")
        lines.append("## Experiment summary")
        lines.append("")
        lines.append(f"- Created: {utc_now_iso()}")
        lines.append(f"- Config hash: `{config.content_hash()}`")
        lines.append(f"- Graph: `{graph.name}`")
        lines.append(f"- Nodes: {graph.n}")
        A = np.asarray(graph.A, dtype=float)
        edges = int(np.count_nonzero(np.triu(A, 1)))
        lines.append(f"- Undirected edges: {edges}")
        lines.append(f"- Graph hash: `{graph_hash(graph)}`")
        if config.tags:
            lines.append(f"- Tags: {', '.join(config.tags)}")
        lines.append("")
        lines.append("## Tasks")
        lines.append("")
        lines.append("| task | kind | status | seconds | artifacts |")
        lines.append("|:---|:---|:---|---:|---:|")
        for r in results:
            lines.append(f"| `{r.name}` | `{r.kind}` | {r.status} | {r.duration_seconds:.3f} | {len(r.artifacts)} |")
        lines.append("")
        for r in results:
            lines.append(f"### {r.name}")
            lines.append("")
            lines.append(f"- Kind: `{r.kind}`")
            lines.append(f"- Status: **{r.status}**")
            lines.append(f"- Runtime: {r.duration_seconds:.3f} seconds")
            if r.error:
                lines.append(f"- Error: `{r.error}`")
            if r.metrics:
                lines.append("")
                lines.append("Metrics:")
                for k, v in r.metrics.items():
                    lines.append(f"- `{k}`: {v}")
            lines.append("")
        lines.append("## Artifacts")
        lines.append("")
        lines.append("| id | task | kind | path | size | sha256 |")
        lines.append("|:---|:---|:---|:---|---:|:---|")
        for a in registry.artifacts:
            digest = (a.sha256 or "")[:12]
            lines.append(f"| `{a.id}` | `{a.task or ''}` | `{a.kind}` | `{a.path or ''}` | {a.size_bytes or 0} | `{digest}` |")
        lines.append("")
        lines.append("## Provenance")
        lines.append("")
        rt = provenance.runtime
        lines.append(f"- Python: `{rt.get('python', {}).get('version', '').split()[0]}`")
        lines.append(f"- NumPy: `{rt.get('packages', {}).get('numpy')}`")
        git = provenance.git
        if git.get("available"):
            lines.append(f"- Git commit: `{git.get('commit')}`")
            lines.append(f"- Git branch: `{git.get('branch')}`")
            lines.append(f"- Git dirty: `{git.get('dirty')}`")
        else:
            lines.append("- Git metadata: unavailable")
        lines.append("")
        return "\n".join(lines)


class HtmlReportBuilder:
    def build(self, markdown_text: str, title: str) -> str:
        # Small Markdown subset renderer for portability.
        body_lines: List[str] = []
        in_table = False
        for raw in markdown_text.splitlines():
            line = raw.rstrip()
            if line.startswith("# "):
                body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
            elif line.startswith("## "):
                body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
            elif line.startswith("### "):
                body_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
            elif line.startswith("- "):
                body_lines.append(f"<p class='bullet'>• {self._inline(line[2:])}</p>")
            elif line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if all(set(c) <= {":", "-", " "} for c in cells):
                    continue
                if not in_table:
                    body_lines.append("<table>")
                    in_table = True
                tag = "th" if not any("<td>" in x for x in body_lines[-3:]) else "td"
                body_lines.append("<tr>" + "".join(f"<{tag}>{self._inline(c)}</{tag}>" for c in cells) + "</tr>")
            else:
                if in_table and not line.startswith("|"):
                    body_lines.append("</table>")
                    in_table = False
                if line:
                    body_lines.append(f"<p>{self._inline(line)}</p>")
                else:
                    body_lines.append("")
        if in_table:
            body_lines.append("</table>")
        css = """
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem auto; max-width: 1100px; line-height: 1.45; color: #1f2937; }
        h1, h2, h3 { color: #111827; }
        code { background: #f3f4f6; padding: 0.1rem 0.25rem; border-radius: 0.25rem; }
        table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.95rem; }
        th, td { border: 1px solid #d1d5db; padding: 0.45rem 0.6rem; text-align: left; }
        th { background: #f9fafb; }
        .bullet { margin-left: 1rem; }
        """
        return "<!doctype html>\n<html><head><meta charset='utf-8'><title>" + html.escape(title) + "</title><style>" + css + "</style></head><body>" + "\n".join(body_lines) + "</body></html>\n"

    def _inline(self, text: str) -> str:
        escaped = html.escape(text)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
        return escaped


# =============================================================================
# Runner
# =============================================================================


class ExperimentRunner:
    """Main orchestration entry point."""

    def __init__(self, config: ExperimentConfig, *, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        ConfigValidator().validate(config)

    @classmethod
    def from_file(cls, path: PathLike, *, verbose: bool = True) -> "ExperimentRunner":
        return cls(ExperimentConfig.load(path), verbose=verbose)

    def run(self) -> JsonDict:
        root = ensure_dir(self.config.output_dir)
        ensure_dir(root / "tasks")
        ensure_dir(root / "manifests")
        logger = ExperimentLogger(root / "experiment.log", verbose=self.verbose)
        registry = ArtifactRegistry(root)
        provenance = ProvenanceTracker(self.config)
        started = time.perf_counter()
        try:
            logger.info(f"starting experiment {self.config.name!r}")
            provenance.add("experiment_start", status="running")
            registry.register_json(root / "config.resolved.json", self.config.to_dict(), name="resolved_config", kind="config")
            if self.config.created_from and Path(self.config.created_from).exists():
                registry.register_file(self.config.created_from, name="input_config", kind="config_source")
            graph = GraphLoader(self.config.created_from).load(self.config.graph)
            registry.register_json(root / "graph_summary.json", self._graph_summary(graph), name="graph_summary", kind="graph_summary")
            bundle_to_npz(graph, str(root / "graph_bundle.npz"))
            registry.register_file(root / "graph_bundle.npz", name="graph_bundle", kind="graph_bundle")
            context = RuntimeContext(
                config=self.config,
                root_dir=root,
                graph=graph,
                registry=registry,
                provenance=provenance,
                logger=logger,
                variables=dict(self.config.variables),
            )
            results = self._run_tasks(context)
            md = MarkdownReportBuilder().build(self.config, graph, results, registry, provenance)
            registry.register_text(root / "report.md", md, name="markdown_report", kind="report", media_type="text/markdown")
            html_text = HtmlReportBuilder().build(md, title=f"RINet Experiment: {self.config.name}")
            registry.register_text(root / "report.html", html_text, name="html_report", kind="report", media_type="text/html")
            provenance.add("experiment_end", status="success", duration_seconds=time.perf_counter() - started)
            registry.register_json(root / "provenance.json", provenance.to_dict(), name="provenance", kind="provenance")
            registry.write_manifest(root / "manifest.json")
            summary = self._final_summary(root, graph, results, registry, provenance, status="success")
            write_json(root / "summary.json", summary)
            logger.info(f"finished experiment {self.config.name!r}")
            return summary
        except Exception as exc:
            provenance.add("experiment_end", status="failed", duration_seconds=time.perf_counter() - started, message=str(exc), metadata={"traceback": traceback.format_exc()})
            try:
                registry.register_json(root / "provenance.json", provenance.to_dict(), name="provenance", kind="provenance")
                registry.write_manifest(root / "manifest.json")
            except Exception:
                pass
            logger.error(f"experiment failed: {exc}")
            if self.config.strict:
                raise
            return {"status": "failed", "error": str(exc), "output_dir": str(root), "provenance": provenance.to_dict()}
        finally:
            logger.close()

    def _run_tasks(self, context: RuntimeContext) -> List[TaskResult]:
        results: List[TaskResult] = []
        completed: Dict[str, str] = {}
        for spec in self._topological_tasks(self.config.tasks):
            if not spec.enabled:
                context.logger.info("skipping disabled task", task=spec.name)
                continue
            failed_deps = [d for d in spec.depends_on if completed.get(d) != "success"]
            if failed_deps:
                msg = f"dependency failure: {failed_deps}"
                result = TaskResult(spec.name, spec.kind, "skipped", utc_now_iso(), utc_now_iso(), 0.0, str(context.task_dir(spec)), error=msg)
                context.task_results[spec.name] = result
                results.append(result)
                completed[spec.name] = "skipped"
                context.logger.warning(msg, task=spec.name)
                continue
            task = TASKS.create(spec)
            out_dir = context.task_dir(spec)
            context.provenance.add("task_start", task=spec.name, status="running", metadata={"kind": spec.kind, "params": spec.params})
            context.logger.info(f"running {spec.kind}", task=spec.name)
            start = time.perf_counter()
            try:
                result = task.run(context, out_dir)
                context.task_results[spec.name] = result
                results.append(result)
                completed[spec.name] = result.status
                context.registry.register_json(out_dir / "task_result.json", result.to_dict(), task=spec.name, kind="task_result")
                context.provenance.add("task_end", task=spec.name, status=result.status, duration_seconds=result.duration_seconds, metadata={"artifacts": result.artifacts})
                context.logger.info(f"completed in {result.duration_seconds:.3f}s", task=spec.name)
            except Exception as exc:
                tb = traceback.format_exc()
                result = TaskResult(
                    name=spec.name,
                    kind=spec.kind,
                    status="failed",
                    started_at=utc_now_iso(),
                    ended_at=utc_now_iso(),
                    duration_seconds=float(time.perf_counter() - start),
                    output_dir=str(out_dir),
                    error=str(exc),
                    traceback=tb,
                )
                context.task_results[spec.name] = result
                results.append(result)
                completed[spec.name] = "failed"
                try:
                    context.registry.register_json(out_dir / "task_result.json", result.to_dict(), task=spec.name, kind="task_result")
                except Exception:
                    pass
                context.provenance.add("task_end", task=spec.name, status="failed", duration_seconds=result.duration_seconds, message=str(exc), metadata={"traceback": tb})
                context.logger.error(str(exc), task=spec.name)
                if self.config.fail_fast and not spec.allow_failure:
                    raise
        return results

    def _topological_tasks(self, tasks: Sequence[TaskSpec]) -> List[TaskSpec]:
        by_name = {t.name: t for t in tasks}
        visited: set[str] = set()
        visiting: set[str] = set()
        order: List[TaskSpec] = []

        def visit(t: TaskSpec) -> None:
            if t.name in visited:
                return
            if t.name in visiting:
                raise ConfigError(f"dependency cycle at task {t.name!r}")
            visiting.add(t.name)
            for dep in t.depends_on:
                if dep in by_name:
                    visit(by_name[dep])
            visiting.remove(t.name)
            visited.add(t.name)
            order.append(t)

        for task in tasks:
            visit(task)
        return order

    def _graph_summary(self, graph: GraphBundle) -> JsonDict:
        A = np.asarray(graph.A, dtype=float)
        return {
            "name": graph.name,
            "n": graph.n,
            "meta": graph.meta,
            "hash": graph_hash(graph),
            "adjacency": matrix_summary(A),
            "edge_count_undirected": int(np.count_nonzero(np.triu(A, 1))),
            "degree": vector_summary(np.sum(A, axis=1)),
        }

    def _final_summary(self, root: Path, graph: GraphBundle, results: Sequence[TaskResult], registry: ArtifactRegistry, provenance: ProvenanceTracker, *, status: str) -> JsonDict:
        return {
            "status": status,
            "name": self.config.name,
            "output_dir": str(root),
            "graph": {"name": graph.name, "n": graph.n, "hash": graph_hash(graph)},
            "tasks": [r.to_dict() for r in results],
            "artifact_count": len(registry.artifacts),
            "manifest": str(root / "manifest.json"),
            "report_markdown": str(root / "report.md"),
            "report_html": str(root / "report.html"),
            "config_hash": self.config.content_hash(),
            "provenance_events": len(provenance.events),
        }


# =============================================================================
# Config templates and convenience API
# =============================================================================


EXAMPLE_CONFIG: JsonDict = {
    "name": "rinet_full_platform_demo",
    "description": "Demonstration of diffusion, spectral geometry, GSP, counterfactual inference, and robustness.",
    "output_dir": "runs/rinet_full_platform_demo",
    "graph": {"kind": "synthetic", "params": {"n": 60, "k": 4, "seed": 0}},
    "tasks": [
        {"name": "forward_seed_0", "kind": "diffusion", "params": {"seed_nodes": [0], "steps": 60, "dt": 0.05}},
        {"name": "spectral_geometry", "kind": "spectral", "params": {"modes": 20, "heat_times": [0.1, 1.0, 3.0]}},
        {"name": "counterfactual_residue_10", "kind": "counterfactual", "params": {"seed_nodes": [0], "target_nodes": [30], "steps": 60, "dt": 0.05, "interventions": [{"type": "remove_node", "node": 10}]}},
        {"name": "gsp_signal", "kind": "gsp", "params": {"seed_nodes": [0, 12], "strengths": [1.0, 0.5]}},
        {"name": "robustness_edge_noise", "kind": "robustness", "params": {"model": "edge_noise", "std": 0.03, "n_trials": 30, "seed_nodes": [0], "steps": 50}},
    ],
}


def write_example_config(path: PathLike) -> Path:
    return write_json(path, EXAMPLE_CONFIG)


def run_experiment(config: Union[ExperimentConfig, Mapping[str, Any], PathLike], *, verbose: bool = True) -> JsonDict:
    if isinstance(config, ExperimentConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = ExperimentConfig.from_mapping(config)
    else:
        cfg = ExperimentConfig.load(config)
    return ExperimentRunner(cfg, verbose=verbose).run()


def available_task_kinds() -> List[str]:
    return TASKS.aliases()


# =============================================================================
# Command-line interface
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rinet-orchestrate",
        description="Run reproducible RINet / AlloGraph experiment configs.",
    )
    parser.add_argument("config", nargs="?", help="Path to JSON/YAML experiment config")
    parser.add_argument("--write-example", help="Write an example config to this path and exit")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging to stdout")
    parser.add_argument("--list-tasks", action="store_true", help="List supported task kinds and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.list_tasks:
        for alias in available_task_kinds():
            print(alias)
        return 0
    if args.write_example:
        path = write_example_config(args.write_example)
        print(f"wrote {path}")
        return 0
    if not args.config:
        parser.error("config path is required unless using --write-example or --list-tasks")
    summary = ExperimentRunner.from_file(args.config, verbose=not args.quiet).run()
    print(stable_json_dumps({"status": summary.get("status"), "output_dir": summary.get("output_dir"), "report_html": summary.get("report_html")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
