# RINet — Residue Interaction Network Inference Toolkit

**RINet** (Residue Interaction Network Inference Toolkit) is an open-source, deterministic framework for constructing **residue interaction networks (RINs)** from protein structures and performing interpretable, structure-first allosteric analysis using explicit **linear graph inference operators**.

RINet is built as a **software product**, not a research prototype. It is designed to be reproducible, auditable, and stable under repeated execution: the same inputs and parameters produce the same graphs, the same operators, and the same results.

This repository provides two tightly coupled components:

- A Python backend library, **`rinet_core`**, that can be imported and embedded in scripts and pipelines.
- A desktop application (macOS), **`rinet_desktop`**, that provides an interactive GUI for analysis, visualization, and export of results.

RINet is intended for exploratory protein structure analysis, hypothesis generation, comparative studies across variants, and workflows where interpretability and deterministic execution are critical.

---

## Table of Contents

[Motivation and Scope](#motivation-and-scope)

[Conceptual Overview](#conceptual-overview) 

[Residue Interaction Networks](#residue-interaction-networks) 

[Graph Construction from Structure](#graph-construction-from-structure) 

[Mathematical Framework](#mathematical-framework) 

[Inference Modes](#inference-modes) 

[Determinism and Reproducibility](#determinism-and-reproducibility) 

[Software Architecture](#software-architecture) 

[Python API](#python-api) 

[Command-Line Usage](#command-line-usage) 

[Desktop Application](#desktop-application) 

[Outputs](#outputs) 

[Validation and Interpretation Guidance](#validation-and-interpretation-guidance) 

[Limitations](#limitations) 

[Intended Use Cases](#intended-use-cases) 

[Installation](#installation) 

[Contributing](#contributing) 

[Citation](#citation) 

[License](#license)

---

## Motivation and Scope

Allostery is the phenomenon by which perturbations at one site in a protein influence distant sites, altering function, binding, catalysis, or regulation. Practical computational tools for allostery frequently fall into one of three categories: long-timescale simulation methods (molecular dynamics and enhanced sampling), statistical or ensemble models (co-evolution, elastic network models, and normal mode approaches), and machine learning methods trained on datasets of structures or sequences. These approaches can be powerful, but many are expensive to run, difficult to reproduce exactly, or opaque in how signals propagate from source to target.

RINet takes a complementary approach that is:

- **Structure-first**: analysis begins from a static protein structure (PDB/mmCIF).
- **Graph-based**: residues become nodes; geometric contacts become edges.
- **Linear and deterministic**: inference uses explicit linear operators on sparse matrices.
- **Interpretable**: every modeling choice is explicit; intermediate quantities are inspectable.

The goal is not to replace physics-heavy simulation. The goal is to provide a lightweight, deterministic analysis layer for rapid screening and hypothesis generation, enabling structured comparisons across proteins, mutants, ligands, conformations, and parameter settings.

---

## Conceptual Overview

RINet runs a sequence of deterministic steps:

1. Parse a protein structure (PDB or mmCIF) and build a residue index with stable labels.
2. Construct a residue interaction network using explicit geometric rules.
3. Choose an operator family (diffusion-like Laplacian dynamics, normalized propagation, or resolvent-style operators).
4. Define a query (source residues, target residues, or a global scan objective).
5. Execute the inference operator deterministically.
6. Summarize results at residue and/or edge level, with transparent scoring definitions.
7. Produce outputs suitable for inspection and downstream comparison.

All inference modes operate on the same underlying network representation so that results are comparable across analyses.

---

## Residue Interaction Networks

A residue interaction network is an undirected weighted graph:

$$
G = (V, E)
$$

Each node $v_i \in V$ corresponds to a residue in the protein. Each edge $(i, j) \in E$ represents a geometric contact between residues $i$ and $j$ under an explicit rule. Each edge has a non-negative weight:

$$
w_{ij} \ge 0
$$

Weights encode an interaction strength under a chosen deterministic scheme (binary, inverse-distance, exponential decay, or user-defined).

RINet intentionally avoids learned contacts. Edges are derived from a fully specified contact definition with fully specified parameters.

---

## Graph Construction from Structure

### Residue Nodes and Stable Labels

Each residue is identified by a stable label (conceptually):

- chain identifier
- residue number
- insertion code
- residue name

This label is treated as the canonical identity throughout the pipeline. Internal indices are derived from these labels, but outputs always preserve stable labeling.

### Contact Definitions

Edges are added according to a configurable contact specification. Common modes include:

**Cα contact mode**  
An edge is added if the distance between Cα atoms is below a cutoff:

$$
\| \mathbf{r}^{(\alpha)}_i - \mathbf{r}^{(\alpha)}_j \| \le d_{\text{cut}}
$$

This is fast and often sufficient for coarse structural coupling analyses.

**Heavy-atom contact mode**  
An edge is added if the minimum distance between any pair of heavy atoms across the two residues is below a cutoff:

$$
\min_{a \in \mathcal{A}(i),\; b \in \mathcal{A}(j)} \| \mathbf{r}_{a} - \mathbf{r}_{b} \| \le d_{\text{cut}}
$$

This is more expensive but more biophysically faithful, particularly for dense cores, salt bridges, and side-chain mediated interactions.

### Edge Weighting

RINet supports deterministic weighting schemes. Let $d_{ij}$ be the contact distance under the selected mode.

**Binary weighting**
$$
w_{ij} = 1
$$

**Inverse-distance weighting**
$$
w_{ij} = \frac{1}{d_{ij} + \varepsilon}
$$

**Exponential decay weighting**
$$
w_{ij} = \exp\left( -\frac{d_{ij}}{\sigma} \right)
$$

Where $\varepsilon > 0$ avoids singularity and $\sigma > 0$ sets the decay length scale.

### Optional Backbone Connectivity

RINet can enforce sequential connectivity by adding edges between consecutive residues in each chain:

$$
(i, i+1) \in E \quad \text{for consecutive residues in the same chain}
$$

This can improve connectivity in sparse contact regimes and supports certain analyses that require connected components to be stable across parameter sweeps.

---

## Mathematical Framework

### Sparse Matrix Representation

The RIN is represented as a sparse adjacency matrix:

$$
A \in \mathbb{R}^{N \times N}
$$

Where $N$ is the number of residues, and:

$$
A_{ij} =
\begin{cases}
w_{ij} & \text{if } (i, j) \in E \\
0      & \text{otherwise}
\end{cases}
$$

RINet stores matrices in CSR (Compressed Sparse Row) form for efficiency and deterministic iteration order.

The degree matrix is:

$$
D = \mathrm{diag}(d_1, \dots, d_N), \quad d_i = \sum_{j=1}^N A_{ij}
$$

The combinatorial Laplacian is:

$$
L = D - A
$$

### Operator Families

RINet focuses on explicit linear operators that support step-by-step interpretability.

#### Continuous-Time Diffusion-Like Dynamics

One canonical model is a diffusion/decay equation:

$$
\frac{d x(t)}{dt} = -\alpha x(t) - \beta L x(t)
$$

Where $x(t) \in \mathbb{R}^N$ is a residue influence state, $\alpha \ge 0$ is a decay term, and $\beta \ge 0$ controls coupling strength.

A deterministic explicit Euler discretization gives:

$$
x_{t+1} = x_t + \Delta t \left( -\alpha x_t - \beta L x_t \right)
$$

This is linear, deterministic, and reproducible.

#### Normalized Propagation

A discrete propagation model uses a row-normalized adjacency:

$$
W = D^{-1} A
$$

With update:

$$
x_{t+1} = \rho W x_t
$$

Where $0 < \rho < 1$ attenuates influence at each step.

This model emphasizes walk-like propagation along weighted edges.

#### Resolvent-Style Steady-State Operator

Another useful deterministic operator is a resolvent form:

$$
x^\* = \left( I + \lambda L \right)^{-1} s
$$

Where $s$ is a seed vector encoding sources and $\lambda > 0$ controls smoothing/coupling. This is a linear system solve on a sparse symmetric positive semidefinite matrix (with appropriate stabilization for components), yielding a single-shot steady-state coupling profile.

RINet may implement one or more of these families; the design intent is that operator families remain explicit and modular so that each operator is independently inspectable and testable.

---

## Inference Modes

RINet provides multiple inference modes built on the same graph representation.

### Forward Influence Propagation

Forward inference answers:

> If I perturb residue(s) $S$, which residues are most affected?

Construct a seed vector $s$:

$$
s_i =
\begin{cases}
1 & i \in S \\
0 & \text{otherwise}
\end{cases}
$$

Run an operator to compute $x$ (either after $T$ steps or at steady state). The final $x$ is interpreted as a coupling profile. Higher magnitude indicates stronger coupling under the chosen operator.

Forward inference is designed to be interpretable. Each residue score can be traced back to network connectivity, edge weights, and operator parameters.

### Inverse Attribution

Inverse inference answers:

> Which residues could plausibly explain a response observed at a target set $T$?

A simple deterministic inverse protocol is:

1. For each candidate source residue $i$, treat $s = e_i$ and compute the response at targets.
2. Score candidate sources by target response magnitude.

For target set $T$, candidate score:

$$
\mathrm{score}(i) = \sum_{j \in T} |x^{(i)}_j|
$$

Where $x^{(i)}$ is the coupling profile from source $i$ under the chosen operator.

This is computationally heavier than forward inference if done naively for all residues, so RINet supports structured candidate sets and operator-specific accelerations where possible, while keeping the procedure deterministic.

### Global Scan Inference

Global scan inference identifies influential edges or residues by systematic perturbation of network elements and measuring deterministic changes in an objective.

One interpretable scan is an edge attenuation scan. For an edge $(i, j)$, define a perturbed adjacency:

$$
A'_{ij} = \gamma A_{ij}, \quad 0 < \gamma < 1
$$

Run a fixed forward query (or an operator-defined objective) and compute an objective difference:

$$
\Delta S_{ij} = S(A') - S(A)
$$

Where $S(\cdot)$ is a deterministic scalar summary (for example, total mass outside the seed set, or target response magnitude). Edges can be ranked by $|\Delta S_{ij}|$ to identify contacts that strongly control propagation under the operator.

The key property is that this scan is completely specified by $(A, \gamma, S, \text{operator parameters})$ and is fully reproducible.

### Mediator Analysis

Mediator analysis aims to identify residues that lie “between” a source and a target in the sense of coupling propagation.

A conservative, explicitly interpretable baseline is:

- Compute a coupling profile from a source set.
- Optionally compute a coupling profile toward a target set (or compute a target-focused objective).
- Score mediators as residues with high coupling magnitude excluding sources/targets.

This definition is intentionally simple and transparent. More sophisticated mediator scoring (betweenness-like measures, flow decompositions, path ensembles) can be layered on later as long as they remain deterministic and explicitly specified.

---

## Determinism and Reproducibility

RINet is deterministic by construction.

- No stochastic sampling is used in graph construction.
- No random number generators are used in inference.
- Contact rules are fully specified by explicit parameters.
- Operator updates are linear and deterministic.
- Sparse matrix operations use stable ordering and explicit numeric types.

For a fixed input structure and fixed parameter set, RINet produces identical results on repeated runs within numerical tolerance determined by floating point arithmetic and the linear solver backend used.

Determinism is treated as a first-class product feature because it enables:

- exact comparisons across mutants and variants
- regression testing across versions
- reproducible figures and tables for collaborators
- auditability of analysis decisions

---

## Software Architecture

RINet is divided into two main components.

### `rinet_core` (Backend Library)

The backend is a pure Python library containing:

- structure parsing and residue labeling
- contact graph construction
- sparse matrix builders
- operator implementations (forward, inverse, scan, mediator)
- result objects with stable serialization and explicit metadata
- testing utilities and deterministic fixtures

The backend is designed for pipeline embedding. It should not require GUI dependencies.

### `rinet_desktop` (Desktop Application)

The desktop application is a PyQt-based GUI that orchestrates the backend.

It provides:

- guided structure import (PDB/mmCIF)
- residue tables with sorting/filtering
- operator selection and parameter controls
- interactive plots of coupling profiles
- network views suitable for demonstration and exploration
- export of results for downstream use

The GUI is intended for interactive exploration and communication of results, not high-throughput batch processing.

---

## Python API

Below are representative examples. The exact module paths may differ depending on your installed version, but the design intent is stable: graph construction produces a bundle-like object, and inference routines accept that bundle plus explicit parameters.

### Minimal Example: Build a RIN and Run Forward Propagation

```python
from rinet_core.graphio.structure import load_structure
from rinet_core.graph.build import build_rin, RINSpec
from rinet_core.inference.forward import run_forward, ForwardSpec

# Load a structure file (PDB or mmCIF)
structure = load_structure("structure.pdb")

# Build a residue interaction network
rin_spec = RINSpec(
    contact_mode="heavy_atom",     # "ca" or "heavy_atom"
    cutoff=4.5,
    weight_scheme="exponential",   # "binary", "inverse_distance", "exponential"
    sigma=2.0,
    add_backbone=True
)

bundle = build_rin(structure, rin_spec)

# Define a forward query
forward_spec = ForwardSpec(
    operator="diffusion",          # "diffusion" or "normalized" or "resolvent"
    steps=50,
    dt=0.1,
    alpha=0.0,
    beta=1.0,
    rho=0.95
)

# Seed residues can be specified by stable labels or internal indices
result = run_forward(bundle, seed_residues=[("A", 42)], spec=forward_spec)

# Access the coupling profile (vector indexed by residues)
x = result.profile
top = result.top_residues(k=25)

print(top[:10])
```
# Example: Inverse Attribution to Explain a Target Region

```python
from rinet_core.inference.inverse import run_inverse, InverseSpec

inverse_spec = InverseSpec(
    operator="resolvent",
    lambda_=0.5,
    candidate_pool="all",  # or a subset selector
    score="target_sum_abs"
)

inverse = run_inverse(
    bundle,
    target_residues=[("A", 100), ("A", 101), ("A", 102)],
    spec=inverse_spec
)

print(inverse.top_sources(k=20))
```

# Example: Edge Scan to Find Controlling Contacts

```python

from rinet_core.inference.scan import run_edge_scan, EdgeScanSpec

scan_spec = EdgeScanSpec(
    operator="diffusion",
    steps=40,
    dt=0.1,
    alpha=0.0,
    beta=1.0,
    perturbation_factor=0.5,  # gamma in the description
    objective="total_mass_outside_seed"
)

scan = run_edge_scan(
    bundle,
    seed_residues=[("A", 42)],
    spec=scan_spec
)

print(scan.top_edges(k=25))
```
RINet’s API is designed so that inference objects carry explicit metadata about every parameter used. If you are running comparisons (mutants, multiple conformations, parameter sweeps), you should keep inference specs fixed across runs to ensure interpretability of differences.

# Command-Line Usage

If you prefer the command line, RINet provides CLI entry points to run common analyses deterministically.

A typical workflow is:

Build a RIN from a structure.

Run one or more inference modes (forward, inverse, scan, mediator).

Write results to files for inspection and comparison.

Example patterns: 

```python
rinet build --in structure.pdb --contact heavy_atom --cutoff 4.5 --weight exponential --sigma 2.0 --add-backbone
rinet forward --in structure.pdb --seed A:42 --operator diffusion --steps 50 --dt 0.1 --alpha 0.0 --beta 1.0
rinet inverse --in structure.pdb --target A:100,A:101 --operator resolvent --lambda 0.5
rinet scan --in structure.pdb --seed A:42 --operator diffusion --steps 40 --dt 0.1 --beta 1.0 --edge-gamma 0.5
```

# Desktop Application

The desktop application provides an interactive, demonstration-friendly interface for running RINet analyses.

Launching the Application
```python
python -m rinet_desktop.main
```

The GUI focuses on interactive inspection: guided structure import with residue labeling and chain selection, inspection of contact definitions and edge weight schemes, residue tables with filtering, sorting, and searching by label, operator selection with explicit parameter panels, coupling profile plots and rank tables for residues and edges, mediator-style views that emphasize source-to-target structure, reproducible export of results with explicit run metadata. The GUI is a thin orchestration layer over rinet_core. If a feature exists in the GUI, it should also be possible via the Python API.

RINet outputs are designed to be human-inspectable (tables and summaries), machine-readable (TSV/CSV/JSON where appropriate), and stable across versions when parameters and inputs are unchanged. Typical outputs include residue coupling profiles (full vectors and ranked subsets), ranked influential residues for a query, ranked influential edges for scan modes, run metadata capturing all parameters used, plots suitable for quick inspection and communication, exact filenames and formats may evolve, but the product requirement is that outputs remain explicit, auditable, and comparable across runs. RINet is a hypothesis-generation tool. The outputs are not meant to be interpreted as quantitative physical energies or experimental effect sizes. Proper interpretation depends on understanding what the operator measures.

# Practical guidance:

Treat coupling profiles as operator-defined structural influence measures, not direct thermodynamic quantities. Keep graph construction parameters fixed when comparing variants. Changing contact mode or cutoff changes the graph itself, making comparisons ambiguous. Use heavy-atom contacts for detailed analyses and Cα contacts for fast coarse scans. If you are targeting a functional region, define the target set explicitly and use inverse or objective-based scan modes rather than interpreting global mass heuristics. Use multiple operator families when you want robustness checks. Agreement across operator families can be more informative than any single operator output. RINet is intentionally conservative: it makes fewer assumptions than a complex physical model, but it also captures fewer physical effects. The main value is interpretability and reproducibility under explicit modeling choices.

# Limitations

RINet has deliberate limitations. It operates on static structures and does not simulate dynamics. Models are linear and do not capture nonlinear coupling or state-dependent interactions. Contact graphs depend on structure quality, missing atoms, alternate conformations, and chosen cutoffs. Results depend on operator choice; there is no universal “correct” operator for all allosteric problems. Mediator analysis is intentionally conservative and may not capture complex multi-pathway effects without extended operator definitions. RINet should be used to generate hypotheses and to support structured comparisons, not as a replacement for physics-based simulation or experimental validation.

# Installation

RINet targets Python 3.9+ (recommended). The backend uses standard scientific Python tooling.

# Backend Library

For development installs:
```python
python -m pip install -e .
```

Or if the project is packaged as separate components:

```python
python -m pip install -e rinet_core
```

# Desktop Application

The desktop app is included as a Python module:

```python
python -m pip install -e rinet_desktop
```

Then launch:

```python
python -m rinet_desktop.main
```

# License

RINet is released under an open-source license. See the LICENSE file for details.
