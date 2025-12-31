# RINet
Residue Interaction Network Inference Toolkit

RINet is an open source, deterministic software framework for constructing residue interaction networks from protein structures and performing interpretable graph based inference on those networks. The system is designed as a software product rather than a research prototype. All computations are explicit, linear, and reproducible. Given the same inputs and parameters, RINet produces the same outputs across repeated runs.

RINet follows a structure first philosophy. It converts static protein structures into residue level graphs using fully specified geometric rules and applies deterministic graph propagation operators to analyze how structural connectivity may mediate long range coupling within proteins.

This repository currently contains two tightly coupled components:

- `allograph_core`, a Python backend library that implements graph construction and inference
- `allograph_desktop`, a macOS desktop application that provides an interactive graphical interface

RINet is the system name and product identity. The internal Python module name `allograph` reflects an earlier internal naming decision and should be understood as an implementation detail. This README describes the system as it exists today and does not assume future refactors or renaming.

RINet is intended for exploratory protein structure analysis, hypothesis generation, and comparative workflows where interpretability, determinism, and auditability are critical.

---

## Table of Contents

- Motivation and Scope  
- Conceptual Overview  
- Residue Interaction Networks  
- Graph Construction from Structure  
- Mathematical Framework  
- Inference Modes  
- Determinism and Reproducibility  
- Software Architecture  
- Python API  
- Command Line Interface  
- Desktop Application  
- Outputs  
- Interpretation Guidance  
- Limitations  
- Installation  
- License  

---

## Motivation and Scope

Allostery refers to the phenomenon by which perturbations at one site in a protein influence distant sites, affecting function, binding, catalysis, or regulation. Many computational approaches to allostery rely on molecular dynamics simulations, elastic network models, statistical coupling analysis, or machine learning methods. While powerful, these approaches can be expensive to run, difficult to reproduce exactly, or opaque in how signals propagate from source to target.

RINet takes a complementary approach based on explicit graph modeling. The goal is not to replace physics based simulation, but to provide a lightweight, deterministic analysis layer that enables rapid and interpretable exploration of structure mediated coupling.

The design principles of RINet are:

- Structure first analysis starting from a static protein structure  
- Explicit residue level graphs derived from geometry  
- Linear and deterministic inference operators  
- Transparent modeling assumptions  
- Stable outputs suitable for comparison across variants  

RINet is particularly useful when the goal is to compare relative coupling patterns across proteins, mutants, or parameter settings under a fixed and auditable modeling framework.

---

## Conceptual Overview

A typical RINet workflow proceeds as follows:

1. Parse a protein structure file in PDB format  
2. Identify residues with resolved C alpha atoms  
3. Construct a residue interaction network using a specified distance cutoff  
4. Represent the network as an adjacency matrix with explicit weights  
5. Choose an inference operator such as diffusion or discrete propagation  
6. Define a query such as a seed residue or a global scan objective  
7. Execute the operator deterministically  
8. Inspect and export residue level scores and metadata  

All inference modes operate on the same underlying graph representation, allowing consistent comparison across different analyses.

---

## Residue Interaction Networks

A residue interaction network is modeled as an undirected weighted graph $G = (V, E)$.

Each node $v_i \in V$ corresponds to a residue in the protein. Each edge $(i, j) \in E$ corresponds to a geometric contact between residues. Edge weights $w_{ij}$ are non negative and explicitly defined.

In RINet, residue interaction networks are derived directly from protein structures using deterministic contact rules. No learned or inferred contacts are used.

---

## Graph Construction from Structure

### Residue Identification

Each residue is identified by a stable label consisting of:

- Chain identifier  
- Residue sequence number  
- Insertion code if present  

Residues without resolved C alpha atoms are excluded. Hetero residues and water molecules are excluded. The internal node index order is derived from structure parsing order, but stable residue labels are preserved in metadata and exported results.

### Contact Definition

RINet currently implements a C alpha distance based contact rule.

For residues $i$ and $j$, an edge is added if:

$$
\| \mathbf{r}_i - \mathbf{r}_j \| \le d_{\text{cut}}
$$

where $\mathbf{r}_i$ and $\mathbf{r}_j$ are the Cartesian coordinates of the C alpha atoms and $d_{\text{cut}}$ is a user specified cutoff in angstroms.

This contact definition is simple, fast, and deterministic, making it suitable for reproducible analysis and parameter sweeps.

### Edge Weighting

Two deterministic edge weighting schemes are currently implemented.

Binary weighting:

$$
w_{ij} = 1
$$

Inverse distance weighting:

$$
w_{ij} = \frac{1}{d_{ij} + \varepsilon}
$$

where $d_{ij}$ is the C alpha distance and $\varepsilon$ is a small constant to avoid singularities.

The selected weighting scheme and cutoff are recorded in the graph metadata.

---

## Mathematical Framework

### Adjacency Matrix

The residue interaction network is represented as a dense adjacency matrix $A \in \mathbb{R}^{N \times N}$, where $N$ is the number of residues.

$$
A_{ij} =
\begin{cases}
w_{ij} & \text{if } (i, j) \in E \\
0 & \text{otherwise}
\end{cases}
$$

The matrix is symmetric and contains only finite values. Validation is performed before inference to ensure correctness.

### Degree Matrix and Laplacian

The degree matrix $D$ is defined as:

$$
D_{ii} = \sum_{j=1}^{N} A_{ij}
$$

The unnormalized graph Laplacian is defined as:

$$
L = D - A
$$

The Laplacian is used as the core operator in diffusion based inference.

---

## Inference Modes

RINet provides several deterministic inference modes implemented in the backend library.

### Forward Influence Propagation

Forward inference answers the question:

If a perturbation originates at one or more residues, which residues are most affected under a chosen operator?

A seed vector $x_0 \in \mathbb{R}^N$ is constructed with nonzero entries at the seed residues. A dynamics operator is applied for a fixed number of steps to produce a final state vector $x_T$.

The magnitude $|x_{T,i}|$ is interpreted as the relative coupling strength of residue $i$.

### Diffusion Dynamics

The diffusion model implements an explicit Euler discretization of Laplacian based propagation:

$$
x_{t+1} = x_t - \Delta t \, L x_t
$$

where $\Delta t$ is a fixed time step and $L$ is the unnormalized graph Laplacian.

This model is linear, deterministic, and stable for sufficiently small $\Delta t$.

### Discrete Hop Propagation

The discrete hop model propagates influence by repeated multiplication with the adjacency matrix:

$$
x_{t+1} = A x_t
$$

After each step, the state vector is normalized to prevent divergence.

This model emphasizes walk based propagation along network edges.

### Global Scan Inference

Global scan inference identifies influential residues by systematic perturbation.

For each residue $i$:

1. Seed only residue $i$  
2. Run forward inference  
3. Compute a scalar score such as $ s_i = \sum_{j=1}^{N} |x^{(i)}_j| $

Residues are ranked by this score. This procedure is deterministic but computationally expensive for large networks.

### Inverse Attribution

Inverse inference is implemented as a simple deterministic baseline.

Given an observed state vector $y$, each residue $i$ is scored by:

$$
\text{score}(i) = \frac{y_i}{\| y \|}
$$

This mode is intentionally conservative and is included as a placeholder for future inverse operators.

### Mediator Analysis

Mediator analysis computes degree centrality as a conservative proxy for mediation:

$$
\text{deg}(i) = \sum_{j=1}^{N} \mathbb{I}(A_{ij} > 0)
$$

Residues with high degree are identified as potential mediators under the assumption that highly connected nodes may facilitate signal propagation.

---

## Determinism and Reproducibility

RINet is deterministic by construction.

- Graph construction uses explicit geometric rules  
- No random sampling is used in inference  
- Dynamics operators are linear and deterministic  
- All parameters are recorded in metadata  

Given the same structure, parameters, and software version, RINet produces identical outputs within floating point tolerance.

---

## Software Architecture

RINet is implemented as a modular system with a strict separation between core logic and user interface.

### Backend Library

The backend library is implemented under the internal Python namespace `allograph.core` and provides:

- `GraphBundle` as the central data structure  
- Structure parsing and graph construction utilities  
- Deterministic inference routines  
- Dynamics operators  
- Serialization and validation utilities  

The backend has no GUI dependencies and can be embedded in scripts or pipelines.

### Desktop Application

The desktop application is implemented under `allograph_desktop` and provides a PySide based graphical interface. It orchestrates the backend library and manages application state.

The GUI does not implement scientific logic. All computations are delegated to the backend.

---

## Python API

A minimal example using the backend library:

```python
from allograph.core.graphio import pdb_to_rin
from allograph.core.inference.forward import run_forward

bundle = pdb_to_rin(
    pdb_path="structure.pdb",
    cutoff=8.0,
    chain=None,
    weight_mode="binary"
)

result = run_forward(
    bundle,
    model="diffusion",
    seed_nodes=[10],
    steps=60,
    dt=0.1
)

state = result.state
meta = result.meta
```
# Command Line Interface

RINet provides a command line interface implemented in allograph.cli.

Example forward propagation on a synthetic network:

```python
python -m allograph.cli forward --demo --seed 5
```

Example scan on a PDB derived network:

```python
python -m allograph.cli scan --pdb structure.pdb --cutoff 8.0
```
Results are written to CSV files with associated metadata.

# Desktop Application

The desktop application can be launched with:

```python
python -m allograph_desktop.main
```

The GUI supports interactive exploration of residue interaction networks and inference results. All operations available in the GUI correspond directly to backend functions.

# Outputs 

RINet produces residue level score vectors, ranked lists of influential residues, serialized graph bundles, CSV and NPZ output files, and metadata capturing all parameters used. Outputs are designed to be human inspectable and machine readable. 

# Interpretation Guidance

RINet outputs should be interpreted as operator defined structural influence measures. They are not physical energies or experimental effect sizes.

Comparisons should only be made between analyses that use identical graph construction and inference parameters. Agreement across multiple operator choices can increase confidence in qualitative conclusions.

RINet operates on static structures and does not model conformational dynamics. All inference operators are linear and do not capture nonlinear effects. Contact graphs depend on structure quality and cutoff selection. Inverse and mediator analyses are intentionally conservative.

RINet is intended for hypothesis generation and structured comparison, not for quantitative prediction of experimental outcomes.

# Installation

RINet requires Python 3.9 or newer.

For development installs:

```python
pip install -e allograph_core
pip install -e allograph_desktop
```

# License

RINet is released under an open source license. See the LICENSE file for details.
