# RINet

RINet (Residue Interaction Network Inference Toolkit) is an open-source, deterministic framework for constructing residue interaction networks (RINs) from protein structures and generating interpretable, structure-first residue coupling profiles using explicit graph propagation operators. RINet emphasizes reproducible execution, transparent assumptions, and exportable analysis artifacts suitable for exploratory protein structure analysis and comparative workflows.

## Overview

Proteins often exhibit allostery, where a perturbation at one site, such as ligand binding or mutation, produces functional effects at distant regions of the structure. While molecular dynamics simulations provide detailed insight into these processes, they are computationally expensive and difficult to compare across variants. Machine learning approaches can scale more easily but often lack interpretability and reproducibility. RINet addresses this gap by modeling a protein structure as a residue interaction network and applying deterministic graph-based inference to analyze how influence propagates through the structure. The goal of RINet is not to replace simulation-based methods, but to provide a lightweight, interpretable, and reproducible tool for hypothesis generation and comparative analysis.

### Residue Interaction Networks

A residue interaction network represents a protein as a graph in which nodes correspond to residues and edges correspond to structural contacts. Contacts may be defined using Cα distance thresholds or heavy-atom proximity rules. Edge weights may be binary or distance-based, depending on the analysis requirements.

Graph construction parameters are explicit and recorded for every run, allowing results to be compared across structures and variants under consistent assumptions.

### Propagation Dynamics

RINet models signal propagation on the residue interaction network using deterministic linear operators. Let \( x_t \) denote a residue-indexed state vector and let \( L \) denote the graph Laplacian derived from the adjacency matrix. A typical update rule is

$$
x_{t+1} = x_t + \Delta t \cdot (-L x_t)
$$

where the step size \( \Delta t \) and number of steps control the spatial extent and smoothness of propagation. Because execution is fully deterministic, identical inputs always produce identical outputs.

## Inference Modes

RINet supports several complementary inference modes. Forward propagation computes how influence spreads from a specified seed residue or region. Scan inference evaluates influence from each residue in turn to identify globally influential sites. Inverse attribution identifies candidate upstream residues that could explain an observed downstream pattern. Mediator analysis highlights residues that act as bridges in influence pathways. Parameter sweeps enable sensitivity analysis over seeds or model parameters. All inference modes produce residue-indexed numerical outputs that can be inspected, compared, and exported.


## Installation

RINet can be installed using pip.

```bash
pip install rinet
```

For development or local modification.

```bash
git clone https://github.com/<username>/rinet.git
cd rinet
pip install -e .
```

## Command Line Interface

After installation, the command line interface is available as `rinet`.

Forward propagation on a synthetic graph.

```bash
rinet forward --demo --seed 5 --out forward.csv
```

Forward propagation from a protein structure.

```bash
rinet forward --pdb 1HHO.pdb --seed 82 --cutoff 8.0 --out forward.csv
```

Global scan inference.

```bash
rinet scan --pdb 1HHO.pdb --cutoff 8.0 --out scan.csv
```

Outputs are written as residue-indexed CSV files suitable for downstream analysis or visualization.

## Desktop Application

RINet also includes a desktop graphical application designed for interactive workflows. The graphical interface supports loading protein structures, configuring graph construction parameters, running inference modes, visualizing residue coupling profiles, inspecting tables of residue scores, and exporting complete analysis reports.

The desktop application can be launched programmatically using.

```bash
python -m rinet.gui
```

## Outputs and Reproducibility

Each RINet analysis produces a set of artifacts intended to support reproducibility and review. These include residue-indexed numerical outputs, visualization-ready plots, and metadata files that record structure identifiers, graph construction parameters, propagation settings, and runtime configuration.

By exporting both results and assumptions, RINet enables exact reproduction of analyses and direct comparison across runs.

## Validation & Limitations 

RINet has been evaluated on hemoglobin, a canonical allosteric protein with well-characterized structure-function relationships. Scan inference on hemoglobin shows statistically significant enrichment of experimentally established allosteric residues compared to random expectation. This demonstrates that RINet captures meaningful structure-based coupling patterns using static structural information alone.

RINet is intended for exploratory and comparative analysis rather than quantitative free-energy prediction. RINet does not explicitly simulate atomic dynamics and does not compute thermodynamic quantities. Results depend on graph construction choices and should be interpreted comparatively rather than as absolute predictors of biological outcome.

RINet is designed for exploratory allosteric analysis, comparative profiling of protein variants, hypothesis generation prior to molecular dynamics simulations, and methodological or educational research in structural biology and computational biophysics.

## Citation

If you use RINet in academic work, please cite.

RINet: A Deterministic Residue Interaction Network Framework for Interpretable Structure-Based Allosteric Analysis  
Akul Kumar, 2025
