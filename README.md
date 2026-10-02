# DiffSAS: A Differentiable StorAge Selection Function-based Transport Model

## Table of Contents
- [Overview](#overview)
- [Repo Structure](#repo-structure)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Reproducing the Paper](#reproducing-the-paper)
- [Data](#data)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [Contacts](#contacts)

## Overview
Uncovering transit time distributions (TTDs) remains a long-standing challenge in understanding watershed transport processes. The StorAge Selection (SAS) function provides a numerically tractable way to derive TTDs from tracer data, but existing SAS models rely on predefined probability distributions, which require prior knowledge and limit flexibility. To address this, we propose a generic SAS formulation using a Mixture Density Network (MDN), representing the SAS function as a weighted combination of three base distributions: Gamma (skewed), Gaussian (symmetric), and Uniform (flat). This approach is implemented in a fully differentiable SAS-based transport model, `DiffSAS`, leveraging the JAX library for efficient training via automatic differentiation.

`DiffSAS` is coupled to an LSTM-based flow model: the LSTM's hidden states can condition the MDN SAS function, and its predicted streamflow can replace observed streamflow, so that transport can be simulated from meteorological forcing alone ("operational mode").

## Repo Structure
```
.
+-- src/DiffFlowTransport
|   +-- sas/                 SAS functions (Uniform, Gamma, storage-dependent Gamma, MDN SAS)
|   +-- transport/           SAS transport model (finite-volume solver)
|   +-- flow/                LSTM flow model
|   +-- utils/               training loops, losses, metrics, data loaders, plotting
+-- Plynlimon/               Lower Hafren watershed (Plynlimon, UK): data, training and postprocessing
+-- OakCreek/                Oak Creek watershed (WA, USA): data, training, benchmark and postprocessing
+-- make_combined_figures.py two-watershed figures for the paper
+-- environment.yml          conda environment
+-- README.md
```

## Installation
1. Clone the repository:
```sh
git clone https://github.com/PeishiJiang/DiffFlowTransport.git
cd DiffFlowTransport
```

2. Create and activate the conda environment (JAX is built for CUDA 12):
```sh
conda env create -f environment.yml
conda activate nn-flow-transport
```

3. Make the package importable by adding `src` to `PYTHONPATH`:
```sh
export PYTHONPATH=${PYTHONPATH}:$(pwd)/src
```

## Getting Started
Each site folder trains an ensemble of MDN-based SAS models coupled to an LSTM flow model. The MDN is conditioned on one of three coupling types (`--couplingtype`):

| Coupling type | Model | MDN conditioning variables |
|---|---|---|
| 1 | MDN<sub>LSTM</sub> | LSTM hidden states *h* |
| 3 | MDN<sub>Q</sub> | normalized log<sub>10</sub> *Q* |
| 4 | MDN<sub>Q,LSTM</sub> | *h* and normalized log<sub>10</sub> *Q* |

Each configuration is an 11-member ensemble: a base model (random seed 42, MLP depth 2 and width 10), five random-seed variants (`--randomseed 0, 1, 10, 20, 1234`), and five architecture variants (`--sasmlpdepth 3-7 --sasmlpwidth 20, 16, 12, 8, 4`). See `job.sh` in each site folder.

Each trained member is saved in the site's model folder as:
- `configs-*.json`: model and training configurations, including the loss curves;
- `flow_model_*.eqx` and `transport_model_*.eqx`: trained LSTM flow and transport models;
- `sim-*.pkl`: simulated streamflow and tracer concentrations.

## Reproducing the Paper
The results in the paper are from `Plynlimon/models-rev2/` and `OakCreek/models-withDiffusion-rev2/`. Run the steps below in each site folder (`Plynlimon/` or `OakCreek/`).

- **Step 1:** Train the LSTM flow model and the SAS model ensembles (GPU):
```sh
bash job.sh
```

> [!NOTE]
> Training the ensembles takes a long time; an HPC system is recommended (see `nersc_job.sh`). `job.sh` calls `train-model-mdn2.py` once per ensemble member.

- **Step 2:** Calibrate the storage-dependent Gamma benchmark (Γ<sub>dynamic</sub>):
  - Oak Creek: run the Nelder-Mead calibration on CPU, which writes `models-withDiffusion-rev2/storage-dependent-gamma-detrended_cumulative-mse.json`:
    ```sh
    python train-StorageDependentGamma.py
    ```
  - Lower Hafren: the benchmark parameters are set in `Postprocess-UQ.py`, with the relative storage taken from the `S_scale` column of `data-agg.csv` (see `CleanData.ipynb`).

- **Step 3:** Evaluate the ensembles and the benchmark, and plot the per-site figures:
```sh
python Postprocess-UQ.py
```
This writes the per-site figures and the performance metrics of all members (`figs/df_metrics.csv`) to `figs/`.

- **Step 4:** From the repo root, assemble the two-watershed figures in `paper/FiguresR2/`:
```sh
python make_combined_figures.py
```

## Data
- **Lower Hafren** (`Plynlimon/data-agg.csv`): daily precipitation, chloride concentration in precipitation, streamflow, evapotranspiration and stream chloride concentration from the Plynlimon experimental catchments (UK Centre for Ecology & Hydrology), 1983-05-03 to 2008-12-31, the same dataset as Harman (2015).
- **Oak Creek** (`OakCreek/data_withDiffusion_withSpinup-correctedZeroFlow.csv`): watershed-scale fluxes and passive tracer concentrations simulated by the Advanced Terrestrial Simulator (ATS). The influx *J* is rainfall plus snowmelt (column `P2`).

## Acknowledgements
This research was supported by both the startup package of Peishi Jiang at the University of Alabama and the U.S. Department of Energy (DOE), Office of Science (SC) Biological and Environmental Research (BER) program, as part of BER's Environmental System Science (ESS) program. This contribution originates from the River Corridor Scientific Focus Area (SFA) at Pacific Northwest National Laboratory. This research used resources of the National Energy Research Scientific Computing Center (NERSC), a DOE Office of Science User Facility supported by the Office of Science of the United States Department of Energy under contract DE-AC02-05CH11231.  Pacific Northwest National Laboratory is operated for the DOE by Battelle Memorial Institute under contract DE-AC05-76RL01830. This paper describes objective technical results and analysis. Any subjective views or opinions that might be expressed in the paper do not necessarily represent the views of the U.S. Department of Energy or the United States Government.

## Citation
Jiang, P., Harman, C. J., Niroula, S., Li, Z., & Chen, X. (in review). Inferring Time-Variable Watershed Transport: A Hybrid SAS Function Approach Via Differentiable Modeling. *Water Resources Research*.

## Contacts
Peishi Jiang (peishi.jiang@ua.edu)
