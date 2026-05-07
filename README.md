# DiffSAS: A Differentiable StorAge Selection Function-based Transport Model

<!-- <p align="center">
<img src="./doc/JAX-CanVeg-v2.png" alt="fishy" class="center" width="400px">
</p> -->

## Table of Contents
- [Overview](#overview)
- [Repo structure](#repo-structure)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Acknowledgments](#acknowledgements)
- [Citation](#citation)
- [Contacts](#contacts)

## Overview
Uncovering transit time distributions (TTDs) remains a long-standing challenge in understanding watershed transport processes. The StorAge Selection (SAS) function provides a numerically tractable way to derive TTDs from tracer data, but existing SAS models rely on predefined probability distributions, which require prior knowledge and limit flexibility. To address this, we propose a generic SAS formulation using a Mixed Density Network (MDN), representing the SAS function as a weighted combination of three base distributions: Gamma (skewed), Gaussian (symmetric), and Uniform (flat). This approach is implemented in a fully differentiable SAS-based transport model, \texttt{DiffSAS}, leveraging the JAX library for efficient training via automatic differentiation.


## Repo Structure
```
.
+-- src/DiffFlowTransport
+-- Plynlimon
+-- OakCreek
+-- environment.yml
+-- README.md
```
- `src/DiffFlowTransport`: providing source codes for DiffSAS.
- `Plynlimon`: providing example codes/notebooks for running DiffSAS at the Lower Hafren watershed (Plynlimon, UK).
- `OakCreek`: providing example codes/notebooks for running DiffSAS at the Oak Creek watershed (WA, USA).
- `environment.yml`: the YAML file for creating the conda virtual environment.
- `README.md`: the readme file.

## Installation
1. Close the repository
```
git clone https://github.com/PeishiJiang/DiffFlowTransport.git
```

2. Create the conda virtual environment:
```
conda env create -f environment.yml
```

3. Activate the virtual environment:
```
conda activate nn-flow-transport
```

## Getting Started
TBD
<!-- We suggest training and running JAX-CanVeg by providing a JSON-based configuration file. See this [post](./doc/MODEL_CONFIG.md) for details. -->



## Examples

### The Lower Hafren Watershed (Plynlimon, UK)
TBD

### The Oak Creek Watershed (WA, USA)
TBD
<!-- <p align="center">
<img src="./doc/HybridBB-v2.png" alt="fishy" class="center" width="600px">
</p> -->

<!-- We demonstrated JAX-CanVeg's hybrid modeling capability by applying the model to simulate the water and carbon fluxes at four flux tower sites in the western United States with varying aridity. To this end, we developed a hybrid version of the Ball-Berry equation that emulates the impact of water stress on stomatal closure (Jiang et al., 2024). The scripts for reproducing the results of the paper are available in the folder `examples`. We applied the differentiable JAX-CanVeg at four flux tower sites to evaluate the performance of a hybrid version of the Ball-Berry equation. The model were trained against both observed latent heat fluxes and net ecosystem exchange. Below we illustrate the application example on [US-Whs](./examples/US-Whs) (which is applicable to the other three flux tower sites, i.e., [US-Me2](./examples/US-Me2), [US-Bi1](./examples/US-Bi1), and [US-Hn1](./examples/US-Hn1)).


- **Step 1:** Train the process-based and hybrid JAX-CanVeg and the pure neural networks:
```sh
python [jax-canveg-folder]/examples/US-Whs/train_models.py
python [jax-canveg-folder]/examples/US-Whs/train_dnns.py
```

> [!NOTE]
> Training multiple JAX-CanVeg models will take a pretty long time. It is suggested to train it in an HPC system. We provide an example of sbatch job script [here](./examples/nersc_job.sh).

- **Step 2:** Evaluate the simulation performance of the trained models including both JAX-CanVeg and DNNs:
```sh
cd [jax-canveg-folder]/examples/US-Whs
python postprocessing.py
```

- **Step 3:** Calculate the parameter sensitivity of selected models:
```sh
cd [jax-canveg-folder]/examples/US-Whs
python calculate_sensitivity.py
```

> [!NOTE]
> This step requires the completion of model training at all four sites. One can modify the code to calculate the sensitivity at specified sites.

- **Step 4:** Visualize the training results using [TrainingAnalysis.ipynb](./examples/TrainingAnalysis.ipynb)

- **Step 5:** Visualize the sensitivity analysis results using [SensitivityAnalysis.ipynb](./examples/SensitivityAnalysis.ipynb)

Some simulations of the trained JAX-CanVeg --
<p align="center">
<img src="./doc/Simulation-US-Whs-ML-0.5.png" alt="fishy" class="center" width="800px">
</p> -->


## Acknowledgements
This research was supported by both the startup package of Peishi Jiang at the University of Alabama and the U.S. Department of Energy (DOE), Office of Science (SC) Biological and Environmental Research (BER) program, as part of BER's Environmental System Science (ESS) program. This contribution originates from the River Corridor Scientific Focus Area (SFA) at Pacific Northwest National Laboratory. This research used resources of the National Energy Research Scientific Computing Center (NERSC), a DOE Office of Science User Facility supported by the Office of Science of the United States Department of Energy under contract DE-AC02-05CH11231.  Pacific Northwest National Laboratory is operated for the DOE by Battelle Memorial Institute under contract DE-AC05-76RL01830. This paper describes objective technical results and analysis. Any subjective views or opinions that might be expressed in the paper do not necessarily represent the views of the U.S. Department of Energy or the United States Government.


## Citation
Jiang, P., Harman, C. J., Niroula, S., Li, Z., & Chen, X. (2025). Deciphering Watershed Transport Process: A Differentiable Hybrid SAS Function Approach. Authorea Preprints.

## Contacts
Peishi Jiang (peishi.jiang@ua.edu)