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

- **Step 1:** Train the LSTM-based flow model and the ensemble SAS models:
```sh
bash [diffsas]/OakCreek/job.sh
```

> [!NOTE]
> Training multiple SAS models will take a pretty long time. It is suggested to train it in an HPC system. The shell scripts basically call [train-model-mdn2-v3.py](./OakCreek/train-model-mdn2-v3.py) for each SAS model training.

- **Step 2:** Predict the streamflow and solute dynamics using the trained models:
```sh
cd [diffsas]/OakCreek
python Postprocess-calibrated-runs.py.py
```

- **Step 3:** Postprocess the trained SAS models via python scripts:
```sh
cd [diffsas]/OakCreek
python Postprocess-UQ.py
```

- **[Optional] Step 4:** Postprocess the trained SAS models by using [Postprocess-UQ.ipynb](./OakCreek/Postprocess-UQ.ipynb) and [Postprocess_mdn.ipynb](./OakCreek/Postprocess_mdn.ipynb)

<!-- ### The Lower Hafren Watershed (Plynlimon, UK) -->


<!-- ### The Oak Creek Watershed (WA, USA) -->


## Acknowledgements
This research was supported by both the startup package of Peishi Jiang at the University of Alabama and the U.S. Department of Energy (DOE), Office of Science (SC) Biological and Environmental Research (BER) program, as part of BER's Environmental System Science (ESS) program. This contribution originates from the River Corridor Scientific Focus Area (SFA) at Pacific Northwest National Laboratory. This research used resources of the National Energy Research Scientific Computing Center (NERSC), a DOE Office of Science User Facility supported by the Office of Science of the United States Department of Energy under contract DE-AC02-05CH11231.  Pacific Northwest National Laboratory is operated for the DOE by Battelle Memorial Institute under contract DE-AC05-76RL01830. This paper describes objective technical results and analysis. Any subjective views or opinions that might be expressed in the paper do not necessarily represent the views of the U.S. Department of Energy or the United States Government.


## Citation
Jiang, P., Harman, C. J., Niroula, S., Li, Z., & Chen, X. (2025). Deciphering Watershed Transport Process: A Differentiable Hybrid SAS Function Approach. Authorea Preprints.

## Contacts
Peishi Jiang (peishi.jiang@ua.edu)