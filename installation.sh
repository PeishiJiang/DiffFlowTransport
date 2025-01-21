#!/bin/sh
conda env create -f environment.yml
conda activate nn-flow-transport
wait

pip install hydroeval
pip install pytest
pip install equinox
pip install tqdm
pip install torch torchvision

# CPU-only (Linux/macOS/Windows)
pip install -U jax
# GPU (NVIDIA, CUDA 12)
#pip install -U "jax[cuda12]"
pip install optax