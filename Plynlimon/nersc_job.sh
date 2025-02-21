#!/bin/bash
#SBATCH -A m1800_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"

module load conda/Miniforge3-24.7.1-0
module load cudatoolkit
conda activate nn-flow-transport
export PYTHONPATH=${PYTHONPATH}:/global/cfs/cdirs/m1800/peishi/DataDrivenFlowTransport/src

srun --exact -u -n 1 -c 1 python ./train-model-mdn-gamma-0.py &
srun --exact -u -n 1 -c 1 python ./train-model-mdn-gamma-1.py &
srun --exact -u -n 1 -c 1 python ./train-model-mdn-one_gamma-1.py &
# srun --exact -u -n 1 -c 1 python ./train-model-mdn-one_gamma-2.py &
wait