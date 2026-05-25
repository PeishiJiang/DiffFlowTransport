#!/bin/bash
# python ./train-model-mdn2-v3.py --cudadevice 1 --couplingtype 1
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 1
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 3
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 4
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 4 --randomseed 0
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 4 --randomseed 1
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 4 --randomseed 1234
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 4 --randomseed 10
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 4 --randomseed 20
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 4 --sasmlpdepth 3 --sasmlpwidth 20
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 4 --sasmlpdepth 4 --sasmlpwidth 16
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 4 --sasmlpdepth 5 --sasmlpwidth 12
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 4 --sasmlpdepth 6 --sasmlpwidth 8
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 4 --sasmlpdepth 7 --sasmlpwidth 4
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 1 --randomseed 0
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 1 --randomseed 1
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 1 --randomseed 1234
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 1 --randomseed 10
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 1 --randomseed 20
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 1 --sasmlpdepth 3 --sasmlpwidth 20
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 1 --sasmlpdepth 4 --sasmlpwidth 16
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 1 --sasmlpdepth 5 --sasmlpwidth 12
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 1 --sasmlpdepth 6 --sasmlpwidth 8
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 1 --sasmlpdepth 7 --sasmlpwidth 4
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 3 --randomseed 0
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 3 --randomseed 1
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 3 --randomseed 1234
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 3 --randomseed 10
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 3 --randomseed 20
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 3 --sasmlpdepth 3 --sasmlpwidth 20
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 3 --sasmlpdepth 4 --sasmlpwidth 16
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 3 --sasmlpdepth 5 --sasmlpwidth 12
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 3 --sasmlpdepth 6 --sasmlpwidth 8
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 3 --sasmlpdepth 7 --sasmlpwidth 4

python ./train-model-mdn2.py --cudadevice 1 --couplingtype 1 --onlyrain True
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 3 --onlyrain True
python ./train-model-mdn2.py --cudadevice 1 --couplingtype 4 --onlyrain True

wait