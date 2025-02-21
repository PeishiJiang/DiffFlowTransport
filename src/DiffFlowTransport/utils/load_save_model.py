"""Functions for loading and saving models."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

import json
from typing import Dict

import equinox as eqx

from ..flow import LSTM
from ..transport import SASTransport
from ..sas import initialize_sas_model


def save_transport_model(filename: str, hyperparams: Dict, model: eqx.Module) -> None:
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_transport_model(filename: str) -> SASTransport:
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        # transport model parameters
        params = hyperparams['transport_specs']

        # SAS model parameters
        sas_Q_params = hyperparams['sas_specs']['Q SAS fun']
        sas_Q = initialize_sas_model(sas_Q_params)
        sas_ET_params = hyperparams['sas_specs']['ET SAS fun']
        sas_ET = initialize_sas_model(sas_ET_params)

        # transport model
        model_skeleton = SASTransport(sas_Q, sas_ET, **params)
        model = eqx.tree_deserialise_leaves(f, model_skeleton)

    return model


def save_flow_model(filename: str, hyperparams: Dict, model: eqx.Module) -> None:
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_flow_model(filename: str) -> eqx.Module:
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())

        # flow model
        model_skeleton = LSTM(**hyperparams)
        model = eqx.tree_deserialise_leaves(f, model_skeleton)

    return model

