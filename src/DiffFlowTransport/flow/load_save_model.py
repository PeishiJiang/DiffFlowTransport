"""Functions for loading and saving the flow model."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

import json
from typing import Dict

import equinox as eqx

from ..flow import LSTM


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

