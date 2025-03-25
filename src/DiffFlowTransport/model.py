"""The differentiable flow and transport model."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

import os
import json
from pathlib import Path

# import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd

# from .flow import RNNFlow
# from .transport import SASTransport
# from .sas import initialize_sas_model
# from .utils import get_scaler
from .flow import load_flow_model, save_flow_model
from .transport import load_transport_model, save_transport_model
from .utils import predict_dl, scale_df
from .utils import make_pytorch_timeseries_dataloader
from .utils import read_obs_csv, get_transport_obs_fluxes

from .sas import SAS_MDN, SAS_MDN2, SAS_GammaMDN, SAS_NormalMDN
from .sas import get_sas_inputs, get_sas_inputs_amount

# from typing import Dict, Optional

# TODO: We need a nice documentation of all the configurations.


class FlowTransport(object):

    def __init__(
        self, f_transport, f_flow, Q_scaler,
        flow_transport_coupling_type, logQminmax, ETminmax
    ):
        """The flow and transport model.

        Args:
            f_transport (str): The model file of the transport model.
            f_flow (str): The model file of the flow model.
            Q_scaler (Callable): The streamflow inverse scaler function.
            flow_transport_coupling_type (str): The type of flow and transport coupling.
            Qminmax (List): The min and max values of Q.
            ETminmax (List): The min and max values of ET.
        
        """
        # Load the transport model
        self.transport_model = load_transport_model(f_transport)

        # Load the flow model
        self.flow_model = load_flow_model(f_flow)

        # Streamflow scaler
        self.Q_scaler = Q_scaler

        # Q and ET bounds
        self.logQminmax, self.ETminmax = logQminmax, ETminmax

        # Coupling type
        self.flow_transport_coupling_type = flow_transport_coupling_type
    
    def run_flow(self, data_loader):
        Q_norm = predict_dl(data_loader, self.flow_model)
        return self.Q_scaler(Q_norm).flatten()

    def run_transport(
        self, J, C_J, ET, Q=None, sas_Q_args=None, sas_ET_args=None, *, dl=None
    ):
        flow_model = self.flow_model
        transport_model = self.transport_model
        coupling_type = self.flow_transport_coupling_type

        logQminmax, ETminmax = self.logQminmax, self.ETminmax

        # Initial condition
        τ_max = transport_model.τ_max
        τ_max = τ_max if τ_max is not None else J.size
        sTmT_init = jnp.zeros([τ_max, 1+transport_model.nm])

        # Calculate the streamflow if not given
        if Q is None:
            Q = self.run_flow(dl)
        
        # Get the arguments of the SAS function
        sas_Q_args, sas_ET_args = get_sas_inputs(
            coupling_type, Q, ET, flow_model, dl, logQminmax, ETminmax
        )

        # Run transport model
        transport_output = transport_model(J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args)

        return transport_output, Q
    
    # def get_mdn_weights(self, Q=None, ET=None, sas_Q_args=None, sas_ET_args=None, dl=None):
    def get_mdn_weights(self, Q=None, ET=None, dl=None):
        """Function for getting the weights of distributions used in the MDN"""
        flow_model = self.flow_model
        transport_model = self.transport_model
        coupling_type = self.flow_transport_coupling_type

        logQminmax, ETminmax = self.logQminmax, self.ETminmax

        # Calculate the streamflow if not given
        if Q is None:
            Q = self.run_flow(dl)

        # Check whether the SAS function uses the MDN model
        # TODO: check sas_ET in the future
        if not isinstance(transport_model.sas_Q, (SAS_MDN, SAS_MDN2, SAS_GammaMDN, SAS_NormalMDN)):
            raise Exception('The streamflow SAS function does not use a MDN model.')
        
        sas_Q_args, sas_ET_args = get_sas_inputs(
            # coupling_type, Q, ET, flow_model.calculate_hidden_states, dl
            coupling_type, Q, ET, flow_model, dl, logQminmax, ETminmax
        )
        
        # if coupling_type == 0: # Take outflux as arguments
        #     sas_Q_args = Q[:,None]
        #     # sas_ET_args = ET[:,None]
        
        # elif coupling_type == 1: # Take LSTM hidden states as arguments
        #     hidden_states = predict_dl(dl, flow_model.calculate_hidden_states)
        #     sas_Q_args = hidden_states
        #     # sas_ET_args = hidden_states
        
        # elif coupling_type == 2: # Take outflux and LSTM hidden states as arguments
        #     hidden_states = predict_dl(dl, flow_model.calculate_hidden_states)
        #     sas_Q_args = jnp.concat([hidden_states, Q[:,None]], axis=1)
        #     # sas_ET_args = jnp.concat([hidden_states, ET[:,None]], axis=1)
        
        results = jax.vmap(transport_model.sas_Q.get_param)(sas_Q_args)
        weights = results[0]

        return weights


def save_model(
    configs, flow_model, transport_model, 
    f_configs='configs.json', saved_folder=Path("./models")
):
    """Function for saving a flow and transport model."""
    # Save the overall configurations
    with open(saved_folder / f_configs, "wb") as f:
        hyperparam_str = json.dumps(configs)
        f.write((hyperparam_str + "\n").encode())
    
    f_flow, f_transport = configs['f_flow'], configs['f_transport']
    flow_params, transport_params = configs['flow_configs'], configs['transport_configs']

    # Save the models
    save_flow_model(saved_folder / f_flow, flow_params, flow_model)
    save_transport_model(saved_folder / f_transport, transport_params, transport_model)


def load_model(f_configs, saved_folder=Path("./models")):
    """Function for loading a flow and transport model."""
    current_folder = os.getcwd()
    # Go to the saved folder
    os.chdir(saved_folder)

    with open(f_configs, "r") as f:
        hyperparams = json.loads(f.readline())
    watershed = hyperparams['watershed_name']

    # Load the data
    f_data = hyperparams['f_data']
    df = read_obs_csv(f_data, watershed)

    # Load the data loader for the flow model
    flow_dl_configs = hyperparams['flow_dl_configs']
    train_configs = hyperparams['train_configs']

    Q_varn = flow_dl_configs['targets'][0]
    scaler, df_norm = scale_df(df, train_configs['scaler_type'])
    Q_scaler = scaler[Q_varn].inverse_transform

    flow_data_loader = make_pytorch_timeseries_dataloader(
        df_norm, **flow_dl_configs, shuffle=False
    )

    # Load the data for the transport model
    cutoff_length = flow_dl_configs['sequence_length']
    df_cut = df.iloc[cutoff_length:]
    transport_data = get_transport_obs_fluxes(df_cut, watershed)
    Q, ET = transport_data[1], transport_data[2]
    logQ = jnp.log10(Q)
    logQ = logQ.at[~jnp.isfinite(logQ)].set(logQ[jnp.isfinite(logQ)].min())
    logQminmax, ETminmax = [logQ.min(), logQ.max()], [ET.min(), ET.max()]

    # Load the flow and transport model
    flow_transport_coupling_type = hyperparams['flow_transport_coupling_type']
    f_flow_model = hyperparams['f_flow']
    f_transport_model = hyperparams['f_transport']
    model = FlowTransport(
        f_transport_model, f_flow_model, Q_scaler, 
        flow_transport_coupling_type, logQminmax, ETminmax
    )

    # Get the loss
    flow_loss = hyperparams['flow_loss']
    transport_loss = hyperparams['transport_loss']
    loss = {"flow_loss": flow_loss, "transport_loss": transport_loss}

    # Change it back
    os.chdir(current_folder)

    return model, loss, flow_data_loader, transport_data, hyperparams


# def get_fluxes_from_df_plynlimon(df):
#     J = jnp.array(df['J'].values)
#     Q = jnp.array(df['Q'].values)
#     ET = jnp.array(df['ET'].values)
#     C_J = jnp.array(df[['Cl mg/l']].values)
#     C_Q_J = jnp.array(df['Q Cl mg/l'].values)
#     time = df.index

#     return J, Q, ET, C_J, C_Q_J, time
