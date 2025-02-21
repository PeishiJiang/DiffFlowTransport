"""The differentiable flow and transport model."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

import os
import json
from pathlib import Path

# import equinox as eqx
import jax.numpy as jnp
import pandas as pd

# from .flow import RNNFlow
# from .transport import SASTransport
# from .sas import initialize_sas_model
# from .utils import get_scaler
from .utils import load_flow_model, load_transport_model
from .utils import save_flow_model, save_transport_model
from .utils import predict_dl, scale_df
from .utils import make_pytorch_timeseries_dataloader

from typing import Dict, Optional

# TODO: We need a nice documentation of all the configurations.


class FlowTransport(object):

    def __init__(
        self, f_transport, f_flow, Q_scaler,
        flow_transport_coupling_type
    ):
        """The flow and transport model.

        Args:
            f_transport (str): The model file of the transport model.
            f_flow (str): The model file of the flow model.
            Q_scaler (Callable): The streamflow inverse scaler function.
            flow_transport_coupling_type (str): The type of flow and transport coupling.
        
        """
        # Load the transport model
        self.transport_model = load_transport_model(f_transport)

        # Load the flow model
        self.flow_model = load_flow_model(f_flow)

        # Streamflow scaler
        self.Q_scaler = Q_scaler

        # Coupling type
        self.flow_transport_coupling_type = flow_transport_coupling_type
    
    def run_flow(self, data_loader):
        Q_norm = predict_dl(data_loader, self.flow_model)
        return self.Q_scaler(Q_norm)

    def run_transport(
        self, J, C_J, ET, Q=None, sas_Q_args=None, sas_ET_args=None, *, dl=None
    ):
        flow_model = self.flow_model
        transport_model = self.transport_model
        coupling_type = self.flow_transport_coupling_type

        # Initial condition
        sTmT_init = jnp.zeros([J.size, 1+transport_model.nm])

        # Calculate the streamflow if not given
        if Q is None:
            Q = self.run_flow(dl)
        
        # Get the arguments of the SAS function
        if coupling_type == 0: # Take outflux as arguments
            sas_Q_args = Q[:,None]
            sas_ET_args = ET[:,None]
        
        elif coupling_type == 1: # Take LSTM hidden states as arguments
            hidden_states = predict_dl(dl, flow_model.calculate_hidden_states)
            sas_Q_args = hidden_states
            sas_ET_args = hidden_states
        
        elif coupling_type == 2: # Take outflux and LSTM hidden states as arguments
            hidden_states = predict_dl(dl, flow_model.calculate_hidden_states)
            sas_Q_args = jnp.concat([hidden_states, Q[:,None]], axis=1)
            sas_ET_args = jnp.concat([hidden_states, ET[:,None]], axis=1)

        # Run transport model
        transport_output = transport_model(J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args)

        return transport_output, Q


    # def run_coupled_flow_transport(
    #     self, flow_dl, J, C_J, Q, ET, sas_Q_args, sas_ET_args
    # ):
    #     flow_model = self.flow_model
    #     transport_model = self.transport_model
    #     Q_scaler = self.Q_scaler
    #     coupling_type = self.flow_transport_coupling_type

    #     # Run the flow model
    #     return

def save_model(
    configs, flow_model, transport_model, 
    f_configs='configs.json', saved_folder=Path("./models")
):
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
    current_folder = os.getcwd()
    # Go to the saved folder
    os.chdir(saved_folder)

    with open(f_configs, "r") as f:
        # hyperparams = json.loads(f.readline().decode())
        hyperparams = json.loads(f.readline())
    watershed = hyperparams['watershed_name']

    # Load the data
    f_data = hyperparams['f_data']
    if watershed.lower() == 'plynlimon':
        df = pd.read_csv(f_data, index_col=1)
        df.index = pd.to_datetime(df.index, format='%m/%d/%y')

    # Load the data loader for the flow model
    flow_dl_configs = hyperparams['flow_dl_configs']
    train_configs = hyperparams['train_configs']
    # train_start, train_end = train_configs['train_start'], train_configs['train_end']
    # test_start, test_end = train_configs['test_start'], train_configs['test_end']

    Q_varn = flow_dl_configs['targets'][0]
    scaler, df_norm = scale_df(df, train_configs['scaler_type'])
    Q_scaler = scaler[Q_varn].inverse_transform

    flow_data_loader = make_pytorch_timeseries_dataloader(
        df_norm, **flow_dl_configs, shuffle=False
    )

    # Load the data for the transport model
    cutoff_length = flow_dl_configs['sequence_length']
    df_cut = df.iloc[cutoff_length:]
    transport_data = get_fluxes_from_df_plynlimon(df_cut)

    # Load the flow and transport model
    flow_transport_coupling_type = hyperparams['flow_transport_coupling_type']
    f_flow_model = hyperparams['f_flow']
    f_transport_model = hyperparams['f_transport']
    model = FlowTransport(
        f_transport_model, f_flow_model, Q_scaler, 
        flow_transport_coupling_type
    )

    # Get the loss
    flow_loss = hyperparams['flow_loss']
    transport_loss = hyperparams['transport_loss']
    loss = {"flow_loss": flow_loss, "transport_loss": transport_loss}

    # Change it back
    os.chdir(current_folder)

    return model, loss, flow_data_loader, transport_data, hyperparams


def get_fluxes_from_df_plynlimon(df):
    J = jnp.array(df['J'].values)
    Q = jnp.array(df['Q'].values)
    ET = jnp.array(df['ET'].values)
    C_J = jnp.array(df[['Cl mg/l']].values)
    C_Q_J = jnp.array(df['Q Cl mg/l'].values)
    time = df.index

    return J, Q, ET, C_J, C_Q_J, time

# class DiffFlowTransport(object):

#     def __init__(
#         self, transport_model: Optional[SASTransport]=None, 
#         flow_model: Optional[RNNFlow]=None, 
#         transport_params: Optional[Dict]=None, 
#         flow_params: Optional[Dict]=None
#     ):
#         """**Arguments:**

#         - `transport_model`: The given transport model `SASTransport`.
#         - `flow_model`: The given flow model `RNNFlow`.
#         - `transport_params`: The parameters of the transport model is `transport_model`=None. `Dict`
#         - `flow_params`: The parameters of the flow model is `flow_model`=None. `Dict`
        
#         """
#         # The transport model
#         if transport_model is None and transport_params is None:
#             raise Exception("Transport model or its parameters is not given")

#         elif transport_model is not None:
#             self.transport = transport_model
        
#         # If the transport model is not given, 
#         # we initialize the transport model based on its configuration
#         elif transport_params is not None:
#             params = transport_params['transport_specs']
#             sas_Q_params = transport_params['sas_specs']['Q SAS fun']
#             sas_ET_params = transport_params['sas_specs']['ET SAS fun']
#             # Initialize the sas function
#             sas_Q = initialize_sas_model(sas_Q_params)
#             sas_ET = initialize_sas_model(sas_ET_params)
#             # Initialize the model
#             self.transport = SASTransport(sas_Q, sas_ET, **params)
        
#         # The flow model
#         if flow_model is None and flow_params is None:
#             raise Exception("Flow model or its parameters is not given")

#         elif flow_model is not None:
#             self.flow = flow_model
        
#         # If the flow model is not given, 
#         # we initialize the flow model based on its configuration
#         elif flow_params is not None:
#             # Get the scalers
#             xdata, ydata = flow_params['xdata'], flow_params['ydata']
#             xscaler_type = flow_params['xscaler_type']
#             yscaler_type = flow_params['yscaler_type']
#             xscaler = get_scaler(xdata, xscaler_type)
#             yscaler = get_scaler(ydata, yscaler_type)
#             # Initialize the flow model
#             params = flow_params["flow_specs"]
#             self.flow = RNNFlow(xscaler, yscaler, **params)


#     def __call__(self, flow_inputs, J, C_J, Q, ET, sTmT_init):
#         """**Arguments:**

#         - `flow_inputs`: The flow model inputs. `(nt, nx)`
#         - `J`: The given flow model `RNNFlow`. `(nt,)`
#         - `C_J`: The parameters of the transport model is `transport_model`=None. `(nt, nm)`
#         - `Q`: The parameters of the flow model is `flow_model`=None. `(nt,)`
#         - `ET`: The parameters of the flow model is `flow_model`=None. `(nt,)`
#         - `sTmT_init`: The parameters of the flow model is `flow_model`=None. `(nτ, 1+nm)`
        
#         """
#         # Run the flow model
#         # TODO:
#         ht, ct, Qrnn = self.flow.calculate_all_states(flow_inputs)

#         # Run the transport model
#         sas_Q_args = ht
#         sas_ET_args = ht
#         sT, mT, mQETs, pQETs, mRs, C_Q = self.transport(
#             J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args)
        
#         return sT, mT, mQETs, pQETs, mRs, C_Q

