"""Determining the number of SAS function inputs."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

import jax.numpy as jnp

from typing import Optional, List

from ..utils import predict_dl


def get_sas_inputs_amount(flow_transport_coupling_type, n_flow_hidden=None):
    """Function for getting the number of SAS function environment dependencies."""
    if flow_transport_coupling_type in [0,3]:
        n_sas_input = 1

    elif flow_transport_coupling_type == 1:
        if n_flow_hidden is None:
            raise Exception(
                f"Number of flow model hidden states are not given with flow_transport_coupling_type {flow_transport_coupling_type}."
            )
        n_sas_input = n_flow_hidden

    elif flow_transport_coupling_type in [2,4]:
        if n_flow_hidden is None:
            raise Exception(
                f"Number of flow model hidden states are not given with flow_transport_coupling_type {flow_transport_coupling_type}."
            )
        n_sas_input = n_flow_hidden + 1

    else:
        n_sas_input = 1
    
    return n_sas_input


def get_sas_inputs(
    flow_transport_coupling_type, Q, ET=None, flow_model=None, dl=None,
    logQminmax=None, ETminmax=None
):
    """Function for getting SAS function environment dependencies."""
    if ET is None:
        ET = Q
    
    # Calculate normalized Q and ET
    # TODO: logQ max/min and ET max/min should be fixed and treated as inputs
    if flow_transport_coupling_type in [3,4]:
        # Calculate logQ
        logQ = jnp.log10(Q)
        logQ = logQ.at[~jnp.isfinite(logQ)].set(logQ[jnp.isfinite(logQ)].min())
        
        # Get the bounds of logQ
        if logQminmax is not None:
            logQmin, logQmax = logQminmax[0], logQminmax[1]
            # # Read the minimum from the loqQ is the given Qmin is not positive.
            # logQmin = jnp.log10(Qmin) if Qmin > 0.0 else logQ.min()
            # logQmax = jnp.log10(Qmax)
        else:
            logQmin, logQmax = logQ.min(), logQ.max()
        
        # Get the bounds of ET
        if ETminmax is not None:
            ETmin, ETmax = ETminmax[0], ETminmax[1]
        else:
            ETmin, ETmax = ET.min(), ET.max()
    
        # Calculate the normalized logQ and ET
        logQ_norm = (logQ - logQmin) / (logQmax - logQmin)
        ET_norm = (ET - ETmin) / (ETmax - ETmin)
        # print(Qmin, Qmax)
        # print(logQ_norm)
        # print(ET_norm)

    # Calculate the SAS function environment dependencies
    if flow_transport_coupling_type == 0: # Take outflux as arguments
        sas_Q_args = jnp.array(Q)[:,None]
        sas_ET_args = jnp.array(ET)[:,None]
    
    elif flow_transport_coupling_type == 1: # Take LSTM hidden states as arguments
        if flow_model is None:
            raise Exception(
                f'The flow model is not given with flow_transport_coupling_type {flow_transport_coupling_type}.'
            )
        if dl is None:
            raise Exception(
                f'The dataloader of the flow model is not given with flow_transport_coupling_type {flow_transport_coupling_type}.'
            )
        hidden_states = predict_dl(dl, flow_model.calculate_hidden_states)
        sas_Q_args = hidden_states
        sas_ET_args = hidden_states
    
    elif flow_transport_coupling_type == 2: # Take outflux and LSTM hidden states as arguments
        if flow_model is None:
            raise Exception(
                f'The flow model is not given with flow_transport_coupling_type {flow_transport_coupling_type}.'
            )
        if dl is None:
            raise Exception(
                f'The dataloader of the flow model is not given with flow_transport_coupling_type {flow_transport_coupling_type}.'
            )
        hidden_states = predict_dl(dl, flow_model.calculate_hidden_states)
        sas_Q_args = jnp.concat([hidden_states, Q[:,None]], axis=1)
        sas_ET_args = jnp.concat([hidden_states, ET[:,None]], axis=1)

    elif flow_transport_coupling_type == 3: # Take logarized and normalized outflux
        sas_Q_args = logQ_norm[:,None]
        sas_ET_args = ET_norm[:,None]

    elif flow_transport_coupling_type == 4: # Take logarized and normalized outflux and LSTM hidden states as arguments
        hidden_states = predict_dl(dl, flow_model.calculate_hidden_states)
        sas_Q_args = jnp.concat([hidden_states, logQ_norm[:,None]], axis=1)
        sas_ET_args = jnp.concat([hidden_states, ET_norm[:,None]], axis=1)

    else: # Take outflux as arguments
        sas_Q_args = jnp.array(Q)[:,None]
        sas_ET_args = jnp.array(ET)[:,None]
    
    return sas_Q_args, sas_ET_args