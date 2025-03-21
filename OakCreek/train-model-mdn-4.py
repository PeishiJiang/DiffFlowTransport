# %%
from pathlib import Path

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

import optax

from DiffFlowTransport.transport import SASTransport
from DiffFlowTransport.flow import LSTM
from DiffFlowTransport.utils import scale_df, make_pytorch_timeseries_dataloader
from DiffFlowTransport.utils import get_transport_obs_fluxes
from DiffFlowTransport.utils import train_flow_model, train_transport_model, mse
from DiffFlowTransport.sas import initialize_sas_model
from DiffFlowTransport.sas import get_sas_inputs, get_sas_inputs_amount
from DiffFlowTransport.model import save_model


# %% [markdown]
# # General configurations

# %%
# Watershed
watershed_name = 'OakCreek'

# Data option
data_option = 'withDiffusion'

# The flow and transport model coupling type
flow_transport_coupling_type = 4

# Label
model_label = f'mdn-couplingtype{flow_transport_coupling_type}'


# %% [markdown]
# # Read the data

# %%
# f_data = f'./data_{data_option}.csv'
f_data = f'./data_{data_option}_withSpinup.csv'
df = pd.read_csv(f_data, index_col=0)
df.index = pd.to_datetime(df.index)
df.head()

# %% [markdown]
# # Parameters

# %% [markdown]
# ## Flow model configuration

# %%
flow_inputs = ['P2', 'Tair', 'SWIN']
# flow_inputs = ['P2']
flow_outputs = ['Q']

flow_params = {
    "n_input": len(flow_inputs),
    "n_output": len(flow_outputs),
    "n_hidden": 20,
    "key": 512,
}

# Skip the first several days
cutoff_length = 365


# %% [markdown]
# ## Transport model configuration

# %%
# Number of SAS inputs depends on the coupling type
n_sas_input = get_sas_inputs_amount(flow_transport_coupling_type, flow_params['n_hidden'])
    

# %%
transport_params = {
    "transport_specs": {
        "dt": 1.,
        "α_Q": 1.,
        "α_ET": 0.,
        "k1": [0.],
        "C_eq": [0.],
        "C_Q_old": [0.],
        # "τ_max": 4000,
        # "C_Q_old": [np.mean(C_J_cut * J_cut) / Q_cut.mean()]
    },
    "sas_specs": {
        "Q SAS fun": {
            "func": "MDN",
            "args": {
                "scale": 4000.0,
                "n_input": n_sas_input,  # Same as the number of hidden states used in the flow LSTM model
                "n_hidden": 10,
                "key": 20,
                "width_size": 10,
                "depth": 2
            }
        },
        "ET SAS fun": {
            "func": "Uniform",
            "args": {
                "scale": 400.0
            }
        }
    }
}


# %% [markdown]
# ## Training configuration and dataloader

# %%
flow_dl_config = {
    "targets": flow_outputs,
    "features": flow_inputs,
    "sequence_length": cutoff_length,
    "batch_size": 512,
}


# %%
train_config = {
    "learning_rate": 0.01,
    "epochs": 200,
    "train_start": '2015-10-01',
    "train_end": '2020-09-30',
    "test_start": '2021-10-01',
    "test_end": '2023-12-28',
    "scaler_type": "minmax"
}


# %% [markdown]
# # Initialize the flow and transport models

# %%
# The transport model
sas_Q = initialize_sas_model(transport_params["sas_specs"]["Q SAS fun"])
sas_ET = initialize_sas_model(transport_params["sas_specs"]["ET SAS fun"])
transport_model = SASTransport(sas_Q, sas_ET, **transport_params["transport_specs"])

# The flow model
flow_model = LSTM(**flow_params)


# %% [markdown]
# # Prepare optimizer and training data

# %% [markdown]
# ## The training data

# %%
train_start, train_end = train_config['train_start'], train_config['train_end']
test_start, test_end = train_config['test_start'], train_config['test_end']

scaler, df_norm = scale_df(df, train_config['scaler_type'])
df_norm_train = df_norm.loc[train_start:train_end].copy()
df_norm_test = df_norm.loc[test_start:test_end].copy()


# %%
train_start_ind = df.index.get_loc(train_start)
train_end_ind = df.index.get_loc(train_end)
test_start_ind = df.index.get_loc(test_start)
test_end_ind = df.index.get_loc(test_end)


# %% [markdown]
# ## Optimizer

# %%
optim = optax.adamw(train_config["learning_rate"])


# %% [markdown]
# # The flow model

# %% [markdown]
# ## Get the flow model training data

# %%
train_loader = make_pytorch_timeseries_dataloader(
    df_norm_train, **flow_dl_config, shuffle=True
)

train_eval_loader = make_pytorch_timeseries_dataloader(
    df_norm_train, **flow_dl_config, shuffle=False
)

test_loader = make_pytorch_timeseries_dataloader(
    df_norm_test, **flow_dl_config, shuffle=False
)

all_loader = make_pytorch_timeseries_dataloader(
    df_norm, **flow_dl_config, shuffle=False
)
next(iter(train_loader));

# %% [markdown]
# ## Train the flow model

# %%
flow_model_new, loss_train_flow, loss_test_flow = train_flow_model(
    flow_model, train_config['epochs'], mse, optim, train_loader, test_loader
)

# %% [markdown]
# # The transport model

# %% [markdown]
# ## Get the transport model training data

# %%
# TODO: skip the initial days with sequence length used by LSTM model
df_cut = df.iloc[cutoff_length:]
dt = transport_params['transport_specs']['dt']
J, Q, ET, C_J, C_Q_J, time = get_transport_obs_fluxes(df_cut, watershed_name)
τ_max = J.size if transport_model.τ_max is None else transport_model.τ_max
sTmT_init = jnp.zeros([τ_max, 2])


# %%
# SAS arguments
sas_Q_args, sas_ET_args = get_sas_inputs(flow_transport_coupling_type, Q, ET, flow_model_new, all_loader)
    

# %%
# Inputs and outputs
mask = jnp.where(~jnp.isnan(C_Q_J))
x = [J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args, mask]
y = C_Q_J[mask][:,None]

# Traing data
# During training, we run the model until the end of the training period
mask_train = mask[0][(mask[0]>train_start_ind) & (mask[0]<train_end_ind)]
x_train = [e[:train_end_ind] for e in x[:-1]] + [mask_train]
y_train = C_Q_J[mask_train][:,None]

# Test data
# During test, we run the model throughout the whole period but evaluate only on the test period
mask_test = mask[0][(mask[0]>test_start_ind) & (mask[0]<test_end_ind)]
x_test = [J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args, mask_test]
y_test = C_Q_J[mask_test][:,None]

y_train.shape, y_test.shape


# %% [markdown]
# ## Train the transport model

# %%
# Train the model
jax.clear_caches()
transport_model_new, loss_train_transport, loss_test_transport = train_transport_model(
    transport_model, train_config['epochs'], mse, optim, x_train, y_train, x_test, y_test
)

# %% [markdown]
# ## Make predictions

# %%
# Run the trained model
sT, mT, mQETs, pQETs, mRs, C_Q = transport_model_new(
    J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args
)
df_cut['C_Q pred'] = C_Q

# %% [markdown]
# # Save the models

# %%
f_flow = f'flow_model_{model_label}.eqx'
f_transport = f'transport_model_{model_label}.eqx'
f_configs = f'configs-{model_label}.json'
dir_models = Path(f'models-{data_option}')


# %%
# Save the overall configurations
configs = {
    "flow_configs": flow_params,
    "transport_configs": transport_params,
    "f_flow": f_flow,
    "f_transport": f_transport,
    "flow_loss": {
        "train": loss_train_flow.tolist(),
        "test": loss_test_flow.tolist()
    },
    "transport_loss": {
        "train": loss_train_transport.tolist(),
        "test": loss_test_transport.tolist()
    },
    "flow_dl_configs": flow_dl_config,
    "train_configs": train_config,
    "flow_transport_coupling_type": flow_transport_coupling_type,
    "f_data": "../" + f_data,
    "watershed_name": watershed_name
}

# %%
save_model(configs, flow_model_new, transport_model_new, f_configs, dir_models)


