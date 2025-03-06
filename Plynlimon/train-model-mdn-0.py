# %%
from pathlib import Path

import pandas as pd

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import equinox as eqx

import optax

from DiffFlowTransport.transport import SASTransport
from DiffFlowTransport.flow import LSTM
from DiffFlowTransport.sas import initialize_sas_model
from DiffFlowTransport.utils import scale_df, make_pytorch_timeseries_dataloader
from DiffFlowTransport.utils.plot import plot_timeseries_obs_1to1, plot_PQET_ST, plot_PQET_quantile
from DiffFlowTransport.utils import train_flow_model, train_transport_model, mse, predict_dl
from DiffFlowTransport.model import load_model, save_model

import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# %% [markdown]
# # Read the data

# %%
f_data = './data.csv'
df = pd.read_csv(f_data, index_col=1)
df.index = pd.to_datetime(df.index, format='%m/%d/%y')

# %%
def get_fluxes_from_df(df, nτ=None):
    J = jnp.array(df['J'].values)
    Q = jnp.array(df['Q'].values)
    ET = jnp.array(df['ET'].values)
    C_J = jnp.array(df[['Cl mg/l']].values)
    C_Q_J = jnp.array(df['Q Cl mg/l'].values)

    # sTmT initial conditions
    if nτ is None:
        sTmT_init = jnp.zeros([J.size, 2])
    else:
        sTmT_init = jnp.zeros([nτ, 2])
    return J, Q, ET, C_J, C_Q_J, sTmT_init


# %% [markdown]
# # Parameters

# %%
# The flow and transport model coupling type
flow_transport_coupling_type = 0

# Watershed
watershed_name = 'Plynlimon'

# Label
# model_label = 'mdn-one_gamma-coupled'
model_label = f'mdn-couplingtype{flow_transport_coupling_type}'


# %% [markdown]
# ## Flow model configuration

# %%
flow_params = {
    "n_input": 1,
    "n_output": 1,
    "n_hidden": 20,
    "key": 512,
}


# %% [markdown]
# ## Transport model configuration

# %%
# Number of SAS inputs depends on the coupling type
if flow_transport_coupling_type == 0:
    n_sas_input = 1

elif flow_transport_coupling_type == 1:
    n_sas_input = flow_params['n_hidden']

elif flow_transport_coupling_type == 2:
    n_sas_input = flow_params['n_hidden'] + 1

else:
    n_sas_input = 1
    

# %%
transport_params = {
    "transport_specs": {
        "dt": 1.,
        "α_Q": 1.,
        "α_ET": 0.,
        "k1": [0.],
        "C_eq": [0.],
        # "C_Q_old": [0.],
        # "C_Q_old": [jnp.mean(C_J * J) / Q.mean() / (1-0.26)]
        # "C_Q_old": [jnp.mean(C_J * J) / Q.mean() * (1-0.26)]
        # "C_Q_old": [jnp.mean(C_J * J) / Q.mean()]
        "C_Q_old": [6.]
    },
    "sas_specs": {
        "Q SAS fun": {
            "func": "MDN",
            "args": {
                "scale": 8215.0,
                "n_input": n_sas_input,  # Same as the number of hidden states used in the flow LSTM model
                "n_hidden": 10,
                "key": 42,
                "width_size": 10,
                "depth": 2
            }
        },
        "ET SAS fun": {
            "func": "Uniform",
            "args": {
                "scale": 737.0
            }
        }
    }
}


# %% [markdown]
# ## Training configuration and dataloader

# %%
flow_dl_config = {
    "targets": ['Q'],
    "features": ['J'],
    "sequence_length": 365,
    "batch_size": 512,
}


# %%
train_config = {
    "learning_rate": 0.01,
    "epochs": 200,
    "train_start": '1989-01-01',
    "train_end": '1998-12-31',
    "test_start": '1999-01-01',
    "test_end": '2008-12-31',
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


# # %%
# ## TODO: Check whether the model is 'incorrectly' initialized !
# x, y = next(iter(train_loader))
# yp = jax.vmap(flow_model)(jnp.array(x))
# plt.plot(yp)

# %% [markdown]
# ## Train the flow model

# %%
flow_model_new, loss_train_flow, loss_test_flow = train_flow_model(
    flow_model, train_config['epochs'], mse, optim, train_loader, test_loader
    # flow_model, 10, mse, optim, train_loader, test_loader
)


# # %%
# fig, ax = plt.subplots(1, 1, figsize=(5, 4))
# ax.plot(loss_train_flow, label='Training losses')
# ax.plot(loss_test_flow, label='Testing losses')
# ax.set(title=f'Losses with training data from {train_start} to {train_end} \n and testing data from {test_start} to {test_end}',
#        ylabel='Loss (MSE)', xlabel='Epochs')
# ax.legend(loc='upper right');


# %% [markdown]
# ## Make predictions and get the hidden states for SAS arguments

# %%
flow_hidden_states = predict_dl(all_loader, flow_model_new.calculate_hidden_states)
Q_lstm = predict_dl(all_loader, flow_model_new)


# %% [markdown]
# # The transport model

# %% [markdown]
# ## Get the transport model training data

# %%
# TODO: skip the initial days with sequence length used by LSTM model
cutoff_length = flow_dl_config['sequence_length']
df_cut = df.iloc[cutoff_length:]
J, Q, ET, C_J, C_Q_J, sTmT_init = get_fluxes_from_df(df_cut)


# %%
# SAS arguments
# sas_Q_args = flow_hidden_states
# sas_ET_args = flow_hidden_states
if flow_transport_coupling_type == 0:
    sas_Q_args = Q[:,None]
    sas_ET_args = ET[:,None]

elif flow_transport_coupling_type == 1:
    sas_Q_args = flow_hidden_states
    sas_ET_args = flow_hidden_states

elif flow_transport_coupling_type == 2:
    sas_Q_args = jnp.concat([flow_hidden_states, Q[:,None]], axis=1)
    sas_ET_args = jnp.concat([flow_hidden_states, ET[:,None]], axis=1)

else:
    sas_Q_args = Q[:,None]
    sas_ET_args = ET[:,None]
    

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


# %%
# Plot the loss
fig, ax = plt.subplots(1, 1, figsize=(5,3))
ax.plot(loss_train_transport, label='train')
ax.plot(loss_test_transport, label='test')
ax.set(xlabel='Epochs', ylabel='Loss', 
       title='$\Omega_Q$: MDN-Gaussian; $\Omega_{ET}$: Invariant Uniform')
ax.legend()


# %% [markdown]
# ## Make predictions

# %%
# Run the trained model
sT, mT, mQETs, pQETs, mRs, C_Q = transport_model_new(
    J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args
)
df_cut['C_Q mdn'] = C_Q


# %%
# plot_timeseries_obs_1to1(
#     obs=df_cut['Q Cl mg/l'].values, sim=df_cut['C_Q mdn'].values, lim=[2,15], timesteps=df_cut.index,
#     varn='Cl ($\Omega_Q$: MDN-Gaussian; $\Omega_{ET}$: Invariant Uniform)'
# )


# # %%
# s, e = test_start, test_end
# plot_timeseries_obs_1to1(
#     obs=df_cut['Q Cl mg/l'][s:e].values, sim=df_cut['C_Q mdn'][s:e].values, lim=[2,15], 
#     timesteps=df_cut[s:e].index,
#     varn='Cl ($\Omega_Q$: MDN-Gaussian; $\Omega_{ET}$: Invariant Uniform)'
# )


# # %%
# plot_PQET_quantile(pQETs, transport_params['transport_specs']['dt'], df_cut.index)


# %% [markdown]
# # Save the models

# %%
f_flow = f'flow_model_{model_label}.eqx'
f_transport = f'transport_model_{model_label}.eqx'
f_configs = f'configs-{model_label}.json'


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
save_model(configs, flow_model_new, transport_model_new, f_configs)


# %% [markdown]
# # Load the models
# model, flow_dl, transport_data = load_model(f_configs)



