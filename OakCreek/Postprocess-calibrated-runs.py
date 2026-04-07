# %%
from pathlib import Path

import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

# Force JAX to use CPU
jax.config.update("jax_platform_name", "cpu")

from DiffFlowTransport.utils.plot import plot_timeseries_obs_1to1, plot_PQET_quantile, plot_2timeseries_obs_1to1
from DiffFlowTransport.utils.plot import plot_PQET_ST, plot_PQET_ST2, plot_young_water
from DiffFlowTransport.utils.plot import plot_mdn_weights, plot_mdn_weights_with_ω
from DiffFlowTransport.utils.plot import plot_young_water_withQ, plot_PQET_ST_esspi, plot_PQET_ST_noselect
from DiffFlowTransport.utils.plot import plot_flow_transport_assessment2
from DiffFlowTransport.utils import compute_metrics
from DiffFlowTransport.model import load_model

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
    

# %% [markdown]
# # Load models

# %%
model_names = [
    'mdn2-couplingtype1-logQ',
    'mdn2-couplingtype3-logQ',
    'mdn2-couplingtype4-logQ',
    'mdn2-couplingtype4-logQ-rd0-mlpd2-mlpw10',
    'mdn2-couplingtype4-logQ-rd1-mlpd2-mlpw10',
    'mdn2-couplingtype4-logQ-rd1234-mlpd2-mlpw10',
    'mdn2-couplingtype4-logQ-rd10-mlpd2-mlpw10',
    'mdn2-couplingtype4-logQ-rd42-mlpd3-mlpw20',
    'mdn2-couplingtype4-logQ-rd42-mlpd4-mlpw16',
    'mdn2-couplingtype4-logQ-rd42-mlpd5-mlpw12',
    'mdn2-couplingtype4-logQ-rd42-mlpd6-mlpw8',
    'mdn2-couplingtype4-logQ-rd42-mlpd7-mlpw4',
    'mdn2-couplingtype1-logQ-rd0-mlpd2-mlpw10',
    'mdn2-couplingtype1-logQ-rd1-mlpd2-mlpw10',
    'mdn2-couplingtype1-logQ-rd1234-mlpd2-mlpw10',
    'mdn2-couplingtype1-logQ-rd10-mlpd2-mlpw10',
    'mdn2-couplingtype1-logQ-rd42-mlpd3-mlpw20',
    'mdn2-couplingtype1-logQ-rd42-mlpd4-mlpw16',
    'mdn2-couplingtype1-logQ-rd42-mlpd5-mlpw12',
    'mdn2-couplingtype1-logQ-rd42-mlpd6-mlpw8',
    'mdn2-couplingtype1-logQ-rd42-mlpd7-mlpw4',
]
# model_labels = [
#     r'$\Gamma_\text{static}$', 
#     r'MDN', 
#     r'MDN$_\text{LSTM}$',
#     r'$\Gamma_\text{static}$-rain', 
#     r'MDN-rain', 
#     r'MDN$_\text{LSTM}$-rain'
# ]

saved_folder = Path("./models-withDiffusion-rev2")

f_configs_set = [f'configs-{model}.json' for model in model_names]
f_sim_set = [f'sim-{model}.pkl' for model in model_names]


# %%
model_set, loss_set, flow_dl_set, transport_data_set, configs_set = [], [], [], [], []
for i,f_configs in enumerate(f_configs_set):
    model, loss, flow_dl, transport_data, configs = load_model(f_configs, saved_folder)
    # model_label = model_labels[i]
    model_set.append(model)
    loss_set.append(loss)
    flow_dl_set.append(flow_dl)
    transport_data_set.append(transport_data)
    configs_set.append(configs)



# %%
# Run all the models
for i,model in enumerate(model_set):
    f_sim = f_sim_set[i]
    f_sim = saved_folder / f_sim
    try:
        print(f'Loading the simulation of the model {model_names[i]} ...')
        # Try to load the model simulation
        with open(f_sim, "rb") as file:
            sim = pickle.load(file)
            transport_output = sim['with Q']['transport']
            transport_output_noQ = sim['without Q']['transport']
            Q_pred = sim['without Q']['Q']
        
    except:
        print(f'Running the model {model_names[i]} ...')
        # If the simulation does not exist, try to run the model
        flow_dl, transport_data = flow_dl_set[i], transport_data_set[i]
        J, Q, ET, C_J, C_Q_J, time = transport_data
        transport_output, _ = model.run_transport(J, C_J, ET, Q, dl=flow_dl)
        transport_output_noQ, Q_pred = model.run_transport(J, C_J, ET, Q=None, dl=flow_dl)

        # Now, we save the simulation result
        sim = {
            "with Q": {"transport": transport_output},
            "without Q": {"transport": transport_output_noQ, "Q": Q_pred}
        }
    
        # Save to pickle
        with open(f_sim, 'wb') as file:
            # Use pickle.dump() to serialize and save the object
            pickle.dump(sim, file)