# %%
from pathlib import Path

import json
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

# Force JAX to use CPU
jax.config.update("jax_platform_name", "cpu")

from DiffFlowTransport.sas import SAS_Uniform, SAS_Gamma_StorageDependent
from DiffFlowTransport.transport import SASTransport
from DiffFlowTransport.utils import get_transport_obs_fluxes
from DiffFlowTransport.utils.plot import plot_timeseries_obs_1to1, plot_PQET_quantile, plot_2timeseries_obs_1to1
# from DiffFlowTransport.utils.plot import plot_PQET_ST, plot_PQET_ST2, plot_young_water
# from DiffFlowTransport.utils.plot import plot_mdn_weights, plot_mdn_weights_with_ω
# from DiffFlowTransport.utils.plot import plot_young_water_withQ, plot_PQET_ST_esspi, plot_PQET_ST_noselect
# from DiffFlowTransport.utils.plot import plot_flow_transport_assessment2,plot_PQET_ST_esspi_ensemble
from DiffFlowTransport.utils.plot import plot_PQ_ST_ensemble, plot_PQ_ensemble
from DiffFlowTransport.utils.plot import plot_young_water_withQ_ensemble, plot_mdn_weights_ensemble
from DiffFlowTransport.utils import compute_metrics
from DiffFlowTransport.model import load_model

import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# %% [markdown]
# # Load models

# %%
model_names = [
    'mdn2-couplingtype1-logQ',
    'mdn2-couplingtype1-logQ-rd0-mlpd2-mlpw10',
    'mdn2-couplingtype1-logQ-rd1-mlpd2-mlpw10',
    'mdn2-couplingtype1-logQ-rd1234-mlpd2-mlpw10',
    'mdn2-couplingtype1-logQ-rd10-mlpd2-mlpw10',
    'mdn2-couplingtype1-logQ-rd20-mlpd2-mlpw10',
    'mdn2-couplingtype1-logQ-rd42-mlpd3-mlpw20',
    'mdn2-couplingtype1-logQ-rd42-mlpd4-mlpw16',
    'mdn2-couplingtype1-logQ-rd42-mlpd5-mlpw12',
    'mdn2-couplingtype1-logQ-rd42-mlpd6-mlpw8',
    'mdn2-couplingtype1-logQ-rd42-mlpd7-mlpw4',
    'mdn2-couplingtype3-logQ',
    'mdn2-couplingtype3-logQ-rd0-mlpd2-mlpw10',
    'mdn2-couplingtype3-logQ-rd1-mlpd2-mlpw10',
    'mdn2-couplingtype3-logQ-rd1234-mlpd2-mlpw10',
    'mdn2-couplingtype3-logQ-rd10-mlpd2-mlpw10',
    'mdn2-couplingtype3-logQ-rd20-mlpd2-mlpw10',
    'mdn2-couplingtype3-logQ-rd42-mlpd3-mlpw20',
    'mdn2-couplingtype3-logQ-rd42-mlpd4-mlpw16',
    'mdn2-couplingtype3-logQ-rd42-mlpd5-mlpw12',
    'mdn2-couplingtype3-logQ-rd42-mlpd6-mlpw8',
    'mdn2-couplingtype3-logQ-rd42-mlpd7-mlpw4',
    'mdn2-couplingtype4-logQ',
    'mdn2-couplingtype4-logQ-rd0-mlpd2-mlpw10',
    'mdn2-couplingtype4-logQ-rd1-mlpd2-mlpw10',
    'mdn2-couplingtype4-logQ-rd1234-mlpd2-mlpw10',
    'mdn2-couplingtype4-logQ-rd10-mlpd2-mlpw10',
    'mdn2-couplingtype4-logQ-rd20-mlpd2-mlpw10',
    'mdn2-couplingtype4-logQ-rd42-mlpd3-mlpw20',
    'mdn2-couplingtype4-logQ-rd42-mlpd4-mlpw16',
    'mdn2-couplingtype4-logQ-rd42-mlpd5-mlpw12',
    'mdn2-couplingtype4-logQ-rd42-mlpd6-mlpw8',
    'mdn2-couplingtype4-logQ-rd42-mlpd7-mlpw4',
]
# model_labels = [r'MDN$_{Q,\text{LSTM}}$'+f'-{i}' for i in range(len(model_names))]
model_labels = [r'MDN$_{\text{LSTM}}$'+f'-{i}' for i in range(11)] + \
               [r'MDN$_{Q}$'+f'-{i}' for i in range(11)] + \
               [r'MDN$_{Q,\text{LSTM}}$'+f'-{i}' for i in range(11)]

saved_folder = Path("./models-rev2")

f_configs_set = [f'configs-{model}.json' for model in model_names]
f_sim_set = [f'sim-{model}.pkl' for model in model_names]


# %%
model_set, loss_set, flow_dl_set, transport_data_set, configs_set = [], [], [], [], []
for i,f_configs in enumerate(f_configs_set):
    model, loss, flow_dl, transport_data, configs = load_model(f_configs, saved_folder)
    model_label = model_labels[i]
    model_set.append(model)
    loss_set.append(loss)
    flow_dl_set.append(flow_dl)
    transport_data_set.append(transport_data)
    configs_set.append(configs)


# %% [markdown]
# # Plot loss

# %%
# import numpy as np
# # [loss_set[i]['flow_loss']['train'][-1] for i in range(len(model_names))]
# [loss_set[i]['flow_loss']['test'][-1] for i in range(len(model_names))]

# %%
fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)
for i,loss in enumerate(loss_set):
    model_label = model_labels[i]
    axes[0].plot(loss['flow_loss']['train'], 'k', label=model_label)
    axes[1].plot(loss['flow_loss']['test'], 'k', label=model_label)

for i,ax in enumerate(axes):
    ax.set(xlabel='Epoch', ylabel='Loss (MSE)', 
           title='Flow model (training)' if i==0 else 'Flow model (test)')
# ax.legend()
plt.savefig('./figs/flow-losses.png', dpi=150, bbox_inches="tight")


# %%
fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)
for i,loss in enumerate(loss_set):
    model_label = model_labels[i]
    axes[0].plot(loss['transport_loss']['train'], 'k', label=model_label)
    axes[1].plot(loss['transport_loss']['test'], 'k', label=model_label)

for i,ax in enumerate(axes):
    ax.set(xlabel='Epoch', ylabel='Loss (MSE)', 
           title='Transport model (training)' if i==0 else 'Transport model (test)')
# ax.legend()
plt.savefig('./figs/transport-losses.png', dpi=150, bbox_inches="tight")


# %%
# Run or load all the models
transport_output_set = []
transport_output_set_noQ = []
flow_output_set = []
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

    # Append the results
    flow_output_set.append(Q_pred)
    transport_output_set.append(transport_output)
    transport_output_set_noQ.append(transport_output_noQ)


# %%
df_set = []
for i,transport_data in enumerate(transport_data_set):
    J, Q, ET, C_J, C_Q, time = transport_data
    sT, mT, mQETs, pQETs, mRs, C_Q_pred = transport_output_set[i]
    C_Q_pred2 = transport_output_set_noQ[i][-1]
    Q_pred = flow_output_set[i]
    data = list(zip(J, Q, Q_pred.flatten(), ET, C_J.flatten(), 
                    C_Q.flatten(), C_Q_pred.flatten(), C_Q_pred2.flatten()))
    df = pd.DataFrame(data, columns=['J', 'Q', 'Q_simb', 'ET', 'C_J', 'C_Q', 'C_Q_sim', 'C_Q_simb']).astype(float)
    df.index = time
    df_set.append(df)


# %% [markdown]
# # Run storage-dependent Gamma model

# %%
df_data = pd.read_csv('./data.csv', index_col=1)
df_data.index = pd.to_datetime(df_data.index, format='%m/%d/%y')
df_data.rename(columns={'Cl mg/l':'C_J', 'Q Cl mg/l':'C_Q'}, inplace=True)
df_data = df_data.loc[time]
df_data.head()


# %%
from DiffFlowTransport.sas import SAS_Uniform, SAS_Gamma_VaryingScale
from DiffFlowTransport.transport import SASTransport

# Parameters
params = {
    "dt": 1.,
    "α_Q": 1.,
    "α_ET": 0.,
    "k1": [0.],
    "C_eq": [0.],
    "C_Q_old": [6.0]
    # "C_Q_old": [7.11]
}

nt = Q.size
sTmT_init = jnp.zeros([nt, 2])

# SAS functions
sas_Q = SAS_Gamma_VaryingScale(a=0.69, scale=8215.)
sas_ET = SAS_Uniform(scale=398.)

# SAS arguments 
sas_Q_args = jnp.array(df_data['S_scale'].values)
sas_ET_args = ET

# Initialize the SAS transport model
sas_transport = SASTransport(sas_Q, sas_ET, **params)

# Run the model
transport_output = sas_transport(
    J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args
)

# Run the model -- operantional mode
Q_sim = df_set[0]['Q_simb'].values
transport_output_op = sas_transport(J, C_J, Q_sim, ET, sTmT_init, sas_Q_args, sas_ET_args)


# %%
# Organize the model results
model_names.append('one_gamma_dynamic')
model_labels.append(r'$\Gamma_\text{dynamic}$')

flow_output_set.append(None)
transport_output_set.append(transport_output)
transport_output_set_noQ.append(transport_output_op)

df_data['C_Q_sim'] = transport_output[-1]
df_data['C_Q_simb'] = transport_output_op[-1]
df_set.append(df_data)


# %% [markdown]
# # Calculate the performance metrics

# %%
# Calculate the performance metrics
df_metrics = pd.DataFrame(
    columns=[
        "model", "model-type", "with-sm", "gaussian", "mdn", "use_lstm_hid", "varn", "train_or_test",
        "rse", "mare", "rmse", "mse", "r2", "kge",
        "nse", "mkge", "cc", "alpha","beta",
    ]
)

test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
train_s, train_e = configs['train_configs']['train_start'], configs['train_configs']['train_end']

for i,model_name in enumerate(model_names):
    print(model_name)
    model_label = model_labels[i]

    if 'rain' in model_label.lower():
        with_sm = False
    else:
        with_sm = True

    model_type = model_label.split('-')[0]
    
    attr = model_name.split('-')
    if len(attr) == 3:
        mdn = attr[1]
    elif attr[0] == 'one_gamma':
        mdn = 'one_gamma'
    else:
        mdn = 'mixed'

    is_gaussian = 'gaussian' in model_name.lower()
    lstm_hid = model_name.endswith('4')
    df = df_set[i]

    df_train = df[train_s:train_e]
    df_test = df[test_s:test_e]

    # Q
    if 'Q_simb' in df:
        metrics1 = compute_metrics(df_train['Q_simb'].values, df_train['Q'].values, mask_naninf=True)
        metrics2 = compute_metrics(df_test['Q_simb'].values, df_test['Q'].values, mask_naninf=True)
        df_metrics.loc[i*6] = [
            model_label, model_type, with_sm, is_gaussian, mdn, lstm_hid, 'Q', 'train', metrics1['rse'], metrics1['mare'], metrics1['rmse'],
            metrics1['mse'], metrics1['r2'], metrics1['kge'], metrics1['nse'], 
            metrics1['mkge'], metrics1['cc'], metrics1['alpha'], metrics1['beta'], 
        ]
        df_metrics.loc[i*6+1] = [
            model_label, model_type, with_sm, is_gaussian, mdn, lstm_hid, 'Q', 'test', metrics2['rse'], metrics2['mare'], metrics2['rmse'],
            metrics2['mse'], metrics2['r2'], metrics2['kge'], metrics2['nse'], 
            metrics2['mkge'], metrics2['cc'], metrics2['alpha'], metrics2['beta'], 
        ]

    # C_Q with Qobs
    metrics1 = compute_metrics(df_train['C_Q_sim'].values, df_train['C_Q'].values, mask_naninf=True)
    metrics2 = compute_metrics(df_test['C_Q_sim'].values, df_test['C_Q'].values, mask_naninf=True)
    df_metrics.loc[i*6+2] = [
        model_label, model_type, with_sm, is_gaussian, mdn, lstm_hid, 'C_Q', 'train', metrics1['rse'], metrics1['mare'], metrics1['rmse'],
        metrics1['mse'], metrics1['r2'], metrics1['kge'], metrics1['nse'], 
        metrics1['mkge'], metrics1['cc'], metrics1['alpha'], metrics1['beta'], 
    ]
    df_metrics.loc[i*6+3] = [
        model_label, model_type, with_sm, is_gaussian, mdn, lstm_hid, 'C_Q', 'test', metrics2['rse'], metrics2['mare'], metrics2['rmse'],
        metrics2['mse'], metrics2['r2'], metrics2['kge'], metrics2['nse'], 
        metrics2['mkge'], metrics2['cc'], metrics2['alpha'], metrics2['beta'], 
    ]

    # C_Q without Qobs
    if 'C_Q_simb' in df:
        metrics1 = compute_metrics(df_train['C_Q_simb'].values, df_train['C_Q'].values, mask_naninf=True)
        metrics2 = compute_metrics(df_test['C_Q_simb'].values, df_test['C_Q'].values, mask_naninf=True)
        df_metrics.loc[i*6+4] = [
            model_label, model_type, with_sm, is_gaussian, mdn, lstm_hid, 'C_{Qb}', 'train', metrics1['rse'], metrics1['mare'], metrics1['rmse'],
            metrics1['mse'], metrics1['r2'], metrics1['kge'], metrics1['nse'], 
            metrics1['mkge'], metrics1['cc'], metrics1['alpha'], metrics1['beta'], 
        ]
        df_metrics.loc[i*6+5] = [
            model_label, model_type, with_sm, is_gaussian, mdn, lstm_hid, 'C_{Qb}', 'test', metrics2['rse'], metrics2['mare'], metrics2['rmse'],
            metrics2['mse'], metrics2['r2'], metrics2['kge'], metrics2['nse'], 
            metrics2['mkge'], metrics2['cc'], metrics2['alpha'], metrics2['beta'], 
        ]


# %%
train_or_test, metrics = "test", ['nse', 'mse', 'cc']
varns = ["C_Q", "C_{Qb}"]
varn_labels = ["$C_Q$", "$C_{Q}$ (op)"]
fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 12))
for j, metric in enumerate(metrics):
    ax = axes[j]
    df_metrics_sub = df_metrics[
        (df_metrics['train_or_test']==train_or_test) & (df_metrics['varn'].isin(varns)) & (df_metrics["with-sm"])
    ]
    ax = sns.barplot(df_metrics_sub, ax=ax, x="model", y=metric, hue="varn",
                     palette=["tab:blue", "tab:orange"],
                     legend="full" if j==len(metrics)-1 else False)
    ax.set(xlabel="", ylabel = f"${metric.upper()}$",
           yscale="linear")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    if j==len(metrics)-1:
        handles, labels = ax.get_legend_handles_labels()
        labels = [varn_labels[i] for i,l in enumerate(labels)]
        ax.legend(handles, labels, ncols=3, frameon=False, bbox_to_anchor=(.5, -0.15), title="")
plt.subplots_adjust(wspace=0.35, hspace=0.25);
plt.suptitle('Lower Hafren Watershed');
plt.savefig('./figs/performances1.png', dpi=150, bbox_inches="tight")


# %%
# Plot the histogram ...
train_or_test, metrics = "test", ['nse', 'mse', 'cc']
varns = ["C_Q", "C_{Qb}"]
varn_labels = ["$C_Q$", "$C_{Q}$ (op)"]
fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
for j, metric in enumerate(metrics):
    ax = axes[j]
    df_metrics_sub = df_metrics[
        (df_metrics['train_or_test']==train_or_test) & (df_metrics['varn'].isin(varns)) & (df_metrics["with-sm"])
    ]
    ax = sns.boxplot(df_metrics_sub, ax=ax, x="model-type", y=metric, hue="varn",
                     palette=["tab:blue", "tab:orange"],
                     legend=False)
                    #  legend="full" if j==len(metrics)-1 else False)
    ax.set(xlabel="", ylabel = f"${metric.upper()}$",
        #    yscale="linear")
        #    yscale="linear", ylim=[0.5, 1.0] if metric in ["nse", "cc"] else None)
           yscale="linear", ylim=[-1, 1] if metric=="nse" else [0,1] if metric=="cc" else None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    # if j==len(metrics)-1:
    #     handles, labels = ax.get_legend_handles_labels()
    #     labels = [varn_labels[i] for i,l in enumerate(labels)]
    #     ax.legend(handles, labels, ncols=3, frameon=False, bbox_to_anchor=(0.0, -0.15), title="")
plt.subplots_adjust(wspace=0.35, hspace=0.25);
plt.suptitle('Lower Hafren Watershed');
plt.savefig('./figs/performances2.png', dpi=150, bbox_inches="tight")


# %%
# Plot the histogram ...
train_or_test, metrics = "test", ['nse', 'mse', 'cc']
varns = ["C_Q", "C_{Qb}"]
varn_labels = ["$C_Q$", "$C_{Q}$ (op)"]
fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
for j, metric in enumerate(metrics):
    ax = axes[j]
    df_metrics_sub = df_metrics[
        (df_metrics['train_or_test']==train_or_test) & (df_metrics['varn'].isin(varns)) & (df_metrics["with-sm"])
    ]
    # ax = sns.barplot(df_metrics_sub, ax=ax, x="model-type", y=metric, hue="varn",
    #                  palette=["tab:blue", "tab:orange"], errorbar='sd',
                    #  legend="full" if j==len(metrics)-1 else False)
    ax = sns.boxplot(df_metrics_sub, ax=ax, x="model-type", y=metric, hue="varn",
                     palette=["tab:blue", "tab:orange"], legend=False)
    ax.set(xlabel="", ylabel = f"${metric.upper()}$",
        #    yscale="linear")
        #    yscale="linear", ylim=[0.5, 1.0] if metric in ["nse", "cc"] else None)
           yscale="linear", ylim=[-1, 1] if metric=="nse" else [0,1] if metric=="cc" else None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    # if j==len(metrics)-1:
    #     handles, labels = ax.get_legend_handles_labels()
    #     labels = [varn_labels[i] for i,l in enumerate(labels)]
    #     ax.legend(handles, labels, ncols=2, frameon=False, bbox_to_anchor=(-0.5, -0.15), title="")
plt.subplots_adjust(wspace=0.35, hspace=0.25);
plt.suptitle('Oak Creek Watershed');
plt.savefig('./figs/performances2.png', dpi=150, bbox_inches="tight")


# %%
# Plot the histogram ...
train_or_test, metrics = "test", ['kge', 'alpha', 'beta', 'cc']
varns = ["C_Q", "C_{Qb}"]
varn_labels = ["$C_Q$", "$C_{Q}$ (op)"]
fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
for j, metric in enumerate(metrics):
    ax = axes[j]
    df_metrics_sub = df_metrics[
        (df_metrics['train_or_test']==train_or_test) & (df_metrics['varn'].isin(varns)) & (df_metrics["with-sm"])
    ]
    ax = sns.boxplot(df_metrics_sub, ax=ax, x="model-type", y=metric, hue="varn",
                     palette=["tab:blue", "tab:orange"], legend=False)
    ax.set(xlabel="", ylabel = f"${metric.upper()}$",
           yscale="linear", ylim=[-1, 1] if metric=="kge" else [0,1] if metric=="cc" else None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    # if j==len(metrics)-1:
    #     handles, labels = ax.get_legend_handles_labels()
    #     labels = [varn_labels[i] for i,l in enumerate(labels)]
    #     ax.legend(handles, labels, ncols=2, frameon=False, bbox_to_anchor=(-0.5, -0.15), title="")
plt.subplots_adjust(wspace=0.35, hspace=0.25);
plt.suptitle('Oak Creek Watershed');
plt.savefig('./figs/performances2-kge.png', dpi=150, bbox_inches="tight")


# %%
# Plot the histogram ...
train_or_test, metrics = "test", ['nse', 'mse', 'cc']
varn, varn_label = "C_Q", "$C_Q$"
fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
for j, metric in enumerate(metrics):
    ax = axes[j]
    df_metrics_sub = df_metrics[
        (df_metrics['train_or_test']==train_or_test) & (df_metrics['varn'].isin(varns)) & 
        (df_metrics["with-sm"]) & (df_metrics['varn']==varn)
    ]
    # ax = sns.barplot(df_metrics_sub, ax=ax, x="model-type", y=metric, errorbar='sd',
    ax = sns.boxplot(df_metrics_sub, ax=ax, x="model-type", y=metric,
                     legend="full" if j==len(metrics)-1 else False)
    ax.set(xlabel="", ylabel = f"${metric.upper()}$ - {varn_label}",
        #    yscale="linear")
        #    yscale="linear", ylim=[0.5, 1.0] if metric in ["nse", "cc"] else None)
           yscale="linear", ylim=[0, 1] if metric=="nse" else [0,1] if metric=="cc" else None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
plt.subplots_adjust(wspace=0.35, hspace=0.25);
plt.suptitle('Lower Hafren Watershed');
plt.savefig('./figs/performances3.png', dpi=150, bbox_inches="tight")


# %%
# Plot the histogram ...
train_or_test, metrics = "test", ['nse', 'mse', 'cc']
varn, varn_label = "C_{Qb}", "$C_Q$ (op)"
fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
for j, metric in enumerate(metrics):
    ax = axes[j]
    df_metrics_sub = df_metrics[
        (df_metrics['train_or_test']==train_or_test) & (df_metrics['varn'].isin(varns)) & 
        (df_metrics["with-sm"]) & (df_metrics['varn']==varn)
    ]
    # ax = sns.barplot(df_metrics_sub, ax=ax, x="model-type", y=metric, errorbar='sd',
    ax = sns.boxplot(df_metrics_sub, ax=ax, x="model-type", y=metric,
                     legend="full" if j==len(metrics)-1 else False)
    ax.set(xlabel="", ylabel = f"${metric.upper()}$ - {varn_label}",
        #    yscale="linear")
        #    yscale="linear", ylim=[0.5, 1.0] if metric in ["nse", "cc"] else None)
           yscale="linear", ylim=[None, 1] if metric=="nse" else [0,1] if metric=="cc" else None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
plt.subplots_adjust(wspace=0.35, hspace=0.25);
plt.suptitle('Lower Hafren Watershed');
plt.savefig('./figs/performances4.png', dpi=150, bbox_inches="tight")


# %% [markdown]
# # Time series prediction

# %%
# Time series of the best CQ
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(
    4, 2, width_ratios=(3, 1), left=0.1, right=0.9,
    bottom=0.1, top=0.9, wspace=0.05, hspace=0.1,
)
model_type_set = ['MDN$_{\\text{LSTM}}$', 'MDN$_{Q}$', 'MDN$_{Q,\\text{LSTM}}$', '$\\Gamma_\\text{dynamic}$']
for i,model_type in enumerate(model_type_set):
    df_metrics_sub = df_metrics[(df_metrics['model-type']==model_type) & (df_metrics['varn']=='C_Q')]
    best_model = df_metrics_sub.iloc[df_metrics_sub['mse'].argmin()]['model']
    ind = model_labels.index(best_model)
    
    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[ind]
    else:
        configs, df = configs_set[ind], df_set[ind]
    sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[ind]
    test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
    dt = configs['transport_configs']['transport_specs']['dt']

    # Plot
    ax1 = fig.add_subplot(gs[i,0])
    ax2 = fig.add_subplot(gs[i,1], sharey=ax1)
    ax1, ax2 = plot_timeseries_obs_1to1(
        obs=df[test_s:test_e]['C_Q'].values, sim=df[test_s:test_e]['C_Q_sim'].values, lim=[4,15], 
        timesteps=df[test_s:test_e].index, units='Cl \n [mg l$^{-1}$]', varn=model_type, 
        label_sim='DiffSAS', axes=[ax1, ax2], legend=False, figsize=(12, 3)
        # label_sim='DiffSAS', figsize=(10,2)
    );
    if i != len(model_type_set)-1:
        ax1.set(xticks=[])
        ax2.set(xticks=[], xlabel=None)
plt.savefig(f'./figs/timeseries-best.png', dpi=150)


# %%
# Time series of the best CQ (op)
fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(
    5, 2, width_ratios=(3, 1), left=0.1, right=0.9,
    bottom=0.1, top=0.9, wspace=0.05, hspace=0.1,
)
model_type_set = ['MDN$_{\\text{LSTM}}$', 'MDN$_{Q}$', 'MDN$_{Q,\\text{LSTM}}$', '$\\Gamma_\\text{dynamic}$']
for i,model_type in enumerate(model_type_set):
    df_metrics_sub = df_metrics[(df_metrics['model-type']==model_type) & (df_metrics['varn']=='C_{Qb}')]
    best_model = df_metrics_sub.iloc[df_metrics_sub['mse'].argmin()]['model']
    ind = model_labels.index(best_model)
    
    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[ind]
    else:
        configs, df = configs_set[ind], df_set[ind]
    sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[ind]
    test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
    dt = configs['transport_configs']['transport_specs']['dt']

    # Plot
    # Flow
    if i == 0:
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1], sharey=ax1)
        plot_timeseries_obs_1to1(
            obs=df[test_s:test_e]['Q'].values, sim=df[test_s:test_e]['Q_simb'].values, lim=[0,110], timesteps=df[test_s:test_e].index,
            units='$Q$ [mm d$^{-1}$]', varn='Streamflow', label_sim='LSTM', axes=[ax1, ax2], legend=False, figsize=(12, 3)
        );
        ax1.set(xticks=[])
        ax2.set(xticks=[], xlabel=None)
    # Transport
    ax1 = fig.add_subplot(gs[i+1,0])
    ax2 = fig.add_subplot(gs[i+1,1], sharey=ax1)
    ax1, ax2 = plot_timeseries_obs_1to1(
        obs=df[test_s:test_e]['C_Q'].values, sim=df[test_s:test_e]['C_Q_simb'].values, lim=[4,15], 
        timesteps=df[test_s:test_e].index, units='Cl \n [mg l$^{-1}$]', varn=model_type + " (operational mode)", 
        label_sim='DiffSAS', axes=[ax1, ax2], legend=False, figsize=(12, 3)
        # label_sim='DiffSAS', figsize=(10,2)
    );
    if i != len(model_type_set)-1:
        ax1.set(xticks=[])
        ax2.set(xticks=[], xlabel=None)
plt.savefig(f'./figs/timeseries-best-op.png', dpi=150)


# %% [markdown]
# # TTDs assessment
del transport_data_set
import gc
gc.collect()

# %%
sT_set, pQ_set = [], []
model_type = 'MDN$_{\\text{LSTM}}$'
for i,model_label in enumerate(model_labels):
    if not model_label.startswith(model_type): continue
    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
    else:
        configs, df = configs_set[i], df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
        test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
        dt = configs['transport_configs']['transport_specs']['dt']

    test_s_ind, test_e_ind = df.index.get_loc(test_s), df.index.get_loc(test_e)
    df_plot = df[test_s:test_e]
    sT_plot, pQ_plot = sT[:,test_s_ind:,...], pQETs[:,test_s_ind:,0]

    sT_set.append(sT_plot)
    pQ_set.append(pQ_plot)

plot_PQ_ensemble(
    Q=df_plot.Q.values, pQ_set=pQ_set, sT_set=sT_set, dt=dt, timesteps=df_plot.index,
    last_age_cut=None, figsize=(7, 7), label=model_type, plot_all_PQs=True, plot_sel_time_style='sd'
);
plt.savefig(f'./figs/ttds-{model_type}.png', dpi=150, bbox_inches="tight")


# %%
sT_set, pQ_set = [], []
model_type = 'MDN$_{Q}$'
for i,model_label in enumerate(model_labels):
    if not model_label.startswith(model_type): continue
    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
    else:
        configs, df = configs_set[i], df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
        test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
        dt = configs['transport_configs']['transport_specs']['dt']

    test_s_ind, test_e_ind = df.index.get_loc(test_s), df.index.get_loc(test_e)
    df_plot = df[test_s:test_e]
    sT_plot, pQ_plot = sT[:,test_s_ind:,...], pQETs[:,test_s_ind:,0]

    sT_set.append(sT_plot)
    pQ_set.append(pQ_plot)

plot_PQ_ensemble(
    Q=df_plot.Q.values, pQ_set=pQ_set, sT_set=sT_set, dt=dt, timesteps=df_plot.index,
    last_age_cut=None, figsize=(7, 7), label=model_type, plot_all_PQs=True, plot_sel_time_style='sd'
);
plt.savefig(f'./figs/ttds-{model_type}.png', dpi=150, bbox_inches="tight")


# %%
sT_set, pQ_set = [], []
model_type = 'MDN$_{Q,\\text{LSTM}}$'
for i,model_label in enumerate(model_labels):
    if not model_label.startswith(model_type): continue
    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
    else:
        configs, df = configs_set[i], df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
        test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
        dt = configs['transport_configs']['transport_specs']['dt']

    test_s_ind, test_e_ind = df.index.get_loc(test_s), df.index.get_loc(test_e)
    df_plot = df[test_s:test_e]
    sT_plot, pQ_plot = sT[:,test_s_ind:,...], pQETs[:,test_s_ind:,0]

    sT_set.append(sT_plot)
    pQ_set.append(pQ_plot)

plot_PQ_ensemble(
    Q=df_plot.Q.values, pQ_set=pQ_set, sT_set=sT_set, dt=dt, timesteps=df_plot.index,
    last_age_cut=None, figsize=(7, 7), label=model_type, plot_all_PQs=True, plot_sel_time_style='sd'
);
plt.savefig(f'./figs/ttds-{model_type}.png', dpi=150, bbox_inches="tight")


# %% [markdown]
# # TTDs assessment - MDN

# %%
distributions = [r'$w_\mathcal{N}$', r'$w_\mathcal{U}$', r'$w_\Gamma$']

# %%
mdn_w_set = []
model_type = 'MDN$_{\\text{LSTM}}$'
# MDN weights
for i,model_label in enumerate(model_labels):
    if not model_label.startswith(model_type): continue

    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[i]
        flow_dl, transport_data = flow_dl_set[i], transport_data_set[i]
        J, Q, ET, C_J, C_Q_J, time = transport_data
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
    else:
        configs, df = configs_set[i], df_set[i]
        flow_dl, transport_data = flow_dl_set[i], transport_data_set[i]
        J, Q, ET, C_J, C_Q_J, time = transport_data
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
        dt = configs['transport_configs']['transport_specs']['dt']
        test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']

    model = model_set[i]
    mdn_w = model.get_mdn_weights(Q=Q, dl=flow_dl)

    test_s_ind, test_e_ind = df.index.get_loc(test_s), df.index.get_loc(test_e)
    df_plot = df[test_s:test_e]
    mdn_w = mdn_w[test_s_ind:test_e_ind+1]
    mdn_w_set.append(mdn_w)

plot_mdn_weights_ensemble(
    Q=df_plot.Q.values, mdn_w_set=mdn_w_set, timesteps=df_plot.index,
    mdn_dists=distributions, figsize=(7,6), label=model_type,
);
plt.savefig(f'./figs/ttds-mdnweights-{model_type}.png', dpi=150, bbox_inches="tight")

# %%
mdn_w_set = []
model_type = 'MDN$_{Q}$'
# MDN weights
for i,model_label in enumerate(model_labels):
    if not model_label.startswith(model_type): continue

    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[i]
        flow_dl, transport_data = flow_dl_set[i], transport_data_set[i]
        J, Q, ET, C_J, C_Q_J, time = transport_data
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
    else:
        configs, df = configs_set[i], df_set[i]
        flow_dl, transport_data = flow_dl_set[i], transport_data_set[i]
        J, Q, ET, C_J, C_Q_J, time = transport_data
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
        dt = configs['transport_configs']['transport_specs']['dt']
        test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
    
    model = model_set[i]
    mdn_w = model.get_mdn_weights(Q=Q, dl=flow_dl)

    test_s_ind, test_e_ind = df.index.get_loc(test_s), df.index.get_loc(test_e)
    df_plot = df[test_s:test_e]
    mdn_w = mdn_w[test_s_ind:test_e_ind+1]
    mdn_w_set.append(mdn_w)

plot_mdn_weights_ensemble(
    Q=df_plot.Q.values, mdn_w_set=mdn_w_set, timesteps=df_plot.index,
    mdn_dists=distributions, figsize=(7,6), label=model_type,
);
plt.savefig(f'./figs/ttds-mdnweights-{model_type}.png', dpi=150, bbox_inches="tight")

# %%
mdn_w_set = []
model_type = 'MDN$_{Q,\\text{LSTM}}$'
# MDN weights
for i,model_label in enumerate(model_labels):
    if not model_label.startswith(model_type): continue

    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[i]
        flow_dl, transport_data = flow_dl_set[i], transport_data_set[i]
        J, Q, ET, C_J, C_Q_J, time = transport_data
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
    else:
        configs, df = configs_set[i], df_set[i]
        flow_dl, transport_data = flow_dl_set[i], transport_data_set[i]
        J, Q, ET, C_J, C_Q_J, time = transport_data
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
        dt = configs['transport_configs']['transport_specs']['dt']
        test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
    
    model = model_set[i]
    mdn_w = model.get_mdn_weights(Q=Q, dl=flow_dl)

    test_s_ind, test_e_ind = df.index.get_loc(test_s), df.index.get_loc(test_e)
    df_plot = df[test_s:test_e]
    mdn_w = mdn_w[test_s_ind:test_e_ind+1]
    mdn_w_set.append(mdn_w)

plot_mdn_weights_ensemble(
    Q=df_plot.Q.values, mdn_w_set=mdn_w_set, timesteps=df_plot.index,
    mdn_dists=distributions, figsize=(7,6), label=model_type,
);
plt.savefig(f'./figs/ttds-mdnweights-{model_type}.png', dpi=150, bbox_inches="tight")

# %% [markdown]
# # $P_Q$ - $S_T$ and $\bar{Q_T}$ - $\bar{S_T}$

# %%
sT_set, pQ_set = [], []
model_type = 'MDN$_{\\text{LSTM}}$'
for i,model_label in enumerate(model_labels):
    if not model_label.startswith(model_type): continue
    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
    else:
        configs, df = configs_set[i], df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
        test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
        dt = configs['transport_configs']['transport_specs']['dt']

    test_s_ind, test_e_ind = df.index.get_loc(test_s), df.index.get_loc(test_e)
    df_plot = df[test_s:test_e]
    sT_plot, pQ_plot = sT[:,test_s_ind:,...], pQETs[:,test_s_ind:,0]

    sT_set.append(sT_plot)
    pQ_set.append(pQ_plot)

plot_PQ_ST_ensemble(Q=df_plot.Q.values, pQ_set=pQ_set, sT_set=sT_set, timesteps=df_plot.index, dt=dt, suptitle=model_type)
plt.savefig(f'./figs/PQ-ST-{model_type}.png', dpi=150, bbox_inches="tight")


# %%
sT_set, pQ_set = [], []
model_type = 'MDN$_{Q}$'
for i,model_label in enumerate(model_labels):
    if not model_label.startswith(model_type): continue
    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
    else:
        configs, df = configs_set[i], df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
        test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
        dt = configs['transport_configs']['transport_specs']['dt']

    test_s_ind, test_e_ind = df.index.get_loc(test_s), df.index.get_loc(test_e)
    df_plot = df[test_s:test_e]
    sT_plot, pQ_plot = sT[:,test_s_ind:,...], pQETs[:,test_s_ind:,0]

    sT_set.append(sT_plot)
    pQ_set.append(pQ_plot)

plot_PQ_ST_ensemble(Q=df_plot.Q.values, pQ_set=pQ_set, sT_set=sT_set, timesteps=df_plot.index, dt=dt, suptitle=model_type)
plt.savefig(f'./figs/PQ-ST-{model_type}.png', dpi=150, bbox_inches="tight")


# %%
sT_set, pQ_set = [], []
model_type = 'MDN$_{Q,\\text{LSTM}}$'
for i,model_label in enumerate(model_labels):
    if not model_label.startswith(model_type): continue
    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
    else:
        configs, df = configs_set[i], df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
        test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
        dt = configs['transport_configs']['transport_specs']['dt']

    test_s_ind, test_e_ind = df.index.get_loc(test_s), df.index.get_loc(test_e)
    df_plot = df[test_s:test_e]
    sT_plot, pQ_plot = sT[:,test_s_ind:,...], pQETs[:,test_s_ind:,0]

    sT_set.append(sT_plot)
    pQ_set.append(pQ_plot)

plot_PQ_ST_ensemble(Q=df_plot.Q.values, pQ_set=pQ_set, sT_set=sT_set, timesteps=df_plot.index, dt=dt, suptitle=model_type)
plt.savefig(f'./figs/PQ-ST-{model_type}.png', dpi=150, bbox_inches="tight")


# %% [markdown]
# # Young water fraction

# %%
sT_set, pQ_set = [], []
model_type = 'MDN$_{\\text{LSTM}}$'
for i,model_label in enumerate(model_labels):
    if not model_label.startswith(model_type): continue
    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
    else:
        configs, df = configs_set[i], df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
        test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
        dt = configs['transport_configs']['transport_specs']['dt']

    test_s_ind, test_e_ind = df.index.get_loc(test_s), df.index.get_loc(test_e)
    df_plot = df[test_s:test_e]
    sT_plot, pQ_plot = sT[:,test_s_ind:,...], pQETs[:,test_s_ind:,0]
    sT_set.append(sT_plot)
    pQ_set.append(pQ_plot)

plot_young_water_withQ_ensemble(
    Q=df_plot.Q.values, pQ_set=pQ_set, timesteps=df_plot.index,
    dt=dt, cutoff_ages=[100], label=model_type, figsize=(7, 6), plot_sel_time_style='sd'
);
plt.savefig(f'./figs/youngwater-{model_type}.png', dpi=150, bbox_inches="tight")


# %%
sT_set, pQ_set = [], []
model_type = 'MDN$_{Q}$'
for i,model_label in enumerate(model_labels):
    if not model_label.startswith(model_type): continue
    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
    else:
        configs, df = configs_set[i], df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
        test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
        dt = configs['transport_configs']['transport_specs']['dt']

    test_s_ind, test_e_ind = df.index.get_loc(test_s), df.index.get_loc(test_e)
    df_plot = df[test_s:test_e]
    sT_plot, pQ_plot = sT[:,test_s_ind:,...], pQETs[:,test_s_ind:,0]
    sT_set.append(sT_plot)
    pQ_set.append(pQ_plot)

plot_young_water_withQ_ensemble(
    Q=df_plot.Q.values, pQ_set=pQ_set, timesteps=df_plot.index,
    dt=dt, cutoff_ages=[100], label=model_type, figsize=(7, 6), plot_sel_time_style='sd'
);
plt.savefig(f'./figs/youngwater-{model_type}.png', dpi=150, bbox_inches="tight")


# %%
sT_set, pQ_set = [], []
model_type = 'MDN$_{Q,\\text{LSTM}}$'
for i,model_label in enumerate(model_labels):
    if not model_label.startswith(model_type): continue
    if model_label == '$\\Gamma_\\text{dynamic}$':
        df = df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
    else:
        configs, df = configs_set[i], df_set[i]
        sT, mT, mQETs, pQETs, mRs, C_Q = transport_output_set[i]
        test_s, test_e = configs['train_configs']['test_start'], configs['train_configs']['test_end']
        dt = configs['transport_configs']['transport_specs']['dt']

    test_s_ind, test_e_ind = df.index.get_loc(test_s), df.index.get_loc(test_e)
    df_plot = df[test_s:test_e]
    sT_plot, pQ_plot = sT[:,test_s_ind:,...], pQETs[:,test_s_ind:,0]
    sT_set.append(sT_plot)
    pQ_set.append(pQ_plot)

plot_young_water_withQ_ensemble(
    Q=df_plot.Q.values, pQ_set=pQ_set, timesteps=df_plot.index,
    dt=dt, cutoff_ages=[100], label=model_type, figsize=(7, 6), plot_sel_time_style='sd'
);
plt.savefig(f'./figs/youngwater-{model_type}.png', dpi=150, bbox_inches="tight")