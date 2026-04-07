# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import jax.numpy as jnp
from DiffFlowTransport.transport import SASTransport
from DiffFlowTransport.sas import SAS_Gamma, SAS_Uniform, SAS_Gamma_StorageDependent
from DiffFlowTransport.utils import get_transport_obs_fluxes


max_iter = 500

# %% [markdown]
# # Read the data

# %%
df = pd.read_csv('data_withDiffusion_withSpinup-correctedZeroFlow.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df.head()

# %% [markdown]
# # Calculate the storage change

# %%
df['dS'] = df['P'] - df['ET'] - df['Q']
fig, axes = plt.subplots(df.shape[1], sharex=True, figsize=(10,12))
for i, ax in enumerate(axes):
    ax.plot(df.iloc[:,i], '.')
    ax.set(title=df.columns[i])


# %% [markdown]
# # Initialize the model and test run

# %% [markdown]
# ## Parameters

# %%
# Parameters
params = {
    "dt": 1.,
    "α_Q": 1.,
    "α_ET": 0.,
    "k1": [0.],
    "C_eq": [0.],
    "C_Q_old": [0.],
}

# Skip the initial days with sequence length used by LSTM model as lookback days
cutoff_length = 365
df_cut = df.iloc[cutoff_length:]

# Other arguments for running the model
nt = df_cut.shape[0]
sTmT_init = jnp.zeros([nt, 2])


# %%
J, Q, ET, C_J, C_Q_J, dS, time = get_transport_obs_fluxes(df_cut, 'Oakcreek', return_dS=True)


# %% [markdown]
# ## SAS functions

# %%
# SAS functions
# sas_Q = SAS_Gamma_StorageDependent(a=0.69, λ=-103., ΔScλ=-103.*48)
sas_Q = SAS_Gamma_StorageDependent(a=0.69, λ=-103., ΔScλ=-103.*48)
sas_ET = SAS_Uniform(scale=400.)

# SAS arguments
sas_Q_args = dS
sas_ET_args = ET

# Initialize the SAS transport model
sas_transport = SASTransport(sas_Q, sas_ET, **params)


# %% [markdown]
# ## Test run

# %%
sT, mT, mQETs, pQETs, mRs, C_Q = sas_transport(
    J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args
)


# %%
plt.plot(C_Q_J, '.')
plt.plot(C_Q, '.')

# %% [markdown]
# # Optimize the model

# %%
train_start, train_end = '2015-10-01', '2020-09-30'
test_start, test_end = '2021-10-01', '2023-12-28'
train_start_ind = df_cut.index.get_loc(train_start)
train_end_ind = df_cut.index.get_loc(train_end)
test_start_ind = df_cut.index.get_loc(test_start)
test_end_ind = df_cut.index.get_loc(test_end)


# %%
mask = jnp.where(~jnp.isnan(C_Q_J))
mask_train = mask[0][(mask[0]>train_start_ind) & (mask[0]<train_end_ind)]
mask_test = mask[0][(mask[0]>test_start_ind) & (mask[0]<test_end_ind)]

# %%
def nse(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    # mask = np.isfinite(obs) & np.isfinite(sim)
    # obs = obs[mask]
    # sim = sim[mask]

    denom = np.sum((obs - np.mean(obs))**2)
    if denom == 0:
        return np.nan  # undefined if obs has zero variance

    return 1 - np.sum((sim - obs)**2) / denom


# Define a function to run the model
def run_model(x):
    # Get the parameters
    a = x[0]
    λ = x[1]
    ΔScλ = x[2]
    scale = x[3]

    # Initialize the SAS functions
    sas_Q = SAS_Gamma_StorageDependent(a=a, λ=λ, ΔScλ=ΔScλ)
    sas_ET = SAS_Uniform(scale=scale)

    # SAS arguments
    sas_Q_args = dS
    sas_ET_args = ET

    # Initialize the SAS transport model
    sas_transport = SASTransport(sas_Q, sas_ET, **params)

    # Run the SAS model
    sT, mT, mQETs, pQETs, mRs, C_Q = sas_transport(
        J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args
    )

    return C_Q

# Define an objective function
def obj_nse(x):
    # Run the model
    C_Q = run_model(x)
    C_Q = C_Q.flatten()

    # Calculate the loss
    # loss = jnp.mean((C_Q_J[mask_train] - C_Q[mask_train]) ** 2)
    loss = -nse(C_Q_J[mask_train], C_Q[mask_train])

    return loss

def obj_mse(x):
    # Run the model
    C_Q = run_model(x)
    C_Q = C_Q.flatten()

    # Calculate the loss
    loss = jnp.mean((C_Q_J[mask_train] - C_Q[mask_train]) ** 2)

    return loss

def cb(xk):
    # Run the model
    C_Q = run_model(xk)
    C_Q = C_Q.flatten()

    # Calculate the loss
    mse_train = jnp.mean((C_Q_J[mask_train] - C_Q[mask_train]) ** 2)
    nse_train = nse(C_Q_J[mask_train], C_Q[mask_train])
    mse_test = jnp.mean((C_Q_J[mask_test] - C_Q[mask_test]) ** 2)
    nse_test = nse(C_Q_J[mask_test], C_Q[mask_test])

    print("iter x =", xk, "mse (train) =", mse_train, "mse (test) =", mse_test, "nse (train) =", nse_train, "nse (test) =", nse_test)  # per-iteration log


# %%
# Train the model using NSE
x0 = np.array([0.69, -103., -103.*48, 400.])
res_nse = minimize(obj_nse, x0, method='nelder-mead', callback=cb,
               options={'xatol': 1e-8, 'disp': True, "maxiter": max_iter}
               )
            #    options={'xatol': 1e-8, 'disp': True, 'maxiter': 100})

# %%
# Train the model using MSE
res_mse = minimize(obj_mse, x0, method='nelder-mead', callback=cb,
               options={'xatol': 1e-8, 'disp': True, "maxiter": max_iter}
               )

# %%
# Save the results
out = {
    "x": res_nse.x.tolist(),
    "fun": float(res_nse.fun),
    "success": bool(res_nse.success),
    "message": str(res_nse.message),
    "nit": int(res_nse.nit),
    "nfev": int(res_nse.nfev),
}
with open("./models-withDiffusion-rev/storage-dependent-gamma-nse.json", "w") as f:
    json.dump(out, f, indent=2)


# Save the results
out = {
    "x": res_mse.x.tolist(),
    "fun": float(res_mse.fun),
    "success": bool(res_mse.success),
    "message": str(res_mse.message),
    "nit": int(res_mse.nit),
    "nfev": int(res_mse.nfev),
}
with open("./models-withDiffusion-rev/storage-dependent-gamma-mse.json", "w") as f:
    json.dump(out, f, indent=2)

# %%
# Run the model using the optimized parameters
# cb(res.x)
# C_Q = run_model(res.x)
# plt.plot(C_Q_J, '.')
# plt.plot(C_Q, '.')


# %%


# %%



