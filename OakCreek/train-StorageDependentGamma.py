# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from DiffFlowTransport.transport import SASTransport
from DiffFlowTransport.sas import SAS_Uniform, SAS_Gamma_StorageDependent
from DiffFlowTransport.utils import get_transport_obs_fluxes


max_iter = 500

# %% [markdown]
# # Read the data

# %%
df = pd.read_csv('data_withDiffusion_withSpinup-correctedZeroFlow.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df.head()

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

# %%
J, Q, ET, C_J, C_Q_J, time = get_transport_obs_fluxes(df_cut, 'Oakcreek')

# %% [markdown]
# ## Relative storage

# %%
# Relative storage ΔS: cumulative J - ET - Q (J = P2, rain + snowmelt), demeaned and
# linearly detrended to remove the drift from water-balance non-closure.
DS_MODE = "detrended_cumulative"
dS_cum = np.cumsum(np.asarray(J - ET - Q))
t_ind = np.arange(dS_cum.size)
dS = jnp.asarray(dS_cum - np.polyval(np.polyfit(t_ind, dS_cum, 1), t_ind))

# Initial age-ranked storage and tracer mass (cold start: all zeros).
sTmT_init = jnp.zeros([nt, 2])

# %% [markdown]
# ## Test run

# %%
# Reference SAS functions to check that the model runs end to end before calibration.
sas_Q = SAS_Gamma_StorageDependent(a=0.69, λ=-103., ΔScλ=-103.*48)
sas_ET = SAS_Uniform(scale=400.)
sas_transport = SASTransport(sas_Q, sas_ET, **params)
sT, mT, mQETs, pQETs, mRs, C_Q = sas_transport(J, C_J, Q, ET, sTmT_init, dS, ET)

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

mask = jnp.where(~jnp.isnan(C_Q_J))
mask_train = mask[0][(mask[0]>train_start_ind) & (mask[0]<train_end_ind)]
mask_test = mask[0][(mask[0]>test_start_ind) & (mask[0]<test_end_ind)]

# %%
def nse(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    denom = np.sum((obs - np.mean(obs))**2)
    if denom == 0:
        return np.nan  # undefined if obs has zero variance
    return 1 - np.sum((sim - obs)**2) / denom


def mse(obs, sim):
    return jnp.mean((obs - sim) ** 2)


# Run the transport model for parameters x = (a, λ, ΔScλ, ET scale); jit-compiled so
# the transport solver is traced once and reused across all objective evaluations.
@jax.jit
def run_model(x):
    sas_Q = SAS_Gamma_StorageDependent(a=x[0], λ=x[1], ΔScλ=x[2])
    sas_ET = SAS_Uniform(scale=x[3])
    sas_transport = SASTransport(sas_Q, sas_ET, **params)
    sT, mT, mQETs, pQETs, mRs, C_Q = sas_transport(J, C_J, Q, ET, sTmT_init, dS, ET)
    return C_Q.flatten()


# Training-period objectives
def obj_nse(x):
    return -nse(C_Q_J[mask_train], run_model(x)[mask_train])


def obj_mse(x):
    return mse(C_Q_J[mask_train], run_model(x)[mask_train])


def callback(xk):
    C_Q = run_model(xk)
    print("iter x =", xk,
          "mse (train) =", mse(C_Q_J[mask_train], C_Q[mask_train]),
          "mse (test) =", mse(C_Q_J[mask_test], C_Q[mask_test]),
          "nse (train) =", nse(C_Q_J[mask_train], C_Q[mask_train]),
          "nse (test) =", nse(C_Q_J[mask_test], C_Q[mask_test]))


# %%
# Starting values of λ for the multistart Nelder-Mead calibration.
LAMBDA0_STARTS = [-0.3, -1.0, -3.0]


def multistart_fit(obj_func, label):
    """Run Nelder-Mead from each starting point in LAMBDA0_STARTS and return the
    best result (lowest objective) and all candidates."""
    candidates = []
    for lam0 in LAMBDA0_STARTS:
        x0 = np.array([0.69, lam0, lam0 * 50., 400.])
        print(f"\n=== {label} fit, start λ0={lam0} (x0={x0.tolist()}) ===")
        res = minimize(obj_func, x0, method='nelder-mead', callback=callback,
                       options={'xatol': 1e-8, 'disp': True, "maxiter": max_iter})
        candidates.append((lam0, res))

    best = min((r for _, r in candidates), key=lambda r: r.fun)
    print(f"\n=== {label} multi-start summary ===")
    for lam0, r in candidates:
        flag = " <- selected" if r is best else ""
        print(f"  λ0={lam0}: fun={r.fun:.6f} success={r.success} nit={r.nit}{flag}")
    return best, candidates


def save_result(res, candidates, label):
    out = {
        "x": res.x.tolist(),
        "fun": float(res.fun),
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "ds_mode": DS_MODE,
        "multistart_lambda0": [lam0 for lam0, _ in candidates],
        "multistart_fun": [float(r.fun) for _, r in candidates],
    }
    fname = f"./models-withDiffusion-rev2/storage-dependent-gamma-{DS_MODE}-{label}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {fname}")


# %%
# Calibrate the benchmark against the training-period NSE and MSE.
for label, obj_func in [("nse", obj_nse), ("mse", obj_mse)]:
    res, candidates = multistart_fit(obj_func, label.upper())
    save_result(res, candidates, label)
