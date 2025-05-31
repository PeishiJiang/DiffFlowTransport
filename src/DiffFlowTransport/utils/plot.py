"""Utility plotting functions."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .metrics import compute_metrics

# Figure configurations parameters
# rc('text', usetex=False)
small_size = 15
medium_size = 25
bigger_size = 30
plt.rc("font", size=small_size)  # controls default text sizes
plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
plt.rc("axes", labelsize=small_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("legend", fontsize=small_size)  # legend fontsize
plt.rc("figure", titlesize=small_size)  # fontsize of the figure title
plt.rc("text", usetex=False)

figsize_1 = (8, 5)
figsize_2 = (5, 5)


def plot_timeseries(
    array, timesteps=None,
    ax=None, title=None,
    label=None, tunit="[day of year]",
    ylabel='',
    alpha=1.0, xticks=None,
    linestyle="-", color="blue",
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_1)
    if timesteps is None:
        ax.plot(array, linestyle, color=color, label=label, alpha=alpha, markersize=3)
    else:
        ax.plot(
            timesteps,
            array,
            linestyle,
            color=color,
            label=label,
            alpha=alpha,
            markersize=3,
        )
    if xticks is None:
        ax.set(
            xlabel=f"Time {tunit}",
            title=title, ylabel=ylabel,
            xlim=[timesteps[0], timesteps[-1]] if timesteps is not None else None,
        )
    else:
        ax.set(
            xlabel=f"Time {tunit}",
            title=title, ylabel=ylabel,
            xlim=[timesteps[0], timesteps[-1]] if timesteps is not None else None,
            xticks=xticks,
        )
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    return ax


def plot_obs_1to1(obs, sim, limx, limy=None, varn="varn", color='black', ax=None, s=0.5, alpha=0.5):
    # ------ observation versus simulation ------
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_2)
    
    # mask = jnp.isfinite(obs)
    # obs, sim = obs[mask], sim[mask]
    # rmse = jnp.mean((obs - sim) ** 2)
    metrics = compute_metrics(sim, obs, True)
    rmse, nse = metrics['rmse'], metrics['nse']
    ax.scatter(obs, sim, color=color, s=s, alpha=alpha)
    if limy is None:
        ax.plot(limx, limx, "k--")
    ax.set(
        xlim=limx,
        ylim=limx if limy is None else limy,
        xlabel="Observation",
        ylabel="Simulation",
        title=f"{varn} (RMSE: {rmse:.3f}; NSE: {nse:.3f})" if limy is None else f"{varn}",
    )
    return ax


def plot_timeseries_obs_1to1(
    obs, sim, lim, timesteps=None, varn="varn", axes=None, linestyle="-",
    units='[-]', label_sim='Simulation', label_obs='Observation'
):
    if axes is None:
        fig = plt.figure(figsize=(15, 4))
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=(3, 1),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.1,
            hspace=0.3,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    else:
        ax1, ax2 = axes[0], axes[1]
    plot_timeseries(
        obs,
        timesteps=timesteps,
        ax=ax1,
        title=None,
        label=label_obs,
        alpha=1.0,
        tunit="",
        linestyle='.',
        color="black",
    )
    plot_timeseries(
        sim,
        timesteps=timesteps,
        ax=ax1,
        title=None,
        label=label_sim,
        alpha=0.7,
        tunit="",
        ylabel=units,
        linestyle=linestyle,
        color="tab:blue",
    )
    ax1.legend()
    ax1.set(title=varn)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    plot_obs_1to1(obs, sim, lim, ax=ax2, s=2, varn="")
    # plt.subplots_adjust(hspace=0.9)
    return fig, ax1, ax2  # pyright: ignore


def plot_timeseries_obs_sim(
    obs, sim, lim, timesteps=None, varn="varn", ax=None, linestyle="-",
    units='[-]', label_sim='Simulation', label_obs='Observation'
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    plot_timeseries(
        obs,
        timesteps=timesteps,
        ax=ax,
        title=None,
        label=label_obs,
        alpha=1.0,
        tunit="",
        linestyle='.',
        color="black",
    )
    plot_timeseries(
        sim,
        timesteps=timesteps,
        ax=ax,
        title=None,
        label=label_sim,
        alpha=0.7,
        tunit="",
        ylabel=units,
        linestyle=linestyle,
        color="tab:blue",
    )
    ax.legend()
    ax.set(title=varn)
    # plt.subplots_adjust(hspace=0.9)
    return ax  # pyright: ignore


def plot_PQET_ST(
    J, Q, sT, pQETs, dt, timesteps, age_cut=1000, axes=None, figsize=None
):
    assert J.shape == Q.shape
    assert len(J) == len(timesteps)
    assert len(Q) == len(timesteps)
    assert sT.shape[0] == pQETs.shape[0]
    assert sT.shape[1] == len(timesteps)+1
    assert pQETs.shape[1] == len(timesteps)

    PQ = jnp.cumsum(pQETs[...,0], axis=0) * dt
    PET = jnp.cumsum(pQETs[...,1], axis=0) * dt
    ST = jnp.cumsum(sT[:,1:], axis=0) * dt

    if figsize is None:
        figsize = (10,12)

    if axes is None:
        fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=False)

    ax = axes[0]
    ax.plot(timesteps, J, 'k')
    ax.set(title='$J$', ylabel='[mm/day]', xlim=[timesteps[0], timesteps[-1]], xticks=[])
    # ax.set(xlim=[df.index[0], df.index[1]])

    ax = axes[1]
    ax.plot(timesteps, Q, 'k')
    ax.set(title='$Q$', ylabel='[mm/day]', xlim=[timesteps[0], timesteps[-1]], xticks=[])

    ax = axes[2]
    im1=ax.imshow(ST[:age_cut], cmap='Blues', origin='lower', aspect='auto')
    ax.set(ylabel='Age $T$ \n [days]', title='$S_T$', xticks=[])
    cb_ax = fig.add_axes([.91,.124+0.32,.04,.1])
    fig.colorbar(im1,orientation='vertical',cax=cb_ax)

    ax = axes[3]
    im2=ax.imshow(PQ[:age_cut], cmap='gist_stern', origin='lower', vmin=0., vmax=1., aspect='auto')
    ax.set(ylabel='Age $T$ \n [days]', title='$P_Q$', xticks=[])
    cb_ax = fig.add_axes([.91,.124+0.16,.04,.1])
    fig.colorbar(im2,orientation='vertical',cax=cb_ax)

    ax = axes[4]
    im3=ax.imshow(PET[:age_cut], cmap='gist_stern', origin='lower', vmin=0., vmax=1., aspect='auto')
    ax.set(ylabel='Age $T$ \n [days]', title='$P_{ET}$')
    ax.set_xticks(np.arange(len(timesteps)))
    ax.set_xticklabels(timesteps.strftime("%Y"), rotation=30, ha="right")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    # plt.colorbar(im3, ax=ax)
    cb_ax = fig.add_axes([.91,.124,.04,.1])
    fig.colorbar(im3,orientation='vertical',cax=cb_ax)

    return axes


def plot_PQET_ST2(sT, pQETs, dt, last_age_cut=100, axes=None, figsize=None, label=''):
    assert sT.shape[0] == pQETs.shape[0]
    assert sT.shape[1] == pQETs.shape[0]+1

    pQ = pQETs[...,0]
    pET = pQETs[...,1]
    PQ = jnp.cumsum(pQ, axis=0) * dt
    PET = jnp.cumsum(pET, axis=0) * dt
    ST = jnp.cumsum(sT[:,1:], axis=0) * dt

    if figsize is None:
        figsize = (10,12)

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    
    for i in range(1,last_age_cut+1):
        axes[0,0].plot(ST[:,-i], PQ[:,-i], color='grey', alpha=0.5)
        axes[1,0].plot(ST[:,-i], PET[:,-i], color='grey', alpha=0.5)
        axes[0,1].plot(ST[:,-i], pQ[:,-i], color='grey', alpha=0.5)
        axes[1,1].plot(ST[:,-i], pET[:,-i], color='grey', alpha=0.5)
    axes[0,0].set(title=label, ylabel=r'$\Omega_Q$')
    axes[1,0].set(xlabel=r'$S_T$', ylabel=r'$\Omega_{ET}$')
    axes[0,1].set(title=label, ylabel=r'$\omega_Q$')
    axes[1,1].set(xlabel=r'$S_T$', ylabel=r'$\omega_{ET}$')
    return axes


def plot_PQET_quantile(pQETs, dt, timesteps, axes=None, figsize=None, label=''):

    def quantile_from_cdf(cdf, quantile=0.5):
        """
        Compute the median from a discretized cumulative density function (CDF).
        
        Parameters:
            x (numpy array): Sorted values corresponding to the CDF.
            cdf (numpy array): 2D array of cumulative densities (each row is a different CDF).
        
        Returns:
            numpy array: Estimated median values for each CDF.
        """
        # Find the first index where CDF >= 0.5 for each row
        # idx = np.apply_along_axis(lambda row: np.searchsorted(row, quantile), axis=1, arr=cdf)
        idx = jax.vmap(jnp.searchsorted, in_axes=(1, None))(cdf, quantile)

        quantile_xs = idx

        # # Initialize median array
        # quantile_xs = []

        # if x is None:
        #     x = jnp.arange(cdf.shape[0])

        # for i in range(cdf.shape[0]):  # Iterate over each CDF row
        #     if idx[i] == 0:
        #         quantile_x = x[0]  # If the first value already meets 0.5
        #     elif cdf[i, idx[i]] == 0.5:
        #         quantile_x = x[idx[i]]  # Exact match
        #     else:
        #         # Linear interpolation between x[idx-1] and x[idx]
        #         x1, x2 = x[idx[i] - 1], x[idx[i]]
        #         cdf1, cdf2 = cdf[i, idx[i] - 1], cdf[i, idx[i]]
        #         quantile_x = x1 + (quantile - cdf1) / (cdf2 - cdf1) * (x2 - x1)
        #     quantile_xs.append(quantile_x)

        return jnp.array(quantile_xs)

    PQ = jnp.cumsum(pQETs[...,0], axis=0) * dt
    PET = jnp.cumsum(pQETs[...,1], axis=0) * dt

    q1 = quantile_from_cdf(PQ, quantile=0.5) * dt
    q2 = quantile_from_cdf(PQ, quantile=0.25) * dt
    q3 = quantile_from_cdf(PQ, quantile=0.75) * dt

    q1b = quantile_from_cdf(PET, quantile=0.5) * dt
    q2b = quantile_from_cdf(PET, quantile=0.25) * dt
    q3b = quantile_from_cdf(PET, quantile=0.75) * dt

    if figsize is None:
        figsize = (10,10)

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax = axes[0]
    ax.fill_between(timesteps, q2, q3, color='blue', alpha=0.2)
    ax.plot(timesteps, q1, 'tab:blue')
    ax.set(title=f'Q age distribution ({label})', ylabel='Age [Days]')

    ax = axes[1]
    ax.fill_between(timesteps, q2b, q3b, color='blue', alpha=0.2)
    ax.plot(timesteps, q1b, 'tab:blue')
    ax.set(title=f'ET age distribution ({label})', ylabel='Age [Days]')

    return axes


def plot_J_Q(J, Q, timesteps, ax=None, figsize=None):
    # Plot
    if figsize is None:
        figsize = (8,4)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
    
    ax.plot(timesteps, Q, 'k')
    ax.set(ylabel='Streamflow \n [mm/d]', ylim=[0, np.max(Q) * 1.2])

    ax2 = ax.twinx()
    ax2.plot(timesteps, J, 'tab:blue')
    ax2.set(ylabel='Rainfall \n [mm/d]', ylim=[np.max(J) * 3, 0])

    return [ax, ax2]


# def plot_QT_ST(Q, pQ, sT, dt=1.0, ax=None, figsize=None):
def plot_PQET_ST3(
    Q, ET, pQETs, sT, dt=1.0, last_age_cut=100, axes=None, figsize=None, label=''
):
    assert sT.shape[0] == pQETs.shape[0]
    assert sT.shape[1] == pQETs.shape[0]+1

    pQETs2 = jnp.maximum(pQETs, 0)

    pQ = pQETs2[...,0]
    pET = pQETs2[...,1]
    PQ = jnp.cumsum(pQ, axis=0) * dt
    PET = jnp.cumsum(pET, axis=0) * dt
    ST = jnp.cumsum(sT[:,1:], axis=0) * dt
    S = ST[-1,:]

    ωQ = (PQ[1:,:] - PQ[:-1,:]) / (ST[1:,:] - ST[:-1,:])
    ωET = (PET[1:,:] - PET[:-1,:]) / (ST[1:,:] - ST[:-1,:])
    ωQ = jnp.nan_to_num(ωQ, nan=0.0, posinf=0.0, neginf=0.0)
    ωET = jnp.nan_to_num(ωET, nan=0.0, posinf=0.0, neginf=0.0)

    ωQ = jnp.maximum(ωQ, 0)
    ωET = jnp.maximum(ωET, 0)

    QT = jax.vmap(lambda a,b: a*b, in_axes=(0,1), out_axes=1)(Q, PQ)
    ETT = jax.vmap(lambda a,b: a*b, in_axes=(0,1), out_axes=1)(ET, PET)
    ST_scaled = jax.vmap(lambda a,b: a/b, in_axes=(1,0), out_axes=1)(ST,S)

    # Calculate the complement
    STC = jax.vmap(lambda a,b: a-b, in_axes=(0,1), out_axes=1)(S, ST)
    QTC = jax.vmap(lambda a,b: a-b, in_axes=(0,1), out_axes=1)(Q, QT)
    ETTC = jax.vmap(lambda a,b: a-b, in_axes=(0,1), out_axes=1)(ET, ETT)

    Qω = jax.vmap(lambda a,b: a*b, in_axes=(0,1), out_axes=1)(Q, ωQ)
    ETω = jax.vmap(lambda a,b: a*b, in_axes=(0,1), out_axes=1)(ET, ωET)

    if figsize is None:
        figsize = (10,12)

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False)
    
    for i in range(1,last_age_cut+1):
        axes[0,0].plot(STC[:,-i], QTC[:,-i], color='grey', alpha=0.5)
        # axes[0,0].plot(ST[:,-i], PQ[:,-i], color='grey', alpha=0.5)
        # axes[1,0].plot(ST_scaled[1:,-i], Qω[:,-i], color='grey', alpha=0.5)
        axes[1,0].plot(ST[1:,-i], Qω[:,-i], color='grey', alpha=0.5)
        # axes[0,1].plot(ST[:,-i], ETT[:,-i], color='grey', alpha=0.5)
        axes[0,1].plot(STC[:,-i], ETTC[:,-i], color='grey', alpha=0.5)
        # axes[1,1].plot(ST_scaled[1:,-i], ETω[:,-i], color='grey', alpha=0.5)
        axes[1,1].plot(ST[1:,-i], ETω[:,-i], color='grey', alpha=0.5)
    axes[0,0].set(title=label, xlabel=r'$\bar{S_T}$', ylabel=r'$\bar{Q_T}$')
    axes[1,0].set(xlabel=r'$S_T / S$', ylabel=r'$Q\omega_{Q}$')
    axes[0,1].set(title=label, xlabel=r'$\bar{S_T}$', ylabel=r'$\bar{ET_T}$')
    axes[1,1].set(xlabel=r'$S_T / S$', ylabel=r'$ET\omega_{ET}$')

    # plt.subplots_adjust(wspace=0.2, hspace=0.2)

    return axes


def plot_PQET_ST_selected(
    Q, pQETs, sT, timesteps, sel_time_ind=None,
    dt=1.0, last_age_cut=500, axes=None, figsize=None, label=''
):
    assert sT.shape[0] == pQETs.shape[0]
    assert sT.shape[1] == pQETs.shape[1]+1

    pQETs2 = jnp.maximum(pQETs, 0)
    pQETs2 = pQETs

    pQ = pQETs2[...,0]
    pET = pQETs2[...,1]
    PQ = jnp.cumsum(pQ, axis=0) * dt
    PET = jnp.cumsum(pET, axis=0) * dt
    ST = jnp.cumsum(sT[:,1:], axis=0) * dt
    S = ST[-1,:]

    ωQ = (PQ[1:,:] - PQ[:-1,:]) / (ST[1:,:] - ST[:-1,:])
    ωET = (PET[1:,:] - PET[:-1,:]) / (ST[1:,:] - ST[:-1,:])
    ωQ = jnp.nan_to_num(ωQ, nan=0.0, posinf=0.0, neginf=0.0)
    ωET = jnp.nan_to_num(ωET, nan=0.0, posinf=0.0, neginf=0.0)

    # ωQ = jnp.maximum(ωQ, 0)
    # ωET = jnp.maximum(ωET, 0)

    if figsize is None:
        figsize = (10,12)

    if axes is None:
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[0.5, 1, 1], hspace=0.3, wspace=0.3)
    
    if sel_time_ind is None:
        nt, nt_cut = Q.size, int(Q.size * 0.5)
        nt_res_mid = int((nt - nt_cut)/2)
        Q2 = Q[nt_cut:]
        sel_time_ind = jnp.argsort(Q2)[:2].tolist() + \
            jnp.argsort(Q2)[nt_res_mid:nt_res_mid+2].tolist() + \
            jnp.argsort(Q2)[-2:].tolist()
        sel_time_ind = [ind+nt_cut for ind in sel_time_ind]
    
    # Plot streamflow
    ax0 = fig.add_subplot(gs[0,:])
    ax0.plot(timesteps, Q, color='k')
    ax0.set(title='Streamflow', ylabel='mm/d')

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Plot SAS
    num_lines = len(sel_time_ind)
    cmap = cm.get_cmap('brg', num_lines)
    colors = [cmap(i) for i in range(num_lines)]
    for i in range(1,last_age_cut+1):
        ax1.plot(ST[:,-i], PQ[:,-i], color='grey', alpha=0.05)
        ax3.plot(ST[1:,-i], ωQ[:,-i], color='grey', alpha=0.05)
        ax2.plot(ST[:,-i], PET[:,-i], color='grey', alpha=0.05)
        ax4.plot(ST[1:,-i], ωET[:,-i], color='grey', alpha=0.05) 
    for i,it in enumerate(sel_time_ind):
        color = colors[i]
        ax0.axvline(x=timesteps[it], color=color, alpha=0.7)
        ax1.plot(ST[:,it], PQ[:,it], color=color, alpha=0.7)
        ax3.plot(ST[1:,it], ωQ[:,it], color=color, alpha=0.7)
        ax2.plot(ST[:,it], PET[:,it], color=color, alpha=0.7)
        ax4.plot(ST[1:,it], ωET[:,it], color=color, alpha=0.7)
    ax1.set(title=label, xlabel=r'$S_T$ [mm]', ylabel=r'$P_Q$')
    ax3.set(xlabel=r'$S_T$ [mm]', ylabel=r'$\omega_{Q}$')
    ax2.set(title=label, xlabel=r'$S_T$ [mm]', ylabel=r'$P_{ET}$')
    ax4.set(xlabel=r'$S_T$ [mm]', ylabel=r'$\omega_{ET}$')
    
    return axes


def plot_PQET_ST_esspi(
    Q, pQ, sT, timesteps, Q_units='[mm d-1]', sel_time_ind='default',
    dt=1.0, last_age_cut=500, axes=None, figsize=None, label=''
):
    assert sT.shape[0] == pQ.shape[0]
    assert sT.shape[1] == pQ.shape[1]+1

    PQ = jnp.cumsum(pQ, axis=0) * dt
    ST = jnp.cumsum(sT[:,1:], axis=0) * dt
    S = ST[-1,:]

    ωQ = (PQ[1:,:] - PQ[:-1,:]) / (ST[1:,:] - ST[:-1,:])
    ωQ = jnp.nan_to_num(ωQ, nan=0.0, posinf=0.0, neginf=0.0)

    if figsize is None:
        figsize = (10,12)

    if axes is None:
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[0.5, 1, 1], hspace=0.3, wspace=0.3)
    
    if sel_time_ind == 'default':
        nt, nt_cut = Q.size, 0
        nt_res_mid = int((nt - nt_cut)/2)
        Q2 = Q[nt_cut:]
        sel_time_ind = jnp.argsort(Q2)[:2].tolist() + \
            jnp.argsort(Q2)[nt_res_mid:nt_res_mid+2].tolist() + \
            jnp.argsort(Q2)[-2:].tolist()
        sel_time_ind = [ind+nt_cut for ind in sel_time_ind]
        num_lines = len(sel_time_ind)
        cmap = cm.get_cmap('brg', num_lines)
        colors = [cmap(i) for i in range(num_lines)]
    
    # Plot streamflow
    ax0 = fig.add_subplot(gs[0,:])
    ax0 = plot_timeseries(
        Q, timesteps=timesteps, ax=ax0, title=None,
        label='Observation', ylabel=f'$Q$ {Q_units}', linestyle='.', color='k'
    )
    ax0.set(title='$Q$', xlabel='')

    # Plot TTDs
    ax1 = fig.add_subplot(gs[1:, :])
    
    # Plot SAS
    for i in range(1,last_age_cut+1):
        ax1.plot(PQ[:,-i], color='grey', alpha=0.05)
    if sel_time_ind is not None:
        for i,it in enumerate(sel_time_ind):
            color = colors[i]
            ax0.axvline(x=timesteps[it], color=color, alpha=0.7)
            ax1.plot(PQ[:,it], color=color, alpha=0.7)
    ax1.set(title='Transit time distribution', xlabel=r'$T$ [d]', ylim=[-0.05,1.1], ylabel=r'$P_Q$')
    
    return [ax0, ax1]


def plot_young_water(
    pQ, timesteps, cutoff_age=100, dt=1., axes=None, figsize=None, label=''
):
    max_young_age = cutoff_age * dt
    PQ = jnp.cumsum(pQ, axis=0) * dt

    # Calculate the young water fraction
    PQ_young_fraction = PQ[cutoff_age,:]

    # Calculate the mean age of the young water fraction (or young water age)
    pQ_young = pQ[:cutoff_age,:]
    ages = jnp.arange(1, cutoff_age+1) * dt
    weighted_pQ_young = jax.vmap(lambda a,b: a*b, in_axes=(1, None))(pQ_young, ages)
    weighted_pQ_young = jnp.sum(weighted_pQ_young, axis=1) / PQ_young_fraction

    # Plot
    if figsize is None:
        figsize = (8,6)

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax = axes[0]
    ax.plot(timesteps, PQ_young_fraction, 'k')
    ax.set(title=f'Young water fraction with age smaller than {max_young_age} days ({label})', 
           ylabel='[-]',ylim=[0, 1])

    ax = axes[1]
    ax.plot(timesteps, weighted_pQ_young, 'k')
    ax.set(title=f'Mean age of young water (smaller than {max_young_age} days) ({label})', 
           ylabel='[Day]')

    return axes


def plot_flow_transport_assessment_esspi(
    J, Q, C_J, C_Q, timesteps,
    Q_sim=None, C_Q_sim=None,
    sim_label='Simulation', obs_label='Observation',
    varn_c='Tracer', Q_units='[mm d-1]', C_units='[-]', 
    figsize=None,
):
    # Create figure and grid spec
    if figsize is None:
        figsize = (10,12)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.2, wspace=0.2)

    # Top two layered subplots (share the same position)
    ax1 = fig.add_subplot(gs[0, :])  # Span both columns
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1,)  # Overlayed plot

    # Plot streamflow simulation
    ax1 = plot_timeseries(
        Q, timesteps=timesteps, ax=ax1, title=None,
        label=obs_label, ylabel=f'$Q$ {Q_units}', linestyle='.', color='k'
    )
    ax1b = ax1.twinx()
    ax1b = plot_timeseries(
        J, timesteps=timesteps, ax=ax1b, title=None,
        label=obs_label, ylabel=f'$J$ {Q_units}', linestyle='.', color='lightblue'
    )
    ax1b.set(ylim=[J.max()*2.0, 0])
    ax1b.set_ylabel(f'$J$ {Q_units}', color="lightblue")
    ax1b.tick_params(axis='y', colors='lightblue')
    if Q_sim is not None:
        metrics = compute_metrics(Q_sim, Q, True)
        rmse, nse = metrics['rmse'], metrics['nse']
        print(f'RMSE: {rmse:.2f}; NSE: {nse:.2f}')
        ax1 = plot_timeseries(
            Q_sim, timesteps=timesteps, ax=ax1, 
            # title=f'Flow simulation (RMSE: {rmse:.2f}; NSE: {nse:.2f})',
            label=sim_label, ylabel=f'$Q$ {Q_units}', alpha=0.7, color='tab:blue'
        )
        ax1.legend(loc='center left')
    ax1.set(xlabel='', title='Water quantity', ylim=[np.nanmin(Q)*0.8, np.nanmax(Q)*1.3])

    # Plot transport simulation
    # C_Q = Q * C_Q
    ax2 = plot_timeseries(
        C_Q, timesteps=timesteps, ax=ax2, title=None,
        label=obs_label, ylabel=r'$C_Q$ ' + f'{C_units}', linestyle='.', color='k'
    )
    ax2b = ax2.twinx()
    ax2b = plot_timeseries(
        C_J, timesteps=timesteps, ax=ax2b, title=None,
        label=obs_label, ylabel=r'$C_J$ ' + f'{C_units}', linestyle='.', color='lightblue'
    )
    ax2b.set(ylim=[J.max()*2.0, 0])
    ax2b.set_ylabel(r'$C_J$ ' + f'{C_units}', color="lightblue")
    ax2b.tick_params(axis='y', colors='lightblue')
    if C_Q_sim is not None:
        # C_Q_sim = Q_sim * C_Q_sim
        metrics = compute_metrics(C_Q_sim, C_Q, True)
        rmse, nse = metrics['rmse'], metrics['nse']
        print(f'RMSE: {rmse:.2f}; NSE: {nse:.2f}')
        ax2 = plot_timeseries(
            C_Q_sim, timesteps=timesteps, ax=ax2, 
            # title=f'{varn_c} concentration simulation by hybrid SAS model (RMSE: {rmse:.2f}; NSE: {nse:.2f})',
            label=sim_label, ylabel=r'$C_Q$ ' + f'{C_units}', alpha=0.7, color='tab:blue'
        )
        ax2.legend(loc='center left')
    ax2.set(xlabel='', title=f'{varn_c} concentration', ylim=[np.nanmin(C_Q)*0.8, np.nanmax(C_Q)*1.3])
        
    return [ax1, ax1b, ax2, ax2b]


def plot_flow_transport_assessment(
    Q, Q_sim, C_Q, C_Q_sim, pQ, timesteps,
    sim_label='Simulation', obs_label='Observation',
    varn_c='Tracer', Q_units='[mm d-1]', C_units='[-]', 
    sel_time_ind=None, last_age_cut=500, dt=1.0, 
    cutoff_age=100, cutoff_time=None, figsize=None,
):
    # Create figure and grid spec
    if figsize is None:
        figsize = (16,12)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.2)

    # Top two layered subplots (share the same position)
    ax1 = fig.add_subplot(gs[0, :])  # Span both columns
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1,)  # Overlayed plot

    # Bottom two side-by-side subplots
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    # Plot streamflow simulation
    metrics = compute_metrics(Q_sim, Q, True)
    rmse, nse = metrics['rmse'], metrics['nse']
    ax1 = plot_timeseries(
        Q, timesteps=timesteps, ax=ax1, title=None,
        label=obs_label, ylabel=Q_units, linestyle='.', color='k'
    )
    ax1 = plot_timeseries(
        Q_sim, timesteps=timesteps, ax=ax1, 
        title=f'Flow simulation by LSTM (RMSE: {rmse:.2f}; NSE: {nse:.2f})',
        label=sim_label, ylabel=Q_units, alpha=0.7, color='tab:blue'
    )
    ax1.set(xlabel='')
    ax1.legend()

    # Plot transport simulation
    metrics = compute_metrics(C_Q_sim, C_Q, True)
    rmse, nse = metrics['rmse'], metrics['nse']
    ax2 = plot_timeseries(
        C_Q, timesteps=timesteps, ax=ax2, title=None,
        label=obs_label, ylabel=C_units, linestyle='.', color='k'
    )
    ax2 = plot_timeseries(
        C_Q_sim, timesteps=timesteps, ax=ax2, 
        title=f'{varn_c} concentration simulation by hybrid SAS model (RMSE: {rmse:.2f}; NSE: {nse:.2f})',
        label=sim_label, ylabel=C_units, alpha=0.7, color='tab:blue'
    )
    ax2.set(xlabel='')
    ax2.legend()

    # Plot TTD for Q
    PQ = jnp.cumsum(pQ, axis=0) * dt
    if sel_time_ind is None:
        if cutoff_time is None:
            nt_cut = int(Q.size * 0.5)
        else:
            nt_cut = cutoff_time
        nt = Q.size
        nt_res_mid = int((nt - nt_cut)/2)
        Q2 = Q[nt_cut:]
        sel_time_ind = jnp.argsort(Q2)[:2].tolist() + \
            jnp.argsort(Q2)[nt_res_mid:nt_res_mid+2].tolist() + \
            jnp.argsort(Q2)[-2:].tolist()
        sel_time_ind = [ind+nt_cut for ind in sel_time_ind]
    num_lines = len(sel_time_ind)
    cmap = cm.get_cmap('brg', num_lines)
    colors = [cmap(i) for i in range(num_lines)]
    for i in range(1,last_age_cut+1):
        ax3.plot(PQ[:,-i], color='grey', alpha=0.05)
    for i,it in enumerate(sel_time_ind):
        color = colors[i]
        ax1.axvline(x=timesteps[it], linewidth=3, color=color, alpha=0.7)
        ax2.axvline(x=timesteps[it], linewidth=3, color=color, alpha=0.7)
        ax3.plot(PQ[:,it], color=color, linewidth=3, alpha=0.7)
    ax3.set(title='Transit time distribution $P_Q$', xlabel=r'Age $T$ [d]', ylabel=r'$P_Q$')

    # Plot young water age
    if cutoff_time is None:
        cutoff_time = int(Q.size * 0.5)
    max_young_age = cutoff_age * dt
    PQ_young_fraction = PQ[cutoff_age,:]
    ax1.axvline(x=timesteps[cutoff_time], linewidth=5, linestyle='--', color='k', alpha=.7)
    ax2.axvline(x=timesteps[cutoff_time], linewidth=5, linestyle='--', color='k', alpha=.7)
    ax4.plot(timesteps[cutoff_time:], PQ_young_fraction[cutoff_time:], 'k')
    for i,it in enumerate(sel_time_ind):
        color = colors[i]
        ax4.axvline(x=timesteps[it], linewidth=3, color=color, alpha=0.7)
    ax4.set(title=f'Young water fraction with age smaller than {max_young_age} days', 
           ylabel='[-]',ylim=[0, 1],xlim=[timesteps[nt_cut],timesteps[-1]])
    
    # Create a colorbar for the lines
    listed_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=num_lines - 1)
    sm = plt.cm.ScalarMappable(cmap=listed_cmap, norm=norm)
    sm.set_array([])

    # Use make_axes_locatable to create a new axis below ax_main
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes("bottom", size="5%", pad=10)
    # Add an inset axis for the colorbar, relative to the main axis
    cax = inset_axes(ax3,
                    width="80%",   # width relative to parent axis
                    height="50%",   # height relative to parent axis
                    loc='lower center',
                    bbox_to_anchor=(0.1, -0.5, 0.8, 0.1),  # (x0, y0, width, height) in axis coords
                    bbox_transform=ax3.transAxes,
                    borderpad=0)

    # Add horizontal colorbar to this new axis
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_ticks([])

    # Add labels to both ends
    cax.text(0, -1.5, 'Low Flow', va='center', ha='left', transform=cax.transAxes, fontsize=15)
    cax.text(1, -1.5, 'High Flow', va='center', ha='right', transform=cax.transAxes, fontsize=15)
        
    return [ax1, ax2, ax3, ax4]


def plot_flow_transport_assessment2(
    sT, Q, Q_sim, C_Q, C_Q_sim, pQ, timesteps,
    sim_label='Simulation', obs_label='Observation',
    varn_c='Tracer', Q_units='[mm d-1]', C_units='[-]', 
    sel_time_ind=None, last_age_cut=500, dt=1.0, 
    cutoff_age=100, cutoff_time=None, figsize=None,
):
    # Create figure and grid spec
    if figsize is None:
        figsize = (16,20)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.2)

    # Top two layered subplots (share the same position)
    ax1 = fig.add_subplot(gs[0, :])  # Span both columns
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1,)  # Overlayed plot

    # Bottom two side-by-side subplots
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[3, 0])
    ax6 = fig.add_subplot(gs[3, 1])

    # Plot streamflow simulation
    metrics = compute_metrics(Q_sim, Q, True)
    rmse, nse = metrics['rmse'], metrics['nse']
    ax1 = plot_timeseries(
        Q, timesteps=timesteps, ax=ax1, title=None,
        label=obs_label, ylabel=Q_units, linestyle='.', color='k'
    )
    ax1 = plot_timeseries(
        Q_sim, timesteps=timesteps, ax=ax1, 
        title=f'Flow simulation by LSTM (RMSE: {rmse:.2f}; NSE: {nse:.2f})',
        label=sim_label, ylabel=Q_units, alpha=0.7, color='tab:blue'
    )
    ax1.set(xlabel='')
    ax1.legend()

    # Plot transport simulation
    metrics = compute_metrics(C_Q_sim, C_Q, True)
    rmse, nse = metrics['rmse'], metrics['nse']
    ax2 = plot_timeseries(
        C_Q, timesteps=timesteps, ax=ax2, title=None,
        label=obs_label, ylabel=C_units, linestyle='.', color='k'
    )
    ax2 = plot_timeseries(
        C_Q_sim, timesteps=timesteps, ax=ax2, 
        title=f'{varn_c} concentration simulation by hybrid SAS model (RMSE: {rmse:.2f}; NSE: {nse:.2f})',
        label=sim_label, ylabel=C_units, alpha=0.7, color='tab:blue'
    )
    ax2.set(xlabel='')
    ax2.legend()

    # Plot TTDs for Q
    PQ = jnp.cumsum(pQ, axis=0) * dt
    if sel_time_ind is None:
        if cutoff_time is None:
            nt_cut = int(Q.size * 0.5)
        else:
            nt_cut = cutoff_time
        nt = Q.size
        nt_res_mid = int((nt - nt_cut)/2)
        Q2 = Q[nt_cut:]
        sel_time_ind = jnp.argsort(Q2)[:2].tolist() + \
            jnp.argsort(Q2)[nt_res_mid:nt_res_mid+2].tolist() + \
            jnp.argsort(Q2)[-2:].tolist()
        sel_time_ind = [ind+nt_cut for ind in sel_time_ind]
    num_lines = len(sel_time_ind)
    cmap = cm.get_cmap('brg', num_lines)
    colors = [cmap(i) for i in range(num_lines)]

    # Plot PQ versus ST
    ST = jnp.cumsum(sT[:,1:], axis=0) * dt
    for i in range(1,last_age_cut+1):
        ax3.plot(ST[:,-i], PQ[:,-i], color='grey', alpha=0.05)
    for i,it in enumerate(sel_time_ind):
        color = colors[i]
        ax3.plot(ST[:,it], PQ[:,it], color=color, linewidth=3, alpha=0.7)
    ax3.set(title='$P_Q$ versus $S_T$', xlabel=r'$S_T$ [mm]', ylabel=r'$P_Q$')

    # Plot QT versus ST
    S = ST[-1,:]
    QT = jax.vmap(lambda a,b: a*b, in_axes=(0,1), out_axes=1)(Q, PQ)
    STC = jax.vmap(lambda a,b: a-b, in_axes=(0,1), out_axes=1)(S, ST)
    QTC = jax.vmap(lambda a,b: a-b, in_axes=(0,1), out_axes=1)(Q, QT)
    for i in range(1,last_age_cut+1):
        ax4.plot(STC[:,-i], QTC[:,-i], color='grey', alpha=0.05)
    for i,it in enumerate(sel_time_ind):
        color = colors[i]
        ax4.plot(STC[:,it], QTC[:,it], color=color, linewidth=3, alpha=0.7)
    ax4.set(title=r'$\bar{Q_T}$ versus $\bar{S_T}$', xlabel=r'$\bar{S_T}$ [mm]', ylabel=r'$\bar{Q_T}$ [mm]')

    # Plot PQ versus T
    for i in range(1,last_age_cut+1):
        ax5.plot(PQ[:,-i], color='grey', alpha=0.05)
    for i,it in enumerate(sel_time_ind):
        color = colors[i]
        ax1.axvline(x=timesteps[it], linewidth=3, color=color, alpha=0.7)
        ax2.axvline(x=timesteps[it], linewidth=3, color=color, alpha=0.7)
        ax5.plot(PQ[:,it], color=color, linewidth=3, alpha=0.7)
    ax5.set(title='$P_Q$ versus $T$', xlabel=r'Age $T$ [d]', ylabel=r'$P_Q$')

    # Plot young water age
    if cutoff_time is None:
        cutoff_time = int(Q.size * 0.5)
    max_young_age = cutoff_age * dt
    PQ_young_fraction = PQ[cutoff_age,:]
    ax1.axvline(x=timesteps[cutoff_time], linewidth=5, linestyle='--', color='k', alpha=.7)
    ax2.axvline(x=timesteps[cutoff_time], linewidth=5, linestyle='--', color='k', alpha=.7)
    ax6.plot(timesteps[cutoff_time:], PQ_young_fraction[cutoff_time:], 'k')
    for i,it in enumerate(sel_time_ind):
        color = colors[i]
        ax6.axvline(x=timesteps[it], linewidth=3, color=color, alpha=0.7)
    ax6.set(title=f'Young water fraction with age smaller than {max_young_age} days', 
           ylabel='[-]',ylim=[0, 1],xlim=[timesteps[nt_cut],timesteps[-1]])
        
    return [ax1, ax2, ax3, ax4, ax5, ax6]