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
            title=title,
            xlim=[timesteps[0], timesteps[-1]] if timesteps is not None else None,
        )
    else:
        ax.set(
            xlabel=f"Time {tunit}",
            title=title,
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
    obs, sim, lim, timesteps=None, varn="varn", axes=None, linestyle="-"
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
        label="Observation",
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
        label="Simulation",
        alpha=0.7,
        tunit="",
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