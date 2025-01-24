"""Utility plotting functions."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu

import pandas as pd
import matplotlib.pyplot as plt

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