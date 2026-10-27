# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inspect generated synthrat trajectories
#
# Checks on a trajectory pickle written by `synthrat.generate_data`, before it is used to
# train a model: what the agent sees at a few moments of an episode, what the trajectories
# look like, and how the velocities are distributed — the last matters because the model's
# labels are discretized velocities, so a long tail decides where the bin edges land.
#
# Generation itself lives in `synthrat/generate_data.py`; this notebook only reads its
# output.

# %%
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from synthrat.sensory import (compute_sensory, get_rect_polar_grid_shape, rehydrate_data)
from synthrat.plotting import plot_episode, visualize_sensory
from synthrat.generate_data import compute_velocity

# %% [markdown]
# ## Set parameters

# %%
thisdir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
datadir = os.path.join(thisdir, 'data')
figdir = os.path.join(thisdir, 'figs')
timestamp = '20260423'
episodefile = os.path.join(datadir, f'ratinabox_rl_valdata_{timestamp}.pkl')
# Episodes drawn in the sensory figure, and trajectories in the grid below.
n_sensory_episodes = 3
n_trajectory_episodes = 9
saveplots = False

os.makedirs(figdir, exist_ok=True)

# %% [markdown]
# ## Load episodes and rebuild the RatInABox objects

# %%
with open(episodefile, 'rb') as f:
    data = pickle.load(f)
episodes = {'track': data['track'], 'hidden': data['hidden']}
print(f"{len(episodes['track'])} episodes from {os.path.basename(episodefile)}")

rehydrated = rehydrate_data(data)
Env, Sensory = rehydrated['Env'], rehydrated['Sensory']
print(f"sensory populations: {sorted(Sensory) if Sensory else 'none saved'}")

# %%
# The boundary cells are laid out as a polar grid, which the model's conv2d embedding
# reads as an image; its shape is worth confirming against the config.
if Sensory and 'field_of_view_boundary' in Sensory \
        and data['sensory_info']['field_of_view_boundary']['cell_arrangement'] == 'rect_polar_grid':
    n_rings, n_neurons_per_ring = get_rect_polar_grid_shape(Sensory['field_of_view_boundary'])
    print(f"field-of-view boundary cells: {n_rings} rings x {n_neurons_per_ring} per ring")

# %% [markdown]
# ## What the agent sees
#
# One row per episode, at a moment part-way through it: the trajectory so far, then each
# sensory population's firing rates.

# %%
n_columns = 1 + len(Sensory)
fig, ax = plt.subplots(n_sensory_episodes, n_columns,
                       figsize=(5 * n_columns, 5 * n_sensory_episodes),
                       sharex='col', sharey='col')
fractions = np.linspace(0, 1, n_sensory_episodes + 2)[1:-1]
for row, episode in enumerate(range(n_sensory_episodes)):
    track = episodes['track'][episode]
    t = int(fractions[row] * track['pos'].shape[0])
    axcurr = {'traj': ax[row, 0]}
    for column, name in enumerate(Sensory.keys()):
        axcurr[name] = ax[row, column + 1]
    visualize_sensory(track, compute_sensory(track, Sensory), t, Env, Sensory,
                      ax=axcurr, fig=fig)
    axcurr['traj'].set_title(f'episode {episode}, t={t}')
if saveplots:
    fig.savefig(os.path.join(figdir, f'sensory_visualization_ep_{timestamp}.pdf'))

# %% [markdown]
# ## Example trajectories

# %%
n_columns = int(np.ceil(np.sqrt(n_trajectory_episodes)))
n_rows = int(np.ceil(n_trajectory_episodes / n_columns))
fig, ax = plt.subplots(n_rows, n_columns, sharex=True, sharey=True,
                       figsize=(5 * n_columns, 5 * n_rows))
for episode, axcurr in zip(range(n_trajectory_episodes), ax.flatten()):
    plot_episode(episodes['track'][episode], Env, axcurr=axcurr)
if saveplots:
    fig.savefig(os.path.join(figdir, f'example_trajectories_{timestamp}.pdf'))

# %% [markdown]
# ## Velocity distributions
#
# Per component: the histogram, the percentiles, and the spacing between percentiles. The
# spacing is plotted on a log axis because it shows directly how much resolution a
# uniform-in-percentile binning would give each part of the range.

# %%
velocities = {'forward_vel': [], 'sideways_vel': [], 'orientation_vel': []}
for episode in episodes['track']:
    forward_vel, sideways_vel, orientation_vel = compute_velocity(episode)
    velocities['forward_vel'].append(forward_vel)
    velocities['sideways_vel'].append(sideways_vel)
    velocities['orientation_vel'].append(orientation_vel)

fig, ax = plt.subplots(3, 3, figsize=(15, 15))
percentiles = np.linspace(0, 100, 26)
for column, (name, values) in enumerate(velocities.items()):
    pooled = np.concatenate(values)
    ax[0, column].hist(pooled, bins=50)
    ax[0, column].set_title(f'histogram of {name}')
    edges = np.percentile(pooled, percentiles)
    ax[1, column].plot(percentiles, edges, '.-')
    ax[1, column].set_title(f'{name} percentiles')
    ax[2, column].plot((edges[:-1] + edges[1:]) / 2, np.diff(edges), '.-')
    ax[2, column].set_title(f'{name} percentile spacing')
    ax[2, column].set_yscale('log')
fig.tight_layout()
if saveplots:
    fig.savefig(os.path.join(figdir, f'velocity_histograms_{timestamp}.pdf'))
