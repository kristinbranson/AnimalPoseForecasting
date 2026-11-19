# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: transformer312
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Same-start divergence: model futures vs. RatInABox futures
#
# For each val episode, `synthrat/samestart_rollouts.py` took the same 16-frame
# burn-in and drew K futures from the trained model and 2K futures from the
# RatInABox policy. Here we measure how far the model's distribution of futures
# is from RatInABox's, at several horizons, with three comparisons per episode
# (all against RatInABox set A = the first K RatInABox futures):
#
# - **model**: model futures vs. set A — the result.
# - **floor**: RatInABox set B (the other K futures) vs. set A. Two sample sets
#   from the *same* distribution; this is the divergence finite K alone produces.
# - **ceiling**: set A of a *different* episode vs. this episode's set A. What
#   the divergence looks like when the future ignores the burn-in entirely.
#
# Metrics (see `synthrat/trajectory_divergence.py`): Jensen–Shannon divergence
# (bits) of forward / sideways / angular velocity histograms and of 2D
# occupancy, Wasserstein-1 of the velocities (m/s, rad/s), and exact
# Wasserstein-1 of the positions at the horizon (m).
#
# Plug-in JS is biased upward at finite K, so read **model − floor** (and
# where the model sits between floor and ceiling), not absolute values.

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import os
import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import tqdm.auto as tqdm

import synthrat
from synthrat.sensory import rehydrate_env
from synthrat import trajectory_divergence as td
import ratinabox.utils
import apf.utils as utils

utils.set_mpl_backend('tkAgg')
ISNOTEBOOK = utils.is_notebook()
if ISNOTEBOOK:
    from IPython.display import display
else:
    plt.ion()
    display = print

# %% [markdown]
# ## Settings

# %%
SYNTHRAT_DIR = synthrat.__path__[0]
# Output folder of synthrat/samestart_rollouts.py
ROLLOUT_DIR = os.path.join(
    SYNTHRAT_DIR, 'data',
    'synthratdefault_20260430T003318_bestepoch100_ratinabox_rl_valdata_20260423_samestart_K20_H80')
# Raw RatInABox data the burn-ins came from: environment, and ground-truth
# kinematics for the histogram bin edges.
RAW_DATA_FILE = os.path.join(SYNTHRAT_DIR, 'data', 'ratinabox_rl_valdata_20260423.pkl')

HORIZONS = [10, 20, 40, 80]            # future frames (0.5, 1, 2, 4 s at dt = 0.05 s)
KINEMATIC_N_BINS = 25
KINEMATIC_PERCENTILES = (0.5, 99.5)    # of ground-truth kinematics; outer bin edges
OCCUPANCY_BIN_SIZE = 0.1               # m
CEILING_SEED = 0                       # picks which other episode each ceiling uses
N_EXAMPLE_EPISODES = 4

COMPARISONS = ('model', 'floor', 'ceiling')
COMPARISON_COLORS = {'model': 'C3', 'floor': 'C0', 'ceiling': 'C7'}
RESULTS_FILE = os.path.join(ROLLOUT_DIR, 'divergence_results.npz')
FIGDIR = os.path.join(ROLLOUT_DIR, 'figs')
os.makedirs(FIGDIR, exist_ok=True)

# %% [markdown]
# ## Load rollouts

# %%
rollout_files = sorted(glob.glob(os.path.join(ROLLOUT_DIR, 'epi*.pkl')))
rollouts = []
for path in tqdm.tqdm(rollout_files, desc='loading'):
    with open(path, 'rb') as f:
        rollouts.append(pickle.load(f))
n_episodes = len(rollouts)
assert n_episodes > 0, f'no rollouts in {ROLLOUT_DIR}'

burn_in = rollouts[0]['burn_in']
dt = rollouts[0]['dt']                                                    # seconds per frame
episodes = np.array([r['epi'] for r in rollouts])                         # (n_episodes,)
model_poses = np.stack([r['model'] for r in rollouts])                    # (n_episodes, K, T, 3)
ratinabox_poses = np.stack([r['ratinabox'] for r in rollouts])            # (n_episodes, 2K, T, 3)
gt_poses = np.stack([r['gt'] for r in rollouts])                          # (n_episodes, T, 3)
n_samples = model_poses.shape[1]
assert ratinabox_poses.shape[1] == 2 * n_samples
assert max(HORIZONS) <= model_poses.shape[2] - burn_in

print(f'{n_episodes} episodes, K = {n_samples} samples, burn-in {burn_in} frames, '
      f'{model_poses.shape[2] - burn_in} future frames, dt = {dt} s')
print('devices used:', {d: sum(r['device_used'] == d for r in rollouts) for d in {r['device_used'] for r in rollouts}})
print(f"mean time per episode: model {np.mean([r['time_model_s'] for r in rollouts]):.1f} s, "
      f"ratinabox {np.mean([r['time_ratinabox_s'] for r in rollouts]):.1f} s")
n_aborted = np.isnan(model_poses).any(axis=(2, 3)).sum()
print(f'model samples with NaN (aborted rollouts): {n_aborted} of {model_poses.shape[0] * n_samples}')

# %% [markdown]
# ## Environment and histogram bin edges
#
# Kinematic bin edges come from all ground-truth val episodes, so every
# comparison uses the same bins.

# %%
with open(RAW_DATA_FILE, 'rb') as f:
    raw_data = pickle.load(f)
Env = rehydrate_env(raw_data['env_info'])

gt_kinematics_per_episode = []
for track in raw_data['track']:
    theta = ratinabox.utils.get_angle(track['head_direction'], is_array=True)  # (T,)
    pose = np.concatenate([track['pos'], theta[:, None]], axis=1)              # (T, 3)
    gt_kinematics_per_episode.append(td.pose_to_kinematics(pose, dt))
gt_kinematics = {name: np.concatenate([k[name] for k in gt_kinematics_per_episode])
                 for name in td.KINEMATIC_NAMES}
del raw_data

kinematic_edges = td.make_kinematic_edges(gt_kinematics, KINEMATIC_N_BINS, KINEMATIC_PERCENTILES)
occupancy_edges = td.make_occupancy_edges(Env.extent, OCCUPANCY_BIN_SIZE)
for name, edges in kinematic_edges.items():
    print(f'{name}: [{edges[0]:.3f}, {edges[-1]:.3f}] {td.KINEMATIC_UNITS[name]}, {len(edges) - 1} bins')

# %% [markdown]
# ## Compute divergences
#
# `divergences[episode, horizon, comparison, metric]`. The ceiling for episode
# i uses episode `(i + shift) % n_episodes` for a random shift in
# [1, n_episodes), so no episode is compared with itself.

# %%
set_a = ratinabox_poses[:, :n_samples]                                    # (n_episodes, K, T, 3)
set_b = ratinabox_poses[:, n_samples:]                                    # (n_episodes, K, T, 3)
ceiling_shift = np.random.default_rng(CEILING_SEED).integers(1, n_episodes) if n_episodes > 1 else 0
ceiling_index = (np.arange(n_episodes) + ceiling_shift) % n_episodes      # (n_episodes,)

divergences = np.full((n_episodes, len(HORIZONS), len(COMPARISONS), len(td.METRIC_NAMES)), np.nan)
for i in tqdm.trange(n_episodes, desc='divergences'):
    others = {'model': model_poses[i], 'floor': set_b[i], 'ceiling': set_a[ceiling_index[i]]}
    for c, comparison in enumerate(COMPARISONS):
        metrics = td.compare_future_sets(others[comparison], set_a[i], burn_in, HORIZONS, dt,
                                         kinematic_edges, occupancy_edges)
        for m, name in enumerate(td.METRIC_NAMES):
            divergences[i, :, c, m] = metrics[name]

# Wall violations over the future (from the last burn-in frame on)
violations = {
    'model': td.count_wall_violations(model_poses.reshape(-1, *model_poses.shape[2:])[:, burn_in - 1:],
                                      Env.walls, Env.extent),
    'ratinabox': td.count_wall_violations(ratinabox_poses.reshape(-1, *ratinabox_poses.shape[2:])[:, burn_in - 1:],
                                          Env.walls, Env.extent),
}

np.savez(RESULTS_FILE, divergences=divergences, episodes=episodes, horizons=np.array(HORIZONS),
         comparisons=np.array(COMPARISONS), metric_names=np.array(td.METRIC_NAMES), dt=dt,
         ceiling_index=ceiling_index,
         **{f'{who}_{kind}': counts for who, v in violations.items() for kind, counts in v.items()})
print(f'saved {RESULTS_FILE}')

# %% [markdown]
# ## Summary table
#
# Mean ± SEM over episodes. `excess` is the paired model − floor difference;
# `relative` = (model − floor) / (ceiling − floor) of the means: 0 means the
# model is as close to RatInABox as a fresh RatInABox sample set, 1 means as
# far as futures from the wrong start. `frac > floor` is the fraction of
# episodes where the model's divergence exceeds the floor's.

# %%
def sem(x, axis=0):
    """Standard error of the mean over `axis`, ignoring NaN."""
    return np.nanstd(x, axis=axis, ddof=1) / np.sqrt(np.sum(~np.isnan(x), axis=axis))


rows = []
c_model, c_floor, c_ceiling = (COMPARISONS.index(c) for c in ('model', 'floor', 'ceiling'))
for m, name in enumerate(td.METRIC_NAMES):
    for h, horizon in enumerate(HORIZONS):
        values = divergences[:, h, :, m]                                  # (n_episodes, n_comparisons)
        excess = values[:, c_model] - values[:, c_floor]                  # (n_episodes,)
        means = np.nanmean(values, axis=0)
        sems = sem(values)
        rows.append({
            'metric': name, 'unit': td.METRIC_UNITS[name], 'horizon_s': horizon * dt,
            **{comparison: f'{means[c]:.4f} ± {sems[c]:.4f}' for c, comparison in enumerate(COMPARISONS)},
            'excess': f'{np.nanmean(excess):.4f} ± {sem(excess):.4f}',
            'relative': (means[c_model] - means[c_floor]) / (means[c_ceiling] - means[c_floor]),
            'frac > floor': np.nanmean(excess > 0),
        })
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(ROLLOUT_DIR, 'divergence_summary.csv'), index=False)
display(summary)

# %% [markdown]
# ## Divergence vs. horizon

# %%
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, (m, name) in zip(axes.flat, enumerate(td.METRIC_NAMES)):
    horizon_s = np.array(HORIZONS) * dt
    for c, comparison in enumerate(COMPARISONS):
        mean = np.nanmean(divergences[:, :, c, m], axis=0)               # (n_horizons,)
        err = sem(divergences[:, :, c, m])
        ax.plot(horizon_s, mean, 'o-', color=COMPARISON_COLORS[comparison], label=comparison)
        ax.fill_between(horizon_s, mean - err, mean + err, color=COMPARISON_COLORS[comparison], alpha=0.25)
    ax.set_title(name)
    ax.set_xlabel('horizon (s)')
    ax.set_ylabel(td.METRIC_UNITS[name])
axes.flat[0].legend()
fig.suptitle(f'Same-start divergence from RatInABox futures ({n_episodes} val episodes, K = {n_samples})')
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'divergence_vs_horizon.png'), dpi=150)

# %% [markdown]
# ## Pooled kinematic distributions
#
# All episodes' future steps up to the longest horizon, model vs. RatInABox
# set A vs. the recorded episode (one sample per burn-in), on the bins used
# for JS.

# %%
max_horizon = max(HORIZONS)
pooled = {
    'ratinabox': td.window_kinematics(set_a.reshape(-1, *set_a.shape[2:]), burn_in, max_horizon, dt),
    'model': td.window_kinematics(model_poses.reshape(-1, *model_poses.shape[2:]), burn_in, max_horizon, dt),
    'recorded': td.window_kinematics(gt_poses, burn_in, max_horizon, dt),
}
pooled_colors = {'ratinabox': 'C0', 'model': 'C3', 'recorded': 'k'}
fig, axes = plt.subplots(1, len(td.KINEMATIC_NAMES), figsize=(15, 4))
for ax, name in zip(axes, td.KINEMATIC_NAMES):
    edges = kinematic_edges[name]
    for who, kinematics in pooled.items():
        ax.hist(kinematics[name], bins=edges, density=True, histtype='step', lw=2,
                color=pooled_colors[who], label=who)
    ax.set_xlabel(f'{name} ({td.KINEMATIC_UNITS[name]})')
axes[0].set_ylabel('density')
axes[0].legend()
fig.suptitle(f'Future-step kinematics, first {max_horizon * dt:g} s after burn-in, pooled over episodes')
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'pooled_kinematics.png'), dpi=150)

# %% [markdown]
# ## Check: are the RatInABox alternatives faithful to recorded RatInABox?
#
# The reference futures come from re-running the RatInABox policy after a
# forced burn-in. The recorded episode is one draw from the real process
# after the same burn-in, so pooled over episodes, recorded futures should
# match RatInABox futures as well as two RatInABox draws match each other.
# All three rows use one future per burn-in (sample 0), so sample sizes are
# equal; each is compared with RatInABox set A sample 0.

# %%
reference = td.window_kinematics(set_a[:, 0], burn_in, max_horizon, dt)
single_future = {
    'recorded': gt_poses,                                                 # (n_episodes, T, 3)
    'ratinabox B': set_b[:, 0],
    'model': model_poses[:, 0],
}
rows = []
for who, poses in single_future.items():
    kinematics = td.window_kinematics(poses, burn_in, max_horizon, dt)
    row = {'vs ratinabox A': who}
    for name in td.KINEMATIC_NAMES:
        row[f'js_{name}'] = td.js_divergence(kinematics[name], reference[name], [kinematic_edges[name]])
        row[f'w1_{name}'] = scipy.stats.wasserstein_distance(kinematics[name], reference[name])
    rows.append(row)
reference_check = pd.DataFrame(rows)
reference_check.to_csv(os.path.join(ROLLOUT_DIR, 'reference_check.csv'), index=False)
display(reference_check)

# %% [markdown]
# ## Wall violations
#
# RatInABox agents can't cross walls; the model is not constrained, so this
# counts how often its futures do.

# %%
for who, v in violations.items():
    n_rows = len(v['crossings'])
    print(f"{who}: {np.mean(v['crossings'] > 0):.3%} of {n_rows} futures cross a wall at least once; "
          f"{np.mean(v['outside'] > 0):.3%} leave the arena; "
          f"mean {v['outside'].mean():.2f} frames outside")

# %% [markdown]
# ## Example episodes
#
# Left: RatInABox set A. Right: model. Burn-in in black, futures colored by
# time (dark = early).

# %%
example_rows = np.linspace(0, n_episodes - 1, N_EXAMPLE_EPISODES).astype(int)
fig, axes = plt.subplots(N_EXAMPLE_EPISODES, 2, figsize=(8, 4 * N_EXAMPLE_EPISODES), squeeze=False)
for row, i in enumerate(example_rows):
    for col, (who, poses) in enumerate((('ratinabox', set_a[i]), ('model', model_poses[i]))):
        ax = axes[row, col]
        Env.plot_environment(fig=fig, ax=ax)
        for pose in poses:
            ax.scatter(pose[burn_in:, 0], pose[burn_in:, 1], c=np.arange(pose.shape[0] - burn_in),
                       s=4, cmap='viridis', alpha=0.6)
        ax.plot(gt_poses[i, :burn_in, 0], gt_poses[i, :burn_in, 1], 'k-', lw=2)
        w1 = divergences[i, -1, COMPARISONS.index('model'), td.METRIC_NAMES.index('w1_position')]
        ax.set_title(f'episode {episodes[i]}: {who}' + (f'\nW1 position at {max_horizon * dt:g} s = {w1:.3f} m'
                                                        if who == 'model' else ''))
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'example_episodes.png'), dpi=150)
