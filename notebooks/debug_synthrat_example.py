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
# # Debug / sanity checks for the RatInABox synthetic-rat dataset
#
# Focused checks on the dataset built by `experiments.synthrat.make_dataset`,
# independent of any trained model. The main one verifies that inverting the
# velocity **labels** reconstructs the true pose — this is the round-trip that
# `simulate` and the training debug plots rely on.
#
# Requires a patched RatInABox — see `environment/README.md`.

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import apf.dataset
import apf.utils as utils
from apf.io import read_config

import experiments.synthrat as synthrat_exp
from synthrat.config import read_config_kwargs, DEFAULTCONFIGFILE
from synthrat.sensory import rehydrate_data

import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# %% [markdown]
# ## Build a small dataset
#
# `debug=True` keeps only a handful of episodes so this is fast.

# %%
config = read_config(DEFAULTCONFIGFILE, **read_config_kwargs)

dataset, info = synthrat_exp.make_dataset(config, config['invalfile'], debug=True)
dataloader = apf.dataset.DataLoader(dataset, batch_size=8, shuffle=False)

print(f'{len(dataset)} chunks, context_length={dataset.context_length}')
print('inputs:', dict(dataset.input_idx))
print('d_output_discrete=%d, nbins=%d' % (dataset.d_output_discrete, dataset.discretize_nbins))

# The true pose is carried per chunk in metadata so it can seed the inversion
# (GlobalVelocity.invert has no default starting pose — without x0 it integrates
# from the origin with zero heading).
example = dataset.item_to_data(utils.convert_torch_to_numpy(next(iter(dataloader))))
assert example.get('metadata', {}).get('pose') is not None, \
    "chunks carry no 'pose' metadata — rebuild with the current make_dataset"
true_pose = np.asarray(example['metadata']['pose'])   # (B, T, 3): x, y, orientation
print('true_pose (metadata) shape:', true_pose.shape)

# %% [markdown]
# ## Check 1 — analytic GlobalVelocity round-trip
#
# Apply `GlobalVelocity` to the true pose, then invert it from each chunk's true
# start pose `x0 = (x, y, orientation)`. With no discretization in the loop this
# must recover the true pose to floating-point precision — it isolates the
# velocity↔pose math from any quantization.

# %%
x0 = true_pose[:, 0, :]                                # (B, 3)
gv = apf.dataset.GlobalVelocity(tspred=[1])
true_vel = gv.apply(true_pose)                         # (B, T, 3)
recov_exact = gv.invert(true_vel, x0=x0)              # (B, T, 3)

exact_pos = np.abs(recov_exact[..., :2] - true_pose[..., :2])
exact_ang = np.abs(utils.modrange(recov_exact[..., 2] - true_pose[..., 2], -np.pi, np.pi))
print('analytic round-trip: max pos err %.2e m, max ang err %.2e deg'
      % (exact_pos.max(), np.degrees(exact_ang.max())))
assert exact_pos.max() < 1e-6 and np.degrees(exact_ang.max()) < 1e-4, \
    "GlobalVelocity forward/inverse is not an identity — inversion math is wrong"
print('PASS: GlobalVelocity forward then inverse recovers the pose exactly.')

# %% [markdown]
# ## Check 2 — full discretized label chain
#
# Invert the actual labels the model is trained on
# (`Discretize → Zscore → GlobalVelocity`) from the true start pose. The residual
# here is the **discretization floor** (bin quantization), not an inversion error,
# so it should be small and shrink if `discretize_nbins` is increased.

# %%
recov_lbl = apf.dataset.apply_inverse_operations(
    example['labels']['velocity'],
    extraargs={'discretize': {'do_sampling': False},
               'globalvelocity': {'x0': x0}})          # (B, T, 3)

lbl_pos = np.abs(recov_lbl[..., :2] - true_pose[..., :2])
lbl_ang = np.abs(utils.modrange(recov_lbl[..., 2] - true_pose[..., 2], -np.pi, np.pi))
print('discretized labels : max pos err %.4f m (mean %.4f), '
      'max ang err %.2f deg (mean %.2f)'
      % (lbl_pos.max(), lbl_pos.mean(),
         np.degrees(lbl_ang.max()), np.degrees(lbl_ang.mean())))
print('(this is the %d-bin discretization floor, not an inversion error)'
      % dataset.discretize_nbins)

# %% [markdown]
# ## Visual check — recovered vs true trajectory
#
# Overlay the true chunk trajectory, the analytic round-trip (should be exactly on
# top), and the label-chain reconstruction (should track within a bin), on the
# environment.

# %%
ratinabox_info = rehydrate_data(info)
Env = ratinabox_info['Env']

nplot = min(3, true_pose.shape[0])
fig, axes = plt.subplots(1, nplot, figsize=(5 * nplot, 5), squeeze=False)
for j in range(nplot):
    ax = axes[0, j]
    Env.plot_environment(fig=fig, ax=ax, autosave=False)
    ax.plot(true_pose[j, :, 0], true_pose[j, :, 1], '-o', ms=4, c='k',
            label='true', zorder=3)
    ax.plot(recov_exact[j, :, 0], recov_exact[j, :, 1], '--', lw=2, c='C2',
            label='analytic round-trip', zorder=4)
    ax.plot(recov_lbl[j, :, 0], recov_lbl[j, :, 1], ':', lw=2, c='C3',
            label='label-chain', zorder=5)
    ax.set_aspect('equal')
    ax.set_title(f'chunk {j}')
axes[0, 0].legend(loc='best', fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Heading check
#
# The reconstructed orientation should track the true orientation (the label chain
# adds only quantization). Plotted per chunk over the context window.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
t = np.arange(true_pose.shape[1])
for j in range(nplot):
    ax.plot(t, np.degrees(true_pose[j, :, 2]), '-o', ms=3, c=f'C{j}', label=f'true chunk {j}')
    ax.plot(t, np.degrees(recov_lbl[j, :, 2]), ':', c=f'C{j}', label=f'label chunk {j}')
ax.set_xlabel('frame in chunk')
ax.set_ylabel('orientation (deg)')
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
plt.show()
