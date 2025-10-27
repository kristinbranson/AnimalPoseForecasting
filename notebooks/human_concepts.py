# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: transformer
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
    
import numpy as np

import matplotlib.pyplot as plt
import apf.io as io
import apf.utils as utils
utils.set_mpl_backend('tkAgg')

from flyllm.config import DEFAULTCONFIGFILE, posenames
from flyllm.features import featglobal, get_sensory_feature_idx
import time
from flyllm.plotting import plot_fly, plot_flies



# %%
# crete a timestamp string if we want to save any results/plots
timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())

# %%
# read inf config
configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_20251007.json"
config = io.read_config(
    configfile,
    default_configfile=DEFAULTCONFIGFILE,
    posenames=posenames,
    featglobal=featglobal,
    get_sensory_feature_idx=get_sensory_feature_idx,
)

# %%
# load in the raw data
rawdata = io.load_raw_npz_data(config['intrainfile'])

#train_dataset, flyids, track, pose, velocity, sensory, dataset_params = make_dataset(config, 'intrainfile', return_all=True, debug=True)

# %%
# what is in the raw data
print(rawdata.keys())
print(rawdata['X'].shape) # (nkpts, 2, nframes, nflies)
print(rawdata['isstart'].shape)

# %%
# plot one fly
flyidx = 1
frame = 0
fig,ax = plt.subplots()
ax.plot(rawdata['X'][:,0,frame,flyidx], rawdata['X'][:,1,frame,flyidx], 'k.')
plot_fly(rawdata['X'][:,:,frame,flyidx], ax=ax, kpt_marker='o', kpt_ms=20, textlabels='keypoints')
ax.axis('equal')


# %%
# plot all flies
fig,ax = plt.subplots()
#ax.plot(rawdata['X'][:,0,frame,:], rawdata['X'][:,1,frame,:], 'k.')
plot_flies(rawdata['X'][:,:,frame,:], ax=ax)
ax.axis('equal')

# %%
centeridx = 7
fps = 150.

# compute the velocity vector
# X is # (nkpts, 2, nframes, nflies)
vel = rawdata['X'][centeridx,:,1:,:] - rawdata['X'][centeridx,:,:-1,:]  # (2, nframes-1, nflies)
vel[:,rawdata['isstart'][1:,:]] = np.nan  # set velocity between end frames and start frames to nan
vel_mmps = vel * fps  # convert to mm/s

velmag = np.linalg.norm(vel, axis=0)  # (nframes-1, nflies)
velmag_mmps = velmag * fps  # convert to mm/s

maxvelmag = 50

# plot the velocity of one fly over time
fig,ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
ax[0].plot(vel_mmps[0,:,flyidx], '-')
ax[0].set_ylabel('Vx')
ax[0].set_ylim([-maxvelmag, maxvelmag])
ax[1].plot(vel_mmps[1,:,flyidx], '-')
ax[1].set_ylabel('Vy')
ax[1].set_ylim([-maxvelmag, maxvelmag])
ax[2].plot(velmag_mmps[:,flyidx], '-')
ax[2].set_ylabel('|V|')
ax[2].set_ylim([0, maxvelmag])
ax[2].set_xlabel('Frame')

ax[2].set_xlim(0,10000)


# %%
# histogram the velocity magnitudes

minvelmag = 1e-1
maxvelmag = 100
nbins = 100
bin_edges = np.logspace(np.log10(minvelmag), np.log10(maxvelmag), num=nbins+1)
counts,bin_edges = np.histogram(velmag_mmps[~np.isnan(velmag_mmps)],bins=bin_edges)
fig,ax = plt.subplots()
ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge')
ax.set_xlabel('Velocity magnitude (mm/s)')
ax.set_ylabel('Fly-frames')
