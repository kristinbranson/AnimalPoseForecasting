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

thresh_stopped = 5
thresh_walking = 15


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
ax[0].plot(rawdata['X'][:,0,frame,flyidx], rawdata['X'][:,1,frame,flyidx], 'k.')
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
sigma = 2 # in frames

# compute the velocity vector
# X is # (nkpts, 2, nframes, nflies)
vel = rawdata['X'][centeridx,:,1:,:] - rawdata['X'][centeridx,:,:-1,:]  # (2, nframes-1, nflies)
vel[:,rawdata['isstart'][1:,:]] = np.nan  # set velocity between end frames and start frames to nan
# make a gaussian filter to smooth vel
from scipy.ndimage import gaussian_filter1d
if sigma > 0:
    vel_fil = gaussian_filter1d(vel, sigma=sigma, axis=1)
else:
    vel_fil = vel

vel_mmps = vel_fil * fps  # convert to mm/s

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
ax[2].plot(velmag_mmps[:,flyidx], '-', label='|V|')
ax[2].axhline(thresh_stopped, color='r', linestyle='--', label='stopped thresh')
ax[2].axhline(thresh_walking, color='g', linestyle='--', label='walking thresh')
ax[2].legend()
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
ylim = ax.get_ylim()
ax.axvline(thresh_stopped, color='r', linestyle='--', label='stopped')
ax.axvline(thresh_walking, color='g', linestyle='--', label='walking')
ax.legend()

# %%
# find time points when the fly is stopped for at least tstopped seconds and transitions to walking
# velmag is # (nframes-1, nflies)
tstopped = .5 # seconds
tfuture = 1 # seconds
tstopped_frames = int(tstopped * fps)
tfuture_frames = int(tfuture * fps)
filstopped = np.ones((tstopped_frames,1),dtype=int)
filfuture = np.ones((tfuture_frames,1),dtype=int)
nflies = velmag_mmps.shape[1]
# nflies = 1

from scipy.ndimage import convolve

def same2valid(x,filsize,debug=False):
    padright = filsize // 2
    padleft = filsize - padright - 1
    if debug:
        print(f'padleft = {padleft}, padright = {padright}, x[padleft-1] = {x[padleft-1]}, x[padleft] = {x[padleft]}, x[-padright-1] = {x[-padright-1]}, x[-padright] = {x[-padright]}')
        assert padleft == 0 or np.all(np.isnan(x[:padleft]))
        assert padright == 0 or np.all(np.isnan(x[-padright:]))
        assert not np.any(np.isnan(x[padleft:-padright]))
    return x[padleft:-padright]

# isstop = np.zeros((1000,nflies))
# isstop[:500] = 1
isstop = velmag_mmps <= thresh_stopped
wasstopped = convolve(isstop.astype(int), filstopped, mode='constant', cval=0) >= tstopped_frames
wasstopped = same2valid(wasstopped, tstopped_frames)
wasstopped = np.r_[np.zeros((tstopped_frames-1, nflies), dtype=bool),wasstopped]

# iswalk = np.zeros((1000,nflies))
# iswalk[500-1+tfuture_frames:] = 1
iswalk = velmag_mmps >= thresh_walking
willwalk = convolve(iswalk.astype(int), filfuture, mode='constant', cval=0) >= 1
willwalk = same2valid(willwalk, tfuture_frames)
willwalk = np.r_[willwalk[1:],np.zeros((tfuture_frames, nflies), dtype=bool)]

willstop = convolve(isstop.astype(int), filfuture, mode='constant', cval=0) >= tfuture_frames
willstop = same2valid(willstop, tfuture_frames)
willstop = np.r_[willstop[1:],np.zeros((tfuture_frames, nflies), dtype=bool)]

startsmoving = wasstopped & willwalk
staysstopped = wasstopped & willstop

# tstartsmoving = np.nonzero(startsmoving)[0]
# print(f'len(tstartsmoving) = {len(tstartsmoving)}')
# for t in tstartsmoving:
#     print(f't = {t}')
#     for off in [-1,0,1]:
#         print(f't = {t+off}, isstop[{t+off}] = {isstop[t+off,0]}, iswalk[{t+off}] = {iswalk[t+off]}, all(isstop[{t+off-tstopped_frames+1}:{t+off+1}]) = {np.all(isstop[t+off-tstopped_frames+1:t+off+1])}, any(iswalk[{t+off+1}:{t+off+tfuture_frames+1}]) = {np.any(iswalk[t+off+1:t+off+tfuture_frames+1,0])}')

# %%
# plot results

fig,ax = plt.subplots(2,1, figsize=(30,10),sharex=True)
tplot = 50000

ax[0].plot(velmag_mmps[:tplot,flyidx], '-', label='|V|')
ax[0].plot(np.nonzero(startsmoving[:tplot,flyidx])[0], velmag_mmps[:tplot][startsmoving[:tplot,flyidx], flyidx], 'g.', label='starts moving')
ax[0].plot(np.nonzero(staysstopped[:tplot,flyidx])[0], velmag_mmps[:tplot][staysstopped[:tplot,flyidx], flyidx], 'r.', label='stays stopped')
ax[0].axhline(thresh_stopped, color='r', linestyle='--', label='stopped thresh')
ax[0].axhline(thresh_walking, color='g', linestyle='--', label='walking thresh')
ax[0].legend()
ax[0].set_ylabel('|V|')
ax[0].set_ylim([0, maxvelmag])

ax[1].plot(isstop[:tplot,flyidx],'-',label='isstop')
ax[1].plot(1.1+iswalk[:tplot,flyidx].astype(float),'-',label='iswalk')
ax[1].plot(2.2+wasstopped[:tplot,flyidx].astype(float),'-',label='wasstopped')
ax[1].plot(3.3+willwalk[:tplot,flyidx].astype(float),'-',label='willwalk')
ax[1].plot(4.4+willstop[:tplot,flyidx].astype(float),'-',label='willstop')
ax[1].plot(5.5+startsmoving[:tplot,flyidx].astype(float),'-',label='startsmoving')
ax[1].plot(6.6+staysstopped[:tplot,flyidx].astype(float),'-',label='staysstopped')
ax[1].set_yticks(np.arange(7)*1.1+.5)
ax[1].set_yticklabels(['isstop','iswalk','wasstopped','willwalk','willstop','startsmoving','staysstopped'])


ax[1].set_xlabel('Frame')
#ax[1].set_xlim(14000,16000)


