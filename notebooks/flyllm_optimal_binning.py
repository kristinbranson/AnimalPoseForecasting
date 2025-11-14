# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: transformer
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
# #%matplotlib widget
# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os

import apf
import apf.utils as utils

import flyllm
from flyllm import optimal_binning

from flyllm.prepare import init_config
from experiments.flyllm import make_dataset

import logging
import tqdm

# format logging prompt to be {time}, {module}:{line number}: {message}
logging.basicConfig(
    format='%(levelname)s:%(name)s:%(lineno)d: %(message)s',
    level=logging.INFO,
    force=True
)
LOG = logging.getLogger(__name__)

utils.set_mpl_backend('tkAgg')
ISNOTEBOOK = utils.is_notebook()

LOG.info('CUDA available: ' + str(torch.cuda.is_available()))
LOG.info('isnotebook: ' + str(ISNOTEBOOK))


# %%
configfile = 'configs/config_fly_llm_predvel_optimalbinning_20251113.json'
outfigdir = 'figs'
debug_uselessdata = False

# path to config file based on code directory
flyllmdir = flyllm.__path__[0]
configfile = os.path.join(flyllmdir,configfile)
outfigdir = os.path.join(flyllmdir,outfigdir)
assert os.path.exists(configfile), f'Config file {configfile} does not exist!'

# make directory if it doesn't exist
if not os.path.exists(outfigdir):
    os.makedirs(outfigdir)

# %%
timestamp = time.strftime('%Y%m%dT%H%M%S')
LOG.info(f'Timestamp for this run: {timestamp}')

# %%
# profile binning optimizer
if False:
    import cProfile
    import pstats

    def profile_test():
        optimal_binning.main(['optimal_binning.py','test'])
        
    cProfile.run("profile_test()",'optimal_binning_profile.out')

    p = pstats.Stats('optimal_binning_profile.out')
    p.sort_stats('cumtime').print_stats(100)

# %%
overrideconfig = {}
# use all data
overrideconfig['categories'] = 'courtship'
config = init_config(configfile=configfile,overrideconfig=overrideconfig)['config']
# don't actually need most of this stuff... 
train_dataset, flyids, track, pose, velocity, sensory, dataset_params, isdata, isstart, useoutputmask = \
    make_dataset(config,'intrainfile',return_all=True,debug=debug_uselessdata)

# %%
from apf.io import get_modeltype_str
modeltype_str = get_modeltype_str(config)
savefilestr = f"flyllm_binning_{modeltype_str}_{timestamp}"
print(f'Save file string: {savefilestr}')

# %%
optimal_binning.debug_synthetic()
#optimal_binning.debug_zscored_velocity()

# %%
# plot percentiles of the data to be discretized
from flyllm.config import posenames, featangle
from matplotlib.ticker import SymmetricalLogLocator

nfeat = len(config['discreteidx'])

prcts = np.logspace(-5,2,100)
prcts = np.vstack((100-prcts,prcts)).T
POS = 1
NEG = 0

prctvals = np.zeros((len(config['discreteidx']),2,len(prcts))) + np.nan
for idx,featidx in enumerate(config['discreteidx']):
    feature_data = velocity.array[...,featidx].copy()
    feature_data = feature_data[(isdata&useoutputmask).T&~np.isnan(feature_data)]
    if featangle[featidx]:
        feature_data *= 180/np.pi
    prctvals[idx,POS] = np.percentile(feature_data[feature_data>=0],prcts[:,0])
    prctvals[idx,NEG] = -np.percentile(feature_data[feature_data<=0],prcts[:,1])

minprctval = 1e-8
fig,ax = plt.subplots(nfeat,2,figsize=(6*2,6*nfeat),sharex=False,squeeze=False)
for idx,featidx in enumerate(config['discreteidx']):
    ax[idx,POS].plot(np.maximum(100-prcts[:,0],minprctval),prctvals[idx,0],'.')
    ax[idx,POS].set_ylabel(f'{idx}: +{posenames[featidx]}')
    ax[idx,POS].invert_xaxis()
    ax[idx,NEG].plot(np.maximum(prcts[:,1],minprctval),prctvals[idx,1],'.')
    ax[idx,NEG].set_ylabel(f'{idx}: -{posenames[featidx]}')

    #ax[idx].set_xscale('log')
    for i in range(2):
        ax[idx,i].set_yscale('log')
        ax[idx,i].set_xscale('log')
        ax[idx,i].yaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=1, subs=range(1,10)))
        plt.minorticks_on()
        ax[idx,i].grid(True,which='both',ls='--')
ax[-1,POS].set_xlabel('100-Percentile')
ax[-1,NEG].set_xlabel('Percentile')
fig.tight_layout()


# %%
config = init_config(configfile=configfile,overrideconfig=overrideconfig)['config']
K = config['discretize_nbins']

allbininfo = []

for idx,featidx in enumerate(config['discreteidx']):

    LOG.info(f'Optimizing bins for feature {featidx}: {posenames[featidx]} ({idx+1} of {len(config["discreteidx"])})')

    # data for current feature
    feature_data = velocity.array[...,featidx].copy()
    feature_data[useoutputmask.T==0] = np.nan  # mask out invalid data

    # split and subsample
    sequences = optimal_binning.split_sequences(feature_data)
    
    optimize_bins_params = config.get('optimize_bins',{}).copy()
    nsamples = optimize_bins_params.pop('nsamples',None)
    sequences = optimal_binning.subsample_sequences(sequences, nsamples=nsamples)

    all_outlier_thresh = optimize_bins_params.pop('all_outlier_thresh',None)
    prctile_outlier_thresh = optimize_bins_params.pop('prctile_outlier_thresh',None)
    if all_outlier_thresh is None:
        if prctile_outlier_thresh is not None:
            outlier_thresh = np.percentile(feature_data[~np.isnan(feature_data)],[prctile_outlier_thresh,100-prctile_outlier_thresh])
    else:
        outlier_thresh = all_outlier_thresh[featidx]
        
    all_discretize_epsilon = config.get('all_discretize_epsilon',None)
    print('all_discretize_epsilon:',all_discretize_epsilon)
    min_width = all_discretize_epsilon[featidx] if all_discretize_epsilon is not None else 0.0
    print('min_width:',min_width)

    edges_fixed, boundary_edges = optimal_binning.set_boundary_edges(sequences,K,outlier_thresh)
    print(edges_fixed)

    edges, prob_matrix, log_lik = optimal_binning.optimize_bin_edges(
        sequences,
        K=K,
        boundary_edges=boundary_edges,
        min_width=min_width,
        edges_fixed=edges_fixed,
        **optimize_bins_params
    )
    
    allbininfo.append({'edges': edges, 'prob_matrix': prob_matrix, 'log_lik': log_lik})

# %%
for idx,featidx in enumerate(config['discreteidx']):
    edges = allbininfo[idx]['edges']
    prob_matrix = allbininfo[idx]['prob_matrix']
    log_lik = allbininfo[idx]['log_lik']
    
    widths = np.diff(edges)
    bin_centers = (edges[:-1]+edges[1:])/2

    if featangle[featidx]:
        edges = edges * 180/np.pi
        widths = widths * 180/np.pi
        bin_centers = bin_centers * 180/np.pi
        units = 'deg/fr'
    else:
        units = 'mm/fr'

    fig,ax = plt.subplots(4,1,figsize=(10,20))
    ax[0].plot(widths,'.')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Bin index')
    ax[0].set_ylabel(f'Bin width ({units})')

    ax[1].plot(bin_centers,widths,'.')
    ax[1].set_yscale('log')
    ax[1].set_xscale('symlog')
    ax[1].set_xlabel(f'Bin center ({units})')
    ax[1].set_ylabel(f'Bin width ({units})')

    ax[2].plot(edges,'.')
    ax[2].set_xlabel('Bin index')
    ax[2].set_ylabel(f'Bin edge ({units})')
    ax[2].set_yscale('symlog')

    ax[3].imshow(prob_matrix)
    ax[3].set_xlabel('To bin index')
    ax[3].set_ylabel('From bin index')
    ax[0].set_title(f'Feature {featidx}: {posenames[featidx]} velocity')

    plt.tight_layout()

    fig.savefig(os.path.join(outfigdir,f'{savefilestr}_{featidx}_{posenames[featidx]}_K{K}.png'))
    fig.savefig(os.path.join(outfigdir,f'{savefilestr}_{featidx}_{posenames[featidx]}_K{K}.pdf'))


# %%
# save results to file
savefile = os.path.join(config['savedir'],f'{savefilestr}_bininfo.npz')
LOG.info(f'Saving binning info to {savefile}')
for i,featidx in enumerate(config['discreteidx']):
    allbininfo[i]['featidx'] = featidx
    allbininfo[i]['posename'] = posenames[featidx]
np.savez_compressed(savefile,bininfo=allbininfo)


# %%
# print bin edges so that they can be copy-pasted into config files
for i,featidx in enumerate(config['discreteidx']):
    edges = allbininfo[i]['edges']
    print(f'{featidx}: {np.array2string(edges,separator=', ')}')
