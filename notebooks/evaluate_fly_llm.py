# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
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
# ### Imports

# %%
# %load_ext autoreload
# %autoreload 2
import linecache
linecache.clearcache()

import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pickle
import datetime

import apf
from apf.training import train
import apf.utils as utils
from apf.simulation import simulate
from apf.models import initialize_model

import flyllm
from flyllm.config import read_config
from flyllm.features import featglobal, get_sensory_feature_idx, compute_pose_distribution_stats        

from flyllm.simulation import animate_pose
import time
import os
from pathlib import Path

from flyllm.prepare import init_flyllm

import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

utils.set_mpl_backend('tkAgg')
ISNOTEBOOK = utils.is_notebook()
if ISNOTEBOOK:
    from IPython.display import HTML, display, clear_output
else:
    plt.ion()

LOG.info('CUDA available: ' + str(torch.cuda.is_available()))
LOG.info('isnotebook: ' + str(ISNOTEBOOK))

# %%
timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
print('Timestamp: ' + timestamp)

# %% [markdown]
# ### Configuration and load data

# %%
# configuration parameters for this model

rootdir = Path(utils.get_code_root())

loadmodelfile = rootdir / 'notebooks/flyllm_models/flypredvel_20251007_20251114T194024_bestepoch200.pth'
configfile = rootdir / 'flyllm/configs/config_fly_llm_predvel_optimalbinning_20251113.json'

# print current directory
outfigdir = rootdir / 'notebooks/figs'
debug_uselessdata = True

# make directory if it doesn't exist
if not outfigdir.exists():
    outfigdir.mkdir(parents=True)

needtraindata = True
needvaldata = True

res = init_flyllm(configfile=configfile,mode='test',loadmodelfile=loadmodelfile,
                debug_uselessdata=debug_uselessdata,
                needtraindata=needtraindata,needvaldata=needvaldata)



# %%
# unpack res
config = res['config']
device = res['device']
train_data = res['train_data']
val_data = res['val_data']
train_dataset = res['train_dataset']
train_dataloader = res['train_dataloader']
val_dataset = res['val_dataset']
val_dataloader = res['val_dataloader']
dataset_params = res['dataset_params']
model = res['model']
criterion = res['criterion']
opt_model = res['opt_model']
modeltype_str = res['modeltype_str']
savetime = res['model_savetime']

pred_stride = 20

# where to save predictions
savepredfile = loadmodelfile.parent / f'{loadmodelfile.stem}_pred{pred_stride}.npz'

# where to save pose statistics
posestatsfile = loadmodelfile.parent / f'{loadmodelfile.stem}_posestats.npz'

# %%
# load/compute pose statistics
if posestatsfile is not None:
    if posestatsfile.exists():
        print("Loading pose stats from ", posestatsfile)
        posestats = np.load(posestatsfile)
    else:
        posestats = compute_pose_distribution_stats(train_data['pose'])
        np.savez(posestatsfile, **posestats)

# %%
# profile stuff 

import cProfile
import pstats

from flyllm.prediction import predict_all

doprofile = False
dodebug = False
from flyllm.dataset import FlyTestDataset
# val_dataset.clear_cuda_cache()
# val_dataset.ncudacaches = 0
# val_dataset.cudaoptimize = False

# def profile_dataset_creation():
#     val_dataset_small = FlyTestDataset(valX[:min(len(valX),100)],config['contextl'],**dataset_params)

def profile_iterating_dataset():

    for i, batch in enumerate(val_dataloader):
        if i > 10:
            break
        
def profile_predict_all():
    return predict_all(dataset=val_dataset, model=opt_model, config=config, keepall=False, earlystop=10, nkeep=50)

if doprofile:

    # get start time
    t0 = datetime.datetime.now()
    #cProfile.run('profile_predict_all()','profile_test.out')
    out = profile_predict_all()
    t1 = datetime.datetime.now()
    print('Elapsed time: ', t1-t0)

    p = pstats.Stats('profile_test.out')
    print('sorted by time')
    stats = p.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(20)
    print('sorted by cum')
    stats = p.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(20)
    

if (not doprofile) and dodebug:
    profile_predict_all()

# %%
# clean up memory allocation before training, particularly if running in a notebook
# and things have crashed before...

import gc

model = model.to(device='cpu')
model.zero_grad()
torch.cuda.empty_cache()
gc.collect()

utils.torch_mem_report(model)

model = model.to(device=device)

memalloc = torch.cuda.memory_allocated() / 1e9
print(f'Cuda memory allocated for model: {memalloc:.3f} GB')
memreserved = torch.cuda.memory_reserved() / 1e9
print(f'Cuda memory reserved for model: {memreserved:.3f} GB')

# %% [markdown]
# ### Evaluate single-iteration predictions

# %%
debugcheat = False
forcecompute = False
from flyllm.prediction import predict_all

if debugcheat or forcecompute or (savepredfile is None) or (not savepredfile.exists()):

    tmpsavepredfile = savepredfile.with_name(savepredfile.stem + '_tmp.npz')
    all_pred, metadata = predict_all(dataset=val_dataset, model=opt_model, config=config, keepall=False, debugcheat=debugcheat, 
                                    savepredfile=tmpsavepredfile, saveinterval=600, stride=pred_stride)

    # save all_pred and metadata to a numpy file
    if (not debugcheat) and (savepredfile is not None):
        print(f'Saving predictions to {savepredfile}')
        np.savez(savepredfile,all_pred=all_pred,metadata=metadata)
else:
    # load all_pred and labelidx from savepredfile
    print(f'Loading predictions from {savepredfile}')
    tmp = np.load(savepredfile,allow_pickle=True)
    all_pred = tmp['all_pred'].item()
    metadata = tmp['metadata'].item()

# %%
# convert to data objects, match with val_dataset labels
pred_data = val_dataset.item_to_data(all_pred,subindex=metadata)

# sanity check -- plot NaN masks for predicted and ground truth labels
fig,ax = plt.subplots(2,1,figsize=(10,6),sharex=True,sharey=True)
ax[0].imshow(np.all(np.isnan(pred_data['labels']['velocity'].array),axis=-1),aspect='auto',cmap='gray',vmin=0,vmax=1,interpolation='none')
ax[0].set_title('Predicted NaN mask')
ax[1].imshow(np.all(np.isnan(val_dataset.labels['velocity'].array),axis=-1),aspect='auto',cmap='gray',vmin=0,vmax=1,interpolation='none')
ax[1].set_title('Ground truth NaN mask')
plt.colorbar(ax[1].images[0], ax=ax, orientation='vertical')

for key in pred_data['labels']:
    print(f'Predicted {key} = {pred_data["labels"][key]}')
    pred_data['labels'][key].print_feature_names()
for key in val_dataset.labels:
    print(f'Label {key} = {val_dataset.labels[key]}')
    val_dataset.labels[key].print_feature_names()
for key in val_dataset.inputs:
    print(f'Input {key} = {val_dataset.inputs[key]}')
    val_dataset.inputs[key].print_feature_names()

# %%
import importlib
import flyllm.evaluation
import apf.dataset
importlib.reload(flyllm.evaluation)  # reload the evaluation
importlib.reload(apf.dataset)
import linecache
linecache.clearcache()
from flyllm.evaluation import compute_error
next_frame_err = compute_error(val_dataset,val_data,pred_data)
print(next_frame_err.keys())

feature_names = next_frame_err['velocity_feature_names']
for key in ['velocity_absdiff_samplemean_datamean', 'velocity_absdiff_samplemin_datamean']:
    assert(len(feature_names) == next_frame_err[key].shape[0]), f'Feature names length {len(feature_names)} does not match next_frame_err shape {next_frame_err[key].shape[0]}'
    print(f'Next frame error for {key}:')
    for i in range(next_frame_err[key].shape[0]):
        print(f'dim {i} ({feature_names[i]}): {next_frame_err[key][i]}')
        
key = 'discrete_velocity_ce_datamean'
feature_names = next_frame_err['discrete_velocity_feature_names']
assert (len(feature_names) == next_frame_err[key].shape[0]), f'Feature names length {len(feature_names)} does not match next_frame_err shape {next_frame_err[key].shape[0]}'
print(f'Next frame error for {key}:')
for i in range(next_frame_err[key].shape[0]):
    print(f'dim {i} ({feature_names[i]}): {next_frame_err[key][i]}')

key = 'continuous_zvelocity_absdiff_datamean'
feature_names = next_frame_err['continuous_zvelocity_feature_names']
assert (len(feature_names) == next_frame_err[key].shape[0]), f'Feature names length {len(feature_names)} does not match next_frame_err shape {next_frame_err[key].shape[0]}'
print(f'Next frame error for {key}:')
for i in range(next_frame_err[key].shape[0]):
    print(f'dim {i} ({feature_names[i]}): {next_frame_err[key][i]}')

# %%
# DEBUG : check inversion operations

import importlib
import apf.dataset
importlib.reload(apf.dataset)
from apf.dataset import invert_to_named, apply_inverse_operations

def check_inversion(datao,datar,isdata=None):
    for id in range(datao.array.shape[0]):
        arr_o = datao.array[id]
        arr_r = datar.array[id]
        if isdata is not None:
            mask = isdata[id]
            arr_o = arr_o[mask]
            arr_r = arr_r[mask]
        if np.all(np.isnan(arr_o)) and np.all(np.isnan(arr_r)):
            print(f'All NaN for id {id}')
            continue
        if np.allclose(arr_o,arr_r,atol=1e-6,equal_nan=True):
            print(f'All close for id {id}')
        else:
            diff = np.abs(arr_o - arr_r)
            maxdiff = np.nanmax(diff)
            meandiff = np.nanmean(diff)
            print(f'Id {id}: meandiff = {meandiff}, maxdiff = {maxdiff}')
            isnanmismatch = np.isnan(arr_o) != np.isnan(arr_r)
            if np.any(isnanmismatch):
                print(f'NaN mismatch for id {id} for {np.count_nonzero(isnanmismatch)} elements')
    if len(datar.feature_names) != len(datao.feature_names):
        print(f'Feature name length mismatch: original {len(datao.feature_names)}, reconstructed {len(datar.feature_names)}')
    for i,featname in enumerate(datao.feature_names):
        if i >= len(datar.feature_names):
            break
        name_r = datar.feature_names[i]
        if featname != name_r:
            print(f'Feature name mismatch at index {i}: original "{featname}", reconstructed "{name_r}"')

print(val_data.keys())
print('track: ' + str(val_data['track'].feature_names))
print('pose: ' + str(val_data['pose'].feature_names))
print('velocity: ' + str(val_data['velocity'].feature_names))
print('sensory: ' + str(val_data['sensory'].feature_names))

track1 = invert_to_named(val_data['pose'],'original',return_data=True)
print('Checking inversion from pose to track:')
check_inversion(val_data['track'],track1,val_data['isdata'].T)
    
pose1 = invert_to_named(val_data['velocity'],'pose',return_data=True)
print('Checking inversion from velocity to pose:')
check_inversion(val_data['pose'],pose1)



# %%
# DEBUG: check inversion of global velocity

import importlib
import apf.dataset
importlib.reload(apf.dataset)  # reload the evaluation
import linecache
linecache.clearcache()

from flyllm.config import posenames, featglobal
print('Pose names: ', posenames)

from apf.dataset import Velocity, GlobalVelocity, Subset, invert_to_named
from experiments.flyllm import featrelative, featangle
#velocity = Velocity(featrelative=featrelative, featangle=featangle)(val_data['pose'], isstart=val_data['isstart'])
pose_global = Subset(include_ids=featglobal)(val_data['pose'])
global_velocity = GlobalVelocity(tspred=[2,5,10])(pose_global,isstart=val_data['isstart'])
global_velocity.print_feature_names()

pose_global1 = invert_to_named(global_velocity,'subset')
print(pose_global1.shape)
print(np.allclose(pose_global.array,pose_global1,equal_nan=True))
for id in range(pose_global.array.shape[0]):
    if np.all(np.isnan(pose_global.array[id,:])) and np.all(np.isnan(pose_global1[id,:])):
        print(f'Pose id {id}: all NaNs')
        continue
    elif np.allclose(pose_global.array[id,:],pose_global1[id,:],equal_nan=True):
        print(f'Pose id {id}: all close')
        continue
    print(f'Pose id {id}: max abs diff = {np.nanmax(np.abs(pose_global.array[id,:] - pose_global1[id,:]))}')
id = 2
tmismatch,fmismatch = np.nonzero(~np.isclose(pose_global.array[id,:],pose_global1[id,:],equal_nan=True))
if len(tmismatch) > 0:
    print(f'pose id {id} first mismatch at time {tmismatch[0]} feature {fmismatch[0]}')
    print(pose_global.array[id,tmismatch[0]-1:tmismatch[0]+2,fmismatch[0]])
    print(pose_global1[id,tmismatch[0]-1:tmismatch[0]+2,fmismatch[0]])

# pose id 2 first mismatch at time 10000 feature 0
# [-7.53946829 -0.07114847 -0.2117333 ]
# [-7.53946829         nan         nan]

# %%
val_dataset.chunk_indices
val_data['flyids']
print(pred_data['labels']['velocity'].shape)
print(pred_data['labels']['velocity'].invertdata)
np.unique(val_data['flyids'][val_data['isdata']])
(val_data['flyids']==11) & (val_data['isdata'])

# %%
# plot multi errors
import importlib
import flyllm.plotting
import apf.dataset
import flyllm.evaluation
importlib.reload(flyllm.plotting)  # reload the evaluation
importlib.reload(apf.dataset)
importlib.reload(flyllm.evaluation)
import linecache
linecache.clearcache()
from flyllm.plotting import plot_pred_vs_true
from apf.dataset import copy_data_subindex
from flyllm.evaluation import labels_to_velocity_samples

savefig = False
featsplot = None
featsplot=[0,1,2,14, 16, 20, 22, 24, 26]
#toplots = [{'id': 3, 'tsplot': np.arange(8500,8800)}, {'id': 435, 'tsplot': np.arange(32400,32700)}, ]
toplots = [{'id': 11, 'tsplot': np.arange(8500,8800)}, {'id': 13, 'tsplot': np.arange(500,10000)}, ]
nsamples = 10
ylim_nstd = 5

for toplot in toplots:
    id = toplot['id']
    tsplot = toplot['tsplot']
    
    tidx,ididx = np.nonzero((val_data['flyids'] == id) & val_data['isdata'])
    assert len(ididx) > 0, f'Fly id {id} not in validation data flyids'
    assert np.all(ididx==ididx[0]), 'Multiple indices for fly id'
    ididx = ididx[0]
    assert len(tidx) > np.max(tsplot), f'Not enough timepoints for fly id {id} to cover tsplot'
    tidx = tidx[tsplot]
    
    pred_example = copy_data_subindex(pred_data['labels']['velocity'], agentidx=[ididx,], frameidx=tidx)
    print(pred_example)
    true_example = copy_data_subindex(val_data['velocity'], agentidx=[ididx,], frameidx=tidx)
    
    fig = plot_pred_vs_true(true_example,pred_example,ylim_nstd=ylim_nstd,nsamples=nsamples,plotbinedges=True)

    # # save this figure as a pdf
    # if savefig:
    #     fig.savefig(os.path.join(outfigdir,f'multi_pred_vs_true_{config["model_nickname"]}_fly{id}_{tsplot[0]}_to_{tsplot[-1]}.pdf'))
    #     plt.close(fig)
    # else:
    #     break

# %%
np.unique(val_data['flyids'])

# %%
bin_edges = true_example.labels.get_discretize_params(zscored=True)['bin_edges']
featdiscretei = 0
nbins = bin_edges.shape[1]
np.stack((bin_edges[featdiscretei][:-1],bin_edges[featdiscretei][1:],np.nan+np.zeros(nbins-1)),axis=1).flatten()


# %%
# store the random state so we can reproduce the same results
randstate_np = np.random.get_state()
randstate_torch = torch.random.get_rng_state()

# %%
# reseed numpy random number generator with randstate_np
np.random.set_state(randstate_np)
# reseed torch random number generator with randstate_torch
torch.random.set_rng_state(randstate_torch)

# %%
# choose data to initialive behavior modeling

animate_pose_params = {'figsizebase': 8,'ms': 4, 'focus_ms':8, 'lw': .75, 'focus_lw': 2}

tpred = np.minimum(4000 + config['contextl'], valdata['isdata'].shape[0] // 2)

# all frames must have real data
burnin = config['contextl'] - 1
contextlpad = burnin + val_dataset.ntspred_max
allisdata = interval_all(valdata['isdata'], contextlpad)
isnotsplit = interval_all(valdata['isstart'] == False, tpred)[1:, ...]
canstart = np.logical_and(allisdata[:isnotsplit.shape[0], :], isnotsplit)
flynum = 2  # 2
t0 = np.nonzero(canstart[:, flynum])[0]
idxstart = np.minimum(8700, len(t0) - 1)
if len(t0) > idxstart:
    t0 = t0[idxstart]
else:
    t0 = t0[0]
fliespred = np.array([flynum, ])

nsamplesfuture = 0 # 32

isreal = np.any(np.isnan(valdata['X'][...,t0:t0+tpred,:]),axis=(0,1,2)) == False
Xkp_init = valdata['X'][...,t0:t0+tpred+val_dataset.ntspred_max,isreal]
scales_pred = []
for flypred in fliespred:
  id = valdata['ids'][t0, flypred]
  scales_pred.append(val_scale_perfly[:,id])
metadata = {'t0': t0, 'ids': fliespred, 'videoidx': valdata['videoidx'][t0], 'frame0': valdata['frames'][t0]}
print(metadata)

ani = animate_predict_open_loop(model, val_dataset, Xkp_init, fliespred, scales_pred, tpred, 
                          debug=False, plotattnweights=False, plotfuture=val_dataset.ntspred_max > 1, 
                          nsamplesfuture=nsamplesfuture, metadata=metadata,
                          animate_pose_params=animate_pose_params)

# %%
# show the animation
HTML(ani.to_html5_video())

# %%
# write the video to file 
vidtime = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
flynumstr = '_'.join([str(x) for x in fliespred])
savevidfile = os.path.join(config['savedir'], f"samplevideo_{modeltype_str}_{savetime}_{vidtime}_t0_{metadata['t0']}_flies{flynumstr}.mp4")
print('Saving animation to file %s...'%savevidfile)
save_animation(ani, savevidfile)
print('Finished writing.')

# %%
# simulate all flies

animate_pose_params = {'figsizebase': 8,'ms': 4, 'focus_ms':8, 'lw': .75, 'focus_lw': 2}

# choose data to initialive behavior modeling
tpred = np.minimum(4000 + config['contextl'], valdata['isdata'].shape[0] // 2)

# all frames must have real data
burnin = config['contextl'] - 1
contextlpad = burnin + val_dataset.ntspred_max
allisdata = interval_all(valdata['isdata'], contextlpad)
isnotsplit = interval_all(valdata['isstart'] == False, tpred)[1:, ...]
canstart = np.logical_and(allisdata[:isnotsplit.shape[0], :], isnotsplit)
flynum = 2  # 2
t0 = np.nonzero(canstart[:, flynum])[0]
idxstart = np.minimum(8700, len(t0) - 1)

if len(t0) > idxstart:
    t0 = t0[idxstart]
else:
    t0 = t0[0]
isreal = np.any(np.isnan(valdata['X'][...,t0:t0+tpred,:]),axis=(0,1,2)) == False
print(isreal)
fliespred = np.nonzero(isreal)[0]

nsamplesfuture = 0 # 32

isreal = np.any(np.isnan(valdata['X'][...,t0:t0+tpred,:]),axis=(0,1,2)) == False
Xkp_init = valdata['X'][...,t0:t0+tpred+val_dataset.ntspred_max,isreal]
scales_pred = []
for flypred in fliespred:
  id = valdata['ids'][t0, flypred]
  scales_pred.append(val_scale_perfly[:,id])
metadata = {'t0': t0, 'ids': fliespred, 'videoidx': valdata['videoidx'][t0], 'frame0': valdata['frames'][t0]}
print(metadata)

ani = animate_predict_open_loop(model, val_dataset, Xkp_init, fliespred, scales_pred, tpred, 
                          debug=False, plotattnweights=False, plotfuture=val_dataset.ntspred_max > 1, 
                          nsamplesfuture=nsamplesfuture, metadata=metadata,
                          animate_pose_params=animate_pose_params)

# %%
vidtime = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
flynumstr = 'all'
savevidfile = os.path.join(config['savedir'], f"samplevideo_{modeltype_str}_{savetime}_{vidtime}_t0_{metadata['t0']}_flies{flynumstr}.mp4")
print('Saving animation to file %s...'%savevidfile)
save_animation(ani, savevidfile)
print('Finished writing.')

# %%
# create dataset for iterative prediction evaluation
max_tpred = 150
iter_val_dataset = FlyTestDataset(valX,config['contextl']+max_tpred+1,**dataset_params,need_labels=True,need_metadata=True,need_init=True,make_copy=True)

# %%
# all_pred_iter is a list of length N of FlyExample objects
# labelidx_iter is an ndarray of which test example each FlyExample corresponds to
all_pred_iter, labelidx_iter = predict_iterative_all(valdata,iter_val_dataset,model,max_tpred,N=10,keepall=False,nsamples=10)

# %%
# plot multi predictions vs true
savefig = True

toff0 = 50
maxsamplesplot = 10
for i in range(len(all_pred_iter)):
    pred_example = all_pred_iter[i]
    examplei = labelidx_iter[i]
    true_example_dict = iter_val_dataset[examplei]
    true_example_obj = FlyExample(example_in=true_example_dict,dataset=iter_val_dataset)
    tsplot = config['contextl']+np.arange(-toff0,max_tpred+1)
    fig = plot_multi_pred_iter_vs_true(pred_example,true_example_obj,ylim_nstd=5,tsplot=tsplot,maxsamplesplot=maxsamplesplot)
    metadata = true_example_obj.metadata

    axs = fig.get_axes()
    for ax in axs:
        ylimcurr = ax.get_ylim()
        ax.plot([toff0+1,toff0+1],ylimcurr,'k--')

    if savefig:
        fig.savefig(os.path.join(outfigdir,f'multi_pred_iter_vs_true_{config["model_nickname"]}_t0_{metadata["t0"]}_fly_{metadata["flynum"]}.pdf'))
        plt.close(fig)
    else:
        break


# %%
# plot pose predictions vs true

savefig = True
for i in range(len(all_pred_iter)):
    pred_example = all_pred_iter[i]
    examplei = labelidx_iter[i]
    true_example_dict = iter_val_dataset[examplei]
    true_example_obj = FlyExample(example_in=true_example_dict,dataset=iter_val_dataset)
    toff0 = 50
    tsplot = config['contextl']+np.arange(-toff0,max_tpred+1)
    fig = plot_pose_pred_iter_vs_true(pred_example,true_example_obj,tsplot=tsplot,maxsamplesplot=10)
    metadata = true_example_obj.metadata

    axs = fig.get_axes()
    for ax in axs:
        ylim_curr = ax.get_ylim()
        ax.plot([toff0+2,toff0+2],ylim_curr,'k--')

    fig.savefig(os.path.join(outfigdir,f'pose_pred_iter_vs_true_{config["model_nickname"]}_t0_{metadata["t0"]}_fly_{metadata["flynum"]}.pdf'))

# %%
# compute error in various ways
# change this to computing error in pose
err_example_iter = []
for i,examplei in enumerate(labelidx_iter):
    pred_exampleobj = all_pred_iter[i]
    true_example_dict = iter_val_dataset[examplei]
    true_example_obj = FlyExample(example_in=true_example,dataset=iter_val_dataset)
    errcurr = true_example_obj.labels.compute_error_iter(pred_exampleobj.labels)
    err_example_iter.append(errcurr)

# combine errors
err_total_iter = FlyExample.combine_errors(err_example_iter)

for k,v in err_total_iter.items():
    if type(v) is np.ndarray:
        print(f'{k}: ndarray {v.shape}')
    else:
        print(f'{k}: {type(v)}')

# %%
# plot multi errors

d_multi = len(err_total_iter['multi_names'])
fig,ax = plt.subplots(d_multi,1,sharex=True,figsize=(16,2*d_multi))

for featnum in range(d_multi):
    miny = 0
    maxy = np.max(err_total_iter['l1_multi'][burnin:,featnum])
    dy = maxy-miny
    ylim = (miny-.1*dy,maxy+.1*dy)
    ax[featnum].plot(err_total_iter['l1_multi'][burnin:,featnum],'.-',label='l1')
    ax[featnum].set_ylim(ylim)
    ax[featnum].set_ylabel(err_total_iter['multi_names'][featnum])

fig.tight_layout()

# %%
# # debug setting pose

# examplei = 0
# example = iter_val_dataset[examplei]
# exampleobj = FlyExample(example_in=example,dataset=iter_val_dataset)
# multi0 = exampleobj.labels.get_multi(use_todiscretize=True)
# next0 = exampleobj.labels.get_next(use_todiscretize=True)
# pose0 = exampleobj.labels.get_next_pose(use_todiscretize=True)
# kp0 = exampleobj.labels.get_next_keypoints(use_todiscretize=True)
# multibad0 = np.zeros(multi0.shape)
# exampleobj.labels.set_multi(multibad0,zscored=True)
# multibad1 = exampleobj.labels.get_multi(use_todiscretize=True,zscored=True)
# assert np.allclose(multibad0,multibad1)
# nextbad1 = exampleobj.labels.get_next(use_todiscretize=True)
# posebad1 = exampleobj.labels.get_next_pose(use_todiscretize=True)
# kpbad1 = exampleobj.labels.get_next_keypoints(use_todiscretize=True)
# print(f'max error between next0 and nextbad1: {np.max(np.abs(next0-nextbad1))}')
# print(f'max error between pose0 and posebad1: {np.max(np.abs(pose0-posebad1))}')
# print(f'max error between kp0 and kpbad1: {np.max(np.abs(kp0-kpbad1))}')
# exampleobj.labels.set_next_pose(pose0)
# next1 = exampleobj.labels.get_next(use_todiscretize=True)
# pose1 = exampleobj.labels.get_next_pose(use_todiscretize=True)
# kp1 = exampleobj.labels.get_next_keypoints(use_todiscretize=True)
# print(f'max error between next0 and next1: {np.max(np.abs(next0-next1))}')
# print(f'max error between pose0 and pose1: {np.max(np.abs(pose0-pose1))}')
# print(f'max error between kp0 and kp1: {np.max(np.abs(kp0-kp1))}')
# assert np.allclose(next0,next1,atol=1e-4)
# assert np.allclose(pose0,pose1,atol=1e-4)
# assert np.allclose(kp0,kp1,atol=1e-4)

# ts = [config['contextl'],]
# exampleobj = FlyExample(example_in=example,dataset=iter_val_dataset)
# multi0 = exampleobj.labels.get_multi(use_todiscretize=True,ts=ts)

# poseall0 = exampleobj.labels.get_next_pose(use_todiscretize=True)
# init0 = poseall0[...,ts[0],:]
# pose0 = exampleobj.labels.get_next_pose(use_todiscretize=True,ts=ts,init_pose=init0)
# next0 = exampleobj.labels.get_next(use_todiscretize=True,ts=ts)
# kp0 = exampleobj.labels.get_next_keypoints(use_todiscretize=True,ts=ts,init_pose=init0)
# assert np.allclose(poseall0[[ts[0],ts[0]+1],:],pose0)
# multibad = exampleobj.labels.get_multi(use_todiscretize=True,zscored=True)
# multibad[...,ts,:] = 0
# exampleobj.labels.set_multi(multibad,zscored=True)
# multibad1 = exampleobj.labels.get_multi(use_todiscretize=True,zscored=True)
# assert np.allclose(multibad,multibad1)
# posebad1 = exampleobj.labels.get_next_pose(use_todiscretize=True,ts=ts,init_pose=init0)
# nextbad1 = exampleobj.labels.get_next(use_todiscretize=True,ts=ts)
# kpbad1 = exampleobj.labels.get_next_keypoints(use_todiscretize=True,ts=ts,init_pose=init0)
# print(f'max error between pose0 and posebad1: {np.max(np.abs(pose0-posebad1))}')
# print(f'max error between next0 and nextbad1: {np.max(np.abs(next0-nextbad1))}')
# print(f'max error between kp0 and kpbad1: {np.max(np.abs(kp0-kpbad1))}')
# exampleobj.labels.set_next_pose(pose0,ts=ts)
# pose1 = exampleobj.labels.get_next_pose(use_todiscretize=True,ts=ts,init_pose=init0)
# next1 = exampleobj.labels.get_next(use_todiscretize=True,ts=ts)
# kp1 = exampleobj.labels.get_next_keypoints(use_todiscretize=True,ts=ts,init_pose=init0)
# print(f'max error between pose0 and pose1: {np.max(np.abs(pose0-pose1))}')
# print(f'max error between next0 and next1: {np.max(np.abs(next0-next1))}')
# print(f'max error between kp0 and kp1: {np.max(np.abs(kp0-kp1))}')
# assert np.allclose(pose0,pose1,atol=1e-4)
# assert np.allclose(next0,next1,atol=1e-4)
# assert np.allclose(kp0,kp1,atol=1e-4)


# %%
# reseed numpy random number generator with randstate_np
np.random.set_state(randstate_np)
# reseed torch random number generator with randstate_torch
torch.random.set_rng_state(randstate_torch)

all_pred_iter_reg, labelidx_iter_reg = predict_iterative_all(valdata,iter_val_dataset,model,max_tpred,keepall=False,
                                                             labelidx=labelidx_iter[[0,]],nsamples=2,dampenconstant=1e-2,
                                                             posestats=posestats,prctilelim=1e-3)


# reseed numpy random number generator with randstate_np
np.random.set_state(randstate_np)
# reseed torch random number generator with randstate_torch
torch.random.set_rng_state(randstate_torch)

all_pred_iter_noreg, labelidx_iter_noreg = predict_iterative_all(valdata,iter_val_dataset,model,max_tpred,keepall=False,
                                                             labelidx=labelidx_iter[[0,]],nsamples=2)

# reseed numpy random number generator with randstate_np
np.random.set_state(randstate_np)
# reseed torch random number generator with randstate_torch
torch.random.set_rng_state(randstate_torch)

all_pred_iter_noreg1, labelidx_iter_noreg1 = predict_iterative_all(valdata,iter_val_dataset,model,max_tpred,keepall=False,
                                                             labelidx=labelidx_iter[[0,]],nsamples=2)



# %%
# plot pose

assert np.allclose(all_pred_iter_noreg1[0].labels.labels_raw['todiscretize'],all_pred_iter_noreg[0].labels.labels_raw['todiscretize'])
nr = 3
nc = 3
nplots = nr*nc
tsplot = config['contextl']-1 + np.arange(0,5*nplots,5)
print(tsplot)
kp_noreg = all_pred_iter_noreg[0].labels.get_next_keypoints(use_todiscretize=True)
kp_reg = all_pred_iter_reg[0].labels.get_next_keypoints(use_todiscretize=True)
pose_noreg = all_pred_iter_noreg[0].labels.get_next_pose(use_todiscretize=True)
print(kp_noreg.shape)

t = tsplot[1]
samplei = 0
fig,axs = plt.subplots(nr,nc,figsize=(nc*5,nr*5))
axs = axs.flatten()

from flyllm.plotting import plot_fly
from flyllm.features import feat2kp

for i,t in enumerate(tsplot):
    ax = axs[i]
    meanpose = pose_noreg[samplei,t].copy()
    meanpose[3:] = posestats['meanrelpose']
    meankp = feat2kp(meanpose,all_pred_iter_noreg[0].labels._scale[0])
    meankp = meankp[:,:,0,0]
    print(meankp.shape)
    plot_fly(meankp,ax=ax,color=[.6,.6,.6],skel_lw=2,kpt_ms=50,kpt_marker='o')
    plot_fly(kp_noreg[samplei,t],ax=ax,color=[0,0,0],skel_lw=2,kpt_ms=100,kpt_marker='x')
    plot_fly(kp_reg[samplei,t],ax=ax,color=[0,0,.6],kpt_ms=100,kpt_marker='+')

    for i in range(meankp.shape[0]):
        ax.plot([kp_noreg[samplei,t,i,0],kp_reg[samplei,t,i,0],meankp[i,0]],
                [kp_noreg[samplei,t,i,1],kp_reg[samplei,t,i,1],meankp[i,1]],'g-')

    ax.set_aspect('equal')
    ax.set_title(f't = {t}')
    
fig.tight_layout()

# %%
dampenconstants = [0,1e-4,1e-3,1e-2,5e-2,1e-1]
prctilelim = 1e-3

all_pred_iter_reg = [None,]*len(dampenconstants)
for i,dampenconstant in enumerate(dampenconstants):
    np.random.set_state(randstate_np)
    torch.random.set_rng_state(randstate_torch)
    all_pred_iter_reg[i], _ = predict_iterative_all(valdata,iter_val_dataset,model,max_tpred,keepall=False,
                                                labelidx=labelidx_iter,nsamples=2,dampenconstant=dampenconstant,
                                                posestats=posestats,prctilelim=prctilelim)

# %%
dampencolors = plt.cm.jet(np.arange(len(dampenconstants))/len(dampenconstants))*.8

savefig = True
toff0 = 50
tsplot = config['contextl']+np.arange(-toff0,max_tpred+1)

for exampleii in range(len(labelidx_iter)):

    fig = None
    examplei = labelidx_iter[exampleii]
    true_example_dict = iter_val_dataset[examplei]
    true_example_obj = FlyExample(example_in=true_example_dict,dataset=iter_val_dataset)
    for i in range(len(dampenconstants)):
        fig = plot_pose_pred_iter_vs_true(all_pred_iter_reg[i][exampleii],true_example_obj,tsplot=tsplot,maxsamplesplot=2,
                                        samplecolors=np.tile(dampencolors[i,:],[2,1]),fig=fig,plottrue=i==len(dampenconstants)-1,
                                        samplelabel=f'dampen = {dampenconstants[i]}')
    axs = fig.get_axes()
    for axi,ax in enumerate(axs):
        # set the y lim so that all lines are visible
        h = ax.get_lines()
        ylimcurr = compute_ylim(h)
        ax.set_ylim(ylimcurr)
        ax.plot([toff0+1,toff0+1],ylimcurr,'k--')
        if axi >= 3:
            meanv = posestats['meanrelpose'][axi-3]
            ax.plot([0,tsplot[-1]-tsplot[0]],[meanv,meanv],'k--')
        
    if savefig:
        metadata = true_example_obj.metadata
        fig.savefig(os.path.join(outfigdir,f'pose_pred_iter_vs_true_dampen_{config["model_nickname"]}_t0_{metadata["t0"]}_fly_{metadata["flynum"]}.pdf'))
        plt.close(fig)
    else:
        plt.show()

plt.close('all')

# %%
savefig = True

for exampleii in range(len(labelidx_iter)):

    fig = None
    examplei = labelidx_iter[exampleii]
    true_example_dict = iter_val_dataset[examplei]
    true_example_obj = FlyExample(example_in=true_example_dict,dataset=iter_val_dataset)
    for i in range(len(dampenconstants)):
        fig = plot_multi_pred_iter_vs_true(all_pred_iter_reg[i][exampleii],true_example_obj,tsplot=tsplot,maxsamplesplot=2,
                                            samplecolors=np.tile(dampencolors[i,:],[2,1]),fig=fig,plottrue=i==len(dampenconstants)-1,
                                            samplelabel=f'dampen = {dampenconstants[i]}')
    axs = fig.get_axes()
    for axi,ax in enumerate(axs):
        # set the y lim so that all lines are visible
        h = ax.get_lines()
        ylimcurr = compute_ylim(h)
        ax.set_ylim(ylimcurr)
        ax.plot([toff0+1,toff0+1],ylimcurr,'k--')
        if axi >= 3:
            meanv = posestats['meanrelpose'][axi-3]
            ax.plot([0,tsplot[-1]-tsplot[0]],[meanv,meanv],'k--')
        
    if savefig:
        metadata = true_example_obj.metadata
        fig.savefig(os.path.join(outfigdir,f'multi_pred_iter_vs_true_dampen_{config["model_nickname"]}_t0_{metadata["t0"]}_fly_{metadata["flynum"]}.pdf'))
        plt.close(fig)
    else:
        plt.show()
        break


# %%
# simulate all flies with dampening

# reseed numpy random number generator with randstate_np
np.random.set_state(randstate_np)
# reseed torch random number generator with randstate_torch
torch.random.set_rng_state(randstate_torch)

dampenconstant = .01
prctilelim = 1e-3

animate_pose_params = {'figsizebase': 8,'ms': 4, 'focus_ms':8, 'lw': .75, 'focus_lw': 2}
predict_iterative_params = {'dampenconstant': dampenconstant, 'prctilelim': prctilelim, 'posestats': posestats}

# choose data to initialive behavior modeling
tpred = np.minimum(4000 + config['contextl'], valdata['isdata'].shape[0] // 2)

# all frames must have real data
burnin = config['contextl'] - 1
contextlpad = burnin + val_dataset.ntspred_max
allisdata = interval_all(valdata['isdata'], contextlpad)
isnotsplit = interval_all(valdata['isstart'] == False, tpred)[1:, ...]
canstart = np.logical_and(allisdata[:isnotsplit.shape[0], :], isnotsplit)
flynum = 2  # 2
t0 = np.nonzero(canstart[:, flynum])[0]
idxstart = np.minimum(8700, len(t0) - 1)

if len(t0) > idxstart:
    t0 = t0[idxstart]
else:
    t0 = t0[0]
isreal = np.any(np.isnan(valdata['X'][...,t0:t0+tpred,:]),axis=(0,1,2)) == False
print(isreal)
fliespred = np.nonzero(isreal)[0]

nsamplesfuture = 0 # 32

isreal = np.any(np.isnan(valdata['X'][...,t0:t0+tpred,:]),axis=(0,1,2)) == False
Xkp_init = valdata['X'][...,t0:t0+tpred+val_dataset.ntspred_max,isreal]
scales_pred = []
for flypred in fliespred:
  id = valdata['ids'][t0, flypred]
  scales_pred.append(val_scale_perfly[:,id])
metadata = {'t0': t0, 'ids': fliespred, 'videoidx': valdata['videoidx'][t0], 'frame0': valdata['frames'][t0]}
print(metadata)

ani = animate_predict_open_loop(model, val_dataset, Xkp_init, fliespred, scales_pred, tpred, 
                          debug=False, plotattnweights=False, plotfuture=val_dataset.ntspred_max > 1, 
                          nsamplesfuture=nsamplesfuture, metadata=metadata,
                          animate_pose_params=animate_pose_params,plottrue=False,
                          predict_iterative_params=predict_iterative_params)

# %%
# write the video to file 
vidtime = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
flynumstr = 'all'
savevidfile = os.path.join(config['savedir'], f"samplevideo_dampen_{dampenconstant}_{modeltype_str}_{savetime}_{vidtime}_t0_{metadata['t0']}_flies{flynumstr}.mp4")
print('Saving animation to file %s...'%savevidfile)
save_animation(ani, savevidfile)
print('Finished writing.')
