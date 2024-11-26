# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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

# %% [markdown]
# ### Imports

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
import datetime
import os
from matplotlib import animation
import pickle

from flyllm.prepare import init_flyllm
from apf.utils import get_dct_matrix, compute_npad, save_animation
from apf.data import process_test_data, interval_all, debug_less_data, chunk_data
from apf.io import read_config, get_modeltype_str, load_and_filter_data, save_model, load_model, parse_modelfile, load_config_from_model_file
from flyllm.config import scalenames, nfeatures, DEFAULTCONFIGFILE, featglobal, posenames, keypointnames
from flyllm.features import compute_features, sanity_check_tspred, get_sensory_feature_idx, compute_scale_perfly, compute_pose_distribution_stats
from flyllm.dataset import FlyTestDataset, FlyMLMDataset
from flyllm.pose import FlyExample, FlyPoseLabels, FlyObservationInputs
from flyllm.plotting import (
    initialize_debug_plots, 
    initialize_loss_plots, 
    update_debug_plots,
    update_loss_plots,
    debug_plot_global_histograms, 
    debug_plot_dct_relative_error, 
    debug_plot_global_error, 
    debug_plot_predictions_vs_labels,
    select_featidx_plot,
)
from apf.models import (
    initialize_model, 
    initialize_loss, 
    compute_loss,
    generate_square_full_mask, 
    sanity_check_temporal_dep,
    criterion_wrapper,
    update_loss_nepochs,
    stack_batch_list,
)
from flyllm.simulation import animate_predict_open_loop
from flyllm.prediction import predict_all
from IPython.display import HTML

import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

mpl_backend = plt.get_backend()
if mpl_backend == 'inline':
    from IPython import display

LOG.info('CUDA available: ' + str(torch.cuda.is_available()))
LOG.info('matplotlib backend: ' + mpl_backend)

# %% [markdown]
# ### Configuration

# %%
# configuration parameters for this model

#loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flypredvel_20241022_epoch200_20241023T140356.pth'
#configfile = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241022.json'
loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flypredvel_20241125_epoch200_20241125T191735.pth'
configfile = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241125.json'
outfigdir = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/figs'

# loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flypredpos_20241023_epoch200_20241028T165510.pth'
# configfile = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predpos_20241023.json'

# set to None if you want to use the full data
#quickdebugdatafile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/tmp_small_usertrainval.pkl'
quickdebugdatafile = None

needtraindata = True
needvaldata = True
traindataprocess = 'test'
valdataprocess = 'test'

res = init_flyllm(configfile=configfile,mode='test',loadmodelfile=loadmodelfile,
                  quickdebugdatafile=quickdebugdatafile,needtraindata=needtraindata,needvaldata=needvaldata,
                  traindataprocess=traindataprocess,valdataprocess=valdataprocess)

# unpack res
config = res['config']
device = res['device']
data = res['data']
scale_perfly = res['scale_perfly']
valdata = res['valdata']
val_scale_perfly = res['val_scale_perfly']
X = res['X']
valX = res['valX']
train_dataset = res['train_dataset']
train_dataloader = res['train_dataloader']
val_dataset = res['val_dataset']
val_dataloader = res['val_dataloader']
dataset_params = res['dataset_params']
model = res['model']
criterion = res['criterion']
opt_model = res['opt_model']
train_src_mask = res['train_src_mask']
is_causal = res['is_causal']

pred_nframes_skip = 20

# where to save predictions
# remove extension from loadmodelfile
savepredfile = loadmodelfile.split('.')[0] + f'_all_pred_skip_{pred_nframes_skip}.npz'

# where to save pose statistics
posestatsfile = loadmodelfile.split('.')[0] + '_posestats.npz'

# %%
# load/compute pose statistics
if posestatsfile is not None:
    if os.path.exists(posestatsfile):
        print("Loading pose stats from ", posestatsfile)
        posestats = np.load(posestatsfile)

    else:
        # check if variable data exists
        if needtraindata == False:
            data, scale_perfly = load_and_filter_data(config['intrainfile'], config, compute_scale_perfly,
                                                    keypointnames=keypointnames)
        posestats = compute_pose_distribution_stats(data,scale_perfly)
        np.savez(posestatsfile, **posestats)

# %%
# profile stuff 

import cProfile
import pstats

from flyllm.prediction import predict_all

doprofile = False
dodebug = True
from flyllm.dataset import FlyTestDataset
val_dataset.clear_cuda_cache()
val_dataset.ncudacaches = 0
val_dataset.cudaoptimize = False

def profile_dataset_creation():
    val_dataset_small = FlyTestDataset(valX[:min(len(valX),100)],config['contextl'],**dataset_params)

def profile_iterating_dataset():
    for i, batch in enumerate(val_dataloader):
        if i > 10:
            break
        
def profile_predict_all():
    return predict_all(val_dataloader, val_dataset, opt_model, config, train_src_mask,keepall=False,earlystop=10)

val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=config['test_batch_size'],
                                              shuffle=False,
                                              pin_memory=val_dataset.cudaoptimize==False,
                                              )

if doprofile:

    # get start time
    print('cudaoptimize = ', val_dataset.cudaoptimize)
    t0 = datetime.datetime.now()
    #cProfile.run('profile_predict_all()','profile_test.out')
    out = profile_predict_all()
    t1 = datetime.datetime.now()
    print('Elapsed time: ', t1-t0)
    print('ncudacaches = ', val_dataset.ncudacaches)

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
# clean up 
import gc
torch.cuda.empty_cache()
gc.collect()

print(torch.cuda.memory_summary())
print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# %% [markdown]
# ### Evaluate single-iteration predictions

# %%
debugcheat = False

if val_dataset.need_labels != debugcheat:
    val_dataset = FlyTestDataset(valX,config['contextl'],**dataset_params,need_labels=debugcheat)

tmpsavepredfile = savepredfile.split('.')[0] + '_tmp.npz'
all_pred, labelidx = predict_all(dataset=val_dataset, model=model, config=config, mask=train_src_mask, keepall=False, debugcheat=debugcheat, 
                                 savepredfile=tmpsavepredfile, saveinterval=600, skipinterval=pred_nframes_skip)

# save all_pred and labelidx to a numpy file
if (not debugcheat) and (savepredfile is not None):
    print(f'Saving predictions to {savepredfile}')
    np.savez(savepredfile,all_pred=all_pred,labelidx=labelidx)

# %%

# load all_pred and labelidx from savepredfile
if savepredfile is not None and os.path.exists(savepredfile):
    print(f'Loading predictions from {savepredfile}')
    tmp = np.load(savepredfile,allow_pickle=True)
    all_pred = tmp['all_pred'].item()
    labelidx = tmp['labelidx']

# %%
# compare predictions to labels

# pred_data is a list of FlyExample objects
pred_data,true_data = val_dataset.create_data_from_pred(all_pred, labelidx)

# compute error in various ways
err_example = []
for pred_example,true_example in tqdm.tqdm(zip(pred_data,true_data),total=len(pred_data)):
    errcurr = true_example.labels.compute_error(pred_example.labels)
    err_example.append(errcurr)

# combine errors
err_total = FlyPoseLabels.combine_errors(err_example)

for k,v in err_total.items():
    print(f'{k}: {type(v)}')
    if type(v) is np.ndarray:
        print(f'    {v.shape}')

# %%
# plot multi errors
from flyllm.plotting import plot_multi_pred_vs_true

featsplot=[0,1,2,14, 16, 20, 22, 24, 26]
toplots = [{'i': 0, 'tsplot': np.arange(8500,8800)}, {'i': 89, 'tsplot': np.arange(32400,32700)}, ]
nsamples = 2
ylim_nstd = 5

for toplot in toplots:
    i = toplot['i']
    tsplot = toplot['tsplot']
    pred_example = pred_data[i]
    true_example = true_data[i]
    id = true_example.metadata['id']

    fig = plot_multi_pred_vs_true(pred_example,true_example,ylim_nstd=ylim_nstd,nsamples=nsamples,tsplot=tsplot)

    # save this figure as a pdf
    fig.savefig(os.path.join(outfigdir,f'multi_pred_vs_true_{config["model_nickname"]}_fly{id}_{tsplot[0]}_to_{tsplot[-1]}.pdf'))

# %%
i = toplots[1]['i']
tsplot = toplots[1]['tsplot']
plt.imshow(pred_data[i].labels.labels_raw['discrete'][tsplot,2,:].T,aspect='auto')
j = np.argmax(np.sum(pred_data[i].labels.labels_raw['discrete'][tsplot,2,:],axis=0))
print(np.diff(dataset_params['discretize_params']['bin_edges'][2][j:j+2]))
print(dataset_params['discretize_epsilon'][2])

# %%
plt.plot(dataset_params['discretize_params']['bin_edges'][2],'.')
np.diff(dataset_params['discretize_params']['bin_edges'][2])


# %%
import re

names = true_example.labels.get_multi_names()
# get indices of names that have 'leg_tip_angle' in the name using regular expressions
idx = [i for i, name in enumerate(names) if re.search('leg_tip_angle', name)]
for i in idx:
    print(f'{i}: {names[i]}')
print(idx)

# %%

# # plot next pose errors
# # I think this doesn't make sense to plot -- it is integrating errors, but also
# # has access to previous frames correct velocities when predicting... 

# i = 0
# pred_example = pred_data[i]
# true_example = true_data[i]
# next_pose_names = true_example.labels.get_next_names()

# #featnum = 58
# nsamples = 100
# #tsplot = np.arange(pred_example.ntimepoints-1)
# tsplot = np.arange(0,100)

# next_pose_pred = pred_example.labels.get_next_pose(nsamples=0,use_todiscretize=False)
# next_pose_pred_sample = pred_example.labels.get_next_pose(nsamples=nsamples,use_todiscretize=False)
# next_pose_pred_meansample = np.nanmean(next_pose_pred_sample,axis=0)
# next_pose_true = true_example.labels.get_next_pose(use_todiscretize=True)
# next_pose_isdiscrete = np.nanmax((np.max(next_pose_pred_sample,axis=0)-np.min(next_pose_pred_sample,axis=0)),axis=0) > 1e-6
# bestsample = np.argmin(np.abs(next_pose_pred_sample-next_pose_true[None,...]),axis=0)

# tabcolors = plt.cm.tab10(np.arange(10))
# colors = {}
# colors['true'] = 'k'
# colors['weightedsum'] = tabcolors[0]
# colors['meansample'] = tabcolors[1]
# colors['minsample'] = tabcolors[2]
# colors['sample'] = tabcolors[3]

# fig,ax = plt.subplots(true_example.labels.d_next_pose,1,sharex=True,figsize=(16,4*true_example.labels.d_next_pose))

# for featnum in range(true_example.labels.d_next_pose):

#     miny = np.nanmin(next_pose_true[tsplot,featnum])
#     maxy = np.nanmax(next_pose_true[tsplot,featnum])
#     dy = maxy-miny
#     ylim = (miny-.1*dy,maxy+.1*dy)

#     if plotsamples:
#         for i in range(next_pose_pred_sample.shape[0]):
#             c = colors['sample'].copy()
#             c[-1] = .05
#             h, = ax[featnum].plot(next_pose_pred_sample[i,tsplot,featnum],'.',color=c)
#             if i == 0:
#                 h.set_label('random sample')
#         ax[featnum].plot(next_pose_pred_meansample[tsplot,featnum],'.-',color=colors['meansample'],label='mean sample')
#         ax[featnum].plot(next_pose_pred_sample[bestsample[tsplot,featnum],tsplot,featnum],'.-',label='best sample',color=colors['minsample'])

#     ax[featnum].plot(next_pose_pred[tsplot,featnum],'.-',label='expected value',color=colors['weightedsum'])
#     ax[featnum].plot(next_pose_true[tsplot,featnum],'.:',label='true',color=colors['true'])
#     ax[featnum].set_ylim(ylim)
#     if featnum == 0:
#         ax[featnum].legend()
#     ax[featnum].set_ylabel(f'{next_pose_names[featnum]}')

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

tpred = np.minimum(10 + config['contextl'], valdata['isdata'].shape[0] // 2)

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
from apf.utils import save_animation

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
max_tpred = 150
iter_val_dataset = FlyTestDataset(valX,config['contextl']+max_tpred+1,**dataset_params,need_labels=True,need_metadata=True,need_init=True,make_copy=True)

# %%
from apf.pose import PoseLabels, ObservationInputs, AgentExample
from flyllm.pose import FlyPoseLabels, FlyObservationInputs, FlyExample
from flyllm.prediction import predict_iterative_all

# all_pred_iter is a list of length N of FlyExample objects
# labelidx_iter is an ndarray of which test example each FlyExample corresponds to
all_pred_iter, labelidx_iter = predict_iterative_all(valdata,iter_val_dataset,model,max_tpred,N=10,keepall=False,nsamples=10)

# %%
# plot multi predictions vs true
from flyllm.plotting import plot_multi_pred_iter_vs_true
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

# %%
# plot pose predictions vs true

import flyllm.plotting
from flyllm.plotting import plot_pose_pred_iter_vs_true

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
FlyExample(example_in=next(iter(iter_val_dataloader)),dataset=iter_val_dataset).labels.get_next_pose(nsamples=10).shape[slice(None)]

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
