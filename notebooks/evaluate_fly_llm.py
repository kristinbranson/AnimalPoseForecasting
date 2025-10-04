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

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pickle

import apf
from apf.training import train
from apf.utils import function_args_from_config, set_mpl_backend
from apf.simulation import simulate
from apf.models import initialize_model

import flyllm
from flyllm.config import read_config
from flyllm.features import featglobal, get_sensory_feature_idx
from flyllm.simulation import animate_pose
import time
import os

from flyllm.prepare import init_flyllm

import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

set_mpl_backend('tkAgg')
mpl_backend = plt.get_backend()
if mpl_backend == 'inline':
    from IPython import display
    from IPython.display import HTML

LOG.info('CUDA available: ' + str(torch.cuda.is_available()))
LOG.info('matplotlib backend: ' + mpl_backend)

# %%
timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
print('Timestamp: ' + timestamp)

# %% [markdown]
# ### Configuration

# %%
# configuration parameters for this model

#loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flypredvel_20241022_epoch200_20241023T140356.pth'
#configfile = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241022.json'

# print current directory
loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flypredvel_20241125_epoch200_20241125T191735.pth'
configfile = 'configs/config_fly_llm_predvel_20241125.json'
outfigdir = 'figs'
debug_uselessdata = True

# make directory if it doesn't exist
if not os.path.exists(outfigdir):
    os.makedirs(outfigdir)

flyllmdir = flyllm.__path__[0]
configfile = os.path.join(flyllmdir,configfile)

needtraindata = True
needvaldata = True

res = init_flyllm(configfile=configfile,mode='test',loadmodelfile=loadmodelfile,
                needtraindata=needtraindata,needvaldata=needvaldata,
                debug_uselessdata=debug_uselessdata)


# %%
# bring out things that are init_flyllm


# %%
# configuration parameters for this model

#loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flypredvel_20241022_epoch200_20241023T140356.pth'
#configfile = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241022.json'
loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flypredvel_20241125_epoch200_20241125T191735.pth'
configfile = 'flyllm/configs/config_fly_llm_predvel_20241125.json'
outfigdir = 'figs'

config = read_config(configfile)

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
modeltype_str = res['modeltype_str']
savetime = res['model_savetime']

pred_nframes_skip = 20

# where to save predictions
# remove extension from loadmodelfile
savepredfile = loadmodelfile.split('.')[0] + f'_all_pred_skip_{pred_nframes_skip}.npz'

# where to save pose statistics
posestatsfile = loadmodelfile.split('.')[0] + '_posestats.npz'

# %%
# load/compute pose statistics
if posestatsfile is not None:
    if False:#os.path.exists(posestatsfile):
        print("Loading pose stats from ", posestatsfile)
        posestats = np.load(posestatsfile)

    else:
        # check if variable data exists
        if 'data' not in locals():
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

savefig = False
featsplot=[0,1,2,14, 16, 20, 22, 24, 26]
toplots = [{'i': 0, 'tsplot': np.arange(8500,8800)}, {'i': 89, 'tsplot': np.arange(32400,32700)}, ]
nsamples = 100
ylim_nstd = 5

for toplot in toplots:
    i = toplot['i']
    tsplot = toplot['tsplot']
    pred_example = pred_data[i]
    true_example = true_data[i]
    id = true_example.metadata['id']

    fig = plot_multi_pred_vs_true(pred_example,true_example,ylim_nstd=ylim_nstd,nsamples=nsamples,tsplot=tsplot,plotbinedges=True,featsplot=featsplot)

    # save this figure as a pdf
    if savefig:
        fig.savefig(os.path.join(outfigdir,f'multi_pred_vs_true_{config["model_nickname"]}_fly{id}_{tsplot[0]}_to_{tsplot[-1]}.pdf'))
        plt.close(fig)
    else:
        break

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
