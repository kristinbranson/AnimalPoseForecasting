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

from apf.utils import get_dct_matrix, compute_npad
from apf.data import process_test_data, interval_all, debug_less_data
from apf.io import read_config, get_modeltype_str, load_and_filter_data, save_model, load_model, parse_modelfile, load_config_from_model_file
from flyllm.config import scalenames, nfeatures, DEFAULTCONFIGFILE, featglobal, posenames, keypointnames
from flyllm.features import compute_features, sanity_check_tspred, get_sensory_feature_idx, compute_scale_perfly
from flyllm.dataset import FlyTestDataset
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

loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flypredvel_20241022_epoch200_20241023T140356.pth'
configfile = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241022.json'
outfigdir = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/figs'

# loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flypredpos_20241023_epoch200_20241028T165510.pth'
# configfile = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predpos_20241023.json'

# set to None if you want to use the full data
#quickdebugdatafile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/tmp_small_usertrainval.pkl'
quickdebugdatafile = None

# whether to create training data
needtraindata = False

#configfile = "/groups/branson/home/eyjolfsdottire/code/MABe2022/config_fly_llm_multitimeglob_discrete_20230907.json"
config = read_config(configfile,
                     default_configfile=DEFAULTCONFIGFILE,
                     get_sensory_feature_idx=get_sensory_feature_idx,
                     featglobal=featglobal,
                     posenames=posenames)

# set loadmodelfile from config if not specified
if loadmodelfile is None and 'loadmodelfile' in config:
    loadmodelfile = config['loadmodelfile']

load_config_from_model_file(loadmodelfile=loadmodelfile,config=config)
assert 'dataset_params' in config, 'dataset_params not in config'

# seed the random number generators
np.random.seed(config['numpy_seed'])
torch.manual_seed(config['torch_seed'])

# set device (cuda/cpu)
device = torch.device(config['device'])
if device.type == 'cuda':
    assert torch.cuda.is_available(), 'CUDA is not available'

pred_nframes_skip = 20

# where to save predictions
# remove extension from loadmodelfile
savepredfile = loadmodelfile.split('.')[0] + f'_all_pred_skip_{pred_nframes_skip}.npz'

# %% [markdown]
# ### Load data

# %%
# load raw data
if quickdebugdatafile is None:
    if needtraindata:
        data, scale_perfly = load_and_filter_data(config['intrainfile'], config, compute_scale_perfly,
                                                keypointnames=keypointnames)
        print(f"training data X shape: {data['X'].shape}")
    valdata, val_scale_perfly = load_and_filter_data(config['invalfile'], config, compute_scale_perfly,
                                                     keypointnames=keypointnames)
    print(f"val data X shape: {valdata['X'].shape}")
    #LOG.warning('DEBUGGING!!!')
    #debug_less_data(data, 10000000)
    #debug_less_data(valdata, 10000)
else:
    print("Loading data from quick debug data file ", quickdebugdatafile)
    with open(quickdebugdatafile,'rb') as f:
        tmp = pickle.load(f)
        data = tmp['data']
        scale_perfly = tmp['scale_perfly']
        valdata = tmp['valdata']
        val_scale_perfly = tmp['val_scale_perfly']

# %% [markdown]
# ### Compute features
# Compute the pose representation and chunk data into sequences for training

# %%
# process data

# if using discrete cosine transform, create dct matrix
# this didn't seem to work well, so probably won't use in the future
if config['dct_tau'] is not None and config['dct_tau'] > 0:
    dct_m, idct_m = get_dct_matrix(config['dct_tau'])
    # this gives the maximum of 
    #   a) max number of frames to lookahead or 
    #   b) dct_tau (number of timepoints for cosine transform)
else:
    dct_m = None
    idct_m = None

# how much to pad outputs by -- depends on how many frames into the future we will predict
npad = compute_npad(config['tspred_global'], dct_m)
chunk_data_params = {'npad': npad, 'minnframes': config['contextl']+1}

compute_feature_params = {
    "simplify_out": config['simplify_out'],
    "simplify_in": config['simplify_in'],
    "dct_m": dct_m,
    "tspred_global": config['tspred_global'],
    "compute_pose_vel": config['compute_pose_vel'],
    "discreteidx": config['discreteidx'],
}

# function for computing features
reparamfun = lambda x, id, flynum, **kwargs: compute_features(
    x, id, flynum, scale_perfly, outtype=np.float32, **compute_feature_params, **kwargs)

val_reparamfun = lambda x, id, flynum, **kwargs: compute_features(
    x, id, flynum, val_scale_perfly, outtype=np.float32, **compute_feature_params, **kwargs)

# process the data

if needtraindata:
    LOG.info('Processing training data...')
    X = process_test_data(data, reparamfun, **chunk_data_params)
    LOG.info(f'{len(X)} training ids, total of {sum([len(x) for x in X])} time points')
LOG.info('Processing val data...')
valX = process_test_data(valdata, val_reparamfun, **chunk_data_params)
LOG.info(f'{len(valX)} val ids, total of {sum([len(x) for x in valX])} time points')

# %% [markdown]
# ### Create Dataset, DataLoader objects

# %%
dataset_params = {
    'max_mask_length': config['max_mask_length'],
    'pmask': config['pmask'],
    'masktype': config['masktype'],
    'simplify_out': config['simplify_out'],
    'pdropout_past': config['pdropout_past'],
    'input_labels': config['input_labels'],
    'dozscore': True,
    'discreteidx': config['discreteidx'],
    'discretize_nbins': config['discretize_nbins'],
    'discretize_epsilon': config['discretize_epsilon'],
    'flatten_labels': config['flatten_labels'],
    'flatten_obs_idx': config['flatten_obs_idx'],
    'flatten_do_separate_inputs': config['flatten_do_separate_inputs'],
    'p_add_input_noise': config['p_add_input_noise'],
    'dct_ms': (dct_m,idct_m),
    'tspred_global': config['tspred_global'],
    'discrete_tspred': config['discrete_tspred'],
    'compute_pose_vel': config['compute_pose_vel'],
    
}
# zscore and discretize parameters
for k in config['dataset_params']:
    dataset_params[k] = config['dataset_params'][k]

if needtraindata:
    LOG.info('Creating training data set...')
    train_dataset = FlyTestDataset(X,config['contextl'],**dataset_params)
    LOG.info(f'Train dataset size: {len(train_dataset)}')

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=config['test_batch_size'],
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    )
    ntrain_batches = len(train_dataloader)
    LOG.info(f'Number of training batches: {ntrain_batches}')

LOG.info('Creating validation data set...')
val_dataset = FlyTestDataset(valX,config['contextl'],**dataset_params)
print(f'Validation dataset size: {len(val_dataset)}')
    
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=config['test_batch_size'],
                                              shuffle=False,
                                              pin_memory=True,
                                              )
nval_batches = len(val_dataloader)
LOG.info(f'Number of validation batches: {nval_batches}')
example = next(iter(val_dataloader))
LOG.info(f'batch keys: {example.keys()}')
sz = example['input'].shape
LOG.info(f'batch input shape = {sz}')

# %%
# check that sizes of everything make sense
ntimepoints_valdata = np.count_nonzero(np.any(np.isnan(valdata['X'])==False,axis=(0,1)))
print(f'ntimepoints_valdata = {ntimepoints_valdata}')
ntimepoints_valX = np.sum([x['input'].shape[0] for x in valX])
print(f'ntimepoints_valX = {ntimepoints_valX}')
nexamples_val = len(val_dataset)
print(f'nexamples_val = {nexamples_val}')
nbatches_val = len(val_dataloader)
print(f'nbatches_val = {nbatches_val}, batch size = {config["test_batch_size"]}, total examples = {nbatches_val*config["test_batch_size"]}')

if needtraindata:
    ntimepoints_traindata = np.count_nonzero(np.any(np.isnan(data['X'])==False,axis=(0,1)))
    print(f'ntimepoints_traindata = {ntimepoints_traindata}')
    ntimepoints_trainX = np.sum([x['input'].shape[0] for x in X])
    print(f'ntimepoints_trainX = {ntimepoints_trainX}')
    nexamples_train = len(train_dataset)
    print(f'nexamples_train = {nexamples_train}')
    nbatches_train = len(train_dataloader)
    print(f'nbatches_train = {nbatches_train}, batch size = {config["test_batch_size"]}, total examples = {nbatches_train*config["test_batch_size"]}')

# %% [markdown]
# ### Load the model

# %%
# create the model
model, criterion = initialize_model(config, val_dataset, device)

# load the model
modeltype_str, savetime = parse_modelfile(loadmodelfile)
loss_epoch = load_model(loadmodelfile, model, device)

# create attention mask
contextl = example['input'].shape[1]
if config['modeltype'] == 'mlm':
    train_src_mask = generate_square_full_mask(contextl).to(device)
    is_causal = False
elif config['modeltype'] == 'clm':
    train_src_mask = torch.nn.Transformer.generate_square_subsequent_mask(contextl, device=device)
    is_causal = True
    #train_src_mask = generate_square_subsequent_mask(contextl).to(device)
else:
    raise ValueError(f'Unknown modeltype: {config["modeltype"]}')

opt_model = torch.compile(model)

# %%
# profile stuff 

import cProfile
import pstats

from flyllm.prediction import predict_all

doprofile = True
dodebug = False
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
# try making context length longer and masking out earlier frames
print(train_src_mask.cpu().numpy()[:5,:5])
tri1 = np.tri(2*config['contextl']-1,k=0)
tri2 = np.tri(2*config['contextl']-1,k=-config['contextl'])
tri = np.log(tri1-tri2)
test_mask = torch.from_numpy(tri).to(device,dtype=torch.float32)
print(tri[:5,:5])
fig,ax = plt.subplots(1,2)
ax[0].imshow(train_src_mask.cpu().numpy())
ax[0].set_title('train_src_mask')
ax[1].imshow(tri)
ax[1].set_title('test_mask')
quadtri1 = np.tri(4*config['contextl']-1,k=0)
quadtri2 = np.tri(4*config['contextl']-1,k=-config['contextl'])
quadtri = np.log(quadtri1-quadtri2)
quad_test_mask = torch.from_numpy(quadtri).to(device,dtype=torch.float32)
quad_test_mask1 = torch.from_numpy(np.log(quadtri1)).to(device,dtype=torch.float32)

val_dataset_small = FlyTestDataset(valX[1:2],config['contextl'],**dataset_params,need_metadata=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset_small,
                                              batch_size=256,
                                              shuffle=False,
                                              pin_memory=True
                                              )
double_val_dataset_small = FlyTestDataset(valX[1:2],2*config['contextl'],**dataset_params)
double_val_dataloader = torch.utils.data.DataLoader(double_val_dataset_small,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    pin_memory=True)
quadruple_val_dataset_small = FlyTestDataset(valX[1:2],4*config['contextl'],**dataset_params)
quadruple_val_dataloader = torch.utils.data.DataLoader(quadruple_val_dataset_small,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       pin_memory=True)
example = next(iter(val_dataloader))
double_example = next(iter(double_val_dataloader))
quadruple_example = next(iter(quadruple_val_dataloader))

with torch.no_grad():
    out = model(example['input'].to(device=device),train_src_mask)
    doubleout = model(double_example['input'].to(device=device),test_mask)
    quadrupleout = model(quadruple_example['input'].to(device=device),quad_test_mask)

for k in out:
    out[k] = out[k].cpu().numpy()
    doubleout[k] = doubleout[k].cpu().numpy()
    quadrupleout[k] = quadrupleout[k].cpu().numpy()

print(f"error in first 511 frames: {np.max(np.abs(doubleout['continuous'][0,:config['contextl']-1]-out['continuous'][0]))}")
fig,ax = plt.subplots(1,2,figsize=(10,5))
err = np.zeros(out['continuous'].shape[0])
for i in range(out['continuous'].shape[0]):
    err[i] = np.abs(out['continuous'][i,-1,:]-doubleout['continuous'][0,config['contextl']-2+i,:]).max()
ax[0].plot(err)
ax[0].set_title('double error')
quaderr = np.abs(quadrupleout['continuous'][0,:2*config['contextl']-1]-doubleout['continuous'][0]).max(axis=-1)
ax[1].plot(quaderr)
ax[1].set_title(f'quadruple error')

# %%
# mess up some inputs and see how far the error affects things

idxmess = range(50,55)
double_example_nan = {}
for k,v in double_example.items():
    double_example_nan[k] = v.clone()
double_example_nan['input'][:,idxmess,:] = (torch.rand_like(double_example_nan['input'][:,idxmess,:])-.5)*9999999
with torch.no_grad():
    doubleoutnan = model(double_example_nan['input'].to(device=device),test_mask)
for k in doubleoutnan:
    doubleoutnan[k] = doubleoutnan[k].cpu().numpy()
    
quadruple_example_nan = {}
for k,v in quadruple_example.items():
    quadruple_example_nan[k] = v.clone()
quadruple_example_nan['input'][:,idxmess,:] = (torch.rand_like(quadruple_example_nan['input'][:,idxmess,:])-.5)*9999999
with torch.no_grad():
    quadrupleoutnan = model(quadruple_example_nan['input'].to(device=device),quad_test_mask)
    quadrupleout1 = model(quadruple_example['input'].to(device=device),quad_test_mask1)
    quadrupleoutnan1 = model(quadruple_example_nan['input'].to(device=device),quad_test_mask1)
for k in quadrupleoutnan:
    quadrupleoutnan[k] = quadrupleoutnan[k].cpu().numpy()
    quadrupleout1[k] = quadrupleout1[k].cpu().numpy()
    quadrupleoutnan1[k] = quadrupleoutnan1[k].cpu().numpy()

np.count_nonzero(np.isnan(doubleoutnan['continuous'][0,:,0]))
fig,ax = plt.subplots(3,3,figsize=(20,10),sharex=True)
ax[0,0].plot(doubleout['continuous'][0,:,:])  
ax[0,0].set_title('doubleout')
ax[0,1].plot(doubleoutnan['continuous'][0,:,:])
ax[0,1].set_title('doubleoutnan')
err = np.mean(np.abs(doubleoutnan['continuous'][0,:,:]-doubleout['continuous'][0,:,:]),axis=1)
ax[0,2].plot(err)
ax[0,2].set_title('error')

ax[1,0].plot(quadrupleout['continuous'][0,:,:])
ax[1,0].set_title('quadrupleout')
ax[1,1].plot(quadrupleoutnan['continuous'][0,:,:])
ax[1,1].set_title('quadrupleoutnan')
err = np.mean(np.abs(quadrupleoutnan['continuous'][0,:,:]-quadrupleout['continuous'][0,:,:]),axis=1)
ax[1,2].plot(err)
ax[1,2].set_title('error')

ax[2,0].plot(quadrupleout1['continuous'][0,:,:])
ax[2,0].set_title('quadrupleout1')
ax[2,1].plot(quadrupleoutnan1['continuous'][0,:,:])
ax[2,1].set_title('quadrupleoutnan1')
err = np.mean(np.abs(quadrupleoutnan1['continuous'][0,:,:]-quadrupleout1['continuous'][0,:,:]),axis=1)
ax[2,2].plot(err)
ax[2,2].set_title('error')



# %%
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
pred_data,true_data = val_dataset.create_data_from_pred(all_pred,labelidx)

# %%

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
for pred_example,true_example in zip(pred_data,true_data):
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

i = 0
pred_example = pred_data[i]
true_example = true_data[i]
tsplot = np.arange(8775,8925)

fig = plot_multi_pred_vs_true(pred_example,true_example,ylim_nstd=5,nsamples=100,tsplot=tsplot)

# save this figure as a pdf
#fig.savefig(os.path.join(outfigdir,f'multi_pred_vs_true_{config["model_nickname"]}_{tsplot[0]}_to_{tsplot[-1]}.pdf'))

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
tpred = np.minimum(1000 + config['contextl'], valdata['isdata'].shape[0] // 2)

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
