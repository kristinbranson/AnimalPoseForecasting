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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Imports

# %%
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
from apf.models import (
    initialize_model,
    initialize_loss,
    generate_square_full_mask,
    sanity_check_temporal_dep,
    criterion_wrapper,
    update_loss_nepochs,
)
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
#loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flymulttimeglob_predposition_20240305_epoch130_20240825T122647.pth'
configfile = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241022.json'
# set to None if you want to use the full data
#quickdebugdatafile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/tmp_small_usertrainval.pkl'
quickdebugdatafile = None

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

# %% [markdown]
# ### Load data

# %%
# load raw data
if quickdebugdatafile is None:
    data, scale_perfly = load_and_filter_data(config['intrainfile'], config, compute_scale_perfly,
                                              keypointnames=keypointnames)
    valdata, val_scale_perfly = load_and_filter_data(config['invalfile'], config, compute_scale_perfly,
                                                     keypointnames=keypointnames)
    #LOG.warning('DEBUGGING!!!')
    #debug_less_data(data, 10000000)
    #debug_less_data(valdata, 10000)
else:
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
chunk_data_params = {'npad': npad}

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
for k in config['dataset_params']:
    dataset_params[k] = config['dataset_params'][k]

LOG.info('Creating validation data set...')
val_dataset = FlyTestDataset(valX,config['contextl'],**dataset_params)
print(f'Validation dataset size: {len(val_dataset)}')
    
LOG.info('Creating training data set...')
train_dataset = FlyTestDataset(X,config['contextl'],**dataset_params)
LOG.info(f'Train dataset size: {len(train_dataset)}')

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config['test_batch_size'],
                                                shuffle=False,
                                                pin_memory=True,
                                                )
ntrain_batches = len(train_dataloader)
LOG.info(f'Number of training batches: {ntrain_batches}')

val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=config['test_batch_size'],
                                              shuffle=False,
                                              pin_memory=True,
                                              )
nval_batches = len(val_dataloader)
LOG.info(f'Number of validation batches: {nval_batches}')
example = next(iter(train_dataloader))
LOG.info(f'batch keys: {example.keys()}')
sz = example['input'].shape
LOG.info(f'batch input shape = {sz}')

# %% [markdown]
# ### Load the model

# %%
# create the model
model, criterion = initialize_model(config, train_dataset, device)

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

# %%
# profile stuff 

import cProfile
import pstats

doprofile = True
dodebug = True
from flyllm.dataset import FlyTestDataset

def profile_dataset_creation():
    val_dataset_small = FlyTestDataset(valX[:min(len(valX),100)],config['contextl'],**dataset_params)

def profile_iterating_dataset():
    for i, batch in enumerate(val_dataloader):
        if i > 10:
            break
        
def profile_predict_all():
    predict_all(val_dataloader, val_dataset, model, config, train_src_mask,keepall=False,earlystop=5)

if doprofile:

    cProfile.run('profile_predict_all()','profile_test.out')
    p = pstats.Stats('profile_test.out')
    p.strip_dirs().sort_stats('tottime').print_stats(20)

if (not doprofile) and dodebug:
    profile_iterating_dataset()

# %% [markdown]
# ### Evaluate tau-frame predictions

# %%
debugcheat = False
savepredfile = 'all_pred.npz'

# compute predictions and labels for all validation data
val_dataset = FlyTestDataset(valX,config['contextl'],**dataset_params,allow_debugcheat=debugcheat)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=config['test_batch_size'],
                                              shuffle=False,
                                              pin_memory=True,
                                              )
all_pred, labelidx = predict_all(val_dataloader, val_dataset, model, config, train_src_mask, keepall=False, debugcheat=debugcheat,earlystop=50)

# save all_pred and labelidx to a numpy file
if savepredfile is not None:
    np.savez('all_pred.npz',all_pred=all_pred,labelidx=labelidx)

# %%
# load all_pred and labelidx from all_pred.npz
savepredfile = 'all_pred.npz'
if savepredfile is not None and os.path.exists('all_pred.npz'):
    tmp = np.load('all_pred.npz',allow_pickle=True)
    all_pred = tmp['all_pred'].item()
    labelidx = tmp['labelidx']
else:
    raise ValueError(f'{savepredfile} does not exist')

# %%
# compare predictions to labels

# pred_data is a list of FlyExample objects
pred_data,true_data = val_dataset.create_data_from_pred(all_pred, labelidx)

# compute error in various ways
err_example = []
for pred_example,true_example in zip(pred_data,true_data):
    errcurr = pred_example.compute_error(true_example=true_example,pred_example=pred_example)
    err_example.append(errcurr)

# combine errors
err_total = FlyExample.combine_errors(err_example)


# %%
for k,v in err_total.items():
    print(f'{k}: {type(v)}')
    if type(v) is np.ndarray:
        print(f'    {v.shape}')

# %%
np.stack([x[k] for x in err_example]).shape

# %%
for k,v in meanerr.items():
    print(f'{k}: {v.shape}')
pred_example.labels.is_zscored()

# %%
# plot errors

i = 0
pred_example = pred_data[i]
true_example = true_data[i]
err = err_example[i]

featnum = 0
featnum = 58
nsamples = 100
#tsplot = np.arange(pred_example.ntimepoints-1)
tsplot = np.arange(8700,8800)

fig,ax = plt.subplots(2,1,sharex=True,figsize=(16,16))
multi_pred = pred_example.labels.get_multi(nsamples=0,collapse_samples=True,use_todiscretize=False)
multi_pred_sample = pred_example.labels.get_multi(nsamples=nsamples,collapse_samples=True,use_todiscretize=False)
multi_pred_meansample = np.nanmean(multi_pred_sample,axis=0)
multi_true = true_example.labels.get_multi(use_todiscretize=True)
multi_isdiscrete = pred_example.labels.get_multi_isdiscrete()
bestsample = np.argmin(np.abs(multi_pred_sample-multi_true[None,...]),axis=0)

plotsamples = multi_isdiscrete[featnum]
tabcolors = plt.cm.tab10(np.arange(10))
colors = {}
colors['true'] = 'k'
colors['weightedsum'] = tabcolors[0]
colors['meansample'] = tabcolors[1]
colors['minsample'] = tabcolors[2]
colors['sample'] = tabcolors[3]

miny = np.nanmin(multi_true[tsplot,featnum])
maxy = np.nanmax(multi_true[tsplot,featnum])
dy = maxy-miny
ylim = (miny-.1*dy,maxy+.1*dy)

if plotsamples:
    for i in range(multi_pred_sample.shape[0]):
        c = colors['sample'].copy()
        c[-1] = .05
        h, = ax[0].plot(multi_pred_sample[i,tsplot,featnum],'.',color=c)
        if i == 0:
            h.set_label('random sample')
    ax[0].plot(multi_pred_meansample[tsplot,featnum],'.-',color=colors['meansample'],label='mean sample')
    ax[0].plot(multi_pred_sample[bestsample[tsplot,featnum],tsplot,featnum],'.-',label='best sample',color=colors['minsample'])

ax[0].plot(multi_pred[tsplot,featnum],'.-',label='expected value',color=colors['weightedsum'])
ax[0].plot(multi_true[tsplot,featnum],'.-',label='true',color=colors['true'])
ax[0].set_ylim(ylim)
ax[0].legend()
ax[0].set_ylabel(f'{err["multi_names"][featnum]}')

ax[1].plot(err['absdiff_multi'][tsplot,featnum],'.-',label='expected value',color=colors['weightedsum'])
if plotsamples:
    ax[1].plot(err['absdiff_multi_samplemean'][tsplot,featnum],'.-',label='mean sample',color=colors['meansample'])
    ax[1].plot(err['absdiff_multi_samplemin'][tsplot,featnum],'.-',label='best sample',color=colors['minsample'])

ax[1].set_ylabel(f'abs error {err["multi_names"][featnum]}')
ax[1].legend()
fig.tight_layout()
