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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Train fly forecasting network

# %% [markdown]
# ### Imports

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
import datetime
import os
from matplotlib import animation
import pickle

from apf.utils import get_dct_matrix, compute_npad
from apf.data import chunk_data, interval_all, debug_less_data
from apf.models import (
    initialize_model,
    initialize_loss,
    generate_square_full_mask,
    sanity_check_temporal_dep,
    criterion_wrapper,
)
from apf.io import read_config, get_modeltype_str, load_and_filter_data, save_model, load_model
from flyllm.config import scalenames, nfeatures, DEFAULTCONFIGFILE, featglobal, posenames, keypointnames
from flyllm.features import compute_features, sanity_check_tspred, get_sensory_feature_idx, compute_scale_perfly
from flyllm.dataset import FlyMLMDataset
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
loadmodelfile = None
restartmodelfile = None
configfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/config_fly_llm_multitimeglob_predictposition_20240305.json'
# set to None if you want to use the full data
quickdebugdatafile = None
#configfile = "/groups/branson/home/eyjolfsdottire/code/MABe2022/config_fly_llm_multitimeglob_discrete_20230907.json"
config = read_config(configfile,
                     default_configfile=DEFAULTCONFIGFILE,
                     get_sensory_feature_idx=get_sensory_feature_idx,
                     featglobal=featglobal,
                     posenames=posenames)

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

# sanity check on computing features when predicting many frames into the future
# sanity_check_tspred(
#     data, 
#     compute_feature_params,
#     npad,
#     scale_perfly,
#     contextl=config['contextl'],
#     t0=510,
#     flynum=0
# )

# chunk the data if we didn't load the pre-chunked cache file
LOG.info('Chunking training data...')
X = chunk_data(data, config['contextl'], reparamfun, **chunk_data_params)
LOG.info(f'{len(X)} training chunks')
LOG.info('Chunking val data...')
valX = chunk_data(valdata, config['contextl'], val_reparamfun, **chunk_data_params)
LOG.info(f'{len(valX)} val chunks')
LOG.info(f'Chunk input shape: {X[0]["input"].shape}')
LOG.info(f'Chunk labels shape: {X[0]["labels"].shape}')


# %%
import tracemalloc
import linecache

def display_top(snapshot, key_type='lineno', limit=None):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    if limit is None:
        limit = len(top_stats)
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
    
def update_plots(figs):
    if mpl_backend == 'inline':
        display.clear_output(wait=True)
        for fig in figs:
            if fig is not None:
                display.display(fig)
    else:
        plt.show()
        plt.pause(.1)


# %% [markdown]
# ### Create Dataset, DataLoader objects

# %%
#import cProfile as profile
#tracemalloc.start()

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
train_dataset_params = {
    'input_noise_sigma': config['input_noise_sigma'],
}

LOG.info('Creating training data set...')
train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)

# snapshot = tracemalloc.take_snapshot()
# display_top(snapshot)
# tracemalloc.stop()

train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)
LOG.info(f'Train dataset size: {len(train_dataset)}')

# zscore and discretize parameters for validation data set based on train data
# we will continue to use these each time we rechunk the data
dataset_params['zscore_params'] = train_dataset.get_zscore_params()
dataset_params['discretize_params'] = train_dataset.get_discretize_params()

LOG.info('Creating validation data set...')
val_dataset = FlyMLMDataset(valX,**dataset_params)
print(f'Validation dataset size: {len(val_dataset)}')

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=True,
                                                pin_memory=True,
                                                )
ntrain_batches = len(train_dataloader)
LOG.info(f'Number of training batches: {ntrain_batches}')

val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=config['batch_size'],
                                              shuffle=False,
                                              pin_memory=True,
                                              )
nval_batches = len(val_dataloader)
LOG.info(f'Number of validation batches: {nval_batches}')

example = next(iter(train_dataloader))
sz = example['input'].shape
LOG.info(f'batch input shape = {sz}')
sz = example['labels'].shape
LOG.info(f'batch labels shape = {sz}')
if 'labels_discrete' in example:
    sz = example['labels_discrete'].shape
    LOG.info(f'batch labels_discrete shape = {sz}')


# %% [markdown]
# ### Set up model and training

# %%
# create the model
model, criterion = initialize_model(config, train_dataset, device)

# optimizer
num_training_steps = config['num_train_epochs'] * ntrain_batches
optimizer = torch.optim.AdamW(model.parameters(), **config['optimizer_args']) 
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=0., total_iters=num_training_steps)

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
    raise

# sanity check on temporal dependences
sanity_check_temporal_dep(train_dataloader, device, train_src_mask, is_causal, model, tmess=300)

modeltype_str = get_modeltype_str(config, train_dataset)
if ('model_nickname' in config) and (config['model_nickname'] is not None):
    modeltype_str = config['model_nickname']
    
if restartmodelfile is not None:
    loss_epoch = load_model(restartmodelfile, model, device, lr_optimizer=optimizer, scheduler=lr_scheduler)
    update_loss_nepochs(loss_epoch, config['num_train_epochs'])
    update_loss_plots(hloss, loss_epoch)
    # loss_epoch = {k: v.cpu() for k,v in loss_epoch.items()}
    epoch = np.nonzero(np.isnan(loss_epoch['train'].cpu().numpy()))[0][0]
    progress_bar.update(epoch * ntrain)
else:
    epoch = 0
    # initialize structure to keep track of loss
    loss_epoch = initialize_loss(train_dataset, config)
last_val_loss = None

savetime = datetime.datetime.now()
savetime = savetime.strftime('%Y%m%dT%H%M%S')


# %% [markdown]
# ### Train

# %%
# set up debug plots
#plt.ion()
debug_params = {}
# if contextl is long, still just look at samples from the first 64 frames
if config['contextl'] > 64:
    debug_params['tsplot'] = np.round(np.linspace(0,64,5)).astype(int)
    debug_params['traj_nsamplesplot'] = 1
hdebug = {}
hdebug['train'] = initialize_debug_plots(train_dataset, train_dataloader, data,name='Train', **debug_params)
hdebug['val'] = initialize_debug_plots(val_dataset, val_dataloader, valdata, name='Val', **debug_params)

hloss = initialize_loss_plots(loss_epoch)

progress_bar = tqdm.tqdm(range(num_training_steps),initial=epoch*ntrain_batches)

ntimepoints_per_batch = train_dataset.ntimepoints
valexample = next(iter(val_dataloader))

# train loop
for epoch in range(epoch, config['num_train_epochs']):
      
    model.train()
    tr_loss = torch.tensor(0.0).to(device)
    if train_dataset.discretize:
        tr_loss_discrete = torch.tensor(0.0).to(device)
        tr_loss_continuous = torch.tensor(0.0).to(device)
    
    nmask_train = 0
    for step, example in enumerate(train_dataloader):
    
        pred = model(example['input'].to(device=device), mask=train_src_mask, is_causal=is_causal)
        loss, loss_discrete, loss_continuous = criterion_wrapper(example, pred, criterion, train_dataset, config)
          
        loss.backward()
        
        # how many timepoints are in this batch for normalization
        if config['modeltype'] == 'mlm':
            nmask_train += torch.count_nonzero(example['mask'])
        else:
            nmask_train += example['input'].shape[0]*ntimepoints_per_batch 
    
        if step % config['niterplot'] == 0:
          
            with torch.no_grad():
                trainpred = model.output(example['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
                valpred = model.output(valexample['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
            update_debug_plots(hdebug['train'],config,model,train_dataset,example,trainpred,name='Train',criterion=criterion,**debug_params)
            update_debug_plots(hdebug['val'],config,model,val_dataset,valexample,valpred,name='Val',criterion=criterion,**debug_params)
            update_plots([hdebug['train']['figpose'],hdebug['train']['figtraj'],hdebug['train']['figstate'],
                          hdebug['val']['figpose'],hdebug['val']['figtraj'],hdebug['val']['figstate'],hloss['fig']])
    
        tr_loss_step = loss.detach()
        tr_loss += tr_loss_step
        if train_dataset.discretize:
            tr_loss_discrete += loss_discrete.detach()
            tr_loss_continuous += loss_continuous.detach()
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
        optimizer.step()
        lr_scheduler.step()
        model.zero_grad()
        
        # update progress bar
        stat = {'train loss': tr_loss.item()/nmask_train,'last val loss': last_val_loss,'epoch': epoch}
        if train_dataset.discretize:
            stat['train loss discrete'] = tr_loss_discrete.item()/nmask_train
            stat['train loss continuous'] = tr_loss_continuous.item()/nmask_train
        progress_bar.set_postfix(stat)
        progress_bar.update(1)
        
        # end of iteration loop
    
    # training epoch complete
    loss_epoch['train'][epoch] = tr_loss.item() / nmask_train
    if train_dataset.discretize:
        loss_epoch['train_discrete'][epoch] = tr_loss_discrete.item() / nmask_train
        loss_epoch['train_continuous'][epoch] = tr_loss_continuous.item() / nmask_train
    
    # compute validation loss after this epoch
    if val_dataset.discretize:
         loss_epoch['val'][epoch],loss_epoch['val_discrete'][epoch],loss_epoch['val_continuous'][epoch] = \
           compute_loss(model,val_dataloader,val_dataset,device,train_src_mask,criterion,config)
    else:
        loss_epoch['val'][epoch] = \
          compute_loss(model,val_dataloader,val_dataset,device,train_src_mask,criterion,config)
    last_val_loss = loss_epoch['val'][epoch].item()
    
    update_loss_plots(hloss, loss_epoch)
    update_plots([hdebug['train']['figpose'],hdebug['train']['figtraj'],hdebug['train']['figstate'],
                  hdebug['val']['figpose'],hdebug['val']['figtraj'],hdebug['val']['figstate'],hloss['fig']])
    
    if (epoch + 1) % config['save_epoch'] == 0:
        savefile = os.path.join(config['savedir'], f"fly{modeltype_str}_epoch{epoch + 1}_{savetime}.pth")
        print(f'Saving to file {savefile}')
        save_model(savefile, model,
                    lr_optimizer=optimizer,
                    scheduler=lr_scheduler,
                    loss=loss_epoch,
                    config=config)
    
    # rechunk the training data
    if np.mod(epoch+1,config['epochs_rechunk']) == 0:
        print(f'Rechunking data after epoch {epoch}')
        X = chunk_data(data,config['contextl'],reparamfun,**chunk_data_params)
      
        train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)
        print('New training data set created')

print('Done training')

# %%
savefile = os.path.join(config['savedir'], f"fly{modeltype_str}_epoch{epoch + 1}_{savetime}.pth")
print(f'Saving to file {savefile}')
save_model(savefile, model,
            lr_optimizer=optimizer,
            scheduler=lr_scheduler,
            loss=loss_epoch,
            config=config)
