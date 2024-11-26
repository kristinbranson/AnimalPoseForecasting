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

from flyllm.prepare import init_flyllm
from apf.utils import get_dct_matrix, compute_npad
from apf.data import chunk_data, interval_all, debug_less_data
from apf.models import (
    initialize_model,
    initialize_loss,
    generate_square_full_mask,
    sanity_check_temporal_dep,
    criterion_wrapper,
    update_loss_nepochs,
)
from apf.io import read_config, get_modeltype_str, load_and_filter_data, save_model, load_model, parse_modelfile
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

restartmodelfile = None
#restartmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flymulttimeglob_predposition_20240305_epoch130_20240825T122647.pth'
configfile = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241125.json'

# set to None if you want to use the full data
quickdebugdatafile = None

needtraindata = True
needvaldata = True
traindataprocess = 'chunk'
valdataprocess = 'chunk'
res = init_flyllm(configfile=configfile,mode='train',restartmodelfile=restartmodelfile,
                  quickdebugdatafile=quickdebugdatafile,needtraindata=needtraindata,needvaldata=needvaldata,
                  traindataprocess=traindataprocess,valdataprocess=valdataprocess)

# unpack outputs
config = res['config']
device = res['device']
data = res['data']
scale_perfly = res['scale_perfly']
valdata = res['valdata']
val_scale_perfly = res['val_scale_perfly']
X = res['X']
valX = res['valX']
npad = res['npad']
reparamfun = res['reparamfun']
val_reparamfun = res['val_reparamfun']
chunk_data_params = res['train_chunk_data_params']
train_dataset = res['train_dataset']
train_dataloader = res['train_dataloader']
val_dataset = res['val_dataset']
val_dataloader = res['val_dataloader']
dataset_params = res['dataset_params']
ntrain_batches = res['ntrain_batches']
nval_batches = res['nval_batches']
model = res['model']
criterion = res['criterion']
num_training_steps = res['num_training_steps']
optimizer = res['optimizer']
lr_scheduler = res['lr_scheduler']
loss_epoch = res['loss_epoch']
epoch = res['epoch']
modeltype_str = res['modeltype_str']
savetime = res['model_savetime']
train_src_mask = res['train_src_mask']
is_causal = res['is_causal']

train_dataset_params = {
    'input_noise_sigma': config['input_noise_sigma'],
}


# %%
# some helper functions for debugging

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


# %%
# profile example creation
import cProfile
import pstats

from flyllm.dataset import FlyMLMDataset

def profile_test():
    train_dataset_small = FlyMLMDataset(X[:min(len(X),100)],**train_dataset_params,**dataset_params)

if False:

    cProfile.run('profile_test()','profile_test.out')
    p = pstats.Stats('profile_test.out')
    p.strip_dirs().sort_stats('cumtime').print_stats(20)

if False:
    profile_test()

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
last_val_loss = loss_epoch['val'][epoch].item()
if np.isnan(last_val_loss):
    last_val_loss = None

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
        assert np.isnan(loss.item()) == False, f'loss is nan at step {step}, epoch {epoch}'
          
        loss.backward()
        
        for weights in model.parameters():
            assert torch.isnan(weights.grad).any() == False, f'nan in gradients at step {step}, epoch {epoch}'
        
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
