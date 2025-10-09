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
import time
import os

import apf
from apf.training import train
from apf.utils import function_args_from_config, set_mpl_backend
from apf.simulation import simulate
from apf.models import initialize_model
import matplotlib.pyplot as plt

import flyllm
from flyllm.config import read_config
from flyllm.features import featglobal, get_sensory_feature_idx
from flyllm.simulation import animate_pose

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
print(torch.cuda.memory_summary())


# %%
timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
print('Timestamp: ' + timestamp)

# %% [markdown]
# ### Set parameters

# %%
configfile = 'configs/config_fly_llm_predvel_20241125.json'
restartmodelfile = None
outfigdir = 'figs'
debug_uselessdata = True

# path to config file based on code directory
flyllmdir = flyllm.__path__[0]
configfile = os.path.join(flyllmdir,configfile)

# make directory if it doesn't exist
if not os.path.exists(outfigdir):
    os.makedirs(outfigdir)

# %% [markdown]
# ### Load configuration and data 

# %%
res = init_flyllm(configfile=configfile,mode='train',restartmodelfile=restartmodelfile,
                debug_uselessdata=debug_uselessdata)

for key in res:
    print(f'{key}: {type(res[key])}')

# unpack outputs
config = res['config']
device = res['device']
train_data = res['train_data']
val_data = res['val_data']
train_dataset = res['train_dataset']
train_dataloader = res['train_dataloader']
val_dataset = res['val_dataset']
val_dataloader = res['val_dataloader']
model = res['model']
criterion = res['criterion']
optimizer = res['optimizer']
lr_scheduler = res['lr_scheduler']
modeltype_str = res['modeltype_str']
loss_epoch = res['loss_epoch']
epoch = res['epoch']
modeltype_str = res['modeltype_str']
savetime = res['model_savetime']

train_dataset_params = {
    'input_noise_sigma': config['input_noise_sigma'],
}


# %% [markdown]
# ### some helper functions for debugging

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
# ### profile example creation

# %%
# import cProfile
# import pstats

# from flyllm.dataset import FlyMLMDataset

# def profile_test():
#     train_dataset_small = FlyMLMDataset(X[:min(len(X),100)],**train_dataset_params,**dataset_params)

# if False:

#     cProfile.run('profile_test()','profile_test.out')
#     p = pstats.Stats('profile_test.out')
#     p.strip_dirs().sort_stats('cumtime').print_stats(20)

# if False:
#     profile_test()

# %%
example_curr = train_dataset[0]
data_curr = train_dataset.item_to_data(example_curr)
data_curr['labels']['velocity'].array.shape
pose = apf.dataset.apply_inverse_operations(data_curr['labels']['velocity'])
print(pose.shape)
print(pose[0,0])
print(example_curr['metadata'].keys())
t0 = example_curr['metadata']['start_frame']
flynum = example_curr['metadata']['agent_id']


# %% [markdown]
# ### set up debug plots

# %%
from flyllm.plotting import initialize_debug_plots, initialize_loss_plots
debug_params = {}
# if contextl is long, still just look at samples from the first 64 frames
if config['contextl'] > 64:
    debug_params['tsplot'] = np.round(np.linspace(0,64,5)).astype(int)
    debug_params['traj_nsamplesplot'] = 1
hdebug = {}
hdebug['train'] = initialize_debug_plots(train_dataset, train_dataloader, train_data, name='Train', **debug_params)
hdebug['val'] = initialize_debug_plots(val_dataset, val_dataloader, val_data, name='Val', **debug_params)

hloss = initialize_loss_plots(loss_epoch)

progress_bar = tqdm.tqdm(range(num_training_steps),initial=epoch*ntrain_batches)

ntimepoints_per_batch = train_dataset.ntimepoints
valexample = next(iter(val_dataloader))
last_val_loss = loss_epoch['val'][epoch].item()
if np.isnan(last_val_loss):
    last_val_loss = None


# %% [markdown]
# ### Train

# %%

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
