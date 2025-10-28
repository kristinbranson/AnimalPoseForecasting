# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
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
from apf.utils import function_args_from_config, set_mpl_backend, is_notebook
from apf.simulation import simulate
from apf.models import initialize_model
import matplotlib.pyplot as plt

import flyllm
from flyllm.config import read_config
from flyllm.features import featglobal, get_sensory_feature_idx
from flyllm.simulation import animate_pose

from flyllm.prepare import init_flyllm
from flyllm.plotting import initialize_debug_plots, initialize_loss_plots

import logging
import tqdm
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

set_mpl_backend('tkAgg')
ISNOTEBOOK = is_notebook()
if ISNOTEBOOK:
    from IPython.display import HTML, display, clear_output
    plt.ioff()
else:
    plt.ion()

LOG.info('CUDA available: ' + str(torch.cuda.is_available()))
LOG.info('isnotebook: ' + str(ISNOTEBOOK))

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

ntrain_batches = len(train_dataloader)
num_training_steps = ntrain_batches * config['num_train_epochs']
valexample = next(iter(val_dataloader))
ntimepoints_per_batch = valexample['input'].shape[0]
last_val_loss = loss_epoch['val'][epoch].item()
if np.isnan(last_val_loss):
    last_val_loss = None


# %% [markdown]
# ### some helper functions for profiling
#

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
# random debugging stuff

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
debug_params = {}
# if contextl is long, still just look at samples from the first 64 frames
if config['contextl'] > 64:
    debug_params['tsplot'] = np.round(np.linspace(0,64,5)).astype(int)
    debug_params['traj_nsamplesplot'] = 1
hdebug = {}
hdebug['train'] = initialize_debug_plots(train_dataset, train_dataloader, train_data, name='Train', **debug_params)
hdebug['val'] = initialize_debug_plots(val_dataset, val_dataloader, val_data, name='Val', **debug_params)
hloss = initialize_loss_plots(loss_epoch)

def refresh_plots(hdebug):
    
    if ISNOTEBOOK:
        if 'display_handles' not in hdebug:
            hdebug['display_handles'] = {}
        for k,fig in hdebug.items():
            if not k.startswith('fig') or fig is None:
                continue 
            if k in hdebug['display_handles']:
                hdebug['display_handles'][k].update(fig)
            else:
                hdebug['display_handles'][k] = display(fig,display_id=k)

    else:
        for k,fig in hdebug.items():
            if not k.startswith('fig') or fig is None:
                continue 
            fig.canvas.draw()
            fig.canvas.flush_events()

if ISNOTEBOOK:
    refresh_plots(hdebug['train'])
    refresh_plots(hdebug['val'])
    refresh_plots(hloss)
else:
    plt.ion()
    plt.show(block=False)



# %%
# functions for plotting visualizations of results during training

from flyllm.plotting import update_debug_plots, update_loss_plots

def end_iter_hook(model=None, step=None, example=None, predfn=None, **kwargs):
    
    assert step is not None
    
    if step % config['niterplot'] != 0:
        return

    assert model is not None
    assert example is not None
    assert predfn is not None

    LOG.info(f'Updating debug plots at step {step}')

    with torch.no_grad():
        trainpred = predfn(example['input'].to(device=device))
        valpred = predfn(valexample['input'].to(device=device))
    update_debug_plots(hdebug['train'],config,model,train_dataset,example,trainpred,name='Train',criterion=criterion,**debug_params)
    update_debug_plots(hdebug['val'],config,model,val_dataset,valexample,valpred,name='Val',criterion=criterion,**debug_params)
    refresh_plots(hdebug['train'])
    refresh_plots(hdebug['val'])
    return

def end_epoch_hook(loss_epoch=None, **kwargs):
    assert loss_epoch is not None
    LOG.info(f'Updating loss plots at end of epoch {epoch}')
    update_loss_plots(hloss, loss_epoch)
    refresh_plots(hloss)
    return

# test the hooks

# trainexample = next(iter(train_dataloader))
# valexample = next(iter(val_dataloader))
# contextl = trainexample['input'].shape[1]
# train_src_mask = torch.nn.Transformer.generate_square_subsequent_mask(contextl, device=device)

# end_iter_hook(model=model,step=0,example=trainexample,predfn=lambda input: model.output(input, mask=train_src_mask, is_causal=True))
# end_epoch_hook(loss_epoch=loss_epoch)


# %%
# clean up memory allocation before training, particularly if running in a notebook
# and things have crashed before...
if device.type == 'cuda':
    import gc
    model = model.to(device='cpu')
    gc.collect()
    torch.cuda.empty_cache()
    model = model.to(device=device)
    memalloc = torch.cuda.memory_allocated() / 1e9
    print(f'Initial cuda memory allocated: {memalloc:.3f} GB')
    memreserved = torch.cuda.memory_reserved() / 1e9
    print(f'Initial cuda memory reserved: {memreserved:.3f} GB')
    print(torch.cuda.memory_summary())

savefilestr = os.path.join(config['savedir'], f"fly{modeltype_str}_{savetime}")

train_args = function_args_from_config(config,train)
train_args['train_dataloader'] = train_dataloader
train_args['val_dataloader'] = val_dataloader
train_args['model'] = model
train_args['loss_epoch'] = loss_epoch
train_args['end_epoch_hook'] = end_epoch_hook
train_args['end_iter_hook'] = end_iter_hook
train_args['optimizer'] = optimizer
train_args['lr_scheduler'] = lr_scheduler
# criterion hard-coded to mixed_causal_criterion
#train_args['criterion'] = criterion
train_args['start_epoch'] = epoch
train_args['savefilestr'] = savefilestr

# can override args here
#train_args['num_train_epochs'] = 100
model, best_model, loss_epoch = train(**train_args)

# %%
# old train code

# progress_bar = tqdm.tqdm(range(num_training_steps),initial=epoch*ntrain_batches)


# # train loop
# for epoch in range(epoch, config['num_train_epochs']):

#     model.train()
#     tr_loss = torch.tensor(0.0).to(device)
#     if train_dataset.discretize:
#         tr_loss_discrete = torch.tensor(0.0).to(device)
#         tr_loss_continuous = torch.tensor(0.0).to(device)
    
#     nmask_train = 0
#     for step, example in enumerate(train_dataloader):
    
#         pred = model(example['input'].to(device=device), mask=train_src_mask, is_causal=is_causal)
#         loss, loss_discrete, loss_continuous = criterion_wrapper(example, pred, criterion, train_dataset, config)
#         assert np.isnan(loss.item()) == False, f'loss is nan at step {step}, epoch {epoch}'
        
#         loss.backward()
        
#         for weights in model.parameters():
#             assert torch.isnan(weights.grad).any() == False, f'nan in gradients at step {step}, epoch {epoch}'
        
#         # how many timepoints are in this batch for normalization
#         if config['modeltype'] == 'mlm':
#             nmask_train += torch.count_nonzero(example['mask'])
#         else:
#             nmask_train += example['input'].shape[0]*ntimepoints_per_batch 
    
#         if step % config['niterplot'] == 0:
        
#             with torch.no_grad():
#                 trainpred = model.output(example['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
#                 valpred = model.output(valexample['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
#             update_debug_plots(hdebug['train'],config,model,train_dataset,example,trainpred,name='Train',criterion=criterion,**debug_params)
#             update_debug_plots(hdebug['val'],config,model,val_dataset,valexample,valpred,name='Val',criterion=criterion,**debug_params)
#             refresh_plots(hdebug)
    
#         tr_loss_step = loss.detach()
#         tr_loss += tr_loss_step
#         if train_dataset.discretize:
#             tr_loss_discrete += loss_discrete.detach()
#             tr_loss_continuous += loss_continuous.detach()
    
#         # gradient clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
#         optimizer.step()
#         lr_scheduler.step()
#         model.zero_grad()
        
#         # update progress bar
#         stat = {'train loss': tr_loss.item()/nmask_train,'last val loss': last_val_loss,'epoch': epoch}
#         if train_dataset.discretize:
#             stat['train loss discrete'] = tr_loss_discrete.item()/nmask_train
#             stat['train loss continuous'] = tr_loss_continuous.item()/nmask_train
#         progress_bar.set_postfix(stat)
#         progress_bar.update(1)
        
#         # end of iteration loop
    
#     # training epoch complete
#     loss_epoch['train'][epoch] = tr_loss.item() / nmask_train
#     if train_dataset.discretize:
#         loss_epoch['train_discrete'][epoch] = tr_loss_discrete.item() / nmask_train
#         loss_epoch['train_continuous'][epoch] = tr_loss_continuous.item() / nmask_train
    
#     # compute validation loss after this epoch
#     if val_dataset.discretize:
#         loss_epoch['val'][epoch],loss_epoch['val_discrete'][epoch],loss_epoch['val_continuous'][epoch] = \
#             compute_loss(model,val_dataloader,val_dataset,device,train_src_mask,criterion,config)
#     else:
#         loss_epoch['val'][epoch] = \
#             compute_loss(model,val_dataloader,val_dataset,device,train_src_mask,criterion,config)
#     last_val_loss = loss_epoch['val'][epoch].item()
    
#     update_loss_plots(hloss, loss_epoch)
#     refresh_plots(hdebug|{'loss': hloss})
    
#     if (epoch + 1) % config['save_epoch'] == 0:
#         savefile = os.path.join(config['savedir'], f"fly{modeltype_str}_epoch{epoch + 1}_{savetime}.pth")
#         print(f'Saving to file {savefile}')
#         save_model(savefile, model,
#                     lr_optimizer=optimizer,
#                     scheduler=lr_scheduler,
#                     loss=loss_epoch,
#                     config=config)
    
#     # rechunk the training data
#     if np.mod(epoch+1,config['epochs_rechunk']) == 0:
#         print(f'Rechunking data after epoch {epoch}')
#         X = chunk_data(data,config['contextl'],reparamfun,**chunk_data_params)

#         train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)
#         print('New training data set created')

# print('Done training')

# %%
# should be in train function

# savefile = os.path.join(config['savedir'], f"fly{modeltype_str}_epoch{epoch + 1}_{savetime}.pth")
# print(f'Saving to file {savefile}')
# save_model(savefile, model,
#             lr_optimizer=optimizer,
#             scheduler=lr_scheduler,
#             loss=loss_epoch,
#             config=config)
