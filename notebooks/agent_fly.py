# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# # TODO today:

# # Something is off with the normalized keypoints, look at Kristin's new data and verify the order in which the keys are

# Compare pose diffusion with L1 vs L2 norm loss functions
# Compare pose diffusion with [64, 128, 256] MPL with [512, 256, 128, 64] MLP (should I try some other variants too?)



# Make a config that has contextlen 64, 6 layers, no convolution on the sensory inputs (or no sensory inputs)
# Can it learn something reasonable in a short time, like 1 hour?
# Once it does, try training it without binning
# Once that works, try training it with diffusion
#     Try it on the velocity features?
#     Try it by always adding a lot of noise (it should learn to ignore those noisy inputs)
#     Try by gradually allowing more noise in later phases

# Once I find a regime that seems to be working:
#     Try swapping out pose velocity for pose 
#     Try swapping out pose for body centric keypoints

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
    
import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pickle

from apf.io import read_config
from apf.training import train
from apf.utils import function_args_from_config, set_mpl_backend, is_notebook
from apf.simulation import simulate
from apf.models import initialize_model

from flyllm.config import DEFAULTCONFIGFILE, posenames
from flyllm.features import featglobal, get_sensory_feature_idx
from flyllm.simulation import animate_pose
from flyllm.plotting import plot_arena
from flyllm.prepare import init_flyllm
import time
import logging
import os

set_mpl_backend('tkAgg')
ISNOTEBOOK = is_notebook()
if ISNOTEBOOK:
    from IPython.display import HTML, display, clear_output
    plt.ioff()
else:
    plt.ion()

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# %%
# configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_20251007.json"
configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_simpler_20251104.json"

mode = 'train' # can toggle to 'train'/'test'
pretrained_modelfile = os.path.join('/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/llmnets',
                                    'predvel_20251007_20251002T000000_epoch200.pth')
restartmodelfile = None
debug_uselessdata=False

# %%
# # modernize model file
# from apf.io import modernize_model_file
# res = init_flyllm(configfile=configfile,needtraindata=True,needvaldata=False,debug_uselessdata=False,mode='train')

# eyrun_modelfile = '/groups/branson/home/eyjolfsdottire/data/flyllm/model_refactored_251002_newdata_cont_cont.pth'
# modelfile = os.path.join('/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/llmnets',
#                         'predvel_20251007_20251002T000000_epoch200.pth')
# state = modernize_model_file(eyrun_modelfile,res['train_dataset'],res['config'],res['device'])

# torch.save(state, modelfile)

# %%
import time
t0 = time.time()

if mode == 'train':
    loadmodelfile = None
else:
    loadmodelfile = pretrained_modelfile
    
res = init_flyllm(configfile=configfile, mode=mode, restartmodelfile=restartmodelfile,
                  loadmodelfile=loadmodelfile, debug_uselessdata=debug_uselessdata,
                  needtraindata=True)

# unpack the results
config = res['config']
if 'train_dataset' in res:
    train_dataset = res['train_dataset']
if 'train_dataloader' in res:
    train_dataloader = res['train_dataloader']
if 'train_data' in res:
    flyids = res['train_data']['flyids']
    track = res['train_data']['track']
    pose = res['train_data']['pose']
    velocity = res['train_data']['velocity']
    sensory = res['train_data']['sensory']
if 'val_dataset' in res:
    val_dataset = res['val_dataset']
if 'val_dataloader' in res:
    val_dataloader = res['val_dataloader']
criterion = res['criterion']
model = res['model']
optimizer = res['optimizer']
lr_scheduler = res['lr_scheduler']
loss_epoch = res['loss_epoch']
start_epoch = res['epoch']
modeltype_str = res['modeltype_str']
device = res['device']
savetime = res['model_savetime']

time.time() - t0


# %%
# Time to load the full dataset: 506 seconds

# %%
# train
if mode == 'train':
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
    
    savefilestr = os.path.join(config['savedir'], f"fly{modeltype_str}_{savetime}")
    
    train_args = function_args_from_config(config,train)
    train_args['train_dataloader'] = train_dataloader
    train_args['val_dataloader'] = val_dataloader
    train_args['model'] = model
    train_args['loss_epoch'] = loss_epoch
    train_args['optimizer'] = optimizer
    train_args['lr_scheduler'] = lr_scheduler
    # criterion hard-coded to mixed_causal_criterion
    #train_args['criterion'] = criterion
    train_args['start_epoch'] = start_epoch
    train_args['savefilestr'] = savefilestr

# %%
from flyllm.plotting import update_debug_plots, update_loss_plots
from flyllm.plotting import initialize_debug_plots, initialize_loss_plots

valexample = next(iter(val_dataloader))

debug_params = {}
# if contextl is long, still just look at samples from the first 64 frames
if config['contextl'] > 64:
    debug_params['tsplot'] = np.round(np.linspace(0,64,5)).astype(int)
    debug_params['traj_nsamplesplot'] = 1
hdebug = {}
hdebug['train'] = initialize_debug_plots(train_dataset, train_dataloader, res['train_data'], name='Train', **debug_params)
# hdebug['val'] = initialize_debug_plots(val_dataset, val_dataloader, res['train_data'], name='Val', **debug_params)

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
    # refresh_plots(hdebug['val'])
    refresh_plots(hloss)
else:
    plt.ion()
    plt.show(block=False)


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
        # valpred = predfn(valexample['input'].to(device=device))
    update_debug_plots(hdebug['train'],config,model,train_dataset,example,trainpred,name='Train',criterion=criterion,**debug_params)
    # update_debug_plots(hdebug['val'],config,model,val_dataset,valexample,valpred,name='Val',criterion=criterion,**debug_params)
    refresh_plots(hdebug['train'])
    # refresh_plots(hdebug['val'])
    return

def end_epoch_hook(loss_epoch=None, epoch=None, **kwargs):
    assert loss_epoch is not None
    LOG.info(f'Updating loss plots at end of epoch {epoch}')
    update_loss_plots(hloss, loss_epoch)
    refresh_plots(hloss)
    return

train_args['end_epoch_hook'] = end_epoch_hook
train_args['end_iter_hook'] = end_iter_hook

# %%
# train_args['num_train_epochs'] = 10
train_args['optimizer_args']['lr'] = 0.0005
train_args['optimizer_args']['lr']

# %%
train_args['savefilestr'] += '_10x_rate'
train_args['savefilestr']

# %%
# from apf.models import initialize_model
# model, criterion = initialize_model(config, train_dataset, device)
# train_args['model'] = model

# %%
# model_file = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_simple_2025111"
# checkpoint = torch.load(model_file, map_location='cuda', weights_only=False)
# model.load_state_dict(checkpoint['model'])

# %%
import time
t0 = time.time()
out = train(**train_args)
# model, best_model, loss_epoch = out
time.time() - t0

# %%

# %%
204 / 60 * 20 # one hour on this smaller data

1:15 hour: so total would > 20 hours

- increase batch size by 4x
1:33 hour: this means that its actually slower... maybe going to a beefier machine would be better here

- skip the sensory features
(did  make a difference)

- on an h100 machine
1 hour (not much faster)
44 minutes with a 4x batch size
27 minutes if I skip flipping

# 44 minutes with reduced data (V3) # this was doing the same since I hadn't filtered, however, filtering didn't save much data
25 minutes for 10 epochs with filtering on movie ids
    1 hour for 30 epochs, lets see if that gets us as much juice as we hope
    next up, train this with visualiation hooks

# %%
tmp = 0

# %%
# Plot the losses
if loss_epoch['val'] is not None:
    idx = torch.argmin(loss_epoch['val']).item()
    print((idx, loss_epoch['val'][idx].item()))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
if loss_epoch['train'] is not None:
    plt.plot(loss_epoch['train'],label='train')
if loss_epoch['val'] is not None:
    plt.plot(loss_epoch['val'],label='val')
    plt.plot(idx, loss_epoch['val'][idx], 'go')
plt.legend()
plt.title('Total loss')

plt.subplot(1, 3, 2)
if 'train_continuous' in loss_epoch and loss_epoch['train_continuous'] is not None:
    plt.plot(loss_epoch['train_continuous'],label='train')
if 'val_continuous' in loss_epoch and loss_epoch['val_continuous'] is not None:
    plt.plot(loss_epoch['val_continuous'],label='val')
    plt.plot(idx, loss_epoch['val_continuous'][idx], 'go')
plt.legend()
plt.title('Continuous loss')

plt.subplot(1, 3, 3)
if 'train_discrete' in loss_epoch and loss_epoch['train_discrete'] is not None:
    plt.plot(loss_epoch['train_discrete'],label='train')
if 'val_discrete' in loss_epoch and loss_epoch['val_discrete'] is not None:
    plt.plot(loss_epoch['val_discrete'],label='val')
    plt.plot(idx, loss_epoch['val_discrete'][idx], 'go')
plt.legend()
plt.title('Discrete loss')
plt.show()
# %%

# %%

# %%
model_file = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_simple_20251119T080019_epoch20.pth"
checkpoint = torch.load(model_file, map_location='cuda', weights_only=False)
model.load_state_dict(checkpoint['model'])

# %%

# %%
# where isdata?
isdata = ~np.all(np.isnan(pose.array),axis=-1)
plt.imshow(isdata,aspect='auto',interpolation='none')
agent_ids = np.nonzero(np.any(isdata,axis=1))[0]
start_frame = 1000
plt.xlabel('Frame')
plt.ylabel('Agent ID')

# %%
train_dataset.sessions
start_frame = 45004
agent_ids = [0, 1, 4, 6]

# %%
# simulate

t0 = time.time()
gt_track, pred_track = simulate(
    dataset=train_dataset,
    model=model,
    track=track,
    pose=pose,
    identities=flyids,
    track_len=3000 + config['contextl'] + 1,
    burn_in=config['contextl'],
    max_contextl=config['contextl'],
    agent_ids=agent_ids,
    start_frame=start_frame,
)
time.time() - t0
# %%
plt.figure(figsize=(10, 5))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plot_arena()

first_frame = config['contextl']
last_frame = None
for agent_id in agent_ids:
    plt.subplot(1, 2, 1)
    x, y = gt_track[agent_id, first_frame:last_frame, :, 0].T
    plt.plot(x, y, '.', markersize=1)
    plt.axis('equal')
    plt.subplot(1, 2, 2)
    x, y = pred_track[agent_id, first_frame:last_frame, :, 0].T
    plt.plot(x, y, '.', markersize=1)
    plt.axis('equal')
plt.show()

# %%

savedir = "flyllm_animations"
if not os.path.exists(savedir):
    os.makedirs(savedir)
if loadmodelfile is not None:
    model_path = loadmodelfile
else:
    model_path = savefilestr
modelname = os.path.split(model_path)[-1].replace('.pth', '')
savevidfile = os.path.join(savedir, f"animation_{modelname}.gif")

ani = animate_pose(
    {'Pred': pred_track.T.copy(), 'True': gt_track.T.copy()}, 
    focusflies=agent_ids, 
    savevidfile=savevidfile,
    contextl=config['contextl']
)

# %%
1+2

# %%
n_flies -1
T -2

[_, _, T, n_flies]

# %%
track = res['train_data']['track'].array.shape

# %%
# # animate_pose??

# %%
