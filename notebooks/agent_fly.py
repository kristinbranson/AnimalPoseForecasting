# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: transformer
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
    
import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pickle

from apf.io import read_config
from apf.training import train
from apf.utils import function_args_from_config
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

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# %%
configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_20251007.json"
mode = 'test' # can toggle to 'train'/'test'
pretrained_modelfile = os.path.join('/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/llmnets',
                                    'predvel_20251007_20251002T000000_epoch200.pth')
restartmodelfile = None
debug_uselessdata=True

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
if mode == 'train':
    loadmodelfile = None
else:
    loadmodelfile = pretrained_modelfile
    
res = init_flyllm(configfile=configfile,mode=mode,restartmodelfile=restartmodelfile,
                loadmodelfile=loadmodelfile,debug_uselessdata=debug_uselessdata,
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

    model, best_model, loss_epoch = train(**train_args)

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
# where isdata?
isdata = ~np.all(np.isnan(pose.array),axis=-1)
plt.imshow(isdata,aspect='auto',interpolation='none')
agent_ids = np.nonzero(np.any(isdata,axis=1))[0]
start_frame = 1000
plt.xlabel('Frame')
plt.ylabel('Agent ID')

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

# +
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
