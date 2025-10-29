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
if False:
    # modernize model file
    from apf.io import modernize_model_file
    configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_20251007.json"

    res = init_flyllm(configfile=configfile,needtraindata=True,needvaldata=False,debug_uselessdata=False,mode='train')

    eyrun_modelfile = '/groups/branson/home/eyjolfsdottire/data/flyllm/model_refactored_251002_newdata_cont_cont.pth'
    modelfile = os.path.join('/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/llmnets',
                            'predvel_20251007_20251002T000000_epoch200.pth')
    state = modernize_model_file(eyrun_modelfile,res['train_dataset'],res['config'],res['device'])

    torch.save(state, modelfile)

# %%
configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_20251007.json"
mode = 'train' # can toggle to 'test'

restartmodelfile = None
loadmodelfile = '/groups/branson/home/eyjolfsdottire/data/flyllm/model_refactored_251002_newdata_cont_cont.pth'
res = init_flyllm(configfile=configfile,mode=mode,restartmodelfile=restartmodelfile,
                loadmodelfile=loadmodelfile,debug_uselessdata=True)


# config = read_config(
#     configfile,
#     default_configfile=DEFAULTCONFIGFILE,
#     posenames=posenames,
#     featglobal=featglobal,
#     get_sensory_feature_idx=get_sensory_feature_idx,
# )

# # Load datasets
# train_dataset, flyids, track, pose, velocity, sensory, dataset_params = make_dataset(config, 'intrainfile', return_all=True, debug=False)

# val_dataset = make_dataset(config, 'invalfile', train_dataset, debug=False)

# # Wrap into dataloader
# train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
# val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# # Initialize the model
# device = torch.device(config['device'])
# model, criterion = initialize_model(config, train_dataset, device)

# # Train the model
# train_args = function_args_from_config(config, train)
# init_loss_epoch = {}
# model, best_model, loss_epoch = train(train_dataloader, val_dataloader, model, loss_epoch=init_loss_epoch, **train_args)

# # OR
# # model_file = 'agentfly_model_20251001T073751.pth'
# # model.load_state_dict(torch.load(model_file))

# %%
# Plot the losses
idx = torch.argmin(loss_epoch['val']).item()
print((idx, loss_epoch['val'][idx].item()))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(loss_epoch['train'],label='train')
plt.plot(loss_epoch['val'],label='val')
plt.plot(idx, loss_epoch['val'][idx], 'go')
plt.legend()
plt.title('Total loss')

plt.subplot(1, 3, 2)
plt.plot(loss_epoch['train_continuous'],label='train')
plt.plot(loss_epoch['val_continuous'],label='val')
plt.plot(idx, loss_epoch['val_continuous'][idx], 'go')
plt.legend()
plt.title('Continuous loss')

plt.subplot(1, 3, 3)
plt.plot(loss_epoch['train_discrete'],label='train')
plt.plot(loss_epoch['val_discrete'],label='val')
plt.plot(idx, loss_epoch['val_discrete'][idx], 'go')
plt.legend()
plt.title('Discrete loss')
plt.show()
# %%

t0 = time.time()
# Look at train_dataset.sessions to select valid fly_ids and frames
agent_ids = [1, 2, 5, 6, 9]
start_frame = 117220
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
if model_path is not None:
    modelname = os.path.split(model_path)[-1].replace('.pth', '')
    savevidfile = os.path.join(savedir, f"animation_{modelname}.gif")
else:
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    savevidfile = os.path.join(savedir, f"animation_{timestamp}.gif")

ani = animate_pose(
    {'Pred': pred_track.T.copy(), 'True': gt_track.T.copy()}, 
    focusflies=agent_ids, 
    savevidfile=savevidfile,
    contextl=config['contextl']
)
