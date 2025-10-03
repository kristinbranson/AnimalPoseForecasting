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
from apf.utils import function_args_from_config, set_mpl_backend
set_mpl_backend('tkAgg')
from apf.simulation import simulate
from apf.models import initialize_model

from flyllm.config import DEFAULTCONFIGFILE, posenames
from flyllm.features import featglobal, get_sensory_feature_idx
from flyllm.simulation import animate_pose
import time

from experiments.flyllm import make_dataset

# %%
timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())

# %%
configfile = "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241125.json"
config = read_config(
    configfile,
    default_configfile=DEFAULTCONFIGFILE,
    posenames=posenames,
    featglobal=featglobal,
    get_sensory_feature_idx=get_sensory_feature_idx,
)

# %%
train_dataset, flyids, track, pose, velocity, sensory = make_dataset(config, 'intrainfile', return_all=True, debug=True)

# %%
val_dataset = make_dataset(config, 'invalfile', train_dataset, debug=True)

# %%
# Wrap into dataloader
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# %%
# Initialize the model
device = torch.device(config['device'])
model, criterion = initialize_model(config, train_dataset, device)

# %%
# Train the model
train_args = function_args_from_config(config, train)
train_args['num_train_epochs'] = 100
init_loss_epoch = {}
model, best_model, loss_epoch = train(train_dataloader, val_dataloader, model, loss_epoch=init_loss_epoch, **train_args)

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

# %% [markdown]
# model_file = f'agentfly_model_{timestamp}.pkl'
# pickle.dump(model, open(model_file, "wb"))
# OR
# model_file = 'agentfly_model_20251001T073751.pkl'
# model = pickle.load(open(model_file, "rb"))

# %%
gt_track, pred_track = simulate(
    dataset=train_dataset,
    model=model,
    track=track,
    pose=pose,
    identities=flyids,
    track_len=4000 + config['contextl'] + 1,
    burn_in=config['contextl'],
    max_contextl=config['contextl'],
    agent_idx=0,
    start_frame=512,
)

# %%
def plot_arena():
    ARENA_RADIUS_MM = 26.689
    n_pts = 1000
    theta = np.arange(0, np.pi*2, np.pi*2 / n_pts)
    x = np.cos(theta) * ARENA_RADIUS_MM
    y = np.sin(theta) * ARENA_RADIUS_MM
    plt.plot(x, y, '-', color=[.8, .8, .8])

plt.figure()
plot_arena()

last_frame = 1000#None
x, y = gt_track[0, :last_frame, :, 0].T
plt.plot(x, y, '.', markersize=1)
x, y = pred_track[0, :last_frame, :, 0].T
plt.plot(x, y, '.', markersize=1)
plt.axis('equal')
plt.show()

# %%
agent_id = 0
savevidfile = f"animation_{timestamp}.gif"
ani = animate_pose({'Pred': pred_track.T.copy(), 'True': gt_track.T.copy()}, focusflies=[agent_id], savevidfile=savevidfile)
