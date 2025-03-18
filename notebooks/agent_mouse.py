# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
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

from flyllm.config import DEFAULTCONFIGFILE

from experiments.spatial_infomax import make_dataset

# +
configfile = '/groups/branson/home/eyjolfsdottire/data/mouse_experiment/config_mouse_20241008.json'

config = read_config(
    configfile,
    default_configfile=DEFAULTCONFIGFILE,
)
# config['discretize_epsilon'] = [0.2, 0.2, .01] # found this empirically for bin sums to look roughtly gaussian
# -

train_dataset, mouseids, pose, velocity, sensory = make_dataset(config, 'intrainfile', return_all=True)

n_bins = 25
plt.figure(figsize=(15 ,3))
labels = ['dfwd', 'dside', 'dang']
for i in range(3):
    plt.subplot(1,3,i+1)
    count = train_dataset.labels['velocity'].array[0, :, i*n_bins:(i+1)*n_bins].sum(0)
    bins = train_dataset.labels['velocity'].operations[-1].bin_centers[i]
    plt.plot(bins, count, '.')
    plt.title(labels[i])
plt.show()

val_dataset = make_dataset(config, 'invalfile', ref_dataset=train_dataset)

# Wrap into dataloader
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# Initialize the model
device = torch.device(config['device'])
model, criterion = initialize_model(config, train_dataset, device)
loss_epoch_init = {}

# Train the model
train_args = function_args_from_config(config, train)
train_args['optimizer_args']['lr'] = 0.0001
train_args['num_train_epochs'] = 200
model, best_model, loss_epoch = train(train_dataloader, val_dataloader, model, loss_epoch=loss_epoch_init, **train_args)

# +
# Plot the losses
idx = np.argmin(loss_epoch['val'])
print((idx, loss_epoch['val'][idx]))

plt.figure()
plt.plot(loss_epoch['train'])
plt.plot(loss_epoch['val'])
plt.plot(idx, loss_epoch['val'][idx], '.g')
# -

model_file = "/groups/branson/home/eyjolfsdottire/data/flyllm/model_refactored_250318_spatialinfomax.pkl"
# pickle.dump(model, open(model_file, "wb"))
model = pickle.load(open(model_file, "rb"))

# +
session = train_dataset.sessions[2]
burn_in = config['contextl'] 

tracks_relative = []
for i in range(100):
    print(i)
    gt_track, pred_track, gt_input, pred_input = simulate(
        dataset=train_dataset,
        model=model,
        track=pose,
        pose=pose,
        identities=mouseids,
        track_len=session.duration,
        burn_in=burn_in,
        max_contextl=config['contextl'],
        agent_idx=0,
        start_frame=session.start_frame,
    )
    tracks_relative.append(pred_track)

# +
plt.figure(figsize=(15, 15))
plt.imshow(sensory.operations[0].heightmap.map, cmap='gray')

for pred_track in tracks_relative:
    x, y = pred_track[0, :, :2].T
    plt.plot(x, y, '.', markersize=1)
    # plt.plot(x[:burn_in], y[:burn_in], '.g', markersize=3)
    
# x, y = gt_track[0, :, :2].T
# plt.plot(x, y, '.r', markersize=1)

plt.axis('equal')
plt.axis('off')
# plt.axis([-3000, 3000, 3000, -3000])
plt.show()

# +
plt.figure(figsize=(5, 5))
plt.imshow(sensory.operations[0].heightmap.map, cmap='gray')


pred_track = tracks_relative[0]

x, y = gt_track[0, :, :2].T
plt.plot(x, y, '.', markersize=3)
x, y = pred_track[0, :, :2].T
plt.plot(x, y, '.', markersize=3)

plt.axis('equal')
plt.axis('off')
plt.show()

# +
plt.figure()

plt.imshow(sensory.operations[0].heightmap.map, cmap='gray')

last_frame = None
x, y = gt_track[0, :last_frame, :2].T
plt.plot(x, y, '.', markersize=3)
x, y = pred_track[0, :last_frame, :2].T
plt.plot(x, y, '.', markersize=3)
plt.plot(x[:burn_in], y[:burn_in], '.g', markersize=3)
plt.axis('equal')
plt.show()

# +
# Important to get a good looking simulation:
# 1) Needed to use absolute heightvalues rather than relative
# 2) Needed to find the right bin_epsilon for feature distribution to look more gaussian (as opposed to flat with big ends)
