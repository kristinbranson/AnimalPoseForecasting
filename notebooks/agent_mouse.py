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
# -

train_dataset, mouseids, pose, velocity, sensory = make_dataset(config, 'intrainfile', return_all=True)

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
train_args['optimizer_args']['lr'] = 0.00001
train_args['num_train_epochs'] = 400
model, best_model, loss_epoch = train(train_dataloader, val_dataloader, model, loss_epoch=loss_epoch_init, **train_args)

# +
# Plot the losses
idx = np.argmin(loss_epoch['val'])
print((idx, loss_epoch['val'][idx]))

plt.figure()
plt.plot(loss_epoch['train'])
plt.plot(loss_epoch['val'])
plt.plot(idx, loss_epoch['val'][idx], '.g')

# +
# model_file = "/groups/branson/home/eyjolfsdottire/data/flyllm/model_refactored_250307_fulldata.pkl"
# pickle.dump(model, open(model_file, "wb"))
# model = pickle.load(open(model_file, "rb"))

# +
session = train_dataset.sessions[1]
burn_in = config['contextl'] 

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

# +
plt.figure()

plt.imshow(sensory.operations[0].heightmap.map, cmap='gray')

last_frame = None
x, y = gt_track[0, :last_frame, :2].T
dx = np.diff(x)
dy = np.diff(y)
length_gt = (dx**2 + dy**2)**.5
plt.plot(x, y, '.', markersize=3)
x, y = pred_track[0, :last_frame, :2].T
dx = np.diff(x)
dy = np.diff(y)
length_pred = (dx**2 + dy**2)**.5
plt.plot(x, y, '.', markersize=3)
plt.plot(x[:burn_in], y[:burn_in], '.g', markersize=3)
plt.axis('equal')
plt.title(f"length gt = {length_gt.sum()}, length pred = {length_pred.sum()}", fontsize=8)
plt.show()

# plt.plot(length_gt)
# plt.plot(length_pred)
# plt.show()
# -





config

# +
op = train_dataset.labels['velocity'].operations[-1]


i = 2
n_bins = op.bin_centers.shape[-1]
counts = train_dataset.labels['velocity'].array[0, :, i*n_bins:(i+1)*n_bins].sum(0)
plt.plot(op.bin_centers[i], counts, '.')
plt.show()
# -



# +
maxval = train_dataset.inputs['sensory'].array[0].max()
minval = train_dataset.inputs['sensory'].array[0].min()

def to_color(data, minval, maxval):
    return (data - minval) / (maxval - minval)


# -

minval = -4
maxval = 4

# +
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.imshow(sensory.operations[0].heightmap.map, cmap='gray')
x, y = gt_track[0, :, :2].T
clr = to_color(gt_input[:, 3:], minval, maxval)
plt.scatter(x, y, s=10, c=clr)
plt.axis('equal')

plt.subplot(1,2,2)
plt.imshow(sensory.operations[0].heightmap.map, cmap='gray')
x, y = pred_track[0, :last_frame, :2].T
clr2 = to_color(pred_input[0, :, 3:].detach().cpu().numpy(), minval, maxval)
plt.scatter(x, y, s=10, c=clr2)
plt.axis('equal')
plt.show()

# +
hm = sensory.operations[0].heightmap.map
# plt.imshow(hm, cmap='gray')
# plt.colorbar()
plt.imshow(hm==1)

plt.figure()
sortd = np.sort(hm.flatten())
plt.plot(sortd)

print(sortd[0], sortd[len(sortd)//2])

0, 0.53, 1

# +
# TODO:

# Verify that in training my data doesn't jump across videos
#    Perhaps I need to use mouseids (even though they are all the same mouse but they are different trials)... yeah I think so
# Check my sensory features, ensure that they are being calculated correctly for simulated track
#    Looks like they are fine, however we just arent representing boundary well as the whisker value is always relative. 
#    Should try to add a feature for boundary proximity, or maybe add absolute wisker values (in addition to /instead of relative to body)
# Train for longer
# Check my bins, might want to use different bin parameters

# Could try training with zscoring rather than binning...
# -




