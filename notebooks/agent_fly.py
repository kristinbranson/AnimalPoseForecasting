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

from flyllm.config import DEFAULTCONFIGFILE, posenames
from flyllm.features import featglobal, get_sensory_feature_idx
from flyllm.simulation import animate_pose

from experiments.flyllm import make_dataset
# -

configfile = "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241125.json"
config = read_config(
    configfile,
    default_configfile=DEFAULTCONFIGFILE,
    posenames=posenames,
    featglobal=featglobal,
    get_sensory_feature_idx=get_sensory_feature_idx,
)

# +
# Try: oddroot, bin_epsilon very large (to get evenly sized bins)

# config['discretize_epsilon'] = np.ones(5) * 1000

config['discreteidx'] = np.array([])
# -



train_dataset, flyids, track, pose, velocity, sensory = make_dataset(config, 'intrainfile', return_all=True, debug=False)

val_dataset = make_dataset(config, 'invalfile', train_dataset, debug=False)

# +
# for i in range(5):
#     plt.figure(figsize=(10, 3))
#     # centers = train_dataset.labels['velocity'].operations[-1].operations[0].bin_centers[i]
#     edges = train_dataset.labels['velocity'].operations[-1].operations[0].bin_edges[i]
#     centers = (edges[:-1] + edges[1:])/2
#     counts = train_dataset.labels['velocity'].array[:, :, i*25:(i+1)*25].reshape([-1, 25]).sum(0)
#     counts_val = val_dataset.labels['velocity'].array[:, :, i*25:(i+1)*25].reshape([-1, 25]).sum(0)
#     plt.plot(centers[:-1], counts[:-1], '.')
#     plt.plot(centers[:-1], counts_val[:-1], '.')
#     plt.show()
# -

# Wrap into dataloader
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# +
# Initialize the model
device = torch.device(config['device'])

# config['variational'] = True

model, criterion = initialize_model(config, train_dataset, device)
# -

# Train the model
train_args = function_args_from_config(config, train)
train_args['num_train_epochs'] = 500
# train_args['optimizer_args']['lr'] *= 10
init_loss_epoch = {}
model, best_model, loss_epoch = train(train_dataloader, val_dataloader, model, loss_epoch=init_loss_epoch, **train_args)
# train_args

# +
# Things I'd like to try:

# VAE
# - using a much smaller hidden state vector, and a multilayer decoder
# - assigning only a subset of the hidden state as variational 

# Other
# - can I have a state vector from which we take the argmax before continuing? 
# -
loss_epoch = init_loss_epoch

# +
# Plot the losses
idx = np.argmin(loss_epoch['val'])
print((idx, loss_epoch['val'][idx]))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(loss_epoch['train'])
plt.plot(loss_epoch['val'])
plt.plot(idx, loss_epoch['val'][idx], '.g')
plt.title('total loss')

plt.subplot(1, 3, 2)
plt.plot(loss_epoch['train_continuous'])
plt.plot(loss_epoch['val_continuous'])
plt.title('continuous loss')

plt.subplot(1, 3, 3)
# plt.plot(loss_epoch['train_discrete'])
# plt.plot(loss_epoch['val_discrete'])
# plt.title('discrete loss')
# plt.show()

# plt.figure()
plt.plot(loss_epoch['train'] - loss_epoch['train_continuous'])
plt.plot(loss_epoch['val'] - loss_epoch['val_continuous'])
plt.title('KLD')
plt.show()
# -

model_file = "/groups/branson/home/eyjolfsdottire/data/flyllm/model_continuous_250428.pkl"
pickle.dump(model, open(model_file, "wb"))
# model = pickle.load(open(model_file, "rb"))

agent_idx = 4
gt_track, pred_track = simulate(
    dataset=train_dataset,
    model=best_model,
    track=track,
    pose=pose,
    identities=flyids,
    track_len=1000 + config['contextl'] + 1,
    burn_in=config['contextl'],
    max_contextl=config['contextl'],
    agent_idx=agent_idx,
    start_frame=800,
)

# +
def plot_arena():
    ARENA_RADIUS_MM = 26.689
    n_pts = 1000
    theta = np.arange(0, np.pi*2, np.pi*2 / n_pts)
    x = np.cos(theta) * ARENA_RADIUS_MM
    y = np.sin(theta) * ARENA_RADIUS_MM
    plt.plot(x, y, '-', color=[.8, .8, .8])

plt.figure()
plot_arena()

last_frame = None
x, y = gt_track[agent_idx, :last_frame, :, 0].T
plt.plot(x, y, '.', markersize=3)
x, y = pred_track[agent_idx, :last_frame, :, 0].T
plt.plot(x, y, '.', markersize=1)
x, y = pred_track[agent_idx, 800:last_frame, :, 0].T
plt.plot(x, y, '.', markersize=1)
plt.axis('equal')
plt.show()
# -





train_dataset.labels['velocity'].operations
val_dataset.labels['velocity'].operations

# agent_id = 0
savevidfile = "/groups/branson/home/eyjolfsdottire/data/flyllm/animation_250428_continuous.gif"
ani = animate_pose({'Pred': pred_track.T.copy(), 'True': gt_track.T.copy()}, focusflies=[agent_idx], savevidfile=savevidfile)

model.encoder.encoder_dict
'velocity': Linear
'pose': ResNet1d
'wall_touch': ResNet1d
'other_flies_vision': ResNet2d
'other_flies_touch': ResNet1d


