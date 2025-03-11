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

from flyllm.config import DEFAULTCONFIGFILE, posenames
from flyllm.features import featglobal, get_sensory_feature_idx
from flyllm.simulation import animate_pose

from experiments.flyllm import make_dataset, initialize_model_wrapper
# -

configfile = "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241125.json"
config = read_config(
    configfile,
    default_configfile=DEFAULTCONFIGFILE,
    posenames=posenames,
    featglobal=featglobal,
    get_sensory_feature_idx=get_sensory_feature_idx,
)

train_dataset, flyids, track, pose, velocity, sensory = make_dataset(config, 'intrainfile', return_all=True, debug=True)

val_dataset = make_dataset(config, 'invalfile', train_dataset, debug=True)

# Wrap into dataloader
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# Initialize the model
device = torch.device(config['device'])
model, criterion = initialize_model_wrapper(config, train_dataset, device)

# Train the model
train_args = function_args_from_config(config, train)
train_args['num_train_epochs'] = 100
init_loss_epoch = {}
model, best_model, loss_epoch = train(train_dataloader, val_dataloader, model, loss_epoch=init_loss_epoch, **train_args)

# +
# Plot the losses
idx = np.argmin(loss_epoch['val'])
print((idx, loss_epoch['val'][idx]))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(loss_epoch['train'])
plt.plot(loss_epoch['val'])
plt.plot(idx, loss_epoch['val'][idx], '.g')

plt.subplot(1, 3, 2)
plt.plot(loss_epoch['train_continuous'])
plt.plot(loss_epoch['val_continuous'])

plt.subplot(1, 3, 3)
plt.plot(loss_epoch['train_discrete'])
plt.plot(loss_epoch['val_discrete'])
plt.show()
# -

model_file = "/groups/branson/home/eyjolfsdottire/data/flyllm/model_refactored_250307_fulldata.pkl"
# pickle.dump(model, open(model_file, "wb"))
# model = pickle.load(open(model_file, "rb"))

gt_track, pred_track = simulate(
    dataset=train_dataset,
    model=model,
    track=track,
    pose=pose,
    flyids=flyids,
    track_len=4000 + config['contextl'] + 1,
    burn_in=config['contextl'],
    max_contextl=config['contextl'],
    agent_id = 0,
    start_frame = 512,
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

last_frame = 1000#None
x, y = gt_track[0, :last_frame, :, 0].T
plt.plot(x, y, '.', markersize=1)
x, y = pred_track[0, :last_frame, :, 0].T
plt.plot(x, y, '.', markersize=1)
plt.axis('equal')
plt.show()
# -

agent_id = 0
savevidfile = "/groups/branson/home/eyjolfsdottire/data/flyllm/animation_250307.gif"
ani = animate_pose({'Pred': pred_track.T.copy(), 'True': gt_track.T.copy()}, focusflies=[agent_id], savevidfile=savevidfile)
