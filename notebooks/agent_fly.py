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
    
import os

# Only set non-interactive backend if not in Jupyter
import matplotlib
try:
    from IPython import get_ipython
    assert 'IPKernelApp' in get_ipython().config
    # Running in Jupyter/IPython, keep default backend
except:
    # Not in Jupyter, use non-interactive backend
    matplotlib.use('tkAgg')


import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from apf.io import read_config
from apf.training import train
from apf.utils import function_args_from_config
from apf.simulation import simulate
from apf.models import initialize_model

from flyllm.config import DEFAULTCONFIGFILE, posenames
from flyllm.features import featglobal, get_sensory_feature_idx
from flyllm.simulation import animate_pose
from flyllm.plotting import plot_arena
import time

from experiments.flyllm import make_dataset, make_diffusion_dataset, make_diffusion_dataset2
# -

configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_20251007.json"
config = read_config(
    configfile,
    default_configfile=DEFAULTCONFIGFILE,
    posenames=posenames,
    featglobal=featglobal,
    get_sensory_feature_idx=get_sensory_feature_idx,
)

config['modelstatetype'] = None
config['diffusion'] = True

# Load datasets
train_dataset, flyids, track, pose, velocity, sensory = make_diffusion_dataset2(config, 'intrainfile', return_all=True, debug=False)

val_dataset = make_diffusion_dataset2(config, 'invalfile', train_dataset, debug=True)

# Wrap into dataloader
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# +
# Initialize the model
device = torch.device(config['device'])

model, criterion = initialize_model(config, train_dataset, device)
# -

model.decoder.t_embeddings.embeddings = model.decoder.t_embeddings.embeddings.to(device)

# Train the model
train_args = function_args_from_config(config, train)
init_loss_epoch = {}
train_args['model_nickname'] = 'predbodycentricvel_diffusion_20251023_round2'
train_args['model_nickname'] = 'predbodycentric_diffusion_test'
train_args['num_train_epochs'] = 10
train_args['save_epoch'] = 100
train_args

model, best_model, loss_epoch, model_path = train(train_dataloader, val_dataloader, model, loss_epoch=init_loss_epoch, **train_args)

example = train_dataset[0]
example['labels'].shape

# +
# OR
model_path = 'flyllm_models/predbodycentricvel_diffusion_20251023_round2_20251027T062129_epoch160.pth'
model.load_state_dict(torch.load(model_path))

model.decoder.t_embeddings.embeddings = model.decoder.t_embeddings.embeddings.to(device)

# +
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
# -

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

train_dataset.sessions

# +
from apf.simulation import simulate_new

t0 = time.time()
# Look at train_dataset.sessions to select valid fly_ids and frames
agent_ids = [1, 2, 5, 6, 9]
# agent_ids = [1]
# start_frame = 117220
start_frame = 20001
gt_track, pred_track = simulate(
    dataset=train_dataset,
    model=model,
    track=track,
    pose=pose,
    identities=flyids,
    track_len=300 + config['contextl'] + 1,
    burn_in=config['contextl'],
    max_contextl=config['contextl'],
    agent_ids=agent_ids,
    start_frame=start_frame,
    debug_with_gt=False
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
plt.subplot(1,2,1)
plt.title('gt')
plt.subplot(1,2,2)
plt.title('pred')
plt.show()
# -

model_path

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
    {'Pred': pred_track[:, first_frame:].T.copy(), 'True': gt_track[:, first_frame:].T.copy()}, 
    focusflies=agent_ids, 
    savevidfile=savevidfile,
    contextl=config['contextl']
)


