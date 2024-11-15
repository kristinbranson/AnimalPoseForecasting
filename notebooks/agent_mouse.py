# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
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

from apf.io import read_config
from apf.dataset import Zscore, Data, Dataset
from apf.training import to_dataloader, init_model, train
from apf.simulation import get_motion_track, get_pred_motion_track, simulate
from apf.utils import function_args_from_config
from apf.models import TransformerModel

from flyllm.config import DEFAULTCONFIGFILE
from spatial_infomax.utils.data_loader import HeightMap

from experiments.spatial_infomax import load_data, compute_global_movement, compute_observation

# +
configfile = '/groups/branson/home/eyjolfsdottire/data/mouse_experiment/config_mouse_20241008.json'

config = read_config(
    configfile,
    default_configfile=DEFAULTCONFIGFILE,
)
# -

# Load data
position, observation, isstart = load_data(config['intrainfile'])

# Compute motion features
tspred = 1
global_motion = compute_global_movement(position, dt=tspred, isstart=isstart)

# +
# Wrap into data class with relevant data operations
motion_zscore = Zscore()

# Discrete output
# motion_data_labels = Data(raw=global_motion.T, operation=Multi([motion_zscore, Discretize()]))

# Continuous output
motion_data_labels = Data(raw=global_motion.T, operation=motion_zscore)

motion_data_input = Data(raw=np.roll(global_motion.T, shift=tspred, axis=-2), operation=motion_zscore)
observation_data = Data(raw=observation.T, operation=Zscore())
# -

# Wrap into dataset
train_dataset = Dataset(
    inputs=[motion_data_input, observation_data],
    labels=[motion_data_labels],
    isstart=isstart,
    context_length=config['contextl'],
)

# Now do the same for validation data and use the dataset operations from the training data
val_position, val_observation, val_isstart = load_data(config['invalfile'])
val_global_motion = compute_global_movement(val_position,  dt=tspred, isstart=val_isstart)
val_motion_data_labels = Data(raw=val_global_motion.T, operation=motion_data_labels.operation)
val_motion_data_input = Data(raw=np.roll(val_global_motion.T, shift=tspred, axis=1), operation=motion_data_input.operation)
val_observation_data = Data(raw=val_observation.T, operation=observation_data.operation)
val_dataset = Dataset(
    inputs=[val_motion_data_input, val_observation_data],
    labels=[val_motion_data_labels],
    isstart=val_isstart,
    context_length=config['contextl'],
)

# Map to dataloaders
device = torch.device(config['device'])
batch_size = config['batch_size']
train_dataloader = to_dataloader(train_dataset, device, batch_size, shuffle=True)
val_dataloader = to_dataloader(val_dataset, device, batch_size, shuffle=False)



# Initialize the model
model_args = function_args_from_config(config, TransformerModel)
model = init_model(train_dataset, model_args).to(device)

# Train the model
train_args = function_args_from_config(config, train)

train_args['num_train_epochs'] = 100

model, best_model, loss_epoch = train(train_dataloader, val_dataloader, model, **train_args)

# Plot the losses
idx = np.argmin(loss_epoch['val'])
print((idx, loss_epoch['val'][idx]))
plt.figure()
plt.plot(loss_epoch['train'])
plt.plot(loss_epoch['val'])
plt.plot(idx, loss_epoch['val'][idx], '.g')
plt.show()



# +
agent_id=0
t0=500
track_len=1000
bg_img=HeightMap().map

dataset = val_dataset

gt_track = val_position[:, t0:t0 + track_len, agent_id]
gt_chunk = dataset.get_chunk(start_frame=t0, duration=track_len, agent_id=agent_id)
if dataset.n_bins is None:
    gt_labels = gt_chunk['labels']
else:
    gt_labels = gt_chunk['labels_discrete'].reshape((-1, dataset.d_output_discrete, dataset.n_bins))
# -

motion_track = get_motion_track(
    labels_operation=val_motion_data_labels.operation,
    motion_labels=gt_labels,
    gt_track=gt_track,
)

pred_track = get_pred_motion_track(
    model=model,
    labels_operation=val_motion_data_labels.operation,
    gt_input=gt_chunk['input'],
    gt_track=gt_track,
)

sim_track = simulate(
    model=model,
    motion_operation=val_motion_data_input.operation,
    motion_labels_operation=val_motion_data_labels.operation,
    observation_operation=val_observation_data.operation,
    compute_observation=compute_observation,
    gt_input=gt_chunk['input'],
    gt_track=gt_track,
    track_len=track_len,
    burn_in=35,
    max_contextl=64,
    noise_factor=0.1,
)

plt.figure(figsize=(10, 10))
if bg_img is not None:
    plt.imshow(bg_img, cmap='gray')
for track in [gt_track, motion_track, pred_track, sim_track]:
    x, y, theta = track
    plt.plot(x, y, '.', linewidth=2, markersize=2)
plt.legend(['gt', 'gt motion', 'pred motion', 'simulation'])
plt.show()






