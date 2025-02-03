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
import copy

from apf.io import read_config
from apf.dataset import Zscore, Discretize, Data, Dataset
from apf.training import to_dataloader, init_model, train
from apf.simulation import get_motion_track, get_pred_motion_track, simulate
from apf.utils import function_args_from_config
from apf.models import TransformerModel
from apf.data import debug_less_data

from flyllm.config import DEFAULTCONFIGFILE, posenames, featrelative, featglobal, featthetaglobal
from flyllm.features import (
    featglobal, get_sensory_feature_idx, kp2feat, compute_sensory_wrapper, compute_movement, compute_global_velocity,
    compute_relpose_velocity, compute_relpose_tspred,
)

from experiments.flyllm import load_npz_data

from experiments.spatial_infomax import set_invalid_ends
# -



configfile = "/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/config_fly_llm_default.json"
# configfile = "/groups/branson/bransonlab/test_data_apf/test_config.json"
config = read_config(
    configfile,
    default_configfile=DEFAULTCONFIGFILE,
    posenames=posenames,
    featglobal=featglobal,
    get_sensory_feature_idx=get_sensory_feature_idx,
)



data, scale_perfly = load_npz_data(config['intrainfile'], config)

valdata, val_scale_perfly = load_npz_data(config['invalfile'], config)

debug_less_data(data, n_frames_per_video=45000, max_n_videos=2)
debug_less_data(valdata, n_frames_per_video=45000, max_n_videos=2)

tmp = 'HelloWorld'
tmp.lower()

.flatten
.get_input_shapes()
.d_input
.flatten_dinput
.flatten_max_doutput
.discretize
.d_output_discrete
.discretize_nbins
.d_output_continuous
.d_output
.discreteidx


def compute_features(data, scale_perfly):
    # Remove all NaN agents (sometimes the last one is a dummy)
    Xkp = data['X']
    
    valid = np.sum(~np.isnan(Xkp[0, 0]), axis=-2) > 0
    Xkp = Xkp[..., valid]
    flyids = data['ids'][..., valid]
    isstart = data['isstart'][..., valid]


    # Compute pose features
    pose = kp2feat(Xkp=Xkp, scale_perfly=scale_perfly, flyid=flyids)
    relpose = pose[featrelative, ...]
    globalpos = pose[featglobal, ...]

    # Compute all sensory features
    n_flies = Xkp.shape[-1]
    sensory = np.array([compute_sensory_wrapper(Xkp, flyid, theta_main=globalpos[featthetaglobal, :, flyid])[0].T 
                        for flyid in range(n_flies)]).T

    # Compute global movement
    Xorigin = globalpos[:2, ...]
    Xtheta = globalpos[2, ...]
    _, n_frames, n_flies = globalpos.shape
    tspred_global = config['tspred_global']
    dXoriginrel, dtheta = compute_global_velocity(Xorigin, Xtheta, tspred_global)
    movement_global = np.concatenate((dXoriginrel[:, [1, 0]], dtheta[:, None, :, :]), axis=1)
    # movement_global = movement_global.reshape((-1, n_frames, n_flies))

    for movement, tspred in zip(movement_global, tspred_global):
        set_invalid_ends(movement, isstart, dt=tspred)
    
    # Compute pose velocity
    tspred_dct = []
    relpose_rep = compute_relpose_velocity(relpose, tspred_dct)
    # relpose_rep = compute_relpose_tspred(relpose, tspred_dct, discreteidx=discreteidx)
    relpose_rep = np.moveaxis(relpose_rep, 2, 0)

    set_invalid_ends(relpose_rep, isstart, dt=1)
    
    # relpose_rep = relpose_rep.reshape((-1, n_frames, n_flies))
    relpose_rep = relpose_rep[0] # we are only using one tspred_dct here (empty, what does that refer to)

    return relpose, globalpos, sensory, movement_global, relpose_rep, Xkp, isstart, flyids

# Compute features
relpose, globalpos, sensory, movement_global, relpose_rep, Xkp, isstart, flyids = \
    compute_features(data, scale_perfly)
val_relpose, val_globalpos, val_sensory, val_movement_global, val_relpose_rep, val_Xkp, val_isstart, val_flyids = \
    compute_features(valdata, val_scale_perfly)



from apf.dataset import DiscretizeSimple, Discretize

op1 = Discretize(nbins = config['discretize_nbins'], bin_epsilon = config['discretize_epsilon'][:3])
op2 = DiscretizeSimple(bin_edges=config['discretize_nbins']-1)
movement = movement_global[0].T

binned1 = op1.apply(movement)
binned2 = op2.apply(movement)

plt.plot(op1.bin_edges[0], '.')
plt.plot(op2.bin_edges[0], '.')
# op2.bin_edges.shape

from apf.dataset import DiscretizeSimple as Discretize
# from apf.dataset import Discretize

# +
# Wrap into data containers

# Input data
relpose_in = Data(raw=relpose.T, operation=Zscore())
val_relpose_in = Data(raw=val_relpose.T, operation = relpose_in.operation)

sensory_in = Data(raw=sensory.T, operation=Zscore())
val_sensory_in = Data(raw=val_sensory.T, operation=sensory_in.operation)

# Output data
global_movement_out = []
val_global_movement_out = []
tspred_global = config['tspred_global']
bin_config = {'nbins': config['discretize_nbins'], 'bin_epsilon': config['discretize_epsilon'][:3]}
# bin_config = {'bin_edges': config['discretize_nbins']-1}
for movement, val_movement, tspred in zip(movement_global, val_movement_global, tspred_global):
    global_movement_out.append(Data(raw=movement.T, operation=Discretize(**bin_config)))
    val_global_movement_out.append(Data(raw=val_movement.T, operation=global_movement_out[-1].operation))


relpose_vel_other_out = Data(raw=relpose_rep.T, operation=Zscore())
val_relpose_vel_other_out = Data(raw=val_relpose_rep.T, operation=relpose_vel_other_out.operation)

# # Handle the case where wing angles are discrete
# bin_config['bin_epsilon'] = config['discretize_epsilon'][3:]
# relpose_vel_wings_out = Data(raw=relpose_rep[-2:].T, operation=Discretize(**bin_config))
# val_relpose_vel_wings_out = Data(raw=val_relpose_rep[-2:].T, operation=relpose_vel_wings_out.operation)

# relpose_vel_other_out = Data(raw=relpose_rep[:-2].T, operation=Zscore())
# val_relpose_vel_other_out = Data(raw=val_relpose_rep[:-2].T, operation=relpose_vel_other_out.operation)

# Motion (output) data as inputs
global_movement_in = Data(raw=np.roll(movement_global[0].T, shift=tspred_global[0], axis=1), operation=Zscore())
val_global_movement_in = Data(raw=np.roll(val_movement_global[0].T, shift=tspred_global[0], axis=1), operation=global_movement_in.operation)

relpose_vel_in = Data(raw=np.roll(relpose_rep.T, shift=1, axis=1), operation=Zscore())
val_relpose_vel_in = Data(raw=np.roll(val_relpose_rep.T, shift=1, axis=1), operation=relpose_vel_in.operation)
# -



# +
# Wrap into dataset
train_dataset = Dataset(
    inputs=[global_movement_in, relpose_vel_in, relpose_in, sensory_in],
    labels=[relpose_vel_other_out] + global_movement_out, # + [relpose_vel_wings_out],
    isstart=isstart,
    context_length=config['contextl'],
)

val_dataset = Dataset(
    inputs=[val_global_movement_in, val_relpose_vel_in, val_relpose_in, val_sensory_in],
    labels=[val_relpose_vel_other_out] + val_global_movement_out, # + [val_relpose_vel_wings_out],
    isstart=val_isstart,
    context_length=config['contextl'],
)
# -



# Wrap into dataloader
device = torch.device(config['device'])
train_dataloader = to_dataloader(train_dataset, device=device, batch_size=config['batch_size'], shuffle=True)
val_dataloader = to_dataloader(val_dataset, device=device, batch_size=config['batch_size'], shuffle=False)

# +
# Initialize the model
model_args = function_args_from_config(config, TransformerModel)
model = init_model(train_dataset, model_args).to(device)

# Train the model
train_args = function_args_from_config(config, train)
# -



# +
import pickle
import os

savedir = '/groups/branson/home/eyjolfsdottire/data/flyllm'
savefile = os.path.join(savedir, 'model_test_20241129_2.pkl') #model_test_20241210
if not os.path.exists(savedir):
    os.makedirs(savedir)

# with open(savefile, 'wb') as f:
#     pickle.dump(model, f)

with open(savefile, 'rb') as f:
    model = pickle.load(f)
# -

train_args['num_train_epochs'] = 90
# train_args['optimizer_args']['lr'] = 0.0001
train_args['optimizer_args']['lr'] = 0.00005

model, best_model, loss_epoch = train(train_dataloader, val_dataloader, model, **train_args)

# Plot the losses
idx = np.argmin(loss_epoch['val'])
print((idx, loss_epoch['val'][idx]))
plt.figure()
plt.plot(loss_epoch['train'])
plt.plot(loss_epoch['val'])
plt.plot(idx, loss_epoch['val'][idx], '.g')
plt.show()

# Plot the losses
idx = np.argmin(loss_epoch['val'])
print((idx, loss_epoch['val'][idx]))
plt.figure()
plt.plot(loss_epoch['train'])
plt.plot(loss_epoch['val'])
plt.plot(idx, loss_epoch['val'][idx], '.g')
plt.show()

# +
# Simulate

# +
# kp2feat

# feat2vel:
#     compute_global_velocity(feat[:3, ...]) # dfwd, dside, dtheta
#         rotate (x, y) by theta
#         compute (dx, dy) with next frame
#         # 
#     compute_relpose_velocity(feat[3:, ...]) # delta

# vel2feat
#     apply_global_velocity(vel[:3])
#         dx, dy = rotate_2d_points(np.array([d_side, d_fwd])[None, :], -theta).flat
#         x = x + dx
#         y = y + dy
#         theta = theta + d_theta
#     apply_relpose_velocity(vel[3:])

# feat2kp



# def apply_global_velocity(global_pos, global_vel):
#     x, y, theta = global_pos
#     dfwd, dside, dtheta = global_vel
#     dx, dy = rotate_2d_points(np.array([dside, dfwd]), -theta)
#     x = x + dx
#     y = y + dy
#     theta = y + dy

# +
# globalpos.shape
# # movement_global[0].shape

# frame_idx = 1000
# agent_idx = 2

# x, y, theta = globalpos[:, frame_idx, agent_idx]
# dfwd, dside, dtehta = movement_global[0, :, frame_idx, agent_idx]

# +
from apf.simulation import apply_motion

# global_new = apply_motion(x, y, theta, dfwd, dside, dtehta)
# np.array(global_new) - globalpos[:, frame_idx+1, agent_idx]
# -

pose = relpose[:, frame_idx, agent_idx]
dpose = relpose_rep[:, frame_idx, agent_idx]
rel_new = pose + dpose
rel_new - relpose[:, frame_idx+1, agent_idx]

config['device']

# +
# Take a chunk from the dataset

# Apply the model prediction to it

# Use the last frame's prediction to apply it
# -



# +
# In both cases we have:

# track
# pose (features)
# motion           compute_motion, apply_motion (for simulation)
# observation      compute_observation

# If we treat motion as a single datachunk we cannot have multiple predictions or split out which operations to apply within motion
# We could:
# - add predictions for future timesteps as auxiliary data chunks (they are not used in apply_motion)
# - create a super-operation that can split data and do different operations on the split

# +
# For each experiment, define


# kp2feat()
# compute_observation(kp)
# compute_motion(feat)
# apply_motion(feat, feat_motion)
# feat2kp()


# Also need to consider that there can be many agents, all functions should handle multiple agents

# +
# In this case: 
#
# track is Xkp
# 
# -







# +
# from simple.simple_plot import plot_flies_in_frame, plot_fly_in_frame

# plt.figure(figsize=(8, 8))
# plt.plot(x, y, '-', color=[.8, .8, .8])
# plot_flies_in_frame(frame_idx, traj_length=0, trk=None, apt_trk=Xkp[2:].transpose([3, 0, 1, 2]), trk_color='k', kp_color='k')
# -



# +
# Initialize simulation

track_len = 5000
burn_in = 200
max_contextl = 512
noise_factor = 0

agent_id = 0
start_frame = 100
curr_frame = burn_in

gt_chunk = train_dataset.get_chunk(start_frame=start_frame, duration=track_len, agent_id=agent_id)
gt_input = gt_chunk['input']
gt_track = Xkp[..., start_frame:start_frame+track_len, :]
gt_pos = globalpos[..., start_frame:start_frame+track_len, agent_id]
gt_pose = relpose[..., start_frame:start_frame+track_len, agent_id]
flyid = np.unique(flyids[start_frame:start_frame+track_len, agent_id])
assert len(flyid) == 1, f"Too many flyids: {flyid}"

# Initialize model input
device = next(model.parameters()).device
model_input = torch.zeros((1, track_len, gt_input.shape[-1]))

model_input[0, :curr_frame, :] = gt_input[:curr_frame]
model_input = model_input.to(device)

# Initialize track
track = copy.deepcopy(gt_track)
track[:, curr_frame:, agent_id] = np.nan

pose = np.concatenate([gt_pos, gt_pose], axis=0)
pose[:, curr_frame:] = np.nan

# +
# Simulate

from apf.simulation import apply_motion
from apf.utils import modrange
from flyllm.config import featangle
from flyllm.features import feat2kp

model.eval()
while curr_frame < track_len:
    # Make a motion prediction
    frame0 = 0
    if max_contextl is not None:
        frame0 = max(0, curr_frame - max_contextl)

    # Apply model to previous frames
    pred = model.output(model_input[:, frame0:curr_frame, :])

    # From prediction to pose velocities
    pos_vel = global_movement_out[0].operation.inverse(pred['discrete'][:, -1:].detach().cpu().numpy()).flatten()
    pose_vel = relpose_vel_other_out.operation.inverse(pred['continuous'][:, -1:].detach().cpu().numpy()).flatten()

    # From pose velocities to pose
    curr_pose = pose[:, curr_frame-1]
    pose[:3, curr_frame] = apply_motion(*curr_pose[:3], *pos_vel)
    pose[3:, curr_frame] = curr_pose[3:] + pose_vel

    pose[featangle, curr_frame] = modrange(pose[featangle, curr_frame], -np.pi, np.pi)
    
    # From pose to kp
    track[..., curr_frame, agent_id] = feat2kp(pose[:, curr_frame], scale_perfly[:, flyid[0]])[..., 0, 0]
    
    # From kp to observation
    sensory = compute_sensory_wrapper(track[:, :, curr_frame:curr_frame+1, :], agent_id, theta_main=pose[2, curr_frame:curr_frame+1])[0].T
    
    # Now, wrap everything into the input
    #input: [global_movement_in, relpose_vel_in, relpose_in, sensory_in]
    curr_in = [
        global_movement_in.operation.apply(pos_vel[None, None, :]),
        relpose_vel_in.operation.apply(pose_vel[None, None, :]),
        relpose_in.operation.apply(curr_pose[None, None, 3:]),
        sensory_in.operation.apply(sensory[None, :, :])
    ]
    curr_in = np.concatenate(curr_in, axis=-1)
    model_input[:, curr_frame, :] = torch.from_numpy(curr_in.astype(np.float32)).to(device)

    curr_frame += 1
# -





# +
from flyllm.plotting import plot_flies, plot_fly

def plot_arena():
    ARENA_RADIUS_MM = 26.689
    n_pts = 1000
    theta = np.arange(0, np.pi*2, np.pi*2 / n_pts)
    x = np.cos(theta) * ARENA_RADIUS_MM
    y = np.sin(theta) * ARENA_RADIUS_MM
    plt.plot(x, y, '-', color=[.8, .8, .8])


# +
f_max = 5000
x, y = gt_pos[:2, :f_max]
x_, y_ = pose[:2, :f_max]

plot_arena()
plt.plot(x, y, '-g', linewidth=2)
plt.plot(x_, y_, '--r')
plt.axis('equal')
plt.show()
# -
frame_idx = 4000
hkpts, hedges, htxt, fig, ax = plot_flies(Xkp[..., start_frame + frame_idx, agent_id:agent_id+1], kptidx=[])
plot_flies(poses=track[:, :, frame_idx, agent_id:agent_id+1], fig=fig, ax=ax, kptidx=[], colors='hsv')
# plt.axis([-30, 30, 30, -30])
plot_arena()
plt.show()

# +
from flyllm.simulation import animate_pose

savevidfile = "/groups/branson/home/eyjolfsdottire/data/flyllm/animation_oldisher5000.gif"
ani = animate_pose({'Pred': track.copy(), 'True': gt_track.copy()}, focusflies=[0], savevidfile=savevidfile)
# -









# +
from flyllm.plotting import plot_flies, plot_fly



hkpts, hedges, htxt, fig, ax = plot_flies(Xkp[..., start_frame + curr_frame, :], kptidx=[])
plot_flies(poses=kp[:, :, 0, :], fig=fig, ax=ax, kptidx=[], colors='hsv')


plt.show()
# -














# +
# apply relpose velocity:

# apply movement_global

# +
# compute_relpose_velocity??

# +


# # Xn, fthorax, thorax_theta = body_centric_kp(Xkp)

# # mid_eye = (Xn[keypointnames.index('right_eye')] + Xn[keypointnames.index('left_eye')]) / 2
# # head_base_x = mid_eye[0]
# # head_base_y = mid_eye[1]

# plt.plot(head_base_x[:, 0])
# plt.plot(relpose[0, :, 0])
# # plt.plot(head_base_y[:, 0])
# # plt.plot(relpose[1, :, 0])
# -





# +
# # # compute_sensory_wrapper??
# _, idxinfo = compute_sensory_wrapper(Xkp, 0, theta_main=globalpos[featthetaglobal, :, 0], returnidx=True)
# idxinfo

# +
# config

# +
# Compute movement features (for prediction)
# out = compute_movement(
#     relpose=relpose,
#     globalpos=globalpos,
#     simplify=False, # or load from config
#     dct_m=None, # or load from config
#     tspred_global=config['tspred_global'],
#     compute_pose_vel=config['compute_pose_vel'],
#     discreteidx=config['discreteidx']
# )
# movement = out[0]
# -







# +
print(globalpos.shape)      # (x, y, theta)

print(relpose.shape)        # see posenames[3:]
print(sensory.shape)
# print(movement.shape)       # 
#     # movement_global: (dfwd, dside, dtheta) * len(tspred)   (39, 19850, 10) # this has dt = [1, ..., 150]
#     # relpose_rep: pose_velocity                             (26, 19850, 10) # this has dt = 1
print(movement_global.shape)
print(relpose_rep.shape)

# +
# chunk from flyllm code:
#   input: (512, 212)  sensory + relpose = 212
#   labels: (512, 65)  movement_global + relpose_rep = 13*3 + 26
# -

186/3













chunk = train_data[0]

chunk['input'].shape  

chunk['labels'].shape

chunk['labels_discrete'].shape  # = 41 * 25

orig_chunk['input'].shape              # 511 x 241          241 - 212 = 29 # all motion features (global and relative)
orig_chunk['labels'].shape             # 511 x 24           26 - 2 # two relpose features are discrete
orig_chunk['labels_discrete'].shape    # 511 x 41 x 25      13*3 + 2 # only discretize one timestep for the wing angles

# +
orig_chunk['metadata']

orig_chunk['input']             sensory, relpose, global_movement_in[0], relpose_vel_in
orig_chunk['labels']            relpose (apart from wing angles)
orig_chunk['labels_discrete']   global_pose * len(tspred) + relpose[wing_angles]
# -

orig_chunk['input'].shape

global_movement_out[4].processed.shape

start_frame = orig_chunk['metadata']['t0']
agent_id = orig_chunk['metadata']['flynum']
duration = orig_chunk['input'].shape[0]
chunk = train_data.get_chunk(start_frame, duration, agent_id)

idx = 1
# idx2 = 29

# inputs
relposenames = ['relpose_' + str(i) for i in range(26)]
sensorynames = ['sensory_' + str(i) for i in range(186)]
globalnames = ['globalvel_' + str(i) for i in range(3)]
relvelnames = ['relvel_' + str(i) for i in range(26)]
names = globalnames + relvelnames + relposenames + sensorynames

# +
# outputs
# -

idx = 0
idx2 = 0

npchunk = chunk['labels_discrete'].detach().cpu().numpy().reshape([duration, -1, 25])

# +
plt.figure(figsize=(15, 5))
plt.imshow(npchunk[:, idx, :].T)

plt.figure(figsize=(15, 5))
plt.imshow(orig_chunk['labels_discrete'][:, idx, :].T)
idx += 1 
# -

idx = 0
frame_id = 100

# idx -=1 
plt.figure(figsize=(15, 5))
plt.plot(npchunk[frame_id, idx, :], '-*', linewidth=2)
plt.plot(orig_chunk['labels_discrete'][frame_id, idx, :], '-*', linewidth=1)
# plt.title(names[idx])
plt.show()
idx += 1

# +
# Notes: 
# - there appears to be some difference in the zscore paramters, but other than that the features look the same for input, labels
# - the binned labels have a similar story, the bin peaks are in similar locations but often shifted
#    - I don't zscore before binning (that should result in the same data anyways)
#    - it is probably a similar issue as with continuous features, that the distribution uses different data...
# - I would probably skip zscoring the sensory features
# - I think it should be fine to try training on this data now and simulate with it
# -



# +
# relpose: (26, 20000, 10)
# sensory: (186, 20000, 10)

# +
raw_chunk = X[10]
raw_chunk['metadata']

flynum = raw_chunk['metadata']['flynum']
t0 = raw_chunk['metadata']['t0']

# +
# Compare relpose features 
a = relpose[:, t0:t0+config['contextl'], flynum].T
b = raw_chunk['input'][:, :26]

plt.figure(figsize=(15, 15))
for i in range(26):
    plt.subplot(6, 5, i+1)
    plt.plot(a[:, i])
    plt.plot(b[:, i])
    plt.title(posenames[3+i], fontsize=9)
    plt.axis('off')
plt.show()

# +
# Compare sensory features
a = sensory[:, t0:t0+config['contextl'], flynum].T
b = raw_chunk['input'][:, 26:]

for key, vals in idxinfo.items():
    print(key)
    
    cols = 10
    rows = int(np.ceil((vals[1]-vals[0]) / 5))
    plt.figure(figsize=(15, rows))
    count = 0
    for i in range(vals[0], vals[1]):
        count += 1
        plt.subplot(rows, cols, count)
        plt.plot(a[:, i])
        plt.plot(b[:, i])
        plt.title(i)
        plt.axis('off')
    plt.show()

# +
# Compare global movement
a = movement_global[..., t0:t0+config['contextl'], flynum].reshape([-1, config['contextl']]).T
b = raw_chunk['labels'][:, :13*3]

plt.figure(figsize=(15, 10))
for i in range(39):
    plt.subplot(5, 9, i+1)
    plt.plot(a[:, i])
    plt.plot(b[:, i])
    featname = posenames[np.mod(i, 3)]
    dt = config['tspred_global'][i//3]
    plt.title(f"{featname} dt={dt}", fontsize=9)
    plt.axis('off')
plt.show()

# +
# Compare relpose_rep
a = relpose_rep[..., t0:t0+config['contextl'], flynum].T
b = raw_chunk['labels'][:, 13*3:]

plt.figure(figsize=(15, 15))
for i in range(26):
    plt.subplot(6, 5, i+1)
    plt.plot(a[:, i])
    plt.plot(b[:, i])
    plt.title(posenames[3+i], fontsize=9)
    plt.axis('off')
plt.show()
# -











# +
# TODO:
# - what is the difference between compute_relpose_tspred, and compute_relpose_velocity, used in compute_movement
#     - only the former uses discreteidx
#     - they don't use the config parameter discrete_tspred, where does that get used?
#     - instead it uses dct_m to determine the tspred

# - Original dataset chunks have the following dimensions:
#       input.shape              # 511 x 241          241 - 212 = 29 # all motion features (global and relative)
#       labels.shape             # 511 x 24           26 - 2         # two relpose features (wing angles) are discrete
#       labels_discrete.shape    # 511 x 41 x 25      13*3 + 2       # only discretize one timestep for the wing angles
#  - to handle this in my framework I had to split relpose_rep into wing_angles and other
# 
# - [DONE] support continuous and discrete outputs
# - [NO-OP] looks like discrete chunk should have shape contextl x n_feat x n_bins but I have it flat... 
#    - Q: why did that work before? A: mixed_loss_criterion takes care of reshaping (so maybe I should not flatten, that's less error prone)

# - [DONE] wrap data into train and validation set
# - [DONE] verify that I'm using the desired parameters for discretize
# - validate that a chunk from my dataset is the same as a chunk from flyllm (this will be a stretch)
#      - old chunks have a shorter context length
#      - old chunks chop off more data (since features are computed after chunking rather than before) 
#   - [DONE] validate raw features
#   - validate processed features

# - Automate the way we split features into binned vs continuous (using info from config)
# - Automate the way we extract the relevant bin_epsilons for Discretizing (using info from config)
# -





# +
from flyllm.features import sanity_check_tspred, compute_features, compute_npad
from apf.data import chunk_data

idct_m = None
dct_m = None
# how much to pad outputs by -- depends on how many frames into the future we will predict
npad = compute_npad(config['tspred_global'], dct_m)

compute_feature_params = {
 "simplify_out": config['simplify_out'],
 "simplify_in": config['simplify_in'],
 "dct_m": dct_m,
 "tspred_global": config['tspred_global'],
 "compute_pose_vel": config['compute_pose_vel'],
 "discreteidx": config['discreteidx'],
}

# function for computing features
reparamfun = lambda x, id, flynum, **kwargs: compute_features(
    x, id, flynum, scale_perfly, outtype=np.float32, **compute_feature_params, **kwargs)

val_reparamfun = lambda x, id, flynum, **kwargs: compute_features(
    x, id, flynum, val_scale_perfly, outtype=np.float32, **compute_feature_params, **kwargs)

# sanity check on computing features when predicting many frames into the future
sanity_check_tspred(
    data, 
    compute_feature_params,
    npad,
    scale_perfly,
    contextl=config['contextl'],
    t0=510,
    flynum=0
)

# chunk the data if we didn't load the pre-chunked cache file
# chunk_data_params = {'npad': npad}
print('Chunking training data...')
X = chunk_data(data, config['contextl'], reparamfun, npad=npad)
# -



raw_chunk = X[0]
print(raw_chunk['input'].shape)
print(raw_chunk['labels'].shape)
# print(raw_chunk['labels_discrete'].shape)



# +
# Compare:

# input and labels chunks with the unprocessed data
# -









# +
from flyllm.dataset import FlyMLMDataset

dataset_params = {
    'max_mask_length': config['max_mask_length'],
    'pmask': config['pmask'],
    'masktype': config['masktype'],
    'simplify_out': config['simplify_out'],
    'pdropout_past': config['pdropout_past'],
    'input_labels': config['input_labels'],
    'dozscore': True,
    'discreteidx': config['discreteidx'],
    'discretize_nbins': config['discretize_nbins'],
    'discretize_epsilon': config['discretize_epsilon'],
    'flatten_labels': config['flatten_labels'],
    'flatten_obs_idx': config['flatten_obs_idx'],
    'flatten_do_separate_inputs': config['flatten_do_separate_inputs'],
    'p_add_input_noise': config['p_add_input_noise'],
    'dct_ms': (dct_m, idct_m),
    'tspred_global': config['tspred_global'],
    'discrete_tspred': config['discrete_tspred'],
    'compute_pose_vel': config['compute_pose_vel'],
}

train_dataset_params = {
    'input_noise_sigma': config['input_noise_sigma'],
}
train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)

# # zscore and discretize parameters for validation data set based on train data
# # we will continue to use these each time we rechunk the data
# dataset_params['zscore_params'] = train_dataset.get_zscore_params()
# dataset_params['discretize_params'] = train_dataset.get_discretize_params()

# val_dataset = FlyMLMDataset(valX,**dataset_params)
# -

orig_chunk = train_dataset[0]

orig_chunk['input'].shape
# orig_chunk.keys()

# +
config['discreteidx'] # Two of the keypoint features (left/right wing angles) are also discretized - it seems

13*3 + 2

# +
# Create dataset
# -

# Load data
position, observation, isstart = load_npz_data(config['intrainfile'])

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
val_position, val_observation, val_isstart = load_npz_data(config['invalfile'])
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






