# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# # TODO today:

# # Something is off with the normalized keypoints, look at Kristin's new data and verify the order in which the keys are

# Compare pose diffusion with L1 vs L2 norm loss functions
# Compare pose diffusion with [64, 128, 256] MPL with [512, 256, 128, 64] MLP (should I try some other variants too?)



# Make a config that has contextlen 64, 6 layers, no convolution on the sensory inputs (or no sensory inputs)
# Can it learn something reasonable in a short time, like 1 hour?
# Once it does, try training it without binning
# Once that works, try training it with diffusion
#     Try it on the velocity features?
#     Try it by always adding a lot of noise (it should learn to ignore those noisy inputs)
#     Try by gradually allowing more noise in later phases

# Once I find a regime that seems to be working:
#     Try swapping out pose velocity for pose 
#     Try swapping out pose for body centric keypoints

# %%
# Look at results from lastest trained models
# - diffusion linear
# - continous linear

# If it looks good, run the simulation on those (this will take forEVER with the diffusion model)
# - can I speed up the simulation? It is quite slow
# - maybe find a way to distribute the task... though that sounds daunting... maybe just parallelize over frame bits? Would a parfor work?

# Run the analysis on that simulation

# Lets say I have results for model with bins, no bins, diffusion (linear layer), ...
# - Now we want to write something about this
# - Bins vs diffusion, sampling jointly vs not
# - 






# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
    
import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pickle

from apf.io import read_config
from apf.training import train
from apf.utils import function_args_from_config, set_mpl_backend, is_notebook
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

set_mpl_backend('tkAgg')
ISNOTEBOOK = is_notebook()
if ISNOTEBOOK:
    from IPython.display import HTML, display, clear_output
    plt.ioff()
else:
    plt.ion()

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# %%
# configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_20251007.json"
# configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_simpler_nobins_diffusion.json"
# configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_simpler_nobins.json"
configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_simpler_20251104.json"

mode = 'train' # can toggle to 'train'/'test'
# pretrained_modelfile = os.path.join('/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/llmnets',
#                                     'predvel_20251007_20251002T000000_epoch200.pth')
# pretrained_modelfile = os.path.join('/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_simple_20251121T075929_nobins_nobins_epoch200.pth')
# pretrained_modelfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_simple_20251121T075929_nobins_nobins_epoch200.pth"
# pretrained_modelfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_simple_20260122T055942_nobins_cont4real_epoch200.pth"
# pretrained_modelfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_simple_20260122T045445_trial_linear_cont4real_epoch200.pth"
pretrained_modelfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_simple_20251119T080019_epoch200.pth"

restartmodelfile = pretrained_modelfile
debug_uselessdata = False

# %%
# Get movie from data

# infile = config['intrainfile']

# data = {}
# with np.load(infile) as data1:
#     for key in data1:
#         LOG.info(f'loading {key}')
#         data[key] = data1[key]
# LOG.info('data loaded')

# data['frames'][0]
# moviepath = os.path.join(data['expdirs'][0], 'movie.ufmf')

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
import time
t0 = time.time()

if mode == 'train' and False:
    loadmodelfile = None
else:
    loadmodelfile = pretrained_modelfile
    
res = init_flyllm(configfile=configfile, mode=mode, restartmodelfile=restartmodelfile,
                  loadmodelfile=loadmodelfile, debug_uselessdata=debug_uselessdata,
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

time.time() - t0


# %%
# Time to load the full dataset: 506 seconds

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

# %%
from flyllm.plotting import update_debug_plots, update_loss_plots
from flyllm.plotting import initialize_debug_plots, initialize_loss_plots

valexample = next(iter(val_dataloader))

debug_params = {}
# if contextl is long, still just look at samples from the first 64 frames
if config['contextl'] > 64:
    debug_params['tsplot'] = np.round(np.linspace(0,64,5)).astype(int)
    debug_params['traj_nsamplesplot'] = 1
hdebug = {}
hdebug['train'] = initialize_debug_plots(train_dataset, train_dataloader, res['train_data'], name='Train', **debug_params)
# hdebug['val'] = initialize_debug_plots(val_dataset, val_dataloader, res['train_data'], name='Val', **debug_params)

hloss = initialize_loss_plots(loss_epoch)


def refresh_plots(hdebug):
    
    if ISNOTEBOOK:
        if 'display_handles' not in hdebug:
            hdebug['display_handles'] = {}
        for k,fig in hdebug.items():
            if not k.startswith('fig') or fig is None:
                continue 
            if k in hdebug['display_handles']:
                hdebug['display_handles'][k].update(fig)
            else:
                hdebug['display_handles'][k] = display(fig,display_id=k)

    else:
        for k,fig in hdebug.items():
            if not k.startswith('fig') or fig is None:
                continue 
            fig.canvas.draw()
            fig.canvas.flush_events()

if ISNOTEBOOK:
    refresh_plots(hdebug['train'])
    # refresh_plots(hdebug['val'])
    refresh_plots(hloss)
else:
    plt.ion()
    plt.show(block=False)


def end_iter_hook(model=None, step=None, example=None, predfn=None, **kwargs):
    
    assert step is not None
    
    if step % config['niterplot'] != 0:
        return

    assert model is not None
    assert example is not None
    assert predfn is not None

    LOG.info(f'Updating debug plots at step {step}')

    with torch.no_grad():
        trainpred = predfn(example['input'].to(device=device))
        # valpred = predfn(valexample['input'].to(device=device))
    update_debug_plots(hdebug['train'],config,model,train_dataset,example,trainpred,name='Train',criterion=criterion,**debug_params)
    # update_debug_plots(hdebug['val'],config,model,val_dataset,valexample,valpred,name='Val',criterion=criterion,**debug_params)
    refresh_plots(hdebug['train'])
    # refresh_plots(hdebug['val'])
    return

def end_epoch_hook(loss_epoch=None, epoch=None, **kwargs):
    assert loss_epoch is not None
    LOG.info(f'Updating loss plots at end of epoch {epoch}')
    update_loss_plots(hloss, loss_epoch)
    refresh_plots(hloss)
    return

train_args['end_epoch_hook'] = end_epoch_hook
train_args['end_iter_hook'] = end_iter_hook

# %%
train_args['optimizer_args']['lr']

# %%
# train_args['num_train_epochs'] = 10
train_args['optimizer_args']['lr'] = 0.0001 # 0.0001
train_args['optimizer_args']['lr']

# %%
train_args['savefilestr'] += '_nobins_cont4real'
train_args['savefilestr']

# %%
example = train_dataset[0]
example['labels'].shape

# %%
# # from apf.models import initialize_model
# model, criterion = initialize_model(config, train_dataset, device)
# train_args['model'] = model

# %%

# %%
# # model_file = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_simple_20251210T061706_diffusion_trial_epoch50.pth"
# model_file = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_simple_20251119T080019_epoch200.pth"


model_file = pretrained_modelfile
checkpoint = torch.load(model_file, map_location='cuda', weights_only=False)
# model.load_state_dict(checkpoint['model'])

# %%
train_args['savefilestr'] += '_trial'

# %%
train_args['model'].decoder.t_embeddings.embeddings = train_args['model'].decoder.t_embeddings.embeddings.to('cuda:0')

# %%
model.decoder.t_embeddings.embeddings = model.decoder.t_embeddings.embeddings.to('cuda:0')

# %%
train_args.keys()
# train_args['start_epoch']
train_args['num_train_epochs'] = 200
train_args['start_epoch'] = 0

# %%
import time
t0 = time.time()
out = train(**train_args)
# model, best_model, loss_epoch = out
time.time() - t0

# %%
tmp = 0

# %%
204 / 60 * 20 # one hour on this smaller data

1:15 hour: so total would > 20 hours

- increase batch size by 4x
1:33 hour: this means that its actually slower... maybe going to a beefier machine would be better here

- skip the sensory features
(did  make a difference)

- on an h100 machine
1 hour (not much faster)
44 minutes with a 4x batch size
27 minutes if I skip flipping

# 44 minutes with reduced data (V3) # this was doing the same since I hadn't filtered, however, filtering didn't save much data
25 minutes for 10 epochs with filtering on movie ids
    1 hour for 30 epochs, lets see if that gets us as much juice as we hope
    next up, train this with visualiation hooks

# %%
tmp = 0

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
tmp = 0

# %%
pretrained_modelfile

# %%
# model_file = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_simple_20251210T061706_diffusion_trial_linear_cont_epoch200.pth"
# model_file = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_simple_20251210T061706_diffusion_trial_linear_cont_epoch200.pth"
model_file = pretrained_modelfile
modelfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_simple_20251119T080019_epoch200.pth"
checkpoint = torch.load(model_file, map_location='cuda', weights_only=False)
model.load_state_dict(checkpoint['model'])

plt.figure()
plt.plot(checkpoint['loss']['train'].detach().cpu().numpy())
plt.plot(checkpoint['loss']['val'].detach().cpu().numpy())
plt.show()

# %%
plt.figure()
plt.plot(checkpoint['loss']['train'].detach().cpu().numpy())
plt.plot(checkpoint['loss']['val'].detach().cpu().numpy())
plt.show()
loadmodelfile = model_file

# %%

# %%
# where isdata?
isdata = ~np.all(np.isnan(pose.array),axis=-1)
plt.imshow(isdata,aspect='auto',interpolation='none')
agent_ids = np.nonzero(np.any(isdata,axis=1))[0]
start_frame = 1000
plt.xlabel('Frame')
plt.ylabel('Agent ID')

# %%
train_dataset.sessions
start_frame = 45004
agent_ids = [0, 1, 4, 6]

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
# %%
example = train_dataset[0]
example['metadata']

# %%
# TODO: plot several future trajectories overlaid on an image
# TODO: make a movie with the predicted trajectories
# TODO: make a movie from the real data and the syntehtic data (using the diffusion model)

# %%
plt.figure(figsize=(10, 5))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plot_arena()

first_frame = config['contextl']
last_frame = 100#None
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
savevidfile

# %%
ani = animate_pose(
    {'Pred': pred_track[:, first_frame:].T.copy(), 'True': gt_track[:, first_frame:].T.copy()}, 
    focusflies=agent_ids, 
    savevidfile=savevidfile,
    contextl=config['contextl']
)

# %%
n_flies -1
T -2

[_, _, T, n_flies]

# %%
track = res['train_data']['track'].array.shape

# %%
# # animate_pose??

# %%

# %%

# %%

# %%

# %%

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

# %%
gt_track.shape

# %%
pred_track.shape

# %%
# TODO:
#
# Simulate one fly at a time for the classification
# Only simulate flies corresponding to training flies (male, courtship...)
# Save track snippets for gt and pred

# %%

# %%
session_agents = np.array([session.agent_id for session in train_dataset.sessions])

# %%
session_id = 0
session = train_dataset.sessions[session_id]
chunk = train_dataset.get_chunk(session

# %%
track.array.shape
# gt_track = track.array[:, start_frame:start_frame + track_len]

# %%
config['contextl'] * 50

# %%
len(train_dataset.sessions)
config['contextl'] * 51

# %%
# session = train_dataset.sessions[7]
# np.arange(session.start_frame, session.start_frame + session.duration, config['contextl'] * 51)
# type(pred_track)
# np.save?

# %%
save_dir = 'train_data/synthetic'
import os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# %%
# simulate
from tqdm.notebook import tqdm
import pickle

track_len = config['contextl'] * 51

for i, session in enumerate(train_dataset.sessions):
    if session.duration < track_len + 1:
        continue

    start_frames = np.arange(session.start_frame, session.start_frame + session.duration - track_len, track_len)
    print(f"Session {i} / {len(train_dataset.sessions)} with {len(start_frames)} chunks...")
    
    for start_frame in start_frames:
        save_str = f"session_{i}_startframe_{start_frame}_agent_id={session.agent_id}.npy"
        
        if os.path.exists(save_str):
            continue
        
        gt_track, pred_track = simulate(
            dataset=train_dataset,
            model=model,
            track=track,
            pose=pose,
            identities=flyids,
            track_len=track_len,
            burn_in=config['contextl'],
            max_contextl=config['contextl'],
            agent_ids=[session.agent_id],
            start_frame=start_frame,
        )
        
        np.save(os.path.join(save_dir, save_str), pred_track[session.agent_id])

# %%
# How about, I run the model on ground truth data and save the hidden state
# Then I run a simulation and I save the hidden states of those as well
#
#
#
# Write a new simulation funciton that saves these different outputs
# I need to modify the model to optionally output its hidden states. 
# Then I train a binary classifier on these hidden states

# %%
len(start_frames), len(np.arange(session.start_frame, session.start_frame + session.duration - track_len, track_len))

# %%

# %%
RL HF stuff instead of GANs

Look at the paper by Kristin and Daniel

When I make another training set, save the activity (hidden states) from the model

Apply a grooming classifier to the data (and others)
(Maybe train classifiers)

Put a warning about overriding config parameters

Push code (in different PRs)

# %%
Discuss with Kristin

DIFFUSION

Diffusion model finished training and loss looks good, but the simulation is still only good for about 200 frames 
My models trained not with binning all froze and then started spiraling after 200-500 frames 
    - linear
    - MLP
    - MLP diffusion
    - linear diffusion (still training)
- I'm thinking of letting a few of these models train even longer to see if that improves it, since validation loss is still decreasing
- I looked a bit at what is happening with the diffusion model, visualizing some weights


DISCRIMINATOR

Data:
- Training data, simulating 3000 frames from fixed interval prompt sequences in training data
- Storing where in training data it is prompted from
- For training, split it into smaller chunks and keep track of distance to prompt
   - that way we can see if classifiction accuracy increases with distance to prompt, which I would expect
Features: 
- will extract same features as used by the model (sensory, posevel, globalvel, ...)
Modeling over time:
- could try convolution, fully connected, transformer, ... (can even use the same transformer as we have for next step prediction)

                                                            
                                                                                        
                                                                                                              
                                                                                                              
                                                                                                              
Training data:
    short sequences of gt and pred trajectories
    features: same as used for the model, could even use the hidden states of running the model as features

Model: 
    - Train a transformer from scratch
    - Use current transformer and classify hidden state of last frame
    - Combo: could use the pretrained transformer from the model and then do the classification

GAN approach:
    - Every so many iterations during training, simulate N frames into the future, then train the discriminator to tell them apart, 
    propagate errors to the network (we penalize good predictions because we want the network to not be able to discriminate)
    - For this to work, how we generate trk from the predicted velocities needs to be differentiable
        Would need to implement it so that it is (or maybe it already is?)
        Would be easier if the output was the track itself
    - The entire sequence of simulation needs to be differentiable so we can propagate gradients back through the weights (this includes
        computing features)

# %%
2000 * 50

# %%
