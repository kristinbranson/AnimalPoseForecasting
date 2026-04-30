# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: transformer312
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Train and evaluate forecasting for ratinabox synthetic rat

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import numpy as np
import torch
import time
import os

import apf
from apf.training import train
import apf.utils as utils
import matplotlib.pyplot as plt

import synthrat
from synthrat import apf_ratinabox
from flyllm.prepare import init_flyllm
from flyllm.plotting import initialize_debug_plots, initialize_loss_plots

import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

utils.set_mpl_backend('tkAgg')
ISNOTEBOOK = utils.is_notebook()
if ISNOTEBOOK:
    from IPython.display import HTML, display, clear_output
else:
    plt.ion()

LOG.info('CUDA available: ' + str(torch.cuda.is_available()))
LOG.info('isnotebook: ' + str(ISNOTEBOOK))

# %%
# set a timestamp for saving results
timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
print('Timestamp: ' + timestamp)

# %% [markdown]
# ## Set parameters

# %%
configfile = 'configs/config_synthrat_20260424.json'
restartmodelfile = None
outfigdir = 'figs'
debug_uselessdata = False

# path to config file based on code directory
synthratdir = synthrat.__path__[0]
configfile = os.path.join(synthratdir,configfile)
assert os.path.exists(configfile), f'Config file {configfile} does not exist!'

# make directory if it doesn't exist
outfigdir = os.path.join(synthratdir,outfigdir)
if not os.path.exists(outfigdir):
    os.makedirs(outfigdir)

# %%
from synthrat.config import read_config_kwargs
import apf.io
mode = 'train'
loadmodelfile = None
restartmodelfile = None

res = apf.io.init_config(configfile=configfile,mode=mode,read_config_kwargs=read_config_kwargs)
config = res['config']
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

res = apf.models.init_state(config=config)
device = res['device']

import pickle, os

CACHE_VERSION = 1

def _cache_path(input_file):
    base = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(os.path.dirname(input_file),
                        f'apf_cache_v{CACHE_VERSION}_{base}.pkl')

def _build_one(input_file):
    """Build a (dataset, data_dict, ratinabox_info) for one input file and
    return them. ratinabox_info comes back as a dict-of-dicts (already pickle-stable)."""
    data = {}
    dataset, ratinabox_info, data['pose'], data['velocity'], \
        data['sensory'], _, data['isstart'] = \
            apf_ratinabox.make_dataset(config=config, filename=input_file,
                                        debug=debug_uselessdata, return_all=True)
    return dataset, data, ratinabox_info

def _save_cache(path, dataset, data, ratinabox_info):
    cached = {
        'ratinabox_info': ratinabox_info,                  # dict-of-dicts
        'pose_array':     data['pose'].array,
        'velocity_array': data['velocity'].array,
        'sensory_array':  data['sensory'].array,           # the expensive one
        'isstart':        data['isstart'],
        'dataset_params': dataset.get_params(),            # pure dict via to_dict()
    }
    print(f'Saving cache to: {path}')
    with open(path, 'wb') as f:
        pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)

def _load_cache(input_file, path):
    with open(path, 'rb') as f:
        cached = pickle.load(f)
    data = {}
    dataset, ratinabox_info, data['pose'], data['velocity'], \
        data['sensory'], _, data['isstart'] = \
            apf_ratinabox.make_dataset(config=config, filename=input_file,
                                        debug=debug_uselessdata, return_all=True,
                                        cached_sensory_array=cached['sensory_array'])
    return dataset, data, ratinabox_info

def _get_or_build(input_file):
    path = _cache_path(input_file)
    if (not debug_uselessdata) and os.path.exists(path):
        print(f'Loading cache: {path}')
        return _load_cache(input_file, path)
    print(f'Building (slow), will cache to: {path}')
    dataset, data, ratinabox_info = _build_one(input_file)
    if not debug_uselessdata:
        _save_cache(path, dataset, data, ratinabox_info)
    return dataset, data, ratinabox_info

train_dataset, train_data, ratinabox_info = _get_or_build(config['intrainfile'])
val_dataset,   val_data,   _              = _get_or_build(config['invalfile'])

# rehydrate ratinabox_info if you still use it as live objects downstream
ratinabox_info = apf_ratinabox.rehydrate_data(ratinabox_info)

train_dataloader = apf.dataset.DataLoader(train_dataset,
                                        batch_size=config['batch_size'],
                                        shuffle=True, pin_memory=True)
val_dataloader = apf.dataset.DataLoader(val_dataset,
                                        batch_size=config['batch_size'],
                                        shuffle=False, pin_memory=True)

model = res['model']
criterion = res['criterion']
optimizer = res['optimizer']
lr_scheduler = res['lr_scheduler']
opt_model = res['opt_model']
modeltype_str = res['modeltype_str']
savetime = res['model_savetime']
loss_epoch = res['loss_epoch']
epoch = res['epoch']

train_dataset_params = {
    'input_noise_sigma': config['input_noise_sigma'],
}

ntrain_batches = len(train_dataloader)
num_training_steps = ntrain_batches * config['num_train_epochs']
trainexample = next(iter(train_dataloader))
ntimepoints_per_batch = trainexample['input'].shape[0]
last_val_loss = loss_epoch['val'][epoch].item()
if np.isnan(last_val_loss):
    last_val_loss = None


# %% [markdown]
# ## set up debug plots

# %%
debug_params = {'nplot': 3}

hdebug = {}
hdebug['train'] = apf_ratinabox.initialize_debug_plots(
    train_dataset, train_dataloader, ratinabox_info, name='Train', **debug_params)
hdebug['val'] = apf_ratinabox.initialize_debug_plots(
    val_dataset, val_dataloader, ratinabox_info, name='Val', **debug_params)
hloss = apf_ratinabox.initialize_loss_plots(loss_epoch)

if ISNOTEBOOK:
    plt.close(hdebug['train']['figsample'])
    plt.close(hdebug['val']['figsample'])
    plt.close(hloss['fig'])

def refresh_plots(hdebugin, prefix=''):
    if ISNOTEBOOK:
        if 'displayed_keys' not in hdebugin:
            hdebugin['displayed_keys'] = set()
        for k, fig in hdebugin.items():
            if not k.startswith('fig') or fig is None:
                continue
            fig.canvas.draw_idle()      # force matplotlib to render the new state
            display_id = f'{prefix}__{k}'
            if k in hdebugin['displayed_keys']:
                display(fig, display_id=display_id, update=True)
            else:
                display(fig, display_id=display_id)
                hdebugin['displayed_keys'].add(k)
    else:
        for k, fig in hdebugin.items():
            if not k.startswith('fig') or fig is None:
                continue
            fig.canvas.draw()
            fig.canvas.flush_events()
            
valexample = next(iter(val_dataloader))

def end_iter_hook(model=None, step=None, example=None, predfn=None, **kwargs):
    if step % config['niterplot'] != 0:
        return
    with torch.no_grad():
        trainpred = predfn(example['input'].to(device=device))
        valpred   = predfn(valexample['input'].to(device=device))
    apf_ratinabox.update_debug_plots(
        hdebug['train'], config, model, train_dataset, ratinabox_info,
        example, trainpred, name='Train', criterion=criterion, **debug_params)
    apf_ratinabox.update_debug_plots(
        hdebug['val'], config, model, val_dataset, ratinabox_info,
        valexample, valpred, name='Val', criterion=criterion, **debug_params)
    refresh_plots(hdebug['train'], 'train')
    refresh_plots(hdebug['val'], 'val')


def end_epoch_hook(loss_epoch=None, epoch=None, **kwargs):
    apf_ratinabox.update_loss_plots(hloss, loss_epoch)
    refresh_plots(hloss, 'loss')
    
if ISNOTEBOOK:
    refresh_plots(hdebug['train'], 'train')
    refresh_plots(hdebug['val'], 'val')
    refresh_plots(hloss, 'loss')
else:
    plt.ion()
    plt.show(block=False)

# %%
savefilestr = os.path.join(config['savedir'], f"fly{modeltype_str}_{savetime}")

train_args = utils.function_args_from_config(config,train)
train_args['train_dataloader'] = train_dataloader
train_args['val_dataloader'] = val_dataloader
train_args['model'] = model
train_args['loss_epoch'] = loss_epoch
train_args['end_epoch_hook'] = end_epoch_hook
train_args['end_iter_hook'] = end_iter_hook
train_args['optimizer'] = optimizer
train_args['lr_scheduler'] = lr_scheduler
# criterion hard-coded to mixed_causal_criterion
#train_args['criterion'] = criterion
train_args['start_epoch'] = epoch
train_args['savefilestr'] = savefilestr

# can override args here
train_args['num_train_epochs'] = 100
model, best_model, loss_epoch = train(**train_args)
