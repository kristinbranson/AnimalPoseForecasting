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
import synthrat.config
from synthrat.config import read_config_kwargs
import apf.io

res = apf.io.init_config(configfile=configfile,mode='train',
                         read_config_kwargs=read_config_kwargs)
print(res['config'])
apf_ratinabox.make_dataset(config=res['config'],
                           filename=res['config']['intrainfile'],
                           debug=True)
