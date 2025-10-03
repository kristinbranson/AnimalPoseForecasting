# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: transformer
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Imports

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

from flyllm.prepare import init_flyllm
from flyllm.dataset import FlyTestDataset
from IPython.display import HTML

import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# %% [markdown]
# ### Configuration

# %%
# configuration parameters for this model

loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flypredvel_20241022_epoch200_20241023T140356.pth'
configfile = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241022.json'
outfigdir = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/figs'

# loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flypredpos_20241023_epoch200_20241028T165510.pth'
# configfile = '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predpos_20241023.json'

# set to None if you want to use the full data
#quickdebugdatafile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/tmp_small_usertrainval.pkl'
quickdebugdatafile = None

# which data to create and how to process
needtraindata = False
needvaldata = True
traindataprocess = 'test'
valdataprocess = 'test'

res = init_flyllm(configfile=configfile,mode='test',loadmodelfile=loadmodelfile,
                  quickdebugdatafile=quickdebugdatafile,needtraindata=needtraindata,needvaldata=needvaldata,
                  traindataprocess=traindataprocess,valdataprocess=valdataprocess)
                #   ,res=res,
                #   doinitconfig=False,doinitstate=False,doinitrawdata=False,doinitprocessdata=False,
                #   doinitdatasets=False)

# unpack the results
config = res['config']
valdata = res['valdata']
val_scale_perfly = res['val_scale_perfly']
valX = res['valX']
val_dataset = res['val_dataset']
val_dataloader = res['val_dataloader']
model = res['model']
device = res['device']
opt_model = res['opt_model']
train_src_mask = res['train_src_mask']

# %%
# try making context length longer and masking out earlier frames

print(train_src_mask.cpu().numpy()[:5,:5])
tri1 = np.tri(2*config['contextl']-1,k=0)
tri2 = np.tri(2*config['contextl']-1,k=-config['contextl'])
tri = np.log(tri1-tri2)
test_mask = torch.from_numpy(tri).to(device,dtype=torch.float32)
print(tri[:5,:5])
fig,ax = plt.subplots(1,2)
ax[0].imshow(train_src_mask.cpu().numpy())
ax[0].set_title('train_src_mask')
ax[1].imshow(tri)
ax[1].set_title('test_mask')

fig.savefig(f'{outfigdir}/band_mask.pdf')

quadtri1 = np.tri(4*config['contextl']-1,k=0)
quadtri2 = np.tri(4*config['contextl']-1,k=-config['contextl'])
quadtri = np.log(quadtri1-quadtri2)
quad_test_mask = torch.from_numpy(quadtri).to(device,dtype=torch.float32)
quad_test_mask1 = torch.from_numpy(np.log(quadtri1)).to(device,dtype=torch.float32)

val_dataset_small = FlyTestDataset(valX[1:2],config['contextl'],**dataset_params,need_metadata=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset_small,
                                              batch_size=256,
                                              shuffle=False,
                                              pin_memory=True
                                              )
double_val_dataset_small = FlyTestDataset(valX[1:2],2*config['contextl'],**dataset_params)
double_val_dataloader = torch.utils.data.DataLoader(double_val_dataset_small,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    pin_memory=True)
quadruple_val_dataset_small = FlyTestDataset(valX[1:2],4*config['contextl'],**dataset_params)
quadruple_val_dataloader = torch.utils.data.DataLoader(quadruple_val_dataset_small,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       pin_memory=True)
example = next(iter(val_dataloader))
double_example = next(iter(double_val_dataloader))
quadruple_example = next(iter(quadruple_val_dataloader))

with torch.no_grad():
    out = model(example['input'].to(device=device),train_src_mask)
    doubleout = model(double_example['input'].to(device=device),test_mask)
    quadrupleout = model(quadruple_example['input'].to(device=device),quad_test_mask)

for k in out:
    out[k] = out[k].cpu().numpy()
    doubleout[k] = doubleout[k].cpu().numpy()
    quadrupleout[k] = quadrupleout[k].cpu().numpy()

print(f"error in first 511 frames: {np.max(np.abs(doubleout['continuous'][0,:config['contextl']-1]-out['continuous'][0]))}")
fig,ax = plt.subplots(1,2,figsize=(20,5))
err = np.zeros(out['continuous'].shape[0])
for i in range(out['continuous'].shape[0]):
    err[i] = np.abs(out['continuous'][i,-1,:]-doubleout['continuous'][0,config['contextl']-2+i,:]).max()
ax[0].plot(err)
ax[0].set_title('double error')
quaderr = np.abs(quadrupleout['continuous'][0,:2*config['contextl']-1]-doubleout['continuous'][0]).max(axis=-1)
ax[1].plot(quaderr)
ax[1].set_title(f'quadruple error')

fig.savefig(f'{outfigdir}/band_vs_causal_mask_mismatch.pdf')

# mess up some inputs and see how far the error affects things

idxmess = range(50,55)
double_example_nan = {}
for k,v in double_example.items():
    double_example_nan[k] = v.clone()
double_example_nan['input'][:,idxmess,:] = (torch.rand_like(double_example_nan['input'][:,idxmess,:])-.5)*9999999
with torch.no_grad():
    doubleoutnan = model(double_example_nan['input'].to(device=device),test_mask)
for k in doubleoutnan:
    doubleoutnan[k] = doubleoutnan[k].cpu().numpy()
    
quadruple_example_nan = {}
for k,v in quadruple_example.items():
    quadruple_example_nan[k] = v.clone()
quadruple_example_nan['input'][:,idxmess,:] = (torch.rand_like(quadruple_example_nan['input'][:,idxmess,:])-.5)*9999999
with torch.no_grad():
    quadrupleoutnan = model(quadruple_example_nan['input'].to(device=device),quad_test_mask)
    quadrupleout1 = model(quadruple_example['input'].to(device=device),quad_test_mask1)
    quadrupleoutnan1 = model(quadruple_example_nan['input'].to(device=device),quad_test_mask1)
for k in quadrupleoutnan:
    quadrupleoutnan[k] = quadrupleoutnan[k].cpu().numpy()
    quadrupleout1[k] = quadrupleout1[k].cpu().numpy()
    quadrupleoutnan1[k] = quadrupleoutnan1[k].cpu().numpy()

np.count_nonzero(np.isnan(doubleoutnan['continuous'][0,:,0]))
fig,ax = plt.subplots(3,3,figsize=(20,10),sharex=True)
ax[0,0].plot(doubleout['continuous'][0,:,:])  
ax[0,0].set_title('doubleout')
ax[0,1].plot(doubleoutnan['continuous'][0,:,:])
ax[0,1].set_title('doubleoutnan')
err = np.mean(np.abs(doubleoutnan['continuous'][0,:,:]-doubleout['continuous'][0,:,:]),axis=1)
ax[0,2].plot(err)
ax[0,2].set_title('error')

ax[1,0].plot(quadrupleout['continuous'][0,:,:])
ax[1,0].set_title('quadrupleout')
ax[1,1].plot(quadrupleoutnan['continuous'][0,:,:])
ax[1,1].set_title('quadrupleoutnan')
err = np.mean(np.abs(quadrupleoutnan['continuous'][0,:,:]-quadrupleout['continuous'][0,:,:]),axis=1)
ax[1,2].plot(err)
ax[1,2].set_title('error')

ax[2,0].plot(quadrupleout1['continuous'][0,:,:])
ax[2,0].set_title('quadrupleout1')
ax[2,1].plot(quadrupleoutnan1['continuous'][0,:,:])
ax[2,1].set_title('quadrupleoutnan1')
err = np.mean(np.abs(quadrupleoutnan1['continuous'][0,:,:]-quadrupleout1['continuous'][0,:,:]),axis=1)
ax[2,2].plot(err)
ax[2,2].set_title('error')

fig.savefig(f'{outfigdir}/band_mask_input_perturb_propagate.pdf')
