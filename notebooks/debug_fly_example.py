# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
## Imports

# %load_ext autoreload
# %autoreload 2

import numpy as np
import torch
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pickle
from flyllm.io import read_config
from flyllm.utils import get_dct_matrix, compute_npad
from flyllm.config import featglobal, featrelative
from flyllm.features import compute_features
from flyllm.data import load_and_filter_data, chunk_data, debug_less_data
from flyllm.dataset import FlyMLMDataset
from flyllm.pose import PoseLabels, FlyExample, ObservationInputs
from flyllm.features import kp2feat
from flyllm.plotting import debug_plot_pose, debug_plot_sample, debug_plot_batch_traj

# %%
## Set parameters, read in data, config
tmpsavefile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/tmp_small_usertrainval.pkl'
configfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/config_fly_llm_multitime_20230825.json'
# configuration parameters for this model
config = read_config(configfile)

# override parameters in config file for testing
# debug velocity representation
#config['compute_pose_vel'] = True
# debug dct
config['dct_tau'] = 4
# debug no multi time-scale predictions
#config['tspred_global'] = [1,]
#config['discrete_tspred'] = [1,]

# read in data, select a subset and save if we don't have the small dataset computed already
if os.path.exists(tmpsavefile):
  with open(tmpsavefile,'rb') as f:
    tmp = pickle.load(f)
    data = tmp['data']
    scale_perfly = tmp['scale_perfly']
else:
  data,scale_perfly = load_and_filter_data(config['intrainfile'],config)
  valdata,val_scale_perfly = load_and_filter_data(config['invalfile'],config)
  T = 10000
  debug_less_data(data,T)
  debug_less_data(valdata,T)
  
  with open(tmpsavefile,'wb') as f:
    pickle.dump({'data': data, 'scale_perfly': scale_perfly, 'valdata': valdata, 'val_scale_perfly': val_scale_perfly},f)

# compute DCT matrix if needed
if config['dct_tau'] is not None and config['dct_tau'] > 0:
  dct_m,idct_m = get_dct_matrix(config['dct_tau'])
else:
  dct_m = None
  idct_m = None
  
# how much to pad outputs by -- depends on how many frames into the future we will predict
npad = compute_npad(config['tspred_global'],dct_m)
chunk_data_params = {'npad': npad}

compute_feature_params = {
    "simplify_out": config['simplify_out'],
    "simplify_in": config['simplify_in'],
    "dct_m": dct_m,
    "tspred_global": config['tspred_global'],
    "compute_pose_vel": config['compute_pose_vel'],
    "discreteidx": config['discreteidx'],
}
# function for computing features
reparamfun = lambda x,id,flynum,**kwargs: compute_features(x,id,flynum,scale_perfly,outtype=np.float32,
                                                          **compute_feature_params,**kwargs)

# chunk the data if we didn't load the pre-chunked cache file
print('Chunking training data...')
X = chunk_data(data,config['contextl'],reparamfun,**chunk_data_params)

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
  'dct_ms': (dct_m,idct_m),
  'tspred_global': config['tspred_global'],
  'discrete_tspred': config['discrete_tspred'],
  'compute_pose_vel': config['compute_pose_vel'],
}
train_dataset_params = {
  'input_noise_sigma': config['input_noise_sigma'],
}
compute_feature_params = {
    "simplify_out": config['simplify_out'],
    "simplify_in": config['simplify_in'],
    "dct_m": dct_m,
    "tspred_global": config['tspred_global'],
    "compute_pose_vel": config['compute_pose_vel'],
    "discreteidx": config['discreteidx'],
}

print('Creating training data set...')
train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)

# %%
## compare flyexample initialized from FlyMLMDataset and from keypoints directly

# flyexample from training dataset
flyexample = train_dataset.data[0]
contextlpad = train_dataset.contextl + npad + 1
Xkp = data['X'][:,:,flyexample.metadata['t0']:flyexample.metadata['t0']+contextlpad,:]
flyexample_kp = FlyExample(Xkp=Xkp,scale=scale_perfly[:,flyexample.metadata['id']],
                            flynum=flyexample.metadata['flynum'],metadata=flyexample.metadata,
                            **train_dataset.get_flyexample_params())
print(f"comparing flyexample initialized from FlyMLMDataset and from keypoints directly")
err = np.max(np.abs(flyexample_kp.labels.labels_raw['continuous']-flyexample.labels.labels_raw['continuous']))
print('max diff between continuous labels: %e'%err)
assert err < 1e-9
err = np.max(np.abs(flyexample_kp.labels.labels_raw['discrete']-flyexample.labels.labels_raw['discrete'])) 
print('max diff between discrete labels: %e'%err)
assert err < 1e-9
err = np.max(np.abs(flyexample_kp.labels.labels_raw['todiscretize']-flyexample.labels.labels_raw['todiscretize']))
print('max diff between todiscretize: %e'%err)  
assert err < 1e-9
multi = flyexample.labels.get_multi(use_todiscretize=True,zscored=False)
multi_kp = flyexample_kp.labels.get_multi(use_todiscretize=True,zscored=False)
err = np.max(np.abs(multi-multi_kp))
print('max diff between multi labels: %e'%err)
assert err < 1e-9

# %%
## compare pose feature representation from original chunked data and PoseLabels.multi
err_chunk_multi = np.max(np.abs(X[0]['labels']-flyexample.labels.get_multi(use_todiscretize=True,zscored=False)))
print('max diff between chunked labels and multi: %e'%err_chunk_multi)  
assert err_chunk_multi < 1e-3


# %%
## helper functions
def data_to_kp_from_metadata(data,metadata,ntimepoints):
  t0 = metadata['t0']
  flynum = metadata['flynum']
  id = metadata['id']
  datakp = data['X'][:,:,t0:t0+ntimepoints+1,flynum].transpose(2,0,1)
  return datakp,id

def compare_dicts(old_ex,new_ex,maxerr=None):
  for k,v in old_ex.items():
    err = 0.
    if not k in new_ex:
      print(f'Missing key {k}')
    elif type(v) is not type(new_ex[k]):
      print(f'Type mismatch for key {k}: {type(v)} vs {type(new_ex[k])}')
    elif type(v) is torch.Tensor:
      v = v.cpu().numpy()
      newv = new_ex[k]
      if type(newv) is torch.Tensor:
        newv = newv.cpu().numpy()
      err = np.nanmax(np.abs(v-newv))
      print(f'max diff {k}: {err:e}')
    elif type(v) is np.ndarray:
      err = np.nanmax(np.abs(v-new_ex[k]))
      print(f'max diff {k}: {err:e}')
    elif type(v) is dict:
      print(f'Comparing dict {k}')
      compare_dicts(v,new_ex[k])
    else:
      try:
        err = np.nanmax(np.abs(v-new_ex[k]))
        print(f'max diff {k}: {err:e}')
      except:
        print(f'not comparing {k}')
    if maxerr is not None:
      assert err < maxerr
      
  missing_keys = [k for k in new_ex.keys() if not k in old_ex]
  if len(missing_keys) > 0:
    print(f'Missing keys: {missing_keys}')

  return


# %%
## compare next frame representation

# extract frames associated with metadata in flyexample
contextl = flyexample.ntimepoints
datakp,id = data_to_kp_from_metadata(data,flyexample.metadata,contextl)
# compute next frame pose feature representation directly
datafeat = kp2feat(datakp.transpose(1,2,0),scale_perfly[:,id])[...,0].T
if config['compute_pose_vel']:

  # compare next frame rep computed from FlyMLMDataset to those from flyexample
  print('\nComparing next frame movements from train_dataset to those from flyexample')
  chunknext = train_dataset.get_next_movements(movements=X[0]['labels'])
  examplenext = flyexample.labels.get_next(use_todiscretize=True,zscored=False)
  err_chunk_next = np.max(np.abs(chunknext-examplenext))
  print('max diff between chunked labels and next: %e'%err_chunk_next)
  assert err_chunk_next < 1e-3
else:
  
  # compare next frame rep computed from FlyMLMDataset to those from flyexample
  print('\nComparing next frame pose feature representation from train_dataset to that from flyexample')
  chunknextcossin = train_dataset.get_next_movements(movements=X[0]['labels'])
  examplenextcossin = flyexample.labels.get_nextcossin(use_todiscretize=True,zscored=False)
  err_chunk_nextcossin = np.max(np.abs(chunknextcossin-examplenextcossin))
  print('max diff between chunked labels and nextcossin: %e'%err_chunk_nextcossin)
  assert err_chunk_nextcossin < 1e-3

  # compare next frame angle rep
  chunknext = train_dataset.convert_cos_sin_to_angle(chunknextcossin)
  examplenext = flyexample.labels.get_next(use_todiscretize=True,zscored=False)
  err_chunk_next = np.max(np.abs(chunknext-examplenext))
  print('max diff between chunked labels and next: %e'%err_chunk_next)
  assert err_chunk_next < 1e-3

  # compare next frame pose feature representation from train_dataset to that from flyexample
  examplefeat = flyexample.labels.get_next_pose(use_todiscretize=True,zscored=False)
  
  err_chunk_data_feat = np.max(np.abs(chunknext[:,featrelative]-datafeat[1:,featrelative]))
  print('max diff between chunked and data relative features: %e'%err_chunk_data_feat)
  assert err_chunk_data_feat < 1e-3

  err_example_chunk_feat = np.max(np.abs(chunknext[:,featrelative]-examplefeat[1:,featrelative]))
  print('max diff between chunked and example relative features: %e'%err_example_chunk_feat)
  assert err_example_chunk_feat < 1e-3

  err_example_data_global = np.max(np.abs(datafeat[:,featglobal]-examplefeat[:,featglobal]))
  print('max diff between data and example global features: %e'%err_example_data_global)
  assert err_example_data_global < 1e-3

  err_example_data_feat = np.max(np.abs(datafeat[:,featrelative]-examplefeat[:,featrelative]))
  print('max diff between data and example relative features: %e'%err_example_data_feat)
  assert err_example_data_feat < 1e-3
  
# compare next frame keypoints
examplekp = flyexample.labels.get_next_keypoints(use_todiscretize=True)
err_mean_example_data_kp = np.mean(np.abs(datakp[:]-examplekp))
print('mean diff between data and example keypoints: %e'%err_mean_example_data_kp)
err_max_example_data_kp = np.max(np.abs(datakp[:]-examplekp))
print('max diff between data and example keypoints: %e'%err_max_example_data_kp)

# plot
_ = debug_plot_pose(flyexample,data=data)  
# elements of the list tspred_global that are smaller than contextl
tsplot = [t for t in train_dataset.tspred_global if t < contextl]
_ = debug_plot_pose(flyexample,pred=flyexample,tsplot=tsplot)

# %%
## compare to legacy fly_llm code, which only supported velocities
if config['compute_pose_vel']:

  import flyllm.legacy.old_fly_llm as old_fly_llm
  
  old_dataset_params = dataset_params.copy()
  old_dataset_params.pop('compute_pose_vel')
  old_train_dataset = old_fly_llm.FlyMLMDataset(X,**train_dataset_params,**old_dataset_params,
                                                discretize_params=train_dataset.get_discretize_params(),
                                                zscore_params=train_dataset.get_zscore_params())

  new_ex = flyexample.get_train_example()
  old_ex = old_train_dataset[0]
  
  compare_dicts(old_ex,new_ex,maxerr=1e-3)

  print('\nComparing old fly llm code to new:')    
  mean_err_discrete = torch.mean(torch.sqrt(torch.sum((old_ex['labels_discrete']-new_ex['labels_discrete'])**2.,dim=-1))).item()
  print('mean error between old and new discrete labels: %e'%mean_err_discrete)
  assert mean_err_discrete < 1e-3

  max_err_continuous = torch.max(torch.abs(old_ex['labels']-new_ex['labels'])).item()
  print('max error between old and new continuous labels: %e'%max_err_continuous)
  assert max_err_continuous < 1e-3
  
  max_err_input = torch.max(torch.abs(new_ex['input']-old_ex['input'])).item()
  print('max error between old and new input: %e'%max_err_input)
  assert max_err_input < 1e-3
  
  max_err_init = torch.max(torch.abs(new_ex['init']-old_ex['init'])).item()
  print('max error between old and new init: %e'%max_err_init)
  assert max_err_init < 1e-3

# %%
## check global future predictions
print('\nChecking that representations of many frames into the future match')
flynum = flyexample.metadata['flynum']
for tpred in flyexample.labels.tspred_global:
  examplefuture = flyexample.labels.get_future_globalpos(use_todiscretize=True,tspred=tpred,zscored=False)
  t = flyexample.metadata['t0']+tpred
  datakpfuture = data['X'][:,:,t:t+contextl,flynum]
  datafeatfuture = kp2feat(datakpfuture,scale_perfly[:,id])[...,0].T  
  err_global_future = np.max(np.abs(datafeatfuture[:,featglobal]-examplefuture[:,0,:]))
  print(f'max diff between data and t+{tpred} global prediction: {err_global_future:e}')
  assert err_global_future < 1e-3

# %%
## check relative future predictions
if flyexample.labels.ntspred_relative > 1:
  examplefuture = flyexample.labels.get_future_relative_pose(zscored=False,use_todiscretize=True)
  for tpred in range(1,flyexample.labels.ntspred_relative):
    t = flyexample.metadata['t0']+tpred
    datakpfuture = data['X'][:,:,t:t+contextl,flynum]
    datafeatfuture = kp2feat(datakpfuture,scale_perfly[:,id])[...,0].T  
    err_relative_future = np.max(np.abs(datafeatfuture[:,featrelative]-examplefuture[:,tpred-1,:]))
    print(f'max diff between data and t+{tpred} relative prediction: {err_relative_future:e}')
    assert err_relative_future < 1e-3

# %%
## check get_train_example and constructor from train_example

# get a training example
print('\nComparing training example from dataset to creating a new FlyExample from that training example, and converting back to a training example')
trainexample = train_dataset[0]
flyexample1 = FlyExample(example_in=trainexample,dataset=train_dataset)
trainexample1 = flyexample1.get_train_example()
compare_dicts(trainexample,trainexample1,maxerr=1e-9)

# %%
## check constructor from batch

# initialize example from batch
print('\nComparing training batch to FlyExample created from that batch converted back to a training batch')
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=config['batch_size'],
                                              shuffle=False,
                                              pin_memory=True,
                                              )
raw_batch = next(iter(train_dataloader))
example_batch = FlyExample(example_in=raw_batch,dataset=train_dataset)
trainbatch1 = example_batch.get_train_example()
compare_dicts(raw_batch,trainbatch1,maxerr=1e-9)

_ = debug_plot_pose(example_batch,data=data)
_ = debug_plot_sample(example_batch,train_dataset)
fig = plt.figure(figsize=(12,12))
_ = debug_plot_batch_traj(example_batch,train_dataset,pred=raw_batch,nsamplesplot=1,fig=fig,ntsplot_global=3,ntsplot_relative=1)
#old_fly_llm.debug_plot_sample_inputs(old_train_dataset,raw_batch)

# %%
## check constructor from keypoints
flyexample = train_dataset.data[0]
train_example0 = flyexample.get_train_example()

Xkp0 = data['X'][:,:,flyexample.metadata['t0']:flyexample.metadata['t0']+contextlpad,:]
flyexample_kp = FlyExample(Xkp=Xkp0,scale=scale_perfly[:,flyexample.metadata['id']],
                            flynum=flyexample.metadata['flynum'],metadata=flyexample.metadata,
                            **flyexample.get_params())
train_example_kp = flyexample_kp.get_train_example()
print('Comparing FlyExample created from keypoints to FlyExample created from training example')
compare_dicts(train_example0,train_example_kp,maxerr=1e-9)
poselabels_kp = PoseLabels(Xkp=Xkp0[...,flynum],scale=scale_perfly[:,flyexample.metadata['id']],
                           metadata=flyexample.metadata,
                           **flyexample.get_poselabel_params()) 
train_labels_kp = poselabels_kp.get_train_labels(namingscheme='train')
print('\nComparing PoseLabels created from keypoints to FlyExample created from training example')
compare_dicts(train_labels_kp,train_example0,maxerr=1e-9)



# %%
## done
print('Goodbye!')
plt.show(block=True)
