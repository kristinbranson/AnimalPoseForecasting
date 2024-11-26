# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
import datetime
import os
from matplotlib import animation
import pickle

from apf.utils import get_dct_matrix, compute_npad, save_animation
from apf.data import process_test_data, interval_all, debug_less_data, chunk_data
from apf.io import read_config, get_modeltype_str, load_and_filter_data, save_model, load_model, parse_modelfile, load_config_from_model_file
from flyllm.config import scalenames, nfeatures, DEFAULTCONFIGFILE, featglobal, posenames, keypointnames
from flyllm.features import compute_features, sanity_check_tspred, get_sensory_feature_idx, compute_scale_perfly, compute_pose_distribution_stats
from flyllm.dataset import FlyTestDataset, FlyMLMDataset
from flyllm.pose import FlyExample, FlyPoseLabels, FlyObservationInputs
from flyllm.plotting import (
    initialize_debug_plots, 
    initialize_loss_plots, 
    update_debug_plots,
    update_loss_plots,
    debug_plot_global_histograms, 
    debug_plot_dct_relative_error, 
    debug_plot_global_error, 
    debug_plot_predictions_vs_labels,
    select_featidx_plot,
)
from apf.models import (
    initialize_model, 
    initialize_loss, 
    compute_loss,
    generate_square_full_mask, 
    sanity_check_temporal_dep,
    criterion_wrapper,
    update_loss_nepochs,
    stack_batch_list,
)
from flyllm.simulation import animate_predict_open_loop
from flyllm.prediction import predict_all
from IPython.display import HTML

import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

mpl_backend = plt.get_backend()
if mpl_backend == 'inline':
    from IPython import display

LOG.info('CUDA available: ' + str(torch.cuda.is_available()))
LOG.info('matplotlib backend: ' + mpl_backend)

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

# whether to create training data
needtraindata = True
needvaldata = False

#configfile = "/groups/branson/home/eyjolfsdottire/code/MABe2022/config_fly_llm_multitimeglob_discrete_20230907.json"
config = read_config(configfile,
                     default_configfile=DEFAULTCONFIGFILE,
                     get_sensory_feature_idx=get_sensory_feature_idx,
                     featglobal=featglobal,
                     posenames=posenames)

# set loadmodelfile from config if not specified
if loadmodelfile is None and 'loadmodelfile' in config:
    loadmodelfile = config['loadmodelfile']

load_config_from_model_file(loadmodelfile=loadmodelfile,config=config)
assert 'dataset_params' in config, 'dataset_params not in config'

# seed the random number generators
np.random.seed(config['numpy_seed'])
torch.manual_seed(config['torch_seed'])

# set device (cuda/cpu)
device = torch.device(config['device'])
if device.type == 'cuda':
    assert torch.cuda.is_available(), 'CUDA is not available'

pred_nframes_skip = 20

# where to save predictions
# remove extension from loadmodelfile
savepredfile = loadmodelfile.split('.')[0] + f'_all_pred_skip_{pred_nframes_skip}.npz'

# where to save pose statistics
posestatsfile = loadmodelfile.split('.')[0] + '_posestats.npz'

# %%
# load raw data
if quickdebugdatafile is None:
    if needtraindata:
        data, scale_perfly = load_and_filter_data(config['intrainfile'], config, compute_scale_perfly,
                                                keypointnames=keypointnames)
        print(f"training data X shape: {data['X'].shape}")
    if needvaldata:
        valdata, val_scale_perfly = load_and_filter_data(config['invalfile'], config, compute_scale_perfly,
                                                        keypointnames=keypointnames)
        print(f"val data X shape: {valdata['X'].shape}")
    #LOG.warning('DEBUGGING!!!')
    #debug_less_data(data, 10000000)
    #debug_less_data(valdata, 10000)
else:
    print("Loading data from quick debug data file ", quickdebugdatafile)
    with open(quickdebugdatafile,'rb') as f:
        tmp = pickle.load(f)
        data = tmp['data']
        scale_perfly = tmp['scale_perfly']
        valdata = tmp['valdata']
        val_scale_perfly = tmp['val_scale_perfly']

# %%

# compute features

# process data

# if using discrete cosine transform, create dct matrix
# this didn't seem to work well, so probably won't use in the future
if config['dct_tau'] is not None and config['dct_tau'] > 0:
    dct_m, idct_m = get_dct_matrix(config['dct_tau'])
    # this gives the maximum of 
    #   a) max number of frames to lookahead or 
    #   b) dct_tau (number of timepoints for cosine transform)
else:
    dct_m = None
    idct_m = None

# how much to pad outputs by -- depends on how many frames into the future we will predict
npad = compute_npad(config['tspred_global'], dct_m)
chunk_data_params = {'npad': npad, 'minnframes': config['contextl']+1}

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

# process the data

if needtraindata:
    LOG.info('Processing training data...')
    X = chunk_data(data, config['contextl'], reparamfun, npad=chunk_data_params['npad'])
    #X = process_test_data(data, reparamfun, **chunk_data_params)
    LOG.info(f'{len(X)} training ids, total of {sum([x['input'].shape[0] for x in X])} time points')
iff needvaldata:
    LOG.info('Processing val data...')
    valX = process_test_data(valdata, val_reparamfun, **chunk_data_params)
    LOG.info(f' {len(valX)} val ids, total of {sum([x['input'].shape[0] for x in valX])} time points')

# %%
from flyllm.features import keypointnames, PXPERMM
bthorax = data['X'][keypointnames.index('base_thorax')]
lthorax = data['X'][keypointnames.index('left_front_thorax')]
rthorax = data['X'][keypointnames.index('right_front_thorax')]
fthorax = (lthorax + rthorax) / 2.
d = np.sqrt(np.sum((bthorax - fthorax)**2, axis=0))
print(f'mean distance between front and back of thorax: {np.nanmean(d)}, std: {np.nanstd(d)}')
print(f'pxpermm = {PXPERMM}, sig_tracking = {.25 / PXPERMM}')

# %%
# how does angle noise vary with kp noise -- distance between front and back of thorax is 1

ltrue = np.array([-.5,.5])
rtrue = np.array([.5,.5])
btrue = np.array([0,-.5])
ftrue = (ltrue+rtrue)/2
dtrue = ftrue-btrue

sigmakps = np.linspace(0,.03,100)
sigmathetas = np.zeros_like(sigmakps)
sigmads = np.zeros_like(sigmakps)
n = 1000

fig,ax = plt.subplots(2,2,figsize=(10,10))
colors = plt.cm.viridis(np.linspace(0,1,len(sigmakps)))
for i, sigmakp in enumerate(sigmakps):
    nul = np.random.normal(scale=sigmakp, size=(2,n))
    nur = np.random.normal(scale=sigmakp, size=(2,n))
    nub = np.random.normal(scale=sigmakp, size=(2,n))
    l = nul + ltrue[:, None]
    r = nur + rtrue[:, None]
    b = nub + btrue[:, None]
    f = (l+r)/2
    d = f-b
    errkpt = np.linalg.norm(d-dtrue[:,None],axis=0)
    theta = np.arctan2(d[1],d[0])-np.pi/2
    ax[0,0].plot(np.sort(theta*180/np.pi),label=f'{sigmakp:.2f}',color=colors[i],lw=.5)
    ax[1,0].plot(np.sort(errkpt),label=f'{sigmakp:.2f}',color=colors[i],lw=.5)
    sigmathetas[i] = np.std(theta)
    sigmads[i] = np.mean(errkpt)

ax[0,0].set_xlabel('sample')
ax[0,0].set_ylabel('theta (deg)')
ax[1,0].set_xlabel('sample')
ax[1,0].set_ylabel('d (mm)')
ax[0,1].plot(sigmakps,sigmathetas*180/np.pi,'.-')
ax[0,1].set_xlabel('sigma kp')
ax[0,1].set_ylabel('sigma theta (deg)')
ax[1,1].plot(sigmakps,sigmads,'.-')
ax[1,1].set_xlabel('sigma kp')
ax[1,1].set_ylabel('mean d (mm)')

# %%
from flyllm.features import compute_noise_params
newepsilon,all_movement,all_delta = compute_noise_params(data,scale_perfly,compute_pose_vel=True,return_extra=True)
catmovement = np.concatenate(list(all_movement.values()),axis=1)
catdelta = np.concatenate(list(all_delta.values()),axis=1)

prctiles = np.linspace(0,100,101)
vmovement = np.percentile(catmovement[config['discreteidx'],:,0],prctiles,axis=1)
vdelta = np.percentile(catdelta[config['discreteidx'],:,0],prctiles,axis=1)

# %%
# fit gaussian to movement
mumovement = np.mean(catmovement[config['discreteidx'],:,0],axis=1)
sigmovement = np.std(catmovement[config['discreteidx'],:,0],axis=1)
normdata = np.zeros((len(config['discreteidx']),catmovement.shape[1]))
for i in range(len(config['discreteidx'])):
    normdata[i] = np.random.normal(size=catmovement.shape[1],scale=sigmovement[i],loc=mumovement[i])

# fit heavy tailed distributions - t and cauchy
import scipy.stats
t_df_movement = np.zeros(len(config['discreteidx']))
t_loc_movement = np.zeros(len(config['discreteidx']))
t_scale_movement = np.zeros(len(config['discreteidx']))
cauchy_loc_movement = np.zeros(len(config['discreteidx']))
cauchy_scale_movement = np.zeros(len(config['discreteidx']))

for i,feati in tqdm.tqdm(enumerate(config['discreteidx']),total=len(config['discreteidx'])):
    t_df_movement[i],t_loc_movement[i],t_scale_movement[i] = scipy.stats.t.fit(catmovement[feati,:,0],1,loc=0,scale=1)
    cauchy_loc_movement[i],cauchy_scale_movement[i] = scipy.stats.cauchy.fit(catmovement[feati,:,0],loc=0,scale=1)
t_data = np.zeros((len(config['discreteidx']),catmovement.shape[1]))
for i in range(len(config['discreteidx'])):
    t_data[i] = scipy.stats.t.rvs(df=t_df_movement[i],loc=t_loc_movement[i],scale=t_scale_movement[i],size=catmovement.shape[1])
cauchy_data = np.zeros((len(config['discreteidx']),catmovement.shape[1]))
for i in range(len(config['discreteidx'])):
    cauchy_data[i] = scipy.stats.cauchy.rvs(loc=cauchy_loc_movement[i],scale=cauchy_scale_movement[i],size=catmovement.shape[1])

vnorm = np.percentile(normdata,prctiles,axis=1)
vt = np.percentile(t_data,prctiles,axis=1)
vcauchy = np.percentile(cauchy_data,prctiles,axis=1)



# %%

print(f't_df_movement: {t_df_movement}')
print(f't_loc_movement: {t_loc_movement}')
print(f't_scale_movement: {t_scale_movement}')

print(f'cauchy_loc_movement: {cauchy_loc_movement}')
print(f'cauchy_scale_movement: {cauchy_scale_movement}')


# %%
nct_df_movement = np.array([1.17467831, 0.,0.,0.,0.])
nct_nc_movement = np.array([1.53077322,0,0,0,0])
nct_loc_movement = np.array([-0.01773452,0,0,0,0])
nct_scale_movement = np.array([0.01829578,0,0,0,0])

# %%
st_a_movement = np.zeros(len(config['discreteidx']))
st_b_movement = np.zeros(len(config['discreteidx']))
st_loc_movement = np.zeros(len(config['discreteidx']))
st_scale_movement = np.zeros(len(config['discreteidx']))

i = 0
feati = config['discreteidx'][i]
st_a_movement[i],st_b_movement[i],st_loc_movement[i],st_scale_movement[i] = \
    scipy.stats.jf_skew_t.fit(catmovement[feati,:,0],loc=0)

print(f'st_a_movement: {st_a_movement}')
print(f'st_b_movement: {st_b_movement}')
print(f'st_loc_movement: {st_loc_movement}')
print(f'st_scale_movement: {st_scale_movement}')

st_data = np.zeros((len(config['discreteidx']),catmovement.shape[1]))
st_data[i] = scipy.stats.jf_skew_t.rvs(st_a_movement[i],st_b_movement[i],st_loc_movement[i],st_scale_movement[i],size=catmovement.shape[1])

vst = np.percentile(st_data,prctiles,axis=1)



# %%
laplace_loc_movement = np.zeros(len(config['discreteidx']))
laplace_scale_movement = np.zeros(len(config['discreteidx']))
for i,feati in tqdm.tqdm(enumerate(config['discreteidx']),total=len(config['discreteidx'])):
    laplace_loc_movement[i] = np.nanmedian(catmovement[feati,:,0])
    laplace_scale_movement[i] = np.mean(np.abs(catmovement[feati,:,0]-laplace_loc_movement[i]))
    
print(f'laplace_loc_movement: {laplace_loc_movement}')
print(f'laplace_scale_movement: {laplace_scale_movement}')

laplace_data = np.zeros((len(config['discreteidx']),catmovement.shape[1]))
for i in range(len(config['discreteidx'])):
    laplace_data[i] = scipy.stats.laplace.rvs(loc=laplace_loc_movement[i],scale=laplace_scale_movement[i],size=catmovement.shape[1])

vlaplace = np.percentile(laplace_data,prctiles,axis=1)

# %%
# look at the actual distribution of tracking errors

from apf.utils import read_matfile
from flyllm.features import kp2feat, PXPERMM
aptgtdata = read_matfile('/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/APTTrueVsPred.mat')
# massage to be in the order as MABe data
apt2mabekpidx = np.r_[np.array([17,19]),np.arange(17)]
apt2mabekpidx
true_kpts = aptgtdata['true_kpts'][:,apt2mabekpidx,:].transpose((1,2,0))/PXPERMM
pred_kpts = aptgtdata['pred_kpts'][:,apt2mabekpidx,:].transpose((1,2,0))/PXPERMM
n = true_kpts.shape[-1]

kpts = np.concatenate([true_kpts[...,None],pred_kpts[...,None]],axis=3)
delta_kpts = pred_kpts-true_kpts

# can either draw from a multivariate normal with empirical covariance, or
# draw from the actual error vectors
sig = np.cov(delta_kpts.reshape(-1,n),rowvar=True)


# %%
epsilon_uncorrelated_gausian_std = compute_noise_params(data,scale_perfly,compute_pose_vel=True,compute_std=True)
print(f'Epsilon when adding uncorrelated gaussian noise to keypoints and computing standard deviation\n:{epsilon_uncorrelated_gausian_std}')

compute_prctile = 50
epsilon_uncorrelated_gaussian_prctile = compute_noise_params(data,scale_perfly,compute_pose_vel=True,
                                                             compute_prctile=compute_prctile,compute_std=False)
print(f'Epsilon when adding uncorrelated gaussian noise to keypoints and computing {compute_prctile}th percentile\n:{epsilon_uncorrelated_gaussian_prctile}')

epsilon_correlated_gaussian_std,_,delta_correlated = compute_noise_params(data,scale_perfly,compute_pose_vel=True,
                                                                          sig_tracking=sig,compute_std=True,
                                                                          return_extra=True)
print(f'Epsilon when adding correlated gaussian noise to keypoints and computing standard deviation:\n{epsilon_correlated_gaussian_std}')

epsilon_correlated_gaussian_prctile = compute_noise_params(data,scale_perfly,compute_pose_vel=True,
                                                           sig_tracking=sig,compute_prctile=compute_prctile,
                                                           compute_std=False)
print(f'Epsilon when adding correlated gaussian noise to keypoints and computing {compute_prctile}th percentile:\n{epsilon_correlated_gaussian_prctile}')

epsilon_correlated_sample_std,_,delta_sample = compute_noise_params(data,scale_perfly,compute_pose_vel=True,delta_kpts=delta_kpts,compute_std=True,
                                                                    return_extra=True)
print(f'Epsilon when using actual error vectors and computing standard deviation:\n{epsilon_correlated_sample_std}')

epsilon_correlated_sample_prctile = compute_noise_params(data,scale_perfly,compute_pose_vel=True,delta_kpts=delta_kpts,
                                                         compute_std=False,compute_prctile=compute_prctile)
print(f'Epsilon when using actual error vectors and computing {compute_prctile}th percentile:\n{epsilon_correlated_sample_prctile}')

epsilon_uncorrelated_sample_std,_,delta_uncorrelated_sample = \
    compute_noise_params(data,scale_perfly,compute_pose_vel=True,delta_kpts=delta_kpts,compute_std=True,
                         return_extra=True,sample_correlated=False)
print(f'Epsilon when using actual error vectors without correlation and computing standard deviation:\n{epsilon_uncorrelated_sample_std}')

epsilon_uncorrelated_sample_prctile = compute_noise_params(data,scale_perfly,compute_pose_vel=True,delta_kpts=delta_kpts,
                                                           compute_std=False,compute_prctile=compute_prctile,sample_correlated=False)
print(f'Epsilon when using actual error vectors without correlation and computing {compute_prctile}th percentile:\n{epsilon_uncorrelated_sample_prctile}')

# %%
catdelta = np.concatenate(list(all_delta.values()),axis=1)
vdelta = np.percentile(catdelta[config['discreteidx'],:,0],prctiles,axis=1)
catdelta = np.concatenate(list(delta_correlated.values()),axis=1)
vdelta_correlated = np.percentile(catdelta[config['discreteidx'],:,0],prctiles,axis=1)
catdelta = np.concatenate(list(delta_sample.values()),axis=1)
vdelta_sample = np.percentile(catdelta[config['discreteidx'],:,0],prctiles,axis=1)
catdelta = np.concatenate(list(delta_uncorrelated_sample.values()),axis=1)
vdelta_uncorrelated_sample = np.percentile(catdelta[config['discreteidx'],:,0],prctiles,axis=1)

# %%
from apf.utils import compute_ylim
nr = 2
nc = int(np.ceil(len(config['discreteidx'])/nr))
fig,ax = plt.subplots(nr,nc,figsize=(nc*5,nr*5),sharex=True)
ax = ax.flatten()

featnames = posenames.copy()
featnames[0] = 'forward'
featnames[1] = 'sideways'

print(f'normal fit:\nmu = {mumovement}\nsigma = {sigmovement}')
print(f't fit:\ndf = {t_df_movement}\nloc = {t_loc_movement}\nscale = {t_scale_movement}')
print(f'cauchy fit:\nloc = {cauchy_loc_movement}\nscale = {cauchy_scale_movement}')

for i,feati in enumerate(config['discreteidx']):
    hylim = []
    ax[i].plot(prctiles[[1,-1]],epsilon_uncorrelated_gausian_std[feati]*np.ones(2),':',color='gray')
    ax[i].plot(prctiles[[1,-1]],-epsilon_uncorrelated_gausian_std[feati]*np.ones(2),':',color='gray')
    ax[i].plot(prctiles[[1,-1]],np.zeros(2),':',color='gray')
    hylim += ax[i].plot(prctiles[1:-1],vmovement[1:-1,i],'k-',label='data',lw=5)
    if i == 0:
        ax[i].plot(prctiles[1:-1],vst[1:-1,i],'-',label='t fit')
    else:
        ax[i].plot(prctiles[1:-1],vt[1:-1,i],'-',label='t fit')
    ax[i].plot(prctiles[1:-1],vcauchy[1:-1,i],'-',label='cauchy fit')
    ax[i].plot(prctiles[1:-1],vlaplace[1:-1,i],'-',label='laplace fit')
    hylim += ax[i].plot(prctiles[1:-1],vnorm[1:-1,i],'-',label='gaussian fit')
    ax[i].set_title(f'{featnames[feati]}')
    ax[i].set_ylim(compute_ylim(hylim,margin=.05))
ax[0].set_xlabel('Percentile')
ax[0].legend()

fig.savefig(os.path.join(outfigdir,'pose_velocity_fits.pdf'))

fig,ax = plt.subplots(nr,nc,figsize=(nc*5,nr*5),sharex=True)
ax = ax.flatten()
for i,feati in enumerate(config['discreteidx']):
    ax[i].plot(prctiles[[1,-1]],epsilon_uncorrelated_gausian_std[feati]*np.ones(2),':',color='gray')
    ax[i].plot(prctiles[[1,-1]],-epsilon_uncorrelated_gausian_std[feati]*np.ones(2),':',color='gray')
    ax[i].plot(prctiles[[1,-1]],np.zeros(2),':',color='gray')
    hylim += ax[i].plot(prctiles[1:-1],vmovement[1:-1,i],'k-',label='data',lw=5)
    hylim += ax[i].plot(prctiles[1:-1],vdelta[1:-1,i],'-',label='uncorrelated gaussian noise')
    ax[i].plot(prctiles[1:-1],vdelta_correlated[1:-1,i],'-',label='correlated gaussian noise')
    ax[i].plot(prctiles[1:-1],vdelta_sample[1:-1,i],'-',label='correlated sample noise')
    ax[i].plot(prctiles[1:-1],vdelta_uncorrelated_sample[1:-1,i],'-',label='uncorrelated sample noise')
    ax[i].set_title(f'{featnames[feati]}')
    ax[i].set_ylim(compute_ylim(hylim,margin=.05))

ax[0].set_xlabel('Percentile')
ax[0].legend()

fig.savefig(os.path.join(outfigdir,'pose_velocity_tracking_noise.pdf'))

# %%
# plot percentiles of the keypoint deltas -- combining keypoints, x and y

nkpts = delta_kpts.shape[0]
vkpt = np.percentile(delta_kpts,prctiles)
vkpt.shape
sigkpt = np.std(delta_kpts)
madkpt = np.mean(np.abs(delta_kpts))
vkptnormal = np.percentile(np.random.normal(size=np.prod(delta_kpts.shape),scale=sigkpt),prctiles)
vkptlaplace = np.percentile(np.random.laplace(loc=0,scale=madkpt,size=np.prod(delta_kpts.shape)),prctiles)
kpt_cauchy_loc,kpt_cauchy_scale = scipy.stats.cauchy.fit(delta_kpts.flatten(),loc=0,scale=10)
vkptcauchy = np.percentile(scipy.stats.cauchy.rvs(loc=kpt_cauchy_loc,scale=kpt_cauchy_scale,size=np.prod(delta_kpts.shape)),prctiles)

fig,ax = plt.subplots()
h = ax.plot(prctiles[1:-1],vkpt[1:-1],'k-',label='data',lw=3)
ax.plot(prctiles[1:-1],vkptcauchy[1:-1],label='cauchy')
ax.plot(prctiles[1:-1],vkptlaplace[1:-1],label='laplace')
ax.plot(prctiles[1:-1],vkptnormal[1:-1],label='normal')
ax.set_ylim(compute_ylim(h,margin=.05))
ax.legend()

fig.savefig(os.path.join(outfigdir,'keypoint_tracking_noise.pdf'))


# %%
# histogram data
nbins = 100
nr = 2
nc = int(np.ceil(len(config['discreteidx'])/nr))
fig,ax = plt.subplots(nr,nc,figsize=(nc*5,nr*5))
ax = ax.flatten()
for i,feati in enumerate(config['discreteidx']):
    counts,binedges = np.histogram(catmovement[feati,:,0],range=(vmovement[1,i],vmovement[-2,i]),bins=nbins)
    counts = counts + 1
    counts = counts/np.sum(counts)
    bincenters = (binedges[:-1]+binedges[1:])/2
    ax[i].plot(bincenters,counts,'k.-')
    ax[i].set_yscale('log')
    ax[i].set_title(f'{featnames[feati]}')
    
fig,ax = plt.subplots(nr,nc,figsize=(nc*5,nr*5))
ax = ax.flatten()
for i,feati in enumerate(config['discreteidx']):
    ax[i].plot(vmovement[1:-1,i],'k.-')
    ax[i].set_title(f'{featnames[feati]}')


# %%
old_epsilon = np.array([0.01322199, 0.01349601, 0.02598117, 0.02033068, 0.01859137, 0.05900833, 0.05741996, 0.02196621, 0.0558432 , 0.02196207, 0.05579313, 0.02626397, 0.12436298, 0.02276208, 0.04065661, 0.02275847, 0.04038093, 0.02625952, 0.12736998, 0.02628111, 0.13280959, 0.02628102, 0.13049692, 0.02439783, 0.0367505, 0.02445746, 0.03742915, 0.02930942, 0.02496748])
print(epsilon_uncorrelated_gausian_std/5)
new_epsilon = old_epsilon.copy()
new_epsilon[2:] = 0
# print new_epsilon with commas between each entry, print 0 if the entry is 0
print(','.join([f'{x:.6f}' if x != 0 else '0' for x in new_epsilon]))
