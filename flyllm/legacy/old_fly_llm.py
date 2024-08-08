import math
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm,animation,colors
import copy
import typing
import tqdm
from itertools import compress
import re
import flyllm.legacy.MABeFlyUtils as mabe
import torch
import transformers
import warnings
import datetime
import collections
# import sklearn.preprocessing
#import sklearn.cluster
#import sklearn.decomposition
import json
import argparse
import pathlib
import pickle
import gzip

print('fly_llm...')

codedir = pathlib.Path(__file__).parent.resolve()
DEFAULTCONFIGFILE = os.path.join(codedir,'config_fly_llm_default.json')
assert os.path.exists(DEFAULTCONFIGFILE)

legtipnames = [
  'right_front_leg_tip',
  'right_middle_leg_tip',
  'right_back_leg_tip',
  'left_back_leg_tip',
  'left_middle_leg_tip',
  'left_front_leg_tip',
]

vision_kpnames_v1 = [
  'antennae_midpoint',
  'tip_abdomen',
  'left_middle_femur_base',
  'right_middle_femur_base',
]

touch_other_kpnames_v1 = [
  'antennae_midpoint',
  'left_front_thorax',
  'right_front_thorax',
  'base_thorax',
  'tip_abdomen',
]

SENSORY_PARAMS = {
  'n_oma': 72,
  'inner_arena_radius': 17.5, # in mm
  'outer_arena_radius': mabe.ARENA_RADIUS_MM,
  'arena_height': 3.5,
  'otherflies_vision_exp': .6,
  'touch_kpnames': mabe.keypointnames,
  #'touch_kpnames': legtipnames,
  'vision_kpnames': vision_kpnames_v1,
  'touch_other_kpnames': touch_other_kpnames_v1,
  'compute_otherflies_touch': True,
  'otherflies_touch_exp': 1.3,
  'otherflies_touch_mult': 0.3110326159171111, # set 20230807 based on courtship male data
}
SENSORY_PARAMS['otherflies_vision_mult'] = 1./((2.*mabe.ARENA_RADIUS_MM)**SENSORY_PARAMS['otherflies_vision_exp'])

"""
data = load_raw_npz_data(inxfile,inyfile=None)
inxfile: npz file with pose data
inyfile: npz file with categories (optional). Default = None. 
Creates dict data with the following fields:
'X': T x maxnflies x d0 array of floats containing pose data for all flies and frames
'videoidx': T x 1 array of ints containing index of video pose is computed from
'ids': T x maxnflies array of ints containing fly id
'frames': T x 1 array of ints containing video frame number
'y': if inyfile is not None, this will be a T x maxnflies x ncategories binary matrix 
     indicating supervised behavior categories
"""
def load_raw_npz_data(infile):
  
  data = {}
  with np.load(infile) as data1:
    for key in data1:
      print(f'loading {key}')
      data[key] = data1[key]
  print('data loaded')
    
  maxnflies = data['ids'].shape[1]
  # ids start at 1, make them start at 0
  data['ids'][data['ids']>=0] -= 1
  # starts of sequences, either because video changes or identity tracking issues
  # or because of filtering of training data
  isstart = (data['ids'][1:,:]!=data['ids'][:-1,:]) | \
    (data['frames'][1:,:] != (data['frames'][:-1,:]+1))
  isstart = np.concatenate((np.ones((1,maxnflies),dtype=bool),isstart),axis=0)
  
  data['isdata'] = data['ids'] >= 0
  data['isstart'] = isstart
  
  data['categories'] = list(data['categories'])
    
  return data

def filter_data_by_categories(data,categories):  
  iscategory = np.ones(data['y'].shape[1:],dtype=bool)
  for category in categories:
    if category == 'male':
      category = 'female'
      val = 0
    else:
      val = 1
    catidx = data['categories'].index(category)
    iscategory = iscategory & (data['y'][catidx,...]==val)
  data['isdata'] = data['isdata'] & iscategory

"""
y = interval_all(x,l)
Computes logical all over intervals of length l in the first dimension
y[i,j] is whether all entries in the l-length interval x[i:i+l,j] are true. 
x: input matrix of any shape. all will be computed over x[i:i+l,j,k]
outputs a matrix y of size (x.shape[0]-l,)+x.shape[1:]). 
"""
def interval_all(x,l):
  csx = np.concatenate((np.zeros((1,)+x.shape[1:],dtype=int),np.cumsum(x,axis=0)),axis=0)
  y = csx[l:-1,...]-csx[:-l-1,...] == l
  return y

"""
X = chunk_data(data,contextl,reparamfun)

"""
def chunk_data(data,contextl,reparamfun,npad=1):
  
  contextlpad = contextl + npad
  
  # all frames for the main fly must have real data
  allisdata = interval_all(data['isdata'],contextlpad)
  isnotsplit = interval_all(data['isstart']==False,contextlpad-1)[1:,...]
  canstart = np.logical_and(allisdata,isnotsplit)

  # X is nkeypts x 2 x T x nflies
  nkeypoints = data['X'].shape[0]
  T = data['X'].shape[2]
  maxnflies = data['X'].shape[3]
  assert T > 2*contextlpad, 'Assumption that data has more frames than 2*(contextl+1) is incorrect, code will fail'
  
  # last possible start frame = T - contextl
  maxt0 = canstart.shape[0]-1
  # X is a dict with chunked data
  X = []
  # loop through ids
  nframestotal = 0
  for flynum in tqdm.trange(maxnflies,desc='Fly'):
    # choose a first frame near the beginning, but offset a bit
    # first possible start
    canstartidx = np.nonzero(canstart[:,flynum])[0]
    if canstartidx.size == 0:
      continue

    mint0curr = canstartidx[0]
    # offset a bit
    t0 = mint0curr + np.random.randint(0,contextl,None)
    # find the next allowed frame
    if canstart[t0,flynum] == False:
      if not np.any(canstart[t0:,flynum]):
        continue
      t0 = np.nonzero(canstart[t0:,flynum])[0][0]+t0
    
    maxt0curr = canstartidx[-1]
    #maxt1curr = maxt0curr+contextlpad-1
    ndata = np.count_nonzero(data['isdata'][:,flynum])
    maxintervals = ndata//contextl+1
    for i in tqdm.trange(maxintervals,desc='Interval'):
      if t0 > maxt0:
        break
      # this is guaranteed to be < T
      t1 = t0+contextlpad-1
      id = data['ids'][t0,flynum]
      xcurr = reparamfun(data['X'][...,t0:t1+1,:],id,flynum,npad=npad)
      xcurr['metadata'] = {'flynum': flynum, 'id': id, 't0': t0, 'videoidx': data['videoidx'][t0,0], 'frame0': data['frames'][t0,0]}
      xcurr['categories'] = data['y'][:,t0:t1+1,flynum].astype(np.float32)
      X.append(xcurr)
      if t0+contextl >= maxt0curr:
        break
      elif canstart[t0+contextl,flynum]:
        t0 = t0+contextl
      else:
        t0 = np.nonzero(canstart[t1+1:,flynum])[0]
        if t0 is None or t0.size == 0:
          break
        t0 = t0[0] + t1 + 1
      nframestotal += contextl

  print(f'In total {nframestotal} frames of data after chunking')

  return X

def compute_noise_params(data,scale_perfly,sig_tracking=.25/mabe.PXPERMM,
                         simplify_out=None):

  # contextlpad = 2
  
  # # all frames for the main fly must have real data
  # allisdata = interval_all(data['isdata'],contextlpad)
  # isnotsplit = interval_all(data['isstart']==False,contextlpad-1)[1:,...]
  # canstart = np.logical_and(allisdata,isnotsplit)

  # X is nkeypts x 2 x T x nflies
  nkeypoints = data['X'].shape[0]
  T = data['X'].shape[2]
  maxnflies = data['X'].shape[3]

  alld = 0.
  n = 0
  # loop through ids
  print('Computing noise parameters...')
  for flynum in tqdm.trange(maxnflies):
    idx0 = data['isdata'][:,flynum] & (data['isstart'][:,flynum]==False)
    # bout starts and ends
    t0s = np.nonzero(np.r_[idx0[0],(idx0[:-1]==False) & (idx0[1:]==True)])[0]
    t1s = np.nonzero(np.r_[(idx0[:-1]==True) & (idx0[1:]==False),idx0[-1]])[0]
    
    for i in range(len(t0s)):
      t0 = t0s[i]
      t1 = t1s[i]
      id = data['ids'][t0,flynum]
      scale = scale_perfly[:,id]
      xkp = data['X'][:,:,t0:t1+1,flynum]
      relpose,globalpos = compute_pose_features(xkp,scale)
      movement = compute_movement(relpose=relpose,globalpos=globalpos,simplify=simplify_out)
      nu = np.random.normal(scale=sig_tracking,size=xkp.shape)
      relpose_pert,globalpos_pert = compute_pose_features(xkp+nu,scale)
      movement_pert = compute_movement(relpose=relpose_pert,globalpos=globalpos_pert,simplify=simplify_out)
      alld += np.nansum((movement_pert-movement)**2.,axis=1)
      ncurr = np.sum((np.isnan(movement)==False),axis=1)
      n+=ncurr
  
  epsilon = np.sqrt(alld/n)
  
  return epsilon.flatten()

def compute_scale_allflies(data):

  maxid = np.max(data['ids'])
  maxnflies = data['X'].shape[3]
  scale_perfly = None

  for flynum in range(maxnflies):

    idscurr = np.unique(data['ids'][data['ids'][:,flynum]>=0,flynum])
    for id in idscurr:
      idx = data['ids'][:,flynum] == id
      s = mabe.compute_scale_perfly(data['X'][...,idx,flynum])
      if scale_perfly is None:
        scale_perfly = np.zeros((s.size,maxid+1))
        scale_perfly[:] = np.nan
      else:
        assert(np.all(np.isnan(scale_perfly[:,id])))
      scale_perfly[:,id] = s.flatten()
      
  return scale_perfly


def plot_scale_stuff(data,scale_perfly):
  
  eps_sex = .05
  nbins = 20
  axlim_prctile = .5
  catidx = data['categories'].index('female')

  maxid = np.max(data['ids'])
  maxnflies = data['X'].shape[3]
  fracfemale = np.zeros(maxid+1)
  nframes = np.zeros(maxid+1)
  minnframes = 40000
  prctiles_compute = np.array([50,75,90,95,99,99.5,99.9])
  midleglength = np.zeros((maxid+1,len(prctiles_compute)))

  for flynum in range(maxnflies):

    idscurr = np.unique(data['ids'][data['ids'][:,flynum]>=0,flynum])
    for id in idscurr:
      idx = data['ids'][:,flynum] == id
      fracfemale[id] = np.count_nonzero(data['y'][catidx,idx,flynum]==1) / np.count_nonzero(idx)
      nframes[id] = np.count_nonzero(idx)
      xcurr = data['X'][:,:,idx,flynum]
      midtip = xcurr[mabe.keypointnames.index('left_middle_leg_tip'),:]
      midbase = xcurr[mabe.keypointnames.index('left_middle_femur_base'),:]
      lmidl = np.sqrt(np.sum((midtip-midbase)**2,axis=0))
      midtip = xcurr[mabe.keypointnames.index('right_middle_leg_tip'),:]
      midbase = xcurr[mabe.keypointnames.index('right_middle_femur_base'),:]
      rmidl = np.sqrt(np.sum((midtip-midbase)**2,axis=0))
      midleglength[id,:] = np.percentile(np.hstack((lmidl,rmidl)),prctiles_compute)

  plotnames = ['thorax_width', 'thorax_length', 'abdomen_length', 'head_width', 'head_height']
  plotidx = np.array([v in plotnames for v in mabe.scalenames])
  #plotidx = np.array([re.search('std',s) is None for s in mabe.scalenames])
  plotidx = np.nonzero(plotidx)[0]
  plotfly = nframes >= minnframes
  fig,ax = plt.subplots(len(plotidx),len(plotidx))
  fig.set_figheight(20)
  fig.set_figwidth(20)
  
  idxfemale = plotfly & (fracfemale>=1-eps_sex)
  idxmale = plotfly & (fracfemale<=eps_sex)

  lims = np.percentile(scale_perfly[:,plotfly],[axlim_prctile,100-axlim_prctile],axis=1)
  
  for ii in range(len(plotidx)):
    i = plotidx[ii]
    for jj in range(len(plotidx)):
      j = plotidx[jj]
      if i == j:
        binedges = np.linspace(lims[0,i],lims[1,i],nbins+1)
        ax[ii,ii].hist([scale_perfly[i,idxfemale],scale_perfly[i,idxmale]],
                     bins=nbins,range=(lims[0,i],lims[1,i]),
                     label=['female','male'])
        ax[ii,ii].set_ylabel('N. flies')
      else:
        ax[jj,ii].plot(scale_perfly[i,idxfemale],
                       scale_perfly[j,idxfemale],'.',label='female')
        ax[jj,ii].plot(scale_perfly[i,idxmale],
                       scale_perfly[j,idxmale],'.',label='male')
        ax[jj,ii].set_ylabel(mabe.scalenames[j])
        ax[jj,ii].set_xlabel(mabe.scalenames[i])
        ax[jj,ii].set_ylim(lims[:,j])
      ax[jj,ii].set_xlim(lims[:,i])
      ax[jj,ii].set_xlabel(mabe.scalenames[i])
  ax[0,0].legend()
  ax[0,1].legend()
  fig.tight_layout()
  
  scalefeat = 'thorax_length'
  scalei = mabe.scalenames.index(scalefeat)
  fig,ax = plt.subplots(2,len(prctiles_compute),sharex='row',sharey='row')
  fig.set_figwidth(20)
  fig.set_figheight(8)
  lims = np.percentile(midleglength[plotfly,:].flatten(),[axlim_prctile,100-axlim_prctile])
  for i in range(len(prctiles_compute)):
    ax[0,i].plot(scale_perfly[scalei,idxfemale],midleglength[idxfemale,i],'.',label='female')
    ax[0,i].plot(scale_perfly[scalei,idxmale],midleglength[idxmale,i],'.',label='male')
    ax[0,i].set_xlabel(scalefeat)
    ax[0,i].set_ylabel(f'{prctiles_compute[i]}th %ile middle leg length')
    ax[1,i].hist([midleglength[idxfemale,i],midleglength[idxmale,i]],
                 bins=nbins,range=(lims[0],lims[1]),label=['female','male'],
                 density=True)
    ax[1,i].set_xlabel(f'{prctiles_compute[i]}th %ile middle leg length')
    ax[1,i].set_ylabel('Density')
  ax[0,0].legend()
  ax[1,0].legend()
  fig.tight_layout()
  
  
  return
  


def pred_apply_fun(pred,fun):
  if isinstance(pred,dict):
    return {k: fun(v) for k,v in pred.items()}
  else:
    return fun(pred)

def debug_plot_otherflies_vision(t,xother,yother,xeye_main,yeye_main,theta_main,
                                 angle0,angle,dist,b_all,otherflies_vision,params):  
  npts = xother.shape[0]
  nflies = xother.shape[1]
  
  rplot = 2*params['outer_arena_radius']
  plt.figure()
  ax = plt.subplot(1,3,1)
  hother = ax.plot(xother[:,:,t],yother[:,:,t],'.-')
  ax.set_aspect('equal')
  #ax.plot(X[:,0,0,flynum],X[:,1,0,flynum],'k.')
  ax.plot(xeye_main[0,0,t],yeye_main[0,0,t],'r.')
  ax.plot([xeye_main[0,0,t],xeye_main[0,0,t]+rplot*np.cos(theta_main[0,0,t])],
          [yeye_main[0,0,t],yeye_main[0,0,t]+rplot*np.sin(theta_main[0,0,t])],'r--')
  for tmpfly in range(nflies):
    ax.plot(xeye_main[0,0,t]+np.c_[np.zeros((npts,1)),np.cos(angle0[:,tmpfly,t])*rplot].T,
            yeye_main[0,0,t]+np.c_[np.zeros((npts,1)),np.sin(angle0[:,tmpfly,t])*rplot].T,
            color=hother[tmpfly].get_color(),alpha=.5)

  ax = plt.subplot(1,3,2)  
  for tmpfly in range(nflies):
    ax.plot(np.c_[np.zeros((npts,1)),np.cos(angle[:,tmpfly,t])].T,
            np.c_[np.zeros((npts,1)),np.sin(angle[:,tmpfly,t])].T,
            color=hother[tmpfly].get_color(),alpha=.5)
  ax.plot(0,0,'r.')
  ax.plot([0,1],[0,0],'r--')
  ax.set_aspect('equal')

  ax = plt.subplot(1,3,3)
  for tmpfly in range(nflies):
    ax.plot(b_all[:,tmpfly,t],dist[:,tmpfly,t],'o',color=hother[tmpfly].get_color())
  ax.set_xlim([-.5,params['n_oma']-.5])
  ax.set_xlabel('bin')
  ax.set_ylabel('dist')

  tmpvision = np.minimum(50,otherflies_vision[:,t])
  ax.plot(tmpvision,'k-')
  
def debug_plot_wall_touch(t,xwall,ywall,distleg,wall_touch):
  plt.figure()
  plt.clf()
  ax = plt.subplot(1,2,1)
  ax.plot(xwall.flatten(),ywall.flatten(),'k.')
  theta_arena = np.linspace(-np.pi,np.pi,100)
  ax.plot(np.cos(theta_arena)*SENSORY_PARAMS['inner_arena_radius'],np.sin(theta_arena)*SENSORY_PARAMS['inner_arena_radius'],'-')
  ax.plot(np.cos(theta_arena)*SENSORY_PARAMS['outer_arena_radius'],np.sin(theta_arena)*SENSORY_PARAMS['outer_arena_radius'],'-')
  hpts = []
  for pti in range(nkptouch):
    hpts.append(ax.plot(xwall[pti,t],ywall[pti,t],'o')[0])
  ax.set_aspect('equal')
  ax = plt.subplot(1,2,2)
  ax.plot(distleg.flatten(),wall_touch.flatten(),'k.')
  ax.plot([0,SENSORY_PARAMS['inner_arena_radius'],SENSORY_PARAMS['outer_arena_radius']],
          [SENSORY_PARAMS['arena_height'],SENSORY_PARAMS['arena_height'],0],'-')
  for pti in range(nkptouch):
    ax.plot(distleg[pti,t],wall_touch[pti,t],'o',color=hpts[pti].get_color())
  ax.set_aspect('equal')

# compute sensory input
#
# inputs:
# xeye_main: x-coordinate of main fly's position for vision. shape = (T).
# yeye_main: y-coordinate of main fly's position for vision. shape = (T).
# theta_main: orientation of main fly. shape = (T).
# xtouch_main: x-coordinates of main fly's keypoints for computing touch inputs (both wall and other fly). shape = (npts_touch,T).
# ytouch_main: y-coordinates of main fly's keypoints for computing touch inputs (both wall and other fly). shape = (npts_touch,T).
# xvision_other: x-coordinate of keypoints on the other flies for computing vision input. shape = (npts_vision,T,nflies)
# yvision_other: y-coordinate of keypoints on the other flies for computing vision input. shape = (npts_vision,T,nflies)
# xtouch_other: x-coordinate of keypoints on the other flies for computing touch input. shape = (npts_touch_other,T,nflies)
# ytouch_other: y-coordinate of keypoints on the other flies for computing touch input. shape = (npts_touch_other,T,nflies)
#
# outputs:
# otherflies_vision: appearance of other flies to input fly. this is computed as a 
# 1. - np.minimum(1.,SENSORY_PARAMS['otherflies_vision_mult'] * dist**SENSORY_PARAMS['otherflies_vision_exp'])
# where dist is the minimum distance to some point on some other fly x,y_vision_other in each of n_oma directions. 
# shape = (SENSORY_PARAMS['n_oma'],T).  
# wall_touch: height of arena chamber at each keypoint in x,y_touch_main. this is computed as
# np.minimum(SENSORY_PARAMS['arena_height'],np.maximum(0.,SENSORY_PARAMS['arena_height'] - 
# (distleg-SENSORY_PARAMS['inner_arena_radius'])*SENSORY_PARAMS['arena_height']/(SENSORY_PARAMS['outer_arena_radius']-
# SENSORY_PARAMS['inner_arena_radius'])))
# shape = (npts_touch,T), 
# otherflies_touch: information about touch from other flies to input fly. this is computed as
# 1. - np.minimum(1.,SENSORY_PARAMS['otherflies_touch_mult'] * dist**SENSORY_PARAMS['otherflies_touch_exp'])
# where dist is the minimum distance over all other flies from each keypoint in x,y_touch_main to each keypoint in x,y_touch_other
# there are two main difference between this and otherflies_vision. first is this uses multiple keypoints on the main and other flies
# and has an output for each of them. conversely, otherflies_vision has an output for each direction. the second difference is
# based on the parameters in SENSORY_PARAMS. The parameters for touch should be set so that the maximum distance over which there is a 
# signal is about how far any keypoint can be from any of the keypoints in x,y_touch_other, which the maximum distance for 
# vision is over the entire arena. 
# shape = (npts_touch*npts_touch_other,T).

def compute_sensory(xeye_main,yeye_main,theta_main,
                    xtouch_main,ytouch_main,
                    xvision_other,yvision_other,
                    xtouch_other,ytouch_other):

  # increase dimensions if only one frame input
  if xvision_other.ndim < 3:
    T = 1
  else:
    T = xvision_other.shape[2]

  npts_touch = xtouch_main.shape[0]
  npts_vision = xvision_other.shape[0]
  npts_touch_other = xtouch_other.shape[0]
  nflies = xvision_other.shape[1]
  
  xvision_other = np.reshape(xvision_other,(npts_vision,nflies,T))
  yvision_other = np.reshape(yvision_other,(npts_vision,nflies,T))
  xtouch_other = np.reshape(xtouch_other,(npts_touch_other,nflies,T))
  ytouch_other = np.reshape(ytouch_other,(npts_touch_other,nflies,T))
  xeye_main = np.reshape(xeye_main,(1,1,T))
  yeye_main = np.reshape(yeye_main,(1,1,T))
  theta_main = np.reshape(theta_main,(1,1,T))
  xtouch_main = np.reshape(xtouch_main,(npts_touch,T))
  ytouch_main = np.reshape(ytouch_main,(npts_touch,T))
  
  # don't deal with missing data :)    
  assert(np.any(np.isnan(xeye_main))==False)
  assert(np.any(np.isnan(yeye_main))==False)
  assert(np.any(np.isnan(theta_main))==False)
  
  # vision bin size
  step = 2.*np.pi/SENSORY_PARAMS['n_oma']

  # compute other flies view

  # convert to this fly's coord system
  dx = xvision_other-xeye_main
  dy = yvision_other-yeye_main
  
  # distance
  dist = np.sqrt(dx**2+dy**2)
  
  # angle in the original coordinate system
  angle0 = np.arctan2(dy,dx)  
  
  # subtract off angle of main fly
  angle = angle0 - theta_main
  angle = mabe.modrange(angle,-np.pi,np.pi)

  # which other flies pass beyond the -pi to pi border
  isbackpos = angle > np.pi/2
  isbackneg = angle < -np.pi/2
  isfront = np.abs(angle) <= np.pi/2
  idxmod = np.any(isbackpos,axis=0) & np.any(isbackneg,axis=0) & (np.any(isfront,axis=0)==False)

  # bin - npts x nflies x T
  b_all = np.floor((angle+np.pi)/step)
  
  # bin range
  # shape: nflies x T
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    minb = np.nanmin(b_all,axis=0)
    maxb = np.nanmax(b_all,axis=0)
    mind = np.nanmin(dist,axis=0) 
    
  # n_oma x 1 x 1
  tmpbins = np.arange(SENSORY_PARAMS['n_oma'])[:,None,None]
  
  # n_oma x nflies x T
  mindrep = np.tile(mind[None,...],(SENSORY_PARAMS['n_oma'],1,1))
  mask = (tmpbins >= minb[None,...]) & (tmpbins <= maxb[None,...])
  
  if np.any(idxmod):
    # this is complicated!! 
    # find the max bin for negative angles
    # and the min bin for positive angles
    # store them in min and max for consistency with non-wraparound
    isbackpos1 = isbackpos[:,idxmod]
    isbackneg1 = isbackneg[:,idxmod]
    bmodneg = b_all[:,idxmod]
    bmodneg[isbackpos1] = np.nan
    minbmod = np.nanmax(bmodneg,axis=0)
    bmodpos = b_all[:,idxmod]
    bmodpos[isbackneg1] = np.nan
    maxbmod = np.nanmin(bmodpos,axis=0)
    mask[:,idxmod] = (tmpbins[...,0] >= maxbmod[None,:]) | (tmpbins[...,0] <= minbmod[None,:])
  
  otherflies_vision = np.nanmin(np.where(mask,mindrep,np.inf),axis=1,initial=np.inf)
  
  otherflies_vision = 1. - np.minimum(1.,SENSORY_PARAMS['otherflies_vision_mult'] * otherflies_vision**SENSORY_PARAMS['otherflies_vision_exp'])

  # t = 249
  # debug_plot_otherflies_vision(t,xother,yother,xeye_main,yeye_main,theta_main,
  #                                 angle0,angle,dist,b_all,otherflies_vision,params)

  # distance from center of arena
  # center of arena is assumed to be [0,0]
  distarena = np.sqrt( xtouch_main**2. + ytouch_main**2 )

  # height of chamber 
  wall_touch = np.zeros(distarena.shape)
  wall_touch[:] = SENSORY_PARAMS['arena_height']
  wall_touch = np.minimum(SENSORY_PARAMS['arena_height'],np.maximum(0.,SENSORY_PARAMS['arena_height'] - (distarena-SENSORY_PARAMS['inner_arena_radius'])*SENSORY_PARAMS['arena_height']/(SENSORY_PARAMS['outer_arena_radius']-SENSORY_PARAMS['inner_arena_radius'])))
  wall_touch[distarena >= SENSORY_PARAMS['outer_arena_radius']] = 0.
  
  # t = 0
  # debug_plot_wall_touch(t,xlegtip_main,ylegtip_main,distleg,wall_touch,params)

  # xtouch_main: npts_touch x T, xtouch_other: npts_touch_other x nflies x T
  if SENSORY_PARAMS['compute_otherflies_touch']:
    dx = xtouch_main.reshape((npts_touch,1,1,T)) - xtouch_other.reshape((1,npts_touch_other,nflies,T)) 
    dy = ytouch_main.reshape((npts_touch,1,1,T)) - ytouch_other.reshape((1,npts_touch_other,nflies,T)) 
    d = np.sqrt(np.nanmin(dx**2 + dy**2,axis=2)).reshape(npts_touch*npts_touch_other,T)
    otherflies_touch = 1. - np.minimum(1.,SENSORY_PARAMS['otherflies_touch_mult'] * d**SENSORY_PARAMS['otherflies_touch_exp'])
  else:
    otherflies_touch = None
  
  return (otherflies_vision, wall_touch, otherflies_touch)

def compute_sensory_torch(xeye_main,yeye_main,theta_main,
                          xtouch_main,ytouch_main,
                          xvision_other,yvision_other,
                          xtouch_other,ytouch_other):

  """
  compute sensory input
  compute_sensory_torch(xeye_main,yeye_main,theta_main,
                          xtouch_main,ytouch_main,
                          xvision_other,yvision_other,
                          xtouch_other,ytouch_other)

  inputs:
  xeye_main: x-coordinate of main fly's position for vision. shape = (T).
  yeye_main: y-coordinate of main fly's position for vision. shape = (T).
  theta_main: orientation of main fly. shape = (T).
  xtouch_main: x-coordinates of main fly's keypoints for computing touch inputs (both wall and other fly). shape = (npts_touch,T).
  ytouch_main: y-coordinates of main fly's keypoints for computing touch inputs (both wall and other fly). shape = (npts_touch,T).
  xvision_other: x-coordinate of keypoints on the other flies for computing vision input. shape = (npts_vision,T,nflies)
  yvision_other: y-coordinate of keypoints on the other flies for computing vision input. shape = (npts_vision,T,nflies)
  xtouch_other: x-coordinate of keypoints on the other flies for computing touch input. shape = (npts_touch_other,T,nflies)
  ytouch_other: y-coordinate of keypoints on the other flies for computing touch input. shape = (npts_touch_other,T,nflies)

  outputs:
  otherflies_vision: appearance of other flies to input fly. this is computed as a 
  1. - np.minimum(1.,SENSORY_PARAMS['otherflies_vision_mult'] * dist**SENSORY_PARAMS['otherflies_vision_exp'])
  where dist is the minimum distance to some point on some other fly x,y_vision_other in each of n_oma directions. 
  shape = (SENSORY_PARAMS['n_oma'],T).  
  wall_touch: height of arena chamber at each keypoint in x,y_touch_main. this is computed as
  np.minimum(SENSORY_PARAMS['arena_height'],np.maximum(0.,SENSORY_PARAMS['arena_height'] - 
  (distleg-SENSORY_PARAMS['inner_arena_radius'])*SENSORY_PARAMS['arena_height']/(SENSORY_PARAMS['outer_arena_radius']-
  SENSORY_PARAMS['inner_arena_radius'])))
  shape = (npts_touch,T), 
  otherflies_touch: information about touch from other flies to input fly. this is computed as
  1. - np.minimum(1.,SENSORY_PARAMS['otherflies_touch_mult'] * dist**SENSORY_PARAMS['otherflies_touch_exp'])
  where dist is the minimum distance over all other flies from each keypoint in x,y_touch_main to each keypoint in x,y_touch_other
  there are two main difference between this and otherflies_vision. first is this uses multiple keypoints on the main and other flies
  and has an output for each of them. conversely, otherflies_vision has an output for each direction. the second difference is
  based on the parameters in SENSORY_PARAMS. The parameters for touch should be set so that the maximum distance over which there is a 
  signal is about how far any keypoint can be from any of the keypoints in x,y_touch_other, which the maximum distance for 
  vision is over the entire arena. 
  shape = (npts_touch*npts_touch_other,T).
  """

  device = xeye_main.device
  dtype = xeye_main.dtype

  # increase dimensions if only one frame input
  if xvision_other.ndim < 3:
    T = 1
  else:
    T = xvision_other.shape[2]

  npts_touch = xtouch_main.shape[0]
  npts_vision = xvision_other.shape[0]
  npts_touch_other = xtouch_other.shape[0]
  nflies = xvision_other.shape[1]

  xvision_other = torch.reshape(xvision_other,(npts_vision,nflies,T))
  yvision_other = torch.reshape(yvision_other,(npts_vision,nflies,T))
  xtouch_other = torch.reshape(xtouch_other,(npts_touch_other,nflies,T))
  ytouch_other = torch.reshape(ytouch_other,(npts_touch_other,nflies,T))
  xeye_main = torch.reshape(xeye_main,(1,1,T))
  yeye_main = torch.reshape(yeye_main,(1,1,T))
  theta_main = torch.reshape(theta_main,(1,1,T))
  xtouch_main = torch.reshape(xtouch_main,(npts_touch,T))
  ytouch_main = torch.reshape(ytouch_main,(npts_touch,T))

  # don't deal with missing data :)
  assert(torch.any(torch.isnan(xeye_main))==False)
  assert(torch.any(torch.isnan(yeye_main))==False)
  assert(torch.any(torch.isnan(theta_main))==False)
  
  # vision bin size
  step = 2.*torch.pi/SENSORY_PARAMS['n_oma']

  # compute other flies view

  # convert to this fly's coord system
  dx = xvision_other-xeye_main
  dy = yvision_other-yeye_main
  
  # distance
  dist = torch.sqrt(dx**2+dy**2)
  
  # angle in the original coordinate system
  angle0 = torch.arctan2(dy,dx)  
  
  # subtract off angle of main fly
  angle = angle0 - theta_main
  angle = torch.fmod(angle + torch.pi, 2 * torch.pi) - torch.pi

  # which other flies pass beyond the -pi to pi border
  isbackpos = angle > torch.pi/2
  isbackneg = angle < -torch.pi/2
  isfront = torch.abs(angle) <= torch.pi/2
  idxmod = torch.any(isbackpos,dim=0) & torch.any(isbackneg,dim=0) & (~torch.any(isfront,dim=0))

  # bin - npts x nflies x T
  b_all = torch.floor((angle+np.pi)/step)
  
  # bin range
  # shape: nflies x T
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    minb = torch.nanmin(b_all, dim=0)
    maxb = torch.nanmax(b_all, dim=0)
    mind = torch.nanmin(dist, dim=0)
    
  # n_oma x 1 x 1
  tmpbins = torch.arange(SENSORY_PARAMS['n_oma'],dtype=dtype,device=device)[:,None,None]

  # n_oma x nflies x T
  mindrep = mind[None,...].repeat((SENSORY_PARAMS['n_oma'],1,1))
  mask = (tmpbins >= minb[None,...]) & (tmpbins <= maxb[None,...])

  if torch.any(idxmod):
    # this is complicated!!
    # find the max bin for negative angles
    # and the min bin for positive angles
    # store them in min and max for consistency with non-wraparound
    isbackpos1 = isbackpos[:,idxmod]
    isbackneg1 = isbackneg[:,idxmod]
    bmodneg = b_all[:,idxmod]
    bmodneg[isbackpos1] = torch.nan
    minbmod = torch.nanmax(bmodneg,dim=0)
    bmodpos = b_all[:,idxmod]
    bmodpos[isbackneg1] = torch.nan
    maxbmod = torch.nanmin(bmodpos,dim=0)
    mask[:,idxmod] = (tmpbins[...,0] >= maxbmod.unsqueeze(0)) | (tmpbins[...,0] <= minbmod.unsqueeze(0))

  otherflies_vision = torch.nanmin(torch.where(mask, mindrep, torch.full_like(mindrep, float('inf'))), dim=1, keepdim=False, initial=float('inf'))
  otherflies_vision = 1. - torch.min(1., SENSORY_PARAMS['otherflies_vision_mult'] * otherflies_vision**SENSORY_PARAMS['otherflies_vision_exp'])

  # t = 249
  # debug_plot_otherflies_vision(t,xother,yother,xeye_main,yeye_main,theta_main,
  #                                 angle0,angle,dist,b_all,otherflies_vision,params)

  # distance from center of arena
  # center of arena is assumed to be [0,0]
  distleg = torch.sqrt( xtouch_main**2. + ytouch_main**2 )

  # height of chamber 
  wall_touch = torch.zeros(distleg.shape,dtype=dtype,device=device)
  wall_touch[:] = SENSORY_PARAMS['arena_height']
  wall_touch = torch.minimum(SENSORY_PARAMS['arena_height'],torch.maximum(0.,SENSORY_PARAMS['arena_height'] - (distleg-SENSORY_PARAMS['inner_arena_radius'])*SENSORY_PARAMS['arena_height']/(SENSORY_PARAMS['outer_arena_radius']-SENSORY_PARAMS['inner_arena_radius'])))
  wall_touch[distleg >= SENSORY_PARAMS['outer_arena_radius']] = 0.
  
  # t = 0
  # debug_plot_wall_touch(t,xlegtip_main,ylegtip_main,distleg,wall_touch,params)

  # xtouch_main: npts_touch x T, xtouch_other: npts_touch_other x nflies x T
  if SENSORY_PARAMS['compute_otherflies_touch']:
    dx = xtouch_main.reshape((npts_touch,1,1,T)) - xtouch_other.reshape((1,npts_touch_other,nflies,T)) 
    dy = ytouch_main.reshape((npts_touch,1,1,T)) - ytouch_other.reshape((1,npts_touch_other,nflies,T)) 
    d = torch.sqrt(torch.nanmin(dx**2 + dy**2,dim=2)).reshape(npts_touch*npts_touch_other,T)
    otherflies_touch = 1. - torch.minimum(1.,SENSORY_PARAMS['otherflies_touch_mult'] * d**SENSORY_PARAMS['otherflies_touch_exp'])
  else:
    otherflies_touch = None
  
  return (otherflies_vision, wall_touch, otherflies_touch)

featorigin = [mabe.posenames.index('thorax_front_x'),mabe.posenames.index('thorax_front_y')]
feattheta = mabe.posenames.index('orientation')
featglobal = featorigin + [feattheta,]
featthetaglobal = 2
featrelative = np.ones(len(mabe.posenames),dtype=bool)
featrelative[featglobal] = False
nrelative = np.count_nonzero(featrelative)
nglobal = len(featglobal)
nfeatures = len(mabe.posenames)
featangle = np.array([re.search('angle$',s) is not None for s in mabe.posenames])
featangle[feattheta] = True

kpvision_other = [mabe.keypointnames.index(x) for x in SENSORY_PARAMS['vision_kpnames']]
kpeye = mabe.keypointnames.index('antennae_midpoint')
kptouch = [mabe.keypointnames.index(x) for x in SENSORY_PARAMS['touch_kpnames']]
nkptouch = len(kptouch)
kptouch_other = [mabe.keypointnames.index(x) for x in SENSORY_PARAMS['touch_other_kpnames']]
nkptouch_other = len(kptouch_other)

narena = 2**10
theta_arena = np.linspace(-np.pi,np.pi,narena+1)[:-1]

def get_sensory_feature_shapes(simplify=None):
  idx = collections.OrderedDict()
  sz = collections.OrderedDict()
  i0 = 0
  i1 = nrelative
  idx['pose'] = [i0,i1]
  sz['pose'] = (nrelative,)
  
  if simplify is None:
    i0 = i1
    i1 = i0+nkptouch
    idx['wall_touch'] = [i0,i1]
    sz['wall_touch'] = (nkptouch,)
    i0 = i1
    i1 = i0 + SENSORY_PARAMS['n_oma']
    idx['otherflies_vision'] = [i0,i1]
    sz['otherflies_vision'] = (SENSORY_PARAMS['n_oma'],)
    i0 = i1
    i1 = i0 + nkptouch*nkptouch_other
    idx['otherflies_touch'] = [i0,i1]
    sz['otherflies_touch'] = (nkptouch*nkptouch_other,)
  return idx,sz

def get_sensory_feature_idx(simplify=None):
  idx,_ = get_sensory_feature_shapes()
  return idx

def split_features(X,simplify=None,axis=-1):
  res = {}
  idx = get_sensory_feature_idx(simplify)
  for k,v in idx.items():
    res[k] = X.take(range(v[0],v[1]),axis=axis)

  return res

def unpack_input(input,featidx,sz,dim=-1):
  
  res = {}
  idx = [slice(None),]*input.ndim
  sz0 = input.shape
  if dim < 0:
    dim = input.ndim+dim
  for k,v in featidx.items():
    idx[dim] = slice(v[0],v[1])
    newsz = sz0[:dim] + sz[k] + sz0[dim+1:]
    res[k] = input[idx].reshape(newsz)
  
  return res

def combine_relative_global(Xrelative,Xglobal,axis=-1):
  X = np.concatenate((Xglobal,Xrelative),axis=axis)
  return X

def combine_relative_global_pose(relpose,globalpos):
  sz = (nfeatures,)+relpose.shape[1:]
  posefeat = np.zeros(sz,dtype=relpose.dtype)
  posefeat[featrelative,...] = relpose
  posefeat[featglobal,...] = globalpos
  return posefeat

def compute_pose_features(X,scale):
  posefeat = mabe.kp2feat(X,scale)
  relpose = posefeat[featrelative,...]
  globalpos = posefeat[featglobal,...]

  return relpose,globalpos

def compute_movement(X=None,scale=None,relpose=None,globalpos=None,simplify=None,dct_m=None,tspred_global=[1,]):
  """
  movement = compute_movement(X=X,scale=scale,...)
  movement = compute_movement(relpose=relpose,globalpos=globalpos,...)

  Args:
      X (ndarray, nkpts x 2 x T x nflies, optional): Keypoints. Can be None only if relpose and globalpos are input. Defaults to None. T>=2
      scale (ndarray, nscale x nflies): Scaling parameters related to an individual fly. Can be None only if relpose and globalpos are input. Defaults to None.
      relpose (ndarray, nrelative x T x nflies or nrelative x T, optional): Relative pose features. T>=2
      If input, X and scale are ignored. Defaults to None.
      globalpos (ndarray, nglobal x T x nflies or nglobal x T, optional): Global position. If input, X and scale are ignored. Defaults to None. T>=2
      simplify (string or None, optional): Whether/how to simplify the output. Defaults to None for no simplification.
  Optional args:
      dct_m (ndarray, nrelative x ntspred_dct+1 x nflies): DCT matrix for pose features. Defaults to None.
      tspred_global (list of int, optional): Time steps to predict for global features. Defaults to [1,].

  Returns:
      movement (ndarray, d_output x T-1 x nflies): Per-frame movement. movement[:,t,i] is the movement from frame 
      t for fly i. 
  """

  if relpose is None or globalpos is None:
    relpose,globalpos = compute_pose_features(X,scale)
    
  nd = relpose.ndim
  assert(nd==2 or nd==3)
  if nd < 3:
    relpose = relpose[...,None]
    globalpos = globalpos[...,None]
  T = relpose.shape[1]
  nflies = relpose.shape[2]

  Xorigin = globalpos[:2,...]
  Xtheta = globalpos[2,...]
  ntspred_global = len(tspred_global)
  nmovement_features = ntspred_global*nglobal
  if (dct_m is not None) and simplify != 'global':
    ntspred_dct = dct_m.shape[0]
    tspred_dct = np.arange(1,ntspred_dct+1)
    tspred_relative = tspred_dct
  else:
    ntspred_dct = 0
    tspred_dct = []
    tspred_relative = [1,]
  nmovement_features += (ntspred_dct+1)*nrelative
  # compute the max of tspred_global and tspred_dct
  tspred_all = np.unique(np.concatenate((tspred_global,tspred_relative)))
  lastT = T - np.max(tspred_all)
  
  # dXoriginrel[tau,:,t,fly] is the global position for fly fly at time t+tau in the coordinate system of the fly at time t
  dXoriginrel = np.zeros((ntspred_global,2,T,nflies),dtype=Xorigin.dtype)
  dXoriginrel[:] = np.nan
  # dtheta[tau,t,fly] is the change in orientation for fly fly at time t+tau from time t
  dtheta = np.zeros((ntspred_global,T,nflies),dtype=Xtheta.dtype)
  dtheta[:] = np.nan
  # dtheta1[t,fly] is the change in orientation for fly from frame t to t+1
  dtheta1 = mabe.modrange(Xtheta[1:,:]-Xtheta[:-1,:],-np.pi,np.pi)

  if simplify != 'global':
    # drelpose1[f,fly] is the change in pose for fly from frame t to t+1
    drelpose1 = relpose[:,1:,:]-relpose[:,:-1,:]
    drelpose1[featangle[featrelative],:,:] = mabe.modrange(drelpose1[featangle[featrelative],:,:],-np.pi,np.pi)
    # drelpose[tau,t,fly] is the change in pose for fly fly at time t+tau from time t
    if dct_m is not None:
      drelpose = np.zeros((relpose.shape[0],T,ntspred_dct+1,nflies),dtype=relpose.dtype)
      drelpose[:] = np.nan
      drelpose[:,:-1,0,:] = drelpose1
    else:
      drelpose = drelpose1
  
  def boxsum(x,n):
    if n == 1:
      return x
    xtorch = torch.tensor(x[:,None,...])
    y = torch.nn.functional.conv2d(xtorch,torch.ones((1,1,n,1),dtype=xtorch.dtype),padding='valid')
    return y[:,0,...].numpy()
  
  for i,toff in enumerate(tspred_global):
    # center and rotate absolute position around position toff frames previous
    dXoriginrel[i,:,:-toff,:] = mabe.rotate_2d_points((Xorigin[:,toff:,:]-Xorigin[:,:-toff,:]).transpose((1,0,2)),Xtheta[:-toff,:]).transpose((1,0,2))
    # compute total change in global orientation in toff frame intervals
    dtheta[i,:-toff,:] = boxsum(dtheta1[None,...],toff)
  for i,toff in enumerate(tspred_dct):
    # compute total change in relative pose in toff frame intervals
    if (dct_m is not None) and (simplify != 'global'):
      drelpose[:,:-toff,i+1,:] = boxsum(drelpose1,toff)

  # only full data up to frame lastT
  dXoriginrel = dXoriginrel[:,:,:lastT,:]
  dtheta = dtheta[:,:lastT,:]
  drelpose = drelpose[:,:lastT]
  if (simplify != 'global'):
    # the pose forecasting papers compute error on the actual pose, not the dct. they just force the network to go through the dct
    # representation first.
    if (dct_m is not None):
      drelpose[:,:,1:,:] = dct_m @ drelpose[:,:,1:,:]
    drelpose = np.moveaxis(drelpose,2,0)

  movement_global = np.concatenate((dXoriginrel[:,[1,0]],dtheta[:,None,:,:]),axis=1)
  movement_global = movement_global.reshape((ntspred_global*nglobal,lastT,nflies))  
  if simplify == 'global':
    movement = movement_global
  else:
    drelpose = drelpose.reshape(((ntspred_dct+1)*nrelative,lastT,nflies))
    movement = np.concatenate((movement_global,drelpose),axis=0)

  # old code
  # dXoriginrel = mabe.rotate_2d_points((Xorigin[:,1:,:]-Xorigin[:,:-1,:]).transpose((1,0,2)),Xtheta[:-1,:]).transpose((1,0,2))
  # forward_vel = dXoriginrel[1,...]
  # sideways_vel = dXoriginrel[0,...]

  # movement = np.zeros([nfeatures,T-1,nflies],relpose.dtype)
  # movement[featrelative,...] = relpose[:,1:,:]-relpose[:,:-1,:]
  # movement[featorigin[0],...] = forward_vel
  # movement[featorigin[1],...] = sideways_vel
  # movement[feattheta,...] = Xtheta[1:,...]-Xtheta[:-1,...]
  # movement[featangle,...] = mabe.modrange(movement[featangle,...],-np.pi,np.pi)

  # if simplify is not None:
  #   if simplify == 'global':
  #     movement = movement[featglobal,...]
  #   else:
  #     raise

  if nd == 2:
    movement = movement[...,0]

  return movement


def compute_sensory_wrapper(Xkp,flynum,theta_main=None,returnall=False):
  
  # other flies positions
  idxother = np.ones(Xkp.shape[-1],dtype=bool)
  idxother[flynum] = False
  Xkp_other = Xkp[:,:,:,idxother]
  
  xeye_main = Xkp[kpeye,0,:,flynum]
  yeye_main = Xkp[kpeye,1,:,flynum]
  xtouch_main = Xkp[kptouch,0,:,flynum]
  ytouch_main = Xkp[kptouch,1,:,flynum]
  xvision_other = Xkp_other[kpvision_other,0,...].transpose((0,2,1))
  yvision_other = Xkp_other[kpvision_other,1,...].transpose((0,2,1))
  xtouch_other = Xkp_other[kptouch_other,0,...].transpose((0,2,1))
  ytouch_other = Xkp_other[kptouch_other,1,...].transpose((0,2,1))
  
  if theta_main is None:
    _,_,theta_main = mabe.body_centric_kp(Xkp[...,[flynum,]])
    theta_main = theta_main[...,0]
  
  otherflies_vision,wall_touch,otherflies_touch = \
    compute_sensory(xeye_main,yeye_main,theta_main+np.pi/2,
                    xtouch_main,ytouch_main,
                    xvision_other,yvision_other,
                    xtouch_other,ytouch_other)
  sensory = np.r_[wall_touch,otherflies_vision]
  if otherflies_touch is not None:
    sensory = np.r_[sensory,otherflies_touch]
  if returnall:
    return sensory,wall_touch,otherflies_vision,otherflies_touch
  else:
    return sensory

def combine_inputs(relpose=None,sensory=None,input=None,labels=None,dim=0):
  if input is None:
    input = np.concatenate((relpose,sensory),axis=dim)
  if labels is not None:
    input = np.concatenate((input,labels),axis=dim)
  return input 

def get_dct_matrix(N: int) -> (np.ndarray,np.ndarray):
  """ Get the Discrete Cosine Transform coefficient matrix
  Copied from https://github.com/dulucas/siMLPe/blob/main/exps/baseline_h36m/train.py
  Back to MLP: A Simple Baseline for Human Motion Prediction
  Guo, Wen and Du, Yuming and Shen, Xi and Lepetit, Vincent and Xavier, Alameda-Pineda and Francesc, Moreno-Noguer
  arXiv preprint arXiv:2207.01567
  2022
  Args:
      N (int): number of time points

  Returns:
      dct_m: array of shape N x N with the encoding coefficients
      idct_m: array of shape N x N with the inverse coefficients
  """
  dct_m = np.eye(N)
  for k in np.arange(N):
    for i in np.arange(N):
      w = np.sqrt(2 / N)
      if k == 0:
        w = np.sqrt(1 / N)
      dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
  idct_m = np.linalg.inv(dct_m)
  return dct_m, idct_m

def unravel_label_index(idx,dct_m=None,tspred_global=[1,]):
  idx = np.array(idx)
  sz = idx.shape
  idx = idx.flatten()
  if dct_m is None:
    dct_tau = 1
  else:
    dct_tau = dct_m.shape[0]+1
  offrelative = len(tspred_global)*nglobal
  ftidx = np.zeros((len(idx),2),dtype=int)
  for ii,i in enumerate(idx):
    if i < offrelative:
      tidx,fidx = np.unravel_index(i,(len(tspred_global),nglobal))
      ftidx[ii] = (fidx,tspred_global[tidx])
    else:
      t,fidx = np.unravel_index(i-offrelative,(dct_tau,nrelative))
      # t = 1 corresponds to next frame
      ftidx[ii] = (fidx+nglobal,t+1)
  return ftidx.reshape(sz+(2,))

def ravel_label_index(ftidx,dct_m=None,tspred_global=[1,]):

  ftidx = np.array(ftidx)
  sz = ftidx.shape
  assert sz[-1] == 2
  ftidx = ftidx.reshape((-1,2))
  
  idx = np.zeros(ftidx.shape[:-1],dtype=int)
  
  if dct_m is None:
    dct_tau = 1
  else:
    dct_tau = dct_m.shape[0]+1
  offrelative = len(tspred_global)*nglobal
  
  for i,ft in enumerate(ftidx):
    fidx = ft[0]
    t = ft[1]
    isglobal = fidx < nglobal
    if isglobal:
      # t = 1 corresponds to next frame
      tidx = np.nonzero(tspred_global==t)[0][0]
      assert tidx is not None
      idx[i] = np.ravel_multi_index((tidx,fidx),(len(tspred_global),nglobal))
    else:
      # t = 1 corresponds to next frame
      idx[i] = np.ravel_multi_index((t-1,fidx-nglobal),(dct_tau,nrelative))+offrelative
  
  return idx.reshape(sz[:-1])

def compute_features(X,id,flynum,scale_perfly,smush=True,outtype=None,
                     simplify_out=None,simplify_in=None,dct_m=None,tspred_global=[1,],
                     npad=1):
  
  res = {}
  
  # convert to relative locations of body parts
  if id is None:
    scale = scale_perfly
  else:
    scale = scale_perfly[:,id]
  
  relpose,globalpos = compute_pose_features(X[...,flynum],scale)
  relpose = relpose[...,0]
  globalpos = globalpos[...,0]
  if npad == 0:
    endidx = None
  else:
    endidx = -npad
  sensory,wall_touch,otherflies_vision,otherflies_touch = \
    compute_sensory_wrapper(X[:,:,:endidx,:],flynum,theta_main=globalpos[featthetaglobal,:endidx],
                            returnall=True)
  res['input'] = combine_inputs(relpose=relpose[:,:endidx],sensory=sensory).T
  #res['input'] = input[:-1,:]

  movement = compute_movement(relpose=relpose,globalpos=globalpos,simplify=simplify_out,dct_m=dct_m,tspred_global=tspred_global)
  if simplify_out is not None:
    if simplify_out == 'global':
      movement = movement[featglobal,...]
    else:
      raise
    
  res['labels'] = movement.T
  res['init'] = globalpos[:,:2]
  res['scale'] = scale
  #res['nextinput'] = input[-1,:]
  
  if not smush:
    res['global'] = globalpos[:,:-1]
    res['relative'] = relpose[:,:-1]
    res['nextglobal'] = globalpos[:,-1]
    res['nextrelative'] = relpose[:,-1]
    if simplify_in == 'no_sensory':
      pass
    else:
      res['wall_touch'] = wall_touch[:,:-1]
      res['otherflies_vision'] = otherflies_vision[:,:-1]
      res['next_wall_touch'] = wall_touch[:,-1]
      res['next_otherflies_vision'] = otherflies_vision[:,-1]
      if otherflies_touch is not None:
        res['otherflies_touch'] = otherflies_touch[:,:-1]
        res['next_otherflies_touch'] = otherflies_touch[:,-1]
    
  # debug_plot_compute_features(X,porigin,theta,Xother,Xnother)
    
  if outtype is not None:
    res = {key: val.astype(outtype) for key,val in res.items()}
  return res


# def compute_features(X,id,flynum,scale_perfly,sensory_params,smush=True,outtype=None,
#                      simplify_out=None,simplify_in=None):
  
#   res = {}
  
#   # convert to relative locations of body parts
#   if id is None:
#     scale = scale_perfly
#   else:
#     scale = scale_perfly[:,id]
#   Xfeat = mabe.kp2feat(X[...,flynum],scale)
#   Xfeat = Xfeat[...,0]
  
#   # compute global coordinates relative to previous frame
#   Xorigin = Xfeat[featorigin,...]
#   Xtheta = Xfeat[feattheta,...]
#   dXoriginrel = mabe.rotate_2d_points((Xorigin[:,1:]-Xorigin[:,:-1]).T,Xtheta[:-1]).T
#   forward_vel = dXoriginrel[1,:]
#   sideways_vel = dXoriginrel[0,:]

#   # compute sensory information

#   if simplify_in == 'no_sensory':
#     input = Xfeat[featrelative,:].T
#     res['input'] = input[:-1,:]
#   else:
#     # other flies positions
#     idxother = np.ones(X.shape[-1],dtype=bool)
#     idxother[flynum] = False
#     Xother = X[:,:,:,idxother]
    
#     xeye_main = X[kpeye,0,:,flynum]
#     yeye_main = X[kpeye,1,:,flynum]
#     xlegtip_main = X[kplegtip,0,:,flynum]
#     ylegtip_main = X[kplegtip,1,:,flynum]
#     xother = Xother[kpother,0,...].transpose((0,2,1))
#     yother = Xother[kpother,1,...].transpose((0,2,1))
#     theta_main = Xfeat[feattheta,...]+np.pi/2
    
#     otherflies_vision,wall_touch = \
#       compute_sensory(xeye_main,yeye_main,theta_main,
#                       xlegtip_main,ylegtip_main,
#                       xother,yother,sensory_params)
#     input = np.r_[Xfeat[featrelative,:],wall_touch[:,:],otherflies_vision[:,:]].T
#     res['input'] = input[:-1,:]
    
#   movement = Xfeat[:,1:]-Xfeat[:,:-1]
#   movement[featorigin[0],:] = forward_vel
#   movement[featorigin[1],:] = sideways_vel
#   movement[featangle,...] = mabe.modrange(movement[featangle,...],-np.pi,np.pi)
#   if simplify_out is not None:
#     if simplify_out == 'global':
#       movement = movement[featglobal,...]
#     else:
#       raise

#   res['labels'] = movement.T
#   res['init'] = Xfeat[featrelative==False,:2]
#   res['scale'] = scale_perfly[:,id]
#   res['nextinput'] = input[-1,:]
  
#   # # check that we can undo correctly
#   # res['labels'].T
#   # thetavel = res['labels'][:,2]
#   # theta0 = res['init'][2,0]
#   # thetare = np.cumsum(np.r_[theta0[None],thetavel],axis=0)
#   # np.max(np.abs(thetare-Xfeat[2,:]))
#   # doriginrel = res['labels'][:,[1,0]]
#   # dxy = mabe.rotate_2d_points(doriginrel,-thetare[:-1])
#   # Xorigin[:,1:]-Xorigin[:,:-1]
#   # np.max(np.sum(np.abs((Xorigin[:,1:]-Xorigin[:,:-1]).T-dxy),axis=0))
  
#   if not smush:
#     res['global'] = Xfeat[featrelative==False,:-1]
#     res['relative'] = Xfeat[featrelative,:-1]
#     res['nextglobal'] = Xfeat[featrelative==False,-1]
#     res['nextrelative'] = Xfeat[featrelative,-1]
#     if simplify_in == 'no_sensory':
#       pass
#     else:
#       res['wall_touch'] = wall_touch[:,:-1]
#       res['otherflies_vision'] = otherflies_vision[:,:-1]
#       res['next_wall_touch'] = wall_touch[:,-1]
#       res['next_otherflies_vision'] = otherflies_vision[:,-1]
    
#   # debug_plot_compute_features(X,porigin,theta,Xother,Xnother)
    
#   if outtype is not None:
#     res = {key: val.astype(outtype) for key,val in res.items()}
#   return res

def compute_otherflies_touch_mult(data,prct=99):
  
  # 1/maxd^exp = mult*maxd^exp
  
  # X is nkeypts x 2 x T x nflies
  nkpts = data['X'].shape[0]
  T = data['X'].shape[2]
  nflies = data['X'].shape[3]
  # isdata is T x nflies
  # X will be nkpts x 2 x N
  X = data['X'].reshape([nkpts,2,T*nflies])[:,:,data['isdata'].flatten()]
  # maximum distance from some keypoint to any keypoint included in kpother
  d = np.sqrt(np.nanmax(np.nanmin(np.sum((X[None,kptouch,:,:] - X[kptouch_other,None,:,:])**2.,axis=2),axis=0),axis=0))
  maxd = np.percentile(d,prct)

  otherflies_touch_mult = 1./((maxd)**SENSORY_PARAMS['otherflies_touch_exp'])
  return otherflies_touch_mult
  
    
def debug_plot_compute_features(X,porigin,theta,Xother,Xnother):
  
  t = 0
  rplot = 5.
  plt.clf()
  ax = plt.subplot(1,2,1)
  mabe.plot_flies(X[:,:,t,:],ax=ax,textlabels='fly',colors=np.zeros((X.shape[-1],3)))
  ax.plot(porigin[0,t],porigin[1,t],'rx',linewidth=2)
  ax.plot([porigin[0,t,0],porigin[0,t,0]+np.cos(theta[t,0])*rplot],
          [porigin[1,t,0],porigin[1,t,0]+np.sin(theta[t,0])*rplot],'r-')
  ax.plot(Xother[kpvision_other,0,t,:],Xother[kpvision_other,1,t,:],'o')
  ax.set_aspect('equal')
  
  ax = plt.subplot(1,2,2)
  ax.plot(0,0,'rx')
  ax.plot([0,np.cos(0.)*rplot],[0,np.sin(0.)*rplot],'r-')
  ax.plot(Xnother[:,0,t,:],Xnother[:,1,t,:],'o')
  ax.set_aspect('equal')

def init_train_bpe(zlabels,transform=True,max_val=10,
                   n_clusters=int(1e3)):  

  def apply_transform(z):
    x = np.zeros(z.shape)
    x = np.sqrt(np.minimum(max_val,np.abs(z)))*np.sign(z)
    return x
  
  def apply_inverse_transform(x):
    z = np.zeros(x.shape)
    z = x**2*np.sign(z)
    return z

  # for |x| <= transform_thresh, use sqrt. above, use log
  if transform:
    x = apply_transform(zlabels)
  else:
    x = zlabels.copy()

  # k-means clustering of inter-frame motion
  alg = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters)
  token = alg.fit_predict(x)
  centers = alg.cluster_centers_  
  err = np.abs(x - centers[token,:])
  
  cmap = plt.get_cmap("rainbow")
  colors = cmap(np.arange(n_clusters)/n_clusters)
  colors = colors[np.random.permutation(n_clusters),:]*.7
  
  nplot = 1000
  nstd = 1
  fig,ax = plt.subplots(2,1,sharex='all')
  ax[0].cla()
  ax[1].cla()
  for i in range(x.shape[1]):
    xrecon = centers[token[:nplot],i]
    ax[0].plot([0,nplot],[i*nstd,]*2,':',color=[.5,.5,.5])
    ax[0].plot(np.arange(nplot),i*nstd*2+x[:nplot,i],'k.-')
    tmpy = np.c_[xrecon,x[:nplot,i],np.zeros(nplot)]
    tmpy[:,2] = np.nan
    tmpx = np.tile(np.arange(nplot)[:,None],(1,3))
    ax[0].plot(tmpx.flatten(),i*nstd*2+tmpy.flatten(),'k-')
    ax[0].scatter(np.arange(nplot),i*nstd*2+xrecon,c=colors[token[:nplot],:],marker='o')
    ax[0].text(0,2*nstd*(i+.5),outnames[i],horizontalalignment='left',verticalalignment='top')

  ax[1].plot(np.arange(nplot),token[:nplot],'k-')
  ax[1].scatter(np.arange(nplot),token[:nplot],c=colors[token[:nplot],:],marker='o')
  ax[1].set_ylabel('Token ID')
  return

def train_bpe(data,scale_perfly,simplify_out=None):
  
  # collect motion data
  nflies = data['X'].shape[3]
  T = data['X'].shape[2]

  isdata = np.any(np.isnan(data['X']),axis=(0,1)) == False
  isstart = (data['ids'][1:,:]!=data['ids'][:-1,:]) | \
    (data['frames'][1:,:] != (data['frames'][:-1,:]+1))
  isstart = np.r_[np.ones((1,nflies),dtype=bool),isstart]
  labels = None

  print('Computing movements over all data')
  for i in tqdm.trange(nflies,desc='animal'):
    isstart = isdata[1:,i] & \
      ((isdata[:-1,i] == False) | \
       (data['ids'][1:,i]!=data['ids'][:-1,i]) | \
       (data['frames'][1:,0] != (data['frames'][:-1,0]+1)))
    isstart = np.r_[isdata[0,i],isstart]
    isend = isdata[:-1,i] & \
      ((isdata[1:,i]==False) | \
       (data['ids'][1:,i]!=data['ids'][:-1,i]) | \
       (data['frames'][1:,0] != (data['frames'][:-1,0]+1)))
    isend = np.r_[isend,isdata[-1,i]]
    t0s = np.nonzero(isstart)[0]
    t1s = np.nonzero(isend)[0]+1
    for j in tqdm.trange(len(t0s),desc='frames'):
      t0 = t0s[j]
      t1 = t1s[j]
      id = data['ids'][t0,i]
      xcurr = compute_features(data['X'][:,:,t0:t1,:],id,i,scale_perfly,None,
                               simplify_in='no_sensory')
                               #simplify_out=simplify_out)
      labelscurr = xcurr['labels']
      if labels is None:
        labels = labelscurr
      else:
        labels = np.r_[labels,labelscurr]
        
        
    # zscore
    mu = np.mean(labels,axis=0)
    sig = np.std(labels,axis=0)
    zlabels = (labels-mu)/sig
    
    
    
  return

def to_size(sz):
  if sz is None:
    sz = (1,)
  elif isinstance(sz,int):
    sz = (sz,)
  elif isinstance(sz,list):
    sz = tuple(sz)
  elif isinstance(sz,tuple):
    pass
  else:
    raise ValueError('Input sz must be an int, list, or tuple')
  return sz

def weighted_sample(w,nsamples=0):
  SMALLNUM = 1e-6
  assert(torch.all(w>=0.))
  nbins = w.shape[-1]
  szrest = w.shape[:-1]
  n = int(np.prod(szrest))
  p = torch.cumsum(w.reshape((n,nbins)),dim=-1)
  assert(torch.all(torch.abs(p[:,-1]-1)<=SMALLNUM))
  p[p>1.] = 1.
  p[:,-1] = 1.
  if nsamples == 0:
    nsamples1 = 1
  else:
    nsamples1 = nsamples
  r = torch.rand((nsamples1,n),device=w.device)
  s = torch.zeros((nsamples1,)+p.shape,dtype=w.dtype,device=w.device)
  s[:] = r[:,None] <= p
  idx = torch.argmax(s,dim=-1)
  if nsamples > 0:
    szrest = (nsamples,)+szrest
  return idx.reshape(szrest)

# def samplebin(X,sz=None):
#   sz = to_size(sz)
#   r = torch.randint(low=0,high=X.numel(),size=sz)
#   return X[r]

def select_bin_edges(movement,nbins,bin_epsilon,outlierprct=0,feati=None):

  n = movement.shape[0]
  lims = np.percentile(movement,[outlierprct,100-outlierprct])
  max_bin_epsilon = (lims[1]-lims[0])/(nbins+1)
  if bin_epsilon >= max_bin_epsilon:
    print(f'{feati}: bin_epsilon {bin_epsilon} bigger than max bin epsilon {max_bin_epsilon}, setting all bins to be the same size')
    bin_edges = np.linspace(lims[0],lims[1],nbins+1)
    return bin_edges

  bin_edges = np.arange(lims[0],lims[1],bin_epsilon)
  bin_edges[-1] = lims[1]
  
  counts,_ = np.histogram(movement,bin_edges)
  mergecounts = counts[1:]+counts[:-1]
  for iter in range(len(bin_edges)-nbins-1):
    mincount = np.min(mergecounts)
    bini = np.random.choice(np.nonzero(mergecounts==mincount)[0],1)[0]
    if bini > 0:
      mergecounts[bini-1] += counts[bini]
    if bini < len(mergecounts)-1:
      mergecounts[bini+1] += counts[bini]
    mergecounts = np.delete(mergecounts,bini)
    counts[bini] = mincount
    counts = np.delete(counts,bini+1)
    bin_edges = np.delete(bin_edges,bini+1)

  return bin_edges
  

def fit_discretize_labels(dataset,featidx,nbins=50,bin_epsilon=None,outlierprct=.001,fracsample=None,nsamples=None):

  # compute percentiles
  nfeat = len(featidx)
  prctiles_compute = np.linspace(0,100,nbins+1)
  prctiles_compute[0] = outlierprct
  prctiles_compute[-1] = 100-outlierprct
  movement = np.concatenate([example['labels'][:,featidx].numpy() for example in dataset],axis=0)
  # bin_edges is nfeat x nbins+1
  if bin_epsilon is not None:
    bin_edges = np.zeros((nfeat,nbins+1),dtype=dataset.dtype)
    for feati in range(nfeat):
      bin_edges[feati,:] = select_bin_edges(movement[:,feati],nbins,bin_epsilon[feati],outlierprct=outlierprct,feati=feati)
  else:
    bin_edges = np.percentile(movement,prctiles_compute,axis=0)
    bin_edges = bin_edges.astype(dataset.dtype).T

  binnum = np.zeros(movement.shape,dtype=int)
  for i in range(nfeat):
    binnum[:,i] = np.digitize(movement[:,i],bin_edges[i,:])
  binnum = np.minimum(np.maximum(0,binnum-1),nbins-1)

  if nsamples is None:
    if fracsample is None:
      fracsample = 1/nbins/5
    nsamples = int(np.round(fracsample*movement.shape[0]))

  # for each bin, approximate the distribution
  samples = np.zeros((nsamples,nfeat,nbins),movement.dtype)
  bin_means = np.zeros((nfeat,nbins),movement.dtype)
  bin_medians = np.zeros((nfeat,nbins),movement.dtype)
  for i in range(nfeat):
    for j in range(nbins):
      movementcurr = torch.tensor(movement[binnum[:,i]==j,i])
      if movementcurr.shape[0] == 0:
        bin_means[i,j] = (bin_edges[i,j]+bin_edges[i,j+1])/2.
        bin_medians[i,j] = bin_means[i,j]
        samples[:,i,j] = bin_means[i,j]
      else:
        samples[:,i,j] = np.random.choice(movementcurr,size=nsamples,replace=True)
        bin_means[i,j] = np.nanmean(movementcurr)
        bin_medians[i,j] = np.nanmedian(movementcurr)
      #kde[j,i] = KernelDensity(kernel='tophat',bandwidth=kde_bandwidth).fit(movementcurr[:,None])

  return bin_edges,samples,bin_means,bin_medians

def discretize_labels(movement,bin_edges,soften_to_ends=False):

  n = movement.shape[0]
  nfeat = bin_edges.shape[0]
  nbins = bin_edges.shape[1]-1

  bin_centers = (bin_edges[:,1:]+bin_edges[:,:-1])/2.
  bin_width = (bin_edges[:,1:]-bin_edges[:,:-1])

  #d = np.zeros((n,nbins+1))
  labels = np.zeros((n,nfeat,nbins),dtype=movement.dtype)#,dtype=bool)
  if soften_to_ends:
    lastbin = 0
  else:
    lastbin = 1
    
  for i in range(nfeat):
    binnum = np.digitize(movement[:,i],bin_edges[i,:])
    binnum = np.minimum(nbins-1,np.maximum(0,binnum-1))
    # soft binning
    # don't soften into end bins
    idxsmall = (movement[:,i] < bin_centers[i,binnum]) & (binnum > lastbin)
    idxlarge = (movement[:,i] > bin_centers[i,binnum]) & (binnum < (nbins-1-lastbin))
    idxedge = (idxsmall==False) & (idxlarge==False)
    # distance from bin center, max should be .5
    d = (np.abs(movement[:,i]-bin_centers[i,binnum]) / bin_width[i,binnum])
    d[idxedge] = 0.
    labels[np.arange(n),i,binnum] = 1. - d
    labels[idxsmall,i,binnum[idxsmall]-1] = d[idxsmall]
    labels[idxlarge,i,binnum[idxlarge]+1] = d[idxlarge]
    
    #d[:,-1] = True
    #d[:,1:-1] = movement[:,i,None] <= bin_edges[None,1:-1,i]
    #labels[:,:,i] = (d[:,:-1] == False) & (d[:,1:] == True)

  return labels

def labels_discrete_to_continuous(labels_discrete,bin_edges):

  sz = labels_discrete.shape
  
  nbins = sz[-1]
  nfeat = sz[-2]
  szrest = sz[:-2]
  n = np.prod(np.array(szrest))
  labels_discrete = torch.reshape(labels_discrete,(n,nfeat,nbins))  
  # nfeat x nbins
  bin_centers = (bin_edges[:,1:]+bin_edges[:,:-1])/2.
  s = torch.sum(labels_discrete,dim=-1)
  assert torch.max(torch.abs(1-s)) < .01, 'discrete labels do not sum to 1'
  movement = torch.sum(bin_centers[None,...]*labels_discrete,dim=-1) / s
  movement = torch.reshape(movement,szrest+(nfeat,))

  return movement


def apply_mask(x,mask,nin=0,maskflagged=False):
  # mask with zeros
  if maskflagged:
    if mask is not None:
      x[mask,:-nin-1] = 0.
      x[mask,-1] = 1.    
  else:
    if mask is None:
      mask = torch.zeros(x.shape[:-1],dtype=x.dtype)
    else:
      x[mask,:-nin] = 0.
    x = torch.cat((x,mask[...,None].type(x.dtype)),dim=-1) 
  return x

def unzscore(x,mu,sig):
  return x*sig + mu
def zscore(x,mu,sig):
  return (x-mu)/sig

class FlyMLMDataset(torch.utils.data.Dataset):
  def __init__(self,data,max_mask_length=None,pmask=None,masktype='block',
               simplify_out=None,simplify_in=None,pdropout_past=0.,maskflag=None,
               input_labels=True,
               dozscore=False,
               zscore_params={},
               discreteidx=None,
               discrete_tspred=[1,],
               discretize_nbins=50,
               discretize_epsilon=None,
               discretize_params={},
               flatten_labels=False,
               flatten_obs_idx=None,
               flatten_do_separate_inputs=False,
               input_noise_sigma=None,
               p_add_input_noise=0,
               dct_ms=None,
               tspred_global=[1,],):

    # copy dicts
    self.data = [example.copy() for example in data]
    self.dtype = data[0]['input'].dtype
    # number of outputs
    self.d_output = self.data[0]['labels'].shape[-1]
    self.d_output_continuous = self.d_output
    self.d_output_discrete = 0
    
    # number of inputs
    self.dfeat = self.data[0]['input'].shape[-1]
    
    self.max_mask_length = max_mask_length
    self.pmask = pmask
    self.masktype = masktype
    self.pdropout_past = pdropout_past
    self.simplify_out = simplify_out # modulation of task to make it easier
    self.simplify_in = simplify_in
    if maskflag is None:
      maskflag = (masktype is not None) or (pdropout_past>0.)
    self.maskflag = maskflag
  
    
    self.dct_m = None
    self.idct_m = None
    if dct_ms is not None:
      self.dct_m = dct_ms[0]
      self.idct_m = dct_ms[1]
    self.tspred_global = tspred_global
    # indices of labels corresponding to the next frame if multiple frames are predicted
    tnext = np.min(self.tspred_global)
    self.nextframeidx_global = self.ravel_label_index([(f,tnext) for f in featglobal])
    if self.simplify_out is None:
      self.nextframeidx_relative = self.ravel_label_index([(i,1) for i in np.nonzero(featrelative)[0]])
    else:
      self.nextframeidx_relative = np.array([])
    self.nextframeidx = np.r_[self.nextframeidx_global,self.nextframeidx_relative]
    if self.dct_m is not None:
      dct_tau = self.dct_m.shape[0]
      # not sure if t+1 should be t+2 here -- didn't add 1 when updating code to make t = 1 mean next frame for relative features
      self.idxdct_relative = np.stack([self.ravel_label_index([(i,t+1) for i in np.nonzero(featrelative)[0]]) for t in range(dct_tau)])
    self.d_output_nextframe = len(self.nextframeidx)
    
    if input_labels:
      assert(masktype is None)
      assert(pdropout_past == 0.)
      
    self.input_labels = input_labels
    if self.input_labels:
      self.d_input_labels = self.d_output_nextframe
    else:
      self.d_input_labels = 0
    
    # which outputs to discretize, which to keep continuous
    self.discrete_idx = np.array([])
    self.discrete_tspred = np.array([1,])
    self.discretize = False
    self.continuous_idx = np.arange(self.d_output)
    self.discretize_nbins = None
    self.discretize_bin_samples = None
    self.discretize_bin_edges = None
    
    self.mu_input = None
    self.sig_input = None
    self.mu_labels = None
    self.sig_labels = None
    
    self.dtype = np.float32
    
    self.flatten_labels = False
    self.flatten_obs = False
    self.flatten_nobs_types = None
    self.flatten_nlabel_types = None
    self.flatten_dinput_pertype = None
    self.flatten_max_dinput = None
    self.flatten_max_doutput = None

    self.input_noise_sigma = input_noise_sigma
    self.p_add_input_noise = p_add_input_noise
    self.set_eval_mode()

    if dozscore:
      print('Z-scoring data...')
      self.zscore(**zscore_params)
      print('Done.')

    if discreteidx is not None:
      print('Discretizing labels...')
      self.discretize_labels(discreteidx,discrete_tspred,nbins=discretize_nbins,bin_epsilon=discretize_epsilon,**discretize_params)
      print('Done.')
    
    self.set_flatten_params(flatten_labels=flatten_labels,flatten_obs_idx=flatten_obs_idx,flatten_do_separate_inputs=flatten_do_separate_inputs)
    
    self.set_train_mode()

    
  @property
  def ntimepoints(self):
    # number of time points
    n = self.data[0]['input'].shape[-2]
    if self.input_labels and not (self.flatten_labels or self.flatten_obs) and not self.ismasked():
      n -= 1
    return n

  @property
  def ntokens_per_timepoint(self):
    if self.flatten_labels or self.flatten_obs:
      ntokens = self.flatten_nobs_types + self.flatten_nlabel_types
    else:
      ntokens = 1
    return ntokens

  @property
  def contextl(self):
    l = self.ntimepoints*self.ntokens_per_timepoint
    if (self.flatten_labels or self.flatten_obs) and not self.ismasked():
      l -= 1
    return l
  
  @property
  def flatten(self):
    return self.flatten_obs or self.flatten_labels
  
  @property
  def continuous(self):
    return (len(self.continuous_idx) > 0)
  
  @property
  def noutput_tokens_per_timepoint(self):
    if self.flatten_labels and self.discretize:
      return len(self.discrete_idx) + int(self.continuous)
    else:
      return 1
    
  @property
  def dct_tau(self):
    if self.dct_m is None:
      return 0
    else:
      return self.dct_m.shape[0]
    
  @property
  def ntspred_relative(self):
    return self.dct_tau + 1
  
  @property
  def ntspred_global(self):
    return len(self.tspred_global)
          
  def set_train_mode(self):
    self.do_add_noise = self.input_noise_sigma is not None and self.p_add_input_noise > 0

  def set_eval_mode(self):
    self.do_add_noise = False
          
  def set_flatten_params(self,flatten_labels=False,flatten_obs_idx=None,flatten_do_separate_inputs=False):
    
    self.flatten_labels = flatten_labels
    self.flatten_obs_idx = flatten_obs_idx
    if self.flatten_labels:
      if self.flatten_obs_idx is None:
        self.flatten_obs_idx = {'all': [0,self.dfeat]}
    self.flatten_obs = (self.flatten_obs_idx is not None) and (len(self.flatten_obs_idx) > 0)
    self.flatten_nobs_types = None
    self.flatten_nlabel_types = None
    self.flatten_dinput_pertype = None
    self.flatten_max_dinput = None
    self.flatten_max_doutput = None
    self.flatten_do_separate_inputs = flatten_do_separate_inputs
    
    if self.flatten_obs:
      self.flatten_nobs_types = len(self.flatten_obs_idx)
    else:
      self.flatten_nobs_types = 1
    if self.flatten_labels:
      self.flatten_nlabel_types = self.d_output_discrete
      if self.d_output_continuous > 0:
        self.flatten_nlabel_types += 1
    else:
      self.flatten_nlabel_types = 1
      
    if self.flatten:
      assert self.input_labels
      self.flatten_dinput_pertype = np.zeros(self.flatten_nobs_types+self.flatten_nlabel_types,dtype=int)
      for i,v in enumerate(self.flatten_obs_idx.values()):
        self.flatten_dinput_pertype[i] = v[1]-v[0]
      if self.flatten_labels and self.discretize:
        self.flatten_dinput_pertype[self.flatten_nobs_types:] = self.discretize_nbins
      if self.d_output_continuous > 0:
        self.flatten_dinput_pertype[-1] = self.d_output_continuous
      self.flatten_max_dinput = np.max(self.flatten_dinput_pertype)
      if self.flatten_do_separate_inputs:
        self.flatten_dinput = np.sum(self.flatten_dinput_pertype)        
      else:
        self.flatten_dinput = self.flatten_max_dinput

      self.flatten_input_type_to_range = np.zeros((self.flatten_dinput_pertype.size,2),dtype=int)
      
      if self.discretize:
        self.flatten_max_doutput = np.maximum(self.discretize_nbins,self.d_output_continuous)
      else:
        self.flatten_max_doutput = self.d_output_continuous

      if self.flatten_do_separate_inputs:
        cs = np.cumsum(self.flatten_dinput_pertype)
        self.flatten_input_type_to_range[1:,0] = cs[:-1]
        self.flatten_input_type_to_range[:,1] = cs
      else:
        self.flatten_input_type_to_range[:,1] = self.flatten_dinput_pertype

      
      # label tokens should be:
      # observations (flatten_nobs_types)
      # discrete outputs (d_output_discrete)
      # continuous outputs (<=1)
      self.idx_output_token_discrete = torch.arange(self.flatten_nobs_types,self.flatten_nobs_types+self.d_output_discrete,dtype=int)
      if self.d_output_continuous > 0:
        self.idx_output_token_continuous = torch.tensor([self.ntokens_per_timepoint-1,])
      else:
        self.idx_output_token_continuous = torch.tensor([])
        
    return
  
  def ravel_label_index(self,ftidx):
    
    idx = ravel_label_index(ftidx,dct_m=self.dct_m,tspred_global=self.tspred_global)
    return idx
  
  def unravel_label_index(self,idx):
      
    ftidx = unravel_label_index(idx,dct_m=self.dct_m,tspred_global=self.tspred_global)
    return ftidx
    
  def discretize_labels(self,discrete_idx,discrete_tspred,nbins=50,
                        bin_edges=None,bin_samples=None,bin_epsilon=None,
                        bin_means=None,bin_medians=None,**kwargs):    
    """
    discretize_labels(self,discrete_idx,discrete_tspred,nbins=50,bin_edges=None,bin_samples=None,bin_epsilon=None,**kwargs)
    For each feature in discrete_idx, discretize the labels into nbins bins. For each example in the data,
    labels_discrete is an ndarray of shape T x len(discrete_idx) x nbins, where T is the number of time points, and 
    indicates whether the label is in each bin, with soft-binning. 
    labels_todiscretize is an ndarray of shape T x len(discrete_idx) with the original continuous labels.
    labels gets replaced with an ndarray of shape T x len(continuous_idx) with the continuous labels.
    discretize_bin_edges is an ndarray of shape len(discrete_idx) x (nbins+1) with the bin edges for each discrete feature.
    discretize_bin_samples is an ndarray of shape nsamples x len(discrete_idx) x nbins with samples from each bin
    """
        
    if not isinstance(discrete_idx,np.ndarray):
      discrete_idx = np.array(discrete_idx)
    if not isinstance(discrete_tspred,np.ndarray):
      discrete_tspred = np.array(discrete_tspred)

    is_bin_epsilon_perfeat = len(bin_epsilon) == len(discrete_idx)
    if is_bin_epsilon_perfeat:
      bin_epsilon_feat = bin_epsilon
      bin_epsilon = []

    # bin_epsilon is per feature
    ftidx = []
    for i,f in enumerate(discrete_idx):
      for t in discrete_tspred:
        if f in featglobal:
          ftidx.append((f,t))
          if is_bin_epsilon_perfeat:
            bin_epsilon.append(bin_epsilon_feat[i])
        else:
          # only t = 1 can be discrete for relative features
          if t == 1:
            ftidx.append((f,1))
            if is_bin_epsilon_perfeat:
              bin_epsilon.append(bin_epsilon_feat[i])
    self.discrete_idx = self.ravel_label_index(ftidx)
    order = np.argsort(self.discrete_idx)
    self.discrete_idx = self.discrete_idx[order]
    bin_epsilon = np.array(bin_epsilon)
    if is_bin_epsilon_perfeat:
      bin_epsilon = bin_epsilon[order]
    
    self.discretize_nbins = nbins
    self.continuous_idx = np.ones(self.d_output,dtype=bool)
    self.continuous_idx[self.discrete_idx] = False
    self.continuous_idx = np.nonzero(self.continuous_idx)[0]
    self.d_output_continuous = len(self.continuous_idx)
    self.d_output_discrete = len(self.discrete_idx)
    self.discrete_tspred = discrete_tspred

    assert((bin_edges is None) == (bin_samples is None))

    if bin_edges is None:
      if self.sig_labels is not None:
        bin_epsilon = np.array(bin_epsilon) / self.sig_labels[self.discrete_idx]
      self.discretize_bin_edges,self.discretize_bin_samples,self.discretize_bin_means,self.discretize_bin_medians = \
        fit_discretize_labels(self,self.discrete_idx,nbins=nbins,bin_epsilon=bin_epsilon,**kwargs)
    else:
      self.discretize_bin_samples = bin_samples
      self.discretize_bin_edges = bin_edges
      self.discretize_bin_means = bin_means
      self.discretize_bin_medians = bin_medians
      assert nbins == bin_edges.shape[-1]-1
        
    for example in self.data:
      example['labels_todiscretize'] = example['labels'][:,self.discrete_idx]
      example['labels_discrete'] = discretize_labels(example['labels_todiscretize'],self.discretize_bin_edges,soften_to_ends=True)
      example['labels'] = example['labels'][:,self.continuous_idx]
    
    self.discretize = True    
    self.discretize_fun = lambda x: discretize_labels(x,self.discretize_bin_edges,soften_to_ends=True)
  
  def get_discretize_params(self):
    discretize_params = {
      'bin_edges': self.discretize_bin_edges.copy(),
      'bin_samples': self.discretize_bin_samples.copy(),
      'bin_means': self.discretize_bin_means.copy(),
      'bin_medians': self.discretize_bin_medians.copy(),
    }
    return discretize_params
    
  def get_bin_edges(self,zscored=False):

    if self.discretize == False:
      return
    
    if zscored or (self.mu_labels is None):
      bin_edges = self.discretize_bin_edges
    else:
      bin_edges = self.unzscore_labels(self.discretize_bin_edges.T,self.discrete_idx).T
    
    return bin_edges
  
  def get_bin_samples(self,zscored=False):
    if self.discretize == False:
      return
    
    if zscored or (self.mu_labels is None):
      bin_samples = self.discretize_bin_samples
    else:
      sz = self.discretize_bin_samples.shape
      bin_samples = self.discretize_bin_samples.transpose(0,2,1).reshape((sz[0]*sz[2],sz[1]))
      bin_samples = self.unzscore_labels(bin_samples,self.discrete_idx)
      bin_samples = bin_samples.reshape((sz[0],sz[2],sz[1])).transpose(0,2,1)
    
    return bin_samples
    
  def remove_labels_from_input(self,input):
    if self.hasmaskflag():
      return input[...,self.d_input_labels:-1]
    else:
      return input[...,self.d_input_labels:]
    
  def metadata_to_index(self,flynum,t0):
    starttoff = self.get_start_toff()
    for i,d in enumerate(self.data):
      if (d['metadata']['t0'] == t0-starttoff) and (d['metadata']['flynum'] == flynum):
        return i
    return None
    
  def hasmaskflag(self):
    return self.ismasked() or self.maskflag or self.pdropout_past > 0
    
  def ismasked(self):
    """Whether this object is a dataset for a masked language model, ow a causal model.
    v = self.ismasked()

    Returns:
        bool: Whether data are masked. 
    """
    return self.masktype is not None
    
  def zscore(self,mu_input=None,sig_input=None,mu_labels=None,sig_labels=None):
    """
    self.zscore(mu_input=None,sig_input=None,mu_labels=None,sig_labels=None)
    zscore the data. input and labels are z-scored for each example in self.data
    and converted to float32. They are stored in place in the dict for each example
    in the dataset. If mean and standard deviation statistics are input, then
    these statistics are used for z-scoring. Otherwise, means and standard deviations
    are computed from this data. 

    Args:
        mu_input (ndarray, dfeat, optional): Pre-computed mean for z-scoring input. 
        If None, mu_input is computed as the mean of all the inputs in self.data. 
        Defaults to None.
        sig_input (ndarray, dfeat, optional): Pre-computed standard deviation for 
        z-scoring input. If mu_input is None, sig_input is computed as the std of all 
        the inputs in self.data. Defaults to None. Do not set this to None if mu_input 
        is not None. 
        mu_labels (ndarray, d_output_continuous, optional): Pre-computed mean for z-scoring labels. 
        If None, mu_labels is computed as the mean of all the labels in self.data. 
        Defaults to None.
        sig_labels (ndarray, dfeat, optional): Pre-computed standard deviation for 
        z-scoring labels. If mu_labels is None, sig_labels is computed as the standard 
        deviation of all the labels in self.data. Defaults to None. Do not set this 
        to None if mu_labels is not None. 
        
    No value returned. 
    """
    
    # must happen before discretizing
    assert self.discretize == False, 'z-scoring should happen before discretizing'
    
    def zscore_helper(f):
      mu = 0.
      sig = 0.
      n = 0.
      for example in self.data:
        # input is T x dfeat
        n += np.sum(np.isnan(example[f]) == False,axis=0)
        mu += np.nansum(example[f],axis=0)
        sig += np.nansum(example[f]**2.,axis=0)
      mu = mu / n
      sig = np.sqrt(sig/n - mu**2.)
      assert(np.any(np.isnan(mu))==False)
      assert(np.any(np.isnan(sig))==False)
              
      return mu,sig

    if mu_input is None:
      self.mu_input,self.sig_input = zscore_helper('input')
    else:
      self.mu_input = mu_input.copy()
      self.sig_input = sig_input.copy()

    self.mu_input = self.mu_input.astype(self.dtype)
    self.sig_input = self.sig_input.astype(self.dtype)
      
    if mu_labels is None:
      self.mu_labels,self.sig_labels = zscore_helper('labels')
    else:
      self.mu_labels = mu_labels.copy()
      self.sig_labels = sig_labels.copy()      

    self.mu_labels = self.mu_labels.astype(self.dtype)
    self.sig_labels = self.sig_labels.astype(self.dtype)
    
    for example in self.data:
      example['input'] = self.zscore_input(example['input'])
      example['labels'] = self.zscore_labels(example['labels'])
      
  def get_zscore_params(self):
    zscore_params = {
      'mu_input': self.mu_input.copy(),
      'sig_input': self.sig_input.copy(),
      'mu_labels': self.mu_labels.copy(),
      'sig_labels': self.sig_labels.copy(),
    }
    return zscore_params
      
  def maskblock(self,inl):
    # choose a mask length
    maxl = min(inl-1,self.max_mask_length)
    l = np.random.randint(1,self.max_mask_length)
    
    # choose mask start
    t0 = np.random.randint(0,inl-l)
    t1 = t0+l
    
    # create mask
    mask = torch.zeros(inl,dtype=bool)
    mask[t0:t1] = True
    
    return mask
  
  def masklast(self,inl):
    mask = torch.zeros(inl,dtype=bool)
    mask[-1] = True
    return mask
  
  def maskind(self,inl,pmask=None):
    if pmask is None:
      pmask = self.pmask
    mask = torch.rand(inl)<=pmask
    if not torch.any(mask):
      imask = np.random.randint(inl)
      mask[imask] = True
    return mask
  
  def set_masktype(self,masktype):
    self.masktype = masktype
    
  def zscore_input(self,rawinput):
    if self.mu_input is None:
      input = rawinput.copy()
    else:
      input = (rawinput-self.mu_input)/self.sig_input
    return input.astype(self.dtype)
  
  def zscore_nextframe_labels(self,rawlabels):
    if self.mu_labels is None:
      labels = rawlabels.copy()
    else:
      # if rawlabels.shape[-1] > self.d_output_continuous:
      #   labels = rawlabels.copy()
      #   labels[...,self.continuous_idx] = (rawlabels[...,self.continuous_idx]-self.mu_labels)/self.sig_labels
      # else:
      labels = (rawlabels-self.mu_labels[self.nextframeidx])/self.sig_labels[self.nextframeidx]
    return labels.astype(self.dtype)
    
  
  def zscore_labels(self,rawlabels):
    if self.mu_labels is None:
      labels = rawlabels.copy()
    else:
      # if rawlabels.shape[-1] > self.d_output_continuous:
      #   labels = rawlabels.copy()
      #   labels[...,self.continuous_idx] = (rawlabels[...,self.continuous_idx]-self.mu_labels)/self.sig_labels
      # else:
      labels = (rawlabels-self.mu_labels)/self.sig_labels
    return labels.astype(self.dtype)

  def unzscore_nextframe_labels(self,zlabels):
    if self.mu_labels is None:
      rawlabels = zlabels.copy()
    else:
      # if zlabels.shape[-1] > self.d_output_continuous:
      #   rawlabels = zlabels.copy()
      #   rawlabels[...,self.continuous_idx] = unzscore(zlabels[...,self.continuous_idx],self.mu_labels,self.sig_labels)
      # else:
      rawlabels = unzscore(zlabels,self.mu_labels[self.nextframeidx],self.sig_labels[self.nextframeidx])
    return rawlabels.astype(self.dtype)


  def unzscore_labels(self,zlabels,featidx=None):
    if self.mu_labels is None:
      rawlabels = zlabels.copy()
    else:
      # if zlabels.shape[-1] > self.d_output_continuous:
      #   rawlabels = zlabels.copy()
      #   rawlabels[...,self.continuous_idx] = unzscore(zlabels[...,self.continuous_idx],self.mu_labels,self.sig_labels)
      # else:
      if featidx is None:
        rawlabels = unzscore(zlabels,self.mu_labels,self.sig_labels)
      else:
        rawlabels = unzscore(zlabels,self.mu_labels[featidx],self.sig_labels[featidx])
    return rawlabels.astype(self.dtype)

  
  def mask_input(self,input,masktype='default'):

    if masktype == 'default':
      masktype = self.masktype
    
    contextl = input.shape[0]
    
    if self.masktype == 'block':
      mask = self.maskblock(contextl)
    elif self.masktype == 'ind':
      mask = self.maskind(contextl)
    elif self.masktype == 'last':
      mask = self.masklast(contextl)
    else:
      mask = None      
    maskflagged = False
    if self.masktype is not None:
      input = apply_mask(input,mask,self.dfeat)
      maskflagged = True
    if self.pdropout_past > 0:
      dropout_mask = self.maskind(contextl,pmask=self.pdropout_past)
      input = apply_mask(input,dropout_mask,self.dfeat,maskflagged)
      maskflagged = True
    else:
      dropout_mask = None
    if self.maskflag and not maskflagged:
      input = apply_mask(input,None)
    
    return input,mask,dropout_mask
  
  def get_input_shapes(self):
    idx,sz = get_sensory_feature_shapes(self.simplify_in)
    if self.input_labels:
      for k,v in idx.items():
        idx[k] = [x + self.d_input_labels for x in v]
      idx['labels'] = [0,self.d_input_labels]
      sz['labels'] = (self.d_input_labels,)
    return idx,sz

  def unpack_input(self,input,dim=-1):
    
    idx,sz = self.get_input_shapes()
    res = unpack_input(input,idx,sz,dim=dim)
    
    return res
    
  def input2pose(self,input):
    """
    pose = self.input2pose(input)
    Extracts the relative pose features. 

    Args:
        input (ndarray or Tensor, (... x nfeat) or (... x (d_input_labels+nfeat+ismasked)): input features
        to process. 

    Returns:
        pose ndarray or Tensor, ... x npose: relative pose features extracted from input
    """
    if input.shape[-1] > self.dfeat:
      input = self.remove_labels_from_input(input)
    res = split_features(input,simplify=self.simplify_in)
    return res['pose']
    
  def get_start_toff(self):
    if self.ismasked() or (self.input_labels == False) or \
      self.flatten_labels or self.flatten_obs:
      starttoff = 0
    else:
      starttoff = 1
    return starttoff
    
  def __getitem__(self,idx: int):
    """
    example = self.getitem(idx)
    Returns dataset example idx. It performs the following processing:
    - Converts the data to tensors.
    - Concatenates labels and feature input into input, and shifts labels in inputs
    depending on whether this is a dataset for a masked LM or a causal LM (see below). 
    - For masked LMs, draw and applies a random mask of type self.masktype. 

    Args:
        idx (int): Index of the example to return. 

    Returns:
        example (dict) with the following fields:

        For masked LMs: 
        example['input'] is a tensor of shape contextl x (d_input_labels + dfeat + 1)
        where example['input'][t,:d_input_labels] is the motion from frame t to t+1 and
        example['input'][t,d_input_labels:-1] are the input features for frame t. 
        example['input'][t,-1] indicates whether the frame is masked or not. If this 
        frame is masked, then example['input'][t,:d_input_labels] will be set to 0. 
        example['labels'] is a tensor of shape contextl x d_output_continuous
        where example['labels'][t,:] is the continuous motion from frame t to t+1. 
        example['labels_discrete'] is a tensor of shape contextl x d_output_discrete x 
        discretize_nbins, where example['labels_discrete'][t,i,:] is one-hot encoding of 
        discrete motion feature i from frame t to t+1. 
        example['init'] is a tensor of shape dglobal, corresponding to the global
        position in frame 0. 
        example['mask'] is a tensor of shape contextl indicating which frames are masked.
        
        For causal LMs:
        example['input'] is a tensor of shape (contextl-1) x (d_input_labels + dfeat).
        if input_labels == True, example['input'][t,:d_input_labels] is the motion from 
        frame t to t+1 and example['input'][t,d_input_labels:] are the input features for 
        frame t+1. example['labels'] is a tensor of shape contextl x d_output
        where example['labels'][t,:] is the motion from frame t+1 to t+2.
        example['init'] is a tensor of shape dglobal, corresponding to the global
        position in frame 1. 
        example['labels_discrete'] is a tensor of shape contextl x d_output_discrete x 
        discretize_nbins, where example['labels_discrete'][t,i,:] is one-hot encoding of 
        discrete motion feature i from frame t+1 to t+2. 

        For all:
        example['scale'] are the scale features for this fly, used for converting from
        relative pose features to keypoints. 
        example['categories'] are the currently unused categories for this sequence.
        example['metadata'] is a dict of metadata about this sequence.
        
    """
    
    datacurr = self.data[idx]
    
    input = torch.as_tensor(datacurr['input'].copy())
    labels = torch.as_tensor(datacurr['labels'].copy())
    if self.discretize:
      labels_todiscretize = torch.as_tensor(datacurr['labels_todiscretize'].copy())
      labels_discrete = torch.as_tensor(datacurr['labels_discrete'].copy())
    else:
      labels_todiscretize = None
      labels_discrete = None
    if self.input_labels:
      # should we use all future predictions, or just the next time point?
      if self.discretize:
        input_labels = torch.zeros((labels.shape[0],self.d_output),dtype=labels.dtype)
        input_labels[:,self.continuous_idx] = labels
        input_labels[:,self.discrete_idx] = labels_todiscretize
        input_labels = input_labels[:,self.nextframeidx] # probably could be made faster
      else:
        input_labels = labels[:,self.nextframeidx].clone()
    else:
      input_labels = None
    
    # add_noise
    if self.do_add_noise:
      eta,input,input_labels,labels,labels_todiscretize,labels_discrete = self.add_noise(input,input_labels,labels,labels_todiscretize,labels_discrete)
    
    # whether we start with predicting the 0th or the 1th frame in the input sequence
    starttoff = self.get_start_toff()

    init = torch.as_tensor(datacurr['init'][:,starttoff].copy())      
    scale = torch.as_tensor(datacurr['scale'].copy())
    categories = torch.as_tensor(datacurr['categories'].copy())
    metadata =copy.deepcopy(datacurr['metadata'])
    metadata['t0'] += starttoff
    metadata['frame0'] += starttoff

    res = {'input': None, 'labels': None, 'labels_discrete': None,
           'labels_todiscretize': None,
           'init': init, 'scale': scale, 'categories': categories,
           'metadata': metadata}

    if self.discretize:
      res['labels_discrete'] = labels_discrete[starttoff:,:,:]
      res['labels_todiscretize'] = labels_todiscretize[starttoff:,:]

    nin = input.shape[-1]
    contextl = input.shape[0]
    input,mask,dropout_mask = self.mask_input(input)
    
    if self.flatten:
      ntypes = self.ntokens_per_timepoint
      #newl = contextl*ntypes
      newlabels = torch.zeros((contextl,ntypes,self.flatten_max_doutput),dtype=input.dtype)
      newinput = torch.zeros((contextl,ntypes,self.flatten_dinput),dtype=input.dtype)
      newmask = torch.zeros((contextl,ntypes),dtype=bool)
      #offidx = np.arange(contextl)*ntypes
      if self.flatten_obs:
        for i,v in enumerate(self.flatten_obs_idx.values()):
          newinput[:,i,self.flatten_input_type_to_range[i,0]:self.flatten_input_type_to_range[i,1]] = input[:,v[0]:v[1]]
          newmask[:,i] = False
      else:
        newinput[:,0,:self.flatten_dinput_pertype[0]] = input
      if self.discretize:
        if self.flatten_labels:
          for i in range(self.d_output_discrete):
            inputnum = self.flatten_nobs_types+i
            newlabels[:,inputnum,:self.discretize_nbins] = labels_discrete[:,i,:]
            newinput[:,inputnum,self.flatten_input_type_to_range[inputnum,0]:self.flatten_input_type_to_range[inputnum,1]] = labels_discrete[:,i,:]
            if mask is None:
              newmask[:,self.flatten_nobs_types+i] = True
            else:
              newmask[:,self.flatten_nobs_types+i] = mask.clone()
          if self.continuous:
            inputnum = -1
            newlabels[:,-1,:labels.shape[-1]] = labels
            newinput[:,-1,self.flatten_input_type_to_range[inputnum,0]:self.flatten_input_type_to_range[inputnum,1]] = labels
            if mask is None:
              newmask[:,-1] = True
            else:
              newmask[:,-1] = mask.clone()
        else:
          newinput[:,-1,:self.d_output] = labels
      newlabels = newlabels.reshape((contextl*ntypes,self.flatten_max_doutput))
      newinput = newinput.reshape((contextl*ntypes,self.flatten_dinput))
      newmask = newmask.reshape(contextl*ntypes)
      if not self.ismasked():
        newlabels = newlabels[1:,:]
        newinput = newinput[:-1,:]
        newmask = newmask[1:]
        
      res['input'] = newinput
      res['input_stacked'] = input
      res['mask_flattened'] = newmask
      res['labels'] = newlabels
      res['labels_stacked'] = labels
    else:
      if self.input_labels:
        input = torch.cat((input_labels[:-starttoff,:],input[starttoff:,:]),dim=-1)
      labels = labels[starttoff:,:]
      res['input'] = input
      res['labels'] = labels

    if mask is not None:
      res['mask'] = mask
    if dropout_mask is not None:
      res['dropout_mask'] = dropout_mask
    return res
  
  def add_noise(self,input,input_labels,labels,labels_todiscretize,labels_discrete):

    # add noise to the input movement and pose
    # desire is to do movement truemovement(t-1->t)
    # movement noisemovement(t-1->t) = truemovement(t-1->t) + eta(t) actually done
    # resulting in noisepose(t) = truepose(t) + eta(t)[featrelative]
    # output should then be fixmovement(t->t+1) = truemovement(t->t+1) - eta(t)
    # input pose: noise_input_pose(t) = truepose(t) + eta(t)[featrelative]
    # input movement: noise_input_movement(t-1->t) = truemovement(t-1->t) + eta(t)
    # output movement: noise_output_movement(t->t+1) = truemovement(t->t+1) - eta(t)
    # movement(t->t+1) = truemovement(t->t+1)

    T = input.shape[0]

    # additive noise
    input_noise_sigma = self.input_noise_sigma
    if self.sig_labels is not None:
      input_noise_sigma = input_noise_sigma / self.sig_labels
    eta = np.zeros((T,self.d_output))
    do_add_noise = np.random.rand(T) <= self.p_add_input_noise
    eta[do_add_noise,:] = input_noise_sigma[None,:]*np.random.randn(np.count_nonzero(do_add_noise),self.d_output)
    eta = torch.as_tensor(eta).to(dtype=labels.dtype,device=labels.device)

    # problem with multiplicative noise is that it is 0 when the movement is 0 -- there should always be some jitter
    # etamult = np.maximum(-self.max_input_noise,np.minimum(self.max_input_noise,self.input_noise_sigma[None,:]*np.random.randn(input.shape[0],self.d_output)))
    # if self.input_labels:
    #   eta = input_labels*etamult
    # else:
    #   labelsprev = torch.zeros((labels.shape[0],nfeatures),dtype=labels.dtype,device=labels.device)
    #   if self.continuous:
    #     labelsprev[1:,self.continuous_idx] = labels[:-1,:]
    #   if self.discretize:
    #     labelsprev[1:,self.discrete_idx] = labels_todiscretize[:-1,:]
    #   eta = labelsprev*etamult

    # input pose
    inputidx = get_sensory_feature_idx(self.simplify_in)
    i0 = inputidx['pose'][0]
    i1 = inputidx['pose'][1]
    input[:,i0:i1] += eta[:,featrelative]
    # input labels
    if self.input_labels:
      input_labels += eta
    # output labels
    if self.continuous:
      labels -= eta[:,self.continuous_idx]
    if self.discretize and np.any(self.input_noise_sigma[self.discrete_idx]!=0.):
      labels_todiscretize -= eta[:,self.discrete_idx]
      labels_discrete = torch.as_tensor(self.discretize_fun(labels_todiscretize.numpy()))
    return eta,input,input_labels,labels,labels_todiscretize,labels_discrete
  
  def __len__(self):
    return len(self.data)
  
  def pred2pose(self,posein,globalin,pred,isnorm=True):
    """
    posenext,globalnext = self.pred2pose(posein,globalin,pred)
    Adds the motion in pred to the pose defined by posein and globalin 
    to compute the pose in the next frame. 

    Args:
        posein (ndarray, dposerel): Unnormalized relative pose features for the current 
        frame globalin (ndarray, dglobal): Global position for the current frame
        pred (ndarray, d_output): Z-scored (if applicable) predicted movement from 
        current frame to the next frame
        isnorm (bool,optional): Whether the input pred is normalized. Default: True.

    Returns:
        posenext (ndarray, dposerel): Unnormalized relative pose features for the next 
        frame. 
        globalnext (ndarray, dglobal): Global position for the next frame. 
        
    Example call: relposenext,globalposnext = self.pred2pose(relposecurr,globalposcurr,zmovementout)        
    """
    if isnorm:
      movement = self.unzscore_labels(pred)
    else:
      movement = pred
      
    if pred.ndim > posein.ndim:
      nsamples = pred.shape[0]
    else:
      nsamples = 1

    # nsamples x tspred_global x nglobal
    movement_global = self.get_global_movement(movement)
    originvelrel = movement_global[:,:,[1,0],...]
    theta0 = globalin[2]
    sz = originvelrel.shape
    originvel = mabe.rotate_2d_points(originvelrel.reshape((-1,sz[-1])),-theta0).reshape(sz)
    globalnext = np.tile(globalin[None,None,:],(nsamples,self.ntspred_global,1))
    globalnext[:,:,:2,...] += originvel
    globalnext[:,:,2,...] += movement_global[:,:,2,...]
    globalnext[:,:,2,...] = mabe.modrange(globalnext[:,:,2,...],-np.pi,np.pi)
    
    if self.simplify_out == 'global':
      posenext = globalnext
    else:
      movement_relative_next = self.get_next_relative_movement(movement)
      if self.dct_m is None:
        # nsamples x ntspred_relative x nfeatrelative
        movement_relative = movement_relative_next[:,None,...]
      else:
        # to do: not sure how this will work with multiple samples
        movement_relative = self.get_relative_movement_dct(movement,iszscored=False)
        movement_relative[0,...] = movement_relative_next
      # nsamples x ntspred_relative x nfeatrelative
      posenext = posein+movement_relative
      #globalnext = globalin+movement[featglobal]
      
    if pred.ndim == posein.ndim:
      # select out the 0th sample
      posenext = posenext[0]
      globalnext = globalnext[0]
      
    return posenext,globalnext
  
  def get_global_movement_idx(self):
    idxglobal = self.ravel_label_index(np.stack(np.meshgrid(featglobal,self.tspred_global),axis=-1))
    return idxglobal
  
  def get_global_movement(self,movement):
    idxglobal = self.get_global_movement_idx()
    movement_global = movement[...,idxglobal]
    return movement_global
  
  def set_global_movement(self,movement_global,movement):
    idxglobal = self.get_global_movement_idx()
    movement[...,idxglobal] = movement_global
    return movement
  
  def get_global_movement_discrete(self,movement_discrete):
    if not self.discretize:
      return None
    idxglobal = self.get_global_movement_idx()
    movement_global_discrete = np.zeros(movement_discrete.shape[:-2]+idxglobal.shape+movement_discrete.shape[-1:],dtype=self.dtype)
    movement_global_discrete[:] = np.nan
    for i in range(idxglobal.shape[0]):
      for j in range(idxglobal.shape[1]):
        idx = idxglobal[i,j]
        didx = np.nonzero(self.discrete_idx==idx)[0]
        if len(didx) == 0:
          continue
        movement_global_discrete[...,i,j,:] = movement_discrete[...,didx[0],:]
    return movement_global_discrete

  def get_next_relative_movement(self,movement):
    movement_next_relative = movement[...,self.nextframeidx_relative]
    return movement_next_relative

  def get_relative_movement_dct(self,movements,iszscored=False):
    movements_dct = movements[...,self.idxdct_relative]
    if not iszscored and self.mu_labels is not None:
      movements_dct = unzscore(movements_dct,self.mu_labels[self.idxdct_relative],self.sig_labels[self.idxdct_relative])
    movements_relative = self.idct_m @ movements_dct
    return movements_relative
  
  def get_next_relative_movement_dct(self,movements,iszscored=True,dozscore=True):
    if self.simplify_out == 'global':
      return movements[...,[]]
    
    if type(movements) is np.ndarray:
      movements = torch.as_tensor(movements)

    movements_dct = movements[...,self.idxdct_relative]
    if not iszscored and self.mu_labels is not None:
      mu = torch.as_tensor(self.mu_labels[self.idxdct_relative]).to(dtype=movements.dtype,device=movements.device)
      sig = torch.as_tensor(self.sig_labels[self.idxdct_relative]).to(dtype=movements.dtype,device=movements.device)
      movements_dct = unzscore(movements_dct,mu,sig)
      
    idct_m0 = torch.as_tensor(self.idct_m[[0,],:]).to(dtype=movements.dtype,device=movements.device)
    dctfeat = movements[...,self.idxdct_relative]
    movements_next_relative = torch.matmult(idct_m0,dctfeat)
    
    if dozscore:
      movements_next_relative = zscore(movements_next_relative,self.mu_labels[self.nextframeidx_relative],self.sig_labels[self.nextframeidx_relative])
      
    return movements_next_relative
  
  def compare_dct_to_next_relative(self,movements):
    movements_next_relative_dct = self.get_next_relative_movement_dct(movements,iszscored=True,dozscore=True)
    movements_next_relative0 = movements[...,self.nextframeidx_relative]
    err = movements_next_relative_dct - movements_next_relative0
    return err
  
  def get_next_movements(self,movements=None,example=None,iszscored=False,use_dct=False,**kwargs):
    if movements is None:
      movements = self.get_full_labels(example=example,**kwargs)
      iszscored = True
      
    if torch.is_tensor(movements):
      movements = movements.numpy()
      
    if iszscored and self.mu_labels is not None:
      movements = unzscore(movements,self.mu_labels,self.sig_labels)
    
    movements_next_global = movements[...,self.nextframeidx_global]
    if self.simplify_out is None:
      if use_dct and self.dct_m is not None:
        dctfeat = movements[...,self.idxdct_relative]
        movements_next_relative = self.idct_m[[0,],:] @ dctfeat
      else:
        movements_next_relative = movements[...,self.nextframeidx_relative]
      movements_next = np.concatenate((movements_next_global,movements_next_relative),axis=-1)
    else:
      movements_next = movements_next_global
    return movements_next
  
  def get_Xfeat(self,input0=None,global0=None,movements=None,example=None,use_dct=False,**kwargs):
    """
    Xfeat = self.get_Xfeat(input0,global0,movements)
    Xfeat = self.get_Xfeat(example=example)

    Unnormalizes initial input input0 and extracts relative pose features. Combines
    these with global0 to get the full set of pose features for initial frame 0. 
    Converts egocentric movements (forward, sideway) to global, and computes the
    full pose features for each frame based on the input movements. 

    Either input0, global0, and movements must be input OR 
    example must be input, and input0, global0, and movements are derived from there.

    Args:
        input0 (ndarray, d_input_labels+dfeat+hasmaskflag): network input for time point 0
        global0 (ndarray, 3): global position at time point 0
        movements (ndarray, T x d_output ): movements[t,:] is the movement from t to t+1

    Returns:
        Xfeat: (ndarray, T+1 x nfeatures): All pose features for frames 0 through T
    """
    
    if example is not None:
      if input0 is None:
        input = self.get_full_inputs(example=example)
        input0 = input[...,0,:]
      if global0 is None:
        global0 = example['init']
      if movements is None:
        movements = self.get_full_labels(example=example,**kwargs)

    szrest = movements.shape[:-1]
    n = np.prod(np.array(szrest))
    
    if torch.is_tensor(input0):
      input0 = input0.numpy()
    if torch.is_tensor(global0):
      global0 = global0.numpy()
    if torch.is_tensor(movements):
      movements = movements.numpy()
    
    if self.mu_input is not None:
      input0 = unzscore(input0,self.mu_input,self.sig_input)
      movements = unzscore(movements,self.mu_labels,self.sig_labels)
      
    movements_next = self.get_next_movements(movements=movements,iszscored=False,use_dct=use_dct)
      
    input0 = split_features(input0,simplify=self.simplify_in)
    Xorigin0 = global0[...,:2]
    Xtheta0 = global0[...,2] 
    thetavel = movements_next[...,feattheta]
    
    Xtheta = np.cumsum(np.concatenate((Xtheta0[...,None],thetavel),axis=-1),axis=-1)
    Xoriginvelrel = movements_next[...,[featorigin[1],featorigin[0]]]
    Xoriginvel = mabe.rotate_2d_points(Xoriginvelrel.reshape((n,2)),-Xtheta[...,:-1].reshape(n)).reshape(szrest+(2,))
    Xorigin = np.cumsum(np.concatenate((Xorigin0[...,None,:],Xoriginvel),axis=-2),axis=-2)
    Xfeat = np.zeros(szrest[:-1]+(szrest[-1]+1,nfeatures),dtype=self.dtype)
    Xfeat[...,featorigin] = Xorigin
    Xfeat[...,feattheta] = Xtheta

    if self.simplify_out == 'global':
      Xfeat[...,featrelative] = np.tile(input0['pose'],szrest[:-1]+(szrest[-1]+1,1))
    else:
      Xfeatpose = np.cumsum(np.concatenate((input0['pose'][...,None,:],movements_next[...,featrelative]),axis=-2),axis=-2)
      Xfeat[...,featrelative] = Xfeatpose
    
    return Xfeat
  
  def get_Xkp(self,example,pred=None,**kwargs):
    """
    Xkp = self.get_Xkp(example,pred=None)

    Call get_Xfeat to get the full pose features based on the initial input and global
    position example['input'] and example['init'] and the per-frame motion in 
    pred (if not None) or example['labels'], example['labels_discrete']. Converts
    the full pose features to keypoint coordinates. 
    
    Args:
        scale (ndarray, dscale): scale parameters for this fly
        example (dict), output of __getitem__: example with fields input, init, labels, and
        scale. 
        pred (ndarray, T x d_output ): movements[t,:] is the movement from t to t+1

    Returns:
        Xkp: (ndarray, nkeypoints x 2 x T+1 x 1): Keypoint locations for frames 0 through T
    """
    
    scale = example['scale']
    if torch.is_tensor(scale):
      scale = scale.numpy()

    if pred is not None:
      movements = self.get_full_pred(pred,**kwargs)
    else:
      movements = None
    Xfeat = self.get_Xfeat(example=example,movements=movements,**kwargs)
    Xkp = self.feat2kp(Xfeat,scale)
    return Xkp

  
  def get_Xkp0(self,input0=None,global0=None,movements=None,scale=None,example=None):
    """
    Xkp = self.get_Xkp(input0,global0,movements)

    Call get_Xfeat to get the full pose features based on the initial input and global
    position input0 and global0 and the per-frame motion in movements. Converts
    the full pose features to keypoint coordinates. 
    
    Either input0, global0, movements, and scale must be input OR 
    example must be input, and input0, global0, movements, and scale are derived from there

    Args:
        input0 (ndarray, d_input_labels+dfeat+hasmaskflag): network input for time point 0
        global0 (ndarray, 3): global position at time point 0
        movements (ndarray, T x d_output ): movements[t,:] is the movement from t to t+1
        scale (ndarray, dscale): scale parameters for this fly
        example (dict), output of __getitem__: example with fields input, init, labels, and
        scale. 

    Returns:
        Xkp: (ndarray, nkeypoints x 2 x T+1 x 1): Keypoint locations for frames 0 through T
    """
    
    if example is not None and scale is None:
      scale = example['scale']
    if torch.is_tensor(scale):
      scale = scale.numpy()
    
    Xfeat = self.get_Xfeat(input0=input0,global0=global0,movements=movements,example=example)
    Xkp = self.feat2kp(Xfeat,scale)
    return Xkp
  
  def feat2kp(self,Xfeat,scale):
    """
    Xkp = self.feat2kp(Xfeat)

    Args:
        Xfeat (ndarray, T x nfeatures): full pose features for each frame
        scale (ndarray, dscale): scale features

    Returns:
        Xkp (ndarray, nkeypoints x 2 x T+1 x 1): keypoints for each frame
    """
    Xkp = mabe.feat2kp(Xfeat.T[...,None],scale[...,None])
    return Xkp
  
  def construct_input(self,obs,movement=None):    
    
    # to do: merge this code with getitem so that we don't have to duplicate
    dtype = obs.dtype
    
    if self.input_labels:
      assert (movement is not None)
    
    if self.flatten:
      xcurr = np.zeros((obs.shape[0],self.ntokens_per_timepoint,self.flatten_dinput),dtype=dtype)

      if self.flatten_obs:
        for i,v in enumerate(self.flatten_obs_idx.values()):
          xcurr[:,i,self.flatten_input_type_to_range[i,0]:self.flatten_input_type_to_range[i,1]] = obs[:,v[0]:v[1]]
      else:
        xcurr[:,0,:self.flatten_dinput_pertype[0]] = obs
  
      if self.input_labels:
         # movement not set for last time points, will be 0s
        if self.flatten_labels:
          for i in range(movement.shape[1]):
            if i < len(self.discrete_idx):
              dmovement = self.discretize_nbins
            else:
              dmovement = len(self.continuous_idx)
            inputnum = self.flatten_nobs_types+i
            xcurr[:-1,inputnum,self.flatten_input_type_to_range[inputnum,0]:self.flatten_input_type_to_range[inputnum,1]] = movement[:,i,:dmovement]
        else:
          inputnum = self.flatten_nobs_types
          xcurr[:-1,inputnum,self.flatten_input_type_to_range[inputnum,0]:self.flatten_input_type_to_range[inputnum,1]] = movement
      xcurr = np.reshape(xcurr,(xcurr.shape[0]*xcurr.shape[1],xcurr.shape[2]))

    else:
      if self.input_labels:      
        xcurr = np.concatenate((movement,obs[1:,...]),axis=-1)
      else:
        xcurr = obs
    
    
    return xcurr
  
  def get_movement_npad(self):
    npad = compute_npad(self.tspred_global,self.dct_m)
    return npad
  
  def predict_open_loop(self,Xkp,fliespred,scales,burnin,model,maxcontextl=np.inf,debug=False,need_weights=False,nsamples=0):
      """
      Xkp = predict_open_loop(self,Xkp,fliespred,scales,burnin,model,sensory_params,maxcontextl=np.inf,debug=False)

      Args:
        Xkp (ndarray, nkpts x 2 x tpred x nflies): keypoints for all flies for all frames.
        Can be nan for frames/flies to be predicted. Will be overwritten. 
        fliespred (ndarray, nfliespred): indices of flies to predict
        scales (ndarray, nscale x nfliespred): scale parameters for the flies to be predicted
        burnin (int): number of frames to use for initialization
        maxcontextl (int, optional): maximum number of frames to use for context. Default np.inf
        debug (bool, optional): whether to fill in from movement computed from Xkp_all

      Returns:
        Xkp (ndarray, nkpts x 2 x tpred x nflies): keypoints for all flies for all frames,
        with predicted frames/flies filled in. 
        
      Example call:
      res = dataset.predict_open_loop(Xkp,fliespred,scales,burnin,model,maxcontextl=config['contextl'],
                                debug=debug,need_weights=plotattnweights,nsamples=nsamplesfuture)
      """
      model.eval()

      with torch.no_grad():
        w = next(iter(model.parameters()))
        dtype = w.cpu().numpy().dtype
        device = w.device
        
      if nsamples == 0:
        nsamples1 = 1
      else:
        nsamples1 = nsamples
        
      # propagate forward with the 0th sample
      selectsample = 0

      tpred = Xkp.shape[-2]
      nfliespred = len(fliespred)
      relpose = np.zeros((nsamples1,tpred,nrelative,nfliespred),dtype=dtype)
      globalpos = np.zeros((nsamples1,tpred,nglobal,nfliespred),dtype=dtype)
      if self.dct_tau == 0:
        ntspred_rel = 1
      else:
        ntspred_rel = self.dct_tau
      relposefuture = np.zeros((nsamples1,tpred,ntspred_rel,nrelative,nfliespred),dtype=dtype)
      globalposfuture = np.zeros((nsamples1,tpred,self.ntspred_global,nglobal,nfliespred),dtype=dtype)
      relposefuture[:] = np.nan
      globalposfuture[:] = np.nan
      sensory = None
      zinputs = None
      if need_weights:
        attn_weights = [None,]*tpred

      if debug:
        movement_true = compute_movement(X=Xkp[...,fliespred],scale=scales,simplify=self.simplify_out).transpose((1,0,2)).astype(dtype)
      
      # outputs -- hide frames past burnin
      Xkp[:,:,burnin:,fliespred] = np.nan
      
      # compute the pose for pred flies for first burnin frames
      relpose0,globalpos0 = compute_pose_features(Xkp[...,:burnin,fliespred],scales)
      relpose[:,:burnin] = relpose0.transpose((1,0,2))
      globalpos[:,:burnin] = globalpos0.transpose((1,0,2))
      # compute one-frame movement for pred flies between first burnin frames 
      movement0 = compute_movement(relpose=relpose0,
                                  globalpos=globalpos0,
                                  simplify=self.simplify_out)
      
      movement0 = movement0.transpose((1,0,2))
      if self.flatten:
        zmovement = np.zeros((tpred-1,self.noutput_tokens_per_timepoint,self.flatten_max_doutput,nfliespred),dtype=dtype)
      else:
        zmovement = np.zeros((tpred-1,movement0.shape[1],nfliespred),dtype=dtype)
        
      for i in range(nfliespred):
        zmovementcurr = self.zscore_nextframe_labels(movement0[...,i])
        if self.flatten:
          if self.discretize:
            zmovement_todiscretize = zmovementcurr[...,self.discrete_idx]
            zmovement_discrete = discretize_labels(zmovement_todiscretize,self.discretize_bin_edges,soften_to_ends=True)
            zmovement[:burnin-1,:len(self.discrete_idx),:self.discretize_nbins,i] = zmovement_discrete
          if self.continuous:
            zmovement[:burnin-1,-1,:len(self.continuous_idx),i] = zmovementcurr[...,self.continuous_idx]            
        else:
          zmovement[:burnin-1,:,i] = zmovementcurr
        
      # compute sensory features for first burnin frames
      if self.simplify_in is None:
        for i in range(nfliespred):
          flynum = fliespred[i]
          sensorycurr = compute_sensory_wrapper(Xkp[...,:burnin,:],flynum,
                                                theta_main=globalpos[0,:burnin,featthetaglobal,i]) # 0th sample
          if sensory is None:
            nsensory = sensorycurr.shape[0]
            sensory = np.zeros((tpred,nsensory,nfliespred),dtype=dtype)
          sensory[:burnin,:,i] = sensorycurr.T
    
      for i in range(nfliespred):
        if self.simplify_in is None:
          rawinputscurr = combine_inputs(relpose=relpose[0,:burnin,:,i],
                                         sensory=sensory[:burnin,:,i],dim=1)
        else:
          rawinputscurr = relpose[:burnin,:,i,0]
          
        zinputscurr = self.zscore_input(rawinputscurr)
        
        # if self.flatten_obs:
        #   zinputscurr = self.apply_flatten_input(zinputscurr)
        # elif self.flatten:
        #   zinputscurr = zinputscurr[:,None,:]
        if zinputs is None:
          zinputs = np.zeros((tpred,)+zinputscurr.shape[1:]+(nfliespred,),dtype=dtype)
        zinputs[:burnin,...,i] = zinputscurr
              
      if self.ismasked():
        # to do: figure this out for flattened models
        masktype = 'last'
        dummy = np.zeros((1,self.d_output))
        dummy[:] = np.nan
      else:
        masktype = None
      
      # start predicting motion from frame burnin-1 to burnin = t
      for t in tqdm.trange(burnin,tpred):
        t0 = int(np.maximum(t-maxcontextl,0))
        
        masksize = t-t0
        if self.flatten:
          masksize*=self.ntokens_per_timepoint
        if self.input_labels:
          masksize -= self.noutput_tokens_per_timepoint
        
        if self.ismasked():
          net_mask = generate_square_full_mask(masksize).to(device)
          is_causal = False
        else:
          net_mask = torch.nn.Transformer.generate_square_subsequent_mask(masksize).to(device)
          is_causal = True
              
        for i in range(nfliespred):
          flynum = fliespred[i]
          zinputcurr = zinputs[t0:t,...,i]
          relposecurr = relpose[0,t-1,:,i] # 0th sample
          globalposcurr = globalpos[0,t-1,:,i] # 0th sample

          
          if self.input_labels:
            zmovementin = zmovement[t0:t-1,...,i]
            if self.ismasked():
              # to do: figure this out for flattened model
              zmovementin = np.r_[zmovementin,dummy]
          else:
            zmovementin = None
          xcurr = self.construct_input(zinputcurr,movement=zmovementin)
          
          # if self.flatten:
          #     xcurr = np.zeros((zinputcurr.shape[0],self.ntokens_per_timepoint,self.flatten_max_dinput),dtype=dtype)
          #     xcurr[:,:self.flatten_nobs_types,:] = zinputcurr
          #     # movement not set for last time points, will be 0s
          #     xcurr[:-1,self.flatten_nobs_types:,:self.flatten_max_doutput] = zmovementin
          #     xcurr = np.reshape(xcurr,(xcurr.shape[0]*xcurr.shape[1],xcurr.shape[2]))
          #     lastidx = xcurr.shape[0]-self.noutput_tokens_per_timepoint
          #   else:
          #     xcurr = np.concatenate((zmovementin,zinputcurr[1:,...]),axis=-1)
          # else:
          #   xcurr = zinputcurr

          
            
          xcurr = torch.tensor(xcurr)
          xcurr,_,_ = self.mask_input(xcurr,masktype)
          
          if debug:
            zmovementout = np.tile(self.zscore_labels(movement_true[t-1,:,i]).astype(dtype)[None],(nsamples1,1))
          else:
            
            if self.flatten:
              # to do: not sure if multiple samples here works
              
              zmovementout = np.zeros((nsamples1,self.d_output),dtype=dtype)
              zmovementout_flattened = np.zeros((self.noutput_tokens_per_timepoint,self.flatten_max_doutput),dtype=dtype)
              
              for token in range(self.noutput_tokens_per_timepoint):

                lastidx = xcurr.shape[0]-self.noutput_tokens_per_timepoint
                masksize = lastidx+token
                if self.ismasked():
                  net_mask = generate_square_full_mask(masksize).to(device)
                else:
                  net_mask = torch.nn.Transformer.generate_square_subsequent_mask(masksize,device=device)

                with torch.no_grad():
                  predtoken = model(xcurr[None,:lastidx+token,...].to(device),mask=net_mask,is_causal=is_causal)
                if token < len(self.discrete_idx):
                  # sample
                  sampleprob = torch.softmax(predtoken[0,-1,:self.discretize_nbins],dim=-1)
                  binnum = int(weighted_sample(sampleprob,nsamples=nsamples1))
                  
                  # store in input
                  xcurr[lastidx+token,binnum[0]] = 1.
                  zmovementout_flattened[token,binnum[0]] = 1.
                  
                  # convert to continuous
                  nsamples_per_bin = self.discretize_bin_samples.shape[0]
                  sample = int(torch.randint(low=0,high=nsamples_per_bin,size=(nsamples,)))
                  zmovementcurr = self.discretize_bin_samples[sample,token,binnum]
                  
                  # store in output
                  zmovementout[:,self.discrete_idx[token]] = zmovementcurr
                else: # else token < len(self.discrete_idx)
                  # continuous
                  zmovementout[:,self.continuous_idx] = predtoken[0,-1,:len(self.continuous_idx)].cpu()
                  zmovementout_flattened[token,:len(self.continuous_idx)] = zmovementout[self.continuous_idx,0]
                  
            else: # else flatten
            
              if need_weights:
                with torch.no_grad():
                  pred,attn_weights_curr = get_output_and_attention_weights(model,xcurr[None,...].to(device),net_mask)
                # dimensions correspond to layer, output frame, input frame
                attn_weights_curr = torch.cat(attn_weights_curr,dim=0).cpu().numpy()
                if i == 0:
                  attn_weights[t] = np.tile(attn_weights_curr[...,None],(1,1,1,nfliespred))
                  attn_weights[t][...,1:] = np.nan
                else:
                  attn_weights[t][...,i] = attn_weights_curr
              else:
                with torch.no_grad():
                  # predict for all frames
                  # masked: movement from 0->1, ..., t->t+1
                  # causal: movement from 1->2, ..., t->t+1
                  # last prediction: t->t+1
                  pred = model.output(xcurr[None,...].to(device),mask=net_mask,is_causal=is_causal)
              if model.model_type == 'TransformerBestState' or model.model_type == 'TransformerState':
                pred = model.randpred(pred)
              # z-scored movement from t to t+1
              pred = pred_apply_fun(pred,lambda x: x[0,-1,...].cpu())
              zmovementout = self.get_full_pred(pred,sample=True,nsamples=nsamples)
              zmovementout = zmovementout.numpy()
            # end else flatten
          # end else debug

          if nsamples == 0:
            zmovementout = zmovementout[None,...]
          # relposenext is nsamples x ntspred_relative x nrelative
          # globalposnext is nsamples x ntspred_global x nglobal
          relposenext,globalposnext = self.pred2pose(relposecurr,globalposcurr,zmovementout)
          relpose[:,t,:,i] = relposenext[:,0] # select next time point, all samples, all features
          globalpos[:,t,:,i] = globalposnext[:,0]
          # relposefuture is (nsamples1,tpred,ntspred_rel,nrelative,nfliespred)
          relposefuture[:,t,:,:,i] = relposenext
          globalposfuture[:,t,:,:,i] = globalposnext
          if self.flatten:
            zmovement[t-1,:,:,i] = zmovementout_flattened
          else:
            zmovement[t-1,:,i] = zmovementout[selectsample,self.nextframeidx]
          # next frame
          featnext = combine_relative_global(relposenext[selectsample,0,:],globalposnext[selectsample,0,:])
          Xkp_next = mabe.feat2kp(featnext,scales[...,i])
          Xkp_next = Xkp_next[:,:,0,0]
          Xkp[:,:,t,flynum] = Xkp_next
        # end loop over flies

        if self.simplify_in is None:
          for i in range(nfliespred):
            flynum = fliespred[i]
            sensorynext = compute_sensory_wrapper(Xkp[...,[t,],:],flynum,
                                                  theta_main=globalpos[selectsample,[t,],featthetaglobal,i])
            sensory[t,:,i] = sensorynext.T
    
        for i in range(nfliespred):
          if self.simplify_in is None:
            rawinputsnext = combine_inputs(relpose=relpose[selectsample,[t,],:,i],
                                          sensory=sensory[[t,],:,i],dim=1)
          else:
            rawinputsnext = relpose[selectsample,[t,],:,i]
          zinputsnext = self.zscore_input(rawinputsnext)         
          zinputs[t,...,i] = zinputsnext
        # end loop over flies
      # end loop over time points

      # if self.flatten:
      #   if self.flatten_obs:
      #     zinputs_unflattened = np.zeros((zinputs.shape[0],self.dfeat,nfliespred))
      #     for i,v in enumerate(self.flatten_obs_idx.values()):
      #       zinputs_unflattened[:,v[0]:v[1],:] = zinputs[:,i,:self.flatten_dinput_pertype[i],:]
      #     zinputs = zinputs_unflattened
      #   else:
      #     zinputs = zinputs[:,0,...]

      if need_weights:
        return Xkp,zinputs,globalposfuture,relposefuture,attn_weights
      else:
        return Xkp,zinputs,globalposfuture,relposefuture
  
  def get_movement_names_global(self):
    return ['forward','sideways','orientation']

  def get_movement_names(self):
    outnames_global = self.get_movement_names_global()
    if self.simplify_out == 'global':
      outnames = outnames_global
    else:        
      outnames = outnames_global + [mabe.posenames[x] for x in np.nonzero(featrelative)[0]]
    return outnames
  
  def get_outnames(self):
    """
    outnames = self.get_outnames()

    Returns:
        outnames (list of strings): names of each output motion
    """
    outnames = self.get_movement_names()
      
    if len(self.tspred_global) > 1 or self.dct_m is not None:
      ftidx = self.unravel_label_index(np.arange(self.d_output,dtype=int))
      outnames = [f'{outnames[ft[0]]}_{ft[1]}' for ft in ftidx]
      
    return outnames
    
  def parse_label_fields(self,example):
    
    labels_discrete = None
    labels_todiscretize = None
    labels_stacked = None
    
    # get labels_continuous, labels_discrete from example
    if isinstance(example,dict):
      if 'labels' in example:
        labels_continuous = example['labels']
      elif 'continuous' in example:
        labels_continuous = example['continuous'] # prediction
      else:
        raise ValueError('Could not find continuous labels')
      if 'labels_discrete' in example:
        labels_discrete = example['labels_discrete']
      elif 'discrete' in example:
        labels_discrete = example['discrete']
      if 'labels_todiscretize' in example:
        labels_todiscretize = example['labels_todiscretize']
      if 'labels_stacked' in example:
        labels_stacked = example['labels_stacked']      
    else:
      labels_continuous = example
    if self.flatten:
      labels_continuous,labels_discrete = self.unflatten_labels(labels_continuous)
          
    return labels_continuous,labels_discrete,labels_todiscretize,labels_stacked

  def unflatten_labels(self,labels_flattened):
    assert self.flatten_labels
    sz = labels_flattened.shape
    newsz = sz[:-2]+(self.ntimepoints,self.ntokens_per_timepoint,self.flatten_max_doutput)
    if not self.ismasked():
      pad = torch.zeros(sz[:-2]+(1,self.flatten_max_doutput),dtype=labels_flattened.dtype,device=labels_flattened.device)
      labels_flattened = torch.cat((pad,labels_flattened),dim=-2)
    labels_flattened = labels_flattened.reshape(newsz)
    if self.d_output_continuous > 0:
      labels_continuous = labels_flattened[...,-1,:self.d_output_continuous]
    else:
      labels_continuous = None
    if self.discretize:
      labels_discrete = labels_flattened[...,self.flatten_nobs_types:,:self.discretize_nbins]
      if self.continuous:
        labels_discrete = labels_discrete[...,:-1,:]
    else:
      labels_discrete = None
    return labels_continuous,labels_discrete
    
  def apply_flatten_input(self,input):
    
    if type(input) == np.ndarray:
      input = torch.Tensor(input)
    
    if self.flatten_obs == False:
      return input
    
    # input is of size ...,contextl,d_input
    sz = input.shape[:-2]
    contextl = input.shape[-2]
    newinput = torch.zeros(sz+(contextl,self.flatten_nobs_types,self.flatten_max_dinput),dtype=input.dtype)

    for i,v in enumerate(self.flatten_obs_idx.values()):
      newinput[...,i,:self.flatten_dinput_pertype[i]] = input[...,v[0]:v[1]]
    return newinput
    
  def unflatten_input(self,input_flattened):
    assert self.flatten_obs
    sz = input_flattened.shape
    if not self.ismasked():
      pad = torch.zeros(sz[:-2]+(1,self.flatten_dinput),dtype=input_flattened.dtype,device=input_flattened.device)
      input_flattened = torch.cat((input_flattened,pad),dim=-2)      
    resz = sz[:-2]+(self.ntimepoints,self.ntokens_per_timepoint,self.flatten_dinput)
    input_flattened = input_flattened.reshape(resz)
    newsz = sz[:-2]+(self.ntimepoints,self.dfeat)
    newinput = torch.zeros(newsz,dtype=input_flattened.dtype)
    for i,v in enumerate(self.flatten_obs_idx.values()):
      newinput[...,:,v[0]:v[1]] = input_flattened[...,i,self.flatten_input_type_to_range[i,0]:self.flatten_input_type_to_range[i,1]]
    return newinput
  
  def get_full_inputs(self,example=None,idx=None,use_stacked=False):
    if example is None:
      example = self[idx]
    if self.flatten_obs:
      if use_stacked and \
        ('input_stacked' in example and example['input_stacked'] is not None):
        return example['input_stacked']
      else:
        return self.unflatten_input(example['input'])
    else:
      return self.remove_labels_from_input(example['input'])
        
  def get_continuous_discrete_labels(self,example):

    # get labels_continuous, labels_discrete from example
    labels_continuous,labels_discrete,_,_ = self.parse_label_fields(example)      
    return labels_continuous,labels_discrete
        
  def get_continuous_labels(self,example):

    labels_continuous,_ = self.get_continuous_discrete_labels(example)
    return labels_continuous
  
  def get_discrete_labels(self,example):
    _,labels_discrete = self.get_continuous_discrete_labels(example)

    return labels_discrete
  
  def get_full_pred(self,pred,**kwargs):
    return self.get_full_labels(example=pred,ispred=True,**kwargs)
        
  def get_full_labels(self,example=None,idx=None,use_todiscretize=False,sample=False,use_stacked=False,ispred=False,nsamples=0):
    
    if self.discretize and sample:
      return self.sample_full_labels(example=example,idx=idx,nsamples=nsamples)
    
    if example is None:
      example = self[idx]

    # get labels_continuous, labels_discrete from example
    labels_continuous,labels_discrete,labels_todiscretize,labels_stacked = \
      self.parse_label_fields(example)
      
    if self.flatten_labels:
      if use_stacked and labels_stacked is not None:
        labels_continuous,labels_discrete = self.unflatten_labels(labels_stacked)
          
    if self.discretize:
      # should be ... x d_output_discrete x discretize_nbins
      sz = labels_discrete.shape
      newsz = sz[:-2]+(self.d_output,)
      labels = torch.zeros(newsz,dtype=labels_discrete.dtype)
      if self.d_output_continuous > 0:
        labels[...,self.continuous_idx] = labels_continuous
      if use_todiscretize and (labels_todiscretize is not None):
        labels[...,self.discrete_idx] = labels_todiscretize
      else:
        labels[...,self.discrete_idx] = labels_discrete_to_continuous(labels_discrete,
                                                                      torch.tensor(self.discretize_bin_edges))
      return labels
    else:
      return labels_continuous.clone()
    
  def sample_full_labels(self,example=None,idx=None,nsamples=0):
    if example is None:
      example = self[idx]
      
    nsamples1 = nsamples
    if nsamples1 == 0:
      nsamples1 = 1
      
    # get labels_continuous, labels_discrete from example
    labels_continuous,labels_discrete,_,_ = self.parse_label_fields(example)
      
    if not self.discretize:
      return labels_continuous
    
    # should be ... x d_output_continuous
    sz = labels_discrete.shape[:-2]
    dtype = labels_discrete.dtype
    newsz = (nsamples1,)+sz+(self.d_output,)
    labels = torch.zeros(newsz,dtype=dtype)
    if self.continuous:
      labels[...,self.continuous_idx] = labels_continuous
    
    # labels_discrete is ... x nfeat x nbins
    nfeat = labels_discrete.shape[-2]
    nbins = labels_discrete.shape[-1]
    szrest = labels_discrete.shape[:-2]
    if len(szrest) == 0:
      n = 1
    else:
      n = np.prod(szrest)
    nsamples_per_bin = self.discretize_bin_samples.shape[0]
    for i in range(nfeat):
      binnum = weighted_sample(labels_discrete[...,i,:].reshape((n,nbins)),nsamples=nsamples)
      sample = torch.randint(low=0,high=nsamples_per_bin,size=(nsamples,n))
      labelscurr = torch.Tensor(self.discretize_bin_samples[sample,i,binnum].reshape((nsamples,)+szrest))
      labels[...,self.discrete_idx[i]] = labelscurr
      
    if nsamples == 0:
      labels = labels[0]

    return labels
      
def get_batch_idx(example,idx):
  
  if isinstance(example,np.ndarray) or torch.is_tensor(example):
    return example[idx,...]
  
  example1 = {}
  for kw,v in example.items():
    if isinstance(v,np.ndarray) or torch.is_tensor(v):
      example1[kw] = v[idx,...]
    elif isinstance(v,dict):
      example1[kw] = get_batch_idx(v,idx)

  return example1

lossfcn_discrete = torch.nn.CrossEntropyLoss()
lossfcn_continuous = torch.nn.L1Loss()

def causal_criterion(tgt,pred):
  d = tgt.shape[-1]
  err = torch.sum(torch.abs(tgt-pred))/d
  return err

def mixed_causal_criterion(tgt,pred,weight_discrete=.5,extraout=False):
  iscontinuous = tgt['labels'] is not None
  isdiscrete = tgt['labels_discrete'] is not None

  if iscontinuous:
    n = np.prod(tgt['labels'].shape[:-1])
  else:
    n = np.prod(tgt['labels_discrete'].shape[:-2])
  if iscontinuous:
    err_continuous = lossfcn_continuous(pred['continuous'],tgt['labels'].to(device=pred['continuous'].device))*n
  else:
    err_continuous = torch.tensor(0.,dtype=tgt['labels_discrete'].dtype,device=tgt['labels_discrete'].device)
  if isdiscrete:
    pd = pred['discrete']
    newsz = (np.prod(pd.shape[:-1]),pd.shape[-1])
    pd = pd.reshape(newsz)
    td = tgt['labels_discrete'].to(device=pd.device).reshape(newsz)
    err_discrete = lossfcn_discrete(pd,td)*n
  else:
    err_discrete = torch.tensor(0.,dtype=tgt['labels'].dtype,device=tgt['labels'].device)
  err = (1-weight_discrete)*err_continuous + weight_discrete*err_discrete
  if extraout:
    return err,err_discrete,err_continuous
  else:
    return err
  
def dct_consistency(pred):
  return

def prob_causal_criterion(tgt,pred):
  d = tgt.shape[-1]
  err = torch.sum(pred['stateprob']*torch.sum(torch.abs(tgt[...,None]-pred['perstate'])/d,keepdim=False,axis=-2))
  return err

def min_causal_criterion(tgt,pred):
  d = tgt.shape[-1]
  errperstate = torch.sum(torch.abs(tgt[...,None]-pred)/d,keepdim=False,dim=tuple(range(pred.ndim - 1)))
  err = torch.min(errperstate,dim=-1)
  return err
    
def masked_criterion(tgt,pred,mask):
  d = tgt.shape[-1]
  err = torch.sum(torch.abs(tgt[mask,:]-pred[mask,:]))/d
  return err

def mixed_masked_criterion(tgt,pred,mask,device,weight_discrete=.5,extraout=False):
  n = torch.count_nonzero(mask)
  err_continuous = lossfcn_continuous(pred['continuous'][mask,:],tgt['labels'].to(device=device)[mask,:])*n
  err_discrete = lossfcn_discrete(pred['discrete'][mask,...],tgt['labels_discrete'].to(device=device)[mask,...])*n
  err = (1-weight_discrete)*err_continuous + weight_discrete*err_discrete
  if extraout:
    return err,err_discrete,err_continuous
  else:
    return err

######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(torch.nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000,
               ntokens_per_timepoint: int = 1):
    super().__init__()

    # during training, randomly zero some of the inputs with probability p=dropout
    self.dropout = torch.nn.Dropout(p=dropout)

    pe = torch.zeros(1,max_len,d_model)
    position = torch.arange(max_len).unsqueeze(1)
    
    # if many tokens per time point, then have a one-hot encoding of token type
    if ntokens_per_timepoint > 1:
      nwave = d_model-ntokens_per_timepoint
      for i in range(ntokens_per_timepoint):
        pe[0,:,nwave+i] = 2*((position[:,0] % ntokens_per_timepoint)==i).to(float)-1
    else:
      nwave = d_model
      
    # compute sine and cosine waves at different frequencies
    # pe[0,:,i] will have a different value for each word (or whatever)
    # will be sines for even i, cosines for odd i,
    # exponentially decreasing frequencies with i
    div_term = torch.exp(torch.arange(0,nwave,2)*(-math.log(10000.0)/nwave))
    nsinwave = int(np.ceil(nwave/2))
    ncoswave = nwave-nsinwave
    pe[0,:,0:nwave:2] = torch.sin(position * div_term[:nsinwave])
    pe[0,:,1:nwave:2] = torch.cos(position * div_term[:ncoswave])

    # buffers will be saved with model parameters, but are not model parameters
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
      x: Tensor, shape [batch_size, seq_len, embedding_dim]
    """

    # add positional encoding
    x = x + self.pe[:,:x.size(1),:]

    # zero out a randomly selected subset of entries
    return self.dropout(x)

class TransformerBestStateModel(torch.nn.Module):

  def __init__(self, d_input: int, d_output: int,
               d_model: int = 2048, nhead: int = 8, d_hid: int = 512,
               nlayers: int = 12, dropout: float = 0.1, nstates: int = 8):
    super().__init__()
    self.model_type = 'TransformerBestState'

    # frequency-based representation of word position with dropout
    self.pos_encoder = PositionalEncoding(d_model,dropout)

    # create self-attention + feedforward network module
    # d_model: number of input features
    # nhead: number of heads in the multiheadattention models
    # dhid: dimension of the feedforward network model
    # dropout: dropout value
    encoder_layers = torch.nn.TransformerEncoderLayer(d_model,nhead,d_hid,dropout,batch_first=True)

    # stack of nlayers self-attention + feedforward layers
    # nlayers: number of sub-encoder layers in the encoder
    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers,nlayers)

    # encoder and decoder are currently not tied together, but maybe they should be? 
    # fully-connected layer from input size to d_model
    self.encoder = torch.nn.Linear(d_input,d_model)

    # for each hidden state, fully connected layer from model to output size
    # concatenated together, so output is size d_output * nstates
    self.decode = torch.nn.Linear(d_model,nstates*d_output)

    # store hyperparameters
    self.d_model = d_model
    self.nstates = nstates
    self.d_output = d_output

    self.init_weights()

  def init_weights(self) -> None:
    pass

  def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, is_causal: bool = False) -> torch.Tensor:
    """
    Args:
      src: Tensor, shape [batch_size,seq_len,dinput]
      src_mask: Tensor, shape [seq_len,seq_len]
    Returns:
      Tensor of shape [batch_size, seq_len, d_output, nstates]
    """

    # project input into d_model space, multiple by sqrt(d_model) for reasons?
    src = self.encoder(src) * math.sqrt(self.d_model)

    # add in the positional encoding of where in the sentence the words occur
    # it is weird to me that these are added, but I guess it would be almost
    # the same to have these be combined in a single linear layer
    src = self.pos_encoder(src)

    # main transformer layers
    transformer_output = self.transformer_encoder(src,mask=src_mask,is_causal=is_causal)

    # output given each hidden state  
    # batch_size x seq_len x d_output x nstates
    output = self.decode(transformer_output).reshape(src.shape[:-1]+(self.d_output,self.nstates))
      
    return output

  def randpred(self,pred):
    contextl = pred.shape[-3]
    draw = torch.randint(0,pred.shape[-1],contextl)
    return pred[...,np.arange(contextl,dtype=int),:,draw]

class TransformerStateModel(torch.nn.Module):

  def __init__(self, d_input: int, d_output: int,
               d_model: int = 2048, nhead: int = 8, d_hid: int = 512,
               nlayers: int = 12, dropout: float = 0.1, nstates: int = 64,
               minstateprob: float = None):
    super().__init__()
    self.model_type = 'TransformerState'

    # frequency-based representation of word position with dropout
    self.pos_encoder = PositionalEncoding(d_model,dropout)

    # create self-attention + feedforward network module
    # d_model: number of input features
    # nhead: number of heads in the multiheadattention models
    # dhid: dimension of the feedforward network model
    # dropout: dropout value
    encoder_layers = torch.nn.TransformerEncoderLayer(d_model,nhead,d_hid,dropout,batch_first=True)

    # stack of nlayers self-attention + feedforward layers
    # nlayers: number of sub-encoder layers in the encoder
    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers,nlayers)

    # encoder and decoder are currently not tied together, but maybe they should be? 
    # fully-connected layer from input size to d_model
    self.encoder = torch.nn.Linear(d_input,d_model)

    # from output of transformer layers to hidden state probabilities
    self.state = torch.nn.Sequential(
      torch.nn.Linear(d_model,nstates),
      torch.nn.Dropout(dropout),
      torch.nn.Softmax(dim=-1)
    )
    if minstateprob is None:
      minstateprob = .01/nstates
    # for each hidden state, fully connected layer from model to output size
    # concatenated together, so output is size d_output * nstates
    self.decode = torch.nn.Linear(d_model,nstates*d_output)

    # store hyperparameters
    self.d_model = d_model
    self.nstates = nstates
    self.d_output = d_output
    self.minstateprob = minstateprob

    self.init_weights()

  def init_weights(self) -> None:
    pass

  def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, is_causal: bool = False) -> torch.Tensor:
    """
    Args:
      src: Tensor, shape [batch_size,seq_len,dinput]
      src_mask: Tensor, shape [seq_len,seq_len]
    Returns:
      output dict with the following fields:
      stateprob: Tensor of shape [batch_size, seq_len, nstates] indicating the 
      probability of each state
      perstate: Tensor of shape [batch_size, seq_len, d_output, nstates] where
      perstate[t,i,:,j] is the output for time t, example i, and state j. 
    """

    # project input into d_model space, multiple by sqrt(d_model) for reasons?
    src = self.encoder(src) * math.sqrt(self.d_model)

    # add in the positional encoding of where in the sentence the words occur
    # it is weird to me that these are added, but I guess it would be almost
    # the same to have these be combined in a single linear layer
    src = self.pos_encoder(src)

    # main transformer layers
    transformer_output = self.transformer_encoder(src,mask=src_mask,is_causal=is_causal)

    output = {}
    # probability of each of the hidden states
    # batch_size x seq_len x nstates
    output['stateprob'] = self.state(transformer_output)

    # make sure that every state has some probability
    if self.training:
      output['stateprob'] = (output['stateprob']+self.minstateprob)/(1+self.nstates*self.minstateprob)
      
    # output given each hidden state  
    # batch_size x seq_len x d_output x nstates
    output['perstate'] = self.decode(transformer_output).reshape(src.shape[:-1]+(self.d_output,self.nstates))
      
    return output
  
  def maxpred(self,pred):
    state = torch.argmax(pred['stateprob'],axis=-1)
    perstate = pred['perstate'].flatten(end_dim=1)
    out = perstate[torch.arange(perstate.shape[0],dtype=int),:,state.flatten()].reshape(pred['perstate'].shape[:-1])
    return out
  
  def randpred(self,pred):
    state = torch.multinomial(pred['stateprob'].flatten(end_dim=-2),1)
    perstate = pred['perstate'].flatten(end_dim=1)
    out = perstate[torch.arange(perstate.shape[0],dtype=int),:,state.flatten()].reshape(pred['perstate'].shape[:-1])
    return out
  
class myTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
  
  def __init__(self,*args,need_weights=False,**kwargs):
    super().__init__(*args,**kwargs)
    self.need_weights = need_weights
  
  def _sa_block(self, x: torch.Tensor,
                attn_mask: typing.Optional[torch.Tensor], 
                key_padding_mask: typing.Optional[torch.Tensor],
                is_causal: bool = False) -> torch.Tensor:
    x = self.self_attn(x, x, x,
                       attn_mask=attn_mask,
                       key_padding_mask=key_padding_mask,
                       need_weights=self.need_weights,
                       is_causal=is_causal)[0]
    return self.dropout1(x)
  def set_need_weights(self,need_weights):
    self.need_weights = need_weights
  
class TransformerModel(torch.nn.Module):

  def __init__(self, d_input: int, d_output: int,
               d_model: int = 2048, nhead: int = 8, d_hid: int = 512,
               nlayers: int = 12, dropout: float = 0.1,
               ntokens_per_timepoint: int = 1,
               input_idx = None, input_szs = None, embedding_types = None, embedding_params = None,
               d_output_discrete = None, nbins = None,
               ):
    super().__init__()
    self.model_type = 'Transformer'

    self.is_mixed = nbins is not None
    if self.is_mixed:
      self.d_output_continuous = d_output
      self.d_output_discrete = d_output_discrete
      self.nbins = nbins
      d_output = self.d_output_continuous + self.d_output_discrete*self.nbins

    # frequency-based representation of word position with dropout
    self.pos_encoder = PositionalEncoding(d_model,dropout,ntokens_per_timepoint=ntokens_per_timepoint)

    # create self-attention + feedforward network module
    # d_model: number of input features
    # nhead: number of heads in the multiheadattention models
    # dhid: dimension of the feedforward network model
    # dropout: dropout value
    encoder_layers = myTransformerEncoderLayer(d_model,nhead,d_hid,dropout,batch_first=True)

    # stack of nlayers self-attention + feedforward layers
    # nlayers: number of sub-encoder layers in the encoder
    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers,nlayers)

    # encoder and decoder are currently not tied together, but maybe they should be? 
    # fully-connected layer from input size to d_model
    
    if input_idx is not None:
      self.encoder = ObsEmbedding(d_model=d_model,input_idx=input_idx,input_szs=input_szs,
                                  embedding_types=embedding_types,embedding_params=embedding_params)
    else:
      self.encoder = torch.nn.Linear(d_input,d_model)

    # fully-connected layer from d_model to input size
    self.decoder = torch.nn.Linear(d_model,d_output)

    # store hyperparameters
    self.d_model = d_model

    self.init_weights()

  def init_weights(self) -> None:
    pass

  def forward(self, src: torch.Tensor, mask: torch.Tensor = None, is_causal: bool = False) -> torch.Tensor:
    """
    Args:
      src: Tensor, shape [seq_len,batch_size,dinput]
      src_mask: Tensor, shape [seq_len,seq_len]
    Returns:
      output Tensor of shape [seq_len, batch_size, ntoken]
    """

    # project input into d_model space, multiple by sqrt(d_model) for reasons?
    src = self.encoder(src) * math.sqrt(self.d_model)

    # add in the positional encoding of where in the sentence the words occur
    # it is weird to me that these are added, but I guess it would be almost
    # the same to have these be combined in a single linear layer
    src = self.pos_encoder(src)

    # main transformer layers
    output = self.transformer_encoder(src,mask=mask,is_causal=is_causal)

    # project back to d_input space
    output = self.decoder(output)
    
    if self.is_mixed:
      output_continuous = output[...,:self.d_output_continuous]
      output_discrete = output[...,self.d_output_continuous:].reshape(output.shape[:-1]+(self.d_output_discrete,self.nbins))
      output = {'continuous': output_continuous, 'discrete': output_discrete}

    return output
  
  def set_need_weights(self,need_weights):
    for layer in self.transformer_encoder.layers:
      layer.set_need_weights(need_weights)

  def output(self,*args,**kwargs):
    
    output = self.forward(*args,**kwargs)
    if self.is_mixed:
      output['discrete'] = torch.softmax(output['discrete'],dim=-1)
    
    return output
  
class ObsEmbedding(torch.nn.Module):
  def __init__(self, d_model: int, input_idx, input_szs, embedding_types, embedding_params):

    super().__init__()

    assert input_idx is not None
    assert input_szs is not None
    assert embedding_types is not None
    assert embedding_params is not None

    self.input_idx = input_idx
    self.input_szs = input_szs

    self.encoder_dict = torch.nn.ModuleDict()
    for k in input_idx.keys():
      emb = embedding_types.get(k,'fc')
      params = embedding_params.get(k,{})
      szcurr = input_szs[k]
      if emb == 'conv1d_feat':
        if len(szcurr) < 2:
          input_channels = 1
        else:
          input_channels = szcurr[1]
        channels = [input_channels,]+params['channels']
        params = {k1:v for k1,v in params.items() if k1 != 'channels'}
        encodercurr = ResNet1d(channels,d_model,d_input=szcurr[0],no_input_channels=True,single_output=True,transpose=False,**params)
      elif emb == 'fc':
        encodercurr = torch.nn.Linear(szcurr[0],d_model)
      elif emb == 'conv1d_time':
        assert(len(szcurr) == 1)
        input_channels = szcurr[0]
        channels = [input_channels,]+params['channels']
        params = {k1:v for k1,v in params.items() if k1 != 'channels'}
        encodercurr = ResNet1d(channels,d_model,no_input_channels=False,single_output=False,transpose=True,**params)
      elif emb == 'conv2d':
        assert(len(szcurr) <= 2)
        if len(szcurr) > 1:
          input_channels = szcurr[1]
          no_input_channels = False
        else:
          input_channels = 1
          no_input_channels = True
        channels = [input_channels,]+params['channels']
        params = {k1:v for k1,v in params.items() if k1 != 'channels'}
        encodercurr = ResNet2d(channels,d_model,no_input_channels=no_input_channels,d_input=szcurr,single_output=True,transpose=True,**params)
      else:
        # consider adding graph networks
        raise ValueError(f'Unknown embedding type {emb}')
      self.encoder_dict[k] = encodercurr
      
  def forward(self,src):
    src = unpack_input(src,self.input_idx,self.input_szs)
    out = 0.
    for k,v in src.items():
      out += self.encoder_dict[k](v)
    return out
  
class Conv1d_asym(torch.nn.Conv1d):
  def __init__(self, *args, padding='same', **kwargs):
    self.padding_off = [0,0]
    padding_sym = padding
    if (type(padding) == tuple) or (type(padding) == list):
      padding_sym = int(np.max(padding))
      for j in range(2):
        self.padding_off[j] = padding_sym - padding[j]
    super().__init__(*args,padding=padding_sym,**kwargs)

  def asymmetric_crop(self,out):
    out = out[...,self.padding_off[0]:out.shape[-1]-self.padding_off[1]]
    return out

  def forward(self,x,*args,**kwargs):
    out = super().forward(x,*args,**kwargs)
    out = self.asymmetric_crop(out)
    return out    
  
class Conv2d_asym(torch.nn.Conv2d):
  def __init__(self, *args, padding='same', **kwargs):
    self.padding_off = [[0,0],[0,0]]
    padding_sym = padding
    if (type(padding) == tuple) or (type(padding) == list):
      padding_sym = list(padding_sym)
      for i in range(2):
        if type(padding[i]) != int:
          padding_sym[i] = int(np.max(padding[i]))
          for j in range(2):
            self.padding_off[i][j] = padding_sym[i] - padding[i][j]
      padding_sym = tuple(padding_sym)
    super().__init__(*args,padding=padding_sym,**kwargs)

  def asymmetric_crop(self,out):
    out = out[...,self.padding_off[0][0]:out.shape[-2]-self.padding_off[0][1],self.padding_off[1][0]:out.shape[-1]-self.padding_off[1][1]]
    return out

  def forward(self,x,*args,**kwargs):
    out = super().forward(x,*args,**kwargs)
    out = self.asymmetric_crop(out)
    return out
  
class ResidualBlock1d(torch.nn.Module):
  
  def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, dilation = 1, padding = 'same', padding_mode='zeros'):
    super().__init__()
    
    self.padding = padding
    self.kernel_size = kernel_size
    self.stride = stride
    self.dilation = dilation
    self.conv1 = torch.nn.Sequential(
      Conv1d_asym(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, padding_mode=padding_mode, dilation=self.dilation),
      torch.nn.BatchNorm1d(out_channels),
      torch.nn.ReLU()
    )
    self.conv2 = torch.nn.Sequential(
      Conv1d_asym(out_channels, out_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding, padding_mode=padding_mode, dilation=self.dilation),
      torch.nn.BatchNorm1d(out_channels)
    )
    if (in_channels != out_channels) or (self.stride > 1):
      self.downsample = torch.nn.Sequential(
        torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
        torch.nn.BatchNorm1d(out_channels)
      )
    else:
      self.downsample = None
    self.relu = torch.nn.ReLU()
    self.out_channels = out_channels
      
  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)
    return out
  
  def compute_output_shape(self,din):
    if type(self.padding) == str:
      if self.padding == 'same':
        return (self.out_channels,din)
      elif self.padding == 'valid':
        padding = 0
      else:
        raise ValueError(f'Unknown padding type {self.padding}')
    if len(self.padding) == 1:
      padding = 2*self.padding
    else:
      padding = np.sum(self.padding)
    dout1 = np.floor((din + padding - self.dilation*(self.kernel_size-1)-1)/self.stride+1)
    dout = (dout1 + padding - self.dilation*(self.kernel_size-1)-1)+1
    sz = (self.out_channels,int(dout))
    return sz
  
class ResidualBlock2d(torch.nn.Module):
  
  def __init__(self, in_channels, out_channels, kernel_size = (3,3), stride = (1,1), dilation = (1,1), padding = 'same', padding_mode = 'zeros'):
    super().__init__()

    self.padding = padding          
    self.kernel_size = kernel_size
    self.stride = stride
    self.dilation = dilation
    self.conv1 = torch.nn.Sequential(
      Conv2d_asym(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, padding_mode=padding_mode, dilation=self.dilation),
      torch.nn.BatchNorm2d(out_channels),
      torch.nn.ReLU()
    )
    self.conv2 = torch.nn.Sequential(
      Conv2d_asym(out_channels, out_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding, padding_mode=padding_mode, dilation=self.dilation),
      torch.nn.BatchNorm2d(out_channels)
    )
    if (in_channels != out_channels) or (np.any(np.array(self.stride) > 1)):
      self.downsample = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
        torch.nn.BatchNorm2d(out_channels)
      )
    else:
      self.downsample = None
    self.relu = torch.nn.ReLU()
    self.out_channels = out_channels
    
  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)
    return out
  
  def compute_output_shape(self,din):
    if type(self.padding) == str:
      if self.padding == 'same':
        return (self.out_channels,)+din
      elif self.padding == 'valid':
        padding = (0,0)
      else:
        raise ValueError(f'Unknown padding type {self.padding}')

    if len(self.padding) == 1:
      padding = [self.padding,self.padding]
    else:
      padding = self.padding
    padding = np.array(padding)
    paddingsum = np.zeros(2,dtype=int)
    for i in range(2):
      if len(padding[i]) == 1:
        paddingsum[i] = 2*padding[i]
      else:
        paddingsum[i] = int(np.sum(padding[i]))
    dout1 = np.floor((np.array(din) + paddingsum - np.array(self.dilation)*(np.array(self.kernel_size)-1)-1)/np.array(self.stride)+1).astype(int)
    dout = ((dout1 + paddingsum - np.array(self.dilation)*(np.array(self.kernel_size)-1)-1)+1).astype(int)
    sz = (self.out_channels,) + tuple(dout)
    return sz
  
class ResNet1d(torch.nn.Module):
  def __init__(self,channels,d_output,d_input=None,no_input_channels=False,single_output=False,transpose=False,**kwargs):
    super().__init__()
    self.channels = channels
    self.d_output = d_output
    self.d_input = d_input
    self.no_input_channels = no_input_channels
    self.transpose = transpose
    self.single_output = single_output
    
    if no_input_channels:
      assert channels[0] == 1
    
    nblocks = len(channels)-1
    self.layers = torch.nn.ModuleList()
    sz = (channels[0],d_input)
    for i in range(nblocks):
      self.layers.append(ResidualBlock1d(channels[i],channels[i+1],**kwargs))
      if d_input is not None:
        sz = self.layers[-1].compute_output_shape(sz[-1])
    if single_output:
      if d_input is None:
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(channels[-1],d_output)
      else:
        self.avg_pool = None
        self.fc = torch.nn.Linear(int(np.prod(sz)),d_output)
    else:
      self.avg_pool = None
      self.fc = torch.nn.Conv1d(channels[-1],d_output,1)
      
  def forward(self,x):

    if self.transpose and not self.no_input_channels:
      x = x.transpose(-1,-2)

    if self.no_input_channels:
      dim = -1
    else:
      dim = -2
    
    sz0 = x.shape
    d_input = sz0[-1]

    if self.single_output and (self.d_input is not None):
      assert d_input == self.d_input
    
    sz = (int(np.prod(sz0[:dim])),self.channels[0],d_input)
    x = x.reshape(sz)
    
    for layer in self.layers:
      x = layer(x)
    if self.avg_pool is not None:
      x = self.avg_pool(x)
    if self.single_output:
      x = torch.flatten(x,1)
    x = self.fc(x)
    
    if self.single_output:
      dimout = -1
    else:
      dimout = -2
    x = x.reshape(sz0[:dim]+x.shape[dimout:])
    
    if self.transpose and not self.single_output:
      x = x.transpose(-1,-2)

    return x
  
class ResNet2d(torch.nn.Module):
  def __init__(self,channels,d_output,d_input=None,no_input_channels=False,single_output=False,transpose=False,**kwargs):
    super().__init__()
    self.channels = channels
    self.d_output = d_output
    self.d_input = d_input
    self.no_input_channels = no_input_channels
    self.transpose = transpose
    
    if no_input_channels:
      assert channels[0] == 1
    
    nblocks = len(channels)-1
    self.layers = torch.nn.ModuleList()
    is_d_input = [False,False]
    if d_input is not None:
      if type(d_input) == int:
        d_input (0,d_input)
      elif len(d_input) < 2:
        d_input = (0,)*(2-len(d_input))+d_input
      is_d_input = [d != 0 for d in d_input]
      sz = (channels[0],) + d_input
    for i in range(nblocks):
      self.layers.append(ResidualBlock2d(channels[i],channels[i+1],**kwargs))
      if d_input is not None:
        sz = self.layers[-1].compute_output_shape(sz[1:])
    self.collapse_dim = []
    if single_output:
      if d_input is None:
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(channels[-1],d_output)
        self.collapse_dim = [-2,-1]
      else:
        self.avg_pool = None
        k = [1,1]
        for i in range(2):
          if is_d_input[i]:
            k[i] = sz[i+1]
            self.collapse_dim.append(-2+i)
        self.fc = torch.nn.Conv2d(channels[-1],d_output,k,padding='valid')
    else:
      self.avg_pool = None
      self.fc = torch.nn.Conv2d(channels[-1],d_output,1)
      
  def forward(self,x):

    if self.transpose and not self.no_input_channels:
      x = torch.movedim(x,-1,-3)

    if self.no_input_channels:
      dim = -2
    else:
      dim = -3
    
    sz0 = x.shape
    d_input = sz0[-2:]
    
    sz = (int(np.prod(sz0[:dim])),self.channels[0])+d_input
    x = x.reshape(sz)
    
    for layer in self.layers:
      x = layer(x)
    if self.avg_pool is not None:
      x = self.avg_pool(x)
      x = torch.flatten(x,1)
      dimout = -1
    else:
      dimout = -3
    x = self.fc(x)
    x = x.reshape(sz0[:dim]+x.shape[dimout:])
    dim_channel = len(sz0[:dim])
    x = torch.squeeze(x,self.collapse_dim)
    
    if self.transpose:
      x = torch.movedim(x,dim_channel,-1)

    return x
    
class DictSum(torch.nn.Module):
  def __init__(self,moduledict):
    super().__init__()
    self.moduledict = moduledict
  def forward(self,x):
    out = 0.
    for k,v in x.items():
      out += self.moduledict[k](v)
    return out

# deprecated, here for backward compatibility
class TransformerMixedModel(TransformerModel):
  
  def __init__(self, d_input: int, d_output_continuous: int = 0,
               d_output_discrete: int = 0, nbins: int = 0,
               **kwargs):
    self.d_output_continuous = d_output_continuous
    self.d_output_discrete = d_output_discrete
    self.nbins = nbins
    d_output = d_output_continuous + d_output_discrete*nbins
    assert d_output > 0
    super().__init__(d_input,d_output,**kwargs)
    
  def forward(self, src: torch.Tensor, mask: torch.Tensor = None, is_causal: bool = False) -> dict:
    output_all = super().forward(src,mask=mask,is_causal=is_causal)
    output_continuous = output_all[...,:self.d_output_continuous]
    output_discrete = output_all[...,self.d_output_continuous:].reshape(output_all.shape[:-1]+(self.d_output_discrete,self.nbins))
    return {'continuous': output_continuous, 'discrete': output_discrete}
  
  def output(self,*args,**kwargs):
    output = self.forward(*args,**kwargs)
    output['discrete'] = torch.softmax(output['discrete'],dim=-1)
    return output
  
def generate_square_full_mask(sz: int) -> torch.Tensor:
  """
  Generates an zero matrix. All words allowed.
  """
  return torch.zeros(sz,sz)

def get_output_and_attention_weights(model, inputs, mask=None, is_causal=False):  

  # set need_weights to True for this function call
  model.set_need_weights(True)

  # where attention weights will be stored, one list element per layer
  activation = [None,]*model.transformer_encoder.num_layers
  def get_activation(layer_num):
    # the hook signature
    def hook(model, inputs, output):
      # attention weights are the second output
      activation[layer_num] = output[1]
    return hook

  # register the hooks
  hooks = [None,]*model.transformer_encoder.num_layers
  for i,layer in enumerate(model.transformer_encoder.layers):
    hooks[i] = layer.self_attn.register_forward_hook(get_activation(i))

  # call the model
  with torch.no_grad():
    output = model.output(inputs, mask=mask, is_causal=is_causal)
  
  # remove the hooks    
  for hook in hooks:
    hook.remove()

  # return need_weights to False
  model.set_need_weights(False)    

  return output,activation
  
def compute_attention_weight_rollout(w0):
  # w0 is nlayers x T x T x ...
  w = np.zeros(w0.shape,dtype=w0.dtype)
  wcurr = np.ones(list(w0.shape)[1:],dtype=w0.dtype)
  # I = np.eye(w0.shape[1],dtype=w0.dtype)
  # sz = np.array(w0.shape[1:])
  # sz[2:] = 1
  # I = I.reshape(sz)

  for i in range(w0.shape[0]):
    wcurr = wcurr * (w0[i,...])
    z = np.maximum(np.sum(wcurr,axis=0,keepdims=True),np.finfo(w0.dtype).eps)
    wcurr = wcurr / z
    w[i,...]  = wcurr
  return w

def save_model(savefile,model,lr_optimizer=None,scheduler=None,loss=None,config=None):
  tosave = {'model':model.state_dict()}
  if lr_optimizer is not None:
    tosave['lr_optimizer'] = lr_optimizer.state_dict()
  if scheduler is not None:
    tosave['scheduler'] = scheduler.state_dict()
  if loss is not None:
    tosave['loss'] = loss
  if config is not None:
    tosave['config'] = config
  tosave['SENSORY_PARAMS'] = SENSORY_PARAMS
  torch.save(tosave,savefile)
  return

def load_model(loadfile,model,device,lr_optimizer=None,scheduler=None,config=None):
  print(f'Loading model from file {loadfile}...')
  state = torch.load(loadfile, map_location=device)
  if model is not None:
    model.load_state_dict(state['model'])
  if lr_optimizer is not None and ('lr_optimizer' in state):
    lr_optimizer.load_state_dict(state['lr_optimizer'])
  if scheduler is not None and ('scheduler' in state):
    scheduler.load_state_dict(state['scheduler'])
  if config is not None:
    load_config_from_model_file(config=config,state=state)
      
  loss = {}
  if 'loss' in state:
    if isinstance(loss,dict):
      loss = state['loss']
    else:
      # backwards compatible
      loss['train'] = loss
      if 'val_loss' in state:
        loss['val'] = state['val_loss']
  return loss

def load_config_from_model_file(loadmodelfile=None,config=None,state=None,no_overwrite=[]):
  if state is None:
    assert loadmodelfile is not None
    print(f'Loading config from file {loadmodelfile}...')
    state = torch.load(loadmodelfile)
  if config is not None and 'config' in state:
    overwrite_config(config,state['config'],no_overwrite=no_overwrite)
  else:
    print(f'config not stored in model file {loadmodelfile}')
  if 'SENSORY_PARAMS' in state:
    for k,v in state['SENSORY_PARAMS'].items():
      SENSORY_PARAMS[k] = v
  else:
    print(f'SENSORY_PARAMS not stored in model file {loadmodelfile}')
  return

def update_loss_nepochs(loss_epoch,nepochs):
  for k,v in loss_epoch.items():
    if v.numel() < nepochs:
      n = torch.zeros(nepochs-v.numel(),dtype=v.dtype,device=v.device)+torch.nan
      loss_epoch[k] = torch.cat((v,n))      
  return


def debug_plot_batch_state(stateprob,nsamplesplot=3,
                          h=None,ax=None,fig=None):
  batch_size = stateprob.shape[0]

  samplesplot = np.round(np.linspace(0,batch_size-1,nsamplesplot)).astype(int)

  if ax is None:
    fig,ax = plt.subplots(nsamplesplot,1)
  if h is None:
    h = [None,]*nsamplesplot

  for i in range(nsamplesplot):
    iplot = samplesplot[i]
    if h[i] is None:
      h[i] = ax[i].imshow(stateprob[iplot,:,:].T,vmin=0.,vmax=1.)
    else:
      h[i].set_data(stateprob[iplot,:,:].T)
    ax[i].axis('auto')
  
  fig.tight_layout(h_pad=0)
  return h,ax,fig


def select_featidx_plot(train_dataset,ntspred_plot,ntsplot_global=None,ntsplot_relative=None):

  if ntsplot_global is None: 
    ntsplot_global = np.minimum(train_dataset.ntspred_global,ntspred_plot)
  if ntsplot_relative is None:
    ntsplot_relative = np.minimum(train_dataset.ntspred_relative,ntspred_plot)
  
  if ntsplot_global == 0:
    tidxplot_global = None
  elif ntsplot_global == 1:
    tidxplot_global = np.zeros((nglobal,1),dtype=int)
  elif ntsplot_global == train_dataset.ntspred_global:
    tidxplot_global = np.tile(np.arange(ntsplot_global,dtype=int)[None,:],(nglobal,1))
  else:
    # choose 0 + a variety of different timepoints for each global feature so that a variety of timepoints are selected
    tidxplot_global = np.concatenate((np.zeros((nglobal,1),dtype=int),
                                      np.round(np.linspace(1,train_dataset.ntspred_global-1,(ntsplot_global-1)*nglobal)).astype(int).reshape(-1,nglobal).T),axis=-1)
  if ntsplot_relative == 0:
    tsplot_relative = None
  elif ntsplot_relative == 1:
    tsplot_relative = np.ones((nrelative,1),dtype=int)
  elif ntsplot_relative == train_dataset.ntspred_relative:
    tsplot_relative = np.tile(np.arange(ntsplot_relative,dtype=int)[None,:]+1,(nrelative,1))
  else:
    # choose 0 + a variety of different timepoints for each feature so that a variety of timepoints are selected
    tsplot_relative = np.concatenate((np.zeros((nrelative,1),dtype=int),
                                      np.round(np.linspace(1,train_dataset.ntspred_relative-1,(ntsplot_relative-1)*nrelative)).astype(int).reshape(-1,nrelative).T),axis=-1)
  ftidx = []
  for fi,f in enumerate(featglobal):
    for ti in range(ntsplot_global):
      ftidx.append((f,train_dataset.tspred_global[tidxplot_global[fi,ti]]))
  for fi,f in enumerate(np.nonzero(featrelative)[0]):
    for ti in range(ntsplot_relative):
      ftidx.append((f,tsplot_relative[fi,ti]))
  featidxplot = train_dataset.ravel_label_index(ftidx)
  return featidxplot

def debug_plot_batch_traj(example,train_dataset,criterion=None,config=None,
                          pred=None,data=None,nsamplesplot=3,
                          true_discrete_mode='to_discretize',
                          pred_discrete_mode='sample',
                          h=None,ax=None,fig=None,label_true='True',label_pred='Pred',
                          ntspred_plot=2):
  batch_size = example['input'].shape[0]
  contextl = train_dataset.ntimepoints
    
  true_color = [0,0,0]
  true_color_y = [.5,.5,.5]
  pred_cmap = lambda x: plt.get_cmap("tab10")(x%10)
  
  if train_dataset.ismasked():
    mask = example['mask']
  else:
    mask = None
  
  if ax is None:
    fig,ax = plt.subplots(1,nsamplesplot,squeeze=False)
    ax = ax[0,:]

  if true_discrete_mode == 'to_discretize':
    true_args = {'use_todiscretize': True}
  elif true_discrete_mode == 'sample':
    true_args = {'sample': True}
  else:
    true_args = {}
    
  if pred_discrete_mode == 'sample':
    pred_args = {'sample': True}
  else:
    pred_args = {}

  featidxplot = select_featidx_plot(train_dataset,ntspred_plot)
  samplesplot = np.round(np.linspace(0,batch_size-1,nsamplesplot)).astype(int)
  for i in range(nsamplesplot):
    iplot = samplesplot[i]
    examplecurr = get_batch_idx(example,iplot)
    zmovement_continuous_true,zmovement_discrete_true = train_dataset.get_continuous_discrete_labels(examplecurr)
    ntimepoints = train_dataset.ntimepoints
    
    #zmovement_true = train_dataset.get_full_labels(examplecurr,**true_args)
    #zmovement_true = example['labels'][iplot,...].numpy()
    #err_movement = None
    err_total = None
    if mask is not None:
      maskcurr = np.zeros(mask.shape[-1]+1,dtype=bool)
      maskcurr[:-1] = mask[iplot,...].numpy()
      maskidx = np.nonzero(maskcurr)[0]
    if pred is not None:
      predcurr = get_batch_idx(pred,iplot)
      zmovement_continuous_pred,zmovement_discrete_pred = train_dataset.get_continuous_discrete_labels(predcurr)
      #zmovement_pred = train_dataset.get_full_labels(predcurr,**pred_args)
      # if mask is not None:
      #   nmask = np.count_nonzero(maskcurr)
      # else:
      #   nmask = np.prod(zmovement_continuous_pred.shape[:-1])
      if criterion is not None:
        err_total,err_discrete,err_continuous = criterion_wrapper(examplecurr,predcurr,criterion,train_dataset,config)
      zmovement_discrete_pred = torch.softmax(zmovement_discrete_pred,dim=-1)
      # err_movement = torch.abs(zmovement_true[maskidx,:]-zmovement_pred[maskidx,:])/nmask
      # err_total = torch.sum(err_movement).item()/d

    elif data is not None:
      #t0 = example['metadata']['t0'][iplot].item()
      #flynum = example['metadata']['flynum'][iplot].item()
      zmovement_pred = None
    else:
      #Xcenter_pred = None
      #Xtheta_pred = None
      zmovement_pred = None
    
    mult = 6.
    d = len(featidxplot)
    outnames = train_dataset.get_outnames()
    contextl = train_dataset.ntimepoints

    ax[i].cla()
    
    for featii,feati in enumerate(featidxplot):
      featidx = np.nonzero(train_dataset.discrete_idx == feati)[0]
      if len(featidx) == 0:
        continue
      featidx = featidx[0]
      im = np.ones((train_dataset.discretize_nbins,contextl,3))
      ztrue = zmovement_discrete_true[:,featidx,:].cpu().T
      ztrue = ztrue - torch.min(ztrue)
      ztrue = ztrue / torch.max(ztrue)
      im[:,:,0] = 1.-ztrue
      if pred is not None:
        zpred = zmovement_discrete_pred[:,featidx,:].detach().cpu().T
        zpred = zpred - torch.min(zpred)
        zpred = zpred / torch.max(zpred)
        im[:,:,1] = 1.-zpred
      ax[i].imshow(im,extent=(0,contextl,(featii-.5)*mult,(featii+.5)*mult),origin='lower',aspect='auto')
      
    for featii,feati in enumerate(featidxplot):
      featidx = np.nonzero(train_dataset.continuous_idx == feati)[0]
      if len(featidx) == 0:
        continue
      featidx = featidx[0]
      ax[i].plot([0,contextl],[mult*featii,]*2,':',color=[.5,.5,.5])
      ax[i].plot(mult*featii + zmovement_continuous_true[:,featidx],'-',color=true_color,label=f'{outnames[feati]}, true')
      if mask is not None:
        ax[i].plot(maskidx,mult*featii + zmovement_continuous_true[maskcurr[:-1],featidx],'o',color=true_color,label=f'{outnames[feati]}, true')

      labelcurr = outnames[feati]
      if pred is not None:
        h = ax[i].plot(mult*featii + zmovement_continuous_pred[:,featidx],'--',label=f'{outnames[feati]}, pred',color=pred_cmap(featii))
        if mask is not None:
          ax[i].plot(maskidx,mult*featii + zmovement_continuous_pred[maskcurr[:-1],featidx],'o',color=pred_cmap(featii),label=f'{outnames[feati]}, pred')
        
    for featii,feati in enumerate(featidxplot):
      labelcurr = outnames[feati]
      ax[i].text(0,mult*(featii+.5),labelcurr,horizontalalignment='left',verticalalignment='top')

    if (err_total is not None):
      if train_dataset.discretize:
        ax[i].set_title(f'Err: {err_total.item(): .2f}, disc: {err_discrete.item(): .2f}, cont: {err_continuous.item(): .2f}')
      else:
        ax[i].set_title(f'Err: {err_total.item(): .2f}')
    ax[i].set_xlabel('Frame')
    ax[i].set_ylabel('Movement')
    ax[i].set_ylim([-mult,mult*d])

  #fig.tight_layout()

  return ax,fig

def debug_plot_pose_prob(example,train_dataset,predcpu,tplot,fig=None,ax=None,h=None,minalpha=.25):
  batch_size = predcpu['stateprob'].shape[0]
  contextl = predcpu['stateprob'].shape[1]
  nstates = predcpu['stateprob'].shape[2]
  if ax is None:
    fig,ax = plt.subplots(1,1)
    
    
  Xkp_true = train_dataset.get_Xkp(example['input'][0,...].numpy(),
                                   example['init'].numpy(),
                                   example['labels'][:tplot+1,...].numpy(),
                                   example['scale'].numpy())
  Xkp_true = Xkp_true[...,0]
  
  order = torch.argsort(predcpu['stateprob'][0,tplot,:])
  rank = torch.argsort(order)
  labels = example['labels'][:tplot,:]
  state_cmap = lambda x: plt.get_cmap("tab10")(rank[x]%10)
  
  if h is None:
    h = {'kpt_true': None, 'kpt_state': [None,]*nstates, 
         'edge_true': None, 'edge_state': [None,]*nstates}
  h['kpt_true'],h['edge_true'],_,_,_ = mabe.plot_fly(Xkp_true[:,:,-1],
                                                     skel_lw=2,color=[0,0,0],
                                                     ax=ax,hkpt=h['kpt_true'],hedge=h['edge_true'])
  for i in range(nstates):

    labelspred = torch.cat((labels,predcpu['perstate'][0,[tplot,],:,i]),dim=0)
  
    Xkp_pred = train_dataset.get_Xkp(example['input'][0,...].numpy(),
                                     example['init'].numpy(),
                                     labelspred,
                                     example['scale'].numpy())
    Xkp_pred = Xkp_pred[...,0]
    p = predcpu['stateprob'][0,tplot,i].item()
    alpha = minalpha + p*(1-minalpha)
    color = state_cmap(i)
    h['kpt_state'][i],h['edge_state'][i],_,_,_ = mabe.plot_fly(Xkp_pred[:,:,-1],
                                                               skel_lw=2,color=color,
                                                               ax=ax,hkpt=h['kpt_state'][i],
                                                               hedge=h['edge_state'][i])
    h['edge_state'][i].set_alpha(alpha)
    h['kpt_state'][i].set_alpha(alpha)
    
  return h,ax,fig

def debug_plot_batch_pose(example,train_dataset,pred=None,data=None,
                          true_discrete_mode='to_discretize',
                          pred_discrete_mode='sample',
                          ntsplot=5,nsamplesplot=3,h=None,ax=None,fig=None,
                          tsplot=None):
 
  
  batch_size = example['input'].shape[0]
  contextl = train_dataset.ntimepoints
  nsamplesplot = np.minimum(nsamplesplot,batch_size)
      
  if tsplot is None:
    tsplot = np.round(np.linspace(0,contextl,ntsplot)).astype(int)
  else:
    ntsplot = len(tsplot)
  samplesplot = np.round(np.linspace(0,batch_size-1,nsamplesplot)).astype(int)

  if ax is None:
    fig,ax = plt.subplots(nsamplesplot,ntsplot,squeeze=False)

  if h is None:
    h = {'kpt0': [], 'kpt1': [], 'edge0': [], 'edge1': []}
  
  if true_discrete_mode == 'to_discretize':
    true_args = {'use_todiscretize': True}
  elif true_discrete_mode == 'sample':
    true_args = {'sample': True}
  else:
    true_args = {}
    
  if pred_discrete_mode == 'sample':
    pred_args = {'sample': True}
  else:
    pred_args = {}
    
  for i in range(nsamplesplot):
    iplot = samplesplot[i]
    examplecurr = get_batch_idx(example,iplot)
    Xkp_true = train_dataset.get_Xkp(examplecurr,**true_args)
    #['input'][iplot,0,...].numpy(),
    #                            example['init'][iplot,...].numpy(),
    #                            example['labels'][iplot,...].numpy(),
    #                            example['scale'][iplot,...].numpy())
    Xkp_true = Xkp_true[...,0]
    t0 = example['metadata']['t0'][iplot].item()
    flynum = example['metadata']['flynum'][iplot].item()
    if pred is not None:
      predcurr = get_batch_idx(pred,iplot)
      Xkp_pred = train_dataset.get_Xkp(examplecurr,pred=predcurr,**pred_args)
      Xkp_pred = Xkp_pred[...,0]
    elif data is not None:
      Xkp_pred = data['X'][:,:,t0:t0+contextl+1,flynum]
    else:
      Xkp_pred = None
    for key in h.keys():
      if len(h[key]) <= i:
        h[key].append([None,]*ntsplot)
        
    minxy = np.nanmin(np.nanmin(Xkp_true[:,:,tsplot],axis=0),axis=-1)
    maxxy = np.nanmax(np.nanmax(Xkp_true[:,:,tsplot],axis=0),axis=-1)
    if Xkp_pred is not None:
      minxy_pred = np.nanmin(np.nanmin(Xkp_pred[:,:,tsplot],axis=0),axis=-1)
      maxxy_pred = np.nanmax(np.nanmax(Xkp_pred[:,:,tsplot],axis=0),axis=-1)
      minxy = np.minimum(minxy,minxy_pred)
      maxxy = np.maximum(maxxy,maxxy_pred)    
    for j in range(ntsplot):
      tplot = tsplot[j]
      if j == 0:
        ax[i,j].set_title(f'fly: {flynum} t0: {t0}')
      else:
        ax[i,j].set_title(f't = {tplot}')
        
      h['kpt0'][i][j],h['edge0'][i][j],_,_,_ = mabe.plot_fly(Xkp_true[:,:,tplot],
                                                             skel_lw=2,color=[0,0,0],
                                                             ax=ax[i,j],hkpt=h['kpt0'][i][j],hedge=h['edge0'][i][j])
      if Xkp_pred is not None:
        h['kpt1'][i][j],h['edge1'][i][j],_,_,_ = mabe.plot_fly(Xkp_pred[:,:,tplot],
                                                              skel_lw=1,color=[0,1,1],
                                                              ax=ax[i,j],hkpt=h['kpt1'][i][j],hedge=h['edge1'][i][j])      
      ax[i,j].set_aspect('equal')
      # minxy = np.nanmin(Xkp_true[:,:,tplot],axis=0)
      # maxxy = np.nanmax(Xkp_true[:,:,tplot],axis=0)
      # if Xkp_pred is not None:
      #   minxy_pred = np.nanmin(Xkp_pred[:,:,tplot],axis=0)
      #   maxxy_pred = np.nanmax(Xkp_pred[:,:,tplot],axis=0)
      #   minxy = np.minimum(minxy,minxy_pred)
      #   maxxy = np.maximum(maxxy,maxxy_pred)
      ax[i,j].set_xlim([minxy[0],maxxy[0]])
      ax[i,j].set_ylim([minxy[1],maxxy[1]])

  return h,ax,fig

def debug_plot_sample_inputs(dataset,example,nplot=3):

  nplot = np.minimum(nplot,example['input'].shape[0])

  fig,ax = plt.subplots(nplot,2,squeeze=False)
  idx = get_sensory_feature_idx(dataset.simplify_in)
  labels = dataset.get_full_labels(example=example,use_todiscretize=True)
  nlabels = labels.shape[-1]
  inputs = dataset.get_full_inputs(example=example,use_stacked=False)
  
  inputidxstart = [x[0] - .5 for x in idx.values()]
  inputidxtype = list(idx.keys())
    
  for iplot in range(nplot):
    ax[iplot,0].cla()
    ax[iplot,1].cla()
    ax[iplot,0].imshow(inputs[iplot,...],vmin=-3,vmax=3,cmap='coolwarm',aspect='auto')
    ax[iplot,0].set_title(f'Input {iplot}')
    #ax[iplot,0].set_xticks(inputidxstart)
    for j in range(len(inputidxtype)):
      ax[iplot,0].plot([inputidxstart[j],]*2,[-.5,inputs.shape[1]-.5],'k-')
      ax[iplot,0].text(inputidxstart[j],inputs.shape[1]-1,inputidxtype[j],horizontalalignment='left')
    lastidx = list(idx.values())[-1][1]
    ax[iplot,0].plot([lastidx-.5,]*2,[-.5,inputs.shape[1]-.5],'k-')

    #ax[iplot,0].set_xticklabels(inputidxtype)
    ax[iplot,1].imshow(labels[iplot,...],vmin=-3,vmax=3,cmap='coolwarm',aspect='auto')
    ax[iplot,1].set_title(f'Labels {iplot}')
  return fig,ax
  
def stack_batch_list(allx,n=None):
  if len(allx) == 0:
    return []
  xv = torch.cat(allx[:n],dim=0)
  nan = torch.zeros((xv.shape[0],1)+xv.shape[2:],dtype=xv.dtype)
  nan[:] = torch.nan
  xv = torch.cat((xv,nan),dim=1)
  xv = xv.flatten(0,1)
  return xv
  
def stackhelper(all_pred,all_labels,all_mask,all_pred_discrete,all_labels_discrete,nplot):

  predv = torch.stack(all_pred[:nplot],dim=0)
  if len(all_mask) > 0:
    maskv = torch.stack(all_mask[:nplot],dim=0)
  else:
    maskv = None
  labelsv = torch.stack(all_labels[:nplot],dim=0)
  s = list(predv.shape)
  s[2] = 1
  nan = torch.zeros(s,dtype=predv.dtype)
  nan[:] = torch.nan
  predv = torch.cat((predv,nan),dim=2)
  predv = predv.flatten(0,2)
  labelsv = torch.cat((labelsv,nan),dim=2)
  labelsv = labelsv.flatten(0,2)
  if maskv is not None:
    maskv = torch.cat((maskv,torch.zeros(s[:-1],dtype=bool)),dim=2)
    maskv = maskv.flatten()
  if len(all_pred_discrete) > 0:
    pred_discretev = torch.stack(all_pred_discrete[:nplot],dim=0)
    s = list(pred_discretev.shape)
    s[2] = 1
    nan = torch.zeros(s,dtype=pred_discretev.dtype)
    nan[:] = torch.nan
    pred_discretev = torch.cat((pred_discretev,nan),dim=2)
    pred_discretev = pred_discretev.flatten(0,2)
  else:
    pred_discretev = None
  if len(all_labels_discrete) > 0:
    pred_discretev = torch.stack(all_labels_discrete[:nplot],dim=0)
    s = list(pred_discretev.shape)
    s[2] = 1
    nan = torch.zeros(s,dtype=pred_discretev.dtype)
    nan[:] = torch.nan
    pred_discretev = torch.cat((pred_discretev,nan),dim=2)
    pred_discretev = pred_discretev.flatten(0,2)
  else:
    pred_discretev = None
  
  return predv,labelsv,maskv,pred_discretev
  
def debug_plot_predictions_vs_labels(predv,labelsv,pred_discretev=None,labels_discretev=None,outnames=None,maskidx=None,ax=None,prctile_lim=.1,naxc=1,featidxplot=None,dataset=None):
  
  d_output = predv.shape[-1]
  
  if featidxplot is not None:
    predv = predv[:,featidxplot]
    labelsv = labelsv[:,featidxplot]
    outnames = [outnames[i] for i in featidxplot]
    if labels_discretev is not None and len(labels_discretev) > 0:
      tmp = torch.zeros((labels_discretev.shape[0],d_output,labels_discretev.shape[2]),dtype=labels_discretev.dtype)
      tmp[:] = np.nan
      tmp[:,dataset.discrete_idx,:] = labels_discretev
      labels_discretev = tmp[:,featidxplot,:]
      tmp = torch.zeros((labels_discretev.shape[0],d_output,labels_discretev.shape[2]),dtype=labels_discretev.dtype)
      tmp[:] = np.nan
      tmp[:,dataset.discrete_idx,:] = pred_discretev
      pred_discretev = tmp[:,featidxplot,:]
    else:
      labels_discretev = None
      pred_discretev = None
    d_output = len(featidxplot)
  
  ismasked = maskidx is not None and len(maskidx) > 0
  naxr = int(np.ceil(d_output/naxc))
  if ax is None:
    fig,ax = plt.subplots(naxr,naxc,sharex='all',figsize=(20,20))
    ax = ax.flatten()
    plt.tight_layout(h_pad=0)

  pred_cmap = lambda x: plt.get_cmap("tab10")(x%10)
  for i in range(d_output):
    ax[i].cla()
    if outnames is not None:
      ti = ax[i].set_title(outnames[i],y=1.0,pad=-14,color=pred_cmap(i),loc='left')
    
    if pred_discretev is not None and len(pred_discretev) > 0 and (not torch.all(torch.isnan(pred_discretev[:,i,:]))):
      predcurr = pred_discretev[:,i,:].T
      labelscurr = labels_discretev[:,i,:].T
      zlabels = torch.max(labelscurr[torch.isnan(labelscurr)==False])
      zpred = torch.max(predcurr[torch.isnan(predcurr)==False])
      im = torch.stack((1-labelscurr/zlabels,1-predcurr/zpred,torch.ones(predcurr.shape)),dim=-1)
      im[torch.isnan(im)] = 1.
      ax[i].imshow(im.numpy(),aspect='auto')
    else:    
      lims = torch.quantile(labelsv[:,i][torch.isnan(labelsv[:,i])==False],torch.tensor([prctile_lim/100.,1-prctile_lim/100.]))
      ax[i].plot(labelsv[:,i],'k-',label='True')
      if ismasked:
        ax[i].plot(maskidx,predv[maskidx,i],'.',color=pred_cmap(i),label='Pred')
      else:
        ax[i].plot(predv[:,i],'-',color=pred_cmap(i),label='Pred')
      #ax[i].set_ylim([-ylim_nstd,ylim_nstd])
      ax[i].set_ylim(lims)
      if outnames is not None:
        plt.setp(ti,color=pred_cmap(i))
  ax[0].set_xlim([0,labelsv.shape[0]])
    
  return fig,ax



def animate_pose(Xkps,focusflies=[],ax=None,fig=None,t0=0,
                 figsizebase=11,ms=6,lw=1,focus_ms=12,focus_lw=3,
                 titletexts={},savevidfile=None,fps=30,trel0=0,
                 inputs=None,nstd_input=3,contextl=10,axinput=None,
                 attn_weights=None,skeledgecolors=None,
                 globalpos_future=None,tspred_future=None,
                 futurecolor=[0,0,0,.25],futurelw=1,futurems=6,
                 futurealpha=.25):
  
  #ani = animate_pose(Xkps,focusflies=focusflies,t0=t0,titletexts=titletexts,trel0=np.maximum(0,config['contextl']-64),
  #                  inputs=inputs,contextl=config['contextl']-1,attn_weights=attn_weights,
  #                  globalpos_future={'Pred': globalposfuture},
  #                  tspred_future=dataset.tspred_global)

  plotinput = inputs is not None and len(inputs) > 0

  # attn_weights[key] should be T x >=contextl x nfocusflies
  plotattn = attn_weights is not None
  
  plotfuture = globalpos_future is not None

  ninputs = 0
  if plotinput:
    inputnames = []
    for v in inputs.values():
      if v is not None:
        inputnames = list(v.keys())
        break
    ninputs = len(inputnames)
    if ninputs == 0:
      plotinput = False
      
  if plotinput or plotattn:
    naxc = len(Xkps)
    naxr = 1
    nax = naxc*naxr
  else:
    nax = len(Xkps)
    naxc = int(np.ceil(np.sqrt(nax)))
    naxr = int(np.ceil(nax/naxc))
  
  if plotattn:
    nsubax = ninputs + 1
  else:
    nsubax = ninputs
  
  # get rid of blank flies
  Xkp = list(Xkps.values())[0]
  T = Xkp.shape[-2]
  isreal = mabe.get_real_flies(Xkp)
  nflies = Xkp.shape[-1]
  isfocusfly = np.zeros(nflies,dtype=bool)
  isfocusfly[focusflies] = True
  for Xkp in Xkps.values():
    assert(nflies == Xkp.shape[-1])
    isreal = isreal | mabe.get_real_flies(Xkp)

  for k,v in Xkps.items():
    Xkps[k] = v[...,isreal]
  focusflies = np.nonzero(isfocusfly[isreal])[0]
    
  nflies = np.count_nonzero(isreal)

  minv = -mabe.ARENA_RADIUS_MM*1.01
  maxv = mabe.ARENA_RADIUS_MM*1.01
  
  h = {}

  trel = trel0
  t = t0+trel
  createdax = False
  if ax is None:
    if fig is None:
      fig = plt.figure()
      if plotinput or plotattn:
        fig.set_figheight(figsizebase*1.5)
      else:
        fig.set_figheight(figsizebase*naxr)
      fig.set_figwidth(figsizebase*naxc)

    if plotinput or plotattn:
      gs = matplotlib.gridspec.GridSpec(3,len(Xkps)*nsubax, figure=fig)
      ax = np.array([fig.add_subplot(gs[:2,nsubax*i:nsubax*(i+1)]) for i in range(len(Xkps))])
    else:
      ax = fig.subplots(naxr,naxc)

    for axcurr in ax:
      axcurr.set_xticks([])      
      axcurr.set_yticks([])
    createdax = True
  else:
    assert(ax.size>=nax)
  ax = ax.flatten()
  if (plotinput or plotattn) and (axinput is None):
    gs = matplotlib.gridspec.GridSpec(3,len(Xkps)*nsubax, figure=fig)
    axinput = {}
    for i,k in enumerate(Xkps):
      if k in inputs:
        axinput[k] = np.array([fig.add_subplot(gs[-1,i*nsubax+j]) for j in range(nsubax)])
        for axcurr in axinput[k][1:]:
          axcurr.set_yticks([])

    createdax = True

  if createdax:
    fig.tight_layout()

  h['kpt'] = []
  h['edge'] = []
  h['ti'] = []
  if plotfuture:
    h['future'] = []
    nsamples = {k: globalpos_future[k].shape[0] for k in globalpos_future.keys()}
  
  titletext_ts = np.array(list(titletexts.keys()))
  
  if 0 in titletexts:
    titletext_str = titletexts[0]
  else:
    titletext_str = ''

  for i,k in enumerate(Xkps):
    
    if plotfuture and k in globalpos_future:
      hfuture = []
      ntsfuture = globalpos_future[k].shape[2]
      for j in range(len(focusflies)):
        futurecolors = plt.get_cmap('jet')(np.linspace(0,1,ntsfuture))
        futurecolors[:,-1] = futurealpha
        hfuturefly = [None,]*ntsfuture
        for tfuturei in range(ntsfuture-1,-1,-1):
          hfuturecurr = ax[i].plot(globalpos_future[k][:,trel,tfuturei,0,j],globalpos_future[k][:,trel,tfuturei,1,j],'.',
                                   color=futurecolors[tfuturei],ms=futurems,lw=futurelw)[0]
          hfuturefly[tfuturei] = hfuturecurr
        # for samplei in range(nsamples[k]):
        #   hfuturecurr = ax[i].plot(globalpos_future[k][samplei,trel,:,0,j],globalpos_future[k][samplei,trel,:,1,j],'.-',color=futurecolor,ms=futurems,lw=futurelw)[0]
        #   hfuturefly.append(hfuturecurr)
        hfuture.append(hfuturefly)
      h['future'].append(hfuture)
    
    hkpt,hedge,_,_,_ = mabe.plot_flies(Xkps[k][...,trel,:],ax=ax[i],kpt_ms=ms,skel_lw=lw,skeledgecolors='tab20')

    for j in focusflies:
      hkpt[j].set_markersize(focus_ms)
      hedge[j].set_linewidth(focus_lw)
    h['kpt'].append(hkpt)
    h['edge'].append(hedge)

    ax[i].set_aspect('equal')
    mabe.plot_arena(ax=ax[i])
    if i == 0:
      hti = ax[i].set_title(f'{titletext_str} {k}, t = {t}')
    else:
      hti = ax[i].set_title(k)
    h['ti'].append(hti)

    ax[i].set_xlim(minv,maxv)
    ax[i].set_ylim(minv,maxv)

  if plotinput or plotattn:
    h['input'] = {}
    t0input = np.maximum(0,trel-contextl)
    contextlcurr = trel0-t0input+1

    if plotinput:
      for k in inputs.keys():
        h['input'][k] = []
        for i,inputname in enumerate(inputnames):
          inputcurr = inputs[k][inputname][trel+1:t0input:-1,:]
          if contextlcurr < contextl:
            pad = np.zeros([contextl-contextlcurr,]+list(inputcurr.shape)[1:])
            pad[:] = np.nan
            inputcurr = np.r_[inputcurr,pad]
          hin = axinput[k][i].imshow(inputcurr,vmin=-nstd_input,vmax=nstd_input,cmap='coolwarm')
          axinput[k][i].set_title(inputname)
          axinput[k][i].axis('auto')
          h['input'][k].append(hin)
    if plotattn:
      for k in attn_weights.keys():
        if k not in h['input']:
          h['input'][k] = []
        # currently only support one focus fly
        hattn = axinput[k][-1].plot(attn_weights[k][trel,-contextl:,0],np.arange(contextl,0,-1))[0]
        #axinput[k][-1].set_xscale('log')
        axinput[k][-1].set_ylim([-.5,contextl-.5])
        axinput[k][-1].set_xlim([0,1])
        axinput[k][-1].invert_yaxis()
        axinput[k][-1].set_title('attention')
        h['input'][k].append(hattn)

  hlist = []
  for hcurr in h.values():
    if type(hcurr) == list:
      hlist+=hcurr
    else:
      hlist+=[hcurr,]

  def update(trel):

      t = t0+trel
      if np.any(titletext_ts<=trel):
        titletext_t = np.max(titletext_ts[titletext_ts<=trel])
        titletext_str = titletexts[titletext_t]
      else:
        titletext_str = ''

      for i,k in enumerate(Xkps):
        mabe.plot_flies(Xkps[k][...,trel,:],ax=ax[0],hkpts=h['kpt'][i],hedges=h['edge'][i])
        if plotfuture and k in globalpos_future:
          ntsfuture = globalpos_future[k].shape[2]
          for j in range(len(focusflies)):
            
            for tfuturei in range(ntsfuture-1,-1,-1):
              h['future'][i][j][tfuturei].set_xdata(globalpos_future[k][:,trel,tfuturei,0,j])
              h['future'][i][j][tfuturei].set_ydata(globalpos_future[k][:,trel,tfuturei,1,j])
            
            # for samplei in range(nsamples[k]):
            #   h['future'][i][j][samplei].set_xdata(globalpos_future[k][samplei,trel,:,0,j])
            #   h['future'][i][j][samplei].set_ydata(globalpos_future[k][samplei,trel,:,1,j])
        if i == 0:
          h['ti'][i].set_text(f'{titletext_str} {k}, t = {t}')
        else:
          h['ti'][i].set_text(k)

      if plotinput or plotattn:
        t0input = np.maximum(0,trel-contextl)
        contextlcurr = trel-t0input+1
      
      if plotinput:
        for k in inputs.keys():
          for i,inputname in enumerate(inputnames):
            
            inputcurr = inputs[k][inputname][trel+1:t0input:-1,:]
            if contextlcurr < contextl:
              pad = np.zeros([contextl-contextlcurr,]+list(inputcurr.shape)[1:])
              pad[:] = np.nan
              inputcurr = np.r_[inputcurr,pad]
            h['input'][k][i].set_data(inputcurr)
            
      if plotattn:
        for k in attn_weights.keys():
          attn_curr = attn_weights[k][trel,-contextl:,0]
          h['input'][k][-1].set_xdata(attn_curr)
          # if any(np.isnan(attn_curr)==False):
          #   axinput[k][-1].set_xlim([0,np.nanmax(attn_curr)])
      return hlist
    
  ani = animation.FuncAnimation(fig, update, frames=range(trel0,T))

  if savevidfile is not None:
    print('Saving animation to file %s...'%savevidfile)
    writer = animation.PillowWriter(fps=30)
    ani.save(savevidfile,writer=writer)
    print('Finished writing.')

  return ani

def json_load_helper(jsonfile):
  with open(jsonfile,'r') as f:
    config = json.load(f)
  config = {k: v for k,v in config.items() if re.search('^_comment',k) is None}
  return config

def read_config(jsonfile):
  
  config = json_load_helper(DEFAULTCONFIGFILE)
  config1 = json_load_helper(jsonfile)  

  # destructive to config
  overwrite_config(config,config1)
  
  config['intrainfile'] = os.path.join(config['datadir'],config['intrainfilestr'])
  config['invalfile'] = os.path.join(config['datadir'],config['invalfilestr'])
  
  if type(config['flatten_obs_idx']) == str:
    if config['flatten_obs_idx'] == 'sensory':
      config['flatten_obs_idx'] = get_sensory_feature_idx()
    else:
      raise ValueError(f"Unknown type {config['flatten_obs_idx']} for flatten_obs_idx")

  if type(config['discreteidx']) == str:
    if config['discreteidx'] == 'global':
      config['discreteidx'] = featglobal.copy()
    else:
      raise ValueError(f"Unknown type {config['discreteidx']} for discreteidx")
  if type(config['discreteidx']) == list:
    for i,v in enumerate(config['discreteidx']):
      if type(v) == str:
        config['discreteidx'][i] = mabe.posenames.index(v)
    config['discreteidx'] = np.array(config['discreteidx'])
    
  if config['modelstatetype'] == 'prob' and config['minstateprob'] is None:
    config['minstateprob'] = 1/config['nstates']
    
  if 'all_discretize_epsilon' in config:
    config['all_discretize_epsilon'] = np.array(config['all_discretize_epsilon'])
    if 'discreteidx' in config and config['discreteidx'] is not None:
      config['discretize_epsilon'] = config['all_discretize_epsilon'][config['discreteidx']]
    
  if 'input_noise_sigma' in config:
    config['input_noise_sigma'] = np.array(config['input_noise_sigma'])
  #elif 'input_noise_sigma_mult' in config and 'all_discretize_epsilon' in config:
  #  config['input_noise_sigma'] = np.zeros(config['all_discretize_epsilon'].shape)
  #  l = len(config['input_noise_sigma_mult'])
  #  config['input_noise_sigma'][:l] = config['all_discretize_epsilon'][:l]*np.array(config['input_noise_sigma_mult'])    
    
  assert config['modeltype'] in ['mlm','clm']
  assert config['modelstatetype'] in ['prob','best',None]
  assert config['masktype'] in ['ind','block',None]
  
  if ('obs_embedding_types' in config) and (type(config['obs_embedding_types']) == dict):
    for k,v in config['obs_embedding_types'].items():
      if v == 'conv1d':
        # modernize
        config['obs_embedding_types'][k] = 'conv1d_feat'
    if 'obs_embedding_params' not in config:
      config['obs_embedding_params'] = {}
    else:
      if type(config['obs_embedding_params']) != dict:
        assert config['obs_embedding_params'] is None
        config['obs_embedding_params'] = {}

    for k,et in config['obs_embedding_types'].items():
      if k not in config['obs_embedding_params']:
        config['obs_embedding_params'][k] = {}
      params = config['obs_embedding_params'][k]
      if et == 'conv1d_time':
        if 'stride' not in params:
          params['stride'] = 1
        if 'dilation' not in params:
          params['dilation'] = 1
        if 'kernel_size' not in params:
          params['kernel_size'] = 2
        if 'padding' not in params:
          w = (params['stride']-1)+(params['kernel_size']*params['dilation'])-1
          params['padding'] = (w,0)
        if 'channels' not in params:
          params['channels'] = [64,256,512]
      elif et == 'conv2d':
        if 'stride' not in params:
          params['stride'] = (1,1)
        elif type(params['stride']) == int:
          params['stride'] = (params['stride'],params['stride'])
        if 'dilation' not in params:
          params['dilation'] = (1,1)
        elif type(params['dilation']) == int:
          params['dilation'] = (params['dilation'],params['dilation'])
        if 'kernel_size' not in params:
          params['kernel_size'] = (2,3)
        elif type(params['kernel_size']) == int:
          params['kernel_size'] = (params['kernel_size'],params['kernel_size'])
        if 'padding' not in params:
          w1 = (params['stride'][0]-1)+(params['kernel_size'][0]*params['dilation'][0])-1
          w2 = (params['stride'][1]-1)+(params['kernel_size'][1]*params['dilation'][1])-1
          w2a = int(np.ceil(w2/2))
          params['padding'] = ((w1,0),(w2a,w2-w2a))
          #params['padding'] = 'same'
        if 'channels' not in params:
          params['channels'] = [16,64,128]
      elif et == 'conv1d_feat':
        if 'stride' not in params:
          params['stride'] = 1
        if 'dilation' not in params:
          params['dilation'] = 1
        if 'kernel_size' not in params:
          params['kernel_size'] = 3
        if 'padding' not in params:
          params['padding'] = 'same'
        if 'channels' not in params:
          params['channels'] = [16,64,128]
      elif et == 'fc':
        pass
      else:
        raise ValueError(f'Unknown embedding type {et}')
        
  
  return config
    
def overwrite_config(config0,config1,no_overwrite=[]):
  # maybe fix: no_overwrite is just a list of parameter names. this may fail in recursive calls
  for k,v in config1.items():
    if k in no_overwrite:
      continue
    if (k in config0) and (config0[k] is not None) and (type(v) == dict):
      overwrite_config(config0[k],config1[k],no_overwrite=no_overwrite)
    else:
      config0[k] = v
  return

def load_and_filter_data(infile,config):
  
  print(f"loading raw data from {infile}...")
  data = load_raw_npz_data(infile)

  if (len(config['discreteidx']) > 0) and config['discretize_epsilon'] is None:
    if (config['all_discretize_epsilon'] is None):
      scale_perfly = compute_scale_allflies(data)
      config['all_discretize_epsilon'] = compute_noise_params(data,scale_perfly,simplify_out=config['simplify_out'])
    config['discretize_epsilon'] = config['all_discretize_epsilon'][config['discreteidx']]

  # filter out data
  print('filtering data...')
  if config['categories'] is not None and len(config['categories']) > 0:
    filter_data_by_categories(data,config['categories'])
    
  # augment by flipping
  if 'augment_flip' in config and config['augment_flip']:
    flipvideoidx = np.max(data['videoidx'])+1+data['videoidx']
    data['videoidx'] = np.concatenate((data['videoidx'],flipvideoidx),axis=0)
    firstid = np.max(data['ids'])+1
    flipids = data['ids'].copy()
    flipids[flipids>=0] += firstid
    data['ids'] = np.concatenate((data['ids'],flipids),axis=0)
    data['frames'] = np.tile(data['frames'],(2,1))
    flipX = mabe.flip_flies(data['X'])
    data['X'] = np.concatenate((data['X'],flipX),axis=2)
    data['y'] = np.tile(data['y'],(1,2,1))
    data['isdata'] = np.tile(data['isdata'],(2,1))
    data['isstart'] = np.tile(data['isstart'],(2,1))

  # compute scale parameters
  print('computing scale parameters...')
  scale_perfly = compute_scale_allflies(data)

  if np.isnan(SENSORY_PARAMS['otherflies_touch_mult']):
    print('computing touch parameters...')
    SENSORY_PARAMS['otherflies_touch_mult'] = compute_otherflies_touch_mult(data)

  # throw out data that is missing scale information - not so many frames
  idsremove = np.nonzero(np.any(np.isnan(scale_perfly),axis=0))[0]
  data['isdata'][np.isin(data['ids'],idsremove)] = False

  return data,scale_perfly

def criterion_wrapper(example,pred,criterion,dataset,config):
  tgt_continuous,tgt_discrete = dataset.get_continuous_discrete_labels(example)
  pred_continuous,pred_discrete = dataset.get_continuous_discrete_labels(pred)
  tgt = {'labels': tgt_continuous, 'labels_discrete': tgt_discrete}
  pred1 = {'continuous': pred_continuous, 'discrete': pred_discrete}
  if config['modeltype'] == 'mlm':
    if dataset.discretize:
      loss,loss_discrete,loss_continuous = criterion(tgt,pred1,mask=example['mask'].to(pred.device),
                                                     weight_discrete=config['weight_discrete'],extraout=True)
    else:
      loss = criterion(tgt_continuous.to(device=pred.device),pred_continuous,
                      example['mask'].to(pred.device))
      loss_continuous = loss
      loss_discrete = 0.
  else:
    if dataset.discretize:
      loss,loss_discrete,loss_continuous = criterion(tgt,pred1,weight_discrete=config['weight_discrete'],extraout=True)
    else:
      loss = criterion(tgt_continuous.to(device=pred.device),pred_continuous)
      loss_continuous = loss
      loss_discrete = 0.
  return loss,loss_discrete,loss_continuous


# MLM - no sensory
#loadmodelfile = os.path.join(savedir,'flymlm_71G01_male_epoch100_202301215712.pth')
# MLM with sensory
#loadmodelfile = os.path.join(savedir,'flymlm_71G01_male_epoch100_202301003317.pth')
# CLM with sensory
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_202301211242.pth')
# CLM with sensory but only global motion output
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch15_202301014322.pth')
#loadmodelfile = None
# CLM, predicting forward, sideways vel
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_202302060458.pth')
# CLM, trained with dropout = 0.8 on movement
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230228T193725.pth')
# CLM, trained with dropout = 0.8 on movement, more wall touch keypoints
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230302T221828.pth')
# CLM, trained with dropout = 0.8 on movement, other fly touch features
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230303T230750.pth')
# CLM, trained with dropout = 1.0 on movement, other fly touch features, 10 layers, 512 context
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230305T135655.pth')
# CLM with mixed continuous and discrete state
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230419T175759.pth')
# CLM with mixed continuous and discrete state, movement input
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230421T223920.pth')
# flattened CLM, forward, sideways, orientation are binned outputs
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230512T202000.pth')
# flattened CLM, forward, sideways, orientation are binned outputs, do_separate_inputs = True
# loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flyclm_flattened_mixed_71G01_male_epoch54_20230517T153613.pth'

def get_modeltype_str(config,dataset):
  if config['modelstatetype'] is not None:
    modeltype_str = f"{config['modelstatetype']}_{config['modeltype']}"
  else:
    modeltype_str = config['modeltype']
  if dataset.flatten:
    modeltype_str += '_flattened'
  if dataset.continuous and dataset.discretize:
    reptype = 'mixed'  
  elif dataset.continuous:
    reptype = 'continuous'
  elif dataset.discretize:
    reptype = 'discrete'
  modeltype_str += f'_{reptype}'
  if config['categories'] is None or len(config['categories']) == 0:
    category_str = 'all'
  else:
    category_str = '_'.join(config['categories'])
  modeltype_str += f'_{category_str}'

  return modeltype_str

def initialize_debug_plots(dataset,dataloader,data,name='',tsplot=None,traj_nsamplesplot=3):

  example = next(iter(dataloader))

  # plot to visualize input features
  fig,ax = debug_plot_sample_inputs(dataset,example)

  # plot to check that we can get poses from examples
  hpose,ax,fig = debug_plot_batch_pose(example,dataset,data=data,tsplot=tsplot)
  ax[-1,0].set_xlabel('Train')

  # plot to visualize motion outputs
  axtraj,figtraj = debug_plot_batch_traj(example,dataset,data=data,
                                         label_true='Label',
                                         label_pred='Raw',
                                         nsamplesplot=traj_nsamplesplot)
  figtraj.set_figheight(18)
  figtraj.set_figwidth(12)
  axtraj[0].set_title(name)
  figtraj.tight_layout()
  
  hdebug = {
    'figpose': fig,
    'axpose': ax,
    'hpose': hpose,
    'figtraj': figtraj,
    'axtraj': axtraj,
    'hstate': None,
    'axstate': None,
    'figstate': None,
    'example': example
  }

  plt.show()
  plt.pause(.001)
  
  return hdebug
  
def update_debug_plots(hdebug,config,model,dataset,example,pred,criterion=None,name='',tsplot=None,traj_nsamplesplot=3):
  
  if config['modelstatetype'] == 'prob':
    pred1 = model.maxpred({k: v.detach() for k,v in pred.items()})
  elif config['modelstatetype'] == 'best':
    pred1 = model.randpred(pred.detach())
  else:
    if isinstance(pred,dict):
      pred1 = {k: v.detach().cpu() for k,v in pred.items()}
    else:
      pred1 = pred.detach().cpu()
  debug_plot_batch_pose(example,dataset,pred=pred1,h=hdebug['hpose'],ax=hdebug['axpose'],fig=hdebug['figpose'],tsplot=tsplot)
  debug_plot_batch_traj(example,dataset,criterion=criterion,config=config,pred=pred1,ax=hdebug['axtraj'],fig=hdebug['figtraj'],nsamplesplot=traj_nsamplesplot)
  if config['modelstatetype'] == 'prob':
    hstate,axstate,figstate = debug_plot_batch_state(pred['stateprob'].detach().cpu(),nsamplesplot=3,
                                                      h=hdebug['hstate'],ax=hdebug['axstate'],fig=hdebug['figstate'])
    hdebug['axstate'][0].set_title(name)

  hdebug['axtraj'][0].set_title(name)
  
def initialize_loss_plots(loss_epoch):

  nax = len(loss_epoch)//2
  assert (nax >= 1) and (nax <= 3)
  hloss = {}
    
  hloss['fig'],hloss['ax'] = plt.subplots(nax,1)
  if nax == 1:
    hloss['ax'] = [hloss['ax'],]
    
  hloss['train'], = hloss['ax'][0].plot(loss_epoch['train'].cpu(),'.-',label='Train')
  hloss['val'], = hloss['ax'][0].plot(loss_epoch['val'].cpu(),'.-',label='Val')
  
  if 'train_continuous' in loss_epoch:
    hloss['train_continuous'], = hloss['ax'][1].plot(loss_epoch['train_continuous'].cpu(),'.-',label='Train continuous')
  if 'train_discrete' in loss_epoch:
    hloss['train_discrete'], = hloss['ax'][2].plot(loss_epoch['train_discrete'].cpu(),'.-',label='Train discrete')
  if 'val_continuous' in loss_epoch:
    hloss['val_continuous'], = hloss['ax'][1].plot(loss_epoch['val_continuous'].cpu(),'.-',label='Val continuous')
  if 'val_discrete' in loss_epoch:
    hloss['val_discrete'], = hloss['ax'][2].plot(loss_epoch['val_discrete'].cpu(),'.-',label='Val discrete')
        
  hloss['ax'][-1].set_xlabel('Epoch')
  hloss['ax'][-1].set_ylabel('Loss')
  for l in hloss['ax']:
    l.legend()
  return hloss

def update_loss_plots(hloss,loss_epoch):

  hloss['train'].set_ydata(loss_epoch['train'].cpu())
  hloss['val'].set_ydata(loss_epoch['val'].cpu())
  if 'train_continuous' in loss_epoch:
    hloss['train_continuous'].set_ydata(loss_epoch['train_continuous'].cpu())
  if 'train_discrete' in loss_epoch:
    hloss['train_discrete'].set_ydata(loss_epoch['train_discrete'].cpu())
  if 'val_continuous' in loss_epoch:
    hloss['val_continuous'].set_ydata(loss_epoch['val_continuous'].cpu())
  if 'val_discrete' in loss_epoch:
    hloss['val_discrete'].set_ydata(loss_epoch['val_discrete'].cpu())
  for l in hloss['ax']:
    l.relim()
    l.autoscale()

def initialize_loss(train_dataset,config):
  loss_epoch = {}
  keys = ['train','val']
  if train_dataset.discretize:
    keys = keys + ['train_continuous','train_discrete','val_continuous','val_discrete']
  for key in keys:
    loss_epoch[key] = torch.zeros(config['num_train_epochs'])
    loss_epoch[key][:] = torch.nan
  return loss_epoch

def initialize_model(d_input,d_output,config,train_dataset,device):

  # architecture arguments
  MODEL_ARGS = {
    'd_model': config['d_model'], 
    'nhead': config['nhead'], 
    'd_hid': config['d_hid'], 
    'nlayers': config['nlayers'],
    'dropout': config['dropout']
    }
  if config['modelstatetype'] is not None:
    MODEL_ARGS['nstates'] = config['nstates']
    assert config['obs_embedding'] == False, 'Not implemented'
    assert train_dataset.flatten == False, 'Not implemented'
  if config['modelstatetype'] == 'prob':
    MODEL_ARGS['minstateprob'] = config['minstateprob']
    
  if config['obs_embedding']:
    MODEL_ARGS['input_idx'],MODEL_ARGS['input_szs'] = train_dataset.get_input_shapes()
    MODEL_ARGS['embedding_types'] = config['obs_embedding_types']
    MODEL_ARGS['embedding_params'] = config['obs_embedding_params']
  if train_dataset.flatten:
    MODEL_ARGS['ntokens_per_timepoint'] = train_dataset.ntokens_per_timepoint
    d_input = train_dataset.flatten_dinput
    d_output = train_dataset.flatten_max_doutput
  elif train_dataset.discretize:
    MODEL_ARGS['d_output_discrete'] = train_dataset.d_output_discrete
    MODEL_ARGS['nbins'] = train_dataset.discretize_nbins
    d_output = train_dataset.d_output_continuous
    
  if config['modelstatetype'] == 'prob':
    model = TransformerStateModel(d_input,d_output,**MODEL_ARGS).to(device)
    criterion = prob_causal_criterion
  elif config['modelstatetype'] == 'min':
    model = TransformerBestStateModel(d_input,d_output,**MODEL_ARGS).to(device)  
    criterion = min_causal_criterion
  else:
    model = TransformerModel(d_input,d_output,**MODEL_ARGS).to(device)
    
    if train_dataset.discretize:
      # this should be maybe be len(train_dataset.discrete_idx) / train_dataset.d_output
      config['weight_discrete'] = len(config['discreteidx']) / nfeatures
      if config['modeltype'] == 'mlm':
        criterion = mixed_masked_criterion
      else:
        criterion = mixed_causal_criterion
    else:
      if config['modeltype'] == 'mlm':
        criterion = masked_criterion
      else:
        criterion = causal_criterion
        
  # if train_dataset.dct_m is not None and config['weight_dct_consistency'] > 0:
  #   criterion = lambda tgt,pred,**kwargs: criterion(tgt,pred,**kwargs) + train_dataset.compare_dct_to_next_relative(pred)
          
  return model,criterion

def compute_loss(model,dataloader,dataset,device,mask,criterion,config):
  
  is_causal = dataset.ismasked() == False
  if is_causal:
    mask = None
    
  model.eval()
  with torch.no_grad():
    all_loss = torch.zeros(len(dataloader),device=device)
    loss = torch.tensor(0.0).to(device)
    if dataset.discretize:  
      loss_discrete = torch.tensor(0.0).to(device)
      loss_continuous = torch.tensor(0.0).to(device)
    nmask = 0
    for i,example in enumerate(dataloader):
      pred = model(example['input'].to(device=device),mask=mask,is_causal=is_causal)
      loss_curr,loss_discrete_curr,loss_continuous_curr = criterion_wrapper(example,pred,criterion,dataset,config)

      if config['modeltype'] == 'mlm':
        nmask += torch.count_nonzero(example['mask'])
      else:
        nmask += example['input'].shape[0]*dataset.ntimepoints
      all_loss[i] = loss_curr
      loss+=loss_curr
      if dataset.discretize:
        loss_discrete+=loss_discrete_curr
        loss_continuous+=loss_continuous_curr
      
    loss = loss.item() / nmask

    if dataset.discretize:
      loss_discrete = loss_discrete.item() / nmask
      loss_continuous = loss_continuous.item() / nmask
      return loss,loss_discrete,loss_continuous
    else:
      return loss
    
def predict_all(dataloader,dataset,model,config,mask):
  
  is_causal = dataset.ismasked() == False
  
  with torch.no_grad():
    w = next(iter(model.parameters()))
    device = w.device

  
  # compute predictions and labels for all validation data using default masking
  all_pred = []
  all_mask = []
  all_labels = []
  all_pred_discrete = []
  all_labels_discrete = []
  with torch.no_grad():
    for example in dataloader:
      pred = model.output(example['input'].to(device=device),mask=mask,is_causal=is_causal)
      if config['modelstatetype'] == 'prob':
        pred = model.maxpred(pred)
      elif config['modelstatetype'] == 'best':
        pred = model.randpred(pred)
      if isinstance(pred,dict):
        pred = {k: v.cpu() for k,v in pred.items()}
      else:
        pred = pred.cpu()
      pred1 = dataset.get_full_pred(pred)
      labels1 = dataset.get_full_labels(example=example,use_todiscretize=True)
      all_pred.append(pred1)
      all_labels.append(labels1)
      if dataset.discretize:
        all_pred_discrete.append(pred['discrete'])
        all_labels_discrete.append(example['labels_discrete'])
      if 'mask' in example:
        all_mask.append(example['mask'])

  return all_pred,all_labels,all_mask,all_pred_discrete,all_labels_discrete

def parse_modelfile(modelfile):
  _,filestr = os.path.split(modelfile)
  filestr,_ = os.path.splitext(filestr)
  m = re.match('fly(.*)_epoch\d+_(\d{8}T\d{6})',filestr)
  if m is None:
    modeltype_str = ''
    savetime = ''
  else:
    modeltype_str = m.groups(1)[0]
    savetime = m.groups(1)[1]
  return modeltype_str,savetime

def compute_all_attention_weight_rollouts(attn_weights0):
  attn_weights_rollout = None
  firstframe = None
  attn_context = None
  tpred = attn_weights0[0].size #to do check
  for t,w0 in enumerate(attn_weights0):
    if w0 is None: 
      continue
    w = compute_attention_weight_rollout(w0)
    w = w[-1,-1,...]
    if attn_weights_rollout is None:
      attn_weights_rollout = np.zeros((tpred,)+w.shape)
      attn_weights_rollout[:] = np.nan
      firstframe = t
      attn_context = w.shape[0]
    if attn_context < w.shape[0]:
      pad = np.zeros([tpred,w.shape[0]-attn_context,]+list(w.shape)[1:])
      pad[:firstframe,...] = np.nan
      attn_context = w.shape[0]
      attn_weights_rollout = np.concatenate((attn_weights_rollout,pad),axis=1)
    attn_weights_rollout[t,:] = 0.
    attn_weights_rollout[t,:w.shape[0]] = w
  return attn_weights_rollout

def get_pose_future(data,scales,tspred_global,ts=None,fliespred=None):

  maxT = data['X'].shape[2]
  if ts is None:
    ts = np.arange(maxT)
  if fliespred is None:
    fliespred = np.arange(data['X'].shape[3])
  
  Xkpfuture = np.zeros((data['X'].shape[0],data['X'].shape[1],len(ts),len(tspred_global),len(fliespred)))
  Xkpfuture[:] = np.nan
  for ti,toff in enumerate(tspred_global):
    idxcurr = ts<maxT-toff
    tscurr = ts[idxcurr]
    Xkpfuture[:,:,idxcurr,ti] = data['X'][:,:,tscurr+toff][...,fliespred]
    isbad = data['videoidx'][tscurr,0] != data['videoidx'][tscurr+toff,0]
    Xkpfuture[:,:,isbad] = np.nan
  
  relposefuture,globalposfuture = compute_pose_features(Xkpfuture,scales)
  if globalposfuture.ndim == 3: # when there is one fly, it gets collapsed
    globalposfuture = globalposfuture[...,None]
    relposefuture = relposefuture[...,None]
  globalposfuture = globalposfuture.transpose(1,2,0,3)
  relposefuture = relposefuture.transpose(1,2,0,3)
  return globalposfuture,relposefuture

def animate_predict_open_loop(model,dataset,data,scale_perfly,config,fliespred,t0,tpred,burnin=None,debug=False,plotattnweights=False,plotfuture=False,nsamplesfuture=1):
    
  #ani = animate_predict_open_loop(model,val_dataset,valdata,val_scale_perfly,config,fliespred,t0,tpred,debug=False,
  #                            plotattnweights=False,plotfuture=train_dataset.ntspred_global>1,nsamplesfuture=nsamplesfuture)
    
  if burnin is None:
    burnin = config['contextl']-1

  Xkp_true = data['X'][...,t0:t0+tpred,:].copy()
  Xkp = Xkp_true.copy()

  #fliespred = np.nonzero(mabe.get_real_flies(Xkp))[0]
  ids = data['ids'][t0,fliespred]
  scales = scale_perfly[:,ids]

  if plotfuture:
    # subtract one from tspred_global, as the tspred_global for predicted data come from the previous frame
    globalposfuture_true,relposefuture_true = get_pose_future(data,scales,[t+1 for t in dataset.tspred_global],ts=np.arange(t0,t0+tpred),fliespred=fliespred)

  model.eval()

  # capture all outputs of predict_open_loop in a tuple
  res = dataset.predict_open_loop(Xkp,fliespred,scales,burnin,model,maxcontextl=config['contextl'],
                                  debug=debug,need_weights=plotattnweights,nsamples=nsamplesfuture)
  Xkp_pred,zinputs,globalposfuture_pred,relposefuture_pred = res[:4]
  if plotattnweights:
    attn_weights0 = res[4]

  Xkps = {'Pred': Xkp_pred.copy(),'True': Xkp_true.copy()}
  #Xkps = {'Pred': Xkp_pred.copy()}
  if len(fliespred) == 1:
    inputs = {'Pred': split_features(zinputs,axis=1)}
  else:
    inputs = None

  if plotattnweights:
    attn_weights = {'Pred': compute_all_attention_weight_rollouts(attn_weights0)}
  else:
    attn_weights = None

  focusflies = fliespred
  titletexts = {0: 'Initialize', burnin: ''}
  
  if plotfuture:
    future_args = {'globalpos_future': {'Pred': globalposfuture_pred, 'True': globalposfuture_true[None,...]},
                   'tspred_future': dataset.tspred_global}
  else:
    future_args = {}
    
  ani = animate_pose(Xkps,focusflies=focusflies,t0=t0,titletexts=titletexts,trel0=np.maximum(0,config['contextl']-64),
                    inputs=inputs,contextl=config['contextl']-1,attn_weights=attn_weights,
                    **future_args)
  
  return ani

def get_interval_ends(tf):
  tf = np.r_[False,tf,False]
  idxstart = np.nonzero((tf[:-1]==False) & (tf[1:] == True))[0]
  idxend = np.nonzero((tf[:-1]==True) & (tf[1:] == False))[0]
  return idxstart,idxend
  
def split_data_by_id(data):
  splitdata = []
  nflies = data['X'].shape[-1]
  for flynum in range(nflies):
    isdata = data['isdata'][:,flynum] & (data['isstart'][:,flynum]==False)
    idxstart,idxend = get_interval_ends(isdata)
    for i in range(len(idxstart)):
      i0 = idxstart[i]
      i1 = idxend[i]
      id = data['ids'][i0,flynum]
      if data['isdata'][i0-1,flynum] and data['ids'][i0-1,flynum] == id:
        i0 -= 1
      splitdata.append({
        'flynum': flynum,
        'id': id,
        'i0': i0,
        'i1': i1,
      })
  return splitdata
  
def explore_representation(configfile):

  config = read_config(configfile)

  np.random.seed(config['numpy_seed'])
  torch.manual_seed(config['torch_seed'])
  device = torch.device(config['device'])

  plt.ion()
  
  data,scale_perfly = load_and_filter_data(config['intrainfile'],config)
  splitdata = split_data_by_id(data)
  
  
  for i in range(len(splitdata)):
    scurr = splitdata[i]
    fcurr = compute_features(data['X'][...,scurr['i0']:scurr['i1'],:],
                             scurr['id'],scurr['flynum'],scale_perfly,smush=False,simplify_in='no_sensory')
    movecurr = fcurr['labels']
    if i == 0:
      move = movecurr
    else:
      move = np.r_[move,movecurr]

  outnames_global = ['forward','sideways','orientation']
  outnames = outnames_global + [mabe.posenames[x] for x in np.nonzero(featrelative)[0]]

  mu = np.nanmean(move,axis=0)
  sig = np.nanstd(move,axis=0)
  zmove = (move-mu)/sig
  
  # pca = sklearn.decomposition.PCA()
  # pca.fit(zmove)
  
  # fig,ax = plt.subplots(1,1)
  # fig.set_figheight(12)
  # fig.set_figwidth(20)
  # clim = np.max(np.abs(pca.components_))*np.array([-1,1])
  # him = ax.imshow(pca.components_,aspect='auto',clim=clim,cmap='RdBu')
  # fig.colorbar(him,ax=ax)
  # ax.set_xlabel('Movement feature')
  # ax.set_ylabel('PC')
  # ax.set_xticks(np.arange(move.shape[1]))
  # ax.set_xticklabels(outnames)
  # ax.set_yticks(np.arange(pca.components_.shape[0]))
  # ax.tick_params(axis='x', labelrotation = 90)
  # ax.invert_yaxis()
  # ax.set_title('PCA weights')
  # fig.tight_layout()
  
  # ica = sklearn.decomposition.FastICA(whiten='unit-variance')
  # ica.fit(zmove)
  
  # fig,ax = plt.subplots(1,1)
  # fig.set_figheight(12)
  # fig.set_figwidth(20)
  # clim = np.mean(np.max(ica.components_,axis=1),axis=0)*np.array([-1,1])
  # him = ax.imshow(ica.components_,aspect='auto',clim=clim,cmap='RdBu')
  # fig.colorbar(him,ax=ax)
  # ax.set_xlabel('Movement feature')
  # ax.set_ylabel('PC')
  # ax.set_xticks(np.arange(move.shape[1]))
  # ax.set_xticklabels(outnames)
  # ax.set_yticks(np.arange(ica.components_.shape[0]))
  # ax.tick_params(axis='x', labelrotation = 90)
  # ax.invert_yaxis()
  # ax.set_title('ICA weights')
  # fig.tight_layout()
  
  
  # spca = sklearn.decomposition.SparsePCA()
  # spca.fit(zmove)
  
  # fig,ax = plt.subplots(1,1)
  # fig.set_figheight(12)
  # fig.set_figwidth(20)
  # clim = np.max(np.abs(spca.components_))*np.array([-1,1])
  # him = ax.imshow(spca.components_,aspect='auto',clim=clim,cmap='RdBu')
  # fig.colorbar(him,ax=ax)
  # ax.set_xlabel('Movement feature')
  # ax.set_ylabel('PC')
  # ax.set_xticks(np.arange(move.shape[1]))
  # ax.set_xticklabels(outnames)
  # ax.set_yticks(np.arange(spca.components_.shape[0]))
  # ax.tick_params(axis='x', labelrotation = 90)
  # ax.invert_yaxis()
  # ax.set_title('SPCA weights')
  # fig.tight_layout()

  bin_edges = np.zeros((nfeatures,config['discretize_nbins']+1))  
  for feati in range(nfeatures):
    bin_edges[feati,:] = select_bin_edges(move[:,feati],config['discretize_nbins'],config['all_discretize_epsilon'][feati],feati=feati)
  
  featpairs = [
    ['left_front_leg_tip_angle','left_front_leg_tip_dist'],
    ['left_middle_femur_base_angle','left_middle_femur_tibia_joint_angle'],
    ['left_middle_femur_tibia_joint_angle','left_middle_leg_tip_angle'],
    ]
  nax = len(featpairs)
  nc = int(np.ceil(np.sqrt(nax)))
  nr = int(np.ceil(nax/nc))
  fig,ax = plt.subplots(nr,nc,squeeze=False)
  ax = ax.flatten()

  for i in range(len(featpairs)):
    feati = [outnames.index(x) for x in featpairs[i]]
    density,_,_ = np.histogram2d(zmove[:,feati[0]],zmove[:,feati[1]],bins=[bin_edges[feati[0],:],bin_edges[feati[1],:]],density=True)
    ax[i].cla()
    X, Y = np.meshgrid(bin_edges[feati[0],1:-1],bin_edges[feati[1],1:-1])
    density = density[1:-1,1:-1]
    him = ax[i].pcolormesh(X,Y,density, norm=matplotlib.colors.LogNorm(vmin=np.min(density[density>0]), vmax=np.max(density)),edgecolors='k')
    ax[i].set_xlabel(outnames[feati[0]])
    ax[i].set_ylabel(outnames[feati[1]])
  fig.tight_layout()

  # ax[i].plot(move[:,feati[0]],move[:,feati[1]],'.',alpha=.02,markersize=1)
  
  
  valdata,val_scale_perfly = load_and_filter_data(config['invalfile'],config)

def debug_add_noise(train_dataset,data,idxsample=0,tsplot=None):
  # debugging adding noise
  train_dataset.set_eval_mode()
  extrue = train_dataset[idxsample]
  train_dataset.set_train_mode()
  exnoise = train_dataset[idxsample]
  exboth = {}
  for k in exnoise.keys():
    if type(exnoise[k]) == torch.Tensor:
      exboth[k] = torch.stack((extrue[k],exnoise[k]),dim=0)
    elif type(exnoise[k]) == dict:
      exboth[k] = {}
      for k1 in exnoise[k].keys():
        exboth[k][k1] = torch.stack((torch.tensor(extrue[k][k1]),torch.tensor(exnoise[k][k1])))
    else:
      raise ValueError('huh')  
  if tsplot is None:
    tsplot = np.round(np.linspace(0,64,4)).astype(int)
  hpose,ax,fig = debug_plot_batch_pose(exboth,train_dataset,data=data,tsplot=tsplot)
  Xfeat_true = train_dataset.get_Xfeat(example=extrue,use_todiscretize=True)
  Xfeat_noise = train_dataset.get_Xfeat(example=exnoise,use_todiscretize=True)

def clean_intermediate_results(savedir):
  modelfiles = list(pathlib.Path(savedir).glob('*.pth'))
  modelfilenames = [p.name for p in modelfiles]
  p = re.compile('^(?P<prefix>.+)_epoch(?P<epoch>\d+)_(?P<suffix>.*).pth$')
  m = [p.match(n) for n in modelfilenames]
  ids = np.array([x.group('prefix')+'___'+x.group('suffix') for x in m])
  epochs = np.array([int(x.group('epoch')) for x in m])
  uniqueids,idx = np.unique(ids,return_inverse=True)
  removed = []
  nremoved = 0
  for i,id in enumerate(uniqueids):
    idxcurr = np.nonzero(ids==id)[0]
    if len(idxcurr) <= 1:
      continue
    j = idxcurr[np.argmax(epochs[idxcurr])]
    idxremove = idxcurr[epochs[idxcurr]<epochs[j]]
    while True:
      print(f'Keep {modelfilenames[j]} and remove the following files:')
      for k in idxremove:
        print(f'Remove {modelfilenames[k]}')
      r = input('(y/n) ?  ')
      if r == 'y':
        for k in idxremove:
          print(f'Removing {modelfiles[k]}')
          os.remove(modelfiles[k])
          removed.append(modelfiles[k])
          nremoved += 1
        break
      elif r == 'n':
        break
      else:
        print('Bad input, response must be y or n')
  print(f'Removed {nremoved} files')
  return removed

def gzip_pickle_dump(filename, data):
  with gzip.open(filename, 'wb') as f:
    pickle.dump(data, f)

def gzip_pickle_load(filename):
  with gzip.open(filename, 'rb') as f:
    return pickle.load(f)

# import h5py
# def hdf5_save(f, d, name="root"):

#   try:
#     if isinstance(f,str):
#       f = h5py.File(f, "w")
#     if isinstance(d,dict):
#       g = f.create_group('dict__'+name)
#       for k,v in d.items():
#         hdf5_save(g,v,name=k)
#     elif isinstance(d,list):
#       if np.all(np.array([isinstance(x,str) for x in d])):
#         g = f.create_dataset('liststr__'+name,data=np.array(d,dtype='S'))
#       else:
#         g = f.create_group('list__'+name)
#         for i,v in enumerate(d):
#           hdf5_save(g,v,name=str(i))
#     elif isinstance(d,np.ndarray):    
#       if np.all(np.array([isinstance(x,str) for x in d])):
#         g = f.create_dataset('ndarraystr__'+name,data=np.array(d,dtype='S'))
#       else:
#         g = f.create_dataset('ndarray__'+name,data=d)
#     else:
#       g = f.create_dataset('other__'+name,data=d)
  
#   except Exception as e:
#     print(f'Error saving {name}')
#     raise e
  
#   return f

def save_chunked_data(savefile,d):
  gzip_pickle_dump(savefile,d)
  return

def load_chunked_data(savefile):
  return gzip_pickle_load(savefile)

def compute_npad(tspred_global,dct_m):
  npad = np.max(tspred_global)
  if dct_m is not None:
    npad = np.maximum(dct_m.shape[0],npad)
  return npad

def sanity_check_tspred(data,config,npad,scale_perfly,dct_m,t0=510,flynum=0):
  # sanity check on computing features when predicting many frames into the future
  # compute inputs and outputs for frames t0:t0+contextl+npad+1 with tspred_global set by config
  # and inputs ant outputs for frames t0:t0+contextl+1 with just next frame prediction.
  # the inputs should match each other 
  # the outputs for each of the config['tspred_global'] should match the next frame 
  # predictions for the corresponding frame
  
  contextl = config['contextl']
  id = data['ids'][t0,flynum]

  contextlpad = contextl+npad
  t1 = t0+contextlpad-1
  x = data['X'][...,t0:t1+1,:]
  xcurr1 = compute_features(x,id,flynum,scale_perfly,outtype=np.float32,dct_m=dct_m,tspred_global=config['tspred_global'],npad=npad)

  contextlpad = contextl+1
  t1 = t0+contextlpad-1
  x = data['X'][...,t0:t1+1,:]
  xcurr0 = compute_features(x,id,flynum,scale_perfly,outtype=np.float32)

  assert np.all(xcurr0['input']==xcurr1['input'])
  ftidx = [(f,1) for f in featglobal] + [(f,1) for f in np.nonzero(featrelative)[0]]
  idx = ravel_label_index(ftidx,dct_m=dct_m,tspred_global=config['tspred_global'])
  assert np.all(xcurr1['labels'][:,idx]==xcurr0['labels'])

  return

def sanity_check_temporal_dep(train_dataloader,device,train_src_mask,is_causal,model,tmess=300):
  # sanity check on temporal dependences
  # create xin2 that is like xin, except xin2 from frame tmess onwards is set to garbage value 100.
  # make sure that model(xin) and model(xin2) matches before frame tmess
  x = next(iter(train_dataloader))
  xin = x['input'].clone()
  xin2 = xin.clone()
  tmess = 300
  xin2[:,tmess:,:] = 100.
  model.eval()
  with torch.no_grad():
    pred = model(xin.to(device),mask=train_src_mask,is_causal=is_causal)
    pred2 = model(xin2.to(device),mask=train_src_mask,is_causal=is_causal)
  if type(pred) == dict:
    for k in pred.keys():
      matches = torch.all(pred2[k][:,:tmess]==pred[k][:,:tmess]).item()
      assert matches
  else:
    matches = torch.all(pred2[:,:tmess]==pred[:,:tmess]).item()
    assert matches

def main(configfile,loadmodelfile=None,restartmodelfile=None):

  # to save time, i saved the chunked data to a pkl file
  tmpsavefile = 'chunkeddata20230905.pkl'
  #tmpsavefile = 'chunkeddata20230828.pkl'
  doloadtmpsavefile = os.path.exists(tmpsavefile)
  #tmpsavefile = None

  # configuration parameters for this model
  config = read_config(configfile)
  
  # set loadmodelfile and restartmodelfile from config if not specified
  if loadmodelfile is None and 'loadmodelfile' in config:
    loadmodelfile = config['loadmodelfile']
  if restartmodelfile is None and 'restartmodelfile' in config:
    loadmodelfile = config['restartmodelfile']
  
  # if loadmodelfile or restartmodelfile specified, use its config
  if loadmodelfile is not None:
    load_config_from_model_file(loadmodelfile,config)
  elif restartmodelfile is not None:
    no_overwrite = ['num_train_epochs',]
    load_config_from_model_file(restartmodelfile,config,no_overwrite=no_overwrite)
  
  print(f"batch size = {config['batch_size']}")

  # seed the random number generators
  np.random.seed(config['numpy_seed'])
  torch.manual_seed(config['torch_seed'])
  
  # set device (cuda/cpu)
  device = torch.device(config['device'])

  plt.ion()

  if doloadtmpsavefile:
    # load cached, pre-chunked data  
    print(f'Loading tmp save file {tmpsavefile}')
    with open(tmpsavefile,'rb') as f:
      tmp = pickle.load(f)
    data = tmp['data']
    scale_perfly = tmp['scale_perfly']
    valdata = tmp['valdata']
    val_scale_perfly = tmp['val_scale_perfly']
    X = tmp['X']
    valX = tmp['valX']
  else:
    # load raw data
    data,scale_perfly = load_and_filter_data(config['intrainfile'],config)
    valdata,val_scale_perfly = load_and_filter_data(config['invalfile'],config)
  
  # if using discrete cpsine transform, create dct matrix
  # this didn't seem to work well, so probably won't use in the future
  if config['dct_tau'] is not None and config['dct_tau'] > 0:
    dct_m,idct_m = get_dct_matrix(config['dct_tau'])
  else:
    dct_m = None
    idct_m = None

  # how much to pad outputs by -- depends on how many frames into the future we will predict
  npad = compute_npad(config['tspred_global'],dct_m)
  chunk_data_params = {'npad': npad}

  # function for computing features
  reparamfun = lambda x,id,flynum,**kwargs: compute_features(x,id,flynum,scale_perfly,outtype=np.float32,
                                                            simplify_out=config['simplify_out'],
                                                            simplify_in=config['simplify_in'],
                                                            dct_m=dct_m,tspred_global=config['tspred_global'],**kwargs)

  val_reparamfun = lambda x,id,flynum,**kwargs: compute_features(x,id,flynum,val_scale_perfly,
                                                                outtype=np.float32,
                                                                simplify_out=config['simplify_out'],
                                                                simplify_in=config['simplify_in'],
                                                                dct_m=dct_m,tspred_global=config['tspred_global'],**kwargs)
  
  # sanity check on computing features when predicting many frames into the future
  sanity_check_tspred(data,config,npad,scale_perfly,dct_m,t0=510,flynum=0)
                                    
  if not doloadtmpsavefile:
    # chunk the data if we didn't load the pre-chunked cache file
    print('Chunking training data...')
    X = chunk_data(data,config['contextl'],reparamfun,**chunk_data_params)
    print('Chunking val data...')
    valX = chunk_data(valdata,config['contextl'],val_reparamfun,**chunk_data_params)
    print('Done.')
    
    if tmpsavefile is not None:
      print('Saving chunked data to file')
      with open(tmpsavefile,'wb') as f:
        pickle.dump({'X': X,'valX': valX, 'data': data, 'valdata': valdata, 'scale_perfly': scale_perfly,'val_scale_perfly': val_scale_perfly},f)
      print('Done.')

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
  }
  train_dataset_params = {
    'input_noise_sigma': config['input_noise_sigma'],
  }

  print('Creating training data set...')
  train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)
  print('Done.')

  # zscore and discretize parameters for validation data set based on train data
  # we will continue to use these each time we rechunk the data
  dataset_params['zscore_params'] = train_dataset.get_zscore_params()
  dataset_params['discretize_params'] = train_dataset.get_discretize_params()

  print('Creating validation data set...')
  val_dataset = FlyMLMDataset(valX,**dataset_params)
  print('Done.')

  # get properties of examples from the first training example
  example = train_dataset[0]
  d_input = example['input'].shape[-1]
  d_output = train_dataset.d_output
  outnames = train_dataset.get_outnames()

  train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=True,
                                                pin_memory=True,
                                                )
  ntrain = len(train_dataloader)

  val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=config['batch_size'],
                                              shuffle=False,
                                              pin_memory=True,
                                              )
  nval = len(val_dataloader)

  example = next(iter(train_dataloader))
  sz = example['input'].shape
  print(f'batch input shape = {sz}')

  # set up debug plots
  debug_params = {}
  # if contextl is long, still just look at samples from the first 64 frames
  if config['contextl'] > 64:
    debug_params['tsplot'] = np.round(np.linspace(0,64,5)).astype(int)
    debug_params['traj_nsamplesplot'] = 1
  hdebug = {}
  hdebug['train'] = initialize_debug_plots(train_dataset,train_dataloader,data,name='Train',**debug_params)
  hdebug['val'] = initialize_debug_plots(val_dataset,val_dataloader,valdata,name='Val',**debug_params)

  # create the model
  model,criterion = initialize_model(d_input,d_output,config,train_dataset,device)

  # optimizer
  num_training_steps = config['num_train_epochs'] * ntrain
  optimizer = transformers.optimization.AdamW(model.parameters(),**config['optimizer_args'])
  lr_scheduler = transformers.get_scheduler('linear',optimizer,num_warmup_steps=0,
                                            num_training_steps=num_training_steps)


  # initialize structure to keep track of loss
  loss_epoch = initialize_loss(train_dataset,config)
  last_val_loss = None

  progress_bar = tqdm.tqdm(range(num_training_steps))

  # create attention mask
  contextl = example['input'].shape[1]
  if config['modeltype'] == 'mlm':
    train_src_mask = generate_square_full_mask(contextl).to(device)
    is_causal = False
  elif config['modeltype'] == 'clm':
    train_src_mask = torch.nn.Transformer.generate_square_subsequent_mask(contextl,device=device)
    is_causal = True
    #train_src_mask = generate_square_subsequent_mask(contextl).to(device)
  else:
    raise

  # sanity check on temporal dependences
  sanity_check_temporal_dep(train_dataloader,device,train_src_mask,is_causal,model,tmess=300)

  modeltype_str = get_modeltype_str(config,train_dataset)
  if ('model_nickname' in config) and (config['model_nickname'] is not None):
    modeltype_str = config['model_nickname']

  hloss = initialize_loss_plots(loss_epoch)
  
  # epoch = 40
  # restartmodelfile = f'llmnets/flyclm_flattened_mixed_71G01_male_epoch{epoch}_20230517T153613.pth'
  # loss_epoch = load_model(restartmodelfile,model,device,lr_optimizer=optimizer,scheduler=lr_scheduler)
  # with torch.no_grad():
  #   pred = model(example['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
  # update_debug_plots(hdebug['train'],config,model,train_dataset,example,pred,name='Train',criterion=criterion)

  # train
  if loadmodelfile is None:
    
    # restart training
    if restartmodelfile is not None:
      loss_epoch = load_model(restartmodelfile,model,device,lr_optimizer=optimizer,scheduler=lr_scheduler)
      update_loss_nepochs(loss_epoch,config['num_train_epochs'])
      update_loss_plots(hloss,loss_epoch)
      #loss_epoch = {k: v.cpu() for k,v in loss_epoch.items()}
      epoch = np.nonzero(np.isnan(loss_epoch['train'].cpu().numpy()))[0][0]
      progress_bar.update(epoch*ntrain)
    else:
      epoch = 0
    
    savetime = datetime.datetime.now()
    savetime = savetime.strftime('%Y%m%dT%H%M%S')
    ntimepoints_per_batch = train_dataset.ntimepoints
    valexample = next(iter(val_dataloader))
    
    for epoch in range(epoch,config['num_train_epochs']):
      
      model.train()
      tr_loss = torch.tensor(0.0).to(device)
      if train_dataset.discretize:
        tr_loss_discrete = torch.tensor(0.0).to(device)
        tr_loss_continuous = torch.tensor(0.0).to(device)

      nmask_train = 0
      for step, example in enumerate(train_dataloader):
        
        pred = model(example['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
        loss,loss_discrete,loss_continuous = criterion_wrapper(example,pred,criterion,train_dataset,config)
          
        loss.backward()
        
        # how many timepoints are in this batch for normalization
        if config['modeltype'] == 'mlm':
          nmask_train += torch.count_nonzero(example['mask'])
        else:
          nmask_train += example['input'].shape[0]*ntimepoints_per_batch 

        if step % config['niterplot'] == 0:
          
          with torch.no_grad():
            trainpred = model.output(example['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
            valpred = model.output(valexample['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
          update_debug_plots(hdebug['train'],config,model,train_dataset,example,trainpred,name='Train',criterion=criterion,**debug_params)
          update_debug_plots(hdebug['val'],config,model,val_dataset,valexample,valpred,name='Val',criterion=criterion,**debug_params)
          plt.show()
          plt.pause(.1)

        tr_loss_step = loss.detach()
        tr_loss += tr_loss_step
        if train_dataset.discretize:
          tr_loss_discrete += loss_discrete.detach()
          tr_loss_continuous += loss_continuous.detach()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
        optimizer.step()
        lr_scheduler.step()
        model.zero_grad()
        
        # update progress bar
        stat = {'train loss': tr_loss.item()/nmask_train,'last val loss': last_val_loss,'epoch': epoch}
        if train_dataset.discretize:
          stat['train loss discrete'] = tr_loss_discrete.item()/nmask_train
          stat['train loss continuous'] = tr_loss_continuous.item()/nmask_train
        progress_bar.set_postfix(stat)
        progress_bar.update(1)
        
        # end of iteration loop

      # training epoch complete
      loss_epoch['train'][epoch] = tr_loss.item() / nmask_train
      if train_dataset.discretize:
        loss_epoch['train_discrete'][epoch] = tr_loss_discrete.item() / nmask_train
        loss_epoch['train_continuous'][epoch] = tr_loss_continuous.item() / nmask_train
      
      # compute validation loss after this epoch
      if val_dataset.discretize:
         loss_epoch['val'][epoch],loss_epoch['val_discrete'][epoch],loss_epoch['val_continuous'][epoch] = \
           compute_loss(model,val_dataloader,val_dataset,device,train_src_mask,criterion,config)
      else:
        loss_epoch['val'][epoch] = \
          compute_loss(model,val_dataloader,val_dataset,device,train_src_mask,criterion,config)
      last_val_loss = loss_epoch['val'][epoch].item()
    
      update_loss_plots(hloss,loss_epoch)
      plt.show()
      plt.pause(.1)

      # rechunk the training data
      if np.mod(epoch+1,config['epochs_rechunk']) == 0:
        print(f'Rechunking data after epoch {epoch}')
        X = chunk_data(data,config['contextl'],reparamfun,**chunk_data_params)
      
        train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)
        print('New training data set created')

      if (epoch+1)%config['save_epoch'] == 0:
        savefile = os.path.join(config['savedir'],f"fly{modeltype_str}_epoch{epoch+1}_{savetime}.pth")
        print(f'Saving to file {savefile}')
        save_model(savefile,model,lr_optimizer=optimizer,scheduler=lr_scheduler,loss=loss_epoch,config=config)

    savefile = os.path.join(config['savedir'],f'fly{modeltype_str}_epoch{epoch+1}_{savetime}.pth')
    save_model(savefile,model,lr_optimizer=optimizer,scheduler=lr_scheduler,loss=loss_epoch,config=config)

    print('Done training')
  else:
    modeltype_str,savetime = parse_modelfile(loadmodelfile)
    loss_epoch = load_model(loadmodelfile,model,device,lr_optimizer=optimizer,scheduler=lr_scheduler)
    update_loss_plots(hloss,loss_epoch)
    
  model.eval()

  # compute predictions and labels for all validation data using default masking
  all_pred,all_labels,all_mask,all_pred_discrete,all_labels_discrete = predict_all(val_dataloader,val_dataset,model,config,train_src_mask)

  # plot comparison between predictions and labels on validation data
  predv = stack_batch_list(all_pred)
  labelsv = stack_batch_list(all_labels)
  maskv = stack_batch_list(all_mask)
  pred_discretev = stack_batch_list(all_pred_discrete)
  labels_discretev = stack_batch_list(all_labels_discrete)
  
  fig,ax = debug_plot_global_histograms(predv,labelsv,train_dataset,nbins=25,subsample=1,compare='pred')
  # glabelsv = train_dataset.get_global_movement(labelsv)
  # gpredprev = torch.zeros(glabelsv.shape)
  # gpredprev[:] = np.nan
  # for i,dt in enumerate(train_dataset.tspred_global):
  #   gpredprev[dt:,i,:] = glabelsv[:-dt,i,:]
  # predprev = torch.zeros(labelsv.shape)
  # predprev[:] = np.nan
  # train_dataset.set_global_movement(gpredprev,predprev)
  # fig,ax = debug_plot_global_histograms(predprev,labelsv,train_dataset,nbins=25,subsample=100)
  
  if train_dataset.dct_m is not None:
    debug_plot_dct_relative_error(predv,labelsv,train_dataset)
  if train_dataset.ntspred_global > 1:
    debug_plot_global_error(predv,labelsv,pred_discretev,labels_discretev,train_dataset)

  # crop to nplot for plotting
  nplot = 8000 #min(len(all_labels),8000//config['batch_size']//config['contextl']+1)
  predv = predv[:nplot,:]
  labelsv = labelsv[:nplot,:]
  if len(maskv) > 0:
    maskv = maskv[:nplot,:]
  pred_discretev = pred_discretev[:nplot,:]
  labels_discretev = labels_discretev[:nplot,:]
  
  if maskv is not None and len(maskv) > 0:
    maskidx = torch.nonzero(maskv)[:,0]
  else:
    maskidx = None
  
  ntspred_plot = np.minimum(4,train_dataset.ntspred_global)
  featidxplot = select_featidx_plot(train_dataset,ntspred_plot)
  naxc = np.maximum(1,int(np.round(len(featidxplot)/nfeatures)))
  fig,ax = debug_plot_predictions_vs_labels(predv,labelsv,pred_discretev,labels_discretev,outnames=outnames,maskidx=maskidx,naxc=naxc,featidxplot=featidxplot,dataset=val_dataset)
  if train_dataset.ntspred_global > 1:
    featidxplot = select_featidx_plot(train_dataset,ntspred_plot=train_dataset.ntspred_global,ntsplot_relative=0)
    naxc = np.maximum(1,int(np.round(len(featidxplot)/nfeatures)))
    fig,ax = debug_plot_predictions_vs_labels(predv,labelsv,pred_discretev,labels_discretev,outnames=outnames,maskidx=maskidx,naxc=naxc,featidxplot=featidxplot,dataset=val_dataset)
  
  if train_dataset.ntspred_global > 1:
    featidxplot = train_dataset.ravel_label_index([(featglobal[0],t) for t in train_dataset.tspred_global])
    fig,ax = debug_plot_predictions_vs_labels(predv,labelsv,pred_discretev,labels_discretev,outnames=outnames,maskidx=maskidx,featidxplot=featidxplot,dataset=val_dataset)

  if train_dataset.dct_tau > 0:
    fstrs = ['left_middle_leg_tip_angle','left_front_leg_tip_angle','left_wing_angle']
    fs = [mabe.posenames.index(x) for x in fstrs]
    featidxplot = train_dataset.ravel_label_index([(f,i+1) for i in range(train_dataset.dct_tau+1) for f in fs])
    fig,ax = debug_plot_predictions_vs_labels(predv,labelsv,pred_discretev,labels_discretev,outnames=outnames,maskidx=maskidx,featidxplot=featidxplot,dataset=val_dataset,naxc=len(fs))

    predrelative_dct = train_dataset.get_relative_movement_dct(predv.numpy())
    labelsrelative_dct = train_dataset.get_relative_movement_dct(labelsv.numpy())
    fsdct = [np.array(mabe.posenames)[featrelative].tolist().index(x) for x in fstrs]
    predrelative_dct = predrelative_dct[:,:,fsdct].astype(train_dataset.dtype)
    labelsrelative_dct = labelsrelative_dct[:,:,fsdct].astype(train_dataset.dtype)
    outnamescurr = [f'{f}_dt{i+1}' for i in range(train_dataset.dct_tau) for f in fstrs]
    fig,ax = debug_plot_predictions_vs_labels(torch.as_tensor(predrelative_dct.reshape((-1,train_dataset.dct_tau*len(fsdct)))),
                                              torch.as_tensor(labelsrelative_dct.reshape((-1,train_dataset.dct_tau*len(fsdct)))),
                                              outnames=outnamescurr,maskidx=maskidx,naxc=len(fstrs))


  # generate an animation of open loop prediction
  tpred = 2000 + config['contextl']

  # all frames must have real data
  
  burnin = config['contextl']-1
  contextlpad = burnin + 1
  allisdata = interval_all(valdata['isdata'],contextlpad)
  isnotsplit = interval_all(valdata['isstart']==False,tpred)[1:,...]
  canstart = np.logical_and(allisdata[:isnotsplit.shape[0],:],isnotsplit)
  flynum = 2
  t0 = np.nonzero(canstart[:,flynum])[0][40000]    
  # flynum = 2
  # t0 = np.nonzero(canstart[:,flynum])[0][0]
  fliespred = np.array([flynum,])

  randstate_np = np.random.get_state()
  randstate_torch = torch.random.get_rng_state()

  nsamplesfuture = 100  
  ani = animate_predict_open_loop(model,val_dataset,valdata,val_scale_perfly,config,fliespred,t0,tpred,debug=False,
                                  plotattnweights=False,plotfuture=train_dataset.ntspred_global>1,nsamplesfuture=nsamplesfuture)

  vidtime = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  savevidfile = os.path.join(config['savedir'],f"samplevideo_{modeltype_str}_{savetime}_{vidtime}.gif")

  print('Saving animation to file %s...'%savevidfile)
  writer = animation.PillowWriter(fps=30)
  ani.save(savevidfile,writer=writer)
  print('Finished writing.')

def debug_plot_histograms(dataset,alpha=1):
  r = np.random.rand(dataset.discretize_bin_samples.shape[0])-.5
  ftidx = dataset.unravel_label_index(dataset.discrete_idx)
  #ftidx[featrelative[ftidx[:,0]],1]+=1
  fs = np.unique(ftidx[:,0])
  ts = np.unique(ftidx[:,1])
  nfs = len(fs)
  fig,ax = plt.subplots(nfs,1,sharey=True)
  fig.set_figheight(17)
  fig.set_figwidth(30)
  colors = mabe.get_n_colors_from_colormap('hsv',dataset.discretize_nbins)
  colors[:,:-1] *= .7
  colors = colors[np.random.permutation(dataset.discretize_nbins),:]
  colors[:,-1] = alpha
  edges = np.zeros((len(fs),2))
  edges[:,0] = np.inf
  edges[:,1] = -np.inf
  bin_edges = dataset.discretize_bin_edges
  bin_samples = dataset.discretize_bin_samples
  if dataset.sig_labels is not None:
    bin_edges = unzscore(bin_edges,dataset.mu_labels[dataset.discrete_idx,None],dataset.sig_labels[dataset.discrete_idx,None])
    bin_samples = unzscore(bin_samples,dataset.mu_labels[None,dataset.discrete_idx,None],dataset.sig_labels[None,dataset.discrete_idx,None])
  for i,idx in enumerate(dataset.discrete_idx):
    f = ftidx[i,0]
    t = ftidx[i,1]
    fi = np.nonzero(fs==f)[0][0]
    ti = np.nonzero(ts==t)[0][0]
    edges[fi,0] = np.minimum(edges[fi,0],bin_edges[i,0])
    edges[fi,1] = np.maximum(edges[fi,1],bin_edges[i,-1])
    for j in range(dataset.discretize_nbins):
      ax[fi].plot(bin_samples[:,i,j],ti+r,'.',ms=.01,color=colors[j])
      ax[fi].plot([bin_edges[i,j],]*2,ti+np.array([-.5,.5]),'k-')
    ax[fi].plot([bin_edges[i,-1],]*2,ti+np.array([-.5,.5]),'k-')
    ax[fi].plot(bin_edges[i,[0,-1]],[ti+.5,]*2,'k-')
    ax[fi].plot(bin_edges[i,[0,-1]],[ti-.5,]*2,'k-')
  fnames = dataset.get_movement_names()
  for i,f in enumerate(fs):
    ax[i].set_title(fnames[f])
    ax[i].set_xlim(edges[i,0],edges[i,1])
    ax[i].set_yticks(np.arange(len(ts)))
    ax[i].set_yticklabels([str(t) for t in ts])
    ax[i].set_ylim(-.5,len(ts)-.5)
    ax[i].set_xscale('symlog')
  ax[-1].set_ylabel('Delta t')
  fig.tight_layout()
  return

def debug_plot_global_histograms(predv,labelsv,train_dataset,nbins=50,subsample=1,compare='time'):
  outnames_global = train_dataset.get_movement_names_global()
  
  # global labels, continuous representation, unzscored
  # ntimepoints x tspred x nglobal
  unz_labelsv = train_dataset.unzscore_labels(labelsv.numpy())
  unz_glabelsv = train_dataset.get_global_movement(unz_labelsv)  

  # global predictions, continuous representation, unzscored
  # ntimepoints x tspred x nglobal
  unz_predv = train_dataset.unzscore_labels(predv.numpy())
  unz_gpredv = train_dataset.get_global_movement(unz_predv)

  if train_dataset.discretize:

    bin_edges = train_dataset.get_bin_edges(zscored=False)
    ftidx = train_dataset.unravel_label_index(train_dataset.discrete_idx)
    bins = []
    for f in featglobal:
      j = np.nonzero(np.all(ftidx == np.array([f,1])[None,...],axis=1))[0][0]
      bins.append(bin_edges[j])
    nbins = train_dataset.discretize_nbins
  else:
    lims = [[np.percentile(unz_glabelsv[::100,:,axi].flatten(),i).item() for i in [.1,99.9]] for axi in range(nglobal)]
    bins = [np.arange(l[0],l[1],nbins+1) for l in lims]

  ntspred = len(train_dataset.tspred_global)
  off0 = .1
  
  if compare == 'time':
    colors = mabe.get_n_colors_from_colormap('jet',len(train_dataset.tspred_global))
    colors[:,:-1] *=.8

    fig,ax = plt.subplots(2,nglobal,figsize=(30,10),sharex='col')
    w = (1-2*off0)/ntspred
    for axj,(datacurr,datatype) in enumerate(zip([unz_glabelsv,unz_gpredv],['label ','pred '])):
      for axi in range(nglobal):
        ax[axj,axi].cla()
        off = off0
        for i in range(unz_glabelsv.shape[1]):
          density,_ = np.histogram(datacurr[::subsample,i,axi],bins=bins[axi],density=True)
          ax[axj,axi].bar(np.arange(nbins)+off,density,width=w,color=colors[i],log=True,
                          align='edge',label=str(train_dataset.tspred_global[i]))
          off+=w
        ax[axj,axi].set_xticks(np.arange(nbins+1))
        ax[axj,axi].set_xticklabels(['%.2f'%x for x in bins[axi]],rotation=90)
        ax[axj,axi].set_title(datatype+outnames_global[axi])
  elif compare == 'pred':
    colors = [[0,.5,.8],[.8,.5,.8]]

    fig,ax = plt.subplots(ntspred,nglobal,figsize=(20,30),sharex='col',sharey='all')
    w = (1-2*off0)/2
    for ti in range(ntspred):
      for fi in range(nglobal):
        axcurr = ax[ti,fi]
        axcurr.cla()
        off = off0
        for i,(datacurr,datatype) in enumerate(zip([unz_glabelsv,unz_gpredv],['label','pred'])):
          density,_ = np.histogram(datacurr[::subsample,ti,fi],bins=bins[fi],density=True)
          axcurr.bar(np.arange(nbins)+off,density,width=w,color=colors[i],log=False,
                     align='edge',label=datatype)
          off+=w
        axcurr.set_xticks(np.arange(nbins+1))
        axcurr.set_xticklabels(['%.2f'%x for x in bins[fi]],rotation=90)
        axcurr.set_title(f'{outnames_global[fi]} t = {train_dataset.tspred_global[ti]}')

  ax[0,0].legend()
  fig.tight_layout()
  
  return fig,ax

def debug_plot_global_error(predv,labelsv,pred_discretev,labels_discretev,train_dataset):
  """
  debug_plot_global_error(predv,labelsv,pred_discretev,labels_discretev,train_dataset)
  inputs:
  predv: torch.Tensor, shape (ntimepoints,d_output), all prediction features in continuous representation
  labelsv: torch.Tensor, shape (ntimepoints,d_output) all labels in continuous representation
  pred_discretev: torch.Tensor, shape (ntimepoints,len(train_dataset.discrete_idx),train_dataset.discretize_nbins), discrete predictions
  labels_discretev: torch.Tensor, shape (ntimepoints,len(train_dataset.discrete_idx),train_dataset.discretize_nbins), discrete labels
  train_dataset: FlyMLMDataset, the training dataset
  """
  outnames_global = train_dataset.get_movement_names_global()
  ntimepoints = predv.shape[0]
  
  # global predictions, continuous representation
  # ntimepoints x tspred x nglobal
  gpredv = train_dataset.get_global_movement(predv)
  
  # global labels, continuous representation
  # ntimepoints x tspred x nglobal
  glabelsv = train_dataset.get_global_movement(labelsv)
  
  # compute L1 error from continuous representations, all global features
  # network predictions
  errcont = np.nanmean(torch.nn.L1Loss(reduction='none')(gpredv,glabelsv),axis=0)
  # just predicting zero velocity all the time
  pred0 = np.zeros((ntimepoints,train_dataset.d_output))
  if train_dataset.mu_labels is not None:
    pred0 = train_dataset.zscore_labels(pred0)
  gpred0 = torch.as_tensor(train_dataset.get_global_movement(pred0))
  err0cont = np.nanmean(torch.nn.L1Loss(reduction='none')(gpred0,glabelsv),axis=0)
  
  # constant velocity predictions: use real labels from dt frames previous. 
  # note we we won't have predictions for the first dt frames
  gpredprev = torch.zeros(glabelsv.shape)
  gpredprev[:] = np.nan
  for i,dt in enumerate(train_dataset.tspred_global):
    gpredprev[dt:,i,:] = glabelsv[:-dt,i,:]
  errprevcont = np.nanmean(torch.nn.L1Loss(reduction='none')(gpredprev,glabelsv),axis=0)
  
  if train_dataset.discretize:
    # ntimepoints x tspred x nglobal x nbins: discretized global predictions
    gpreddiscretev = torch.as_tensor(train_dataset.get_global_movement_discrete(pred_discretev))
    # ntimepoints x tspred x nglobal x nbins: discretized global labels
    glabelsdiscretev = torch.as_tensor(train_dataset.get_global_movement_discrete(labels_discretev))
    # cross entropy error
    errdiscrete = np.nanmean(torch.nn.CrossEntropyLoss(reduction='none')(gpreddiscretev.moveaxis(-1,1),glabelsdiscretev.moveaxis(-1,1)),axis=0)

    zerodiscretev = train_dataset.discretize_fun(pred0)
    gzerodiscretev = torch.as_tensor(train_dataset.get_global_movement_discrete(zerodiscretev))
    err0discrete = np.nanmean(torch.nn.CrossEntropyLoss(reduction='none')(gzerodiscretev.moveaxis(-1,1),glabelsdiscretev.moveaxis(-1,1)),axis=0)

    gpredprevdiscrete = torch.zeros(gpreddiscretev.shape,dtype=gpreddiscretev.dtype)
    gpredprevdiscrete[:] = np.nan
    for i,dt in enumerate(train_dataset.tspred_global):
      gpredprevdiscrete[dt:,i,:,:] = glabelsdiscretev[:-dt,i,:,:]
    errprevdiscrete = np.nanmean(torch.nn.CrossEntropyLoss(reduction='none')(gpredprevdiscrete.moveaxis(-1,1),glabelsdiscretev.moveaxis(-1,1)),axis=0)

  if train_dataset.discretize:
    nc = 2
  else:
    nc = 1
  nr = nglobal
  fig,ax = plt.subplots(nr,nc,sharex=True,squeeze=False)
  fig.set_figheight(10)
  fig.set_figwidth(12)
  #colors = mabe.get_n_colors_from_colormap('viridis',train_dataset.dct_tau)
  for i in range(nglobal):
    ax[i,0].plot(errcont[:,i],'o-',label=f'Pred')
    ax[i,0].plot(err0cont[:,i],'s-',label=f'Zero')
    ax[i,0].plot(errprevcont[:,i],'s-',label=f'Prev')
    if train_dataset.discretize:
      ax[i,1].plot(errdiscrete[:,i],'o-',label=f'Pred')
      ax[i,1].plot(err0discrete[:,i],'s-',label=f'Zero')
      ax[i,1].plot(errprevdiscrete[:,i],'s-',label=f'Prev')
      ax[i,0].set_title(f'{outnames_global[i]} L1 error')
      ax[i,1].set_title(f'{outnames_global[i]} CE error')
    else:
      ax[i,0].set_title(outnames_global[i])
  ax[-1,-1].set_xticks(np.arange(train_dataset.ntspred_global))
  ax[-1,-1].set_xticklabels([str(t) for t in train_dataset.tspred_global])
  ax[-1,-1].set_xlabel('Delta t')
  ax[0,0].legend()
  plt.tight_layout()  
  
  return

def debug_plot_dct_relative_error(predv,labelsv,train_dataset):
  dt = train_dataset.dct_tau
  dcterr = np.sqrt(np.nanmean((predv[:,train_dataset.idxdct_relative]-labelsv[:,train_dataset.idxdct_relative])**2.,axis=0))
  dcterr0 = np.sqrt(np.nanmean((labelsv[:,train_dataset.idxdct_relative])**2.,axis=0))
  dcterrprev = np.sqrt(np.nanmean((labelsv[:-dt,train_dataset.idxdct_relative]-labelsv[dt:,train_dataset.idxdct_relative])**2.,axis=0))
  
  nc = int(np.ceil(np.sqrt(nrelative)))
  nr = int(np.ceil(nrelative/nc))
  fig,ax = plt.subplots(nr,nc,sharex=True,sharey=True)
  fig.set_figheight(14)
  fig.set_figwidth(23)
  ax = ax.flatten()
  for i in range(nrelative,nc*nr):
    ax[i].remove()
  ax = ax[:nrelative]
  for i in range(nrelative):
    ax[i].plot(dcterr[:,i],'o-',label=f'pred')
    ax[i].plot(dcterr0[:,i],'s-',label=f'zero')
    ax[i].plot(dcterrprev[:,i],'s-',label=f'prev')
    ax[i].set_title(mabe.posenames[np.nonzero(featrelative)[0][i]])
  ax[-1].set_xticks(np.arange(train_dataset.dct_tau))
  ax[(nc-1)*nr-1].set_xlabel('DCT feature')
  ax[0].legend()
  plt.tight_layout()  
  
  predrelative_dct = train_dataset.get_relative_movement_dct(predv.numpy())
  labelsrelative_dct = train_dataset.get_relative_movement_dct(labelsv.numpy())
  zpredrelative_dct = np.zeros(predrelative_dct.shape)
  zlabelsrelative_dct = np.zeros(labelsrelative_dct.shape)
  for i in range(predrelative_dct.shape[1]):
    zpredrelative_dct[:,i,:] = zscore(predrelative_dct[:,i,:],train_dataset.mu_labels[train_dataset.nextframeidx_relative],
                                      train_dataset.sig_labels[train_dataset.nextframeidx_relative])
    zlabelsrelative_dct[:,i,:] = zscore(labelsrelative_dct[:,i,:],train_dataset.mu_labels[train_dataset.nextframeidx_relative],
                                        train_dataset.sig_labels[train_dataset.nextframeidx_relative])
  idcterr = np.sqrt(np.nanmean((zpredrelative_dct-zlabelsrelative_dct)**2.,axis=0))
  nexterr = np.sqrt(np.nanmean((train_dataset.get_next_relative_movement(predv)-train_dataset.get_next_relative_movement(labelsv))**2,axis=0))
  err0 = np.sqrt(np.nanmean((zlabelsrelative_dct)**2,axis=0))
  errprev = np.sqrt(np.nanmean((zlabelsrelative_dct[:-dt,:,:]-zlabelsrelative_dct[dt:,:,:])**2,axis=0))
  
  plt.figure()
  plt.clf()
  plt.plot(idcterr[0,:],'s-',label='dct pred')
  plt.plot(nexterr,'o-',label='next pred')
  plt.plot(err0[0,:],'s-',label='zero')
  plt.plot(errprev[0,:],'s-',label='prev')
  plt.legend()
  plt.xticks(np.arange(nrelative))
  plt.gca().set_xticklabels([mabe.posenames[i] for i in np.nonzero(featrelative)[0]])
  plt.xticks(rotation=90)
  plt.title('Next frame prediction')
  plt.tight_layout()
  
  
  fig,ax = plt.subplots(nr,nc,sharex=True,sharey=True)
  fig.set_figheight(14)
  fig.set_figwidth(23)
  ax = ax.flatten()
  for i in range(nrelative,nc*nr):
    ax[i].remove()
  ax = ax[:nrelative]
  for i in range(nrelative):
    ax[i].plot(idcterr[:,i],'o-',label=f'pred')
    ax[i].plot(err0[:,i],'s-',label=f'zero')
    ax[i].plot(errprev[:,i],'s-',label=f'prev')
    ax[i].set_title(mabe.posenames[np.nonzero(featrelative)[0][i]])
  ax[-1].set_xticks(np.arange(train_dataset.dct_tau))
  ax[(nc-1)*nr-1].set_xlabel('Delta t')
  ax[0].legend()
  plt.tight_layout()  
  
  return

def debug_plot_histogram_edges(train_dataset):
  bin_edges = train_dataset.get_bin_edges(zscored=False)
  ftidx = train_dataset.unravel_label_index(train_dataset.discrete_idx)
  fs = np.unique(ftidx[:,0])
  ts = np.unique(ftidx[:,1])
  fig,ax = plt.subplots(1,len(fs),sharey=True)
  movement_names = train_dataset.get_movement_names()
  for i,f in enumerate(fs):
    ax[i].cla()
    idx = ftidx[:,0]==f
    tscurr = ftidx[idx,1]
    tidx = mabe.npindex(ts,tscurr)
    ax[i].plot(bin_edges[idx,:],tidx,'.-')
    ax[i].set_title(movement_names[f])
    ax[i].set_xscale('symlog')
  ax[0].set_yticks(np.arange(len(ts)))
  ax[0].set_yticklabels([str(t) for t in ts])
  return fig,ax

if __name__ == "__main__":

  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-c',type=str,required=False,help='Path to config file',metavar='configfile',dest='configfile')
  parser.add_argument('-l',type=str,required=False,help='Path to model file to load',metavar='loadmodelfile',dest='loadmodelfile')
  parser.add_argument('-r',type=str,required=False,help='Path to model file to restart training from',metavar='restartmodelfile',dest='restartmodelfile')
  parser.add_argument('--clean',type=str,required=False,help='Delete intermediate networks saved in input directory.',metavar='cleandir',dest='cleandir')
  args = parser.parse_args()

  if args.cleandir is not None:
    assert os.path.isdir(args.cleandir)      
    removedfiles = clean_intermediate_results(args.cleandir)
  else:
    main(args.configfile,loadmodelfile=args.loadmodelfile,restartmodelfile=args.restartmodelfile)
  #explore_representation(args.configfile)
