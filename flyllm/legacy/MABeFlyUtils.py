"""
Useful functions for interacting with the files X<split>_seq.npy.

FlyDataset is a pytorch Dataset for loading and grabbing frames.
plot_fly(), plot_flies(), and animate_pose_sequence()
can be used to visualize a fly, a frame, and a sequence of frames.
"""

import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm,animation,colors,collections
import copy
import tqdm
from itertools import compress
import re

# local path to data -- modify this!
datadir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/seqdata20220307'

# data frame rate
FPS = 150.
# size of the arena the flies are enclosed in
ARENA_RADIUS_MM = 26.689
ARENA_RADIUS_PX = 507.611429 # median over all videos
PXPERMM = ARENA_RADIUS_PX/ARENA_RADIUS_MM

# names of our keypoint features
keypointnames = [
  'wing_right',
  'wing_left',
  'antennae_midpoint',
  'right_eye',
  'left_eye',
  'left_front_thorax',
  'right_front_thorax',
  'base_thorax',
  'tip_abdomen',
  'right_middle_femur_base',
  'right_middle_femur_tibia_joint',
  'left_middle_femur_base',
  'left_middle_femur_tibia_joint',
  'right_front_leg_tip',
  'right_middle_leg_tip',
  'right_back_leg_tip',
  'left_back_leg_tip',
  'left_middle_leg_tip',
  'left_front_leg_tip',
]

posenames = [
  'thorax_front_x',
  'thorax_front_y',
  'orientation',
  'head_base_x',
  'head_base_y',
  'head_angle',
  'abdomen_angle',
  'left_middle_femur_base_dist',
  'left_middle_femur_base_angle',
  'right_middle_femur_base_dist',
  'right_middle_femur_base_angle',
  'left_middle_femur_tibia_joint_dist',
  'left_middle_femur_tibia_joint_angle',
  'left_front_leg_tip_dist',
  'left_front_leg_tip_angle',
  'right_front_leg_tip_dist',
  'right_front_leg_tip_angle',
  'right_middle_femur_tibia_joint_dist',
  'right_middle_femur_tibia_joint_angle',
  'left_middle_leg_tip_dist',
  'left_middle_leg_tip_angle',
  'right_middle_leg_tip_dist',
  'right_middle_leg_tip_angle',
  'left_back_leg_tip_dist',
  'left_back_leg_tip_angle',
  'right_back_leg_tip_dist',
  'right_back_leg_tip_angle',
  'left_wing_angle',
  'right_wing_angle',
 ]

scalenames = [
  'thorax_width',
  'thorax_length',
  'abdomen_length',
  'wing_length',
  'head_width',
  'head_height',
  'std_thorax_width',
  'std_thorax_length',
  'std_abdomen_length',
  'std_wing_length',
  'std_head_width',
  'std_head_height',
]



# hard-code indices of keypoints and skeleton edges
keypointidx = np.arange(19,dtype=int)
skeleton_edges = np.array([
  [ 7,  8],
  [10, 14],
  [11, 12],
  [12, 17],
  [ 7, 11],
  [ 9, 10],
  [ 7,  9],
  [ 5,  7],
  [ 2,  3],
  [ 2,  7],
  [ 5, 18],
  [ 6, 13],
  [ 7, 16],
  [ 7, 15],
  [ 2,  4],
  [ 6,  7],
  [ 7,  0],
  [ 7,  1]
  ])
# keypoints for computing distances between pairs of flies
fidxdist = np.array([2,7,8])
# hard-coded coordinates
coordidx = np.arange(19,dtype=int)
# hard-coded body orientation
bodyoriidx = 20
# hard-coded head index
headidx = 2
# hard-coded tail index
tailidx = 8

# edges to connect subsets of the keypoints that maybe make sense
skeleton_edge_names = [
  ('base_thorax','tip_abdomen'),
  ('left_middle_femur_tibia_joint','left_middle_leg_tip'),
  ('right_middle_femur_tibia_joint','right_middle_leg_tip'),
  ('left_middle_femur_base','left_middle_femur_tibia_joint'),
  ('base_thorax','right_middle_femur_base'),
  ('right_middle_femur_base','right_middle_femur_tibia_joint'),
  ('base_thorax','left_middle_femur_base'),
  ('left_front_thorax','base_thorax'),
  ('antennae_midpoint','right_eye'),
  ('antennae_midpoint','base_thorax'),
  ('left_front_thorax','right_front_thorax'),
  ('left_front_thorax','left_front_leg_tip'),
  ('right_front_thorax','right_front_leg_tip'),
  ('base_thorax','left_back_leg_tip'),
  ('base_thorax','right_back_leg_tip'),
  ('left_eye','right_eye'),
  ('antennae_midpoint','left_eye'),
  ('right_front_thorax','base_thorax'),
  ('base_thorax','wing_left'),
  ('base_thorax','wing_right')
]

# for computing distances between pairs of flies
distkeypointnames = [
  'antennae_midpoint',
  'base_thorax',
  'tip_abdomen'
]

YTYPES = {'seq':0,'tgt':1,'frame':2}

def modrange(x,l,u):
  return np.mod(x-l,u-l)+l

def npindex(big,small):
  # big and small should be 1D arrays
  order = np.argsort(big)
  bigsorted = big[order]
  idx = np.searchsorted(bigsorted,small,side='left')
  idx[bigsorted[idx] != small] = -1    
  return idx

def circle_line_intersection(linex, liney, linedx, linedy, circler, circlex=None, circley=None):
  # intersection of a line and a circle from
  # https://mathworld.wolfram.com/Circle-LineIntersection.html

  if circlex is not None:
    linex = linex - circlex
  if circley is not None:
    liney = liney - circley
    
  D = linex * (liney+linedy)-(linex+linedx) * liney
  z1 = np.sqrt(circler**2-D**2)
  z2 = linedx * z1
  z2[linedy<0] = -z2[linedy<0]
  z3 = np.abs(linedy) * z1
  z4 = D * linedy
  z5 = -D * linedx
  xint1 = z4+z2
  yint1 = z5+z3
  xint2 = z4-z2
  yint2 = z5-z3
  m = (xint1-linex) * linedx+(yint1-liney) * linedy
  xtmp = xint1.copy()
  ytmp = yint1.copy()
  xint1[m<0] = xint2[m<0]
  yint1[m<0] = yint2[m<0]
  xint2[m<0] = xtmp[m<0]
  yint2[m<0] = ytmp[m<0]
  return xint1,yint1,xint2,yint2

"""
pytorch Dataset for loading the data (__init__) and grabbing frames.
Utilities for visualizing one frame of data also provided.
"""
class FlyDataset(Dataset):
  
  """
  data = FlyDataset(xtrainfile)
  Load in the data from file xtrainfile and initialize member variables
  Optional annotations from ytrainfile
  """
  def __init__(self,xfile,yfile=None,ntgtsout=None,balance=True,minweight=.1,maxweight=100,
               weight_restart=None,weight_set=None,binary=False,arena_radius=None,arena_center=None):
    
    self.preprocessed = False

    xdata = np.load(xfile,allow_pickle=True).item()
    # X is a dictionary keyed by random 20-character strings
    self.X = xdata['keypoints']
    # X[seqid] is nkeypoints x d=2 x seql x maxnflies x
    for k,x in self.X.items():
      self.X[k] = x.transpose((2,3,0,1))
    
    # seqids is a list of the 20-character strings for accessing sequences
    self.seqids = list(self.X.keys())
    
    # featurenames is a list of pairs of strings with a string descriptor
    # of each feature. featurenames[j][j] corresponds to X[seqid][:,:,j,j]
    self.featurenames = [re.sub('_x_mm$','',x[0]) for x in xdata['vocabulary']]
    
    # number of sequences
    self.nseqs = len(self.X)
    
    # get the sizes of inputs
    firstx = list(self.X.values())[0]
    # sequence length in frames
    self.seqlength = firstx.shape[2]
    # middle frame of the sequence, used for ordering flies by distance to target fly
    self.ctrf = int(np.ceil(self.seqlength/2))
    # maximum number of flies
    self.ntgts = firstx.shape[3]
    # number of flies to output
    if ntgtsout is None:
      self.ntgtsout = self.ntgts
    else:
      self.ntgtsout = ntgtsout
    # number of features
    self.nfeatures = firstx.shape[0]
    # number of coordinates -- 2
    self.d = firstx.shape[1]
    
    # y is optional, and not None only if annotations are provided in ytrainfile
    # y is a dictionary keyed by the same 20-character strings as X, with
    # X[seqid] corresponding to y[seqid]
    # y[seqid] is ncategories x seql x maxnflies
    self.y = None
    self.categorynames = None
    self.ncategories = 0
    if yfile is not None:
      ydata = np.load(yfile,allow_pickle=True).item()
      self.y = ydata['annotations']
      if 'annotations' in xdata:
        for id in ydata['annotations']:
          ydata['annotations'][id] = np.concatenate((xdata['annotations'][id],ydata['annotations'][id]),axis=0)
      firsty = list(self.y.values())[0]
      self.ncategories = firsty.shape[0]
    
      if 'vocabulary' in ydata:
        # categorynames is a list of string descriptors of each category
        self.categorynames = ydata['vocabulary']
        if 'categories' in xdata:
          self.categorynames = xdata['categories'] + self.categorynames
      
      # check that sizes match
      assert( len(self.y) == self.nseqs )
      assert( set(self.y.keys()) == set(self.X.keys()) )
      assert( firsty.shape[1] == self.seqlength )
      assert( firsty.shape[2] == self.ntgts )
      
    # for balanced error
    self.balance = False
    self.weights = None
    self.yvalues = None
      
    # features to use for computing inter-fly distances
    self.fidxdist = np.sort(np.array([self.featurenames.index(x) for x in distkeypointnames]))
    assert(self.fidxdist.size == len(distkeypointnames))
    
    # not all flies have data for each sequence
    self.set_idx2seqnum()
    
    # which feature numbers correspond to keypoints and are on the defined skeleton?
    self.keypointidx = np.sort(np.array([self.featurenames.index(x) for x in keypointnames]))
    assert(self.keypointidx.size == len(keypointnames))

    self.skeleton_edges = np.zeros((len(skeleton_edge_names),2),dtype=int)
    for i in range(len(skeleton_edge_names)):
      for j in range(2):
        self.skeleton_edges[i,j] = self.featurenames.index(skeleton_edge_names[i][j])

    self.arena_radius = arena_radius
    if arena_center is None:
      arena_center = np.zeros(2)
    self.arena_center = arena_center

    self.normalize = False
    self.mu = None
    self.sig = None
    self.n_zscore = None
    
    # # for z-scoring
    # self.normalize = normalize
    # if self.normalize:
    #   self.zscore(restart=zscore_restart,set=zscore_set)

  def set_idx2seqnum(self):
    
    assert self.preprocessed == False, 'set_idx2seqnum can only be called on original data'
    i = 0
    self.idx2seqnum = np.zeros(self.nseqs*self.ntgts,dtype=np.int32)
    self.idx2tgt = np.zeros(self.nseqs*self.ntgts,dtype=np.int32)
    for k,v in self.X.items():
      seqi = self.seqids.index(k)
      # x is nfeatures x d x seqlength x ntgts
      tgtidx = np.where(np.all(np.isnan(v),axis=(0,1,2))==False)[0]
      ncurr = len(tgtidx)
      self.idx2seqnum[i:i+ncurr] = seqi
      self.idx2tgt[i:i+ncurr] = tgtidx
      i += ncurr
    self.idx2seqnum = self.idx2seqnum[:i]
    self.n = i
    self.idx2tgt = self.idx2tgt[:i]

  def reformat_x(self,x):
    x = x.reshape((self.nfeatures*self.d,x.shape[2],x.shape[3]))
    return x
  
  def get_zscore_stats(self):
    return {'mu':self.mu,'sig':self.sig,'n':self.n_zscore}
    
  def zscore(self, restart=None, set=None):
    
    self.normalize = True
    
    if set is None:
      self.mu,self.sig,self.n_zscore = compute_zscore_stats(self,restart=restart)
    else:
      self.mu = set['mu']
      self.sig = set['sig']
      self.n_zscore = set['n']
    
    for i in range(self.getnseqxs()):
      x = self.getseqx(i)
      # normalize
      z = (x-self.mu)/self.sig

      # set nans to the mean value
      isreal = get_real_flies(x)
      replace = np.isnan(x)
      replace[:,:,:,isreal==False] = False
      x[replace] = 0.

      self.setseqx(i,x)
      
    return
  
  def preprocess_all(self,rotate=True,arena=True,translate=True):
    assert self.preprocessed == False, 'Cannot preprocess more than once'
    assert self.normalize == False, 'Preprocessing cannot happen after z-scoring'

    i = 0
    xraw = self.getitem(i)['x']
    xproc,newfeaturenames = self.preprocess(xraw,tgt=None,rotate=rotate,arena=arena,translate=translate,returnnames=True)
    newnfeatures = xproc.shape[0]
    X = np.zeros((self.n,)+xproc.shape,dtype=xproc.dtype)
    X[i,...] = xproc
    
    print('Preprocessing all data')
    for i in tqdm.trange(1,self.n):
      xraw = self.getitem(i)['x']
      X[i,...] = self.preprocess(xraw,tgt=None,rotate=rotate,arena=arena,translate=translate)
      
    self.X = X
    self.preprocessed = True
    self.featurenames = newfeaturenames
    self.nfeatures = newnfeatures
    
    return
  
  def unnormalize(self,x):
    if not self.normalize or self.mu is None:
      return x
    ndims = x.ndim
    s = np.ones(ndims,dtype=int)
    # nfeatures x 2 x ...
    if x.shape[0] == self.nfeatures*self.d:
      s[0] = self.nfeatures*self.d
      return x*self.sig.reshape(s)+self.mu.reshape(s)
    elif x.shape[0] == self.nfeatures:
      s[0] = self.nfeatures
      s[1] = self.d
      return x*self.sig.reshape(s)+self.mu.reshape(s)
    else:
      assert True, 'x must be nfeatures x d x ... or nfeatures*d x ...'
      
  def get_weight_stats(self):
    return {'yvalues':self.yvalues,'ycounts':self.ycounts,'yweights':self.yweights,'n':self.n_reweight}
    
  def balance_reweight(self,restart=None,set=None,binary=False):
    
    self.balance = True
    
    if set is None:
      self.yweights,self.yvalues,self.ycounts,self.n_reweight = \
        compute_class_weights(self,restart=restart,binary=binary,)
    else:
      self.yvalues = set['yvalues']
      self.ycounts = set['ycounts']
      self.yweights = set['yweights']
      self.n_reweight = set['n']
      
    # pre-compute weights for each frame, fly
    self.weights = {}
    for seqid,y in self.y.items():
      w = np.zeros(y.shape)
      for c in range(self.ncategories):
        ycurr = y[c,...]
        wcurr = np.zeros(ycurr.shape)
        for i in range(len(self.yvalues[c])):
          v = self.yvalues[c][i]
          wcurr[ycurr==v] = self.yweights[c][i]
        w[c,...] = wcurr
      self.weights[seqid] = w
  
  def select_seq(self,doselect):

    assert self.preprocessed==False,'select_seq can only be run on data that is not preprocessed'

    for i in np.where(doselect==False)[0]:
      id = self.seqids[i]
      del self.X[id]
      del self.y[id]
      if self.weights is not None:
        del self.weights[id]

    self.seqids = list(np.array(self.seqids)[doselect])
    self.nseqs = len(self.seqids)

    self.set_idx2seqnum()
    
  def compute_y_values(self,yvalues=None,binary=True):
    return compute_y_values(self.y,yvalues=yvalues,binary=binary)
  
  def compute_y_counts(self,yvalues=None,ycounts=None,n=0):
    
    if yvalues is None:
      yvalues = self.yvalues
      
    if ycounts is None:
      ycounts = []
      for c in range(self.ncategories):
        ycounts.append({})
      
    for c in range(self.ncategories):
      for v in yvalues[c]:
        if not v in ycounts[c]:
          ycounts[c][v] = 0
         
    # loop through all labels and count
    for c in range(self.ncategories):
      for v in yvalues[c]:
        for y in self.y.values():
          ycounts[c][v] += np.count_nonzero(v==y[c,...])
        n+=ycounts[c][v]

    return ycounts,n
  
  def compute_y_seq_counts(self,yvalues=None,ycounts=None,n=0):
    
    if yvalues is None:
      yvalues = self.yvalues
      
    if ycounts is None:
      ycounts = []
      for c in range(self.ncategories):
        ycounts.append({})
      
    for c in range(self.ncategories):
      for v in yvalues[c]:
        if not v in ycounts[c]:
          ycounts[c][v] = 0
         
    # loop through all labels and count
    for c in range(self.ncategories):
      for v in yvalues[c]:
        for y in self.y.values():
          if np.any(v==y[c,...]):
            ycounts[c][v] += 1

    return ycounts,self.nseqs
    
  # total number of sequence-fly pairs
  def __len__(self):
    return self.n
  
  def preprocess(self,x,tgt=None,rotate=True,arena=True,translate=True,returnnames=False):
    # put tgt first, sort flies according to distance to tgt
    # rotate so that tgt in center frame is pointed to right

    # sort flies by distance to target
    #nkpts = len(dataset.fidxdist)
    if tgt is None:
      x1 = x.copy()
    else:
      d = self.get_fly_dists(x, tgt)
      order = np.argsort(d)
      # x is nkeypoints x d x seqlength x ntgts
      x1 = x[...,order[:self.ntgtsout]]
    
    if returnnames:
      featurenames = self.featurenames.copy()

    # vector from tail to head
    midt = self.seqlength//2
    midhead = x[headidx,:,midt,0]
    if rotate:
      midtail = x[tailidx,:,midt,0]
      cs = midhead-midtail
      z = np.sqrt(np.sum(np.square(cs)))
      cs = cs / z # c[0] = cos, c[1] = sin
      # rotate around origin (0,0)
      #[linex,liney] = R*[x,y]
      #      =   [cos(theta)*x + sin(theta)*y
      #           -sin(theta)*x + cos(theta)*y]
      x2 = cs[0]*x1[coordidx,0,...]+cs[1]*x1[coordidx,1,...]
      x1[coordidx,1,...] = -cs[1]*x1[coordidx,0,...]+cs[0]*x1[coordidx,1,...]
      x1[coordidx,0,...] = x2
      
    if arena:
      xwall,wallfeaturenames = self.circular_arena_wall_features(x1)
      x1 = np.concatenate((x1,xwall),axis=0)
      if returnnames:
        featurenames = featurenames + wallfeaturenames
        
    if translate:
      x1[coordidx,...] = x1[coordidx,...]-midhead.reshape([1,2,1,1])
      
    if returnnames:
      return x1,featurenames
    else:
      return x1
  
  def circular_arena_wall_features(self,x,bodyangle=None,DEBUG=False):
    
    # x is nkeypts x d x seqlength x ntargets
    x = np.array(x,ndmin=4)
    seqlength = x.shape[2]
    ntgts = x.shape[3]
    if self.arena_radius is None:
      print('arena_radius not set')
      raise
    if self.arena_center is None:
      print('arena_center not set')
      raise
    
    # distance to wall: find circler that this point is along
    # compute distance along this circler
    p = x[headidx,...]-self.arena_center.reshape([2,1,1])
    arena_angle = np.arctan2(p[1,...],p[0,...])
    arena_r = np.sqrt(np.sum(np.square(p),axis=0))
    mindwall = self.arena_radius - arena_r
    #maxdwall = arena_r + self.arena_radius
    
    # angle to wall: angle to this closest point on the wall:
    # difference in angle between body orientation and absolute
    # angle to closest point
    if bodyangle is None:
      head = x[headidx,...]
      tail = x[tailidx,...]
      cs = head-tail
      bodyangle = np.arctan2(cs[1,...],cs[0,...])
    cosbodyangle = np.cos(bodyangle)
    sinbodyangle = np.sin(bodyangle)
      
    angle2wall = arena_angle-bodyangle
    cosangle2wall = np.cos(angle2wall)
    sinangle2wall = np.sin(angle2wall)
    # 1 x 2 x seqlength x ntgts
    angle2wall = np.concatenate((cosangle2wall.reshape((1,1,seqlength,ntgts)),sinangle2wall.reshape((1,1,seqlength,ntgts))),axis=1)
    
    # forward distance to wall
    xint1,yint1,xint2,yint2 = circle_line_intersection(p[0,...], p[1,...], cosbodyangle, sinbodyangle, self.arena_radius)
    forwarddist = np.sqrt((xint1-p[0,...])**2+(yint1-p[1,...])**2)
    #backwarddist = np.sqrt((xint2-p[0,...])**2+(yint2-p[1,...])**2)
    # 1 x 2 x seqlength x ntgts
    dist2wall = np.concatenate((mindwall.reshape((1,1,seqlength,ntgts)),forwarddist.reshape((1,1,seqlength,ntgts))),axis=1)
    
    # 1 x 2 x seqlength x ntgts
    #forwardbackwarddistwall = np.concatenate((forwarddist.reshape((1,1,seqlength,ntgts)),backwarddist.reshape((1,1,seqlength,ntgts))),axis=1)

    # sideways distance to wall
    xint1,yint1,xint2,yint2 = circle_line_intersection(p[0,...], p[1,...], sinbodyangle, -cosbodyangle, self.arena_radius)
    sidedist1 = np.sqrt((xint1-p[0,...])**2+(yint1-p[1,...])**2)
    sidedist2 = np.sqrt((xint2-p[0,...])**2+(yint2-p[1,...])**2)
    
    dirwalldist = np.concatenate((sidedist1.reshape((1,1,seqlength,ntgts)),sidedist2.reshape((1,1,seqlength,ntgts))),axis=1)
    #xwall = np.concatenate((dist2wall,angle2wall,forwardbackwarddistwall,dirwalldist),axis=0)
    xwall = np.concatenate((dist2wall,angle2wall,dirwalldist),axis=0)

    wallfeaturenames = [('min_dist2wall','forward_dist2wall'),('cos_angle2wall','sin_angle2wall'),('right_dist2wall','left_dist2wall')]
    #wallfeaturenames = ['dist2wall','angle2wall','forbackdist2wall','sidedist2wall']
    
    if DEBUG:
      theta = np.linspace(-np.pi,np.pi,360)
      midt = seqlength//2
      hkpts,hedges,htxt,fig,ax = self.plot_flies(x[:,:,midt,:],textlabels='id')
      ax.plot(ARENA_RADIUS_MM*np.cos(theta),ARENA_RADIUS_MM*np.sin(theta),'k-',zorder=-10)
      for i in range(ntgts):
        ax.plot(head[0,midt,i]+np.array([0,cosbodyangle[midt,i]])*dist2wall[0,1,midt,i],
                head[1,midt,i]+np.array([0,sinbodyangle[midt,i]])*dist2wall[0,1,midt,i],'c-')
      a = np.arctan2(xwall[1,1,midt,:],xwall[1,0,midt,:])+bodyangle[midt,:]
      
      for i in range(ntgts):
        ax.plot(head[0,midt,i]+np.array([0,cosbodyangle[midt,i]])*xwall[0,1,midt,i],
                head[1,midt,i]+np.array([0,sinbodyangle[midt,i]])*xwall[0,1,midt,i],'c-')
        
      for i in range(ntgts):
        ax.plot(head[0,midt,i]+np.array([0,np.cos(a[i])])*xwall[0,0,midt,i],
                head[1,midt,i]+np.array([0,np.sin(a[i])])*xwall[0,0,midt,i],'m-')
      
      for i in range(ntgts):
        ax.plot(head[0,midt,i]+np.array([0,sinbodyangle[midt,i]])*xwall[2,0,midt,i],
            head[1,midt,i]+np.array([0,-cosbodyangle[midt,i]])*xwall[2,0,midt,i],'g-')
        ax.plot(head[0,midt,i]+np.array([0,-sinbodyangle[midt,i]])*xwall[2,1,midt,i],
            head[1,midt,i]+np.array([0,cosbodyangle[midt,i]])*xwall[2,1,midt,i],'b-')

    return xwall,wallfeaturenames
  
  def select_categories(self,newcategories):
    idx = [self.categorynames.index(c) for c in newcategories]
    if self.yvalues is not None:
      self.yvalues = [self.yvalues[i] for i in idx]
    doremove = np.zeros(self.nseqs,dtype=bool)
    for i in range(self.nseqs):
      seqid = self.seqids[i]
      self.y[seqid] = self.y[seqid][idx,...]
      if np.all(np.isnan(self.y[seqid])):
        doremove[i] = True
    self.removeseqs(doremove)
    self.categorynames = newcategories
    self.ncategories = len(newcategories)
    
    
  def removeseqs(self,doremove):
    idxremove = np.where(doremove)[0]
    if self.preprocessed:
      self.X.delete(idxremove,axis=0)
    else:
      for i in idxremove:
        _ = self.X.pop(self.seqids[i])
    if self.y is not None:
      for i in idxremove:
        _ = self.y.pop(self.seqids[i])
    if self.weights is not None:
      for i in idxremove:
        _ = self.weights.pop(self.seqids[i])
        
    newseqids = list(compress(self.seqids,list(doremove==False)))
    seqtgtidxremove = np.isin(self.idx2seqnum,idxremove)
    self.idx2seqnum = self.idx2seqnum[seqtgtidxremove==False]
    for i in range(len(self.idx2seqnum)):
      self.idx2seqnum[i] = newseqids.index(self.seqids[self.idx2seqnum[i]])
    self.idx2tgt = self.idx2tgt[seqtgtidxremove==False]
    self.seqids = newseqids
    self.nseqs = len(self.seqids)
    self.n = len(self.idx2seqnum)
    return
  
  def getseqx(self,seqi):
    if self.preprocessed:
      return self.X[seqi,...]
    else:
      return self.X[self.seqids[seqi]]
    
  def setseqx(self,seqi,x):
    if self.preprocessed:
      self.X[seqi,...] = x
    else:
      self.X[self.seqids[seqi]] = x
    return
    
  def getnseqxs(self):
    if self.preprocessed:
      return self.n
    else:
      return self.nseqs
  
  """
  Unlabeled dataset:
  x,seqid,seqnum,tgt = data.getitem(idx)
  Labeled dataset:
  x,y,seqid,seqnum,tgt = data.getitem(idx)
  The getitem() function inputs an index between 0 and len(data)-1
  and returns a data example corresponding to a sequence (seqid or seqnum)
  and fly (tgt) corresponding to input idx.
  The output data x is reordered so that the selected fly is first, and
  other flies are ordered by distance to that fly.
  Input:
  idx: Integer between 0 and len(dataset)-1 specifying which sample to select.
  Output:
  x is an ndarray of size nkeypoints x 2 x seql x maxnflies input data sample
  y (only output for labeled dataset) is an ndarray of size
  ncategories x seqlength output data sample for target 0
  w (only output for labeled dataset) is an
  ncategories x seqlength array of weights for each sample and category for target 0
  seqid: 20-character string identifying the selected sequence
  seqnum: integer identifying the selected sequence
  tgt: Integer identifying the selected fly.
  """
  # def getitem(self,idx):
  #   seqnum = self.idx2seqnum[idx]
  #   tgt = self.idx2tgt[idx]
  #   seqid = self.seqids[seqnum]
  #
  #   # sort flies by distance to target
  #   #nkpts = len(dataset.fidxdist)
  #   x = self.X[seqid]
  #   d = self.get_fly_dists(x, tgt)
  #   order = np.argsort(d)
  #   # x is nkeypoints x d x seqlength x ntgts
  #   x = x[...,order[:self.ntgtsout]]
  #
  #   if self.y is None:
  #     return x,seqid,seqnum,tgt
  #
  #   y = self.y[seqid][:,:,tgt]
  #   if self.balance:
  #     w = self.weights[seqid][...,tgt]
  #   else:
  #     w = np.ones(y.shape)
  #   return x,y,w,seqid,seqnum,tgt
  
  def getx(self,idx):
    if self.preprocessed:
      return self.X[idx,...]
    else:
      return self.getitem(idx)['x']
  
  def seqtgt2idx(self,seq=None,seqnum=None,tgt=None):
    if seqnum is None:
      seqnum = self.seqids.index(seq)
    idx = np.where(np.logical_and(self.idx2seqnum==seqnum,self.idx2tgt==tgt))[0]
    assert len(idx) == 1, 'Sanity check: found %d matches for seq %d and tgt %d'%(len(idx),seqnum,tgt)
    if np.isscalar(tgt):
      idx = idx[0]
    return idx
  
  def getitem(self,idx):
    
    seqnum = self.idx2seqnum[idx]
    tgt = self.idx2tgt[idx]
    seqid = self.seqids[seqnum]

    # sort flies by distance to target
    #nkpts = len(dataset.fidxdist)
    retval = {'seqid': seqid, 'seqnum': seqnum, 'tgt': tgt}
    if self.preprocessed:
      retval['x'] = self.X[idx,...]
    else:
      x = self.X[seqid]
      d = self.get_fly_dists(x, tgt)
      order = np.argsort(d)
      # x is nkeypoints x d x seqlength x ntgts
      retval['x'] = x[...,order[:self.ntgtsout]]
    if self.y is not None:
      retval['y'] = self.y[seqid][:,:,tgt]
      if self.balance:
        retval['weights'] = self.weights[seqid][...,tgt]
      else:
        retval['weights'] = np.ones(retval['y'].shape)
    return retval
  
  """
  Unlabeled dataset:
  x = data[idx]
  Labeled dataset:
  x,y = data[idx]
  The __getitem__() function inputs an index between 0 and len(data)-1
  and returns a data example corresponding to a sequence (seqid or seqnum)
  and fly (tgt) corresponding to input idx.
  The output data x (and optionally y) are reordered so that the selected fly
  is first, and other flies are ordered by distance to that fly.
  Input:
  idx: Integer between 0 and len(dataset)-1 specifying which sample to select.
  Output:
  Dictionary with the following entries:
  x is an ndarray of size (nkeypoints*d) x seqlength x ntgts data sample
  y (only output for labeled dataset) is an ndarray of size
  ncategories x seqlength output data sample for target 0
  weights (only output for labeled dataset) is an
  ncategories x seqlength array of weights for each sample and category for target 0
  seqnum: integer identifying the selected sequence
  tgt: Integer identifying the selected fly.
  """
  def __getitem__(self,idx):
    retval = self.getitem(idx)
    retval['x'] = self.reformat_x(retval['x'])
    return retval

  def rotate(self,x,theta,inplace=True):
    # rotate all trajectories around origin
    
    #[linex,liney] = R*[x,y]
    #      =   [cos(theta)*x - sin(theta)*y
    #           sin(theta)*x + cos(theta)*y]
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    if inplace:
      x1 = x
    else:
      x1 = x.copy()

    
    pass

  """
  d = data.get_fly_dists(x,tgt=0)
  Compute the distance between fly tgt and all other flies. This is defined as the
  minimum distance between any pair of the following keypoints:
  'antennae','end_notum_x','end_abdomen'
  at middle frame data.ctrf
  Input:
  x: ndarray of size nkeypoints x d x seqlength x ntgts data sample, sequence of data for all flies
  tgt: (optional) which fly to compute distances to. Default is 0.
  Output:
  d: Array of length nflies with the squared distance to the selected target.
  """
  def get_fly_dists(self, x, tgt=0):
    nkpts = len(self.fidxdist)
    d = np.min(np.sum((x[self.fidxdist,:,self.ctrf,:].reshape((nkpts,1,self.d,self.ntgts))-
                x[self.fidxdist,:,self.ctrf,tgt].reshape((1,nkpts,self.d,1)))**2.,axis=2),
               axis=(0,1))
    return d
  
  def compute_interfly_dists(self,x,tgt=0):
    # x is nkpts x 2 x seql x ntgts
    nkpts = len(self.fidxdist)
    d = np.min(np.sum((x[self.fidxdist,:,:,:].reshape((nkpts,1,x.shape[1],x.shape[2],x.shape[3]))-
                x[self.fidxdist,:,:,tgt].reshape((1,nkpts,x.shape[1],x.shape[2],1)))**2.,axis=2),
               axis=(0,1))
    return d
  
  """
  hkpt,hedge,htxt,fig,ax = data.plot_fly(pose=None, idx=None, f=None,
                                        fig=None, ax=None, kptcolors=None, color=None, name=None,
                                        plotskel=True, plotkpts=True, hedge=None, hkpt=None)
  Visualize the single fly position specified by pose
  Inputs:
  pose: Optional. nfeatures x 2 ndarray. Default is None.
  idx: Data sample index to plot. Only used if pose is not input. Default: None.
  f: Frame of data sample idx to plot. Only used if pose is not input. Default: dataset.ctrf.
  fig: Optional. Handle to figure to plot in. Only used if ax is not specified. Default = None.
  ax: Optional. Handle to axes to plot in. Default = None.
  kptcolors: Optional. Color scheme for each keypoint. Can be a string defining a matplotlib
  colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to 'hsv'.
  Default: None
  color: Optional. Color for edges plotted. If None, it is set to [.6,.6,.6]. efault: None.
  name: Optional. String defining an identifying label for these plots. Default None.
  plotskel: Optional. Whether to plot skeleton edges. Default: True.
  plotkpts: Optional. Whether to plot key points. Default: True.
  hedge: Optional. Handle of edges to update instead of plot new edges. Default: None.
  hkpt: Optional. Handle of keypoints to update instead of plot new key points. Default: None.
  """
  def plot_fly(self, x=None, idx=None, f=None, **kwargs):
    if x is not None:
      return plot_fly(pose=self.unnormalize(x), kptidx=self.keypointidx, skelidx=self.skeleton_edges, **kwargs)
    else:
      assert(idx is not None)
      if f is None:
        f = self.ctrf
      x = self.getitem(idx)
      return plot_fly(pose=self.unnormalize(x['x'][:,:,f, 0]), kptidx=self.keypointidx, skelidx=self.skeleton_edges, **kwargs)
    
  """
  hkpt,hedge,fig,ax = data.plot_flies(poses=None,, idx=None, f=None,
                                      colors=None,kptcolors=None,hedges=None,hkpts=None,
                                      **kwargs)
  Visualize all flies for a single frame specified by poses.
  Inputs:
  poses: Required. nfeatures x 2 x nflies ndarray.
  idx: Data sample index to plot. Only used if pose is not input. Default: None.
  f: Frame of data sample idx to plot. Only used if pose is not input. Default: dataset.ctrf.
  colors: Optional. Color scheme for edges plotted for each fly. Can be a string defining a matplotlib
  colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to the Dark3
  colormap I defined in get_Dark3_cmap(). Default: None.
  kptcolors: Optional. Color scheme for each keypoint. Can be a string defining a matplotlib
  colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to [0,0,0].
  Default: None
  hedges: Optional. List of handles of edges, one per fly, to update instead of plot new edges. Default: None.
  hkpts: Optional. List of handles of keypoints, one per fly,  to update instead of plot new key points.
  Default: None.
  Extra arguments: All other arguments will be passed directly to plot_fly.
  """
  def plot_flies(self, x=None, idx=None, f=None, **kwargs):
    if x is not None:
      return plot_flies(poses=self.unnormalize(x), kptidx=self.keypointidx, skelidx=self.skeleton_edges, **kwargs)
    else:
      assert(idx is not None)
      if f is None:
        f = self.ctrf
      x,_,_,_ = self.get_item(idx=idx)
      return plot_flies(self.unnormalize(x[:, :, f, :]), self.keypointidx, self.skeleton_edges, **kwargs)
  
  def deepcopy(self):
    return copy.deepcopy(self)

# def compute_y_values(dataset,yvalues=None,binary=False):
#
#   if binary:
#     # total number of non-nan examples
#     yvalues = []
#     for c in range(dataset.ncategories):
#       yvalues.append([0,1])
#     return yvalues
#
#   # this will only work for categorical data
#   if yvalues is None:
#     yvalues = [set(),] * dataset.ncategories
#   else:
#     for c in range(dataset.ncategories):
#       if type(yvalues[c]) == list:
#         yvalues[c] = set(yvalues[c])
#
#   for y in dataset.y.values():
#     for c in range(dataset.ncategories):
#       ycurr = y[c,:]
#       yvaluescurr = set(np.unique(ycurr[np.isnan(ycurr)==False]))
#       yvalues[c] = yvalues[c].union(yvaluescurr)
#
#   yvalues = list(map(lambda x: list(x), yvalues))
#   return yvalues

def compute_y_values(y,yvalues=None,binary=False):
  
  y0 = next(iter(y.values()))
  ncategories = y0.shape[0]
  
  if binary:
    # total number of non-nan examples
    yvalues = []
    for c in range(ncategories):
      yvalues.append([0,1])
    return yvalues
  
  # this will only work for categorical data
  if yvalues is None:
    yvalues = [set(),] * ncategories
  else:
    for c in range(ncategories):
      if type(yvalues[c]) == list:
        yvalues[c] = set(yvalues[c])

  for y in y.values():
    for c in range(ncategories):
      ycurr = y[c,:]
      yvaluescurr = set(np.unique(ycurr[np.isnan(ycurr)==False]))
      yvalues[c] = yvalues[c].union(yvaluescurr)
      
  yvalues = list(map(lambda x: list(x), yvalues))
  return yvalues

def compute_class_weights(dataset, restart=None, minweight=.1,maxweight=100.,binary=False):

  dorestart = restart is not None
  if dorestart:
    yvalues = restart['yvalues']
    ycounts = restart['ycounts']
    n = restart['n']
  else:
    ycounts = None
    yvalues = None
    n = 0
  
  # compute all possible y values
  yvalues = dataset.compute_y_values(binary=binary)
  
  # compute frequency of each value
  ycounts,n = dataset.compute_y_counts(yvalues=yvalues,ycounts=ycounts,n=n)
        
  # compute weights
  yweights = []
  for c in range(dataset.ncategories):
    
    # convert dictionary into np array
    ycountscurr = np.zeros(len(yvalues[c]))
    for k,v in ycounts[c].items():
      i = yvalues[c].index(k)
      ycountscurr[i] = v
      
    # what should the total weight be for this category
    w0 = n / dataset.ncategories / np.maximum(1, len(yvalues[c]))
    
    # compute weight to be inversely proportional to ycounts
    w = w0 / ycountscurr
    w = np.minimum(np.maximum(w,minweight),maxweight)
    yweights.append(w)
  
  
  return yweights,yvalues,ycounts,n

def get_ytypes(dataset,ytypes=None):
  # which categories are per sequence, per fly, per frame
  if ytypes is None:
    ytypes = [YTYPES['seq'], ] * dataset.ncategories
  for y in dataset.y.values():
    for c in range(dataset.ncategories):
      if ytypes[c] == YTYPES['frame']:
        continue
      miny = np.min(y[c,...],axis=1)
      maxy = np.min(y[c,...],axis=1)
      if np.any(maxy>miny):
        ytypes[c] = YTYPES['frame']
      elif (ytypes[c] != YTYPES['frame']) and (np.max(maxy) > np.min(y)):
        ytypes[c] = YTYPES['tgt']
  return ytypes

def compute_zscore_stats(dataset,restart=None):
  
  dorestart = restart is not None
  if dorestart:
    mu=restart['mu']
    sig=restart['sig']
    n=restart['n']
    n0 = n
    mu0 = mu.copy()
    musum = mu*n0
  else:
    musum = 0.
    sig = 0.
    n = 0

  for i in range(dataset.getnseqxs()):
    x = dataset.getseqx(i)
    isgood = np.isnan(x) == False
    musum += np.nansum(x, axis=(2, 3))
    n += np.sum(isgood,axis=(2,3))
  mu = musum/n
  mu = mu.reshape((mu.shape[0], mu.shape[1], 1, 1))

  # compute standard deviation
  
  # if restarting, renormalize sig2 around new mu
  # sum^n0( (x-mu)^2 ) = sum^n0( (x-mu0)^2 ) + n0*(mu0-mu)^2
  if dorestart:
    sig = sig + n0*(mu0-mu)**2
  
  for i in range(dataset.getnseqxs()):
    x = dataset.getseqx(i)
    sig += np.nansum((x-mu)**2., axis=(2, 3))
  sig = np.sqrt(sig/n)
  sig = sig.reshape((sig.shape[0], sig.shape[1], 1, 1))
  
  return mu,sig,n

def split_train_holdout(dataset_all,frac_holdout=0.1,sampletype='weighted'):
  
  dataset_holdout = dataset_all.deepcopy()
  dataset_train = dataset_all.deepcopy()
  isholdout,pholdout = select_holdout(dataset_all,frac_holdout,sampletype)
  
  dataset_holdout.select_seq(isholdout)
  dataset_train.select_seq(isholdout==False)
  
  return dataset_train,dataset_holdout,pholdout,isholdout
  
def select_holdout(dataset,frac_holdout,sampletype='weighted',DEBUG=False):
  wseq = np.ones(dataset.nseqs)
  if sampletype=='weighted':
    for i in range(dataset.nseqs):
      id = dataset.seqids[i]
      y = dataset.y[id]
      w = dataset.weights[id]
      istgt = np.all(np.isnan(y),axis=(0,1))==False
      wseq[i] = np.mean(np.sum(w[:,:,istgt],axis=0))
  elif sampletype=='uniform':
    pass # use wseq = 1
  else:
    print('Unknown sampletype %s'%sampletype)
    raise
  pholdout = wseq / np.sum(wseq)
  idxholdout = np.random.choice(np.arange(dataset.nseqs,dtype=int),size=int(np.round(dataset.nseqs*frac_holdout)),
                                replace=False,p=pholdout)
  isholdout = np.zeros(dataset.nseqs,dtype=bool)
  isholdout[idxholdout] = True
  
  if DEBUG:
    # print some info about what was selected
    countsholdout = []
    countstrain = []
    for i in range(len(dataset.yvalues)):
      countsholdout.append(np.zeros(len(dataset.yvalues[i])))
      countstrain.append(np.zeros(len(dataset.yvalues[i])))
    for k in np.where(isholdout)[0]:
      id = dataset.seqids[k]
      y = dataset.y[id]
      for i in range(len(dataset.yvalues)):
        for j in range(len(dataset.yvalues[i])):
          countsholdout[i][j] += np.count_nonzero(y[i,...]==dataset.yvalues[i][j])
    for k in np.where(isholdout==False)[0]:
      id = dataset.seqids[k]
      y = dataset.y[id]
      for i in range(len(dataset.yvalues)):
        for j in range(len(dataset.yvalues[i])):
          countstrain[i][j] += np.count_nonzero(y[i,...]==dataset.yvalues[i][j])
    for i in range(len(dataset.yvalues)):
      print('%s:'%dataset.categorynames[i])
      for j in range(len(dataset.yvalues[i])):
        print('  %d: train: %d,%f holdout: %d,%f'%(dataset.yvalues[i][j],
                                                   countstrain[i][j],countstrain[i][j]/(countstrain[i][j]+countsholdout[i][j]),
                                                   countsholdout[i][j],countsholdout[i][j]/(countstrain[i][j]+countsholdout[i][j])))

  return isholdout,pholdout
  
"""
dark3cm = get_Dark3_cmap()
Returns a new matplotlib colormap based on the Dark2 colormap.
I didn't have quite enough unique colors in Dark2, so I made Dark3 which
is Dark2 followed by all Dark2 colors with the same hue and saturation but
half the brightness.
"""
def get_Dark3_cmap():
  dark2 = list(cm.get_cmap('Dark2').colors)
  dark3 = dark2.copy()
  for c in dark2:
    chsv = colors.rgb_to_hsv(c)
    chsv[2] = chsv[2]/2.
    crgb = colors.hsv_to_rgb(chsv)
    dark3.append(crgb)
  dark3cm = colors.ListedColormap(tuple(dark3))
  return dark3cm

"""
isreal = get_real_flies(x)
Returns which flies in the input ndarray x correspond to real data (are not nan).
Input:
x: ndarray of arbitrary dimensions, as long as the tgtdim-dimension corresponds to targets.
tgtdim: dimension corresponding to targets. default: -1 (last)
"""
def get_real_flies(x,tgtdim=-1):
  # x is ... x ntgts
  dims = list(range(x.ndim))
  if tgtdim < 0:
    tgtdim = x.ndim+tgtdim
  dims.remove(tgtdim)
    
  isreal = np.all(np.isnan(x),axis=tuple(dims))==False
  return isreal

"""
fig,ax,isnewaxis = set_fig_ax(fig=None,ax=None)
Create new figure and/or axes if those are not input.
Returns the handles to those figures and axes.
isnewaxis is whether a new set of axes was created.
"""
def set_fig_ax(fig=None,ax=None):
  if ax is None:
    if fig is None:
      fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    isnewaxis = True
  else:
    isnewaxis = False
  return fig, ax, isnewaxis

def get_n_colors_from_colormap(colormapname,n):
  # Get the colormap from matplotlib
  colormap = plt.cm.get_cmap(colormapname)

  # Generate a range of values from 0 to 1
  values = np.linspace(0, 1, n)

  # Get 'n' colors from the colormap
  colors = colormap(values)

  return colors

"""
hkpt,hedge,htext,fig,ax = plot_fly(pose=None, kptidx=None, skelidx=None,
                                  fig=None, ax=None, kptcolors=None, color=None, name=None,
                                  plotskel=True, plotkpts=True, hedge=None, hkpt=None)
Visualize the single fly position specified by pose
Inputs:
pose: Required. nfeatures x 2 ndarray.
kptidx: Required. 1-dimensional array specifying which keypoints to plot
skelidx: Required. nedges x 2 ndarray specifying which keypoints to connect with edges
fig: Optional. Handle to figure to plot in. Only used if ax is not specified. Default = None.
If None, a new figure is created.
ax: Optional. Handle to axes to plot in. Default = None. If None, new axes are created.
kptcolors: Optional. Color scheme for each keypoint. Can be a string defining a matplotlib
colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to 'hsv'.
Default: None
color: Optional. Color for edges plotted. If None, it is set to [.6,.6,.6]. efault: None.
name: Optional. String defining an identifying label for these plots. Default None.
plotskel: Optional. Whether to plot skeleton edges. Default: True.
plotkpts: Optional. Whether to plot key points. Default: True.
hedge: Optional. Handle of edges to update instead of plot new edges. Default: None.
hkpt: Optional. Handle of keypoints to update instead of plot new key points. Default: None.
"""
def plot_fly(pose=None, kptidx=keypointidx, skelidx=skeleton_edges, fig=None, ax=None, kptcolors=None, color=None, name=None,
             plotskel=True, plotkpts=True, hedge=None, hkpt=None, textlabels=None, htxt=None, kpt_ms=6, skel_lw=1,
             kpt_alpha=1.,skel_alpha=1., skeledgecolors=None, kpt_marker='.'):
  # plot_fly(x,fig=None,ax=None,kptcolors=None):
  # x is nfeatures x 2
  assert(pose is not None)
  assert(kptidx is not None)
  assert(skelidx is not None)

  fig,ax,isnewaxis = set_fig_ax(fig=fig,ax=ax)
  isreal = get_real_flies(pose[:,:,np.newaxis])
  
  hkpts = None
  hedges = None
  htxt = None
  if plotkpts:
    if isreal:
      xc = pose[kptidx,0]
      yc = pose[kptidx,1]
    else:
      xc = []
      yc = []
    if hkpt is None:
      if kptcolors is None:
        kptcolors = 'hsv'
      if (type(kptcolors) == list or type(kptcolors) == np.ndarray) and len(kptcolors) == 3:
        kptname = 'keypoints'
        if name is not None:
          kptname = name + ' ' + kptname
        hkpt = ax.plot(xc,yc,kpt_marker,color=kptcolors,label=kptname,zorder=10,ms=kpt_ms,alpha=kpt_alpha)[0]
      else:
        if type(kptcolors) == str:
          kptcolors = plt.get_cmap(kptcolors)
        hkpt = ax.scatter(xc,yc,c=np.arange(len(kptidx)),marker=kpt_marker,cmap=kptcolors,s=kpt_ms,alpha=kpt_alpha,zorder=10)
    else:
      if type(hkpt) == matplotlib.lines.Line2D:
        hkpt.set_data(xc,yc)
      else:
        hkpt.set_offsets(np.column_stack((xc,yc)))
        
  if textlabels is not None:
    xc = pose[kptidx,0]
    yc = pose[kptidx,1]
    xc[np.isnan(xc)] = 0.
    yc[np.isnan(yc)] = 0.
    if textlabels == 'keypoints':
      if htxt is None:
        htxt = []
        for i in range(len(xc)):
          htxt.append(plt.text(xc[i],yc[i],'%d: %s'%(i+1,keypointnames[i]),horizontalalignment='left',visible=isreal))
      else:
        for i in range(len(xc)):
          htxt[i].set_visible(isreal)
          htxt[i].set_data(xc[i],yc[i])
    else:
      if htxt is None:
        htxt = plt.text(xc[0],yc[0],textlabels,horizontalalignment='left',visible=isreal)
      else:
        htxt.set_visible(isreal)
        htxt.set_data(xc[0],yc[0])
        
  if plotskel:
    nedges = skelidx.shape[0]
    if isreal:
      segments = pose[skelidx,:]
      #xc = np.concatenate((pose[skelidx,0],np.zeros((nedges,1))+np.nan),axis=1)
      #yc = np.concatenate((pose[skelidx,1],np.zeros((nedges,1))+np.nan),axis=1)
    else:
      segments = np.zeros((nedges,2))+np.nan
      #xc = np.array([])
      #yc = np.array([])
    if hedge is None:
      if color is None:
        color = [.6,.6,.6]
      if type(color) == str:
        color = get_n_colors_from_colormap(color,nedges)
      
      hedge = collections.LineCollection(pose[skelidx,:],colors=color,linewidths=skel_lw,alpha=skel_alpha)
      ax.add_collection(hedge)
      #edgename = 'skeleton'
      #if name is not None:
      #  edgename = name + ' ' + edgename
      #hedge = ax.plot(xc.flatten(),yc.flatten(),'-',color=color,label=edgename,zorder=0,lw=skel_lw,alpha=skel_alpha)[0]
    else:
      hedge.set_segments(segments)
      #hedge.set_data(xc.flatten(),yc.flatten())

  if isnewaxis:
    ax.axis('equal')

  return hkpt,hedge,htxt,fig,ax
 
"""
hkpt,hedge,fig,ax = plot_flies(poses=None, kptidx=None, skelidx=None,
                               colors=None,kptcolors=None,hedges=None,hkpts=None,
                               **kwargs)
Visualize all flies for a single frame specified by poses.
Inputs:
poses: Required. nfeatures x 2 x nflies ndarray.
kptidx: Required. 1-dimensional array specifying which keypoints to plot
skelidx: Required. nedges x 2 ndarray specifying which keypoints to connect with edges
colors: Optional. Color scheme for edges plotted for each fly. Can be a string defining a matplotlib
colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to the Dark3
colormap I defined in get_Dark3_cmap(). Default: None.
kptcolors: Optional. Color scheme for each keypoint. Can be a string defining a matplotlib
colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to [0,0,0].
Default: None
hedges: Optional. List of handles of edges, one per fly, to update instead of plot new edges. Default: None.
hkpts: Optional. List of handles of keypoints, one per fly,  to update instead of plot new key points.
Default: None.
Extra arguments: All other arguments will be passed directly to plot_fly.
"""
def plot_flies(poses=None,fig=None,ax=None,colors=None,kptcolors=None,hedges=None,hkpts=None,htxt=None,textlabels=None,skeledgecolors=None,**kwargs):

  fig,ax,isnewaxis = set_fig_ax(fig=fig,ax=ax)
  if colors is None and skeledgecolors is None:
    colors = get_Dark3_cmap()
  if kptcolors is None:
    kptcolors = [0,0,0]
  nflies = poses.shape[-1]
  if colors is not None and (not (type(colors) == list or type(colors) == np.ndarray)):
    if type(colors) == str:
      cmap = cm.get_cmap(colors)
    else:
      cmap = colors
    colors = cmap(np.linspace(0.,1.,nflies))
    
  if hedges is None:
    hedges = [None,]*nflies
  if hkpts is None:
    hkpts = [None,]*nflies
  if htxt is None:
    htxt = [None,]*nflies

  textlabels1 = textlabels
  
  for fly in range(nflies):
    if not (textlabels is None or (textlabels == 'keypoints')):
      textlabels1 = '%d'%fly
    if skeledgecolors is not None:
      colorcurr = skeledgecolors
    else:
      colorcurr = colors[fly,...]
    hkpts[fly],hedges[fly],htxt[fly],fig,ax = plot_fly(poses[...,fly],fig=fig,ax=ax,color=colorcurr,
                                               kptcolors=kptcolors,hedge=hedges[fly],hkpt=hkpts[fly],
                                               htxt=htxt,textlabels=textlabels1,**kwargs)
    
  if isnewaxis:
    ax.axis('equal')
  
  return hkpts,hedges,htxt,fig,ax

def plot_annotations(y=None,yvalues=None,names=None,fig=None,ax=None,
                     patchcolors=None,binarycolors=False,color0=None,axcolor=[.7,.7,.7]):
  
  y = y.copy()
  fig,ax,isnewaxis = set_fig_ax(fig=fig,ax=ax)
  ncategories = y.shape[0]
  seql = y.shape[1]
  if yvalues is None:
    yvalues = compute_y_values({'0': y})
  maxnvalues = np.max(np.array(list(map(lambda x: len(x),yvalues))))

  if binarycolors:
    if color0 is None:
      color0 = [1,1,1]
    if patchcolors is None:
      patchcolors = 'tab10'
    if type(patchcolors) == str:
      patchcolors = cm.get_cmap(patchcolors)
    if not (type(patchcolors) == list or type(patchcolors) == np.ndarray):
      patchcolors = patchcolors(np.linspace(0.,1.,ncategories))
    colors1 = patchcolors.copy()
  else:
  
    if patchcolors is None:
      patchcolors = 'tab10'
    if type(patchcolors) == str:
      patchcolors = cm.get_cmap(patchcolors)
    if not (type(patchcolors) == list or type(patchcolors) == np.ndarray):
      patchcolors = patchcolors(np.linspace(0.,1.,maxnvalues))
  
  hpatch = []
  for c in range(ncategories):
    
    values = np.asarray(yvalues[c])
    if values.dtype == bool:
      values = values.astype(int)
    if y.dtype == bool:
      y = y.astype(int)
    novalue = np.min(values)-1
    y[np.isnan(y)] = novalue
    
    ycurr = y[c,:]
    ycurr = np.insert(ycurr,0,novalue)
    ycurr = np.append(ycurr,novalue)
    hpatchcurr = []
    if binarycolors:
      patchcolors = np.vstack((color0,colors1[c,:3]))
    for i in range(len(values)):
      patchcolor = patchcolors[i,:]
      v = values[i]
      t0s = np.where(np.logical_and(ycurr[:-1] != v,ycurr[1:] == v))[0]-.5
      t1s = np.where(np.logical_and(ycurr[:-1] == v,ycurr[1:] != v))[0]-.5
      # xy should be nrects x 4 x 2
      xy = np.zeros((len(t0s),4,2))
      xy[:,0,0] = t0s
      xy[:,1,0] = t0s
      xy[:,2,0] = t1s
      xy[:,3,0] = t1s
      xy[:,0,1] = c-.5
      xy[:,1,1] = c+.5
      xy[:,2,1] = c+.5
      xy[:,3,1] = c-.5
      catpath = matplotlib.path.Path.make_compound_path_from_polys(xy)
      patch = matplotlib.patches.PathPatch(catpath,facecolor=patchcolor,edgecolor=patchcolor)
      hpatchcurr.append(ax.add_patch(patch))
    hpatch.append(hpatchcurr)
  
  ax.set_xlim([-1,seql])
  ax.set_ylim([-1,ncategories])
  ax.set_facecolor(axcolor)
  if names is not None and len(names) == ncategories:
    ax.set_yticks(np.arange(0,ncategories))
    ax.set_yticklabels(names)
  return hpatch,fig,ax

def plot_arena(ax=None):
  if ax is None:
    ax = plt.gca()
  theta = np.linspace(0,2*np.pi,360)
  h = ax.plot(ARENA_RADIUS_MM*np.cos(theta),ARENA_RADIUS_MM*np.sin(theta),'k-',zorder=-10)
  return h


"""
animate_pose_sequence(seq=None, kptidx=None, skelidx=None,
                      start_frame=0,stop_frame=None,skip=1,
                      fig=None,ax=None,savefile=None,
                      **kwargs)
Visualize all flies for the input sequence of frames seq.
Inputs:
seq: Required. nfeatures x 2 x seql x nflies ndarray.
kptidx: Required. 1-dimensional array specifying which keypoints to plot
skelidx: Required. nedges x 2 ndarray specifying which keypoints to connect with edges
start_frame: Which frame of the sequence to start plotting at. Default: 0.
stop_frame: Which frame of the sequence to end plotting on. Default: None. If None, the
sequence length (seq.shape[0]) is used.
skip: How many frames to skip between plotting. Default: 1.
fig: Optional. Handle to figure to plot in. Only used if ax is not specified. Default = None.
If None, a new figure is created.
ax: Optional. Handle to axes to plot in. Default = None. If None, new axes are created.
savefile: Optional. Name of video file to save animation to. If None, animation is displayed
instead of saved.
Extra arguments: All other arguments will be passed directly to plot_flies.
"""
def animate_pose_sequence(seq=None,start_frame=0,stop_frame=None,skip=1,
                          fig=None,ax=None,
                          annotation_sequence=None,
                          savefile=None,
                          **kwargs):
    
  if stop_frame is None:
    stop_frame = seq.shape[2]
  fig,ax,isnewaxis = set_fig_ax(fig=fig,ax=ax)
  
  isreal = get_real_flies(seq)
  seq = seq[...,isreal]

  # plot the arena wall
  theta = np.linspace(0,2*np.pi,360)
  ax.plot(ARENA_RADIUS_MM*np.cos(theta),ARENA_RADIUS_MM*np.sin(theta),'k-',zorder=-10)
  minv = -ARENA_RADIUS_MM*1.01
  maxv = ARENA_RADIUS_MM*1.01
  
  # first frame
  f = start_frame
  h = {}
  h['kpts'],h['edges'],h['txt'],fig,ax = plot_flies(poses=seq[...,f,:],fig=fig,ax=ax,**kwargs)
  h['frame'] = plt.text(-ARENA_RADIUS_MM*.99,ARENA_RADIUS_MM*.99,'Frame %d (%.2f s)'%(f,float(f)/FPS),
                        horizontalalignment='left',verticalalignment='top')
  ax.set_xlim(minv,maxv)
  ax.set_ylim(minv,maxv)
  ax.axis('off')
  ax.axis('equal')
  fig.tight_layout(pad=0)
  #ax.margins(0)
  
  def update(f):
    nonlocal fig
    nonlocal ax
    nonlocal h
    h['kpts'],h['edges'],h['txt'],fig,ax = plot_flies(poses=seq[...,f,:],fig=fig,ax=ax,
                                             hedges=h['edges'],hkpts=h['kpts'],htxt=h['txt'],**kwargs)
    h['frame'].set_text('Frame %d (%.2f s)'%(f,float(f)/FPS))
    return h['edges']+h['kpts']
  
  ani = animation.FuncAnimation(fig, update, frames=np.arange(start_frame,stop_frame,skip,dtype=int))
  if savefile is not None:
    print('Saving animation to file %s...'%savefile)
    writer = animation.PillowWriter(fps=30)
    ani.save(savefile,writer=writer)
    print('Finished writing.')
  else:
    plt.show()
   
"""
DatasetTest()
Driver function to test loading and accessing data.
"""
def DatasetTest():
  
  # file containing the data
  xfile = os.path.join(datadir,'Xusertrain_seq.npy')
  assert(os.path.exists(xfile))
  
  # load data and initialize Dataset object
  datatrain = FlyDataset(xfile)
  #datatrain.plot_fly(idx=456)
  
  # get one sample sequence.
  data = datatrain.getitem(234)
  # which sequence and fly did this correspond to?
  print('x.shape = {shape}, seqid = {seqid}, seqnum = {seqnum}, tgt = {tgt}'.format(shape=str(data['x'].shape),seqid=data['seqid'],seqnum=data['seqnum'],tgt=data['tgt']))
  # make sure the flies are sorted by distance to the first fly.
  print('These distances should be increasing, with nans (if any) at the end): %s' % str(datatrain.get_fly_dists(data['x'])))
  #hkpts,hedges,fig,ax = datatrain.plot_flies(x[:,:,datatrain.ctrf,:],colors='hsv')
  
  # Set savefile if we want to save a video
  #savefile = 'seq.gif'
  savefile=None

  # plot that sequence
  animate_pose_sequence(seq=datatrain.unnormalize(data['x']),kptidx=datatrain.keypointidx,skelidx=datatrain.skeleton_edges,savefile=savefile)

def MakeSequenceMovie():
  # file containing the data
  xfile = os.path.join(datadir,'Xusertrain_seq.npy')
  assert(os.path.exists(xfile))
  yfile = os.path.join(datadir,'yusertrain_seq.npy')
  assert(os.path.exists(yfile))
  
  movietype = 'aIPg'
  
  # Set savefile if we want to save a video
  #savefile = 'aIPg_seq2.gif'
  savefile=None
  
  # load data and initialize Dataset object
  datatrain = FlyDataset(xfile,yfile)
  #datatrain.plot_fly(idx=456)
  
  if movietype == 'random':
  
    # get one sample sequence.
    idx = 234
  
  elif movietype == 'aIPg':
  
    cat = 'aIPgpublished3_newstim_offvsstrong'
    cati = datatrain.categorynames.index(cat)
    beh = 'perframe_aggression'
    behi = datatrain.categorynames.index(beh)
    
    n = np.zeros(len(datatrain))
    
    for i in range(len(datatrain)):
      sample = datatrain[i]
      #n[j] = np.count_nonzero(np.logical_and(sample['y'][cati]==1,sample['y'][behi]==1))
      n[i] = np.count_nonzero(sample['y'][cati]==1)

    idx = np.argmax(n)
    print('Found %d frames with both aggression and aIPg strong'%n[idx])
    
  elif movietype == 'pC1d':
  
    cat = 'pC1dpublished1_newstim_offvsstrong'
    cati = datatrain.categorynames.index(cat)
    beh = 'perframe_aggression'
    behi = datatrain.categorynames.index(beh)
    
    n = np.zeros(len(datatrain))
    
    for i in range(len(datatrain)):
      sample = datatrain[i]
      #n[j] = np.count_nonzero(np.logical_and(sample['y'][cati]==1,sample['y'][behi]==1))
      n[i] = np.count_nonzero(sample['y'][cati]==1)
      
    idx = np.argmax(n)
    print('Found %d frames with both aggression and pC1d strong'%n[idx])
    
  else:
    print('unknown movie type %s'%movietype)
    raise
    
  data = datatrain.getitem(idx)
  # which sequence and fly did this correspond to?
  print('x.shape = {shape}, seqid = {seqid}, seqnum = {seqnum}, tgt = {tgt}'.format(shape=str(data['x'].shape),seqid=data['seqid'],seqnum=data['seqnum'],tgt=data['tgt']))
  # make sure the flies are sorted by distance to the first fly.
  print('These distances should be increasing, with nans (if any) at the end): %s' % str(datatrain.get_fly_dists(data['x'])))
  #hkpts,hedges,fig,ax = datatrain.plot_flies(x[:,:,datatrain.ctrf,:],colors='hsv')

  # plot that sequence
  animate_pose_sequence(seq=datatrain.unnormalize(data['x']),kptidx=datatrain.keypointidx,skelidx=datatrain.skeleton_edges,savefile=savefile)


def CountLabelsPerFly():
  
  tasks = [
    'perframe_aggression',
    'perframe_chase',
    'perframe_courtship',
    'perframe_highfence2',
    'perframe_wingext',
    'perframe_wingflick'
  ]
  yfiles = ['ytesttrain_seq.npy','ytest1_seq.npy','ytest2_seq.npy']
  counts = {}
  doskip = np.zeros((len(tasks),11))
  for yfile in yfiles:
    data = np.load(os.path.join(datadir,yfile),allow_pickle=True).item()
    taskidx = np.array(list(map(lambda x: data['vocabulary'].index(x), tasks)))
    counts[yfile] = np.zeros((len(tasks),11,2))
    for seqid,y in data['annotations'].items():
      for v in range(2):
        counts[yfile][:,:,v] += np.sum(y[taskidx,:,:]==v,axis=1)
    for taski in range(len(taskidx)):
      for fly in range(11):
        print('%s, %s, %s: %d, %d'%(yfile,tasks[taski],fly,counts[yfile][taski,fly,0],counts[yfile][taski,fly,1]))
  doskip = np.any(counts['ytesttrain_seq.npy']==0,axis=2)
  for yfile in yfiles:
    doskip = np.logical_or(doskip,np.all(counts[yfile]==0,axis=2))
  
  print('skip:')
  for taski in range(len(tasks)):
    for fly in range(11):
      if doskip[taski,fly]:
        print('%s, fly %d'%(tasks[taski],fly))
  print('keep:')
  for taski in range(len(tasks)):
    for fly in range(11):
      if not doskip[taski,fly]:
        print('%s, fly %d'%(tasks[taski],fly))
  
  return

def DebugPreprocess():
  # file containing the data
  xfile = os.path.join(datadir,'Xusertrain_seq.npy')
  assert(os.path.exists(xfile))
  yfile = os.path.join(datadir,'yusertrain_seq.npy')
  assert(os.path.exists(yfile))
  
  tgt = 0
  
  # load data and initialize Dataset object
  datatrain = FlyDataset(xfile,yfile,arena_center=np.array([0,0]),arena_radius=ARENA_RADIUS_MM)
  datatrain.preprocess_all(rotate=True,arena=True,translate=True)
  x1 = datatrain.getx(100)
  procfeaturenames = datatrain.featurenames
  #x1,procfeaturenames = datatrain.preprocess(x,tgt,rotate=True,arena=True,translate=True)
  
  # x is nkeypoints x d x seqlength x ntgts
  midt = datatrain.seqlength//2
  nflies = x1.shape[-1]
  cmap = get_Dark3_cmap()
  colors = cmap(np.linspace(0.,1.,nflies))

  head = x1[headidx,:,midt,:]
  tail = x1[tailidx,:,midt,:]
  cs = head-tail
  bodyangle = np.arctan2(cs[1,...],cs[0,...])
  cosbodyangle = np.cos(bodyangle)
  sinbodyangle = np.sin(bodyangle)
  #arena_center = -x[headidx,:,midt,tgt]
  #theta = np.linspace(0,2*np.pi,360)
  
  hkpts,hedges,htxt,fig,ax = datatrain.plot_flies(x1[:,:,midt,:],colors=colors,textlabels='id')

  # arena wall
  #ax.plot(ARENA_RADIUS_MM*np.cos(theta)+arena_center[0],ARENA_RADIUS_MM*np.sin(theta)+arena_center[1],'k-',zorder=-10)
  
  procfeaturenames1 = list(map(lambda x: x[0],procfeaturenames))
  fidx = procfeaturenames1.index('min_dist2wall')
  dist2wall = x1[fidx,:,midt,:]
  fidx = procfeaturenames1.index('cos_angle2wall')
  angle2wall = np.arctan2(x1[fidx,1,midt,:],x1[fidx,0,midt,:])+bodyangle
  fidx = procfeaturenames1.index('right_dist2wall')
  sidedist2wall = x1[fidx,:,midt,:]

  # arena angle and distance to wall
  for i in range(nflies):
    ax.plot(head[0,i]+dist2wall[0,i]*np.array([0,np.cos(angle2wall[i])]),
            head[1,i]+dist2wall[0,i]*np.array([0,np.sin(angle2wall[i])]),'m-')
  
  # forward distance to wall
  for i in range(nflies):
    ax.plot(head[0,i]+np.array([0,cosbodyangle[i]])*dist2wall[1,i],
            head[1,i]+np.array([0,sinbodyangle[i]])*dist2wall[1,i],'c-')
    
  # sideways distance to wall
  for i in range(nflies):
    ax.plot(head[0,i]+np.array([0,sinbodyangle[i]])*sidedist2wall[0,i],
            head[1,i]+np.array([0,-cosbodyangle[i]])*sidedist2wall[0,i],'g-')
    ax.plot(head[0,i]+np.array([0,-sinbodyangle[i]])*sidedist2wall[1,i],
            head[1,i]+np.array([0,cosbodyangle[i]])*sidedist2wall[1,i],'b-')
    
  return
  

def circle_fit_error(mu,x,forcemu=None):

  if np.ndim(mu) == 1:
    mu = mu[:,np.newaxis]

  x = np.atleast_3d(x)
  m = mu.shape[1]
  n = x.shape[1]
  # mu is 2 x 1 x m
  mu = mu[:,np.newaxis,:]
  # x is 2 x n x 1

  if forcemu is not None:
    # forcemu is of length 2
    mu1 = np.tile(forcemu.reshape((2,1,1)),(1,1,m))
    mu1[np.isnan(forcemu),...] = mu
    mu = mu1

  # dx is 2 x n x m
  dx = x - mu
  # n x m
  z = np.sqrt(np.sum(dx**2.,axis=0))
  # m
  rho = np.mean(z,axis=0)
  # n x m
  theta = np.arctan2(dx[1,...]/z,dx[0,...]/z)
  err = np.nanmean( (np.cos(theta)*rho.reshape((1,m)) - dx[0,...])**2. + \
                    (np.sin(theta)*rho.reshape((1,m)) - dx[1,...])**2.,axis=0)
  return np.sqrt(err)

import scipy.optimize as optimize
def fit_circle(x,mu0=None,forcemu=None,algorithm='sample',lb=None,ub=None,nsamples=100):

  if mu0 is None:
    mu0 = np.mean(x,axis=1)

  if forcemu is None:
    doforce = np.zeros(2,dtype=bool)
    forcemu = mu0
  else:
    doforce = np.isnan(forcemu)==False

  if algorithm == 'leastsq':
    res = optimize.least_squares(lambda mu: circle_fit_error(mu,x,forcemu),mu0[doforce==False])
    mufit = res.x
  elif algorithm == 'sample':
    assert lb is not None
    assert ub is not None

    if not any(doforce):
      nsamples1 = int(np.ceil(np.sqrt(nsamples)))
      mux_try,muy_try = np.meshgrid(np.linspace(lb[0],ub[0],nsamples1),np.linspace(lb[1],ub[1],nsamples1))
      mu_try = np.vstack((mux_try.flatten(),muy_try.flatten()))
    else:
      mu_try = np.linspace(lb[doforce==False],ub[doforce==False],nsamples).reshape([1,nsamples])

    err = circle_fit_error(mu_try,x,forcemu=forcemu)
    i = np.argmin(err)
    mufit = mu_try[:,i]
    res = {'mu': mu_try, 'err': err, 'minerr': err[i]}

  mu = forcemu
  mu[doforce==False] = mufit
  dx=x-mu[:,np.newaxis]
  z=np.sqrt(np.sum(dx**2.,axis=0))
  rho=np.mean(z)

  return mu,rho,res

def circle_rotation(x,mu):
  dx = x - mu[:,np.newaxis]
  theta = np.arctan2(dx[1,:],dx[0,:])
  return theta

def reconstruct_headpos(peye,headangle,meanheadwidth,meanheadheight):
  deye = np.vstack((np.cos(headangle),np.sin(headangle)))*meanheadwidth/2.
  dant = np.vstack((-np.sin(headangle),np.cos(headangle)))*meanheadheight
  p = {'left_eye': peye - deye, 'right_eye': peye + deye, 'antenna': peye + dant}
  return p

def rotate_2d_points(X,theta):
  costheta = np.cos(theta)
  sintheta = np.sin(theta)
  xr = X[:,0,...]*costheta+X[:,1,...]*sintheta
  yr = -X[:,0,...]*sintheta+X[:,1,...]*costheta
  Xr = np.concatenate((xr[:,np.newaxis,...],yr[:,np.newaxis,...]),axis=1)
  return Xr

def compute_scale_perfly(Xcurr):

  if np.ndim(Xcurr) >= 3:
    T = Xcurr.shape[2]
  else:
    T = 1
  if np.ndim(Xcurr) >= 4:
    nflies = Xcurr.shape[3]
  else:
    nflies = 1

  scale_perfly = np.zeros((len(scalenames),nflies))
  rfthorax = Xcurr[keypointnames.index('right_front_thorax'),...]
  lfthorax = Xcurr[keypointnames.index('left_front_thorax'),...]
  scale_perfly[scalenames.index('thorax_width'),:] = np.nanmedian(np.sqrt(np.sum((rfthorax-lfthorax)**2.,axis=0)),axis=0)
  scale_perfly[scalenames.index('std_thorax_width'),:]=np.nanstd(np.sqrt(np.sum((rfthorax-lfthorax)**2.,axis=0)),axis=0)
  fthorax=(rfthorax+lfthorax)/2.
  bthorax = Xcurr[keypointnames.index('base_thorax'),...]
  midthorax = (fthorax+bthorax)/2.
  scale_perfly[scalenames.index('thorax_length'),:] = np.nanmedian(np.sqrt(np.sum((fthorax-bthorax)**2,axis=0)),axis=0)
  scale_perfly[scalenames.index('std_thorax_length'),:]=np.nanstd(np.sqrt(np.sum((fthorax-bthorax)**2,axis=0)),axis=0)
  abdomen = Xcurr[keypointnames.index('tip_abdomen'),...]
  scale_perfly[scalenames.index('abdomen_length'),:]=np.nanmedian(np.sqrt(np.sum((bthorax-abdomen)**2.,axis=0)),axis=0)
  scale_perfly[scalenames.index('std_abdomen_length'),:]=np.nanstd(np.sqrt(np.sum((bthorax-abdomen)**2.,axis=0)),axis=0)
  lwing = Xcurr[keypointnames.index('wing_left'),...]
  rwing = Xcurr[keypointnames.index('wing_right'),...]
  scale_perfly[scalenames.index('wing_length'),:]=np.nanmedian(np.sqrt(np.sum(np.concatenate(((lwing-midthorax)**2.,(rwing-midthorax)**2.),axis=1),axis=0)),axis=0)
  scale_perfly[scalenames.index('std_wing_length'),:]=np.nanstd(np.sqrt(np.sum(np.concatenate(((lwing-midthorax)**2.,(rwing-midthorax)**2.),axis=1),axis=0)),axis=0)

  reye = Xcurr[keypointnames.index('right_eye'),...]
  leye = Xcurr[keypointnames.index('left_eye'),...]
  eye = (leye+reye)/2.
  ant = Xcurr[keypointnames.index('antennae_midpoint'),...]
  headwidth = np.sqrt(np.sum((reye-leye)**2.,axis=0))
  scale_perfly[scalenames.index('head_width'),...] = np.nanmedian(headwidth)
  scale_perfly[scalenames.index('std_head_width'),...] = np.nanstd(headwidth)
  headheight = np.sqrt(np.sum((eye-ant)**2.,axis=0))
  scale_perfly[scalenames.index('head_height'),...] = np.nanmedian(headheight)
  scale_perfly[scalenames.index('std_head_height'),...] = np.nanstd(headheight)

  return scale_perfly

def angledist2xy(origin,angle,dist):
  u = np.vstack((np.cos(angle[np.newaxis,...]),np.sin(angle[np.newaxis,...])))
  d = u*dist[np.newaxis,...]
  xy = origin + d
  return xy

def feat2kp(Xfeat,scale_perfly,flyid=None):
  # Xfeat is nfeatures x T x nflies
  ndim = np.ndim(Xfeat)
  if ndim >= 2:
    T = Xfeat.shape[1]
  else:
    T = 1
  if ndim >= 3:
    nflies = Xfeat.shape[2]
  else:
    nflies = 1
  if np.ndim(scale_perfly)==1:
    scale_perfly = scale_perfly[:,None]
      
  if flyid is None:
    assert(scale_perfly.shape[1]==nflies)
    flyid=np.tile(np.arange(nflies,dtype=int)[np.newaxis,:],(T,1))

  Xfeat = Xfeat.reshape((Xfeat.shape[0],T,nflies))

  porigin = Xfeat[[posenames.index('thorax_front_x'),posenames.index('thorax_front_y')],...]
  thorax_theta = Xfeat[posenames.index('orientation'),...]

  # Xkpn will be normalized by the following translation and rotation
  Xkpn = np.zeros((len(keypointnames),2,T,nflies))
  Xkpn[:] = np.nan

  # thorax
  thorax_width = scale_perfly[scalenames.index('thorax_width'),flyid].reshape((T,nflies))
  thorax_length = scale_perfly[scalenames.index('thorax_length'),flyid].reshape((T,nflies))
  Xkpn[keypointnames.index('left_front_thorax'),0,...] = -thorax_width/2.
  Xkpn[keypointnames.index('left_front_thorax'),1,...] = 0.
  Xkpn[keypointnames.index('right_front_thorax'),0,...] = thorax_width/2.
  Xkpn[keypointnames.index('right_front_thorax'),1,...] = 0.
  Xkpn[keypointnames.index('base_thorax'),0,...] = 0.
  Xkpn[keypointnames.index('base_thorax'),1,...] = -thorax_length

  # head
  bhead = Xfeat[[posenames.index('head_base_x'),posenames.index('head_base_y')],...]
  headangle = Xfeat[posenames.index('head_angle'),...]+np.pi/2.
  headwidth = scale_perfly[scalenames.index('head_width'),flyid].reshape((T,nflies))
  headheight = scale_perfly[scalenames.index('head_height'),flyid].reshape((T,nflies))
  cosha = np.cos(headangle-np.pi/2.)
  sinha = np.sin(headangle-np.pi/2.)
  leye = bhead.copy()
  leye[0,...] -= headwidth/2.*cosha
  leye[1,...] -= headwidth/2.*sinha
  reye = bhead.copy()
  reye[0,...] += headwidth/2.*cosha
  reye[1,...] += headwidth/2.*sinha
  Xkpn[keypointnames.index('left_eye'),...] = leye
  Xkpn[keypointnames.index('right_eye'),...] = reye
  Xkpn[keypointnames.index('antennae_midpoint'),...] = angledist2xy(bhead,headangle,headheight)

  # abdomen
  pthorax = np.zeros((2,T,nflies))
  pthorax[1,...] = -thorax_length
  abdomenangle = Xfeat[posenames.index('abdomen_angle'),...]-np.pi/2.
  abdomendist = scale_perfly[scalenames.index('abdomen_length'),flyid].reshape((T,nflies))
  Xkpn[keypointnames.index('tip_abdomen'),...] = angledist2xy(pthorax,abdomenangle,abdomendist)

  # front legs
  legangle = np.pi-Xfeat[posenames.index('left_front_leg_tip_angle'),...]
  legdist = Xfeat[posenames.index('left_front_leg_tip_dist'),...]
  Xkpn[keypointnames.index('left_front_leg_tip'),...] = angledist2xy(np.zeros((2,T,nflies)),legangle,legdist)

  legangle = Xfeat[posenames.index('right_front_leg_tip_angle'),...]
  legdist = Xfeat[posenames.index('right_front_leg_tip_dist'),...]
  Xkpn[keypointnames.index('right_front_leg_tip'),...] = angledist2xy(np.zeros((2,T,nflies)),legangle,legdist)

  # middle leg femur base
  pmidthorax = np.zeros((2,T,nflies))
  pmidthorax[1,...] = -thorax_length/2.

  lfemurbaseangle = np.pi-Xfeat[posenames.index('left_middle_femur_base_angle'),...]
  legdist = Xfeat[posenames.index('left_middle_femur_base_dist'),...]
  lfemurbase = angledist2xy(pmidthorax,lfemurbaseangle,legdist)
  Xkpn[keypointnames.index('left_middle_femur_base'),...] = lfemurbase

  rfemurbaseangle = Xfeat[posenames.index('right_middle_femur_base_angle'),...]
  legdist = Xfeat[posenames.index('right_middle_femur_base_dist'),...]
  rfemurbase = angledist2xy(pmidthorax,rfemurbaseangle,legdist)
  Xkpn[keypointnames.index('right_middle_femur_base'),...] = rfemurbase

  # middle leg femur tibia joint

  # lftangleoffset = np.pi-lftangle-(np.pi-lbaseangle)
  #                = lbaseangle - lftangle
  # lftangle = lbaseangle - lftangleoffset
  lftangleoffset = Xfeat[posenames.index('left_middle_femur_tibia_joint_angle'),...]
  lftangle = lfemurbaseangle-lftangleoffset
  legdist=Xfeat[posenames.index('left_middle_femur_tibia_joint_dist'),...]
  lftjoint = angledist2xy(lfemurbase,lftangle,legdist)
  Xkpn[keypointnames.index('left_middle_femur_tibia_joint'),...] = lftjoint

  rftangleoffset = Xfeat[posenames.index('right_middle_femur_tibia_joint_angle'),...]
  rftangle = rfemurbaseangle+rftangleoffset
  legdist=Xfeat[posenames.index('right_middle_femur_tibia_joint_dist'),...]
  rftjoint = angledist2xy(rfemurbase,rftangle,legdist)
  Xkpn[keypointnames.index('right_middle_femur_tibia_joint'),...] = rftjoint

  # middle leg tip
  ltipoffset = Xfeat[posenames.index('left_middle_leg_tip_angle'),...]
  ltipangle = lftangle-ltipoffset
  legdist=Xfeat[posenames.index('left_middle_leg_tip_dist'),...]
  ltip = angledist2xy(lftjoint,ltipangle,legdist)
  Xkpn[keypointnames.index('left_middle_leg_tip'),...] = ltip

  rtipoffset = Xfeat[posenames.index('right_middle_leg_tip_angle'),...]
  rtipangle = rftangle+rtipoffset
  legdist=Xfeat[posenames.index('right_middle_leg_tip_dist'),...]
  rtip = angledist2xy(rftjoint,rtipangle,legdist)
  Xkpn[keypointnames.index('right_middle_leg_tip'),...] = rtip

  # back leg
  legangle = np.pi-Xfeat[posenames.index('left_back_leg_tip_angle'),...]
  legdist = Xfeat[posenames.index('left_back_leg_tip_dist'),...]
  Xkpn[keypointnames.index('left_back_leg_tip'),...] = angledist2xy(pthorax,legangle,legdist)

  legangle = Xfeat[posenames.index('right_back_leg_tip_angle'),...]
  legdist = Xfeat[posenames.index('right_back_leg_tip_dist'),...]
  Xkpn[keypointnames.index('right_back_leg_tip'),...] = angledist2xy(pthorax,legangle,legdist)

  wingangle = np.pi+Xfeat[posenames.index('left_wing_angle'),...]
  wingdist = scale_perfly[scalenames.index('wing_length'),flyid].reshape((T,nflies))
  Xkpn[keypointnames.index('wing_left'),...] = angledist2xy(pmidthorax,wingangle,wingdist)

  wingangle = -Xfeat[posenames.index('right_wing_angle'),...]
  Xkpn[keypointnames.index('wing_right'),...] = angledist2xy(pmidthorax,wingangle,wingdist)

  Xkp = rotate_2d_points(Xkpn,-thorax_theta)+porigin[np.newaxis,...]

  return Xkp

def body_centric_kp(Xkp):

  ndim = np.ndim(Xkp)
  if ndim >= 3:
    T = Xkp.shape[2]
  else:
    T = 1
  if ndim >= 4:
    nflies = Xkp.shape[3]
  else:
    nflies = 1

  sz = Xkp.shape[:2]
  Xkp = Xkp.reshape(sz+(T,nflies))

  bthorax = Xkp[keypointnames.index('base_thorax'),...]
  lthorax = Xkp[keypointnames.index('left_front_thorax'),...]
  rthorax = Xkp[keypointnames.index('right_front_thorax'),...]
  fthorax = (lthorax+rthorax)/2.
  d = fthorax - bthorax

  # center on mean point of "shoulders", rotate so that thorax points "up"
  thorax_theta = modrange(np.arctan2(d[1,...],d[0,...])-np.pi/2.,-np.pi,np.pi)
  porigin = fthorax
  Xn = rotate_2d_points(Xkp-porigin[np.newaxis,...],thorax_theta)

  return Xn,porigin,thorax_theta

def kp2feat(Xkp,scale_perfly=None,flyid=None,return_scale=False):
  """
  Xkp is nkeypoints x 2 [x T [x nflies]]
  """

  ndim = np.ndim(Xkp)
  if ndim >= 3:
    T = Xkp.shape[2]
  else:
    T = 1
  if ndim >= 4:
    nflies = Xkp.shape[3]
  else:
    nflies = 1

  sz = Xkp.shape[:2]
  Xkp = Xkp.reshape(sz+(T,nflies))

  assert((flyid is None) or scale_perfly is not None)
  #assert((flyid is None) == (scale_perfly is None))

  if scale_perfly is None:
    scale_perfly = compute_scale_perfly(Xkp)
    flyid=np.tile(np.arange(nflies,dtype=int)[np.newaxis,:],(T,1))

  Xn,fthorax,thorax_theta = body_centric_kp(Xkp)

  Xfeat = np.zeros((len(posenames),T,nflies))
  Xfeat[posenames.index('thorax_front_x'),...] = fthorax[0,...]
  Xfeat[posenames.index('thorax_front_y'),...]= fthorax[1,...]
  Xfeat[posenames.index('orientation'),...] = thorax_theta

  # thorax_length can be a scalar or an array of size T x nflies
  thorax_length = scale_perfly[scalenames.index('thorax_length'),flyid]
  if np.isscalar(thorax_length) or thorax_length.size == 1:
    pass
  elif thorax_length.size == nflies:
    thorax_length = thorax_length.reshape((1,nflies))
  elif thorax_length.size == T*nflies:    
    thorax_length = thorax_length.reshape((T,nflies))
  else:
    raise ValueError(f'thorax_length size {thorax_length.size} is unexpected')
  pthoraxbase = np.zeros((2,T,nflies))
  pthoraxbase[1,...] = -thorax_length
  d = Xn[keypointnames.index('tip_abdomen'),...]-pthoraxbase
  Xfeat[posenames.index('abdomen_angle'),...] = modrange(np.arctan2(d[1,...],d[0,...])+np.pi/2,-np.pi,np.pi)

  ant = Xn[keypointnames.index('antennae_midpoint'),...]
  eye = (Xn[keypointnames.index('right_eye'),...]+Xn[keypointnames.index('left_eye'),...])/2.

  # represent with the midpoint of the eyes and the angle from here to the antenna
  Xfeat[posenames.index('head_base_x'),...] = eye[0,...]
  Xfeat[posenames.index('head_base_y'),...] = eye[1,...]
  Xfeat[posenames.index('head_angle'),...] = np.arctan2(ant[1,:]-eye[1,:],ant[0,:]-eye[0,:])-np.pi/2.

  # parameterize the front leg tips based on angle and distance from origin (shoulder mid-point)
  d = Xn[keypointnames.index('left_front_leg_tip'),...]
  Xfeat[posenames.index('left_front_leg_tip_dist'),...] = np.sqrt(np.sum(d**2,axis=0))
  Xfeat[posenames.index('left_front_leg_tip_angle'),...] = modrange(np.pi-np.arctan2(d[1,:],d[0,:]),-np.pi,np.pi)
  d = Xn[keypointnames.index('right_front_leg_tip'),...]
  Xfeat[posenames.index('right_front_leg_tip_dist'),...] = np.sqrt(np.sum(d**2,axis=0))
  Xfeat[posenames.index('right_front_leg_tip_angle'),...] = np.arctan2(d[1,:],d[0,:])

  # for middle leg femur base, compute angle around and distance from
  # halfway between the thorax base and thorax front
  pmidthorax = np.zeros((2,T,nflies))
  pmidthorax[1,...] = -thorax_length/2.
  d = Xn[keypointnames.index('left_middle_femur_base'),...]-pmidthorax
  Xfeat[posenames.index('left_middle_femur_base_dist'),...]=np.sqrt(np.sum(d**2,axis=0))
  Xfeat[posenames.index('left_middle_femur_base_angle'),...] = modrange(np.pi-np.arctan2(d[1,:],d[0,:]),-np.pi,np.pi)
  d = Xn[keypointnames.index('right_middle_femur_base'),...]-pmidthorax
  Xfeat[posenames.index('right_middle_femur_base_dist'),...]=np.sqrt(np.sum(d**2,axis=0))
  Xfeat[posenames.index('right_middle_femur_base_angle'),...] = np.arctan2(d[1,:],d[0,:])

  # femur tibia joint is represented as distance from and angle around femur base
  d = Xn[keypointnames.index('left_middle_femur_tibia_joint'),...]-\
          Xn[keypointnames.index('left_middle_femur_base'),...]
  Xfeat[posenames.index('left_middle_femur_tibia_joint_dist'),...]=np.sqrt(np.sum(d**2,axis=0))
  left_angle = modrange(np.pi-np.arctan2(d[1,:],d[0,:]),-np.pi,np.pi)
  Xfeat[posenames.index('left_middle_femur_tibia_joint_angle'),...] = \
    modrange(left_angle-Xfeat[posenames.index('left_middle_femur_base_angle'),...],-np.pi,np.pi)

  d = Xn[keypointnames.index('right_middle_femur_tibia_joint'),...]-\
          Xn[keypointnames.index('right_middle_femur_base'),...]
  Xfeat[posenames.index('right_middle_femur_tibia_joint_dist'),...]=np.sqrt(np.sum(d**2,axis=0))
  right_angle = np.arctan2(d[1,:],d[0,:])
  Xfeat[posenames.index('right_middle_femur_tibia_joint_angle'),...] = \
    modrange(right_angle-Xfeat[posenames.index('right_middle_femur_base_angle'),...],-np.pi,np.pi)

  # middle leg tip is represented as distance from and angle around femur tibia joint
  d = Xn[keypointnames.index('left_middle_leg_tip'),...]-\
          Xn[keypointnames.index('left_middle_femur_tibia_joint'),...]
  Xfeat[posenames.index('left_middle_leg_tip_dist'),...]=np.sqrt(np.sum(d**2,axis=0))
  Xfeat[posenames.index('left_middle_leg_tip_angle'),...] = \
    modrange(np.pi-np.arctan2(d[1,:],d[0,:])-left_angle,-np.pi,np.pi)

  d = Xn[keypointnames.index('right_middle_leg_tip'),...]-\
          Xn[keypointnames.index('right_middle_femur_tibia_joint'),...]
  Xfeat[posenames.index('right_middle_leg_tip_dist'),...]=np.sqrt(np.sum(d**2,axis=0))
  Xfeat[posenames.index('right_middle_leg_tip_angle'),...] = \
    modrange(np.arctan2(d[1,:],d[0,:])-right_angle,-np.pi,np.pi)

  # for the back legs, use the thorax base as the origin
  d = Xn[keypointnames.index('left_back_leg_tip'),...]-pthoraxbase
  Xfeat[posenames.index('left_back_leg_tip_dist'),...] = np.sqrt(np.sum(d**2,axis=0))
  Xfeat[posenames.index('left_back_leg_tip_angle'),...] = modrange(np.pi-np.arctan2(d[1,:],d[0,:]),-np.pi,np.pi)
  d = Xn[keypointnames.index('right_back_leg_tip'),...]-pthoraxbase
  Xfeat[posenames.index('right_back_leg_tip_dist'),...] = np.sqrt(np.sum(d**2,axis=0))
  Xfeat[posenames.index('right_back_leg_tip_angle'),...] = np.arctan2(d[1,:],d[0,:])

  # wings relative to thorax middle
  d = Xn[keypointnames.index('wing_left'),...]-pmidthorax
  Xfeat[posenames.index('left_wing_angle'),...] = modrange(-np.pi+np.arctan2(d[1,:],d[0,:]),-np.pi,np.pi)
  d = Xn[keypointnames.index('wing_right'),...]
  Xfeat[posenames.index('right_wing_angle'),...] = -np.arctan2(d[1,:],d[0,:])

  if return_scale:
    return Xfeat,scale_perfly,flyid
  else:
    return Xfeat
  
def get_flip_idx():

  isright = np.array([re.search('right',kpn) is not None for kpn in keypointnames])
  flipidx = np.arange(len(keypointnames),dtype=int)
  idxright = np.nonzero(isright)[0]
  for ir in idxright:
    kpnr = keypointnames[ir]
    kpnl = kpnr.replace('right','left')
    il = keypointnames.index(kpnl)
    flipidx[ir] = il
    flipidx[il] = ir
  
  return flipidx

def flip_flies(X,arena_center=[0,0],flipdim=0):
  flipX = X.copy()
  flipidx = get_flip_idx()
  for i in range(len(flipidx)):
    flipX[i,flipdim,...] = arena_center[flipdim]-X[flipidx[i],flipdim,...]
    flipX[i,1-flipdim,...] = X[flipidx[i],1-flipdim,...]

  return flipX

# def explore_pose(X,scale_perfly,flyid,all_dataset):
#
#   pb = X[keypointnames.index('base_thorax'),...]
#   pl = X[keypointnames.index('left_front_thorax'),...]
#   pr = X[keypointnames.index('right_front_thorax'),...]
#   pf = (pl+pr)/2.
#   d = pf - pb
#
#   # center on mean point of "shoulders", rotate so that thorax points "up"
#   thorax_theta = np.arctan2(d[1,...],d[0,...])-np.pi/2.
#   porigin = (pl+pr)/2.
#
#   Xn = rotate_2d_points(X-porigin[np.newaxis,...],thorax_theta)
#
#   reX = {}
#   reX['thorax_front'] = pf
#   reX['orientation'] = thorax_theta
#   reX['thorax_width'] = np.sqrt(np.sum((pr-pl)**2,axis=0))
#   reX['thorax_length'] = np.sqrt(np.sum(d**2,axis=0))
#
#   #R = np.array([[np.cos(thorax_theta[0]),-np.sin(thorax_theta[0])],[np.sin(thorax_theta[0]),np.cos(thorax_theta[0])]])
#   #Xn = (X-porigin)@R
#
#   # eyelen = np.sqrt(np.sum((Xn[keypointnames.index('right_eye'),...]-\
#   #                          Xn[keypointnames.index('left_eye'),...])**2.,axis=0))
#   # torsolen = np.sqrt(np.sum((pf-pb)**2.,axis=0))
#   # torsowid = np.sqrt(np.sum((pr-pl)**2.,axis=0))
#
#   # examine fly head positions
#
#   pant = Xn[keypointnames.index('antennae_midpoint'),...]
#   peye = (Xn[keypointnames.index('right_eye'),...]+\
#            Xn[keypointnames.index('left_eye'),...])/2.
#
#   headangle = np.arctan2(pant[1,:]-peye[1,:],pant[0,:]-peye[0,:])
#   delta_headangle = np.pi/8
#   sample_headangle = np.linspace(np.pi/2-delta_headangle,np.pi/2+delta_headangle,200)
#
#   xhead=np.vstack(list(map(lambda x:np.expand_dims(x,axis=0),
#                            [Xn[keypointnames.index('antennae_midpoint'),...],
#                             Xn[keypointnames.index('right_eye'),...],
#                             Xn[keypointnames.index('left_eye'),...]])))
#
#   cmap=cm.plasma
#   plt.figure()
#   rtmp = .2
#
#   for i in range(len(sample_headangle)):
#     j = np.argmin(np.abs(sample_headangle[i]-headangle))
#     headanglecurr = headangle[j]
#     color = cmap(np.minimum(1,np.maximum(0,(headanglecurr-sample_headangle[0])/(2*delta_headangle))))
#     if i == 0 or jprev != j:
#       plt.plot(np.concatenate((xhead[:,0,j],[xhead[0,0,j],])),
#                np.concatenate((xhead[:,1,j],[xhead[0,1,j],])),'.-',color=color,lw=1)
#       plt.plot([peye[0,j]-np.cos(headanglecurr)*rtmp,pant[0,j]+np.cos(headanglecurr)*rtmp],
#                [peye[1,j]-np.sin(headanglecurr)*rtmp,pant[1,j]+np.sin(headanglecurr)*rtmp],'-',color=color,lw=.5)
#
#     jprev = j
#   plt.axis('equal')
#
#   # based on this plot, i think it makes most sense to represent with the midpoint of the eyes
#   # and the angle from here to the antenna, phead and headangle above
#   # It looks like head size and shape is fairly stereotyped, except when the flies are doing
#   # weird stuff, so let's compute the average head shape. should be symmetrical!
#   headwidth = np.sqrt(np.sum((Xn[keypointnames.index('right_eye'),...]-\
#                               Xn[keypointnames.index('left_eye'),...])**2.,axis=0))
#   meanheadwidth = np.nanmedian(headwidth)
#   headheight = np.sqrt(np.sum((peye-pant)**2.,axis=0))
#   meanheadheight = np.nanmedian(headheight)
#   reX['head_base'] = (Xn[keypointnames.index('right_eye'),...]+\
#                       Xn[keypointnames.index('left_eye'),...])/2.
#
#   reX['head_angle'] = np.arctan2(Xn[keypointnames.index('antennae_midpoint'),1,:]-reX['head_base'][1,:],
#                                  Xn[keypointnames.index('antennae_midpoint'),0,:]-reX['head_base'][0,:])-np.pi/2.
#
#   frontlegtip = np.concatenate((Xn[keypointnames.index('right_front_leg_tip'),...],
#                                 np.array([-1,1]).reshape((2,1))*Xn[keypointnames.index('left_front_leg_tip'),...]),axis=1)
#
#   plt.figure()
#   flyid2=np.tile(flyid,2)
#   idx=np.random.permutation(frontlegtip.shape[1])
#   plt.scatter(frontlegtip[0,idx],frontlegtip[1,idx],marker='.',s=.2,c=flyid2[idx],cmap=cm.hsv,alpha=.05)
#   all_dataset.plot_fly(Xn[:,:,0],ax=plt.gca())
#   plt.axis('equal')
#
#   forcemu = np.zeros(2)
#   forcemu[:] = np.nan
#   lb = np.zeros(2)
#   ub = np.zeros(2)
#   lb[0] = 0
#   ub[0] = np.mean(Xn[keypointnames.index('right_front_thorax'),0,:])
#   lb[1] = np.mean(Xn[keypointnames.index('base_thorax'),1,:])
#   ub[1] = np.mean(Xn[keypointnames.index('right_front_thorax'),1,:])
#   mufrontleg,rhofrontleg,res = fit_circle(frontlegtip,forcemu=forcemu,lb=lb,ub=ub,nsamples=1000,algorithm='leastsq')
#
#   # based on this plot and analysis, i think it makes most sense to fully parameterize the front leg tips
#   # I think it makes most sense to use angle and distance from origin (shoulder mid-point)
#   pcurr = Xn[keypointnames.index('left_front_leg_tip'),...]
#   reX['left_front_leg_tip_dist'] = np.sqrt(np.sum(pcurr**2,axis=0))
#   reX['left_front_leg_tip_angle'] = modrange(np.pi-np.arctan2(pcurr[1,:],pcurr[0,:]),-np.pi,np.pi)
#   pcurr = Xn[keypointnames.index('right_front_leg_tip'),...]
#   reX['right_front_leg_tip_dist'] = np.sqrt(np.sum(pcurr**2,axis=0))
#   reX['right_front_leg_tip_angle'] = np.arctan2(pcurr[1,:],pcurr[0,:])
#
#   # for middle legs, I'm using halfway between the thorax base and thorax front as the base
#   # each subsequent point is based on the point on the leg
#   pmidthorax = np.vstack((np.zeros(X.shape[2]),-scale_perfly['thorax_length'][flyid]/2.))
#   pcurr = Xn[keypointnames.index('left_middle_femur_base'),...]-pmidthorax
#   reX['left_middle_femur_base_dist']=np.sqrt(np.sum(pcurr**2,axis=0))
#   reX['left_middle_femur_base_angle'] = modrange(np.pi-np.arctan2(pcurr[1,:],pcurr[0,:]),-np.pi,np.pi)
#   pcurr = Xn[keypointnames.index('right_middle_femur_base'),...]-pmidthorax
#   reX['right_middle_femur_base_dist']=np.sqrt(np.sum(pcurr**2,axis=0))
#   reX['right_middle_femur_base_angle'] = np.arctan2(pcurr[1,:],pcurr[0,:])
#
#   pcurr = Xn[keypointnames.index('left_middle_femur_tibia_joint'),...]-\
#           Xn[keypointnames.index('left_middle_femur_base'),...]
#   reX['left_middle_femur_tibia_joint_dist']=np.sqrt(np.sum(pcurr**2,axis=0))
#   left_angle = modrange(np.pi-np.arctan2(pcurr[1,:],pcurr[0,:]),-np.pi,np.pi)
#   reX['left_middle_femur_tibia_joint_angle'] = modrange(left_angle-reX['left_middle_femur_base_angle'],-np.pi,np.pi)
#
#   pcurr = Xn[keypointnames.index('right_middle_femur_tibia_joint'),...]-\
#           Xn[keypointnames.index('right_middle_femur_base'),...]
#   reX['right_middle_femur_tibia_joint_dist']=np.sqrt(np.sum(pcurr**2,axis=0))
#   right_angle = np.arctan2(pcurr[1,:],pcurr[0,:])
#   reX['right_middle_femur_tibia_joint_angle'] = modrange(right_angle-reX['right_middle_femur_base_angle'],-np.pi,np.pi)
#
#   pcurr = Xn[keypointnames.index('left_middle_leg_tip'),...]-\
#           Xn[keypointnames.index('left_middle_femur_tibia_joint'),...]
#   reX['left_middle_leg_tip_dist']=np.sqrt(np.sum(pcurr**2,axis=0))
#   reX['left_middle_leg_tip_angle'] = modrange(np.pi-np.arctan2(pcurr[1,:],pcurr[0,:])-left_angle,-np.pi,np.pi)
#
#   pcurr = Xn[keypointnames.index('right_middle_leg_tip'),...]-\
#           Xn[keypointnames.index('right_middle_femur_tibia_joint'),...]
#   reX['right_middle_leg_tip_dist']=np.sqrt(np.sum(pcurr**2,axis=0))
#   reX['right_middle_leg_tip_angle'] = modrange(np.arctan2(pcurr[1,:],pcurr[0,:])-right_angle,-np.pi,np.pi)
#
#   # for the back legs, use the thorax base as the origin
#   pthoraxbase = np.vstack((np.zeros(X.shape[2]),-scale_perfly['thorax_length'][flyid]))
#   pcurr = Xn[keypointnames.index('left_back_leg_tip'),...]-pthoraxbase
#   reX['left_back_leg_tip_dist']=np.sqrt(np.sum(pcurr**2,axis=0))
#   reX['left_back_leg_tip_angle'] = modrange(np.pi-np.arctan2(pcurr[1,:],pcurr[0,:]),-np.pi,np.pi)
#   pcurr = Xn[keypointnames.index('right_back_leg_tip'),...]-pthoraxbase
#   reX['right_back_leg_tip_dist']=np.sqrt(np.sum(pcurr**2,axis=0))
#   reX['right_back_leg_tip_angle'] = np.arctan2(pcurr[1,:],pcurr[0,:])
#
#   # plot wings relative to origin
#   pcurr = Xn[keypointnames.index('wing_left'),...]
#   reX['left_wing_angle'] = modrange(np.pi-np.arctan2(pcurr[1,:],pcurr[0,:]),-np.pi,np.pi)
#   pcurr = Xn[keypointnames.index('wing_right'),...]
#   reX['left_wing_angle'] = np.arctan2(pcurr[1,:],pcurr[0,:])
#
#   all_dataset


if __name__ == "__main__":
  DebugPreprocess()
  