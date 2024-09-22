import numpy as np
import torch
import copy
import typing

from flyllm.config import featglobal, featrelative, featangle, posenames, keypointnames
from apf.utils import modrange, rotate_2d_points
from flyllm.features import (
    compute_features,
    split_features,
    get_sensory_feature_shapes,
    relfeatidx_to_cossinidx,
    ravel_label_index,
    unravel_label_index,
    feat2kp,
    kp2feat
)
from apf.pose import AgentExample, PoseLabels, ObservationInputs

if typing.TYPE_CHECKING:
    from flyllm.dataset import FlyLLMDataset


# helper functions
def modernize_fly_params(params):
    if 'tspred_global' in params:
        nfeatures = len(posenames)
        tspred = [None,]*nfeatures
        isdct = np.zeros(nfeatures, dtype=bool)
        for i in range(nfeatures):
            tspred[i] = [1,]
            if featrelative[i] == False:
                tspred[i] = params['tspred_global'].copy()
            elif params['dct_m'] is not None:
                tspred[i] = np.arange(params['ntspred_relative'])
                isdct[i] = True
        params['tspred'] = tspred
        params['isdct'] = isdct
    return params

def remove_implicit_params(params):
    if 'tspred' in params:
        del params['tspred']
    if 'isdct' in params:
        del params['isdct']
    return params

class FlyObservationInputs(ObservationInputs):
    """
    FlyObservationInputs
    Class for handling observations/inputs to network
    Represents the observation inputs for a fly for multiple time points. 
    Can be used with batches of observations. 

    Main properties:
    input: ( pre_sz x ) ntimepoints x d_input. ndarray with the observations of the fly at each time point
    metadata: dictionary with metadata about which fly and video frames the observations were derived from
    d_input: number of observation features
    ntimepoints: number of time points
    pre_sz: size of the input, not empty when storing a batch
    
    Main methods:
    
    __init__: Constructor for initializing from a training example or from keypoints. 
    get_inputs(zscored=False, makecopy=True): Returns the inputs, optionally un-zscoring. 
    get_split_inputs(zscored=False, makecopy=True): Returns the inputs split into different types of features.
    get_inputs_type(type, zscored=False, makecopy=True): Returns inputs of a specific type.
    get_train_inputs(input_labels=None, do_add_noise=False, labels=None): Returns the inputs for training, optionally adding noise.
    
    set_inputs(input, zscored=False, ts=None): Sets the inputs. 
    set_inputs_from_keypoints(Xkp, fly, scale=None, ts=None): Sets the inputs from keypoints.
    """

    def __init__(self, example_in=None, Xkp=None, agent=0, scale=None, dataset=None, dozscore=False, npad=None, **kwargs):
        """
        Constructor for initializing from an example or from keypoints.

        To initialize from an example computed with features.compute_features or a training example, pass in example_in:
        example_in: dictionary with the example. This can be the output of compute_features or FlyExample.get_train_example(). 
            Required fields:
            'input': ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the fly at each time point
            Optional fields:
            'input_init': ndarray of size (pre_sz x ) 1 x d_input with the observations of the fly on the 
            first frame of the sequence. This should be part of a training example in which the first frame
            has been cropped from input for a causal network. 
            'metadata': dictionary with metadata about which fly and video frames the observations were derived from

        To initialize from keypoints, pass in the following:
        Xkp: ndarray of size (pre_sz x ) ntimepoints x nfeatures x 2 with the keypoints for all flies. 
        fly: index of the main fly.
        scale: scale parameters for converting from keypoints to features.
        
        Optional parameters:
        dataset: FlyMLMDataset object. Used to get parameters for computing features, etc.
        dozscore: Whether to z-score the inputs. Only used if example_in is input. Set this to true if the example input
        should be z-scored, i.e. it is not already z-scored. Default is False. 
        npad: Number of frames to crop from the end of the sequence when computing features. Only used if Xkp is input. 
        This is used to manually set the number of frames to crop from the end of the sequence. This parameter may be
        obsolete. 
        
        Optional parameters defined by the dataset configuration, input to set_params():
            zscore_params
            do_input_labels
            starttoff
            flatten_labels
            flatten_obs
            discreteidx
            tspred_global
            discrete_tspred
            ntspred_relative
            discretize_params
            is_velocity
            simplify_out
            simplify_in
            flatten_obs_idx
            dct_m
            idct_m        
        TODO- put these all in a dict. 
        """

        super().__init__(example_in=example_in, Xkp=Xkp, agent=agent, scale=scale, 
                         dataset=dataset, dozscore=dozscore, npad=npad, **kwargs)

        # indices for splitting observation features by type
        self._sensory_feature_idx, self._sensory_feature_szs = \
            get_sensory_feature_shapes(simplify=self._simplify_in)

        return

    def get_compute_features_params(self):
        """
        get_compute_features_params()
        Returns the parameters for compute_features, used to compute inputs from keypoints.
        """
        cfparams = super().get_compute_features_params()
        if hasattr(self, '_simplify_in'):
            cfparams['simplify_in'] = self._simplify_in
        if hasattr(self, '_simplify_out'):
            cfparams['simplify_out'] = self._simplify_out
        if hasattr(self, '_tspred_global'):
            cfparams['tspred_global'] = self._tspred_global
        if hasattr(self, '_is_velocity'):
            cfparams['compute_pose_vel'] = self._is_velocity
        return cfparams

    def get_split_inputs(self, **kwargs):
        """
        get_split_inputs(zscored=False, makecopy=True)
        Returns a dict with the inputs split into different types of features.
        Optional arguments:
        zscored: Whether the inputs should be z-scored. Default is False, i.e. the non-zscored inputs are returned.
        makecopy: Whether to make a copy of the inputs. Default is True.
        Splitting is done by the split_features function from features.py. Currently returns:
            'pose': (pre_sz x ) ntimepoints x nfeatures_relative
            'wall_touch': (pre_sz x ) ntimepoints x len(config.SENSORYPARAMS['touch_kpnames']) 
            'otherflies_vision': (pre_sz x ) ntimepoints x config.SENSORYPARAMS['n_oma']
            'otherflies_touch': (pre_sz x ) ntimepoints x (len(config.SENSORYPARAMS['touch_kpnames']) * len(config.SENSORYPARAMS['touch_other_kpnames']) )
        """
        input = self.get_inputs(**kwargs)
        input = split_features(input)
        return input

    def get_inputs_type(self, type, **kwargs):
        """
        get_inputs_type(type, zscored=False, makecopy=True)
        Returns the inputs of a specific type, ndarray of size (pre_sz x ) ntimepoints x d_input_type(type)
        Inputs:
        type: string, type of feature to return. Options are 'pose', 'wall_touch', 'otherflies_vision', 'otherflies_touch'
        Optional arguments:
        zscored: Whether the inputs should be z-scored. Default is False, i.e. the non-zscored inputs are returned.
        makecopy: Whether to make a copy of the inputs. Default is True.
        Sizes of outputs, given type:
            'pose': (pre_sz x ) ntimepoints x nfeatures_relative
            'wall_touch': (pre_sz x ) ntimepoints x len(config.SENSORYPARAMS['touch_kpnames']) 
            'otherflies_vision': (pre_sz x ) ntimepoints x config.SENSORYPARAMS['n_oma']
            'otherflies_touch': (pre_sz x ) ntimepoints x (len(config.SENSORYPARAMS['touch_kpnames']) * len(config.SENSORYPARAMS['touch_other_kpnames']) )
        """
        input = self.get_split_inputs(**kwargs)
        return input[type]

    def set_inputs_from_keypoints(self, Xkp, fly, scale, ts=None, npad=None):
        """
        set_inputs_from_keypoints(Xkp, fly, scale=None, ts=None, npad=None)
        Sets the inputs from keypoints. This calls compute_features on the keypoints to compute the sensory features, then
        stores these with self.set_inputs(zscored=False,ts=ts). 
        Inputs:
        Xkp: ndarray of keypoints for all flies and time points, size (pre_sz x ) nkeypoints x 2 x ntimepoints x nflies
        fly: index of the main fly
        scale: scale parameters for converting from keypoints to features
        Optional:
        ts: Time points to set the inputs. If None, the inputs are set for all time points.
        npad: Number of frames to crop from the end of the sequence when computing features.        
        """
        example = compute_features(Xkp, flynum=fly, scale_perfly=scale, **self.get_compute_features_params(), npad=npad, compute_labels=False)
        input = example['input']
        self.set_inputs(input, zscored=False, ts=ts)
        return

class FlyPoseLabels(PoseLabels):
    """
    FlyPoseLabels
    Class for handling pose labels for a fly for multiple time points. Can be used with batches.
    
    The data are stored in labels_raw, which is the format that will be used for training and prediction. 
    This is the 'labels' output of compute_features, with the following additions:
    zscoring: if zscore_params is not None, then the labels will be z-scored.
    discretization: if discretize_params is not None, then features in discrete_idx will be discretized. 
    The continuous features are stored in labels_raw['continuous'] and the discrete features are stored in
    labels_raw['discrete']. If available, the continuous versions of the discrete features are stored in
    labels_raw['todiscretize'] when we want to invert feature computations. 
    init_pose stores the initial pose of the fly. Since some of the features represent velocities and/or
    are in egocentric coordinate systems, we will need to integrate to invert the feature computation, and
    the start of integration is init_pose. 
    The PoseLabels class allows access to the following representations of the data:
    - labels_raw: representation used for training/prediction, made up of continuous and discrete labels (get_raw_labels())
    - multi: continuous version of the labels, with discrete features converted to continuous values. (get_multi())
      This conversion can be done in multiple ways, depending on the parameters:
      -- use_todiscretize=True: use the continuous versions of the discrete features in labels_raw['todiscretize']
      -- nsamples > 1: sample from the continuous distribution of the discrete features.
      -- nsamples = 0: use the mean of the continuous distribution of the discrete features.
    - un-zscored multi: multi before z-scoring (zscored=False)
    - multi_idct: if dct_m is not None, then multi_idct applies the inverse dct transform to relative features. 
    - un-zscored multi_idct: multi_idct before z-scoring (zscored=False)
    - next frame pose features, with angles represented as cos/sin pairs if applicable (get_nextcossin)
    - next frame pose features (get_next_pose), with angles represented in radians (get_next_pose)
    - next frame keypoints (get_next_keypoints)

    Main properties:
    labels_raw: Dictionary with the raw labels. This will have the following keys:
    'continuous': ndarray of size (pre_sz x ) ntimepoints x d_continuous with the continuous pose for the fly
    'discrete': ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the binned pose for the fly
    'todiscretize': ndarray of size (pre_sz x ) ntimepoints x d_discrete with the continuous versions of the discrete pose.
        
    Main methods:

    __init__: Constructor for initializing from an example or from keypoints.
    
    get_train_labels(): Returns the training labels as a dict. This is the labels portion of the dict ingested by the 
    forecasting model. It offsets labels for causal models as necessary, and cropped frames are stored in keys with
    'init' in their names. There is enough information in this dict to recreate this PoseLabels object. 
    'continuous' will be or size (pre_sz x ) (ntimepoints-starttoff) x d_continuous with the continuous pose for the fly
    and 'discrete' will be of size (pre_sz x ) (ntimepoints-starttoff) x d_discrete x nbins with the binned pose for the fly.
    get_multi(): Returns the continuous version of the full labels, with discrete features converted to continuous values.
    This implements sampling from distributions of discrete features. The output will be of size 
    (pre_sz x ) ntimepoints x d_multi. 
    get_multi_idct(): Returns the continuous version of the full labels. Multi-time predictions that have been
    combined with the DCT have had the inverse DCT applied. The output will be of size (pre_sz x ) ntimepoints x d_multi.
    get_nextcossin(): Returns all features associated with one-frame predictions. The output will be of size
    (pre_sz x ) ntimepoints x d_next_cossin. 
    get_next(): Returns all features associated with one-frame predictions. For angle features that have been represented
    by cos,sin pairs, this will convert them to angles in radians. The output will be of size (pre_sz x ) ntimepoints x d_next.
    get_next_pose(): Returns the pose features for one-frame predictions. This pose representation should be a function of 
    just a single frame's trajectory. For any feature that is a velocity or in a relative coordinate system, this will 
    integrate velocities and do coordinate transformations to get the single-frame pose. The 'init' features will be used
    to start the integration. The output will be of size (pre_sz x ) (ntimepoints+1) x d_next_pose. This +1 is because the
    first frame is the initial pose.
    get_next_keypoints(): Returns the keypoints for one-frame predictions. The output will be of size 
    (pre_sz x ) ntimepoints x nkeypoints x 2.
    
    set_prediction(): Sets labels for time points ts, usually an output of the forecasting model. 
    
    """
    def __init__(self, example_in=None,
                 Xkp=None, scale=None, metadata=None,
                 dozscore=False, dodiscretize=False,
                 dataset=None, **kwargs):
        
        super().__init__(example_in=example_in, Xkp=Xkp, scale=scale, metadata=metadata, 
                         dozscore=dozscore, dodiscretize=dodiscretize, dataset=dataset, **kwargs)

        return
    
    @classmethod
    def agentexample_to_poselabels_params(cls,params):
        """
        flyexample_to_poselabels_params(params) (static)
        Converts the parameters in the dict params from FlyExample parameters for the PoseLabels object.
        Returns this dict of parameters for PoseLabels.
        """
        
        super(cls,cls).agentexample_to_poselabels_params(params)
        params = remove_implicit_params(params)
        toremove = ['simplify_in',]
        for k in toremove:
            if k in params:
                del params[k]
        return params

    def get_params(self):
        """
        get_params()
        Returns the parameters for the PoseLabels object.
        """
        kwlabels = super().get_params()
        kwlabels = remove_implicit_params(kwlabels)
        kwlabels['tspred_global'] = self._tspred_global
        kwlabels['ntspred_relative'] = self._ntspred_relative
        kwlabels['is_velocity'] = self._is_velocity
        kwlabels['simplify_out'] = self._simplify_out

        return kwlabels

    def set_params(self, params, **kwargs):
        """
        set_params(params, override=True)
        Sets the parameters for the PoseLabels object. 
        params: Dict of parameters to set. Each key,value pair in the dict will be set as an attribute of the FlyExample object,
        with the key prefixed by an underscore. The exception are those parameters defined in synonyms, which will get different 
        names. 
        override: Whether to override existing parameters. If False, will not overwrite existing parameters. Default is True. 
        """
        #params = modernize_fly_params(params)
        super().set_params(params, **kwargs)

        return

    @property
    def tspred_global(self):
        """
        tspred_global
        Which frames into the future are predicted for global features.
        """
        return self._tspred_global
    
    @property
    def ntspred_relative(self):
        """
        ntspred_relative
        Number of frames into the future that are predicted for relative features.
        """
        return self._ntspred_relative

    # def is_dct(self):
    #     """
    #     is_dct()
    #     Returns whether the DCT is computed for relative features. 
    #     """
    #     return self.ntspred_relative > 1

    def get_scale(self, makecopy=True):
        """
        get_scale(makecopy=True)
        Returns the scale for computing features from keypoints.
        Optional parameters:
        makecopy: whether to return a copy of the data. Default is True.
        """
        if makecopy:
            return self._scale.copy()
        else:
            return self._scale

    def get_categories(self, makecopy=True):
        """
        get_categories(makecopy=True)
        Returns the categories from the MABe dataset. Currently not used for anything, may be buggy.
        Optional parameters:
        makecopy: whether to return a copy of the data. Default is True.
        """
        if self._categories is None:
            return None
        if makecopy:
            return self._categories.copy()
        else:
            return self._categories

    @property
    def _idx_nextglobal_to_next(self):
        """
        _idx_nextglobal_to_next
        Convert from nextglobal indices to next indices. 
        Returns ndarray of indices of next pose that are global
        """
        return np.array(featglobal)

    @property
    def _isdct(self):
        if self._dct_m is None:
            nfeatures = len(posenames)
            return np.zeros(nfeatures, dtype=bool)
        else:
            return featrelative.copy()
        
    @property
    def _tspred(self):
        nfeatures = len(posenames)
        tspred = [None,]*nfeatures
        for i in range(nfeatures):
            if featrelative[i] == False:
                tspred[i] = self._tspred_global.copy()
            else:
                tspred[i] = np.arange(self._ntspred_relative)
        return tspred

    @property
    def d_next_global(self):
        """
        d_next_global
        Returns the number of global features in the next frame pose.
        """
        return len(self._idx_nextglobal_to_next)

    @property
    def _idx_nextrelative_to_next(self):
        """
        idx_nextrelative_to_next
        Convert from nextrelative indices to next indices.
        Returns ndarray of indices of next pose that are relative
        """
        if self._simplify_out is None:
            return np.nonzero(featrelative)[0]
        else:
            return np.array([])

    @property
    def d_next_relative(self):
        """
        d_next_relative
        Returns the number of relative features in the next frame pose.
        """
        return len(self._idx_nextrelative_to_next)

    @property
    def d_next(self):
        """
        d_next
        Returns the total number of features in the next frame pose.
        """
        return self.d_next_global + self.d_next_relative
    
    @property
    def d_next_pose(self):
        """
        d_next_pose
        Returns the total number of features in the next frame pose, including the global and relative features.
        """
        return self.d_next

    @property
    def is_angle_next(self):
        """
        is_angle_next
        Returns a boolean array indicating which features in the next frame pose are angles.
        """
        return featangle

    @property
    def _idx_nextcossinglobal_to_nextcossin(self):
        """
        _idx_nextcossinglobal_to_nextcossin
        Convert from nextcossinglobal indices to nextcossin indices.
        Returns ndarray of indices of next cossin pose that are global. 
        """
        return np.arange(self.d_next_global)

    @property
    def d_next_cossin_global(self):
        """
        d_next_cossin_global
        Returns the number of global features in the next cossin representation.
        """
        return len(self._idx_nextcossinglobal_to_nextcossin)

    @property
    def _idx_nextglobal_to_nextcossinglobal(self):
        """
        _idx_nextglobal_to_nextcossinglobal
        Convert from nextglobal indices to nextcossinglobal indices.
        """
        return np.arange(self.d_next_global)

    def _get_idx_nextrelative_to_nextcossinrelative(self):
        """
        get_idx_nextrelative_to_nextcossinrelative()
        Returns indices for converting from nextrelative to nextcossinrelative, and
        the total number of relative features in the next cossin representation.
        """
        if self._is_velocity:
            return np.arange(self.d_next_relative), self.d_next_relative
        else:
            return relfeatidx_to_cossinidx(self._idx_nextdiscrete_to_next)

    @property
    def _idx_nextrelative_to_nextcossinrelative(self):
        """
        _idx_nextrelative_to_nextcossinrelative
        Convert from nextrelative indices to nextcossinrelative indices.
        """
        idx, _ = self._get_idx_nextrelative_to_nextcossinrelative()
        return idx

    @property
    def d_next_cossin_relative(self):
        """
        d_next_cossin_relative
        Returns the number of relative features in the next cossin representation.
        """
        # rewrote to speed up code
        # number of next relative features + number of relative features that are angles and not discretized
        nrelative = np.count_nonzero(featrelative)
        iscossin = featangle & featrelative
        iscossin[self._idx_nextdiscrete_to_next] = False
        
        return nrelative + np.count_nonzero(iscossin)

    @property
    def d_next_cossin(self):
        """
        d_next_cossin
        Returns the total number of features in the next cossin representation.
        """
        return self.d_next_cossin_relative + self.d_next_cossin_global

    @property
    def _idx_nextcossinrelative_to_nextcossin(self):
        """
        _idx_nextcossinrelative_to_nextcossin
        Convert from nextcossinrelative indices to nextcossin indices.
        """
        return np.setdiff1d(np.arange(self.d_next_cossin), self._idx_nextcossinglobal_to_nextcossin)

    @property
    def _idx_next_to_nextcossin(self):
        """
        _idx_next_to_nextcossin
        Convert from next indices to nextcossin indices.
        Seems kind of involved, maybe we should store this?? 
        """
        idx = list(range(self.d_next))
        idx_nextglobal_to_next = self._idx_nextglobal_to_next
        idx_nextglobal_to_nextcossinglobal = self._idx_nextglobal_to_nextcossinglobal
        idx_nextcossinglobal_to_nextcossin = self._idx_nextcossinglobal_to_nextcossin
        idx_nextrelative_to_next = self._idx_nextrelative_to_next
        idx_nextrelative_to_nextcossinrelative = self._idx_nextrelative_to_nextcossinrelative
        idx_nextcossinrelative_to_nextcossin = self._idx_nextcossinrelative_to_nextcossin

        for inextglobal in range(self.d_next_global):
            inext = idx_nextglobal_to_next[inextglobal]
            inextcossinglobal = idx_nextglobal_to_nextcossinglobal[inextglobal]
            inextcossin = idx_nextcossinglobal_to_nextcossin[inextcossinglobal]
            idx[inext] = inextcossin

        for inextrel in range(self.d_next_relative):
            inext = idx_nextrelative_to_next[inextrel]
            inextcossinrelative = idx_nextrelative_to_nextcossinrelative[inextrel]
            inextcossin = idx_nextcossinrelative_to_nextcossin[inextcossinrelative]
            idx[inext] = inextcossin
        return idx

    @property
    def d_multi_relative(self):
        """
        d_multi_relative
        Returns the number of relative features in the multi representation.
        """
        return self.d_next_cossin_relative * self.ntspred_relative

    @property
    def d_multi_global(self):
        """
        d_multi_global
        Returns the number of global features in the multi representation.
        """
        return self.d_next_cossin_global * len(self.tspred_global)

    @property
    def d_multi(self):
        """
        d_multi
        Returns the total number of features in the multi representation.
        """
        return self.d_multi_global + self.d_multi_relative
    
    @property
    def d_multidiscrete(self):
        """
        d_multidiscrete
        Returns the total number of discrete features in the multi representation.
        """
        # compute faster
        d_discrete_global = np.intersect1d(self._idx_nextglobal_to_next, self._idx_nextdiscrete_to_next).size
        d_discrete_relative = np.intersect1d(self._idx_nextrelative_to_next, self._idx_nextdiscrete_to_next).size
        d_multidiscrete = d_discrete_global * len(self.tspred_global) + d_discrete_relative
        return d_multidiscrete
        #return super().d_multidiscrete

    @property
    def d_multicontinuous(self):
        """
        d_multicontinuous
        Returns the number of continuous features in the multi representation.
        """
        return self.d_multi - self.d_multidiscrete

    @property
    def _idx_multifeattpred_to_multi(self):
        """
        idx_multifeattpred_to_multi
        Convert from (feat,tpred) to multi indices.
        Dictionary where keys are (feat,tpred) and values are the multi index.
        """
        # look up table from (feat,tpred) to multi index
        # dict
        idx_multifeattpred_to_multi = {}
        for idx, ft in enumerate(self._idx_multi_to_multifeattpred):
            idx_multifeattpred_to_multi[tuple(ft.tolist())] = idx
        return idx_multifeattpred_to_multi

    @property
    def _multi_isrelative(self):
        """
        _multi_isrelative
        Returns a boolean array indicating which features in the multi representation are relative.
        """
        idx_nextcossinrelative_to_nextcossin = self._idx_nextcossinrelative_to_nextcossin
        idx_multi_to_multifeattpred = self._idx_multi_to_multifeattpred
        isrelative = np.array([ft[0] in idx_nextcossinrelative_to_nextcossin for ft in idx_multi_to_multifeattpred])
        return isrelative

    @property
    def _idx_multirelative_to_multi(self):
        """
        _idx_multirelative_to_multi
        Convert from multirelative to multi indices.
        Returns indices of multi that correspond to relative features.
        """
        isrelative = self._multi_isrelative
        return np.nonzero(isrelative)[0]

    @property
    def _idx_multiglobal_to_multi(self):
        """
        _idx_multiglobal_to_multi
        Convert from multiglobal to multi indices.
        Returns indices of multi that correspond to global features.
        """
        isrelative = self._multi_isrelative
        return np.nonzero(isrelative == False)[0]

    def feattpred_to_multi(self, ftidx):
        """
        feattpred_to_multi(ftidx)
        Converts from pairs of (feature,tpred) to multi indices.
        ftidx: ndarray of size ... x 2. ftidx[...,0] are the feature indices and [...,1] are the number of frames into the
        future.
        Returns an ndarray of size ... x 1 with the multi indices.
        """
        # the order of features is a bit different than the default in PoseLabels
        # for backward compatibility
        idx = ravel_label_index(ftidx, ntspred_relative=self.ntspred_relative,
                                tspred_global=self.tspred_global, nrelrep=self.d_next_cossin_relative)
        return idx

    def multi_to_feattpred(self, idx):
        """
        multi_to_feattpred(idx)
        Converts from multi indices to pairs of (feature,tpred).
        idx: ndarray of size ... x 1. idx are the multi indices.
        Returns an ndarray of size ... x 2 with the feature indices and number of frames into the future.
        """
        # the order of features is a bit different than the default in PoseLabels
        # for backward compatibility
        ftidx = unravel_label_index(idx, ntspred_relative=self.ntspred_relative, tspred_global=self.tspred_global,
                                    nrelrep=self.d_next_cossin_relative)
        return ftidx

    def _multi_to_multiidct(self, multi):
        """
        multi_to_multiidct(multi)
        Performs the inverse DCT on relative pose features of multi to convert to the multi_idct representation, if applicable. 
        TODO- check that we are using multiidct rather than multi when we should be... 
        Parameters:
        multi: ndarray of size (pre_sz x ) ntimepoints x d_multi with the multi representation of the labels.
        Returns:
        multi_idct: ndarray of size (pre_sz x ) ntimepoints x d_multi with the multi_idct representation of the labels.
        """
        
        # if not DCT, just return multi
        if not self.is_dct():
            return multi

        # allocate multi_idct
        multi_idct = multi.copy()
        
        idct_m = self._idct_m.T

        # features to convert
        idx_nextcossinrelative_to_nextcossin = self._idx_nextcossinrelative_to_nextcossin
        idx_multi_to_multifeattpred = self._idx_multi_to_multifeattpred
        
        # for each realtive feature
        for irel in range(self.d_next_cossin_relative):
            i = idx_nextcossinrelative_to_nextcossin[irel]
            # find all features of multi for the relative feature (all tspred_relative)
            idxfeat = np.nonzero((idx_multi_to_multifeattpred[:, 0] == i) & \
                                 (idx_multi_to_multifeattpred[:, 1] > 1))[0]
            # make sure features are in order
            assert np.all(idx_multi_to_multifeattpred[idxfeat, 1] == np.arange(2, self.ntspred_relative + 1))
            # apply inverse DCT
            multi_dct = multi[..., idxfeat].reshape((-1, self.ntspred_relative - 1))
            multi_idct[..., idxfeat] = (multi_dct @ idct_m).reshape((multi.shape[:-1]) + (self.ntspred_relative - 1,))
        return multi_idct

    def _get_idx_mutli_to_futureglobal(self, tspred=None):
        """
        _get_idx_mutli_to_futureglobal(tspred=None)
        Returns the indices of multi that correspond to predictions for global features. If tspred is input,
        then will only return indices for those tspred into the future. 
        Parameter:
        tspred: values for tspred into the future to return indices for. If None, all global feature indices
        will be returned. Otherwise, only indices within tspred will be returned. Default is None.
        """
        
        # array of size d_next_cossin x 2 with feature number and tpred
        idx_multi_to_multifeattpred = self._idx_multi_to_multifeattpred
        # check for global features
        idx = np.isin(idx_multi_to_multifeattpred[:, 0], self._idx_nextcossinglobal_to_nextcossin)
        if tspred is not None:
            # check for tpred
            idx = idx & (np.isin(idx_multi_to_multifeattpred[:, 1], tspred))
        return idx

    def _multi_to_futureglobal(self, multi, tspred=None):
        """
        _multi_to_futureglobal(multi, tspred=None)
        Convert from the full multi representation (either multi or multi_idct) to the future global features.
        Parameters:
        multi: ndarray of size (pre_sz x ) ntimepoints x d_multi with the multi or multi_idct representation of the labels.
        Optional parameters:
        tspred: values for tspred into the future to return indices for. If None, all global feature indices are returned.
        Returns:
        futureglobalvel: ndarray of size (pre_sz x ) ntimepoints x (len(tspred)*d_next_cossin_global) with the future global
        features. 
        """
        # if tspred is None, returns all global features
        idx = self._get_idx_mutli_to_futureglobal(tspred)
        return multi[..., idx]

    def get_future_global(self, tspred=None, **kwargs):
        """
        get_future_global(tspred=None, use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False, ts=None)
        Returns the global features for frames into the future tspred. 
        Optional parameters:
        tspred: values for tspred into the future to return indices for. If None, all global features for all tspred are returned.
        Default: None.
        use_todiscretize: whether to use the continuous versions of the discrete labels, if available, to 
        convert discrete to continuous. Default is False.
        nsamples: Method for converting from discrete to continuous. If 0, the weighted mean of bin centers is computed. If > 0,
        specifies the number of samples to take according to the bin distributions. Default is 0.
        zscored: whether to return the z-scored version of multi. If False, multi will be unzscored. Default is False.
        collapse_samples: whether to collapse the samples dimension if nsamples=1 the first dimension. Default is False.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are returned. Default is None.
        Returns:
        futureglobalvel: ndarray of size (nsamples x) (pre_sz x ) ntimepoints x ntspred x d_next_cossin_global with the future global
        features.
        """
        
        # get multi representation of data, pass through almost all arguments
        multi = self.get_multi(**kwargs)
        
        # convert to futureglobal
        futureglobalvel = self._multi_to_futureglobal(multi, tspred=tspred)
        
        # how many tspred, reshape
        if tspred is None:
            ntspred = len(self.tspred_global)
        elif hasattr(tspred, '__len__'):
            ntspred = len(tspred)
        else:
            ntspred = 1
            tspred = [tspred, ]

        futureglobalvel = futureglobalvel.reshape((futureglobalvel.shape[:-1]) + (ntspred, self.d_next_cossin_global))

        return futureglobalvel

    def get_future_global_as_discrete(self, tspred=None, ts=None):
        """
        get_future_global_as_discrete(tspred=None, ts=None)
        Returns the future global features as discrete features. Will be nan for global features that are not discrete.
        Optional parameters:
        tspred: values for tspred into the future to return indices for. If None, all global features for all tspred are returned.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done.
        Returns:
        labels_discrete: ndarray of size (pre_sz x ) ntimepoints x ntspred x d_next_cossin_global x nbins with the future global
        features as discrete features.
        """
        
        # TODO: add some checks that global are discrete
        if not self.is_discretized():
            return None
        
        # get the raw data
        labels_raw = self.get_raw_labels(format='standard', ts=ts, makecopy=False)
        
        # allocate
        labels_discrete = np.zeros(self.pre_sz + (self.ntimepoints, self.d_multi, self._discretize_nbins),
                                   dtype=self.dtype)
        # initialize to nan
        labels_discrete[:] = np.nan
        # stored discrete features
        labels_discrete[..., self._idx_multidiscrete_to_multi, :] = labels_raw['discrete']
        # subselect for future global and tspred
        idx = self._get_idx_mutli_to_futureglobal(tspred)
        labels_discrete = labels_discrete[..., idx, :]

        # reshape
        if tspred is None:
            ntspred = len(self.tspred_global)
        elif hasattr(tspred, '__len__'):
            ntspred = len(tspred)
        else:
            ntspred = 1
        labels_discrete = labels_discrete.reshape(
            self.pre_sz + (self.ntimepoints, ntspred, self.d_next_cossin_global, self._discretize_nbins))
        return labels_discrete

    def _futureglobal_to_futureglobalpos(self, globalpos0, futureglobalvel):
        """
        _futureglobal_to_futureglobalpos(globalpos0, futureglobalvel)
        Convert global features to global positions by adding velocities and initial position, with necessary
        coordinate transforms. 
        Parameters:
        globalpos0: ndarray of size (pre_sz x ) ntimepoints x d_next_global with the initial global positions.
        futureglobalvel: ndarray of size (pre_sz x ) ntimepoints x ntspred x d_next_cossin_global with the 
        future global features for each tspred. 
        Returns:
        futureglobalpos: ndarray of size (pre_sz x ) ntimepoints x ntspred x d_next_global with the future global
        position, consisting of the (x,y) coordinate and orientation. 
        """

        szrest = futureglobalvel.shape[:-3]
        n = int(np.prod(szrest))
        T = futureglobalvel.shape[-3]
        ntspred = futureglobalvel.shape[-2]
        
        # reshape so that tspred is a dimension
        futureglobalvel = futureglobalvel.reshape((n, T, ntspred, self.d_next_global))
        globalpos0 = globalpos0[..., :T, :].reshape((n, T, self.d_next_global))

        # origin
        xorigin0 = np.tile(globalpos0[..., None, :2], (1, 1, ntspred, 1))
        # orientation
        xtheta0 = np.tile(globalpos0[..., None, 2], (1, 1, ntspred))
        
        # forward and sideways velocities
        xoriginvelrel = futureglobalvel[..., [1, 0]]  # forward=y then sideways=x
        # rotate based on coordinate systems
        xoriginvel = rotate_2d_points(xoriginvelrel.reshape((n * T * ntspred, 2)),
                                      -xtheta0.reshape(n * T * ntspred)).reshape((n, T, ntspred, 2))
        xorigin = xorigin0 + xoriginvel
        
        xtheta = modrange(xtheta0 + futureglobalvel[..., 2], -np.pi, np.pi)
        futureglobalpos = np.concatenate((xorigin, xtheta[..., None]), axis=-1)

        return futureglobalpos.reshape(szrest + (T, ntspred, self.d_next_global))

    def get_future_globalpos(self, tspred=None, **kwargs):
        """
        get_future_globalpos(tspred=None, use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False, ts=None)
        Returns the global position for frames into the future tspred.
        Optional parameters:
        tspred: values for tspred into the future to return indices for. If None, all global features for all tspred are returned.
        Default: None.
        use_todiscretize: whether to use the continuous versions of the discrete labels, if available, to
        convert discrete to continuous. Default is False.
        nsamples: Method for converting from discrete to continuous. If 0, the weighted mean of bin centers is computed. If > 0,
        specifies the number of samples to take according to the bin distributions. Default is 0.
        zscored: whether to return the z-scored version of multi. If False, multi will be unzscored. Default is False.
        collapse_samples: whether to collapse the samples dimension if nsamples=1 the first dimension. Default is False.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are returned. Default is None.
        Returns:
        futureglobalpos: ndarray of size (pre_sz x ) ntimepoints x ntspred x d_next_global with the future global position, 
        consisting of the (x,y) coordinate and orientation.
        """
        globalpos0 = self.get_next_pose_global(**kwargs)
        futureglobal = self.get_future_global(tspred=tspred, **kwargs)
        futureglobalpos = self._futureglobal_to_futureglobalpos(globalpos0, futureglobal)
        return futureglobalpos

    def _multiidct_to_futurecossinrelative(self, multi_idct, tspred=None):
        """
        _multiidct_to_futurecossinrelative(multi_idct, tspred=None)
        Convert from the multi_idct representation (full labels, relatve features have been un-DCTed) to the future relative features.
        Parameters:
        multi_idct: ndarray of size (pre_sz x ) ntimepoints x d_multi with the multi_idct representation of the labels.
        Optional parameters:
        tspred: values for tspred into the future to return indices for. If None, all relative features for all tspred are returned.
        Default: None.
        Returns:
        futurerelcs: ndarray of size (pre_sz x ) ntimepoints x ntspred x d_next_cossin_relative with the future relative features
        """
        if not self.is_dct():
            return np.zeros(self.pre_sz + (self.ntimepoints, 0), dtype=multi_idct.dtype)
        if tspred is None:
            tspred = np.arange(2, self.ntspred_relative + 1)
        elif not hasattr(tspred, '__len__'):
            tspred = [tspred, ]
        ntspred = len(tspred)
        idx_multi_to_multifeattpred = self._idx_multi_to_multifeattpred
        idxfeat = np.nonzero(np.isin(idx_multi_to_multifeattpred[:, 0], self._idx_nextcossinrelative_to_nextcossin) & \
                             np.isin(idx_multi_to_multifeattpred[:, 1], tspred))[0]
        return multi_idct[..., idxfeat].reshape((multi_idct.shape[:-1]) + (ntspred, self.d_next_cossin_relative))

    def _get_future_cossin_relative(self, tspred=None, **kwargs):
        """
        _get_future_cossin_relative(tspred=None, use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False, ts=None)
        Returns the future relative features for frames into the future tspred.
        Optional parameters:
        tspred: values for tspred into the future to return indices for. If None, all relative features for all tspred are returned.
        Default: None.
        zscored: whether to return the z-scored version of multi. If False, multi will be unzscored. Default is False.
        collapse_samples: whether to collapse the samples dimension if nsamples=1 the first dimension. Default is False.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are returned. Default is None.
        Returns:
        futurerelcs: ndarray of size (pre_sz x ) ntimepoints x ntspred x d_next_cossin_relative with the future relative features
        """

        multi_idct = self.get_multi_idct(**kwargs)
        futurerelcs = self._multiidct_to_futurecossinrelative(multi_idct, tspred=tspred)
        return futurerelcs

    def get_future_relative(self, tspred=None, **kwargs):
        """
        get_future_relative(tspred=None, use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False, ts=None)
        Returns the future relative features for frames into the future tspred.
        Optional parameters:
        tspred: values for tspred into the future to return indices for. If None, all relative features for all tspred are returned.
        Default: None.
        zscored: whether to return the z-scored version of multi. If False, multi will be unzscored. Default is False.
        collapse_samples: whether to collapse the samples dimension if nsamples=1 the first dimension. Default is False.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are returned. Default is None.
        Returns:
        futurerel: ndarray of size (pre_sz x ) ntimepoints x ntspred x d_next_relative with the future relative features
        """
        futurerelcs = self._get_future_cossin_relative(tspred=tspred, **kwargs)
        futurerel = np.moveaxis(self._nextcossinrelative_to_nextrelative(np.moveaxis(futurerelcs, -2, 0)), 0, -2)
        return futurerel

    def get_future_relative_pose(self, tspred=None, zscored=False, **kwargs):
        """
        get_future_relative_pose(tspred=None, ts=None)
        Returns the future relative pose for frames into the future tspred.
        Optional parameters:
        tspred: values for tspred into the future to return indices for. If None, all relative features for all tspred are returned.
        Default: None.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. 
        Returns:
        futurerelpose: ndarray of size (pre_sz x ) ntimepoints x ntspred x d_next_relative with the future relative pose.
        """
        assert zscored == False
        futurerel = self.get_future_relative(tspred=tspred, zscored=False,**kwargs)
        if not self._is_velocity:
            return futurerel
        # if self._is_velocity, then futurerel is a velocity, so add it to current next pose relative
        relpose0 = self.get_next_pose_relative(**kwargs)
        return futurerel + relpose0[..., :-1, None, :]

    def _nextcossinglobal_to_nextglobal(self, next_cossinglobal):
        """
        _nextcossinglobal_to_nextglobal(next_cossinglobal)
        Convert the global features in the next_cossin representation to the next_global representation.
        cos,sin representation only affects relative features, so this is the identity. 
        Parameters:
        next_cossinglobal: ndarray of size (pre_sz x ) ntimepoints x d_next_cossin_global with the global features in the
        next_cossin representation.
        Returns:
        next_global: ndarray of size (pre_sz x ) ntimepoints x d_next_global with the global features in the next_global
        representation.
        """
        return next_cossinglobal

    def _nextcossinrelative_to_nextrelative(self, next_cossin_relative):
        """
        _nextcossinrelative_to_nextrelative(next_cossin_relative)
        Convert the relative features in the next_cossin representation to the next_relative representation. All 
        relative angle features are converted from cos,sin to radians using the arctan2 function.
        Parameters:
        next_cossin_relative: ndarray of size (pre_sz x ) ntimepoints x d_next_cossin_relative with the relative features in the
        next_cossin representation.
        Returns:
        next_relative: ndarray of size (pre_sz x ) ntimepoints x d_next_relative with the relative features in the next_relative
        representation.
        """
        # reshape so we can mostly ignore pre_sz
        szrest = next_cossin_relative.shape[:-2]
        T = next_cossin_relative.shape[-2]
        n = int(np.prod(szrest))
        next_cossin_relative = next_cossin_relative.reshape((n, T, self.d_next_cossin_relative))

        # allocate
        next_relative = np.zeros((n, T, self.d_next_relative), dtype=next_cossin_relative.dtype)
        idx_nextrelative_to_nextcossinrelative = self._idx_nextrelative_to_nextcossinrelative
        for inext in range(self.d_next_relative):
            # indices of the next_cossin_relative features that correspond to inext
            inextcossin = idx_nextrelative_to_nextcossinrelative[inext]
            # if this is an ndarray, then we need to convert from cos,sin to radians
            if type(inextcossin) is np.ndarray:
                next_relative[..., inext] = np.arctan2(next_cossin_relative[..., inextcossin[1]],
                                                       next_cossin_relative[..., inextcossin[0]])
            else:
                # otherwise just copy
                next_relative[..., inext] = next_cossin_relative[..., inextcossin]

        # reshape back
        next_relative = next_relative.reshape(szrest + (T, self.d_next_relative))
        
        return next_relative

    def _nextcossin_to_next(self, next_cossin):
        """
        _nextcossin_to_next(next_cossin)
        Convert from the next_cossin representation to the next representation. This will convert the cos,sin representation
        of relative angles to radians.
        Parameters:
        next_cossin: ndarray of size (pre_sz x ) ntimepoints x d_next_cossin with the next_cossin representation of the labels.
        Returns:
        next: ndarray of size (pre_sz x ) ntimepoints x d_next with the next representation of the labels.
        """
        next = np.zeros(next_cossin.shape[:-1] + (self.d_next,), dtype=next_cossin.dtype)
        # convert global features (does nothing)
        next[..., self._idx_nextglobal_to_next] = \
            self._nextcossinglobal_to_nextglobal(next_cossin[..., self._idx_nextcossinglobal_to_nextcossin])
        # convert relative features
        next[..., self._idx_nextrelative_to_next] = \
            self._nextcossinrelative_to_nextrelative(next_cossin[..., self._idx_nextcossinrelative_to_nextcossin])
        return next

    def _globalvel_to_globalpos(self, globalvel, starttoff=0, init_pose=None):
        """
        _globalvel_to_globalpos(globalvel, starttoff=0, init_pose=None)
        Convert from global velocities to global positions by integrating.
        Parameters:
        globalvel: ndarray of size (pre_sz x ) ntimepoints x d_next_global with the global velocities.
        starttoff: offset for the initial pose. Default is 0.
        init_pose: ndarray of size (pre_sz x ) d_next_global with the initial pose. If None, 
        self._init_pose is used. Default is None.
        Returns:
        globalpos: ndarray of size (pre_sz x ) ntimepoints x d_next_global with the global positions.
        """

        n = globalvel.shape[0]
        T = globalvel.shape[1]

        if init_pose is None:
            init_pose = self._init_pose[...,starttoff]
            
        globalpos0 = init_pose[..., self._idx_nextglobal_to_next]
        xorigin0 = globalpos0[..., :2]
        xtheta0 = globalpos0[..., 2]

        thetavel = globalvel[..., 2]
        xtheta = np.cumsum(np.concatenate((xtheta0.reshape((n, 1)), thetavel), axis=-1), axis=-1)

        xoriginvelrel = globalvel[..., [1, 0]]  # forward=y then sideways=x
        xoriginvel = rotate_2d_points(xoriginvelrel.reshape((n * T, 2)), -xtheta[..., :-1].reshape(n * T)).reshape(
            (n, T, 2))
        xorigin = np.cumsum(np.concatenate((xorigin0.reshape(n, 1, 2), xoriginvel), axis=-2), axis=-2)

        globalpos = np.concatenate((xorigin, xtheta[..., None]), axis=-1)
        return globalpos

    def _relrep_to_relpose(self, relrep, init_pose=None, starttoff=0):
        """
        _relrep_to_relpose(relrep, init_pose=None, starttoff=0)
        Convert from relative representation to relative poses. If self._is_velocity is True, then 
        this will require integrating from the initial pose. Otherwise, initial pose is just concatenated.
        Parameters:
        relrep: ndarray of size (pre_sz x ) ntimepoints x d_next_relative with the relative representation.
        Optional parameters:
        init_pose: ndarray of size (pre_sz x ) d_next with the initial pose. If None, self._init_pose is used
        to derive this. Default is None.
        starttoff: offset for the initial pose. Default is 0.
        Returns:
        relpose: ndarray of size (pre_sz x ) ntimepoints x d_next_relative with the relative poses.
        """

        n = relrep.shape[0]
        T = relrep.shape[1]

        if init_pose is None:
            init_pose = self._init_pose[..., starttoff]
        relpose0 = init_pose[..., self._idx_nextrelative_to_next]

        if self._is_velocity:
            relpose = np.cumsum(np.concatenate((relpose0.reshape((n, 1, -1)), relrep), axis=-2), axis=-2)
        else:
            relpose = np.concatenate((relpose0.reshape((n, 1, -1)), relrep), axis=-2)

        return relpose

    def _next_to_nextpose(self, next, init_pose=None):
        """
        _next_to_nextpose(next, init_pose=None)
        Convert from the next representation to the next pose representation. Will integrate velocities
        starting from init_pose. 
        Parameters:
        next: ndarray of size (pre_sz x ) ntimepoints x d_next with the next representation of the labels.
        Optional parameters:
        init_pose: ndarray of size (pre_sz x ) d_next with the initial pose. If None, self._init_pose is used.
        Default is None.
        Returns:
        pose: ndarray of size (pre_sz x ) ntimepoints x d_next with the next pose representation of the labels.
        """

        szrest = next.shape[:-2]
        n = int(np.prod(szrest))
        starttoff = 0
        T = next.shape[-2]
        next = next.reshape((n, T, self.d_next))
        globalvel = next[..., self._idx_nextglobal_to_next]
        globalpos = self._globalvel_to_globalpos(globalvel, starttoff=starttoff, init_pose=init_pose)

        relrep = next[..., self._idx_nextrelative_to_next]
        relpose = self._relrep_to_relpose(relrep, init_pose=init_pose, starttoff=starttoff)

        pose = np.concatenate((globalpos, relpose), axis=-1)
        pose[..., self.is_angle_next] = modrange(pose[..., self.is_angle_next], -np.pi, np.pi)

        pose = pose.reshape(szrest + (pose.shape[-2], self.d_next))

        return pose

    def _nextpose_to_next(self, nextpose):
        """
        _nextpose_to_next(nextpose)
        Convert from the next pose representation to the next representation. Differences between
        pairs of frames will be computed for all velocity features.
        Parameters:
        nextpose: ndarray of size (pre_sz x ) ntimepoints x d_next with the next pose representation of the labels.
        Returns:
        next: ndarray of size (pre_sz x ) ntimepoints x d_next with the next representation of the labels.
        init_pose: ndarray of size (pre_sz x ) d_next with the initial pose.
        """

        szrest = nextpose.shape[:-2]
        n = int(np.prod(szrest))
        T = nextpose.shape[-2]
        nextpose = nextpose.reshape((n, T, self.d_next))
        init_pose = nextpose[..., 0, :]
        
        # if is_velocity, then compute diff for all features
        if self._is_velocity:
            next = np.diff(nextpose, axis=1)
        else:
            # otherwise just compute for global features
            idx_nextglobal_to_next = self._idx_nextglobal_to_next
            # offset by 1 to get the next frame's relative features
            next = nextpose[..., 1:, :].copy()
            next[..., idx_nextglobal_to_next] = np.diff(nextpose[..., idx_nextglobal_to_next], axis=1)
        next[..., self.is_angle_next] = modrange(next[..., self.is_angle_next], -np.pi, np.pi)
        next = next.reshape(szrest + (T - 1, self.d_next))

        return next, init_pose

    def _next_to_nextvelocity(self, next):
        """
        _next_to_nextvelocity(next)
        Convert from next representation to next velocity representation, with differences computed for all features.
        If self._is_velocity is True, then this will be the identity.
        Parameters:
        next: ndarray of size (pre_sz x ) ntimepoints x d_next with the next representation of the labels.
        Returns:
        vel: ndarray of size (pre_sz x ) ntimepoints x d_next with the next velocity representation of the labels.
        """

        if self._is_velocity:
            return next

        szrest = next.shape[:-2]
        n = int(np.prod(szrest))
        T = next.shape[-2]
        idx_nextrelative_to_next = self._idx_nextrelative_to_next
        velrel = np.zeros((n, T + 1, self.d_next_relative), dtype=next.dtype)
        velrel[:, 0, :] = self._init_pose[idx_nextrelative_to_next]
        velrel[:, 1:, :] = next[..., idx_nextrelative_to_next].reshape((n, T, self.d_next_relative))
        velrel[:, :-1, :] = np.diff(velrel, axis=1)
        velrel = velrel[:, :-1, :]
        velrel = velrel.reshape(szrest + (T, self.d_next_relative))
        vel = next.copy()
        vel[..., idx_nextrelative_to_next] = velrel

        return vel

    def get_next_pose(self, init_pose=None, zscored=False, ts=None, **kwargs):
        """
        get_next_pose(init_pose=None, zscored=False, ts=None, use_todiscretize=False, nsamples=0)
        Returns the next pose representation of the labels. This will integrate velocities from init_pose.
        Optional parameters:
        init_pose: ndarray of size (pre_sz x ) d_next with the initial pose. If None, self._init_pose is used.
        Default is None.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. 
        use_to_discretize: whether to use the continuous versions of the discrete labels, if available, to
        convert discrete to continuous. Default is False.
        nsamples: Method for converting from discrete to continuous. If 0, the weighted mean of bin centers is computed. If > 0,
        specifies the number of samples to take according to the bin distributions. Default is 0.
        Returns:
        nextpose: ndarray of size (pre_sz x ) ntimepoints x d_next with the next pose representation of the labels.
        """
        
        assert zscored == False, 'zscored must be False'
        if (ts is not None) and np.array(ts)[0] > 0:
            assert init_pose is not None, 'init_pose must be provided if ts[0] > 0'

        # only deal with un-zscored data, as we will be concatenating with init, which is not zscored
        next = self.get_next(zscored=False,ts=ts,**kwargs)
        # global will always be velocity, still need to do an integration
        next_pose = self._next_to_nextpose(next, init_pose=init_pose)
        return next_pose

    def _next_to_nextrelative(self, next):
        """
        _next_to_nextrelative(next)
        Convert from the next representation to the next_relative representation. This subselects the 
        relative features and returns those.
        Parameters:
        next: ndarray of size (pre_sz x ) ntimepoints x d_next with the next representation of the labels.
        Returns:
        next_relative: ndarray of size (pre_sz x ) ntimepoints x d_next_relative with the next_relative representation of the labels.
        """
        next_relative = next[..., self._idx_nextrelative_to_next]
        return next_relative

    def next_to_nextglobal(self, next):
        """
        next_to_nextglobal(next)
        Convert from the next representation to the next_global representation. This subselects the
        global features and returns those.
        Parameters:
        next: ndarray of size (pre_sz x ) ntimepoints x d_next with the next representation of the labels.
        Returns:
        next_global: ndarray of size (pre_sz x ) ntimepoints x d_next_global with the next_global representation of the labels.
        """
        next_global = next[..., self._idx_nextglobal_to_next]
        return next_global

    def get_next_pose_relative(self, zscored=False, **kwargs):
        """
        get_next_pose_relative(ts=None)
        Returns the relative pose for the next frame.
        Optional parameters:
        ts: indices of the time points to return. Time points must be contiguous, limited checking done.
        Returns:
        nextpose_relative: ndarray of size (pre_sz x ) ntimepoints x d_next_relative with the relative pose.
        """
        assert zscored == False, 'zscored must be False'
        nextpose = self.get_next_pose(**kwargs)
        nextpose_relative = self._next_to_nextrelative(nextpose)
        return nextpose_relative

    def get_next_pose_global(self, **kwargs):
        """
        get_next_pose_global(ts=None,use_todiscretize=False, nsamples=0)
        Returns the global pose for the next frame.
        Optional parameters:
        ts: indices of the time points to return. Time points must be contiguous, limited checking done.
        use_todiscretize: whether to use the continuous versions of the discrete labels, if available, to
        convert discrete to continuous. Default is False.
        nsamples: Method for converting from discrete to continuous. If 0, the weighted mean of bin centers is computed. If 1,
        then it will sample from the bin distributions. Default is 0.
        Returns:
        nextpose_global: ndarray of size (pre_sz x ) ntimepoints x d_next_global with the global pose.
        """
        nextpose = self.get_next_pose(**kwargs)
        nextpose_global = self.next_to_nextglobal(nextpose)
        return nextpose_global

    def set_next_pose(self, nextpose, **kwargs):
        """
        set_next_pose(nextpose,nsamples=0)
        Sets the next pose representation of the labels. This will only update the next frame labels, not the
        future frame labels. Note that if there are future features that are discrete and self.is_todiscretize() == False, then
        todiscretize for those features will be set based on sampling strategies described by nsamples.
        Parameters:
        nextpose: ndarray of size (pre_sz x ) ntimepoints x d_next with the next pose representation of the labels.
        Optional parameters:
        nsamples: Method for converting from discrete to continuous when getting multi to fill in. If 0, the weighted mean of bin
        centers is computed. If 1, then it will sample from the bin distributions. Default is 0.
        """
        self._pre_sz = nextpose.shape[:-2]
        next, init_pose = self._nextpose_to_next(nextpose)
        self._init_pose = init_pose.T
        self.set_next(next, zscored=False, **kwargs)

    def _nextpose_to_nextkeypoints(self, pose):
        """
        _nextpose_to_nextkeypoints(pose)
        Convert from the next pose representation to the next keypoints representation. 
        Parameters:
        pose: ndarray of size (pre_sz x ) ntimepoints x d_next with the next pose representation of the labels.
        Returns:
        kp: ndarray of size (pre_sz x ) ntimepoints x nkpts x 2 with the next keypoints representation of the labels.
        """

        if self._scale.ndim == 1:
            nflies = 1
        else:
            nflies = int(np.prod(self._scale.shape[:-1]))

        # input to feat2kp is expected to be an np.ndarray with shape nfeatures x T x nflies
        if nflies == 1:
            szrest = pose.shape[:-1]
            n = int(np.prod(szrest))
            pose = pose.reshape((n, self.d_next)).T
            scale = self._scale
        else:
            szrest = pose.shape[:-2]
            T = pose.shape[-2]
            n = int(np.prod(szrest))
            assert n == nflies
            pose = pose.reshape((n, T, self.d_next)).transpose((1, 2, 0))
            scale = self._scale.reshape((nflies, -1)).T
        kp = feat2kp(pose, scale)
        if nflies == 1:
            kp = kp[..., 0].transpose((2, 0, 1))
            kp = kp.reshape(szrest + kp.shape[-2:])
        else:
            # kp will be nkpts x 2 x T x nflies
            kp = kp.transpose((3, 2, 0, 1))
            # kp is now nflies x T x nkpts x 2
            kp = kp.reshape(szrest + kp.shape[1:])
        return kp
    
    @property
    def n_keypoints(self):
        return len(keypointnames)

    def compute_features(self, Xkp, scale=None):
        
        if (scale is None) and (self._scale is not None):
            scale = self._scale

        # function for computing features
        example = compute_features(Xkp[..., None], scale_perfly=scale, outtype=np.float32,
                                   simplify_out=self._simplify_out,
                                   dct_m=self._dct_m,
                                   tspred_global=self.tspred_global,
                                   compute_pose_vel=self._is_velocity,
                                   discreteidx=self._idx_nextdiscrete_to_next,
                                   simplify_in='no_sensory')
        
        if self._init_pose is None:
            self._init_pose = kp2feat(Xkp[:,:,:2],scale)[...,0]
        
        return example,scale

    def get_next_velocity(self, **kwargs):
        """
        get_next_velocity(use_todiscretize=False, nsamples=0, collapse_samples=False, ts=None)
        Returns the next frame features in velocity representation.
        Optional parameters:
        use_todiscretize: whether to use the continuous versions of the discrete labels, if available, to
        convert discrete to continuous. Default is False.
        nsamples: Method for converting from discrete to continuous. If 0, the weighted mean of bin centers is computed. If > 0,
        specifies the number of samples to take according to the bin distributions. Default is 0.
        collapse_samples: whether to collapse the samples dimension if nsamples=1 the first dimension. Default is False.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done.
        Returns:
        vel: ndarray of size (pre_sz x ) ntimepoints x d_next with the next velocity representation of the labels.
        """

        next = self.get_next(**kwargs)

        # global will always be velocity
        if self._is_velocity:
            return next

        next_vel = self._next_to_nextvelocity(next)

        return next_vel

    def set_zscore_params(self, zscore_params):
        """
        set_zscore_params(zscore_params)
        Set the zscore parameters.
        Parameters:
        zscore_params: dictionary with keys 'mu_labels' and 'sigma_labels' containing the mean and standard deviation
        """
        self._zscore_params = zscore_params
        return

    def add_next_noise(self, eta_next, zscored=False):
        """
        add_next_noise(eta_next, zscored=False)
        Add noise to the next frame features. Currently not used/debugged.
        Parameters:
        eta_next: ndarray of size (pre_sz x ) ntimepoints x d_next with the noise to add.
        zscored: whether the input eta_next is zscored. If it is not, then eta_next will be zscored (if self.is_zscored()) before
        adding. Default is False.
        """
        next = self.get_next(zscored=zscored)
        next = next + eta_next
        self.set_next(next, zscored=zscored)

    def get_nextglobal_names(self):
        """
        get_nextglobal_names()
        Returns the names of the global features.
        Returns:
        nextglobal_names: list of names of the global features.
        """
        return ['forward', 'sideways', 'orientation']

    def get_nextrelative_names(self):
        """
        get_nextrelative_names()
        Returns the names of the relative features.
        Returns:
        nextrelative_names: list of names of the relative features.
        """
        idx_nextrelative_to_next = self._idx_nextrelative_to_next
        return [posenames[i] for i in idx_nextrelative_to_next]

    def get_next_names(self):
        """
        get_next_names()
        Returns the names of the next features.
        Returns:
        next_names: list of names of the next features.
        """
        next_names = [None, ] * self.d_next
        next_names_global = self.get_nextglobal_names()
        next_names_relative = self.get_nextrelative_names()
        for i, inext in enumerate(self._idx_nextglobal_to_next):
            next_names[inext] = next_names_global[i]
        for i, inext in enumerate(self._idx_nextrelative_to_next):
            next_names[inext] = next_names_relative[i]
        return next_names

    def get_multi_names(self):
        """
        get_multi_names()
        Returns the names of the multi features.
        Returns:
        multi_names: list of names of the multi features.
        """
        ft = self._idx_multi_to_multifeattpred
        ismulti = (np.max(self.tspred_global) > 1) or (self.ntspred_relative > 1)
        multi_names = [None, ] * self.d_multi
        nextcs_names = self.get_nextcossin_names()
        for i in range(self.d_multi):
            if ismulti:
                multi_names[i] = nextcs_names[ft[i, 0]] + '_' + str(ft[i, 1])
        return multi_names

    def select_featidx_plot(self, ntsplot=None, ntsplot_global=None, ntsplot_relative=None):
        """
        select_featidx_plot(ntsplot=None, ntsplot_global=None, ntsplot_relative=None)
        Select a hopefully representative subset of the features to plot. This will return the indices and the 
        feature-time pairs.
        Optional parameters:
        ntsplot_global: Number of tspred into the future to plot for global features. If None, will default
        to ntsplot_global = ntsplot. If both are None, then will plot all global features. If ntsplot_global
        is less than the total number of tspred_global, then it will try to select different ntsplot_global
        for each global feature to cover the range of tspred_global. 
        ntsplot_relative: Number of tspred into the future to plot for relative features. If None, will default
        ntsplot. As with ntsplot_global, if ntsplot_relative is less than ntspred_relative,
        then it will try to select different ntsplot_relative for each relative feature to cover the range of
        ntspred_relative.
        ntsplot: Default ntsplot for both global and relative features. If None, and ntsplot_global
        or ntsplot_relative is not None, then all timepoints will be selected. 
        """

        idx_multi_to_multifeattpred = self._idx_multi_to_multifeattpred
        idx_multifeattpred_to_multi = self._idx_multifeattpred_to_multi
        ntspred_global = len(self.tspred_global)
        
        # number of tsplot for global
        if ntsplot_global is None and ntsplot is not None:
            ntsplot_global = ntsplot
            
        if ntsplot_global is None or (ntsplot >= ntspred_global):
            # select all features
            idxglobal = self._idx_multiglobal_to_multi
            ftglobal = idx_multi_to_multifeattpred[idxglobal, :]
            ntsplot_global = ntspred_global
        else:
            d_next_global = self.d_next_global
            # which indices of tspred_global to select for each feature
            tidxplot_global = np.concatenate((np.zeros((d_next_global, 1), dtype=int),
                                              np.round(np.linspace(1, ntspred_global - 1,
                                                                   (ntsplot_global - 1) * d_next_global)).astype(
                                                  int).reshape(-1, d_next_global).T), axis=-1)
            ftglobal = []
            idxglobal = []
            for fi, f in enumerate(self._idx_nextcossinglobal_to_nextcossin):
                for ti in range(ntsplot_global):
                    ftcurr = (f, self.tspred_global[tidxplot_global[fi, ti]])
                    ftglobal.append(ftcurr)
                    idxglobal.append(idx_multifeattpred_to_multi[ftcurr])
            ftglobal = np.array(ftglobal)
            idxglobal = np.array(idxglobal)

        ntspred_relative = self.ntspred_relative
        # number of tsplot for relative
        if ntsplot_relative is None and ntsplot is not None:
            ntsplot_relative = ntsplot
        if ntsplot_relative is None or (ntsplot_relative >= ntspred_relative):
            # select all features
            idxrelative = self._idx_multirelative_to_multi
            ftrelative = idx_multi_to_multifeattpred[idxrelative, :]
            ntsplot_relative = ntspred_relative
        elif ntsplot_relative == 0:
            idxrelative = np.zeros((0,), dtype=int)
            ftrelative = np.zeros((0, 2), dtype=int)
        else:
            d_next_cossin_relative = self.d_next_cossin_relative
            # which tsplot to select for each feature
            tplot_relative = np.concatenate((np.ones((d_next_cossin_relative, 1), dtype=int),
                                             np.round(np.linspace(2, ntspred_relative, (
                                                         ntsplot_relative - 1) * d_next_cossin_relative)).astype(
                                                 int).reshape(-1, d_next_cossin_relative).T), axis=-1)
            ftrelative = []
            idxrelative = []
            for fi, f in enumerate(self._idx_nextcossinrelative_to_nextcossin):
                for ti in range(ntsplot_relative):
                    ftcurr = (f, tplot_relative[fi, ti])
                    ftrelative.append(ftcurr)
                    idxrelative.append(idx_multifeattpred_to_multi[ftcurr])
            ftrelative = np.array(ftrelative)
            idxrelative = np.array(idxrelative)
        idx = np.concatenate((idxglobal, idxrelative), axis=0)
        ft = np.concatenate((ftglobal, ftrelative), axis=0)
        order = np.argsort(idx, axis=0)
        idx = idx[order]
        ft = ft[order]

        return idx, ft

class FlyExample(AgentExample):
    """
    
    FlyExample
    Class for handling input observations/inputs and pose/label outputs for a fly for multiple time points. 
    Can be used with batches. 
        
    Main properties:
    labels: PoseLabels object with the pose labels
    inputs: ObservationInputs object with the input observations 
    
    Main methods:
    __init__: Constructor for initializing from a training example or from keypoints.
    get_train_example(do_add_noise=False): Returns the training example consisting of inputs and labels. 
    copy_subindex(idx_pre=None, ts=None, needinit=True): Returns a copy of the FlyExample with a subset 
    of the examples (if batched) and/or a subset of the time points.
    
    TODO- separate out fly specific stuff, rename to AgentExample
    
    """
    
    _labelsClass = FlyPoseLabels
    _inputsClass = FlyObservationInputs
    
    def __init__(self,example_in: typing.Optional[dict] = None,
                 dataset: typing.Optional['FlyLLMDataset'] = None,
                 Xkp: typing.Optional[np.ndarray] = None,
                 agentnum: typing.Optional[int] = None,
                 scale: typing.Optional[np.ndarray] = None,
                 metadata: typing.Optional[dict] = None,
                 dozscore: bool = False,
                 dodiscretize: bool = False,
                 **kwargs):
        
        """
        __init__(example_in=None, dataset=None, Xkp=None, agentnum=None, scale=None, metadata=None,
                 dozscore=False, dodiscretize=False, **kwargs)
        Constructor for initializing from a training example or from keypoints.
        
        To initialize from an example computed with features.compute_features or a training example, pass in example_in:
        example_in: dictionary with the example. This can be the output of compute_features or FlyExample.get_train_example(). 
            Required fields:
            'input': ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the fly at each time point
            'labels' or 'continuous': ndarray of size (pre_sz x ) ntimepoints x d_continuous. Continuous pose labels for the fly
            at each time point. TODO- check that this will work without continuous inputs.
            'labels_discrete' or 'discrete': ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins. Binned pose labels 
            for the fly. TODO- check that this will work without discretized inputs. 
            
            Optional fields:
            'labels_init' or 'continuous_init': ndarray of size (pre_sz x ) tinit x d_continuous with the initial continuous pose
            labels for the fly. Used if the first frame of the sequence has been cropped as a training example for a causal network.
            'labels_discrete_init' or 'discrete_init': ndarray of size (pre_sz x ) tinit x d_discrete x nbins with the initial
            discretized pose. Used if the first frame of the sequence has been cropped as a training example for a causal network.
            'labels_todiscretize' or 'todiscretize': ndarray of size (pre_sz x ) ntimepoints x d_discrete with the continuous
            versions of the discrete labels. As discrete is non-invertible, this can be used when getting keypoints or other 
            continuous representations of the data. 
            'metadata': dictionary with metadata about which fly and video frames the observations were derived from.
            'categories': dictionary with the categories from the MABe dataset. Currently not used for anything, may be buggy. 
            'mask': ndarray of size (pre_sz x ) ntimepoints with a mask for the labels.
            'init_all' or 'init': ndarray of size (pre_sz x ) ntimepoints x d_next with the initial pose of the fly
            'input_init': ndarray of size (pre_sz x ) 1 x d_input with the observations of the fly on the 
            first frame of the sequence. This should be part of a training example in which the first frame
            has been cropped from input for a causal network. 
            'metadata': dictionary with metadata about which fly and video frames the observations were derived from

        To initialize from keypoints, pass in the following:
        Xkp: ndarray of size (pre_sz x ) ntimepoints x nfeatures x 2 with the keypoints for all flies. 
        fly: index of the main fly.
        scale: scale parameters for converting from keypoints to features.
        
        Optional parameters:
        dataset: FlyMLMDataset object. Used to get parameters for computing features, etc.
        dozscore: Whether to z-score the inputs. Only used if example_in is input. Set this to true if the example input
        should be z-scored, i.e. it is not already z-scored. Default is False. 
        npad: Number of frames to crop from the end of the sequence when computing features. Only used if Xkp is input. 
        This is used to manually set the number of frames to crop from the end of the sequence. This parameter may be
        obsolete. 
        
        Optional parameters defined by the dataset configuration, input to set_params():
            zscore_params
            do_input_labels
            starttoff
            flatten_labels
            flatten_obs
            discreteidx
            tspred_global
            discrete_tspred
            ntspred_relative
            discretize_params
            is_velocity
            simplify_out
            simplify_in
            flatten_obs_idx
            dct_m
            idct_m        
        To-do: put these all in a dict. 
        
        """
        
        super().__init__(example_in=example_in,
                        dataset=dataset,
                        Xkp=Xkp,
                        agentnum=agentnum,
                        scale=scale,
                        metadata=metadata,
                        dozscore=dozscore,
                        dodiscretize=dodiscretize,
                        **kwargs)

        return
    
    def compute_features(self, Xkp, agentnum, scale):
        """
        compute_features(Xkp, agentnum, scale)
        Compute sensory and pose label features from keypoints by calling features.compute_features with the 
        correct parameters. 
        Parameters:
        Xkp: ndarray of keypoints for all flies and time points, size (pre_sz x ) nkeypoints x 2 x ntimepoints x nflies
        agentnum: index of the main fly
        scale: scale parameters for converting from keypoints to features.
        Output:
        example: dictionary with the computed features:
            'input': ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the fly at each time point
            'labels': ndarray of size (pre_sz x ) ntimepoints x d_multi with the pose representation for the fly at each
            time point.
            'init': ndarray of size (pre_sz x ) ntimepoints x d_next with the initial pose of the fly. 
            'scale': scale parameters for converting from keypoints to features and vice-versa. 
        """

        example = compute_features(Xkp, flynum=agentnum, scale_perfly=scale, outtype=np.float32,
                                   simplify_in=self._simplify_in,
                                   simplify_out=self._simplify_out,
                                   dct_m=self._dct_m,
                                   tspred_global=self._tspred_global,
                                   compute_pose_vel=self._is_velocity,
                                   discreteidx=self._discreteidx)

        return example

    def set_params(self, params, override=True):
        """
        set_params(params, override=True)
        Sets the parameters for the FlyExample object. 
        params: Dict of parameters to set. Each key,value pair in the dict will be set as an attribute of the FlyExample object,
        with the key prefixed by an underscore. The exception are those parameters defined in synonyms, which will get different 
        names. 
        override: Whether to override existing parameters. If False, will not overwrite existing parameters. Default is True. 
        """
        #params = modernize_fly_params(params)
        synonyms = {'compute_pose_vel': 'is_velocity'}
        super().set_params(params,override=override,synonyms=synonyms)

    @classmethod
    def get_default_params(cls):
        """
        get_default_params()
        Returns the default parameters for the FlyExample object as a dict.
        """
        
        params =  super(cls,cls).get_default_params()
        params = remove_implicit_params(params)
        params['tspred_global'] = [1, ]
        params['ntspred_relative'] = 1
        params['is_velocity'] = False
        params['simplify_out'] = None
        params['simplify_in'] = None

        return params

    @classmethod
    def get_params_from_dataset(cls,dataset):
        """
        get_params_from_dataset(dataset)
        Returns the parameters for the FlyExample object taken from a FlyMLMDataset object as a dict.
        """
        
        params = super(cls,cls).get_params_from_dataset(dataset)
        params = remove_implicit_params(params)
        params['tspred_global'] = dataset.tspred_global
        params['ntspred_relative'] = dataset.ntspred_relative
        params['is_velocity'] = dataset.compute_pose_vel
        params['simplify_out'] = dataset.simplify_out
        params['simplify_in'] = dataset.simplify_in

        return params
        
    def get_train_input_shapes(self):
        idx,sz = self.inputs.get_input_shapes()
        idx = copy.deepcopy(self.inputs._sensory_feature_idx)
        sz = copy.deepcopy(self.inputs._sensory_feature_szs)
        if self._do_input_labels:
            d_input_labels = self.labels.get_d_labels_input()
            for k, v in idx.items():
                idx[k] = [x + d_input_labels for x in v]
            idx['labels'] = [0, d_input_labels]
            sz['labels'] = (d_input_labels,)
        return idx, sz

FlyPoseLabels._exampleClass = FlyExample
FlyObservationInputs._exampleClass = FlyExample