import numpy as np
import torch
import copy

from flyllm.config import featglobal, featrelative, featangle, posenames, nfeatures
from apf.data import weighted_sample, discretize_labels
from apf.utils import modrange, rotate_2d_points, len_wrapper, dict_convert_torch_to_numpy
from flyllm.features import (
    compute_features,
    combine_inputs,
    split_features,
    get_sensory_feature_idx,
    get_sensory_feature_shapes,
    compute_sensory_wrapper,
    relfeatidx_to_cossinidx,
    ravel_label_index,
    unravel_label_index,
    zscore,
    unzscore,
    feat2kp,
    kp2feat
)


class ObservationInputs:
    """
    ObservationInputs
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

    def __init__(self, example_in=None, Xkp=None, fly=0, scale=None, dataset=None, dozscore=False, npad=None, **kwargs):
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

        # to do: deal with flattening

        # initialize the inputs and metadata
        self._input = None
        self._metadata = None

        # set parameters 
        self.set_params(kwargs)
        if dataset is not None:
            self.set_params(self.get_params_from_dataset(dataset), override=False)
        default_params = FlyExample.get_default_params()
        self.set_params(default_params, override=False)

        # indices for splitting observation features by type
        self._sensory_feature_idx, self._sensory_feature_szs = \
            get_sensory_feature_shapes(simplify=self._simplify_in)

        if example_in is not None:
            # if example_in is provided, set the inputs from the example
            self.set_example(example_in, dozscore=dozscore)
        elif Xkp is not None:
            # if Xkp is provided, set the inputs from the keypoints
            # use npad to set number of frames to crop from the end manually, 
            # particularly if there is no label sequence associated with this observation object
            self.set_inputs_from_keypoints(Xkp, fly, scale, npad=npad)

        # flattened network stuff -- currently not implemented
        if self._flatten_obs_idx is not None:
            self._flatten_max_dinput = np.max(list(self._flatten_obs_idx.values()))
        else:
            self._flatten_dinput_pertype = np.array(self.d_input)
            self._flatten_max_dinput = self.d_input

        if self._flatten_obs:
            flatten_dinput_pertype = np.array([v[1] - v[0]] for v in self._flatten_obs_idx.values())
            self._flatten_input_type_to_range = np.zeros((self._flatten_dinput_pertype.size, 2), dtype=int)
            cs = np.cumsum(flatten_dinput_pertype)
            self._flatten_input_type_to_range[1:, 0] = cs[:-1]
            self._flatten_input_type_to_range[:, 1] = cs

        return

    @property
    def input(self):
        """
        input
        ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the fly at each time point.
        """
        return self._input
    
    @property
    def metadata(self):
        """
        metadata
        Dictionary with metadata about which fly and video frames the observations were derived from.
        """
        return self._metadata
    
    @property
    def d_input(self):
        """
        d_input
        Number of observation features
        """
        if self._input is None:
            return 0
        else:
            return self._input.shape[-1]
    
    @property
    def pre_sz(self):
        """
        pre_sz
        Size of the input, not empty when storing a batch
        """
        if self._input is None:
            return ()
        else:
            return self._input.shape[:-2]

    @staticmethod
    def flyexample_to_observationinput_params(params):
        """
        flyexample_to_observationinput_params(params) (static)
        Converts parameters from FlyExample to ObservationInputs format.
        """
        kwinputs = copy.deepcopy(params)
        zscore_params_input, _ = FlyExample.split_zscore_params(params['zscore_params'])
        kwinputs['zscore_params'] = zscore_params_input
        return kwinputs

    @staticmethod
    def get_default_params():
        """
        get_default_params() (static)
        Returns the default parameters for ObservationInputs. 
        These are set from FlyExample.get_default_params().
        """
        params = FlyExample.get_default_params()
        params = ObservationInputs.flyexample_to_observationinput_params(params)
        return params

    def get_params(self):
        """
        get_params()
        Returns a dict of the dataset-related parameters. 
        """
        params = {
            'zscore_params': self._zscore_params,
            'simplify_in': self._simplify_in,
            'flatten_obs': self._flatten_obs,
        }
        return params

    def set_params(self, params, override=True):
        """
        set_params(params, override=True)
        Sets the parameters for the ObservationInputs object.
        For each key-value pair in params, sets the attribute with the key to the value.
        """
        for k, v in params.items():
            k = '_' + k
            if override or (not hasattr(self, k)) or (getattr(self, k) is None):
                setattr(self, k, v)

    @staticmethod
    def get_params_from_dataset(dataset):
        """
        get_params_from_dataset(dataset) (static)
        Returns the dataset-related parameters for ObservationInputs computed from input dataset.
        """
        params = FlyExample.get_params_from_dataset(dataset)
        params = ObservationInputs.flyexample_to_observationinput_params(params)
        return params
      
    def get_compute_features_params(self):
        """
        get_compute_features_params()
        Returns the parameters for compute_features, used to compute inputs from keypoints.
        """
        cfparams = {'outtype': np.float32}
        if hasattr(self, '_simplify_in'):
          cfparams['simplify_in'] = self._simplify_in
        if hasattr(self, '_simplify_out'):
          cfparams['simplify_out'] = self._simplify_out
        if hasattr(self, '_dct_m'):
          cfparams['dct_m'] = self._dct_m
        if hasattr(self, '_tspred_global'):
          cfparams['tspred_global'] = self._tspred_global
        if hasattr(self, '_is_velocity'):
          cfparams['compute_pose_vel'] = self._is_velocity
        if hasattr(self, '_discreteidx'):
          cfparams['discreteidx'] = self._discreteidx
        return cfparams
      
    def set_example(self, example_in, dozscore=False):
        """
        set_example(example_in, dozscore=False)
        Sets the inputs from an example.
        Parameters:
        example_in: a dict output from compute_features or a training example
            Required fields:
                'input': ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the fly at each time point
            Optional fields:
                'input_init': ndarray of size (pre_sz x ) 1 x d_input with the observations of the fly on the 
                first frame of the sequence. This should be part of a training example in which the first frame
                has been cropped from input for a causal network. 
                'metadata': dictionary with metadata about which fly and video frames the observations were derived from
        Optional:
        dozscore: Whether to z-score the inputs. Only used if example_in is input. Set this to true if the example input
        should be z-scored, i.e. it is not already z-scored. Default is False. 
        """
        self._input = example_in['input']
        if 'input_init' in example_in:
            # if this is a training example, then the first frame may have been cropped off for temporal alignment 
            # (i.e. starttoff = 1). input_init is this cropped input. Concatenate. 
            self._input = np.concatenate((example_in['input_init'], self._input), axis=-2)

        if dozscore and self.is_zscored():
            # zscore the input
            self._input = zscore(self._input, self._zscore_params['mu_input'], self._zscore_params['sig_input'])

        if 'metadata' in example_in:
            self._metadata = example_in['metadata']
        else:
            self._metadata = None
            
    @property
    def ntimepoints(self):
        """
        ntimepoints
        Number of time points
        """
        if self._input is None:
            return 0
        return self._input.shape[-2]

    def is_zscored(self):
        """
        is_zscored()
        Returns True if the inputs are z-scored, False otherwise.
        """
        return self._zscore_params is not None

    def get_raw_inputs(self, makecopy=True):
        """
        get_raw_inputs(makecopy=True)
        Returns the raw inputs, optionally making a copy.
        Could probably be removed, doesn't do much... 
        """
        if makecopy:
            return self._input.copy()
        else:
            return self._input

    def get_inputs(self, zscored=False, **kwargs):
        """
        get_inputs(zscored=False, makecopy=True)
        Returns the inputs, an ndarray of size (pre_sz x ) ntimepoints x d_input.
        Optional arguments:
        zscored: Whether the inputs should be z-scored. Default is False, i.e. the non-zscored inputs are returned. 
        makecopy: Whether to make a copy of the inputs. Default is True.
        """
        input = self.get_raw_inputs(**kwargs)

        # todo: deal with flattening
        if self.is_zscored() and zscored == False:
            input = unzscore(input, self._zscore_params['mu_input'], self._zscore_params['sig_input'])

        return input

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

    def set_zscore_params(self, zscore_params):
        """
        set_zscore_params(zscore_params)
        Sets the z-score parameters for the inputs. Should only be called by FlyExample.set_zscore_params.
        """
        self._zscore_params = zscore_params
        return

    def add_pose_noise(self, train_inputs, eta_pose, zscored=False):
        """
        add_pose_noise(train_inputs, eta_pose, zscored=False)
        Add noise to the pose features of the inputs. 
        This is copied over from original fly_llm.py code and hasn't been tested.
        """
        idx = self._sensory_feature_idx['pose']
        if self.is_zscored() and (zscored == False):
            eta_pose = zscore(eta_pose, self._zscore_params['mu_input'][..., idx[0]:idx[1]],
                              self._zscore_params['sig_input'][..., idx[0]:idx[1]])
        train_inputs[..., idx[0]:idx[1]] += eta_pose

    def set_inputs(self, input, zscored=False, ts=None):
        """
        set_inputs(input, zscored=False, ts=None)
        Sets the inputs. 
        Parameters:
        inputs: ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the fly at each time point
        Optional parameters:
        zscored: Whether the inputs are z-scored. Default is False, i.e. the inputs are not z-scored. If self.is_zscored(), 
        the stored inputs will be z-scored if z_scored is False.
        ts: Time points to set the inputs. If None, the inputs are set for all time points. 
        """
        if zscored == False and self.is_zscored():
            input = zscore(input, self._zscore_params['mu_input'], self._zscore_params['sig_input'])
        if ts is None:
            self._input = input
        else:
            assert self._input is not None, 'input has not been initialized'
            self._input[..., ts, :] = input
        return

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

    def add_noise(self, train_inputs, input_labels=None, labels=None):
        """
        add_noise(train_inputs, input_labels=None, labels=None)
        Add noise to the pose of train_inputs. Currently not debugged
        """
        assert (np.prod(self.pre_sz) == 1)
        T = self.ntimepoints
        if input_labels is not None:
            d_labels = input_labels.shape[-1]
        else:
            d_labels = self.sensory_feature.szs['pose']
        # additive noise
        eta = np.zeros((T, d_labels))
        do_add_noise = np.random.rand(T) <= self.p_add_input_noise
        eta[do_add_noise, :] = self.input_noise_sigma[None, :] * np.random.randn(np.count_nonzero(do_add_noise),
                                                                                 self.d_output)
        if input_labels is None:
            eta_pose = eta
        else:
            eta_input_labels = labels.next_to_input_labels(eta)
            eta_pose = labels.next_to_nextpose(eta)
            input_labels += eta_input_labels
        train_inputs = self.add_pose_noise(train_inputs, eta_pose)
        return train_inputs, input_labels, eta

    def get_sensory_feature_idx(self):
        """
        get_sensory_feature_idx()
        Returns the indices of the different types of sensory features in the input as a dict
        with keys feature types and values the start and end indices. 
        """
        return get_sensory_feature_idx(self._simplify_in)

    def get_train_inputs(self, input_labels=None, do_add_noise=False, labels=None):
        """
        get_train_inputs(input_labels=None, do_add_noise=False, labels=None)
        Returns the inputs for training. If input_labels is not None, it will concatenate
        the input_labels and the inputs. For causal networks, it will offset these by 
        1 so that input_labels correspond to the previous frame, so 
        train_inputs[...,t,:] is the concatenation of input_labels[...,t,:]  and 
        inputs[...,t+starttoff,:].
        Optional arguments:
        input_labels: ndarray of size (pre_sz x ) ntimepoints x d_input_labels with the input labels. 
        If this is None, then input_labels are not concatenated. Default: None
        do_add_noise: Whether to add noise to the inputs. Default: False. Hasn't been debugged. 
        labels: PoseLabels object. Used to add noise to the pose features. Required if do_add_noise is True.
        Returns:
        res: dictionary with the following keys:
            'input': ndarray of size (pre_sz x ) (ntimepoints-starttoff) x (d_input_labels + d_input) with 
            the inputs for training/evaluating the network. 
            'input_init': ndarray of size (pre_sz x ) starttoff x d_input with the sensory features
            that were cropped off when aligning with the previous frame input labels. 
            'eta': added noise if do_add_noise is True, None otherwise.
        """

        train_inputs = self.get_raw_inputs()

        # makes a copy
        if do_add_noise:
            train_inputs, input_labels, eta = self.add_noise(train_inputs, input_labels, labels)
        else:
            eta = None

        # makes a copy
        train_inputs = torch.tensor(train_inputs)
        train_inputs_init = None

        if not self._flatten_obs:
            # offset input_labels from inputs so that input_labels correspond to the previous frame
            # then concatenate
            if input_labels is not None:
                train_inputs_init = train_inputs[..., :self._starttoff, :]
                train_inputs = torch.cat((torch.tensor(input_labels[..., :self.ntimepoints-self._starttoff, :]),
                                          train_inputs[..., self._starttoff:, :]), dim=-1)
        else:
            # haven't debugged flattened stuff
            ntypes = len(self._flatten_obs_idx)
            flatinput = torch.zeros(self.pre_sz + (self.ntimepoints, ntypes, self._flatten_max_dinput),
                                    dtype=train_inputs.dtype)
            for i, v in enumerate(self._flatten_obs_idx.values()):
                flatinput[..., i,
                self._flatten_input_type_to_range[i, 0]:self._flatten_input_type_to_range[i, 1]] = train_inputs[:,
                                                                                                 v[0]:v[1]]

            train_inputs = flatinput

        return {'input': train_inputs, 'eta': eta, 'input_init': train_inputs_init}

class FlyExample:
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
    
    TODO- add methods for setting from example, keypoints, currently can only be done from constructor.
    TODO- separate out fly specific stuff, rename to AgentExample
    
    """
    def __init__(self, example_in=None, dataset=None, Xkp=None, flynum=None, scale=None, metadata=None,
                 dozscore=False, dodiscretize=False, **kwargs):
        """
        __init__(example_in=None, dataset=None, Xkp=None, flynum=None, scale=None, metadata=None,
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

        # set parameters from extra arguments or dataset
        self.set_params(kwargs)
        if dataset is not None:
            self.set_params(self.get_params_from_dataset(dataset), override=False)
        default_params = FlyExample.get_default_params()
        self.set_params(default_params, override=False)
        if self._dct_m is not None and self._idct_m is None:
            self._idct_m = np.linalg.inv(self._dct_m)

        if example_in is not None:
            
            # copy the example

            # copy the dict, not the arrays
            example_in = {k: v for k, v in example_in.items()}
            # copy the metadata, deep copy
            if (example_in is not None) and ('metadata' in example_in):
                example_in['metadata'] = copy.deepcopy(example_in['metadata'])
        elif Xkp is not None:
            
            # create example from keypoints
            
            # compute pose and observation representations from keypoints
            example_in = self.compute_features(Xkp, flynum, scale)
            # copy the metadata, deep copy
            example_in['metadata'] = copy.deepcopy(metadata)
            # if params set that zscoring and discretizing are to be done, then set 
            # that the example_in requires these
            dozscore = True
            dodiscretize = True

        # whether the input is an example from compute_features or a training example from
        # get_train_example
        is_train_example = (example_in is not None) and ('input' in example_in) \
                           and (type(example_in['input']) is torch.Tensor)

        if is_train_example:
            # concatenate inits if train example
            example_in = dict_convert_torch_to_numpy(example_in)
            # if we offset the example, adjust back metadata
            if ('metadata' in example_in) and \
                (example_in['metadata'] is not None) and \
                ('frame0' in example_in['metadata']):
                if 'labels_init' in example_in:
                    starttoff = example_in['labels_init'].shape[-2]
                elif 'continuous_init' in example_in:
                    starttoff = example_in['continuous_init'].shape[-2]
                else:
                    starttoff = 0
                example_in['metadata']['frame0'] -= starttoff

        # create PoseLabels object from the example
        self._labels = PoseLabels(example_in, dozscore=dozscore, dodiscretize=dodiscretize,
                                 **self.get_poselabel_params())

        
        if is_train_example and self._do_input_labels:
            self._remove_labels_from_input(example_in)

        self._inputs = ObservationInputs(example_in, dozscore=dozscore, **self.get_observationinputs_params())

        self.set_zscore_params(self._zscore_params)

        if (example_in is not None) and ('metadata' in example_in):
            self._metadata = example_in['metadata']

        return
    
    @property
    def labels(self):
        """
        labels
        PoseLabels object with the pose labels
        """
        return self._labels
    
    @property
    def inputs(self):
        """
        inputs
        ObservationInputs object with the input observations
        """
        return self._inputs
    
    @property
    def pre_sz(self):
        """
        pre_sz
        Size of the input, not empty when storing a batch
        """
        return self._labels.pre_sz
    
    @property
    def metadata(self):
        """
        metadata
        Dictionary with metadata about which fly and video frames the observations were derived from.
        """
        return self._metadata
    
    def compute_features(self, Xkp, flynum, scale):
        """
        compute_features(Xkp, flynum, scale)
        Compute sensory and pose label features from keypoints by calling features.compute_features with the 
        correct parameters. 
        Parameters:
        Xkp: ndarray of keypoints for all flies and time points, size (pre_sz x ) nkeypoints x 2 x ntimepoints x nflies
        flynum: index of the main fly
        scale: scale parameters for converting from keypoints to features.
        Output:
        example: dictionary with the computed features:
            'input': ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the fly at each time point
            'labels': ndarray of size (pre_sz x ) ntimepoints x d_multi with the pose representation for the fly at each
            time point.
            'init': ndarray of size (pre_sz x ) ntimepoints x d_next with the initial pose of the fly. 
            'scale': scale parameters for converting from keypoints to features and vice-versa. 
        """

        example = compute_features(Xkp, flynum=flynum, scale_perfly=scale, outtype=np.float32,
                                   simplify_in=self._simplify_in,
                                   simplify_out=self._simplify_out,
                                   dct_m=self._dct_m,
                                   tspred_global=self._tspred_global,
                                   compute_pose_vel=self._is_velocity,
                                   discreteidx=self._discreteidx)

        return example

    def copy(self):
        """
        copy()
        Returns a copy of the FlyExample object. 
        """
        return self.copy_subindex()
    
    def copy_subindex(self, idx_pre=None, ts=None, needinit=True):
        """
        copy_subindex(idx_pre=None, ts=None, needinit=True)
        Returns a copy of the FlyExample object with a subset of the examples (if batched) and/or a subset of the time points.
        Optional parameters:
        idx_pre: indices of the examples to copy. If None, all examples are copied. Only relevant if the example
        is batched. Default is None.
        ts: indices of the time points to copy. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are copied. Default is None.
        needinit: Whether we need the initial pose. If ts[0] is not 0, then we would have to integrate across all
        timepoints up through ts[0] to get the initial pose. If the initial pose will never be used, set needinit to 
        False to avoid this computation. init_pose will then be nan, and things like labels.get_next_keypoints() will return
        nans. Default is True.
        Return value:
        new: a copy of the FlyExample object with the specified indices and time points
        """

        # TODO - this shares a lot of code with PoseLabels.copy_subindex, should be refactored

        # get a copy of the raw data
        example = self.get_raw_example(makecopy=True)

        # if idx_pre is specified, subselect these indices 
        if idx_pre is not None:
            ks = ['continuous', 'discrete', 'todiscretize', 'input', 'init', 'scale', 'categories']
            for k in ks:
                if k in example:
                    example[k] = example[k][idx_pre]
            if example['metadata'] is not None:
              for k in example['metadata'].keys():
                  example['metadata'][k] = example['metadata'][k][idx_pre]

        # if ts is specified, subselect these time points
        if ts is not None:

            # convert ts to ndarray
            if type(ts) is slice:
                ts = range(*ts.indices(self.ntimepoints))
            ts = np.atleast_1d(np.array(ts))
            
            # if ts[0] > 0 and needinit, compute the initial pose by integrating
            toff = ts[0]
            if toff > 0:
                if needinit:
                    next_pose = self.labels.get_next_pose(ts=np.arange(toff+1),use_todiscretize=self.labels.is_todiscretize())
                    init_pose = next_pose[-2:]
                    example['init'] = init_pose.T                    
                else:
                    example['init'][:] = np.nan # set to nans so that we know this is bad data

            # subselect from categories, padding is weird here. 
            # might need debugging, we don't use categories yet
            if example['categories'] is not None:
                cattextra = example['categories'].shape[-1] - example['continuous'].shape[-2]
                if hasattr(ts, '__len__'):
                    example['categories'] = example['categories'][..., ts[0]:ts[-1] + cattextra, :]
                else:
                    example['categories'] = example['categories'][..., ts:ts + cattextra, :]
            
            # subselect main fields
            ks = ['continuous', 'discrete', 'todiscretize', 'input']
            for k in ks:
                if not k in example:
                    continue
                if k == 'discrete':
                    example[k] = example[k][..., ts, :, :]
                else:
                    example[k] = example[k][..., ts, :]
            if (example['metadata'] is not None):
                if 't0' in example['metadata']:
                    example['metadata']['t0'] += toff
                if 'frame0' in example['metadata']:
                    example['metadata']['frame0'] += toff

        # create the new FlyExample from the example dict
        new = FlyExample(example_in=example, **self.get_params())
        return new

    def _remove_labels_from_input(self, example_in):
        """
        _remove_labels_from_input(example_in) (private)
        If example_in is a training example and do_input_labels is True, then the inputs will be the concatenation of
        the previous frame and the current observations. Remove the labels from the input.
        """
        if not self._do_input_labels:
            return

        d_labels = self.labels.get_d_labels_input()
        example_in['input'] = example_in['input'][..., d_labels:]

    def set_params(self, params, override=True):
        """
        set_params(params, override=True)
        Sets the parameters for the FlyExample object. 
        params: Dict of parameters to set. Each key,value pair in the dict will be set as an attribute of the FlyExample object,
        with the key prefixed by an underscore. The exception are those parameters defined in synonyms, which will get different 
        names. 
        override: Whether to override existing parameters. If False, will not overwrite existing parameters. Default is True. 
        """
        synonyms = {'compute_pose_vel': 'is_velocity'}
        for k, v in params.items():
            if k in synonyms:
                k = synonyms[k]
            k = '_' + k
            if override or (not hasattr(self, k)) or (getattr(self, k) is None):
                setattr(self, k, v)

    @staticmethod
    def get_default_params():
        """
        get_default_params()
        Returns the default parameters for the FlyExample object as a dict.
        """

        params = {
            'zscore_params': None,
            'do_input_labels': True,
            'starttoff': 1,
            'flatten_labels': False,
            'flatten_obs': False,
            'discreteidx': [],
            'tspred_global': [1, ],
            'discrete_tspred': [1, ],
            'ntspred_relative': 1,
            'discretize_params': None,
            'is_velocity': False,
            'simplify_out': None,
            'simplify_in': None,
            'flatten_obs_idx': None,
            'dct_m': None,
            'idct_m': None,
        }
        return params

    def get_params(self):
        """
        get_params()
        Returns the parameters for the FlyExample object as a dict.
        """
        default_params = FlyExample.get_default_params()
        params = {k: getattr(self, '_'+k) for k in default_params.keys()}
        return params

    @staticmethod
    def get_params_from_dataset(dataset):
        """
        get_params_from_dataset(dataset)
        Returns the parameters for the FlyExample object taken from a FlyMLMDataset object as a dict.
        """
        params = {
            'zscore_params': dataset.get_zscore_params(),
            'do_input_labels': dataset.input_labels,
            'starttoff': dataset.get_start_toff(),
            'flatten_labels': dataset.flatten_labels,
            'flatten_obs': dataset.flatten_obs,
            'discreteidx': dataset.discretefeat,
            'tspred_global': dataset.tspred_global,
            'discrete_tspred': dataset.discrete_tspred,
            'ntspred_relative': dataset.ntspred_relative,
            'discretize_params': dataset.get_discretize_params(),
            'is_velocity': dataset.compute_pose_vel,
            'simplify_out': dataset.simplify_out,
            'simplify_in': dataset.simplify_in,
            'flatten_obs_idx': dataset.flatten_obs_idx,
            'dct_m': dataset.dct_m,
            'idct_m': dataset.idct_m,
        }
        return params

    def get_poselabel_params(self):
        """
        get_poselabel_params()
        Returns the parameters for the PoseLabels object as a dict.
        """
        params = self.get_params()
        params = PoseLabels.flyexample_to_poselabels_params(params)

        return params

    def get_observationinputs_params(self):
        """
        get_observationinputs_params()
        Returns the parameters for the ObservationInputs object as a dict.
        """
        params = self.get_params()
        params = ObservationInputs.flyexample_to_observationinput_params(params)
        return params

    @property
    def ntimepoints(self):
        """
        ntimepoints
        Number of time points
        """
        # number of time points
        return self.labels.ntimepoints

    # @property
    # def szrest(self):
    #     return self._labels.szrest

    # def get_labels(self):
    #     return self._labels

    # def get_inputs(self):
    #     return self._inputs

    def get_metadata(self, makecopy=True):
        """
        get_metadata(makecopy=True)
        Returns the metadata for the FlyExample object. If makecopy is True, returns a deep copy of the metadata. Default: True.
        """
        if makecopy:
            return copy.deepcopy(self.metadata)
        else:
            return self.metadata

    def get_raw_example(self, format='standard', makecopy=True):
        """
        get_raw_example(format='standard', makecopy=True)
        Returns the raw example for the FlyExample object as a dict. 
        Optional arguments:
        format: Format of the labels. If 'standard', then the key names used within the object are returned. These are 
        'continuous' and 'discrete. If 'original', then it will use the key names that were input when the FlyExample was created.
        Default is 'standard'.
        makecopy: Whether to make a copy of the example. Default is True.
        """
        example = self.labels.get_raw_labels(format=format, makecopy=makecopy)
        example['input'] = self.inputs.get_raw_inputs(makecopy=makecopy)
        example['metadata'] = self.get_metadata(makecopy=makecopy)
        return example

    def get_input_labels(self):
        """
        get_input_labels()
        Returns the input labels for the FlyExample object.
        """
        if self._do_input_labels == False:
            return None
        else:
            return self.labels.get_input_labels()

    def get_n_input_labels(self):
        """
        get_n_input_labels()
        Returns the number of input labels for the FlyExample object.
        """
        if self._do_input_labels:
            return self.labels.get_d_labels_input()
        else:
            return 0

    def get_train_example(self, do_add_noise=False):
        """
        get_train_example(do_add_noise=False)
        Returns the training example consisting of inputs and labels.
        Optional arguments:
        do_add_noise: Whether to add noise to the inputs. Default is False.
        Returns a dictionary with the following keys:
            'input': ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the fly at each time point
            'labels': ndarray of size (pre_sz x ) ntimepoints x d_continuous with the continuous pose labels for the fly
            at each time point
            'labels_discrete': ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the binned pose labels
            for the fly at each time point
            'labels_todiscretize': ndarray of size (pre_sz x ) ntimepoints x d_discrete with the continuous versions of the
            discrete labels. As discrete is non-invertible, this can be used when getting keypoints or other continuous
            representations of the data. 
            'init': ndarray of size (pre_sz x ) d_next with the initial pose of the fly
            'scale': scale parameters for converting from keypoints to features and vice-versa
            'categories': dictionary with the categories from the MABe dataset. Currently not used for anything, may be buggy.
            'metadata': dictionary with metadata about which fly and video frames the observations were derived from
            'input_init': ndarray of size (pre_sz x ) 1 x d_input with the observations of the fly on the first frame of the 
            sequence. This should be part of a training example in which the first frame has been cropped from input for a causal
            network.
            'labels_init': ndarray of size (pre_sz x ) 1 x d_continuous with the initial continuous pose labels for the fly
            'labels_discrete_init': ndarray of size (pre_sz x ) 1 x d_discrete x nbins with the initial discretized pose
            'labels_todiscretize_init': ndarray of size (pre_sz x ) 1 x d_discrete with the initial continuous versions of the
            discrete labels. As discrete is non-invertible, this can be used when getting keypoints or other continuous
            representations of the data. 
            'init_all': ndarray of size (pre_sz x ) starttoff x d_next with the initial pose of the fly
        """

        # to do: add noise
        metadata = self.get_train_metadata()
        input_labels = self.get_input_labels()

        train_inputs = self.inputs.get_train_inputs(input_labels=input_labels,
                                                    labels=self.labels,
                                                    do_add_noise=do_add_noise)
        train_labels = self.labels.get_train_labels(added_noise=train_inputs['eta'])

        flatten = self._flatten_labels or self._flatten_obs
        assert flatten == False, 'flatten not implemented'
        
        res = {'input': train_inputs['input'], 'labels': train_labels['continuous'],
               'labels_discrete': train_labels['discrete'],
               'labels_todiscretize': train_labels['todiscretize'],
               'init': train_labels['init'], 'scale': train_labels['scale'],
               'categories': train_labels['categories'],
               'metadata': metadata,
               'input_init': train_inputs['input_init'],
               'labels_init': train_labels['continuous_init'],
               'labels_discrete_init': train_labels['discrete_init'],
               'labels_todiscretize_init': train_labels['todiscretize_init'],
               'init_all': train_labels['init_all'], }

        return res

    def get_train_metadata(self):
        """
        get_train_metadata()
        Returns the metadata for the training example, offset by starttoff.
        """
        starttoff = self._starttoff
        metadata = self.get_metadata()
        if metadata is None:
            return None
        metadata = copy.deepcopy(metadata)
        metadata['t0'] += starttoff
        metadata['frame0'] += starttoff
        return metadata

    @staticmethod
    def split_zscore_params(zscore_params):
        """
        split_zscore_params(zscore_params) (static)
        Splits the zscore_params into input and labels zscore parameters.
        """
        if zscore_params is not None:
            zscore_params_input = {'mu_input': zscore_params['mu_input'], 'sig_input': zscore_params['sig_input']}
            zscore_params_labels = {'mu_labels': zscore_params['mu_labels'], 'sig_labels': zscore_params['sig_labels']}
        else:
            zscore_params_input = None
            zscore_params_labels = None
        return zscore_params_input, zscore_params_labels

    def set_zscore_params(self, zscore_params):
        """
        set_zscore_params(zscore_params)
        Sets the zscore parameters for the FlyExample object.
        """
        zscore_params_input, zscore_params_labels = FlyExample.split_zscore_params(zscore_params)
        self.inputs.set_zscore_params(zscore_params_input)
        self.labels.set_zscore_params(zscore_params_labels)


class PoseLabels:
    """
    PoseLabels
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
    labels_raw: Dictionary with the raw labels. 
    
    Main methods:
        
    """
    def __init__(self, example_in=None,
                 Xkp=None, scale=None, metadata=None,
                 dozscore=False, dodiscretize=False,
                 dataset=None, **kwargs):

        # set parameters
        self.set_params(kwargs)
        if dataset is not None:
            self.set_params(self.get_params_from_dataset(dataset), override=False)
        default_params = PoseLabels.get_default_params()
        self.set_params(default_params, override=False)

        # initialize
        self._label_keys = {}
        self._labels_raw = {}
        self._pre_sz = None
        self._metadata = metadata
        self._categories = None
        self._init_pose = None

        # initialize from example_in
        if example_in is not None:
            self.set_raw_example(example_in, dozscore=dozscore, dodiscretize=dodiscretize)
        elif Xkp is not None:
            # initialize from keypoints
            self.set_keypoints(Xkp, scale)

        if 'continuous' in self.labels_raw:
            assert self.d_multicontinuous == self.labels_raw['continuous'].shape[-1]
        if self.is_discretized() and 'discrete' in self.labels_raw:
            assert self.d_multidiscrete == self.labels_raw['discrete'].shape[-2]

        return
    
    @property
    def labels_raw(self):
        """
        labels_raw
        Dictionary with the raw labels. 
        """
        return self._labels_raw
    
    @property
    def pre_sz(self):
        """
        pre_sz
        Size of the input, not empty when storing a batch
        """
        return self._pre_sz
    
    @property
    def metadata(self):
        """
        metadata
        Dictionary with metadata about which fly and video frames the observations were derived from.
        """
        return self._metadata
        
    @property
    def zscore_params(self):
        """
        zscore_params
        Z-score parameters for the PoseLabels object. 
        TODO- maybe don't make these publicly available?
        """
        return self._zscore_params

    def __str__(self):
        """
        __str__()
        Returns a string representation of the PoseLabels object. 
        """
        s = f'PoseLabels:\n'
        if len(self.labels_raw) == 0:
            s += 'No data set'
            return s
        s += f'  pre size: {self.pre_sz}\n'
        s += f'  ntimepoints: {self.ntimepoints}\n'
        if self.is_continuous():
            s += f'  continuous dim: {self.labels_raw["continuous"].shape[-1]}\n'
        if self.is_discretized():
            s += f'  discrete dim: {self.labels_raw["discrete"].shape[-2]}\n'
            s += f'  nbins: {self.labels_raw["discrete"].shape[-1]}\n'
        return s

    def set_prediction(self, predin, ts=None, zscored=True, use_todiscretize=None, nsamples=1):
        """
        set_prediction(predin, ts=None, zscored=True, use_todiscretize=None, nsamples=1)
        Sets labels_raw according to the input prediction predin for time points ts, usually an 
        output of the forecasting model. Only fields of labels_raw will be set, not init_pose. This will set
        labels_raw['todiscretize'], either based on predin['todiscretize'] if use_todiscretize==True or
        by creating a continuous version of the discrete features based on nsamples parameter. 
        Parameters:
        predin: dictionary with the prediction. This can be the output of a forecasting model or of
        compute_features. It should have the following fields:
            'continuous': ndarray of size (pre_sz x ) ntimepoints x d_continuous with the continuous pose for the fly
            'discrete': ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the binned pose for the fly
            'todiscretize': ndarray of size (pre_sz x ) ntimepoints x d_discrete with the continuous versions of the 
            discrete pose. 
        Optional parameters:
        ts: indices of the time points to set. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are set (but not init_pose). Default is None.
        zscored: Whether the prediction is z-scored. Default is True. If it is not zscored and self.is_zscored(), then
        the prediction will be z-scored.
        use_todiscretize: Whether to use the continuous versions of the discrete features in predin. If None, then will use 
        'todiscretize' if it is in predin. Default is None.
        nsamples: How to set the continuous version of the discrete features, if use_todiscretize==False. If nsamples=0, then
        the mean of the continuous distribution of the discrete features will be used. If nsamples == 1, then the continuous
        features will be set by sampling from the distribution defined by the discrete features. Default is 1.
        """

        assert nsamples <= 1

        # convert to ndarray if torch tensors
        pred = {k: v.numpy() if type(v) is torch.Tensor else v for k, v in predin.items()}
        
        if ts is None:
            ts = slice(self._starttoff,None)
        
        # store the prediction in labels_raw
        # if discretized and use_todiscretize==False, this will sample from discrete to set labels_raw['todiscretize']
        if use_todiscretize is None:
            use_todiscretize = 'todiscretize' in pred
        multi = self.raw_labels_to_multi(pred, use_todiscretize=use_todiscretize, nsamples=nsamples, zscored=zscored, collapse_samples=True)
        
        # if discretized, send through the discretized values, otherwise sample will be discretized
        # since binning is soft, this will be slightly different than the original discrete
        if self.is_discretized():
            multi_discrete = pred['discrete']
        else:
            multi_discrete = None
        self.set_multi(multi,multi_discrete=multi_discrete,zscored=zscored,ts=ts)
        
        return

    def set_raw_example(self, example_in, dozscore=False, dodiscretize=False):
        """
        set_raw_example(example_in, dozscore=False, dodiscretize=False)
        Sets the labels_raw from the input example_in.
        Parameters:
        example_in: dictionary with the example. This can be the output of compute_features or get_train_example()/get_train_labels()
            Required fields:
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
            'metadata': dictionary with metadata about which fly and video frames the observations were derived from
        """

        # if example_in is None, reinitialize
        if example_in is None:
            self._labels_raw = {}
            self._label_keys = {}
            self._metadata = None
            self._categories = None
            self._pre_sz = None
            self._init_pose = None
            return

        # number of cropped frames at the beginning
        tinit = None

        # continous pose
        if 'labels' in example_in:
            labels_in = example_in['labels']
            self._label_keys['continuous'] = 'labels'
        elif 'continuous' in example_in:
            labels_in = example_in['continuous']
            self._label_keys['continuous'] = 'continuous'
        else:
            labels_in = None
            #raise ValueError('labels_in must contain labels or continuous')
        
        if labels_in is not None:
            # if this is a training example, labels_init/continuous will be the cropped frames
            # from continuous
            if 'labels_init' in example_in and example_in['labels_init'] is not None:
                labels_in = np.concatenate((example_in['labels_init'], labels_in), axis=-2)
                tinit = example_in['labels_init'].shape[-2]
            elif 'continuous_init' in example_in and example_in['continuous_init'] is not None:
                labels_in = np.concatenate((example_in['continuous_init'], labels_in), axis=-2)
                tinit = example_in['continuous_init'].shape[-2]
            self.labels_raw['continuous'] = np.atleast_2d(labels_in)

        # discrete pose
        if 'labels_discrete' in example_in:
            labels_discrete = example_in['labels_discrete']
            self._label_keys['discrete'] = 'labels_discrete'
        elif 'discrete' in example_in:
            labels_discrete = example_in['discrete']
            self._label_keys['discrete'] = 'discrete'
        else:
            labels_discrete = None
        if labels_discrete is not None:
            labels_discrete = np.atleast_3d(labels_discrete)
            # if this is a training example, labels_discrete_init will be the cropped frames
            # from discrete
            if 'labels_discrete_init' in example_in and example_in['labels_discrete_init'] is not None:
                labels_discrete = np.concatenate((example_in['labels_discrete_init'], labels_discrete), axis=-3)
                if tinit is None:
                    tinit = example_in['labels_discrete_init'].shape[-3]
            elif 'discrete_init' in example_in and example_in['discrete_init'] is not None:
                labels_discrete = np.concatenate((example_in['discrete_init'], labels_discrete), axis=-3)
                if tinit is None:
                    tinit = example_in['discrete_init'].shape[-3]
            self.labels_raw['discrete'] = labels_discrete

        # if tinit wasn't set from continuous or discrete data, set to 0
        if tinit is None:
            tinit = 0

        # continuous version of discrete pose
        if 'labels_todiscretize' in example_in:
            labels_todiscretize = example_in['labels_todiscretize']
            self._label_keys['todiscretize'] = 'labels_todiscretize'
        elif 'todiscretize' in example_in:
            labels_todiscretize = example_in['todiscretize']
            self._label_keys['todiscretize'] = 'todiscretize'
        else:
            labels_todiscretize = None
        if labels_todiscretize is not None:
            # if this is a training example, labels_todiscretize_init will be the cropped frames
            # from todiscretize
            labels_todiscretize = np.atleast_2d(labels_todiscretize)
            if 'labels_todiscretize_init' in example_in and example_in['labels_todiscretize_init'] is not None:
                labels_todiscretize = np.concatenate((example_in['labels_todiscretize_init'], labels_todiscretize),
                                                     axis=-2)
            elif 'todiscretize_init' in example_in and example_in['todiscretize_init'] is not None:
                labels_todiscretize = np.concatenate((example_in['todiscretize_init'], labels_todiscretize), axis=-2)
            self.labels_raw['todiscretize'] = labels_todiscretize

        # batch size
        if self.is_continuous():
            self._pre_sz = self.labels_raw['continuous'].shape[:-2]
        else:
            self._pre_sz = self.labels_raw['discrete'].shape[:-3]

        # mask, not used yet, not debugged
        if 'mask' in example_in:
            self.labels_raw['mask'] = np.atleast_1d(example_in['mask'])
            if tinit > 0:
                self.labels_raw['mask'] = np.concatenate(
                    (np.zeros(self.pre_sz + (tinit,), dtype=bool), self.labels_raw['mask']), axis=-1)

        # scale for computing features from keypoints and vv
        if 'scale' in example_in:
            self._scale = example_in['scale']
        if 'metadata' in example_in:
            self._metadata = example_in['metadata']
        else:
            self._metadata = None

        # categories, not used yet, not debugged
        if 'categories' in example_in:
            self._categories = example_in['categories']

        # zscore and discretize if needed
        if dozscore and self.is_zscored():
            self.labels_raw['continuous'] = self.zscore_multi(self.labels_raw['continuous'])
        if dodiscretize and self.is_discretized():
            self.discretize_multi(self.labels_raw)

        # initial pose        
        if 'init_all' in example_in:
          # output of get_train_example/get_train_labels, need to use init_all
          self._init_pose = example_in['init_all']
        elif 'init' in example_in:
            self._init_pose = example_in['init']

    def copy(self):
        """
        copy()
        Returns a copy of the PoseLabels object.
        """
        return self.copy_subindex()

    def copy_subindex(self, idx_pre=None, ts=None, needinit=True):
        """
        copy_subindex(idx_pre=None, ts=None, needinit=True)
        Returns a copy of the PoseLabels object with a subset of the examples (if batched) and/or a subset of the time points.
        Optional parameters:
        idx_pre: indices of the labels to copy. If None, all labels are copied. Only relevant if the labels
        are batched. Default is None.
        ts: indices of the time points to copy. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are copied. Default is None.
        needinit: Whether we need the initial pose. If ts[0] is not 0, then we would have to integrate across all
        timepoints up through ts[0] to get the initial pose. If the initial pose will never be used, set needinit to 
        False to avoid this computation. init_pose will then be nan, and things like labels.get_next_keypoints() will return
        nans. Default is True.
        Return value:
        new: a copy of the FlyExample object with the specified indices and time points
        """

        labels = self.get_raw_labels(makecopy=True)
        labels['metadata'] = self.get_metadata(makecopy=True)
        init_next = self.get_init_pose()

        if idx_pre is not None:
            ks = ['continuous', 'discrete', 'todiscretize', 'init', 'scale', 'categories', 'mask']
            for k in ks:
                if k in labels:
                    labels[k] = labels[k][idx_pre]
            if labels['metadata'] is not None:
                for k in labels['metadata'].keys():
                    labels['metadata'][k] = labels['metadata'][k][idx_pre]
            init_next = init_next[idx_pre]

        if ts is not None:
            
            # convert ts to ndarray
            if type(ts) is slice:
                ts = range(*ts.indices(self.ntimepoints))
            ts = np.atleast_1d(np.array(ts))
            assert np.all(np.diff(ts) == 1), 'ts must be consecutive'
            
            # if ts[0] > 0 and needinit, compute the initial pose by integrating
            toff = ts[0]
            if toff > 0:
                if needinit:
                    next_pose = self.get_next_pose(ts=np.arange(toff+1),use_todiscretize=self.is_todiscretize())
                    init_pose = next_pose[-2:]
                    init_next = init_pose.T
                else:
                    init_next[:] = np.nan # set to nans so that we know this is bad data            
            
            ks = ['continuous', 'discrete', 'todiscretize', 'mask']

            # subselect from categories, padding is weird here. 
            # might need debugging, we don't use categories yet
            if 'categories' in labels and labels['categories'] is not None:
                cattextra = labels['categories'].shape[-1] - labels['continuous'].shape[-2]
            if hasattr(ts, '__len__'):
                assert np.all(np.diff(ts) == 1), 'ts must be consecutive'
                toff = ts[0]
                if 'categories' in labels and labels['categories'] is not None:
                    labels['categories'] = labels['categories'][..., ts[0]:ts[-1] + cattextra, :]
            else:
                toff = ts
                if 'categories' in labels:
                    labels['categories'] = labels['categories'][..., ts:ts + cattextra, :]
                    
            # subselect main fields
            ks = ['continuous', 'discrete', 'todiscretize', 'input']
            for k in ks:
                if k not in labels:
                    continue
                if k == 'discrete':
                    labels[k] = labels[k][..., ts, :, :]
                else:
                    labels[k] = labels[k][..., ts, :]
            if (labels['metadata'] is not None) and ('t0' in labels['metadata']):
                if 't0' in labels['metadata']:
                    labels['metadata']['t0'] += toff
                if 'frame0' in labels['metadata']:
                    labels['metadata']['frame0'] += toff

        # create new PoseLabels object
        new = PoseLabels(example_in=labels, init_next=init_next, **self.get_params())
        return new

    def erase_labels(self):
        """
        erase_labels()
        Sets all labels to nan.
        """
        if self.is_continuous() and 'continuous' in self.labels_raw:
            self.labels_raw['continuous'][..., self._starttoff:, :] = np.nan
        if self.is_discretized():
            if 'discrete' in self.labels_raw:
                self.labels_raw['discrete'][..., self._starttoff:, :, :] = np.nan
            if 'todiscretize' in self.labels_raw:
                self.labels_raw['todiscretize'][..., self._starttoff:, :] = np.nan
        return

    @staticmethod
    def flyexample_to_poselabels_params(params):
        """
        flyexample_to_poselabels_params(params) (static)
        Converts the parameters in the dict params from FlyExample parameters for the PoseLabels object.
        Returns this dict of parameters for PoseLabels.
        """
        if 'zscore_params' in params:
            _, zscore_params_labels = FlyExample.split_zscore_params(params['zscore_params'])
            params['zscore_params'] = zscore_params_labels
        toremove = ['do_input_labels', 'flatten_obs', 'simplify_in', 'flatten_obs_idx']
        for k in toremove:
            if k in params:
                del params[k]
        return params

    @staticmethod
    def get_default_params():
        """
        get_default_params()
        Returns the default parameters for the PoseLabels object.
        """
        params = FlyExample.get_default_params()
        params = PoseLabels.flyexample_to_poselabels_params(params)
        return params

    def get_params_from_dataset(self, dataset):
        """
        get_params_from_dataset(dataset)
        Returns the parameters for the PoseLabels object from the FlyMLMDataset.
        """
        params = FlyExample.get_params_from_dataset(dataset)
        params = PoseLabels.flyexample_to_poselabels_params(params)
        return params

    def get_params(self):
        """
        get_params()
        Returns the parameters for the PoseLabels object.
        """
        kwlabels = {
            'zscore_params': self._zscore_params,
            'discreteidx': self._idx_nextdiscrete_to_next,
            'tspred_global': self._tspred_global,
            'discrete_tspred': self._discrete_tspred,
            'ntspred_relative': self._ntspred_relative,
            'discretize_params': self._discretize_params,
            'is_velocity': self._is_velocity,
            'simplify_out': self._simplify_out,
            'starttoff': self._starttoff,
            'flatten_labels': self._flatten_labels,
            'dct_m': self._dct_m,
            'idct_m': self._idct_m,
        }
        return kwlabels

    def set_params(self, params, override=True):
        """
        set_params(params, override=True)
        Sets the parameters for the PoseLabels object. 
        params: Dict of parameters to set. Each key,value pair in the dict will be set as an attribute of the FlyExample object,
        with the key prefixed by an underscore. The exception are those parameters defined in synonyms, which will get different 
        names. 
        override: Whether to override existing parameters. If False, will not overwrite existing parameters. Default is True. 
        """
        translatedict = {'discreteidx': 'idx_nextdiscrete_to_next'}
        for k, v in params.items():
            if k in translatedict:
                k = translatedict[k]
            k = '_' + k
            if override or (not hasattr(self, k)) or (getattr(self, k) is None):
                setattr(self, k, v)
                
        if hasattr(self,'_discretize_params') \
            and (self._discretize_params is not None) and ('bin_edges' in self._discretize_params) \
            and (self._discretize_params['bin_edges'] is not None):
            self._discretize_nbins = self._discretize_params['bin_edges'].shape[-1] - 1
        else:
            self._discretize_nbins = 0

    def _labels_raw_to_ntimepoints(self,labels_raw):
        """
        _labels_raw_to_ntimepoints(labels_raw)
        Returns the number of time points in the input labels_raw.
        Parameter:
        labels_raw: Dictionary with the raw labels, must have 'continuous' and/or 'discrete' fields.
        """
        # number of time points
        if len(labels_raw) == 0:
            return 0
        if 'continuous' in labels_raw:
            if labels_raw['continuous'].ndim == 0:
                return 0
            elif labels_raw['continuous'].ndim == 1:
                return 1
            else:
                return labels_raw['continuous'].shape[-2]
        else:
            if labels_raw['discrete'].ndim < 2:
                return 0
            elif labels_raw['discrete'].ndim == 2:
                return 1
            else:
                return labels_raw['discrete'].shape[-3]

    @property
    def ntimepoints(self):
        """
        ntimepoints
        Number of time points in the labels.
        """
        return self._labels_raw_to_ntimepoints(self.labels_raw)

    @property
    def ntimepoints_train(self):
        """
        ntimepoints_train
        Number of time points in the train labels, which excludes the initial frames that are cropped.
        """
        return self.ntimepoints - self._starttoff

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

    @property
    def dtype(self):
        """
        dtype
        Data type of the labels.
        """
        if self.is_continuous():
            return self.labels_raw['continuous'].dtype
        else:
            return self.labels_raw['discretized'].dtype

    @property
    def shape(self):
        """
        shape
        Shape of the full labels.
        """
        return self.pre_sz + (self.ntimepoints, self.get_d_labels_full(),)

    @property
    def d_labels_full(self):
        """
        d_labels_full
        Total number of features in the labels (d_multi)
        """
        return self.d_multi

    def is_dct(self):
        """
        is_dct()
        Returns whether the DCT is computed for relative features. 
        """
        return self.ntspred_relative > 1

    def get_init_pose(self, starttoff=None, makecopy=False):
        """
        get_init_pose(starttoff=None, makecopy=False)
        Returns the initial pose of the fly. 
        Optional parameters:
        starttoff: which frame of init_pose to return. If none, it will return all frames. Default is None.
        makecopy: whether to return a copy of the data. Default is False.
        """
        if starttoff is None:
            init_pose = self._init_pose
        else:
            init_pose = self._init_pose[:, starttoff]
        if makecopy:
            init_pose = init_pose.copy()
        return init_pose

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

    def get_metadata(self, makecopy=True):
        """
        get_metadata(makecopy=True)
        Returns the metadata about which fly and video frames the pose labels were derived from.
        Optional parameters:
        makecopy: whether to return a copy of the data. Default is True.
        TODO - maybe consolidate @property metadata with get_metadata
        """
        if makecopy:
            return copy.deepcopy(self.metadata)
        else:
            return self.metadata

    def get_d_labels_input(self):
        """
        get_d_labels_input()
        Returns the number of features concatenated with the input when creating a training example.
        """
        return self.d_next_cossin

    @property
    def _idx_nextglobal_to_next(self):
        """
        _idx_nextglobal_to_next
        Convert from nextglobal indices to next indices. 
        Returns ndarray of indices of next pose that are global
        """
        return np.array(featglobal)

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
    def is_angle_next(self):
        """
        is_angle_next
        Returns a boolean array indicating which features in the next frame pose are angles.
        """
        return featangle

    @property
    def _idx_nextcontinuous_to_next(self):
        """
        _idx_nextcontinuous_to_next
        Convert from nextcontinuous indices to next indices.
        Returns ndarray of indices of next pose (next frame, global + relative) that are continuous
        """
        iscontinuous = np.ones(self.d_next, dtype=bool)
        iscontinuous[self._idx_nextdiscrete_to_next] = False
        return np.nonzero(iscontinuous)[0]

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
        _, d = self._get_idx_nextrelative_to_nextcossinrelative()
        return d

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
    def _idx_nextcossindiscrete_to_nextcossin(self):
        """
        _idx_nextcossindiscrete_to_nextcossin
        Convert from nextcossindiscrete indices to nextcossin indices.
        Returns a list of indices of next cossin pose that are discrete.
        """
        idx_next_to_nextcossin = self._idx_next_to_nextcossin
        idx = np.array([idx_next_to_nextcossin[inext] for inext in self._idx_nextdiscrete_to_next])
        return idx

    @property
    def _idx_nextcossincontinuous_to_nextcossin(self):
        """
        _idx_nextcossincontinuous_to_nextcossin
        Convert from nextcossincontinuous indices to nextcossin indices.
        Returns a list of indices of next cossin pose that are continuous.
        """
        idx = []
        idx_next_to_nextcossin = self._idx_next_to_nextcossin
        for inext in self._idx_nextcontinuous_to_next:
            inextcossin = idx_next_to_nextcossin[inext]
            if type(inextcossin) is np.ndarray:
                idx.extend(inextcossin.tolist())
            else:
                idx.append(inextcossin)
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
    def _idx_nextcossin_to_multi(self):
        """
        _idx_nextcossin_to_multi
        Convert from nextcossin indices to multi indices.
        Returns a list of indices of multi pose that are next cossin pose.
        """
        assert (np.min(self.tspred_global) == 1)
        return self.feattpred_to_multi([(f, 1) for f in range(self.d_next_cossin)])

    @property
    def _idx_multi_to_multifeattpred(self):
        """
        idx_multi_to_multifeattpred
        Convert from multi indices to which feature and frames into the future are predicted. 
        Returns an ndarray of size d_nextcossin x 2.
        """
        # look up table from multi index to (feat,tpred)
        # d_multi x 2 array
        return self.multi_to_feattpred(np.arange(self.d_multi))

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

    @property
    def _multi_isdiscrete(self):
        """
        _multi_isdiscrete
        Returns a boolean array indicating which features in the multi representation are discrete.
        """
        idx_multi_to_multifeattpred = self._idx_multi_to_multifeattpred
        isdiscrete = (np.isin(idx_multi_to_multifeattpred[:, 0], self._idx_nextcossindiscrete_to_nextcossin) & \
                      (idx_multi_to_multifeattpred[:, 1] == 1)) | \
                     (np.isin(idx_multi_to_multifeattpred[:, 0], self._idx_nextcossinglobal_to_nextcossin) & \
                      np.isin(idx_multi_to_multifeattpred[:, 1], self._discrete_tspred))
        return isdiscrete

    @property
    def _idx_multidiscrete_to_multi(self):
        """
        _idx_multidiscrete_to_multi
        Convert from multidiscrete to multi indices.
        Returns indices of multi that correspond to discrete features.
        """
        isdiscrete = self._multi_isdiscrete
        return np.nonzero(isdiscrete)[0]
    
    def get_multi_isdiscrete(self,idx=None):
        """
        get_multi_isdiscrete(idx=None)
        Returns a boolean array indicating which features in the multi representation are discrete.
        Optional parameters:
        idx: indices of the multi features to return. If None, all features are returned. Default is None.
        """
        if idx is None:
            return self._multi_isdiscrete.copy()
        else:
            return self._multi_isdiscrete[idx].copy()

    @property
    def _idx_multicontinuous_to_multi(self):
        """
        _idx_multicontinuous_to_multi
        Convert from multicontinuous to multi indices.
        Returns indices of multi that correspond to continuous features.
        """
        isdiscrete = self._multi_isdiscrete
        return np.nonzero(isdiscrete == False)[0]

    @property
    def _idx_multi_to_multidiscrete(self):
        """
        _idx_multi_to_multidiscrete
        Convert from multi to multidiscrete indices. 
        Returns an ndarray which is -1 if this multi index is not discrete, and the index into multidiscrete otherwise.
        """
        isdiscrete = self.get_multi_isdiscrete()
        idx = np.zeros(self.d_multi, dtype=int)
        idx[:] = -1
        idx[isdiscrete] = np.arange(np.count_nonzero(isdiscrete))
        return idx

    @property
    def _idx_multi_to_multicontinuous(self):
        """
        _idx_multi_to_multicontinuous
        Convert from multi to multicontinuous indices.
        Returns an ndarray which is -1 if this multi index is not continuous, and the index into multicontinuous otherwise.
        """
        iscontinuous = self.get_multi_isdiscrete() == False
        idx = np.zeros(self.d_multi, dtype=int)
        idx[:] = -1
        idx[iscontinuous] = np.arange(np.count_nonzero(iscontinuous))
        return idx

    @property
    def d_multidiscrete(self):
        """
        d_multidiscrete
        Returns the number of discrete features in the multi representation.
        """
        return len(self._idx_multidiscrete_to_multi)

    @property
    def d_multicontinuous(self):
        """
        d_multicontinuous
        Returns the number of continuous features in the multi representation.
        """
        return len(self._idx_multicontinuous_to_multi)

    def feattpred_to_multi(self, ftidx):
        """
        feattpred_to_multi(ftidx)
        Converts from pairs of (feature,tpred) to multi indices.
        ftidx: ndarray of size ... x 2. ftidx[...,0] are the feature indices and [...,1] are the number of frames into the
        future.
        Returns an ndarray of size ... x 1 with the multi indices.
        """
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
        ftidx = unravel_label_index(idx, ntspred_relative=self.ntspred_relative, tspred_global=self.tspred_global,
                                    nrelrep=self.d_next_cossin_relative)
        return ftidx

    def is_zscored(self):
        """
        is_zscored()
        Returns whether the stored labels are zscored.
        """
        return self._zscore_params is not None

    def is_discretized(self):
        """
        is_discretized()
        Returns whether any of the stored labels are discretized.
        """
        return self._discretize_params is not None

    def is_continuous(self):
        """
        is_continuous()
        Returns whether any of the stored labels are continuous.
        """
        return 'continuous' in self.labels_raw

    def is_todiscretize(self):
        """
        is_todiscretize()
        Returns whether the continuous versions of the discrete features are stored.
        """
        return self.is_discretized() and ('todiscretize' in self.labels_raw)

    def is_masked(self):
        """
        is_masked()
        Returns whether the labels have a mask in labels_raw. Currently not used/debugged.
        """
        return 'mask' in self.labels_raw

    def get_raw_labels(self, format='standard', ts=None, makecopy=True):
        """
        get_raw_labels(format='standard', ts=None, makecopy=True)
        Returns the raw labels as a dict. 
        Optional parameters:
        format: whether to return the labels in the standard format or the format used for training. Default is 'standard'.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are returned. Default is None.
        makecopy: whether to return a copy of the data. Default is True.
        """
        labels_out = {}
        for kin in self.labels_raw.keys():
            if format == 'standard':
                kout = kin
            else:
                kout = self._label_keys[kin]
            if makecopy:
                labels_out[kout] = self.labels_raw[kin].copy()
            else:
                labels_out[kout] = self.labels_raw[kin]
            if ts is not None:
                if kin == 'discrete':
                    labels_out[kout] = labels_out[kout][..., ts, :, :]
                else:
                    labels_out[kout] = labels_out[kout][..., ts, :]

        labels_out['init'] = self.get_init_pose(makecopy=makecopy)
        labels_out['scale'] = self.get_scale(makecopy=makecopy)
        labels_out['categories'] = self.get_categories(makecopy=makecopy)

        return labels_out

    def get_raw_labels_tensor_copy(self, **kwargs):
        """
        get_raw_labels_tensor_copy()
        Returns the raw labels as a dict, with the numpy arrays copied and converted to torch tensors.
        """
        raw_labels = self.get_raw_labels(makecopy=False, **kwargs)
        labels_out = {}
        for k, v in raw_labels.items():
            if type(v) is np.ndarray:
                labels_out[k] = torch.tensor(v)
        return labels_out

    def get_ntokens(self):
        """
        get_ntokens()
        For flattened labels, currently not tested/fully implemented. 
        Number of tokens that the labels are broken into.
        """
        return self.d_multidiscrete + int(self.is_continuous())

    def get_flatten_max_doutput(self):
        """
        get_flatten_max_doutput()
        For flattened labels, currently not tested/fully implemented.
        Maximum dimensionality of any output token.
        """
        return np.max(self.d_multicontinuous, self._discretize_nbins)

    def get_train_labels(self, added_noise=None, namingscheme='standard'):
        """
        get_train_labels(do_add_noise=False, namingscheme='standard')
        Returns the training labels as a dict. 
        Optional arguments:
        do_add_noise: Whether to add noise to the inputs. Default is False.
        namingscheme: Whether to use the standard naming scheme ('standard') or the naming scheme used for training ('train'). 
        Default is 'standard'.
        Returns a dictionary with the following keys:
            'labels': ndarray of size (pre_sz x ) ntimepoints x d_continuous with the continuous pose labels for the fly
            at each time point
            'labels_discrete': ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the binned pose labels
            for the fly at each time point
            'labels_todiscretize': ndarray of size (pre_sz x ) ntimepoints x d_discrete with the continuous versions of the
            discrete labels. As discrete is non-invertible, this can be used when getting keypoints or other continuous
            representations of the data. 
            'init': ndarray of size (pre_sz x ) d_next with the initial pose of the fly
            'scale': scale parameters for converting from keypoints to features and vice-versa
            'categories': dictionary with the categories from the MABe dataset. Currently not used for anything, may be buggy.
            'metadata': dictionary with metadata about which fly and video frames the observations were derived from
            sequence. This should be part of a training example in which the first frame has been cropped from input for a causal
            network.
            'labels_init': ndarray of size (pre_sz x ) 1 x d_continuous with the initial continuous pose labels for the fly
            'labels_discrete_init': ndarray of size (pre_sz x ) 1 x d_discrete x nbins with the initial discretized pose
            'labels_todiscretize_init': ndarray of size (pre_sz x ) 1 x d_discrete with the initial continuous versions of the
            discrete labels. As discrete is non-invertible, this can be used when getting keypoints or other continuous
            representations of the data. 
            'init_all': ndarray of size (pre_sz x ) starttoff x d_next with the initial pose of the fly
        """

        # makes a copy
        raw_labels = self.get_raw_labels_tensor_copy()

        # to do: add noise
        assert added_noise is None, 'not implemented'

        # if naming scheme is standard, use keys like 'continuous', 'discrete', etc.
        rename_dict = {k: k for k in  ['discrete', 'continuous', 'todiscretize', 'continuous_init', 'discrete_init', 'todiscretize_init']}
                        
        # if naming scheme is train, use keys like 'labels', 'labels_discrete', etc.
        if namingscheme == 'train':
          rename_dict['discrete'] = 'labels_discrete'
          rename_dict['continuous'] = 'labels'
          rename_dict['todiscretize'] = 'labels_todiscretize'
          rename_dict['continuous_init'] = 'labels_init'
          rename_dict['discrete_init'] = 'labels_discrete_init'
          rename_dict['todiscretize_init'] = 'labels_todiscretize_init'

        train_labels = {}

        # offset labels by starttoff so that we can include previous labels as input for causal llms
        if self.is_discretized():
            train_labels[rename_dict['discrete']] = raw_labels['discrete'][..., self._starttoff:, :, :]
            train_labels[rename_dict['todiscretize']] = raw_labels['todiscretize'][..., self._starttoff:, :]
            train_labels[rename_dict['discrete_init']] = raw_labels['discrete'][..., :self._starttoff, :, :]
            train_labels[rename_dict['todiscretize_init']] = raw_labels['todiscretize'][..., :self._starttoff, :]
        else:
            train_labels[rename_dict['discrete']] = None
            train_labels[rename_dict['todiscretize']] = None
            train_labels[rename_dict['discrete_init']] = None
            train_labels[rename_dict['todiscretize_init']] = None

        # init_all has all frames of init
        train_labels['init_all'] = raw_labels['init']
        train_labels['init'] = raw_labels['init'][..., self._starttoff]
        
        # extra stuff
        # scale: scale parameters for converting from keypoints to features and vice-versa
        train_labels['scale'] = raw_labels['scale']
        
        # categories: categories from MABe, currently not used/debugged
        if 'categories' in raw_labels:
            train_labels['categories'] = raw_labels['categories']
        else:
            train_labels['categories'] = None

        if not self._flatten_labels:
            # offset labels by starttoff so that we can include previous labels as input for causal llms
            train_labels[rename_dict['continuous']] = raw_labels['continuous'][..., self._starttoff:, :]
            train_labels[rename_dict['continuous_init']] = raw_labels['continuous'][..., :self._starttoff, :]
            if 'mask' in raw_labels:
                train_labels['mask'] = raw_labels['mask'][..., self._starttoff:]
        else:
            # flattened rep -- currently not debugged
            contextl = self.ntimepoints
            dtype = raw_labels['continuous'].dtype
            ntokens = self.get_ntokens()
            flatten_max_doutput = self.get_flatten_max_doutput()
            flatlabels = torch.zeros(self.pre_sz + (contextl, ntokens, flatten_max_doutput), dtype=dtype)
            for i in range(self.d_output_discrete):
                # inputnum = self.flatten_nobs_types+i
                flatlabels[..., i, :self._discretize_nbins] = raw_labels['discrete'][..., i, :]
                # newinput[:,inputnum,self.flatten_input_type_to_range[inputnum,0]:self.flatten_input_type_to_range[inputnum,1]] = raw_labels['labels_discrete'][:,i,:]
                # if mask is None:
                #   newmask[:,self.flatten_nobs_types+i] = True
                # else:
                #   newmask[:,self.flatten_nobs_types+i] = mask.clone()
            if self.continuous:
                # inputnum = -1
                flatlabels[..., -1, :self.d_multicontinuous] = raw_labels['continuous']
                # newinput[:,-1,self.flatten_input_type_to_range[inputnum,0]:self.flatten_input_type_to_range[inputnum,1]] = raw_labels['labels']
                # if mask is None:
                #   newmask[:,-1] = True
                # else:
                #   newmask[:,-1] = mask.clone()
            train_labels[rename_dict['continuous']] = flatlabels
            train_labels['continuous_stacked'] = raw_labels['continuous']
            train_labels[rename_dict['continuous_init']] = None

        return train_labels

    def get_mask(self, makecopy=True, ts=None):
        """
        get_mask(makecopy=True, ts=None)
        Returns the mask for the labels. Currently not used/debugged.
        """
        if not self.is_masked():
            return None
        labels_raw = self.get_raw_labels(format='standard', ts=ts, makecopy=makecopy)
        return labels_raw['mask']

    def unzscore_multi(self, multi):
        """
        unzscore_multi(multi)
        Unzscores the input multi labels.
        Parameter:
        multi: ndarray of size (pre_sz x ) ntimepoints x d_multi with the multi labels.
        Output:
        multi: ndarray of size (pre_sz x ) ntimepoints x d_multi with the unzscored multi labels.
        TODO - maybe should be made private?
        """
        if not self.is_zscored():
            return multi
        multi = unzscore(multi, self._zscore_params['mu_labels'], self._zscore_params['sig_labels'])
        return multi

    def zscore_multi(self, multi_unz):
        """
        zscore_multi(multi_unz)
        Zscores the input multi labels.
        Parameter:
        multi_unz: ndarray of size (pre_sz x ) ntimepoints x d_multi with the unzscored multi labels.
        Output:
        multi: ndarray of size (pre_sz x ) ntimepoints x d_multi with the zscored multi labels.
        TODO - maybe should be made private?
        """
        if not self.is_zscored():
            return multi_unz
        multi = zscore(multi_unz, self._zscore_params['mu_labels'], self._zscore_params['sig_labels'])
        return multi

    def labels_discrete_to_continuous(self, labels_discrete, epsilon=1e-3):
        """
        labels_discrete_to_continuous(labels_discrete, epsilon=1e-3)
        Converts discrete labels to continuous labels by taking the weighted sum of the bin centers.
        Parameters:
        labels_discrete: ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the discrete labels.
        epsilon: small number to check that the discrete labels sum to 1. Default is 1e-3.
        Output:
        continuous: ndarray of size (pre_sz x ) ntimepoints x d_discrete with the continuous version of the labels.
        """
        assert self.is_discretized()
        sz = labels_discrete.shape
        nbins = sz[-1]
        nfeat = sz[-2]
        szrest = sz[:-2]
        n = int(np.prod(np.array(szrest)))
        labels_discrete = labels_discrete.reshape((n, nfeat, nbins))

        # nfeat x nbins
        bin_centers = self._discretize_params['bin_medians']
        s = np.sum(labels_discrete, axis=-1)
        assert np.max(np.abs(1 - s)) < epsilon, 'discrete labels do not sum to 1'
        continuous = np.sum(bin_centers[None, ...] * labels_discrete, axis=-1) / s
        continuous = np.reshape(continuous, szrest + (nfeat,))
        return continuous

    def sample_discrete_labels(self, labels_discrete, nsamples=1):
        """
        sample_discrete_labels(labels_discrete, nsamples=1)
        Samples continuous labels from the discrete labels.
        Parameters:
        labels_discrete: ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the discrete labels.
        nsamples: number of samples to take. Default is 1.
        Output:
        continuous: ndarray of size nsamples x (pre_sz x ) ntimepoints with the continuous version of the labels."""
        assert self.is_discretized()

        sz = labels_discrete.shape
        nbins = sz[-1]
        nfeat = sz[-2]
        szrest = sz[:-2]
        n = int(np.prod(np.array(szrest)))
        labels_discrete = labels_discrete.reshape((n, nfeat, nbins))
        bin_samples = self._discretize_params['bin_samples']
        nsamples_per_bin = bin_samples.shape[0]
        continuous = np.zeros((nsamples,) + szrest + (nfeat,), dtype=labels_discrete.dtype)
        for f in range(nfeat):
            # to do make weighted_sample work with numpy directly
            binnum = weighted_sample(torch.tensor(labels_discrete[:, f, :]), nsamples=nsamples).numpy()
            sample = np.random.randint(low=0, high=nsamples_per_bin, size=(nsamples, n))
            curr = bin_samples[sample, f, binnum].reshape((nsamples,) + szrest)
            continuous[..., f] = curr

        return continuous

    def raw_labels_to_multi(self, labels_raw, use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False):
        """
        raw_labels_to_multi(labels_raw, use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False)
        Converts the raw labels dict to multi representation. 
        Parameters:
        labels_raw: dictionary with the raw labels.
        use_todiscretize: whether to use the continuous versions of the discrete labels. Default is False.
        nsamples: How to convert discrete to continuous if use_todiscretize is False. If 0, use the weighted mean of the bin centers. 
        If >0, sample according to bin values. If >1, sample nsamples times. Default is 0.
        Optional parameters:
        use_todiscretize: whether to use the continuous versions of the discrete labels. Default is False.
        nsamples: Method for converting from discrete to continuous. If 0, the weighted mean of bin centers is computed. If > 0,
        specifies the number of samples to take according to the bin distributions. Default is 0.
        zscored: whether to return the z-scored version of multi. If False, multi will be unzscored. Default is False.
        collapse_samples: whether to collapse the samples dimension if nsamples=1 the first dimension. Default is False.
        Returns:
        multi: ndarray of size (nsamples x ) (pre_sz x ) ntimepoints x d_multi with the multi representation of the labels.
        """
        
        # to do: add flattening support here
        
        # allocate multi
        T = self._labels_raw_to_ntimepoints(labels_raw)
        multisz = self.pre_sz + (T, self.d_multi)
        if (nsamples > 1) or (nsamples == 1 and not collapse_samples):
            multisz = (nsamples,) + multisz
        multi = np.zeros(multisz, dtype=labels_raw['continuous'].dtype)
        multi[:] = np.nan

        if self.is_discretized():
            if use_todiscretize:
                assert 'todiscretize' in self.labels_raw
                # shape is pre_sz x T x d_multi_discrete
                labels_discrete = labels_raw['todiscretize']
            elif nsamples > 0:
                # shape is nsamples x pre_sz x T x d_multi_discrete
                labels_discrete = self.sample_discrete_labels(labels_raw['discrete'], nsamples)
                if nsamples == 1 and collapse_samples:
                    labels_discrete = labels_discrete[0, ...]
            else:
                labels_discrete = self.labels_discrete_to_continuous(labels_raw['discrete'])

            # store labels_discrete in multi
            multi[..., self._idx_multidiscrete_to_multi] = labels_discrete

        # get continuous
        multi[..., self._idx_multicontinuous_to_multi] = labels_raw['continuous']

        # unzscore
        if zscored == False and self.is_zscored():
            multi = self.unzscore_multi(multi)

        return multi

    def get_multi(self, use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False, ts=None):
        """
        get_multi(use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False, ts=None)
        Returns the multi representation of the labels.
        Optional parameters:
        use_todiscretize: whether to use the continuous versions of the discrete labels, if available, to 
        convert discrete to continuous. Default is False.
        nsamples: Method for converting from discrete to continuous. If 0, the weighted mean of bin centers is computed. If > 0,
        specifies the number of samples to take according to the bin distributions. Default is 0.
        zscored: whether to return the z-scored version of multi. If False, multi will be unzscored. Default is False.
        collapse_samples: whether to collapse the samples dimension if nsamples=1 the first dimension. Default is False.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are returned. Default is None.
        Returns:
        multi: ndarray of size (nsamples x ) (pre_sz x ) ntimepoints x d_multi with the multi representation of the labels.
        """

        labels_raw = self.get_raw_labels(format='standard', ts=ts, makecopy=False)
        multi = self.raw_labels_to_multi(labels_raw, use_todiscretize=use_todiscretize, nsamples=nsamples, 
                                         zscored=zscored, collapse_samples=collapse_samples)

        return multi

    def get_multi_discrete(self, makecopy=True, ts=None):
        """
        get_multi_discrete(makecopy=True, ts=None)
        Returns the discrete features of the labels. 
        Optional parameters:
        makecopy: whether to return a copy of the data. Default is True.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are returned. Default is None.
        Returns:
        multi_discrete: ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the discrete features.
        """
        if not self.is_discretized():
            nts = len_wrapper(ts, self.ntimepoints)
            return np.zeros((self.pre_sz + (nts, 0, 0)), dtype=self.dtype)
        labels_raw = self.get_raw_labels(format='standard', ts=ts, makecopy=makecopy)
        return labels_raw['discrete']

    def set_multi(self, multi, multi_discrete=None, zscored=False, ts=None):
        """
        set_multi(multi, multi_discrete=None, zscored=False, ts=None)
        Sets the labels, stored in _labels_raw, from the input continuous representation. 
        Parameters:
        multi: ndarray of size (pre_sz x ) ntimepoints x d_multi with the multi representation of the labels.
        Optional parameters:
        multi_discrete: ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the discrete representation of the labels.
        If None, it will be computed from the continuous labels. Default is None.
        zscored: whether the input multi is zscored. If it is not, then multi will be zscored (if self.is_zscored()) before
        storing. Default is False.
        ts: indices of the time points to set. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are set. Default is None.
        """

        multi = np.atleast_2d(multi)

        # zscore
        if self.is_zscored() and (zscored == False):
            multi = self.zscore_multi(multi)

        # get current raw labels
        labels_raw = self.get_raw_labels(format='standard', makecopy=False)

        if ts is None:
            ts = slice(None)
        elif np.isscalar(ts):
            ts = [ts,]

        # set continuous
        labels_raw['continuous'][...,ts,:] = multi[...,self._idx_multicontinuous_to_multi]

        # set discrete
        if self.is_discretized():
            labels_raw['todiscretize'][...,ts,:] = multi[..., self._idx_multidiscrete_to_multi]
            if multi_discrete is None:
                labels_raw['discrete'][...,ts,:,:] = discretize_labels(labels_raw['todiscretize'][..., ts, :],
                                                                    self._discretize_params['bin_edges'],
                                                                    soften_to_ends=True)
            else:
                labels_raw['discrete'][...,ts,:,:] = multi_discrete

        return

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
        multi_idct = np.zeros(multi.shape, dtype=multi.dtype)
        
        idct_m = self.idct_m.T

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

    def get_multi_idct(self, **kwargs):
        """
        get_multi_idct(use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False, ts=None)
        Returns the multi_idct representation of the labels -- the full labels, with relative features un-DCTed.
        Optional parameters:
        use_todiscretize: whether to use the continuous versions of the discrete labels, if available, to
        convert discrete to continuous. Default is False.
        nsamples: Method for converting from discrete to continuous. If 0, the weighted mean of bin centers is computed. If > 0,
        specifies the number of samples to take according to the bin distributions. Default is 0.
        zscored: whether to return the z-scored version of multi. If False, multi will be unzscored. Default is False.
        collapse_samples: whether to collapse the samples dimension if nsamples=1 the first dimension. Default is False.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are returned. Default is None.
        Returns:
        multi_idct: ndarray of size (nsamples x ) (pre_sz x ) ntimepoints x d_multi with the multi_idct representation of the labels.
        """
        multi = self.get_multi(**kwargs)
        return self._multi_to_multiidct(multi)

    def multiidct_to_futurecossinrelative(self, multi_idct, tspred=None):
        """
        multiidct_to_futurecossinrelative(multi_idct, tspred=None)
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

    def get_future_cossin_relative(self, tspred=None, **kwargs):

        multi_idct = self.get_multi_idct(**kwargs)
        futurerelcs = self.multiidct_to_futurecossinrelative(multi_idct, tspred=tspred)
        return futurerelcs

    def get_future_relative(self, tspred=None, **kwargs):
        futurerelcs = self.get_future_cossin_relative(tspred=tspred, **kwargs)
        futurerel = np.moveaxis(self.nextcossinrelative_to_nextrelative(np.moveaxis(futurerelcs, -2, 0)), 0, -2)
        return futurerel

    def get_future_relative_pose(self, tspred=None, **kwargs):
        futurerel = self.get_future_relative(tspred=tspred, **kwargs)
        if not self._is_velocity:
            return futurerel
        relpose0 = self.get_next_pose_relative(**kwargs)
        return futurerel + relpose0[..., :-1, None, :]

    def multi_to_nextcossin(self, multi):
        next_cossin = multi[..., self._idx_nextcossin_to_multi]
        return next_cossin

    def get_nextcossin(self, **kwargs):
        # note that multi_idct is ignored since we don't use the dct representation for the next frame
        multi = self.get_multi(**kwargs)
        return self.multi_to_nextcossin(multi)

    def set_nextcossin(self, nextcossin, **kwargs):
        nextcossin = np.atleast_2d(nextcossin)
        multi = self.get_multi(**kwargs)
        multi[..., self._idx_nextcossin_to_multi] = nextcossin
        self.set_multi(multi, **kwargs)

    def nextcossinglobal_to_nextglobal(self, next_cossinglobal):
        return next_cossinglobal

    def nextcossinrelative_to_nextrelative(self, next_cossin_relative):
        szrest = next_cossin_relative.shape[:-2]
        T = next_cossin_relative.shape[-2]
        n = int(np.prod(szrest))
        next_cossin_relative = next_cossin_relative.reshape((n, T, self.d_next_cossin_relative))
        next_relative = np.zeros((n, T, self.d_next_relative), dtype=next_cossin_relative.dtype)
        idx_nextrelative_to_nextcossinrelative = self._idx_nextrelative_to_nextcossinrelative
        for inext in range(self.d_next_relative):
            inextcossin = idx_nextrelative_to_nextcossinrelative[inext]
            if type(inextcossin) is np.ndarray:
                next_relative[..., inext] = np.arctan2(next_cossin_relative[..., inextcossin[1]],
                                                       next_cossin_relative[..., inextcossin[0]])
            else:
                next_relative[..., inext] = next_cossin_relative[..., inextcossin]
        next_relative = next_relative.reshape(szrest + (T, self.d_next_relative))
        return next_relative

    def nextcossin_to_next(self, next_cossin):
        next = np.zeros(next_cossin.shape[:-1] + (self.d_next,), dtype=next_cossin.dtype)
        next[..., self._idx_nextglobal_to_next] = \
            self.nextcossinglobal_to_nextglobal(next_cossin[..., self._idx_nextcossinglobal_to_nextcossin])
        next[..., self._idx_nextrelative_to_next] = \
            self.nextcossinrelative_to_nextrelative(next_cossin[..., self._idx_nextcossinrelative_to_nextcossin])
        return next

    def next_to_nextcossin(self, next):
        szrest = next.shape[:-1]
        n = np.prod(szrest)
        next_cossin = np.zeros((n, self.d_next_cossin), dtype=next.dtype)
        idx_next_to_nextcossin = self._idx_next_to_nextcossin
        for inext in range(self.d_next):
            inextcossin = idx_next_to_nextcossin[inext]
            if type(inextcossin) is np.ndarray:
                next_cossin[..., inextcossin[0]] = np.cos(next[..., inext])
                next_cossin[..., inextcossin[1]] = np.sin(next[..., inext])
            else:
                next_cossin[..., inextcossin] = next[..., inext]
        return next_cossin

    def next_to_input_labels(self, next):
        return self.next_to_nextcossin(next)

    def get_input_labels(self, **kwargs):
        return self.get_nextcossin(zscored=True, use_todiscretize=True, **kwargs)

    def get_next(self, **kwargs):
        next_cossin = self.get_nextcossin(**kwargs)
        return self.nextcossin_to_next(next_cossin)

    def set_next(self, next, **kwargs):
        nextcossin = self.next_to_nextcossin(next)
        self.set_nextcossin(nextcossin, **kwargs)

    def convert_idx_next_to_nextcossin(self, idx_next):

        if not hasattr(idx_next, '__len__'):
            idx_next = [idx_next, ]

        idx_next_to_nextcossin = self._idx_next_to_nextcossin

        idx_next_cossin = []
        for i in idx_next:
            ic = idx_next_to_nextcossin[i]
            if type(ic) is np.ndarray:
                idx_next_cossin = idx_next_cossin + ic.tolist()
            else:
                idx_next_cossin.append(ic)

        return idx_next_cossin

    def convert_idx_nextcossin_to_multi(self, idx_nextcossin):
        idx_nextcossin_to_multi = self._idx_nextcossin_to_multi
        idx_multi = idx_nextcossin_to_multi[idx_nextcossin]
        return idx_multi

    def convert_idx_next_to_multi(self, idx_next):
        idx_next_cossin = self.convert_idx_next_to_nextcossin(idx_next)
        idx_multi = self.convert_idx_nextcossin_to_multi(idx_next_cossin)

        return idx_multi

    def convert_idx_next_to_multi_anyt(self, idx_next):
        idx_next_cossin = self.convert_idx_next_to_nextcossin(idx_next)
        idx_multi_to_multifeattpred = self._idx_multi_to_multifeattpred
        idx_multi_anyt = np.nonzero(np.isin(idx_multi_to_multifeattpred[:, 0], idx_next_cossin))[0]
        ts = idx_multi_to_multifeattpred[idx_multi_anyt, 1]
        return idx_multi_anyt, ts

    def globalvel_to_globalpos(self, globalvel, starttoff=0, init_pose=None):

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

    def relrep_to_relpose(self, relrep, init_pose=None, starttoff=0):

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

    def next_to_nextpose(self, next, init_pose=None):

        szrest = next.shape[:-2]
        n = int(np.prod(szrest))
        starttoff = 0
        T = next.shape[-2]
        next = next.reshape((n, T, self.d_next))
        globalvel = next[..., self._idx_nextglobal_to_next]
        globalpos = self.globalvel_to_globalpos(globalvel, starttoff=starttoff, init_pose=init_pose)

        relrep = next[..., self._idx_nextrelative_to_next]
        relpose = self.relrep_to_relpose(relrep, init_pose=init_pose, starttoff=starttoff)

        pose = np.concatenate((globalpos, relpose), axis=-1)
        pose[..., self.is_angle_next] = modrange(pose[..., self.is_angle_next], -np.pi, np.pi)

        pose = pose.reshape(szrest + (pose.shape[-2], self.d_next))

        return pose

    def nextpose_to_next(self, nextpose):

        szrest = nextpose.shape[:-2]
        n = int(np.prod(szrest))
        T = nextpose.shape[-2]
        nextpose = nextpose.reshape((n, T, self.d_next))
        init_pose = nextpose[..., 0, :]
        if self._is_velocity:
            next = np.diff(nextpose, axis=1)
        else:
            idx_nextglobal_to_next = self._idx_nextglobal_to_next
            next = nextpose[..., 1:, :].copy()
            next[..., idx_nextglobal_to_next] = np.diff(nextpose[..., idx_nextglobal_to_next], axis=1)
        next[..., self.is_angle_next] = modrange(next[..., self.is_angle_next], -np.pi, np.pi)
        next = next.reshape(szrest + (T - 1, self.d_next))

        return next, init_pose

    def next_to_nextvelocity(self, next):

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
        
        assert zscored == False, 'zscored must be False'
        if (ts is not None) and np.array(ts)[0] > 0:
            assert init_pose is not None, 'init_pose must be provided if ts[0] > 0'

        # only deal with un-zscored data, as we will be concatenating with init, which is not zscored
        next = self.get_next(zscored=False,ts=ts,**kwargs)
        # global will always be velocity, still need to do an integration
        next_pose = self.next_to_nextpose(next, init_pose=init_pose)
        return next_pose

    def next_to_nextrelative(self, next):
        next_relative = next[..., self._idx_nextrelative_to_next]
        return next_relative

    def next_to_nextglobal(self, next):
        next_global = next[..., self._idx_nextglobal_to_next]
        return next_global

    def get_next_pose_relative(self, **kwargs):
        nextpose = self.get_next_pose(**kwargs)
        nextpose_relative = self.next_to_nextrelative(nextpose)
        return nextpose_relative

    def get_next_pose_global(self, **kwargs):
        nextpose = self.get_next_pose(**kwargs)
        nextpose_global = self.next_to_nextglobal(nextpose)
        return nextpose_global

    def set_next_pose(self, nextpose):
        self._pre_sz = nextpose.shape[:-2]
        next, init_pose = self.nextpose_to_next(nextpose)
        self._init_pose = init_pose.T
        self.set_next(next, zscored=False)

    def nextpose_to_nextkeypoints(self, pose):

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

    def get_next_keypoints(self, **kwargs):

        next_pose = self.get_next_pose(**kwargs)
        next_keypoints = self.nextpose_to_nextkeypoints(next_pose)
        return next_keypoints

    def discretize_multi(self, example):
        if not self.is_discretized():
            return
        assert example['continuous'].shape[-1] == self.d_multi
        discretize_idx = self._idx_multidiscrete_to_multi
        example['todiscretize'] = example['continuous'][..., discretize_idx].copy()
        example['discrete'] = discretize_labels(example['todiscretize'], self._discretize_params['bin_edges'],
                                                soften_to_ends=True)
        example['continuous'] = example['continuous'][..., self._idx_multicontinuous_to_multi]
        return

    def set_keypoints(self, Xkp, scale=None):

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

        self.set_raw_example(example,dozscore=self.is_zscored(),dodiscretize=self.is_discretized())
        
        if self._init_pose is None:
          self._init_pose = kp2feat(Xkp[:,:,:2],scale)[...,0]
        
        return
    
    def get_next_velocity(self, **kwargs):

        next = self.get_next(**kwargs)

        # global will always be velocity
        if self._is_velocity:
            return next

        next_vel = self.next_to_nextvelocity(next)

        return next_vel

    def set_zscore_params(self, zscore_params):
        self._zscore_params = zscore_params
        return

    def add_next_noise(self, eta_next, zscored=False):
        next = self.get_next(zscored=zscored)
        next = next + eta_next
        self.set_next(next, zscored=zscored)

    def get_nextglobal_names(self):
        return ['forward', 'sideways', 'orientation']

    def get_nextrelative_names(self):
        idx_nextrelative_to_next = self._idx_nextrelative_to_next
        return [posenames[i] for i in idx_nextrelative_to_next]

    def get_next_names(self):
        next_names = [None, ] * self.d_next
        next_names_global = self.get_nextglobal_names()
        next_names_relative = self.get_nextrelative_names()
        for i, inext in enumerate(self._idx_nextglobal_to_next):
            next_names[inext] = next_names_global[i]
        for i, inext in enumerate(self._idx_nextrelative_to_next):
            next_names[inext] = next_names_relative[i]
        return next_names

    def get_nextcossin_names(self):
        next_names = self.get_next_names()
        idx_next_to_nextcossin = self._idx_next_to_nextcossin
        next_names_cossin = [None, ] * self.d_next_cossin
        for i, ics in enumerate(idx_next_to_nextcossin):
            if hasattr(ics, '__len__'):
                next_names_cossin[ics[0]] = next_names[i] + '_cos'
                next_names_cossin[ics[1]] = next_names[i] + '_sin'
            else:
                next_names_cossin[ics] = next_names[i]
        return next_names_cossin

    def get_multi_names(self):
        ft = self._idx_multi_to_multifeattpred
        ismulti = (np.max(self.tspred_global) > 1) or (self.ntspred_relative > 1)
        multi_names = [None, ] * self.d_multi
        nextcs_names = self.get_nextcossin_names()
        for i in range(self.d_multi):
            if ismulti:
                multi_names[i] = nextcs_names[ft[i, 0]] + '_' + str(ft[i, 1])
        return multi_names

    def select_featidx_plot(self, ntsplot=None, ntsplot_global=None, ntsplot_relative=None):

        idx_multi_to_multifeattpred = self._idx_multi_to_multifeattpred
        idx_multifeattpred_to_multi = self._idx_multifeattpred_to_multi
        ntspred_global = len(self.tspred_global)
        if ntsplot_global is None and ntsplot is not None:
            ntsplot_global = ntsplot
        if ntsplot_global is None or (ntsplot >= ntspred_global):
            idxglobal = self._idx_multiglobal_to_multi
            ftglobal = idx_multi_to_multifeattpred[idxglobal, :]
            ntsplot_global = ntspred_global
        else:
            d_next_global = self.d_next_global
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
        if ntsplot_relative is None and ntsplot is not None:
            ntsplot_relative = ntsplot
        if ntsplot_relative is None or (ntsplot_relative >= ntspred_relative):
            idxrelative = self._idx_multirelative_to_multi
            ftrelative = idx_multi_to_multifeattpred[idxrelative, :]
            ntsplot_relative = ntspred_relative
        elif ntsplot_relative == 0:
            idxrelative = np.zeros((0,), dtype=int)
            ftrelative = np.zeros((0, 2), dtype=int)
        else:
            d_next_cossin_relative = self.d_next_cossin_relative
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
