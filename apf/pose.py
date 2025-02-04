import numpy as np
import torch
import copy
import typing

# if typing.TYPE_CHECKING:
#     from apf.dataset import AgentLLMDataset

from apf.data import weighted_sample, discretize_labels
from apf.utils import len_wrapper, dict_convert_torch_to_numpy, zscore, unzscore, pre_tile_array, pad_axis_array, get_cuda_device

class AgentParams:
    
    def __init__(self):
        return
    
    @classmethod
    def get_params_from_dataset(cls,dataset):
        """
        get_params_from_dataset(dataset)
        Returns the parameters for the AgentExample object taken from a AgentMLMDataset object as a dict.
        """
        if type(dataset) is str:
            return {'todo': dataset}
        
        params = {
            'zscore_params': dataset.get_zscore_params(),
            'do_input_labels': dataset.input_labels,
            'starttoff': dataset.get_start_toff(),
            'flatten_labels': dataset.flatten_labels,
            'flatten_obs': dataset.flatten_obs,
            'discreteidx': dataset.discretefeat,
            'discrete_tspred': dataset.discrete_tspred, # todo make obsolete
            'tspred': [1,], # todo: add this to dataset
            'isdct': False, # todo: add this to dataset
            'discretize_params': dataset.get_discretize_params(),
            'flatten_obs_idx': dataset.flatten_obs_idx,
            'dct_m': dataset.dct_m,
            'idct_m': dataset.idct_m,
        }
        return params
    
    @classmethod
    def get_default_params(cls):
        """
        get_default_params()
        Returns the default parameters for the AgentExample object as a dict.
        """

        params = {
            'zscore_params': None,
            'do_input_labels': True,
            'starttoff': 1,
            'flatten_labels': False,
            'flatten_obs': False,
            'discreteidx': [],
            'tspred': [1,], # list of lists, each sublist is the frames into the future to predict for each feature
            'isdct': False, # can be list-like, with an entry for each feature
            'discrete_tspred': [1, ], # todo make obsolete
            'discretize_params': None,
            'flatten_obs_idx': None,
            'dct_m': None,
            'idct_m': None,
        }
        return params
    
    @classmethod
    def split_zscore_params(cls,zscore_params):
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

    @classmethod
    def example_to_poselabels_params(cls,params):
        """
        example_to_poselabels_params(cls,params) (classmethod)
        Converts the parameters in the dict params from AgentExample parameters for the PoseLabels object.
        Returns this dict of parameters for PoseLabels.
        """
        if 'zscore_params' in params:
            _, zscore_params_labels = cls.split_zscore_params(params['zscore_params'])
            params['zscore_params'] = zscore_params_labels
        toremove = ['do_input_labels', 'flatten_obs', 'simplify_in', 'flatten_obs_idx']
        for k in toremove:
            if k in params:
                del params[k]
        return params
    
    @classmethod
    def example_to_observationinput_params(cls,params):
        """
        example_to_observationinput_params(cls,params) (class)
        Converts parameters from AgentExample to ObservationInputs format.
        """
        kwinputs = params #copy.deepcopy(params)
        zscore_params_input, _ = cls.split_zscore_params(params['zscore_params'])
        kwinputs['zscore_params'] = zscore_params_input
        return kwinputs

class ObservationInputs:
    """
    ObservationInputs
    Class for handling observations/inputs to network
    Represents the observation inputs for an agent for multiple time points. 
    Can be used with batches of observations. 

    Main properties:
    input: ( pre_sz x ) ntimepoints x d_input. ndarray with the observations of the agent at each time point
    metadata: dictionary with metadata about which agent and video frames the observations were derived from
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
    set_inputs_from_keypoints(Xkp, agent, scale=None, ts=None): Sets the inputs from keypoints.
    """

    _paramsClass = AgentParams

    def __init__(self, example_in=None, Xkp=None, agent=0, scale=None, dataset=None, dozscore=False, npad=None, **kwargs):
        """
        Constructor for initializing from an example or from keypoints.

        To initialize from an example computed with features.compute_features or a training example, pass in example_in:
        example_in: dictionary with the example. This can be the output of compute_features or AgentExample.get_train_example(). 
            Required fields:
            'input': ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the agent at each time point
            Optional fields:
            'input_init': ndarray of size (pre_sz x ) 1 x d_input with the observations of the agent on the 
            first frame of the sequence. This should be part of a training example in which the first frame
            has been cropped from input for a causal network. 
            'metadata': dictionary with metadata about which agent and video frames the observations were derived from

        To initialize from keypoints, pass in the following:
        Xkp: ndarray of size (pre_sz x ) ntimepoints x nfeatures x 2 with the keypoints for all flies. 
        agent: index of the main agent.
        scale: scale parameters for converting from keypoints to features.
        
        Optional parameters:
        dataset: AgentMLMDataset object. Used to get parameters for computing features, etc.
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
            discrete_tspred
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
        self.cuda_input = None

        # set parameters 
        self.set_params(kwargs)
        if dataset is not None:
            self.set_params(self.get_params_from_dataset(dataset), override=False)
        default_params = self.get_default_params()
        self.set_params(default_params, override=False)

        # indices for splitting observation features by type
        self._sensory_feature_idx = {}
        self._sensory_feature_szs = {}

        if example_in is not None:
            # if example_in is provided, set the inputs from the example
            self.set_example(example_in, dozscore=dozscore)
        elif Xkp is not None:
            # if Xkp is provided, set the inputs from the keypoints
            # use npad to set number of frames to crop from the end manually, 
            # particularly if there is no label sequence associated with this observation object
            self.set_inputs_from_keypoints(Xkp, agent, scale, npad=npad)

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
        ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the agent at each time point.
        """
        return self._input
    
    @property
    def metadata(self):
        """
        metadata
        Dictionary with metadata about which agent and video frames the observations were derived from.
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

    @classmethod
    def get_default_params(cls):
        """
        get_default_params() (class)
        Returns the default parameters for ObservationInputs. 
        These are set from AgentExample.get_default_params().
        """
        params = cls._paramsClass.get_default_params()
        params = cls._paramsClass.example_to_observationinput_params(params)
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


    @classmethod
    def get_params_from_dataset(cls,dataset):
        """
        get_params_from_dataset(cls,dataset) (classmethod)
        Returns the dataset-related parameters for ObservationInputs computed from input dataset.
        """
        params = cls._paramsClass.get_params_from_dataset(dataset)
        params = cls._paramsClass.example_to_observationinput_params(params)
        return params
    
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
            
    def pre_tile(self,reps, tile_metadata=True):
        """
        pre_tile(reps)
        Tile the observations along the pre_sz dimension(s) by rep. 
        Parameters:
        rep: The number of repetitions along each axis.
        tile_metadata: whether to tile the metadata, default is True
        """
        reps = np.atleast_1d(reps)
        
        nextra = len(reps) - len(self.pre_sz)
        if nextra > 0:
            # add nextra dimensions to the start of each array
            newdims = tuple(range(nextra))
            self._input = np.expand_dims(self._input,newdims)

        ndims = self._input.ndim
        reps1 = np.ones(ndims,dtype=int)
        reps1[:len(reps)] = reps
        self._input = np.tile(self._input, reps1)
        
        if tile_metadata:
            if self._metadata is not None:
                metadata_pre_sz = np.array(self._metadata['t0']).shape
                if metadata_pre_sz != self.pre_sz:
                    if nextra > 0:
                        for k in self._metadata.keys():
                            if self._metadata[k] is not None:
                                self._metadata[k] = np.expand_dims(self._metadata[k],newdims)
                    for k in self._metadata.keys():
                        if self._metadata[k] is not None:
                            self._metadata[k] = np.tile(self._metadata[k], reps)
        
        return

            
    def expand_allocate(self,newT=None,newpre_sz=None):
        if newT is not None:
            Tpad = newT - self.ntimepoints
            self._input = pad_axis_array(self._input, -2, Tpad)
        if newpre_sz is not None:
            self._input = pre_tile_array(self._input, 2, newpre_sz)
                        
        return
    
    def get_compute_features_params(self):
        """
        get_compute_features_params()
        Returns the parameters for compute_features, used to compute inputs from keypoints.
        """
        cfparams = {'outtype': np.float32}
        if hasattr(self, '_dct_m'):
            cfparams['dct_m'] = self._dct_m
        if hasattr(self, '_discreteidx'):
            cfparams['discreteidx'] = self._discreteidx
        return cfparams
            
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
    
    def get_zscore_params(self,makecopy=True):
        """
        get_zscore_params()
        Returns the z-score parameters for the inputs.
        Inputs:
        makecopy: Whether to make a copy of the z-score parameters. Default is True.
        Output:
        zscore_params: dict with keys 'mu_input' and 'sig_input' with the mean and standard deviation of the inputs.
        """
        if self.is_zscored() and makecopy:
            return copy.deepcopy(self._zscore_params)
        else:
            return self._zscore_params
        
    def get_raw_inputs(self, makecopy=True, ts=None, cache=None):
        """
        get_raw_inputs(makecopy=True)
        Returns the raw inputs, optionally making a copy.
        Could probably be removed, doesn't do much... 
        """
        if cache is None:
            input = self._input
        else:
            input = cache
        if ts is not None:
            input = input[...,ts,:]
        if (cache is None) and makecopy:
            return input.copy()
        else:
            return input
        
    def get_raw_inputs_tensor_copy(self, **kwargs):
        """
        get_raw_input_tensor_copy()
        Returns the raw input converted to a torch tensor.
        """
        raw_input = self.get_raw_inputs(makecopy=False, cache=self.cuda_input, **kwargs)
        if self.cuda_input is None:
            raw_input = torch.tensor(raw_input)
        return raw_input

    def set_cuda_cache(self):
        self.cuda_input = torch.tensor(self._input).to(device=get_cuda_device())
                
    def clear_cuda_cache(self):
        del self.cuda_input
        self.cuda_input = None


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
        """
        input = self.get_inputs(**kwargs)
        return {'sensory': input}

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
        Sets the z-score parameters for the inputs. Should only be called by AgentExample.set_zscore_params.
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
        if type(train_inputs) is torch.Tensor:
            eta_pose = torch.tensor(eta_pose,device=train_inputs.device)
        train_inputs[..., idx[0]:idx[1]] += eta_pose

    def set_inputs(self, input, zscored=False, ts=None):
        """
        set_inputs(input, zscored=False, ts=None)
        Sets the inputs. 
        Parameters:
        inputs: ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the agent at each time point
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

    def set_inputs_from_keypoints(self, Xkp, agent, scale, ts=None, npad=None):
        """
        set_inputs_from_keypoints(Xkp, agent, scale=None, ts=None, npad=None)
        Sets the inputs from keypoints. This calls compute_features on the keypoints to compute the sensory features, then
        stores these with self.set_inputs(zscored=False,ts=ts). 
        Inputs:
        Xkp: ndarray of keypoints for all flies and time points, size (pre_sz x ) nkeypoints x 2 x ntimepoints x nflies
        agent: index of the main agent
        scale: scale parameters for converting from keypoints to features
        Optional:
        ts: Time points to set the inputs. If None, the inputs are set for all time points.
        npad: Number of frames to crop from the end of the sequence when computing features.        
        """
        
        pre_sz = Xkp.shape[:-4]
        ntimepoints = len_wrapper(Xkp.shape[-2])
        # set input to be empty
        input = np.zeros(pre_sz+(ntimepoints,0))
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
        return copy.deepcopy(self._sensory_feature_idx)

    def get_train_inputs(self, input_labels=None, do_add_noise=False, labels=None, makecopy=True, ts=None):
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

        # makes a copy
        train_inputs = self.get_raw_inputs_tensor_copy(ts=ts)

        if do_add_noise:
            train_inputs, input_labels, eta = self.add_noise(train_inputs, input_labels, labels)
        else:
            eta = None

        # makes a copy
        train_inputs_init = None
        
        if ts is None:
            n = self.ntimepoints
        else:
            n = len(ts)

        if not self._flatten_obs:
            # offset input_labels from inputs so that input_labels correspond to the previous frame
            # then concatenate
            if input_labels is not None:
                input_labels = input_labels[..., :n-self._starttoff, :]
                if type(input_labels) is np.ndarray:
                    input_labels = torch.tensor(input_labels,device=train_inputs.device)
                train_inputs_init = train_inputs[..., :self._starttoff, :]
                train_inputs = torch.cat((input_labels,train_inputs[..., self._starttoff:, :]), dim=-1)
        else:
            # haven't debugged flattened stuff
            ntypes = len(self._flatten_obs_idx)
            flatinput = torch.zeros(self.pre_sz + (n, ntypes, self._flatten_max_dinput),
                                    dtype=train_inputs.dtype)
            for i, v in enumerate(self._flatten_obs_idx.values()):
                flatinput[..., i,
                self._flatten_input_type_to_range[i, 0]:self._flatten_input_type_to_range[i, 1]] = train_inputs[:,v[0]:v[1]]

            train_inputs = flatinput

        return {'input': train_inputs, 'eta': eta, 'input_init': train_inputs_init}
    
    def get_input_shapes(self):
        idx = copy.deepcopy(self._sensory_feature_idx)
        sz = copy.deepcopy(self._sensory_feature_szs)
        return idx,sz
    
    
    def __str__(self):
        """
        __str__()
        Returns a string representation of the ObservationInputs object. 
        """
        s = f'{self.__class__.__name__}:\n'
        if len(self._input) == 0:
            s += 'No data set'
            return s
        s += f'  pre size: {self.pre_sz}\n'
        s += f'  ntimepoints: {self.ntimepoints}\n'
        s += f'  input dim: {self.d_input}\n'
        return s


class PoseLabels:
    """
    PoseLabels
    Class for handling pose labels for a agent for multiple time points. Can be used with batches.
    
    The data are stored in labels_raw, which is the format that will be used for training and prediction. 
    This is the 'labels' output of compute_features, with the following additions:
    zscoring: if zscore_params is not None, then the labels will be z-scored.
    discretization: if discretize_params is not None, then features in discrete_idx will be discretized. 
    The continuous features are stored in labels_raw['continuous'] and the discrete features are stored in
    labels_raw['discrete']. If available, the continuous versions of the discrete features are stored in
    labels_raw['todiscretize'] when we want to invert feature computations. 
    init_pose stores the initial pose of the agent. Since some of the features represent velocities and/or
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
    'continuous': ndarray of size (pre_sz x ) ntimepoints x d_continuous with the continuous pose for the agent
    'discrete': ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the binned pose for the agent
    'todiscretize': ndarray of size (pre_sz x ) ntimepoints x d_discrete with the continuous versions of the discrete pose.
        
    Main methods:

    __init__: Constructor for initializing from an example or from keypoints.
    
    get_train_labels(): Returns the training labels as a dict. This is the labels portion of the dict ingested by the 
    forecasting model. It offsets labels for causal models as necessary, and cropped frames are stored in keys with
    'init' in their names. There is enough information in this dict to recreate this PoseLabels object. 
    'continuous' will be or size (pre_sz x ) (ntimepoints-starttoff) x d_continuous with the continuous pose for the agent
    and 'discrete' will be of size (pre_sz x ) (ntimepoints-starttoff) x d_discrete x nbins with the binned pose for the agent.
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
    
    _paramsClass = AgentParams
    
    def __init__(self, example_in=None,
                 Xkp=None, scale=None, metadata=None,
                 dozscore=False, dodiscretize=False,
                 dataset=None, init_next=None, **kwargs):

        # set parameters
        default_params = self.get_default_params()
        self.set_params(default_params,updatesizes=False)
        if dataset is not None:
            self.set_params(self.get_params_from_dataset(dataset),updatesizes=False)
        self.set_params(kwargs, updatesizes=True)

        # initialize
        self._label_keys = {}
        self._labels_raw = {}
        self._pre_sz = None
        self._metadata = metadata
        self._categories = None
        self._init_pose = init_next
        self.cuda_labels_raw = None
        self.cuda_input_labels = None
        
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
        Dictionary with metadata about which agent and video frames the observations were derived from.
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
        s = f'{self.__class__.__name__}:\n'
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
            'continuous': ndarray of size (pre_sz x ) ntimepoints x d_continuous with the continuous pose for the agent
            'discrete': ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the binned pose for the agent
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
            'labels' or 'continuous': ndarray of size (pre_sz x ) ntimepoints x d_continuous. Continuous pose labels for the agent
            at each time point. TODO- check that this will work without continuous inputs.
            'labels_discrete' or 'discrete': ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins. Binned pose labels 
            for the agent. TODO- check that this will work without discretized inputs. 
            
            Optional fields:
            'labels_init' or 'continuous_init': ndarray of size (pre_sz x ) tinit x d_continuous with the initial continuous pose
            labels for the agent. Used if the first frame of the sequence has been cropped as a training example for a causal network.
            'labels_discrete_init' or 'discrete_init': ndarray of size (pre_sz x ) tinit x d_discrete x nbins with the initial
            discretized pose. Used if the first frame of the sequence has been cropped as a training example for a causal network.
            'labels_todiscretize' or 'todiscretize': ndarray of size (pre_sz x ) ntimepoints x d_discrete with the continuous
            versions of the discrete labels. As discrete is non-invertible, this can be used when getting keypoints or other 
            continuous representations of the data. 
            'metadata': dictionary with metadata about which agent and video frames the observations were derived from.
            'categories': dictionary with the categories from the MABe dataset. Currently not used for anything, may be buggy. 
            'mask': ndarray of size (pre_sz x ) ntimepoints with a mask for the labels.
            'init_all' or 'init': ndarray of size (pre_sz x ) ntimepoints x d_next with the initial pose of the agent
            'metadata': dictionary with metadata about which agent and video frames the observations were derived from
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

        d_multicontinuous = 0
        d_multidiscrete = 0

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
            d_multicontinuous = labels_in.shape[-1]
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
            d_multidiscrete = labels_discrete.shape[-2]
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
        if 'continuous' in self.labels_raw:
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

        # categories, not used yet, not debugged
        if 'categories' in example_in:
            self._categories = example_in['categories']

        # zscore and discretize if needed
        if dozscore and self.is_zscored():
            self.labels_raw['continuous'] = self.zscore_multi(self.labels_raw['continuous'])
        if dodiscretize and self.is_discretized():
            self._discretize_multi(self.labels_raw)

        # initial pose        
        if 'init_all' in example_in:
          # output of get_train_example/get_train_labels, need to use init_all
          self._init_pose = example_in['init_all']
        elif 'init' in example_in:
            self._init_pose = example_in['init']
                                    
        # # set sizes
        # #self._d_next_discrete = len(self._idx_nextdiscrete_to_next)
        # self._d_multi = d_multicontinuous + d_multidiscrete
        # ntspred_continuous = 1
        # # TODO - maybe allow multi-frame continuous predictions without dct?
        # if self._dct_m is not None:
        #     ntspred_continuous += self._dct_m.shape[0]
        # #self._d_next_continuous = self.d_multicontinuous / ntspred_continuous
        # self._d_next = self.d_next_continuous + self.d_next_discrete
        # np.sum([len(x) for x in self._tspred_nextcossin])

        # # make tspred and isdct have an entry for each d_next        
        # if not hasattr(self._tspred[0],'__len__'):
        #     self._tspred = [self._tspred,]*self._d_next
        # if not hasattr(self._isdct,'__len__'):
        #     self._isdct = np.zeros(self._d_next,dtype=bool)+self._isdct

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
        new: a copy of the AgentExample object with the specified indices and time points
        """

        if ts is not None:            
            # convert ts to ndarray
            if type(ts) is slice:
                ts = range(*ts.indices(self.ntimepoints))
            ts = np.atleast_1d(np.array(ts))
            assert np.all(np.diff(ts) == 1), 'ts must be consecutive'

        labels = self.get_raw_labels(makecopy=True,needinit=False)
        labels['metadata'] = self.get_metadata(makecopy=True)

        if needinit:
            init_next = self.get_init_pose(ts=ts)
        else:
            init_next = np.zeros(self._init_pose.shape)
            init_next[:] = np.nan
        labels['init'] = init_next

        if idx_pre is not None:
            ks = ['continuous', 'discrete', 'todiscretize', 'init', 'scale', 'categories', 'mask']
            for k in ks:
                if k in labels:
                    labels[k] = labels[k][idx_pre]
            if labels['metadata'] is not None:
                for k in labels['metadata'].keys():
                    labels['metadata'][k] = labels['metadata'][k][idx_pre]
            if needinit:
                init_next = init_next[idx_pre]

        if ts is not None:
            
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
        new = self.__class__(example_in=labels, **self.get_params())
        return new

    def pre_tile(self,reps, tile_metadata=True):
        """
        pre_tile(reps   )
        Tile the labels along the pre_sz dimension(s) by rep. 
        Parameters:
        rep: The number of repetitions along each axis.
        tile_metadata: Whether to tile the metadata. Default is True.
        """
        
        reps = np.atleast_1d(reps)
        
        labels = self.get_raw_labels(makecopy=True,needinit=False)
        if tile_metadata:
            labels['metadata'] = self.get_metadata(makecopy=True)
        ks = ['continuous', 'discrete', 'todiscretize', 'init', 'scale', 'categories', 'mask']
        
        nextra = len(reps) - len(self.pre_sz)
        if nextra > 0:
            # add nextra dimensions to the start of each array
            newdims = tuple(range(nextra))
            for k in ks:
                if k in labels and (labels[k] is not None):
                    labels[k] = np.expand_dims(labels[k],newdims)
        
        for k in ks:
            if k in labels and (labels[k] is not None):
                ndims = labels[k].ndim
                reps1 = np.ones(ndims,dtype=int)
                reps1[:len(reps)] = reps
                labels[k] = np.tile(labels[k], reps1)

        if tile_metadata and labels['metadata'] is not None:
            if nextra > 0:
                for k in labels['metadata'].keys():
                    if (labels['metadata'][k] is not None):
                        labels['metadata'][k] = np.expand_dims(labels['metadata'][k],newdims)
            for k in labels['metadata'].keys():
                if labels['metadata'][k] is not None:
                    labels['metadata'][k] = np.tile(labels['metadata'][k], reps)
                
        # will update pre_sz
        self.set_raw_example(labels)
        
        return

    def erase_labels(self,ts=None):
        """
        erase_labels()
        Sets all labels to nan.
        """
        if ts is None:
            ts = slice(self._starttoff,None)
        if self.is_continuous() and 'continuous' in self.labels_raw:
            self.labels_raw['continuous'][..., ts, :] = np.nan
        if self.is_discretized():
            if 'discrete' in self.labels_raw:
                self.labels_raw['discrete'][..., ts, :, :] = np.nan
            if 'todiscretize' in self.labels_raw:
                self.labels_raw['todiscretize'][..., ts, :] = np.nan
        return

    @classmethod
    def get_default_params(cls):
        """
        get_default_params(cls)
        Returns the default parameters for the PoseLabels object.
        """
        params = cls._paramsClass.get_default_params()
        params = cls._paramsClass.example_to_poselabels_params(params)
        return params

    def get_params_from_dataset(self, dataset):
        """
        get_params_from_dataset(dataset)
        Returns the parameters for the PoseLabels object from the AgentMLMDataset.
        """
        params = self._paramsClass.get_params_from_dataset(dataset)
        params = self._paramsClass.example_to_poselabels_params(params)
        return params

    def get_params(self):
        """
        get_params()
        Returns the parameters for the PoseLabels object.
        """
        kwlabels = {
            'zscore_params': self._zscore_params,
            'discreteidx': self._idx_nextdiscrete_to_next,
            'discrete_tspred': self._discrete_tspred, # todo make obsolete
            'discretize_params': self._discretize_params,
            'starttoff': self._starttoff,
            'flatten_labels': self._flatten_labels,
            'dct_m': self._dct_m,
            'idct_m': self._idct_m,
            'tspred': self._tspred, # how many frames into the future to predict for each feature of next
        }
        return kwlabels

    def set_params(self, params, override=True, translatedict=None, updatesizes=True):
        """
        set_params(params, override=True)
        Sets the parameters for the PoseLabels object. 
        params: Dict of parameters to set. Each key,value pair in the dict will be set as an attribute of the AgentExample object,
        with the key prefixed by an underscore. The exception are those parameters defined in synonyms, which will get different 
        names. 
        override: Whether to override existing parameters. If False, will not overwrite existing parameters. Default is True. 
        """

        if translatedict is None:
            translatedict = {}
        translatedict['discreteidx'] = 'idx_nextdiscrete_to_next'
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
            
        if updatesizes:
            self.update_sizes()

    def update_sizes(self):
        """
        update_sizes()
        Update cached sizes, should be called after setting parameters is complete
        """
                    
        # make tspred and isdct have an entry for each d_next        
        self._d_next = len(self._tspred)
        if not hasattr(self._tspred[0],'__len__'):
            self._tspred = [self._tspred,]*self._d_next
        elif len(self._tspred) < self._d_next:
            self._tspred = self._tspred + [self._tspred[-1],]*(self._d_next-len(self._tspred))
        elif len(self._tspred) > self._d_next:
            self._tspred = self._tspred[:self._d_next]
            
        if not hasattr(self._isdct,'__len__'):
            self._isdct = np.zeros(self._d_next,dtype=bool)+self._isdct
        elif len(self._isdct) < self._d_next:
            self._isdct = np.concatenate((self._isdct, np.zeros(self._d_next-len(self._isdct),dtype=bool)))
        elif len(self._isdct) > self._d_next:
            self._isdct = self._isdct[:self._d_next]
        
        self._d_multi = np.sum([len(x) for x in self._tspred_nextcossin])
        # cache indices
        self._idx_multi_to_multifeattpred = self.multi_to_feattpred(np.arange(self.d_multi))
        self._multi_isdiscrete = self.compute_multi_isdiscrete()
        self._idx_nextcossin_to_multi = self.compute_idx_nextcossin_to_multi()

        
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
        return (self._dct_m is not None) and np.any(self._isdct)

    def get_init_pose(self,starttoff=None,makecopy=False,ts=None):
        """
        get_init_pose(starttoff=None, makecopy=False,ts=None)
        Returns the initial pose of the agent. 
        Optional parameters:
        starttoff: which frame of init_pose to return. If none, it will return all frames. Default is None.
        makecopy: whether to return a copy of the data. Default is False.
        ts: time points selected
        """
        if ts is None:
            toff = 0
        else:
            ts = np.atleast_1d(ts)
            toff = ts[0]
        if toff == 0:
            init_pose = self._init_pose
        else:
            next_pose = self.get_next_pose(ts=np.arange(toff+1),use_todiscretize=self.is_todiscretize())
            init_pose = next_pose[-2:].T
        if starttoff is not None:
            init_pose = self._init_pose[:, starttoff]
        if makecopy:
            init_pose = init_pose.copy()
        return init_pose

    # def get_init_pose(self, starttoff=None, makecopy=False):
    #     """
    #     get_init_pose(starttoff=None, makecopy=False)
    #     Returns the initial pose of the agent. 
    #     Optional parameters:
    #     starttoff: which frame of init_pose to return. If none, it will return all frames. Default is None.
    #     makecopy: whether to return a copy of the data. Default is False.
    #     """
    #     if starttoff is None:
    #         init_pose = self._init_pose
    #     else:
    #         init_pose = self._init_pose[:, starttoff]
    #     if makecopy:
    #         init_pose = init_pose.copy()
    #     return init_pose

    def get_metadata(self, makecopy=True):
        """
        get_metadata(makecopy=True)
        Returns the metadata about which agent and video frames the pose labels were derived from.
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
    def d_next_discrete(self):
        """
        d_next_discrete
        Returns the total number of discrete features in the next frame pose.
        """
        return len(self._idx_nextdiscrete_to_next)
    
    @property
    def d_next_continuous(self):
        """
        d_next_continuous
        Returns the total number of continuous features in the next frame pose.
        """
        return len(self._idx_nextcontinuous_to_next)

    @property
    def d_next(self):
        """
        d_next
        Returns the total number of features in the next frame pose.
        """
        return self._d_next
    
    @property
    def d_next_pose(self):
        """
        d_next_pose
        Returns the total number of features in the next frame pose.
        """
        return self.d_next

    @property
    def is_cossinangle_next(self):
        """
        is_angle_next
        Returns a boolean array indicating which features in the next frame pose are angles that will be 
        converted to cos/sin representation
        """
        return np.zeros(self.d_next,dtype=bool)

    @property
    def _idx_nextcontinuous_to_next(self):
        """
        _idx_nextcontinuous_to_next
        Convert from nextcontinuous indices to next indices.
        Returns ndarray of indices of next pose that are continuous
        """
        iscontinuous = np.ones(self.d_next, dtype=bool)
        iscontinuous[self._idx_nextdiscrete_to_next] = False
        return np.nonzero(iscontinuous)[0]

    @property
    def d_next_cossin(self):
        """
        d_next_cossin
        Returns the total number of features in the next cossin representation.
        """
        
        nangle = np.count_nonzero(self.is_cossinangle_next)
        return self.d_next + nangle

    @property
    def _idx_next_to_nextcossin(self):
        """
        _idx_next_to_nextcossin
        Convert from next indices to nextcossin indices.
        Seems kind of involved, maybe we should store this?? 
        """
        idx = []
        off = 0
        for inext in range(self.d_next):
            if self.is_cossinangle_next[inext]:
                idx.append([off, off + 1])
                off += 2
            else:
                idx.append(inext)
                off += 1
        return idx

    @property
    def _idx_nextcossin_to_next(self):
        """
        _idx_nextcossin_to_next
        Convert from nextcossin indices to next indices.
        Returns a list of indices of next pose that are next cossin pose.
        """
        idx = np.zeros(self.d_next_cossin, dtype=int)
        idx_next_to_nextcossin = self._idx_next_to_nextcossin
        for inext,inextcossin in enumerate(idx_next_to_nextcossin):
            idx[inextcossin] = inext
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
    def _isdct_nextcossin(self):
        isdct_nextcossin = np.zeros(self.d_next_cossin, dtype=bool)
        idx_next_to_nextcossin = self._idx_next_to_nextcossin
        for inext,inextcossin in enumerate(idx_next_to_nextcossin):
            isdct_nextcossin[inextcossin] = self._isdct[inext]
        return isdct_nextcossin
        
    @property
    def _tspred_nextcossin(self):
        """
        _tspred_nextcossin
        Returns the number of frames into the future to predict for each feature of next cossin.
        """
        tspred_nextcossin = [None,]*self.d_next_cossin
        idx_nextcossin_to_next = self._idx_nextcossin_to_next
        for inextcossin,inext in enumerate(idx_nextcossin_to_next):
            tspred_nextcossin[inextcossin] = self._tspred[inext]
        return tspred_nextcossin

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
    def d_multi(self):
        """
        d_multi
        Returns the total number of features in the multi representation.
        """
        return self._d_multi

    def compute_idx_nextcossin_to_multi(self):
        """
        compute_idx_nextcossin_to_multi
        Convert from nextcossin indices to multi indices.
        Returns a list of indices of multi pose that are next cossin pose.
        """
        # list of lists
        assert np.max([np.min(ts) for ts in self._tspred]) == 1
        return self.feattpred_to_multi([(f, 1) for f in range(self.d_next_cossin)])

    # @property
    # def _idx_multi_to_multifeattpred(self):
    #     """
    #     idx_multi_to_multifeattpred
    #     Convert from multi indices to which feature and frames into the future are predicted. 
    #     Returns an ndarray of size d_nextcossin x 2.
    #     """
    #     # look up table from multi index to (feat,tpred)
    #     # d_multi x 2 array
    #     return self.multi_to_feattpred(np.arange(self.d_multi))

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

    def compute_multi_isdiscrete(self):
        """
        compute_multi_isdiscrete
        Returns a boolean array indicating which features in the multi representation are discrete.
        """
        
        idx_multi_to_multifeattpred = self._idx_multi_to_multifeattpred
        # features that are discrete in the next frame prediction and either this is the next frame prediction or
        # isdct is false
        isdiscrete = np.isin(idx_multi_to_multifeattpred[:, 0], self._idx_nextcossindiscrete_to_nextcossin) & \
                      ( (idx_multi_to_multifeattpred[:, 1] == 1) | (self._isdct_nextcossin[idx_multi_to_multifeattpred[:,0]]==False) )
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
        
    def get_multi_iscontinuous(self,idx=None):
        """
        get_multi_iscontinuous(idx=None)
        Returns a boolean array indicating which features in the multi representation are continuous.
        Optional parameters:
        idx: indices of the multi features to return. If None, all features are returned. Default is None.
        """
        return self.get_multi_isdiscrete(idx) == False

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

        tspred = self._tspred_nextcossin
        isdct = self._isdct_nextcossin

        ftidx = np.array(ftidx)
        sz = ftidx.shape
        assert sz[-1] == 2
        ftidx = ftidx.reshape((-1, 2))

        idx = np.zeros(len(ftidx),dtype=int)
        ntspred = np.array([len(ts) for ts in tspred])
        off = np.r_[0,np.cumsum(ntspred[:-1])]
        for i in range(len(ftidx)):
            f,t = ftidx[i]
            if isdct[f]:
                t -= 1
            tidx = np.nonzero(tspred[f] == t)[0]
            idx[i] = off[f] + tidx
        
        return idx.reshape(sz[:-1])

    def multi_to_feattpred(self, idx):
        """
        multi_to_feattpred(idx)
        Converts from multi indices to pairs of (feature,tpred).
        idx: ndarray of size ... x 1. idx are the multi indices.
        Returns an ndarray of size ... x 2 with the feature indices and number of frames into the future.
        """
        tspred = self._tspred_nextcossin
        isdct = self._isdct_nextcossin
        
        idx = np.array(idx)
        sz = idx.shape
        idx = idx.flatten()

        ntspred = np.array([len(ts) for ts in tspred])
        off = np.r_[0,np.cumsum(ntspred[:-1])]
        fs = np.searchsorted(off,idx,side='right') - 1
        tidxs = idx - off[fs]
        ts = np.zeros(fs.shape,dtype=int)
        for i in range(len(idx)):
            f = fs[i]
            tidx = tidxs[i]
            ts[i] = tspred[f][tidx] 
            # add 1 -- 0 corresponds to next frame, 1:dct_tau correspond to 1:dct_tau
            if isdct[f]:
                ts[i] += 1

        return np.c_[fs,ts].reshape(sz + (2,))
        
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
    
    def get_raw_labels_basic(self, format='standard', ts=None, makecopy=True, ks=None, cache=None):
        """
        get_raw_labels_basic(ts=None, makecopy=True)
        Returns the raw labels as a dict. 
        Optional parameters:
        format: whether to return the labels in the standard format or the format used for training. Default is 'standard'.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are returned. Default is None.
        makecopy: whether to return a copy of the data. Default is True.
        """
        labels_out = {}
        if cache is None:
            labels_in = self.labels_raw
        else:
            labels_in = cache
            
        if ks is None:
            ks = labels_in.keys()
        for kin in ks:
            if format == 'standard':
                kout = kin
            else:
                kout = self._label_keys[kin]

            if (cache is None) and makecopy:
                labels_out[kout] = labels_in[kin].copy()
            else:
                labels_out[kout] = labels_in[kin]
            if ts is not None:
                if kin == 'discrete':
                    labels_out[kout] = labels_out[kout][..., ts, :, :]
                else:
                    labels_out[kout] = labels_out[kout][..., ts, :]

        return labels_out
        

    def get_raw_labels(self, format='standard', ts=None, makecopy=True, needinit=True,
                       needscale=True, needcategories=True, cache=None):
        """
        get_raw_labels(format='standard', ts=None, makecopy=True)
        Returns the raw labels as a dict. 
        Optional parameters:
        format: whether to return the labels in the standard format or the format used for training. Default is 'standard'.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are returned. Default is None.
        makecopy: whether to return a copy of the data. Default is True.
        """
        labels_out = self.get_raw_labels_basic(format=format, ts=ts, makecopy=makecopy, cache=cache)

        if needinit:
            labels_out['init'] = self.get_init_pose(ts=ts,makecopy=makecopy)
        
        if needscale:
            labels_out['scale'] = self.get_scale(makecopy=makecopy)
        
        if needcategories:
            labels_out['categories'] = self.get_categories(makecopy=makecopy)

        return labels_out

    def set_cuda_cache(self,cache_input_labels=True):
        self.cuda_labels_raw = {}
        raw_labels = self.get_raw_labels_basic(makecopy=False)
        for k, v in raw_labels.items():
            if type(v) is np.ndarray:
                self.cuda_labels_raw[k] = torch.tensor(v).to(get_cuda_device())
        if cache_input_labels:
            self.cuda_input_labels = self.get_input_labels()
            self.cuda_input_labels = torch.tensor(self.cuda_input_labels).to(device=get_cuda_device())
            
    def clear_cuda_cache(self):
        del self.cuda_labels_raw
        del self.cuda_input_labels
        self.cuda_labels_raw = None
        self.cuda_input_labels = None

    def get_raw_labels_tensor_copy(self, **kwargs):
        """
        get_raw_labels_tensor_copy()
        Returns the raw labels as a dict, with the numpy arrays copied and converted to torch tensors.
        """
        raw_labels = self.get_raw_labels(makecopy=False, cache=self.cuda_labels_raw, **kwargs)
        labels_out = {}
        for k, v in raw_labels.items():
            if type(v) is np.ndarray:
                labels_out[k] = torch.tensor(v)
            else:
                labels_out[k] = v
        return labels_out


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

    def get_train_labels(self, added_noise=None, namingscheme='standard',ts=None, needinit=True):
        """
        get_train_labels(do_add_noise=False, namingscheme='standard')
        Returns the training labels as a dict. 
        Optional arguments:
        do_add_noise: Whether to add noise to the inputs. Default is False.
        namingscheme: Whether to use the standard naming scheme ('standard') or the naming scheme used for training ('train'). 
        Default is 'standard'.
        Returns a dictionary with the following keys:
            'labels': ndarray of size (pre_sz x ) ntimepoints x d_continuous with the continuous pose labels for the agent
            at each time point
            'labels_discrete': ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the binned pose labels
            for the agent at each time point
            'labels_todiscretize': ndarray of size (pre_sz x ) ntimepoints x d_discrete with the continuous versions of the
            discrete labels. As discrete is non-invertible, this can be used when getting keypoints or other continuous
            representations of the data. 
            'init': ndarray of size (pre_sz x ) d_next with the initial pose of the agent
            'scale': scale parameters for converting from keypoints to features and vice-versa
            'categories': dictionary with the categories from the MABe dataset. Currently not used for anything, may be buggy.
            'metadata': dictionary with metadata about which agent and video frames the observations were derived from
            sequence. This should be part of a training example in which the first frame has been cropped from input for a causal
            network.
            'labels_init': ndarray of size (pre_sz x ) 1 x d_continuous with the initial continuous pose labels for the agent
            'labels_discrete_init': ndarray of size (pre_sz x ) 1 x d_discrete x nbins with the initial discretized pose
            'labels_todiscretize_init': ndarray of size (pre_sz x ) 1 x d_discrete with the initial continuous versions of the
            discrete labels. As discrete is non-invertible, this can be used when getting keypoints or other continuous
            representations of the data. 
            'init_all': ndarray of size (pre_sz x ) starttoff x d_next with the initial pose of the agent
        """

        # makes a copy
        raw_labels = self.get_raw_labels_tensor_copy(ts=ts,needinit=needinit)

        # to do: add noise
        assert added_noise is None, 'not implemented'

        # if naming scheme is standard, use keys like 'continuous', 'discrete', etc.
        rename_dict = {k: k for k in  ['discrete', 'continuous', 'todiscretize', 'continuous_init', 'discrete_init', 'todiscretize_init']}
                        
        # if naming scheme is train, use keys like 'labels', 'labels_discrete', etc.
        if namingscheme == 'train':
            rename_dict['discrete'] = 'labels_discrete'
            rename_dict['continuous'] = 'labels'
            rename_dict['todiscretize'] = 'labels_todiscretize'
            if needinit:
                rename_dict['continuous_init'] = 'labels_init'
                rename_dict['discrete_init'] = 'labels_discrete_init'
                rename_dict['todiscretize_init'] = 'labels_todiscretize_init'

        train_labels = {}

        # offset labels by starttoff so that we can include previous labels as input for causal llms
        if self.is_discretized():
            train_labels[rename_dict['discrete']] = raw_labels['discrete'][..., self._starttoff:, :, :]
            train_labels[rename_dict['todiscretize']] = raw_labels['todiscretize'][..., self._starttoff:, :]
            if needinit:
                train_labels[rename_dict['discrete_init']] = raw_labels['discrete'][..., :self._starttoff, :, :]
                train_labels[rename_dict['todiscretize_init']] = raw_labels['todiscretize'][..., :self._starttoff, :]
        else:
            train_labels[rename_dict['discrete']] = None
            train_labels[rename_dict['todiscretize']] = None
            if needinit:
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
            if ts is None:
                contextl = self.ntimepoints
            else:
                contextl = len(ts)
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
        labels_raw = self.get_raw_labels_basic(ts=ts, makecopy=makecopy, ks=['mask'])
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

    def zscore_multi_continuous(self,multi_continuous):
        """
        zscore_multi_continuous(multi_continuous)
        Zscores the input multi_continuous labels.
        Parameter:
        multi_continuous: ndarray of size (pre_sz x ) ntimepoints x d_multicontinuous with the continuous multi labels.
        Output:
        multi_continuous: ndarray of size (pre_sz x ) ntimepoints x d_multicontinuous with the zscored multi_continuous labels.
        """
        if not self.is_zscored():
            return multi_continuous
        multi_continuous = zscore(multi_continuous, self._zscore_params['mu_labels'][self._idx_multicontinuous_to_multi], 
                                  self._zscore_params['sig_labels'][self._idx_multicontinuous_to_multi])
        return multi_continuous

    
    def unzscore_multi_continuous(self,multi_continuous):
        """
        unzscore_multi_continuous(multi_continuous)
        Unzscores the input multi_continuous labels.
        Parameter:
        multi_continuous: ndarray of size (pre_sz x ) ntimepoints x d_multicontinuous with the continuous multi labels.
        Output:
        multi_continuous: ndarray of size (pre_sz x ) ntimepoints x d_multicontinuous with the unzscored multi_continuous labels.
        """
        if not self.is_zscored():
            return multi_continuous
        multi_continuous = unzscore(multi_continuous, self._zscore_params['mu_labels'][self._idx_multicontinuous_to_multi], 
                                    self._zscore_params['sig_labels'][self._idx_multicontinuous_to_multi])
        return multi_continuous

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

        s = np.sum(labels_discrete, axis=-1)

        # if the sum is not 1, including if nan, set to 1, but nan out results after
        isbad = np.abs(1 - s) >= epsilon
        s[isbad] = 1.

        # nfeat x nbins
        bin_centers = self._discretize_params['bin_medians']
        continuous = np.sum(bin_centers[None, ...] * labels_discrete, axis=-1) / s
        continuous[isbad] = np.nan
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
            idxgood = binnum >= 0
            ngood = np.count_nonzero(idxgood)
            sample = np.random.randint(low=0, high=nsamples_per_bin, size=(ngood,))
            curr = np.zeros(nsamples*n)
            curr[:] = np.nan
            curr[idxgood.flatten()] = bin_samples[sample, f, binnum[idxgood]]
            curr = curr.reshape((nsamples,) + szrest)
            continuous[..., f] = curr

            # old code that assumed that all weights were good
            # binnum = weighted_sample(torch.tensor(labels_discrete[:, f, :]), nsamples=nsamples).numpy()
            # idxgood = binnum >= 0
            # sample = np.random.randint(low=0, high=nsamples_per_bin, size=(nsamples, n))
            # curr = np.zeros((nsamples,n))
            # curr = bin_samples[sample, f, binnum].reshape((nsamples,) + szrest)
            # continuous[..., f] = curr

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
            if use_todiscretize and 'todiscretize' in self.labels_raw:
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

        labels_raw = self.get_raw_labels_basic(ts=ts, makecopy=False)
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
        labels_raw = self.get_raw_labels_basic(ts=ts, makecopy=makecopy, ks=['discrete'])
        return labels_raw['discrete']

    def get_multi_continuous(self, makecopy=True, ts=None, zscored=False):
        """
        get_multi_continuous(makecopy=True, ts=None)
        Returns the continuous features of the labels.
        Optional parameters:
        makecopy: whether to return a copy of the data. Default is True.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are returned. Default is None.
        Returns:
        multi_continuous: ndarray of size (pre_sz x ) ntimepoints x d_continuous with the continuous features.
        """
        labels_raw = self.get_raw_labels_basic(ts=ts, makecopy=makecopy, ks=['continuous'])
        multi_continuous = labels_raw['continuous']
        if (zscored == False) and self.is_zscored():
            return self.unzscore_multi_continuous(multi_continuous)
        return multi_continuous

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

        if ts is None:
            ts = slice(None)
        elif np.isscalar(ts):
            ts = [ts,]

        # set continuous
        self.labels_raw['continuous'][...,ts,:] = multi[...,self._idx_multicontinuous_to_multi]

        # set discrete
        if self.is_discretized():
            self.labels_raw['todiscretize'][...,ts,:] = multi[..., self._idx_multidiscrete_to_multi]
            if multi_discrete is None:
                self.labels_raw['discrete'][...,ts,:,:] = discretize_labels(self.labels_raw['todiscretize'][..., ts, :],
                                                                            self._discretize_params['bin_edges'],
                                                                            soften_to_ends=True)
            else:
                self.labels_raw['discrete'][...,ts,:,:] = multi_discrete

        return
    
    def expand_allocate(self,newT=None,newpre_sz=None):
        if newT is not None:
            Tpad = newT - self.ntimepoints
            if 'continuous' in self.labels_raw:
                self.labels_raw['continuous'] = pad_axis_array(self.labels_raw['continuous'], -2, Tpad, constant_values=np.nan)                
            if 'discrete' in self.labels_raw:
                self.labels_raw['discrete'] = pad_axis_array(self.labels_raw['discrete'], -3, Tpad, constant_values=np.nan)
            if 'todiscretize' in self.labels_raw:
                self.labels_raw['todiscretize'] = pad_axis_array(self.labels_raw['todiscretize'], -2, Tpad, constant_values=np.nan)
        if newpre_sz is not None:
            if 'continuous' in self.labels_raw:
                self.labels_raw['continuous'] = pre_tile_array(self.labels_raw['continuous'],2,newpre_sz)
            if 'discrete' in self.labels_raw:
                self.labels_raw['discrete'] = pre_tile_array(self.labels_raw['discrete'],3,newpre_sz)
            if 'todiscretize' in self.labels_raw:
                self.labels_raw['todiscretize'] = pre_tile_array(self.labels_raw['discretize'],2,newpre_sz)
            metadata = self.get_metadata()
            if metadata is not None:
                for k in metadata.keys():
                    metadata[k] = pre_tile_array(metadata[k],1,newpre_sz)
        return

    def _multi_to_multiidct1(self, multi):
        """
        multi_to_multiidct(multi)
        Performs the inverse DCT on continuous pose features of multi to convert to the multi_idct representation, if applicable. 
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
        dct_tau = idct_m.shape[0]

        # features to convert
        isdct = self._isdct_nextcossin
        fs = np.nonzero(isdct)[0]
        ftcurr = np.zeros((dct_tau,2),dtype=int)
        ftcurr[:,1] = np.arange(2,dct_tau+2)

        for f in fs:
            ftcurr[:,0] = f
            idx_multi = self.feattpred_to_multi(ftcurr)
            multi_curr = multi[..., idx_multi].reshape((-1, dct_tau))
            multi_idct[..., idx_multi] = (multi_curr @ idct_m).reshape((multi.shape[:-1]) + (dct_tau,))
        
        return multi_idct

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

    def _multi_to_nextcossin(self, multi):
        """
        _multi_to_nextcossin(multi)
        Convert from the full multi (or multi_idct) representation to the next_cossin representation -- both global
        and relative features, but only for the next frame prediction. multi and multi_idct both work, as we don't use
        the next frame prediction in the DCT representation.
        Parameters:
        multi: ndarray of size (pre_sz x ) ntimepoints x d_multi with the multi representation of the labels.
        Returns:
        next_cossin: ndarray of size (pre_sz x ) ntimepoints x d_next_cossin with the next_cossin representation of the labels.
        """
        next_cossin = multi[..., self._idx_nextcossin_to_multi]
        return next_cossin

    def get_nextcossin(self, **kwargs):
        """
        get_nextcossin(use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False, ts=None)
        Returns the next_cossin representation of the labels -- both global and relative features, but only for the next 
        frame prediction. Angles in the relative features will be in the cos,sin representaiton. 
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
        next_cossin: ndarray of size (pre_sz x ) ntimepoints x d_next_cossin with the next_cossin representation of the labels.
        """
        
        # note that multi_idct is ignored since we don't use the dct representation for the next frame
        multi = self.get_multi(**kwargs)
        return self._multi_to_nextcossin(multi)

    def set_nextcossin(self, nextcossin, use_todiscretize=True,**kwargs):
        """
        set_nextcossin(nextcossin, zscored=False, ts=None, nsamples=0)
        Sets the next_cossin representation of the labels. This will only update the next frame labels, not the 
        future frame labels. Note that if there are future features that are discrete and self.is_todiscretize() == False, then 
        todiscretize for those features will be set based on sampling strategies described by nsamples. 
        Parameters:
        nextcossin: ndarray of size (pre_sz x ) ntimepoints x d_next_cossin with the next_cossin representation of the labels.
        Optional parameters:
        zscored: whether the input nextcossin is zscored. If it is not, then nextcossin will be zscored (if self.is_zscored()) before
        storing. Default is False.
        nsamples: Method for converting from discrete to continuous when getting multi to fill in. If 0, the weighted mean of bin 
        centers is computed. If 1, then it will sample from the bin distributions. Default is 0.
        """
        if self.is_todiscretize():
            assert use_todiscretize == True
        nextcossin = np.atleast_2d(nextcossin)
        # get all multi features
        # if self.is_todiscretize() == False, then get_multi will convert from discrete to continuous for all features. 
        # This will get overwritten for the next frame features, but not for the future frame features.
        # TODO - maybe do this better?
        multi = self.get_multi(use_todiscretize=True,**kwargs)
        # fill in next features
        multi[..., self._idx_nextcossin_to_multi] = nextcossin
        self.set_multi(multi, **kwargs)

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
        idx_next_to_nextcossin = self._idx_next_to_nextcossin
        for inext in range(self.d_next):
            # indices of the next_cossin features that correspond to inext
            inextcossin = idx_next_to_nextcossin[inext]
            if np.isnumeric(inextcossin):
                # scalar then just copy
                next[..., inext] = next_cossin[..., inextcossin]
            else:
                # if two indices, then we need to convert from cos,sin to radians
                next[..., inext] = np.arctan2(next_cossin[..., inextcossin[1]], next_cossin[..., inextcossin[0]])
        return next

    def _next_to_nextcossin(self, next):
        """
        _next_to_nextcossin(next)
        Convert from the next representation to the next_cossin representation. This will convert the relative angles in radians
        to cos,sin.
        Parameters:
        next: ndarray of size (pre_sz x ) ntimepoints x d_next with the next representation of the labels.
        Returns:
        next_cossin: ndarray of size (pre_sz x ) ntimepoints x d_next_cossin with the next_cossin representation of the labels.
        """
        szrest = next.shape[:-1]
        n = np.prod(szrest)
        next_cossin = np.zeros((n, self.d_next_cossin), dtype=next.dtype)
        next = next.reshape((n, self.d_next))
        idx_next_to_nextcossin = self._idx_next_to_nextcossin
        for inext in range(self.d_next):
            # indices of the next_cossin features that correspond to inext
            inextcossin = idx_next_to_nextcossin[inext]
            # if two indices, then we need to convert from radians to cos,sin
            if type(inextcossin) is np.ndarray:
                next_cossin[..., inextcossin[0]] = np.cos(next[..., inext])
                next_cossin[..., inextcossin[1]] = np.sin(next[..., inext])
            else:
                # otherwise just copy
                next_cossin[..., inextcossin] = next[..., inext]
        next_cossin = next_cossin.reshape(szrest + (self.d_next_cossin,))
        return next_cossin

    def next_to_input_labels(self, next):
        """
        next_to_input_labels(next)
        Convert from the next representation to the input_labels representation. This will convert the relative angles in radians.
        This is currently only used by ObservationInputs class for adding noise. 
        Parameters:
        next: ndarray of size (pre_sz x ) ntimepoints x d_next with the next representation of the labels.
        Returns:
        input_labels: ndarray of size (pre_sz x ) ntimepoints x d_input_labels with the input_labels representation of the labels.
        """
        return self._next_to_nextcossin(next)

    def get_input_labels(self,ts=None):
        """
        get_input_labels()
        Returns the labels that will be input from the previous frame to the model. 
        This will be an ndarray of size (pre_sz x ) ntimepoints x d_input_labels.
        """
        assert (not self.is_discretized) or self.is_todiscretize(), 'get_input_labels only works when todiscretize is set'
        # if ts is not None:
        #     ts = np.atleast_1d(ts)
        #     ts = np.r_[ts[0]-1,ts]
        
        if self.cuda_input_labels is not None:
            return self.cuda_input_labels[...,ts,:]
        
        return self.get_nextcossin(zscored=self.is_zscored(), use_todiscretize=True, ts=ts)

    def get_next(self, **kwargs):
        """
        get_next(use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False, ts=None)
        Returns the next-frame features. 
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
        next: ndarray of size (pre_sz x ) ntimepoints x d_next with the next representation of the labels.
        """
        next_cossin = self.get_nextcossin(**kwargs)
        return self._nextcossin_to_next(next_cossin)

    def set_next(self, next, **kwargs):
        """
        set_next(next, zscored=False, ts=None, nsamples=0)
        Sets the next representation of the labels. This will only update the next frame labels, not the 
        future frame labels. Note that if there are future features that are discrete and self.is_todiscretize() == False, then 
        todiscretize for those features will be set based on sampling strategies described by nsamples. 
        Parameters:
        next: ndarray of size (pre_sz x ) ntimepoints x d_next with the next representation of the labels.
        Optional parameters:
        zscored: whether the input next is zscored. If it is not, then next will be zscored (if self.is_zscored()) before
        storing. Default is False.
        nsamples: Method for converting from discrete to continuous when getting multi to fill in. If 0, the weighted mean of bin 
        centers is computed. If 1, then it will sample from the bin distributions. Default is 0.
        """
        
        # convert to nextcossin
        nextcossin = self._next_to_nextcossin(next)
        # set nextcossin
        self.set_nextcossin(nextcossin, **kwargs)

    def _convert_idx_next_to_nextcossin(self, idx_next):
        """
        _convert_idx_next_to_nextcossin(idx_next)
        Convert the input indices for the next representation to the indices for the next_cossin representation.
        The return value will be a list of all indices, with correspondences potentially lost. 
        Parameters:
        idx_next: list/ndarray/scalar of indices for the next representation
        Returns:
        idx_next_cossin: list of indices for the next_cossin representation
        """

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

    def _convert_idx_nextcossin_to_multi(self, idx_nextcossin):
        """
        _convert_idx_nextcossin_to_multi(idx_nextcossin)
        Convert the input indices for the next_cossin representation to the indices for the multi representation.
        Parameters:
        idx_nextcossin: ndarray of indices for the next_cossin representation.
        Returns:
        idx_multi: ndarray of indices for the multi representation.
        """
        idx_nextcossin_to_multi = self._idx_nextcossin_to_multi
        idx_multi = idx_nextcossin_to_multi[idx_nextcossin]
        return idx_multi

    def _convert_idx_next_to_multi(self, idx_next):
        """
        _convert_idx_next_to_multi(idx_next)
        Convert the input indices for the next representation to the indices for the multi representation.
        Parameters:
        idx_next: list/ndarray/scalar of indices for the next representation
        Returns:
        idx_multi: ndarray of indices for the multi representation.
        """
        idx_next_cossin = self._convert_idx_next_to_nextcossin(idx_next)
        idx_multi = self._convert_idx_nextcossin_to_multi(idx_next_cossin)

        return idx_multi

    def get_multiidx_for_featidx(self, featidx):
        """
        get_multiidx_for_featidx(idx_next)
        Returns the indices of multi, and the associated tspred, for each feature in featidx.
        Correspondences may be lost, so this probably makes the most sense used with featidx
        as a scalar. 
        Parameters:
        featidx: list/ndarray/scalar of indices for the next representation
        Returns:
        idx_multi: ndarray of indices for the multi representation.
        ts: ndarray of tspred for each feature in featidx.
        """
        idx_next_cossin = self._convert_idx_next_to_nextcossin(featidx)
        idx_multi_to_multifeattpred = self._idx_multi_to_multifeattpred
        idx_multi_anyt = np.nonzero(np.isin(idx_multi_to_multifeattpred[:, 0], idx_next_cossin))[0]
        ts = idx_multi_to_multifeattpred[idx_multi_anyt, 1]
        return idx_multi_anyt, ts

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
        
        if init_pose is None:
            init_pose = self.get_init_pose()

        nextpose = np.concatenate((init_pose, next), axis=-2)

        return nextpose

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

        next = nextpose[...,1:,:]
        init_pose = nextpose[...,[0,],:]

        return next, init_pose

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

    @property
    def n_keypoints(self):
        return self._d_next//2

    def _nextpose_to_nextkeypoints(self, pose):
        """
        _nextpose_to_nextkeypoints(pose)
        Convert from the next pose representation to the next keypoints representation. 
        Parameters:
        pose: ndarray of size (pre_sz x ) ntimepoints x d_next with the next pose representation of the labels.
        Returns:
        kp: ndarray of size (pre_sz x ) ntimepoints x nkpts x 2 with the next keypoints representation of the labels.
        """

        kp = pose.reshape(pose.shape[:-1] + (pose.shape[-1]//2, 2))
        
        return kp

    def get_next_keypoints(self, **kwargs):
        """
        get_next_keypoints(use_todiscretize=False, nsamples=0, collapse_samples=False, ts=None)
        Returns the labels in keypoint representation.
        Optional parameters:
        use_todiscretize: whether to use the continuous versions of the discrete labels, if available, to
        convert discrete to continuous. Default is False.
        nsamples: Method for converting from discrete to continuous. If 0, the weighted mean of bin centers is computed. If > 0,
        specifies the number of samples to take according to the bin distributions. Default is 0.
        collapse_samples: whether to collapse the samples dimension if nsamples=1 the first dimension. Default is False.
        ts: indices of the time points to return. Time points must be contiguous, limited checking done. ts should work if it is
        an ndarray, a list, a scalar, a slice, or a range. If None, all time points are returned. Default is None.
        Returns:
        kp: ndarray of size (nsamples x ) (pre_sz x ) ntimepoints x nkpts x 2 with the keypoint representation of the labels.
        """
        next_pose = self.get_next_pose(**kwargs)
        next_keypoints = self._nextpose_to_nextkeypoints(next_pose)
        return next_keypoints

    def _discretize_multi(self, example):
        """
        _discretize_multi(example)
        Discretize the features in example['continuous'] that should be discretized
        and store them in example['discrete']. The original continuous versions are 
        stored in example['todiscretize']. Dictionary input example is modified 
        in place.
        Parameters:
        example: dictionary with key 'continuous' containing the full
        continuous representation of the labels.
        """
        
        if not self.is_discretized():
            return
        assert example['continuous'].shape[-1] == self.d_multi
        discretize_idx = self._idx_multidiscrete_to_multi
        example['todiscretize'] = example['continuous'][..., discretize_idx]#.copy()
        example['discrete'] = discretize_labels(example['todiscretize'], self._discretize_params['bin_edges'],
                                                soften_to_ends=True)
        example['continuous'] = example['continuous'][..., self._idx_multicontinuous_to_multi]
        return

    def compute_features(self,Xkp,scale=None):
        
        raise NotImplementedError
    
        if (scale is None) and (self._scale is not None):
            scale = self._scale
            
        example = compute_features(Xkp[..., None], scale=scale, outtype=np.float32,
                                   dct_m=self._dct_m,
                                   tspred_global=self.tspred_global,
                                   discreteidx=self._idx_nextdiscrete_to_next)
        init_pose = example['init']
        return example,init_pose   

    def set_keypoints(self, Xkp, scale=None):
        """
        set_keypoints(Xkp, scale=None)
        Set the keypoints representation of the labels. This will compute the features and store them in the
        labels_raw dictionary.
        Parameters:
        Xkp: ndarray of size (pre_sz x ) nkpts x 2 x ntimepoints with the keypoints representation of the labels.
        Optional parameters:
        scale: ndarray of size (pre_sz x ) nkpts x 2 with the scale for each keypoint. If None, self._scale is used.
        Note that _scale is not set. Default is None.
        """

        # function for computing features
        example,init_pose = self.compute_features(Xkp, scale=scale)

        self.set_raw_example(example,dozscore=self.is_zscored(),dodiscretize=self.is_discretized())
        
        if self._init_pose is None:
            self._init_pose = init_pose
        
        return

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

    def get_next_names(self):
        """
        get_next_names()
        Returns the names of the next features.
        Returns:
        next_names: list of names of the next features.
        """
        
        next_names = [None, ] * self.d_next        
        for i in range(self.d_next):
            if i % 2 == 0:
                coord = 'x'
            else:
                coord = 'y'
            next_names[i] = f'kp_{i//2}_{coord}'
        return next_names

    def get_nextcossin_names(self):
        """
        get_nextcossin_names()
        Returns the names of the next_cossin features.
        Returns:
        next_names_cossin: list of names of the next_cossin features.
        """
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
        """
        get_multi_names()
        Returns the names of the multi features.
        Returns:
        multi_names: list of names of the multi features.
        """
        ft = self._idx_multi_to_multifeattpred
        ismulti = np.max([np.max(ts) for ts in self._tspred]) > 1
        multi_names = [None, ] * self.d_multi
        nextcs_names = self.get_nextcossin_names()
        for i in range(self.d_multi):
            if ismulti:
                multi_names[i] = nextcs_names[ft[i, 0]] + '_' + str(ft[i, 1])
        return multi_names
    
    def get_discrete_names(self):
        """
        get_discrete_names()
        Returns the names of the discrete features.
        Returns:
        discrete_names: list of names of the discrete features.
        """
        multi_names = self.get_multi_names()
        discrete_names = [multi_names[i] for i in self._idx_multidiscrete_to_multi]
        return discrete_names
    
    def get_continuous_names(self):
        """
        get_continuous_names()
        Returns the names of the continuous features
        Returns:
        continuous_names: list of names of the continuous features.
        """
        multi_names = self.get_multi_names()
        continuous_names = [multi_names[i] for i in self._idx_multicontinuous_to_multi]
        return continuous_names
    
    def get_zscore_params(self,makecopy=True):
        """
        get_zscore_params()
        Returns the zscore parameters.
        Inputs: 
        makecopy: whether to return a copy of the zscore parameters. Default is True.
        Returns:
        zscore_params: dictionary with keys 'mu_labels' and 'sigma_labels' containing the mean and standard deviation
        """
        if self.is_zscored() and makecopy:
            return copy.deepcopy(self._zscore_params)
        else:
            return self._zscore_params
    
    def get_discretize_params(self,makecopy=True,zscored=True):
        """
        get_discretize_params()
        Returns the discretize parameters.
        Inputs: 
        makecopy: whether to return a copy of the discretize parameters. Default is True.
        zscored: whether the discretize parameters should be zscored, if self.is_zscored(). Default is True.
        Returns:
        discretize_params: dictionary with keys 'bin_edges' containing the bin edges for discretization
        """
        if not self.is_discretized():
            return None
        if self.is_zscored() and (zscored == False):
            idx_discrete = self._idx_multidiscrete_to_multi
            params = {}
            for k,v in self._discretize_params.items():
                # v may have extra dimensions                    
                params[k] = unzscore(v,self._zscore_params['mu_labels'][idx_discrete,None],self._zscore_params['sig_labels'][idx_discrete,None])
            return params
        elif makecopy:
            return copy.deepcopy(self._discretize_params)
        else:
            return self._discretize_params
    
    def compute_error(true_labels,pred_labels,nsamples=10,collapsetime=True):
        """
        compute_error(true_labels,pred_labels,nsamples=10,collapsetime=True)
        Computes the error between the true and predicted labels in various ways.
        Parameters:
        true_labels: (self) PoseLabels object with the true labels. 
        pred_labels: PoseLabels object with the predicted labels. 
        nsamples: Number of samples to use when computing the error for discrete features. Default is 10.
        collapsetime: whether to average over the time dimension when computing errors. Default is True.
        Returns:
        Keys are named with this vocabulary:
        'l1': Mean of L1 error over time points
        'mse': Mean of squared error over time points
        'total': Average over features
        'samplemean': Average of nsamples samples drawn
        'samplemin': Closest of nsamples samples drawn
        'zscored': Features are z-scored -- unit is standard deviations
        
        err: Dictionary with the error metrics.
        'idxgood': ndarray of size ntimepoints with a mask for the time points that are not nan
        'n': Number of time points that are not nan
        'multi_names': list of the names of the multi features
        'l1_multi': ndarray of size d_multi with the L1 error for each multi feature averaged over all time points. 
        For discrete predictions, the weighted sum of bin centers is used. 
        'mse_multi' ndarray of size d_multi with the squared error for each multi feature averaged over all time points. 
        For discrete predictions, the weighted sum of bin centers is used. 
        'l1_multi_samplemean': ndarray of size d_multi with the L1 error for each multi feature from the mean of the 
        nsamples samples drawn, averaged over all time points. 
        'mse_multi_samplemean': ndarray of size d_multi with the squared error for each multi feature from the mean of the
        nsamples samples drawn, averaged over all time points.
        'absdiff_multi_samplemean': ndarray of size ntimepoints x d_multi computed from the mean of absdiff_multi_sample
        for each time point. 
        'absdiff_multi_samplemin': ndarray of size ntimepoints x d_multi computed from the minimum of absdiff_multi_sample
        for each time point. 
        'l1_multi_samplemin': ndarray of size d_multi with the mean L1 error for the closest sample, computed from 
        absdiff_multi_samplemin.
        'mse_multi_samplemin': ndarray of size d_multi with the mean squared error for the closest sample, computed from 
        absdiff_multi_samplemin.
        If 'discrete' is in the labels, then the following keys are included:
            'ce_discrete_mean': ndarray of size d_multi_discrete with the cross-entropy error for the discrete labels,
            averaged over all time points
            'total_ce_discrete_mean': Average of ce_discrete_mean over features
            'discrete_names': list of the names of the discrete features
        If 'continuous' is in labels, then the following are included:
            'l1_continuous_raw': ndarray of size d_multi_continuous with the L1 error for each continuous feature averaged
            over time points
            'mse_continuous_raw': ndarray of size d_multi_continuous with the squared error for each continuous feature averaged
            over time points
            'total_l1_continuous_raw': Average of l1_continuous_raw over features
            'total_mse_continuous_raw': Average of mse_continuous_raw over features
            'continuous_names': list of the names of the continuous features
        'l2_nextkp_mean': ndarray of size d_next with the mean L2 error (Euclidean distance) for the next keypoints, averaged 
        over all time points
        """
                
        pred_labels_dict = pred_labels.get_train_labels()
        true_labels_dict = true_labels.get_train_labels()
        
        starttoff = true_labels._starttoff

        err = {}
        
        idxgood = True
        if 'continuous' in pred_labels_dict:
            idxgood = idxgood & (torch.any(torch.isnan(true_labels_dict['continuous']),dim=-1)==False)
            idxgood = idxgood & (torch.any(torch.isnan(pred_labels_dict['continuous']),dim=-1)==False)
        if 'discrete' in pred_labels_dict:
            idxgood = idxgood & (torch.any(torch.isnan(true_labels_dict['discrete']),dim=(-1,-2))==False)
            idxgood = idxgood & (torch.any(torch.isnan(pred_labels_dict['discrete']),dim=(-1,-2))==False)
            
        err['idxgood'] = idxgood.cpu().numpy()
        err['n'] = np.sum(err['idxgood']).item()

        # multi representation - contains most things
        err['multi_names'] = true_labels.get_multi_names()     

        ksmeta = ['idxgood','multi_names','discrete_names','continuous_names','n']

        multi_label = true_labels.get_multi(use_todiscretize=True)
        multi_label = multi_label[...,starttoff:,:]
        multi_pred_weightedsum = pred_labels.get_multi(nsamples=0,collapse_samples=True)#,use_todiscretize=True)
        multi_pred_weightedsum = multi_pred_weightedsum[...,starttoff:,:]
        absdiff_multi = np.abs(multi_pred_weightedsum-multi_label)
        assert np.all(np.isnan(absdiff_multi[idxgood])==False), 'absdiff_multi has unexpected nans'

        err['l1_multi'] = absdiff_multi
        err['mse_multi'] = np.square(absdiff_multi)

        multi_pred_sample = pred_labels.get_multi(nsamples=nsamples,collapse_samples=False)
        multi_pred_sample = multi_pred_sample[...,starttoff:,:]

        # there are a few different ways to draw from the distributions, currently only for discrete features
        if true_labels.is_discretized():
            absdiff_multi_sample = np.abs(multi_pred_sample-multi_label[None,...])
            absdiff_multi_samplemean = np.nanmean(absdiff_multi_sample,axis=0)
            absdiff_multi_samplemin = np.nanmin(absdiff_multi_sample,axis=0)
            err['l1_multi_samplemean'] = absdiff_multi_samplemean
            err['l1_multi_samplemin'] = absdiff_multi_samplemin
            err['mse_multi_samplemean'] = np.square(absdiff_multi_samplemean)
            err['mse_multi_samplemin'] = np.square(absdiff_multi_samplemin)

            # cross-entropy error for discrete features
            n = pred_labels_dict['discrete'].shape[0]
            d = pred_labels_dict['discrete'].shape[1]
            nbins = pred_labels_dict['discrete'].shape[-1]
            
            ce_discrete = torch.nn.functional.cross_entropy(pred_labels_dict['discrete'].reshape(-1,nbins),
                                                            true_labels_dict['discrete'].reshape(-1,nbins),reduction='none').reshape(n,d)
            ce_discrete = ce_discrete.cpu().numpy()
            err['ce_discrete_mean'] = ce_discrete            
            err['discrete_names'] = true_labels.get_discrete_names()
        
        # # this doesn't make sense -- will integrate over time
        # true_kp = true_labels.get_next_keypoints()
        # pred_kp = pred_labels.get_next_keypoints()

        # l2_err_kp = np.sqrt(np.sum(np.square(true_kp-pred_kp),axis=-1))
        # err['l2_nextkp_mean'] = l2_err_kp

        # raw error
        if 'continuous' in true_labels_dict:
            err['l1_continuous_raw'] = np.abs(true_labels_dict['continuous']-pred_labels_dict['continuous'])
            err['mse_continuous_raw'] = np.square(true_labels_dict['continuous']-pred_labels_dict['continuous'])
            err['continuous_names'] = true_labels.get_continuous_names()

        # can combine z-scored errors across features
        if true_labels.is_zscored():
            
            # T x dmulti
            multi_label_z = true_labels.get_multi(use_todiscretize=True,zscored=True)
            multi_label_z = multi_label_z[...,starttoff:,:]
            # T x dmulti
            multi_pred_weightedsum_z = pred_labels.get_multi(nsamples=0,collapse_samples=True,zscored=True)
            multi_pred_weightedsum_z = multi_pred_weightedsum_z[...,starttoff:,:]
            # T x dmulti
            absdiff_multi_z = np.abs(multi_pred_weightedsum_z-multi_label_z)

            # dmulti
            err['l1_multi_zscored'] = absdiff_multi_z
            err['mse_multi_zscored'] = np.square(absdiff_multi_z)

            if true_labels.is_discretized():

                # nsamples x T x dmulti
                multi_pred_sample_z = pred_labels.get_multi(nsamples=nsamples,collapse_samples=False,zscored=True)
                multi_pred_sample_z = multi_pred_sample_z[...,starttoff:,:]

                # nsamples x T x dmulti
                absdiff_multi_sample_z = np.abs(multi_pred_sample_z-multi_label_z[None,...])
                absdiff_multi_samplemean_z = np.nanmean(absdiff_multi_sample_z,axis=0)
                absdiff_multi_samplemin_z = np.nanmin(absdiff_multi_sample_z,axis=0)
                # dmulti
                err['l1_multi_samplemean_zscored'] = absdiff_multi_samplemean_z
                err['mse_multi_samplemean_zscored'] = np.square(absdiff_multi_samplemean_z)
                        
                # dmulti
                err['l1_multi_samplemin_zscored'] = absdiff_multi_samplemin_z
                err['mse_multi_samplemin_zscored'] = np.square(absdiff_multi_samplemin_z)

        # return as np arrays
        for k,v in err.items():
            if type(v) is torch.Tensor:
                err[k] = v.cpu().numpy()

        if collapsetime:
            for k in err.keys():
                if k in ksmeta:
                    continue
                err[k] = np.nanmean(err[k],axis=-2)
        
        ks_collapse_features = ['ce_discrete_mean','l1_continuous_raw','mse_continuous_raw','l1_multi_zscored','mse_multi_zscored',
                                'l1_multi_samplemean_zscored','mse_multi_samplemean_zscored','l1_multi_samplemin_zscored',
                                'mse_multi_samplemin_zscored']
        for k in ks_collapse_features:
            if k in err:
                err['total_'+k] = np.nanmean(err[k],axis=-1)
                    
        return err
        
    def get_next_pose_closest_sample(true_labels,pred_labels,nsamples=10,uselasttimepoint=False,usesum=False):
        true_next_pose = true_labels.get_next_pose(use_todiscretize=True)
        pred_next_pose_samples = pred_labels.get_next_pose(nsamples=nsamples)

        if uselasttimepoint:
            ts = -1
        else:
            ts = range(true_next_pose.shape[-2])
        if not usesum:
            fs = range(true_next_pose.shape[-1])
            
        l1err = np.abs(true_next_pose[None,...,ts,:]-pred_next_pose_samples[:,...,ts,:])
        if usesum:
            l1err = np.sum(l1err,axis=-1)

        sampleidx = np.argmin(l1err,axis=0)

        if usesum:
            if uselasttimepoint:
                return pred_next_pose_samples[sampleidx,...]
            else:
                return pred_next_pose_samples[sampleidx,...,ts,:]
        else:
            if uselasttimepoint:
                return pred_next_pose_samples[sampleidx,...,fs]
            else:
                tidx,fidx = np.meshgrid(ts,range(true_next_pose.shape[-1]))
    
    def compute_error_iter(true_labels,pred_labels,nsamples=10):
        """
        compute_error_iter(true_labels, pred_labels, nsamples=10)
        Computes the error between the true and predicted labels in various ways. This is an iterator version
        of compute_error. 
        Parameters:
        true_labels: (self) PoseLabels object with the true labels. 
        """
        
        err = {}
        
        # if predicting velocities, this will integrate from the beginning of the time sequence
        # (pre_sz x) T x d_next_pose
        true_next_pose = true_labels.get_next_pose(use_todiscretize=True)

        pred_next_pose_expected = pred_labels.get_next_pose(nsamples=0)
        err['l1_next_pose_expected'] = np.abs(true_next_pose-pred_next_pose_expected)
        
        # (pre_sz x) T x d_next_pose
        if true_labels.is_discretized():
            pred_next_pose_closest_sample = true_labels.get_next_pose_closest_sample(pred_labels,nsamples=nsamples)
            err['l1_next_pose_samplemin'] = np.abs(true_next_pose-pred_next_pose_closest_sample)
            
        
        return err
        
        
            
    @staticmethod
    def combine_errors(err_example):
        err_total = {}
        for k,v in err_example[0].items():
            if k == 'n':
                err_total[k] = sum([x[k] for x in err_example])
            elif k == 'idxgood':
                err_total[k] = np.concatenate([x[k] for x in err_example])
            elif type(v) == np.ndarray or np.isscalar(v):
                err_total[k] = np.average(np.stack([x[k] for x in err_example]),
                                        weights=np.stack([x['n'] for x in err_example]),
                                        axis=0)
            else:
                err_total[k] = v

        return err_total

    
    
    @classmethod
    def label_dict_to_pre_sz(cls, label_dict):
        """
        label_dict_to_pre_sz(label_dict)
        Returns the pre_sz of the input dict. 
        """
        if 'continuous' in label_dict:
            return label_dict['continuous'].shape[:-2]
        elif 'labels' in label_dict:
            return label_dict['labels'].shape[:-2]
        elif 'discrete' in label_dict:
            return label_dict['discrete'].shape[:-3]
        elif 'labels_discrete' in label_dict:
            return label_dict['labels_discrete'].shape[:-3]
        else:
            raise ValueError('label_dict must have continuous, labels, discrete, or labels_discrete')

    @classmethod 
    def label_dict_to_ntimepoints(cls,label_dict):
        """
        label_dict_to_ntimepoints(label_dict)
        Returns the ntimepoints of the input dict. 
        """
        if 'continuous' in label_dict:
            return label_dict['continuous'].shape[-2]
        elif 'labels' in label_dict:
            return label_dict['labels'].shape[-2]
        elif 'discrete' in label_dict:
            return label_dict['discrete'].shape[-3]
        elif 'labels_discrete' in label_dict:
            return label_dict['labels_discrete'].shape[-3]
        else:
            raise ValueError('label_dict must have continuous, labels, discrete, or labels_discrete')        

class AgentExample:
    """
    
    AgentExample
    Class for handling input observations/inputs and pose/label outputs for an agent for multiple time points. 
    Can be used with batches. 
        
    Main properties:
    labels: PoseLabels object with the pose labels
    inputs: ObservationInputs object with the input observations 
    
    Main methods:
    __init__: Constructor for initializing from a training example or from keypoints.
    get_train_example(do_add_noise=False): Returns the training example consisting of inputs and labels. 
    copy_subindex(idx_pre=None, ts=None, needinit=True): Returns a copy of the AgentExample with a subset 
    of the examples (if batched) and/or a subset of the time points.
        
    """
    
    _labelsClass = PoseLabels
    _inputsClass = ObservationInputs
    _paramsClass = AgentParams
    
    def __init__(self,example_in: typing.Optional[dict] = None,
                 dataset: typing.Any = None,
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
        example_in: dictionary with the example. This can be the output of compute_features or AgentExample.get_train_example(). 
            Required fields:
            'input': ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the agent at each time point
            'labels' or 'continuous': ndarray of size (pre_sz x ) ntimepoints x d_continuous. Continuous pose labels for the agent
            at each time point. TODO- check that this will work without continuous inputs.
            'labels_discrete' or 'discrete': ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins. Binned pose labels 
            for the agent. TODO- check that this will work without discretized inputs. 
            
            Optional fields:
            'labels_init' or 'continuous_init': ndarray of size (pre_sz x ) tinit x d_continuous with the initial continuous pose
            labels for the agent. Used if the first frame of the sequence has been cropped as a training example for a causal network.
            'labels_discrete_init' or 'discrete_init': ndarray of size (pre_sz x ) tinit x d_discrete x nbins with the initial
            discretized pose. Used if the first frame of the sequence has been cropped as a training example for a causal network.
            'labels_todiscretize' or 'todiscretize': ndarray of size (pre_sz x ) ntimepoints x d_discrete with the continuous
            versions of the discrete labels. As discrete is non-invertible, this can be used when getting keypoints or other 
            continuous representations of the data. 
            'metadata': dictionary with metadata about which agent and video frames the observations were derived from.
            'categories': dictionary with the categories from the MABe dataset. Currently not used for anything, may be buggy. 
            'mask': ndarray of size (pre_sz x ) ntimepoints with a mask for the labels.
            'init_all' or 'init': ndarray of size (pre_sz x ) ntimepoints x d_next with the initial pose of the agent
            'input_init': ndarray of size (pre_sz x ) 1 x d_input with the observations of the agent on the 
            first frame of the sequence. This should be part of a training example in which the first frame
            has been cropped from input for a causal network. 
            'metadata': dictionary with metadata about which agent and video frames the observations were derived from

        To initialize from keypoints, pass in the following:
        Xkp: ndarray of size (pre_sz x ) ntimepoints x nfeatures x 2 with the keypoints for all flies. 
        agentnum: index of the main agent.
        scale: scale parameters for converting from keypoints to features.
        
        Optional parameters:
        dataset: AgentMLMDataset object. Used to get parameters for computing features, etc.
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
            discrete_tspred
            ntspred_relative
            discretize_params
            flatten_obs_idx
            dct_m
            idct_m        
        To-do: put these all in a dict. 
        
        """        

        # set parameters from extra arguments or dataset
        self.set_params(kwargs)
        if dataset is not None:
            self.set_params(self._paramsClass.get_params_from_dataset(dataset), override=False)
        default_params = self.get_default_params()
        self.set_params(default_params, override=False)
        if self._dct_m is not None and self._idct_m is None:
            self._idct_m = np.linalg.inv(self._dct_m)
        
        # set the metadata
        self._metadata = None
        if metadata is not None:
            self._metadata = metadata #copy.deepcopy(metadata)

        if example_in is not None:
            
            self.set_example(example_in,dozscore=dozscore,dodiscretize=dodiscretize)

        elif Xkp is not None:
            
            # create example from keypoints
            self.set_keypoints(Xkp, agentnum, scale, metadata=metadata)
        
        else:
            
            self.set_example(None)

        return
        
    def set_keypoints(self,Xkp,agentnum,scale,metadata=None):
        """
        set_keypoints(Xkp,agentnum,scale,metadata=None)
        Set the example from keypoints. This will compute the features and store them in the labels_raw dictionary.
        Parameters:
        Xkp: ndarray of size (pre_sz x ) ntimepoints x nfeatures x 2 x nflies with the keypoints for all flies.
        agentnum: index of the main agent.
        scale: scale parameters for converting from keypoints to features.
        metadata: dictionary with metadata about which agent and video frames the observations were derived from.
        """
        # compute pose and observation representations from keypoints
        example_in = self.compute_features(Xkp, agentnum, scale)
        # metadata will by copied in set_example
        if metadata is not None:
            example_in['metadata'] = metadata

        # if params set that zscoring and discretizing are to be done, then set 
        # that the example_in requires these
        dozscore = True
        dodiscretize = True
        
        self.set_example(example_in,dozscore=dozscore,dodiscretize=dodiscretize)
        return
    
    def set_example(self,example_in,dozscore=False,dodiscretize=False):
        """
        set_example(example_in,dozscore=False,dodiscretize=False)
        Set the example from a dictionary. This will create the PoseLabels and ObservationInputs objects.
        Parameters:
        example_in: dictionary with the example. This can be the output of compute_features or AgentExample.get_train_example().
        dozscore: whether to z-score the inputs. Default is False.
        dodiscretize: whether to discretize the labels. Default is False.
        """
        
        # copy the example

        # copy the dict, not the arrays
        if example_in is not None:
            example_in = {k: v for k, v in example_in.items()}
            # copy the metadata, deep copy
            if (example_in is not None) and ('metadata' in example_in):
                example_in['metadata'] = example_in['metadata']#copy.deepcopy(example_in['metadata'])

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
        self._labels = self._labelsClass(example_in, dozscore=dozscore, dodiscretize=dodiscretize,
                                         **self.get_poselabel_params())

        if is_train_example and self._do_input_labels:
            self._remove_labels_from_input(example_in)

        self._inputs = self._inputsClass(example_in, dozscore=dozscore, **self.get_observationinputs_params())
        
        if (example_in is not None) and ('metadata' in example_in):
            self._metadata = example_in['metadata']

        
        return
    
    def pre_tile(self,reps):
        """
        pre_tile(reps   )
        Tile the example along the pre_sz dimension(s) by rep. 
        Parameters:
        rep: The number of repetitions of A along each axis.
        """
        reps = np.atleast_1d(reps)
        self._labels.pre_tile(reps,tile_metadata=True)
        self._inputs.pre_tile(reps,tile_metadata=False)
        return
    
    def expand_allocate(self,newT=None,newpre_sz=None):
        self._labels.expand_allocate(newT=newT,newpre_sz=newpre_sz)
        self._inputs.expand_allocate(newT=newT,newpre_sz=newpre_sz)
    
    @property
    def d_input(self):
        """
        d_input
        Number of observation features
        """
        return self._inputs.d_input
    
    @property
    def d_labels(self):
        """
        d_labels
        Total number of pose label features
        """
        return self._labels.d_multi
    
    @property
    def d_labels_discrete(self):
        """
        d_labels_discrete
        Total number of discrete pose label features
        """
        return self.labels.d_multidiscrete
    
    @property
    def d_labels_continuous(self):
        """
        d_labels_continuous
        Total number of continuous pose label features
        """
        return self.labels.d_multicontinuous
    
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
        Dictionary with metadata about which agent and video frames the observations were derived from.
        """
        return self._metadata
    
    def compute_features(self, Xkp, agentnum, scale):
        """
        compute_features(Xkp, agentnum, scale)
        Compute sensory and pose label features from keypoints by calling features.compute_features with the 
        correct parameters. 
        Parameters:
        Xkp: ndarray of keypoints for all flies and time points, size (pre_sz x ) nkeypoints x 2 x ntimepoints x nflies
        agentnum: index of the main agent
        scale: scale parameters for converting from keypoints to features.
        Output:
        example: dictionary with the computed features:
            'input': ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the agent at each time point
            'labels': ndarray of size (pre_sz x ) ntimepoints x d_multi with the pose representation for the agent at each
            time point.
            'init': ndarray of size (pre_sz x ) ntimepoints x d_next with the initial pose of the agent. 
            'scale': scale parameters for converting from keypoints to features and vice-versa. 
        """

        example = {}
        example['labels'] = Xkp[...,1:,agentnum]#.copy()
        pre_sz = Xkp.shape[:-4]
        ntimepoints = Xkp.shape[-2]
        example['input'] = np.zeros(pre_sz+(ntimepoints,0))
        example['init'] = Xkp[...,[0,],agentnum]#.copy()

        return example

    def copy(self):
        """
        copy()
        Returns a copy of the AgentExample object. 
        """
        return self.copy_subindex()
    
    def copy_subindex(self, idx_pre=None, ts=None, needinit=True):
        """
        copy_subindex(idx_pre=None, ts=None, needinit=True)
        Returns a copy of the AgentExample object with a subset of the examples (if batched) and/or a subset of the time points.
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
        new: a copy of the AgentExample object with the specified indices and time points
        """
        
        if ts is not None:
            # convert ts to ndarray
            if type(ts) is slice:
                ts = range(*ts.indices(self.ntimepoints))
            ts = np.atleast_1d(np.array(ts))

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

            # if ts[0] > 0 and needinit, compute the initial pose by integrating
            if needinit:
                init_pose = self.labels.get_init_pose(ts=ts)
                example['init'] = init_pose
            else:
                example['init'][:] = np.nan # set to nans so that we know this is bad data
            toff = ts[0]

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

        # create the new AgentExample from the example dict
        new = self.__class__(example_in=example, **self.get_params())
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

    def set_params(self, params, override=True, synonyms = None):
        """
        set_params(params, override=True)
        Sets the parameters for the AgentExample object. 
        params: Dict of parameters to set. Each key,value pair in the dict will be set as an attribute of the AgentExample object,
        with the key prefixed by an underscore. The exception are those parameters defined in synonyms, which will get different 
        names. 
        override: Whether to override existing parameters. If False, will not overwrite existing parameters. Default is True. 
        """
        if synonyms is None:
            synonyms = {}
        for k, v in params.items():
            if k in synonyms:
                k = synonyms[k]
            k = '_' + k
            if override or (not hasattr(self, k)) or (getattr(self, k) is None):
                setattr(self, k, v)
                

    @classmethod
    def get_default_params(cls):
        """
        get_default_params()
        Returns the default parameters for the AgentExample object as a dict.
        """

        params = {
            'zscore_params': None,
            'do_input_labels': True,
            'starttoff': 1,
            'flatten_labels': False,
            'flatten_obs': False,
            'discreteidx': [],
            'tspred': [1,], # list of lists, each sublist is the frames into the future to predict for each feature
            'isdct': False, # can be list-like, with an entry for each feature
            'discrete_tspred': [1, ], # todo make obsolete
            'discretize_params': None,
            'flatten_obs_idx': None,
            'dct_m': None,
            'idct_m': None,
        }
        return params

    def get_params(self):
        """
        get_params()
        Returns the parameters for the AgentExample object as a dict.
        """
        default_params = self.__class__.get_default_params()
        params = {k: getattr(self, '_'+k) for k in default_params.keys()}
        return params

    def get_poselabel_params(self):
        """
        get_poselabel_params()
        Returns the parameters for the PoseLabels object as a dict.
        """
        params = self.get_params()
        params = self._paramsClass.example_to_poselabels_params(params)

        return params

    def get_observationinputs_params(self):
        """
        get_observationinputs_params()
        Returns the parameters for the ObservationInputs object as a dict.
        """
        params = self.get_params()
        params = self._paramsClass.example_to_observationinput_params(params)
        return params

    @property
    def ntimepoints(self):
        """
        ntimepoints
        Number of time points
        """
        # number of time points
        return self.labels.ntimepoints

    def get_metadata(self, makecopy=True):
        """
        get_metadata(makecopy=True)
        Returns the metadata for the AgentExample object. If makecopy is True, returns a deep copy of the metadata. Default: True.
        """
        if makecopy:
            return copy.deepcopy(self.metadata)
        else:
            return self.metadata

    def get_raw_example(self, format='standard', makecopy=True, needinit=True, needscale=True, needcategories=True):
        """
        get_raw_example(format='standard', makecopy=True)
        Returns the raw example for the AgentExample object as a dict. 
        Optional arguments:
        format: Format of the labels. If 'standard', then the key names used within the object are returned. These are 
        'continuous' and 'discrete. If 'original', then it will use the key names that were input when the AgentExample was created.
        Default is 'standard'.
        makecopy: Whether to make a copy of the example. Default is True.
        """
        example = self.labels.get_raw_labels(format=format, makecopy=makecopy, 
                                             needinit=needinit, needscale=needscale, needcategories=needcategories)
        example['input'] = self.inputs.get_raw_inputs(makecopy=makecopy)
        example['metadata'] = self.get_metadata(makecopy=makecopy)
        return example

    def get_input_labels(self,**kwargs):
        """
        get_input_labels()
        Returns the input labels for the AgentExample object.
        """
        if self._do_input_labels == False:
            return None
        else:
            return self.labels.get_input_labels(**kwargs)

    def get_n_input_labels(self):
        """
        get_n_input_labels()
        Returns the number of input labels for the AgentExample object.
        """
        if self._do_input_labels:
            return self.labels.get_d_labels_input()
        else:
            return 0

    def get_train_example(self, do_add_noise=False, ts=None, makecopy=True, needmetadata=True, needinput=True, needlabels=True, needinit=True):
        """
        get_train_example(do_add_noise=False)
        Returns the training example consisting of inputs and labels.
        Optional arguments:
        do_add_noise: Whether to add noise to the inputs. Default is False.
        Returns a dictionary with the following keys:
            'input': ndarray of size (pre_sz x ) ntimepoints x d_input with the observations of the agent at each time point
            'labels': ndarray of size (pre_sz x ) ntimepoints x d_continuous with the continuous pose labels for the agent
            at each time point
            'labels_discrete': ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the binned pose labels
            for the agent at each time point
            'labels_todiscretize': ndarray of size (pre_sz x ) ntimepoints x d_discrete with the continuous versions of the
            discrete labels. As discrete is non-invertible, this can be used when getting keypoints or other continuous
            representations of the data. 
            'init': ndarray of size (pre_sz x ) d_next with the initial pose of the agent
            'scale': scale parameters for converting from keypoints to features and vice-versa
            'categories': dictionary with the categories from the MABe dataset. Currently not used for anything, may be buggy.
            'metadata': dictionary with metadata about which agent and video frames the observations were derived from
            'input_init': ndarray of size (pre_sz x ) 1 x d_input with the observations of the agent on the first frame of the 
            sequence. This should be part of a training example in which the first frame has been cropped from input for a causal
            network.
            'labels_init': ndarray of size (pre_sz x ) 1 x d_continuous with the initial continuous pose labels for the agent
            'labels_discrete_init': ndarray of size (pre_sz x ) 1 x d_discrete x nbins with the initial discretized pose
            'labels_todiscretize_init': ndarray of size (pre_sz x ) 1 x d_discrete with the initial continuous versions of the
            discrete labels. As discrete is non-invertible, this can be used when getting keypoints or other continuous
            representations of the data. 
            'init_all': ndarray of size (pre_sz x ) starttoff x d_next with the initial pose of the agent
        """

        res = {}
        # to do: add noise
            
        if needinput:
            input_labels = self.get_input_labels(ts=ts)

            train_inputs = self.inputs.get_train_inputs(input_labels=input_labels,
                                                        labels=self.labels,
                                                        do_add_noise=do_add_noise,
                                                        makecopy=makecopy,
                                                        ts=ts)
            res['input'] = train_inputs['input']
            if needinit:
                res['input_init'] = train_inputs['input_init']
                
        if needlabels:
            train_labels = self.labels.get_train_labels(added_noise=train_inputs['eta'],ts=ts)
                        
            flatten = self._flatten_labels or self._flatten_obs
            assert flatten == False, 'flatten not implemented'
            
            if 'continuous' in train_labels:
                res['labels'] = train_labels['continuous']
            if 'discrete' in train_labels:
                res['labels_discrete'] = train_labels['discrete']
            if 'todiscretize' in train_labels:
                res['labels_todiscretize'] = train_labels['todiscretize']
            if needinit:
                res['init'] = train_labels['init']
                if 'continuous_init' in train_labels:
                    res['labels_init'] = train_labels['continuous_init']
                if 'discrete_init' in train_labels:
                    res['labels_discrete_init'] = train_labels['discrete_init']
                if 'todiscretize_init' in train_labels:
                    res['labels_todiscretize_init'] = train_labels['todiscretize_init']
                if 'init_all' in train_labels:
                    res['init_all'] = train_labels['init_all']
            if needmetadata:
                if 'scale' in train_labels:
                    res['scale'] = train_labels['scale']
                if 'categories' in train_labels:
                    res['categories'] = train_labels['categories']

        if needmetadata:
            metadata = self.get_train_metadata()
            if ts is not None:
                ts = np.atleast_1d(ts)
                metadata['t0'] += ts[0]
                metadata['frame0'] += ts[0]
            res['metadata'] = metadata

        return res

    def clear_cuda_cache(self):
        self.labels.clear_cuda_cache()
        self.inputs.clear_cuda_cache()
        return
        
    def set_cuda_cache(self,needinput=True,needlabels=True):
        if needinput:
            self.inputs.set_cuda_cache()
        if needlabels:
            self.labels.set_cuda_cache(cache_input_labels=self._do_input_labels)
        return
    
    def get_train_metadata(self):
        """
        get_train_metadata()
        Returns the metadata for the training example, offset by starttoff.
        """
        starttoff = self._starttoff
        metadata = self.get_metadata(makecopy=True)
        if metadata is None:
            return None
        metadata['t0'] += starttoff
        metadata['frame0'] += starttoff
        return metadata

    def set_zscore_params(self, zscore_params):
        """
        set_zscore_params(zscore_params)
        Sets the zscore parameters for the AgentExample object.
        """
        zscore_params_input, zscore_params_labels = self._paramsClass.split_zscore_params(zscore_params)
        self.inputs.set_zscore_params(zscore_params_input)
        self.labels.set_zscore_params(zscore_params_labels)
        
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
    
    def __str__(self):
        """
        __str__()
        Returns a string representation of the AgentExample object. 
        """
        s = f'{self.__class__.__name__}:\n'
        s += 'labels:\n'
        s += str(self.labels)
        s += 'inputs:\n'
        s += str(self.inputs)
        return s
    
    @staticmethod
    def expand_allocate_raw_example(raw_example,newT=None,newpre_sz=None):
        
        if 'input' in raw_example:
            oldT = raw_example['input'].shape[-2]
        elif 'labels' in raw_example:
            oldT = raw_example['labels'].shape[-2]
        elif 'continuous' in raw_example:
            oldT = raw_example['continuous'].shape[-2]
        elif 'labels_discrete' in raw_example:
            oldT = raw_example['labels_discrete'].shape[-2]
        elif 'discrete' in raw_example:
            oldT = raw_example['discrete'].shape[-2]
            
        keys2dim = {'input':2,'labels':2,'continuous':2,'labels_discrete':3,'discrete':3,'todiscretize':2,'labels_todiscretize':2}
        for k,d in keys2dim.items():
            if k in raw_example:
                if newT is not None:
                    oldT = raw_example[k].shape[-d]
                    Tpad = newT - oldT
                    raw_example[k] = pad_axis_array(raw_example[k],d,Tpad,constant_values=np.nan)
                if newpre_sz is not None:
                    raw_example[k] = pre_tile_array(raw_example[k],d,newpre_sz)

        return
