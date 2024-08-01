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

        # to do: deal with flattening

        self._input = None
        self._metadata = None

        self.set_params(kwargs)
        if dataset is not None:
            self.set_params(self.get_params_from_dataset(dataset), override=False)
        default_params = FlyExample.get_default_params()
        self.set_params(default_params, override=False)

        self._sensory_feature_idx, self._sensory_feature_szs = \
            get_sensory_feature_shapes(simplify=self._simplify_in)
        if example_in is not None:
            self.set_example(example_in, dozscore=dozscore)
        elif Xkp is not None:
            # use npad to set number of frames to crop from the end manually, 
            # particularly if there is no label sequence associated with this observation object
            self.set_inputs_from_keypoints(Xkp, fly, scale, npad=npad)

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
        return self._input
    
    @property
    def metadata(self):
        return self._metadata
    
    @property
    def d_input(self):
        if self._input is None:
            return 0
        else:
            return self._input.shape[-1]
    
    @property
    def pre_sz(self):
        if self._input is None:
            return ()
        else:
            return self._input.shape[:-2]

    @staticmethod
    def flyexample_to_observationinput_params(params):
        kwinputs = copy.deepcopy(params)
        zscore_params_input, _ = FlyExample.split_zscore_params(params['zscore_params'])
        kwinputs['zscore_params'] = zscore_params_input
        return kwinputs

    @staticmethod
    def get_default_params():
        params = FlyExample.get_default_params()
        params = ObservationInputs.flyexample_to_observationinput_params(params)
        return params

    def get_params(self):
        params = {
            'zscore_params': self._zscore_params,
            'simplify_in': self._simplify_in,
            'flatten_obs': self._flatten_obs,
        }
        return params

    def set_params(self, params, override=True):
        for k, v in params.items():
            k = '_' + k
            if override or (not hasattr(self, k)) or (getattr(self, k) is None):
                setattr(self, k, v)

    @staticmethod
    def get_params_from_dataset(dataset):
        params = FlyExample.get_params_from_dataset(dataset)
        params = ObservationInputs.flyexample_to_observationinput_params(params)
        return params
      
    def get_compute_features_params(self):
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
        self._input = example_in['input']
        if 'input_init' in example_in:
            self._input = np.concatenate((example_in['input_init'], self._input), axis=-2)

        if dozscore:
            self._input = zscore(self._input, self._zscore_params['mu_input'], self._zscore_params['sig_input'])

        if 'metadata' in example_in:
            self._metadata = example_in['metadata']
        else:
            self._metadata = None
            
    @property
    def ntimepoints(self):
        if self._input is None:
            return 0
        return self._input.shape[-2]

    def is_zscored(self):
        return self._zscore_params is not None

    def get_raw_inputs(self, makecopy=True):
        if makecopy:
            return self._input.copy()
        else:
            return self._input

    def get_inputs(self, zscored=False, **kwargs):
        input = self.get_raw_inputs(**kwargs)

        # todo: deal with flattening
        if self.is_zscored() and zscored == False:
            input = unzscore(input, self._zscore_params['mu_input'], self._zscore_params['sig_input'])

        return input

    def get_split_inputs(self, **kwargs):
        input = self.get_inputs(**kwargs)
        input = split_features(input)
        return input

    def get_inputs_type(self, type, **kwargs):
        input = self.get_split_inputs(**kwargs)
        return input[type]

    def set_zscore_params(self, zscore_params):
        self._zscore_params = zscore_params
        return

    def add_pose_noise(self, train_inputs, eta_pose, zscored=False):
        idx = self._sensory_feature_idx['pose']
        if self.is_zscored() and (zscored == False):
            eta_pose = zscore(eta_pose, self._zscore_params['mu_input'][..., idx[0]:idx[1]],
                              self._zscore_params['sig_input'][..., idx[0]:idx[1]])
        train_inputs[..., idx[0]:idx[1]] += eta_pose

    def set_inputs(self, input, zscored=False, ts=None):
        if zscored == False and self.is_zscored():
            input = zscore(input, self._zscore_params['mu_input'], self._zscore_params['sig_input'])
        if ts is None:
            self._input = input
        else:
            self._input[..., ts, :] = input
        return

    def set_inputs_from_keypoints(self, Xkp, fly, scale=None, ts=None, npad=None):
        example = compute_features(Xkp, flynum=fly, scale_perfly=scale, **self.get_compute_features_params(), npad=npad, compute_labels=False)
        input = example['input']
        self.set_inputs(input, zscored=False, ts=ts)
        return

    def add_noise(self, train_inputs, input_labels=None, labels=None):
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
        return get_sensory_feature_idx(self._simplify_in)

    def get_train_inputs(self, input_labels=None, do_add_noise=False, labels=None):

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
            if input_labels is not None:
                train_inputs_init = train_inputs[..., :self._starttoff, :]
                train_inputs = torch.cat((torch.tensor(input_labels[..., :self.ntimepoints-self._starttoff, :]),
                                          train_inputs[..., self._starttoff:, :]), dim=-1)
        else:
            ntypes = len(self._flatten_obs_idx)
            flatinput = torch.zeros(self.pre_sz + (self.ntimepoints, ntypes, self._flatten_max_dinput),
                                    dtype=train_inputs.dtype)
            for i, v in enumerate(self._flatten_obs_idx.values()):
                flatinput[..., i,
                self._flatten_input_type_to_range[i, 0]:self._flatten_input_type_to_range[i, 1]] = train_inputs[:,
                                                                                                 v[0]:v[1]]

            train_inputs = flatinput

        return {'input': train_inputs, 'eta': eta, 'input_init': train_inputs_init}


# The above code is attempting to define a Python class named `FlyExample`, but it contains a syntax
# error. In Python, the class definition should include a colon `:` after the class name. The correct
# syntax should be:
class FlyExample:
    def __init__(self, example_in=None, dataset=None, Xkp=None, flynum=None, scale=None, metadata=None,
                 dozscore=False, dodiscretize=False, **kwargs):

        self.set_params(kwargs)
        if dataset is not None:
            self.set_params(self.get_params_from_dataset(dataset), override=False)
        default_params = FlyExample.get_default_params()
        self.set_params(default_params, override=False)
        if self.dct_m is not None and self.idct_m is None:
            self.idct_m = np.linalg.inv(self.dct_m)

        if example_in is not None:
            # copy the dict, not the arrays
            example_in = {k: v for k, v in example_in.items()}
            # copy the metadata, deep copy
            if (example_in is not None) and ('metadata' in example_in):
                example_in['metadata'] = copy.deepcopy(example_in['metadata'])
        elif Xkp is not None:
            # compute pose and observation representations from keypoints
            example_in = self.compute_features(Xkp, flynum, scale, metadata)
            # copy the metadata, deep copy
            example_in['metadata'] = copy.deepcopy(metadata)
            # if params set that zscoring and discretizing are to be done, then set 
            # that the example_in requires these
            dozscore = True
            dodiscretize = True

        is_train_example = (example_in is not None) and ('input' in example_in) \
                           and (type(example_in['input']) is torch.Tensor)

        if is_train_example:
            example_in = dict_convert_torch_to_numpy(example_in)
            # if we offset the example, adjust back
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

        self.labels = PoseLabels(example_in, dozscore=dozscore, dodiscretize=dodiscretize,
                                 **self.get_poselabel_params())

        if is_train_example and self.do_input_labels:
            self.remove_labels_from_input(example_in)

        self.inputs = ObservationInputs(example_in, dozscore=dozscore, **self.get_observationinputs_params())

        self.set_zscore_params(self.zscore_params)

        self.pre_sz = self.labels.pre_sz

        if (example_in is not None) and ('metadata' in example_in):
            self.metadata = example_in['metadata']

        return
      
    def compute_features(self, Xkp, flynum=0, scale=None, metadata=None):

        example = compute_features(Xkp, flynum=flynum, scale_perfly=scale, outtype=np.float32,
                                   simplify_in=self.simplify_in,
                                   simplify_out=self.simplify_out,
                                   dct_m=self.dct_m,
                                   tspred_global=self.tspred_global,
                                   compute_pose_vel=self.is_velocity,
                                   discreteidx=self.discreteidx)

        return example

    def copy(self):
        return self.copy_subindex()
    
    def copy_subindex(self, idx_pre=None, ts=None, needinit=True):

        example = self.get_raw_example(makecopy=True)

        if idx_pre is not None:
            ks = ['continuous', 'discrete', 'todiscretize', 'input', 'init', 'scale', 'categories']
            for k in ks:
                example[k] = example[k][idx_pre]
            if example['metadata'] is not None:
              for k in example['metadata'].keys():
                  example['metadata'][k] = example['metadata'][k][idx_pre]

        if ts is not None:
            if type(ts) is slice:
                ts = range(*ts.indices(self.ntimepoints))
            ts = np.atleast_1d(np.array(ts))
            ks = ['continuous', 'discrete', 'todiscretize', 'input']
            toff = ts[0]
            if toff > 0:
                if needinit:
                    next_pose = self.labels.get_next_pose(ts=np.arange(toff+1),use_todiscretize=self.labels.is_todiscretize())
                    init_pose = next_pose[-2:]
                    example['init'] = init_pose.T                    
                else:
                    example['init'][:] = np.nan # set to nans so that we know this is bad data
                    
            if example['categories'] is not None:
                cattextra = example['categories'].shape[-1] - example['continuous'].shape[-2]
                if hasattr(ts, '__len__'):
                    example['categories'] = example['categories'][..., ts[0]:ts[-1] + cattextra, :]
                else:
                    example['categories'] = example['categories'][..., ts:ts + cattextra, :]
            for k in ks:
                if k == 'discrete':
                    example[k] = example[k][..., ts, :, :]
                else:
                    example[k] = example[k][..., ts, :]
            if (example['metadata'] is not None) and ('t0' in example['metadata']):
                example['metadata']['t0'] += toff

        new = FlyExample(example_in=example, **self.get_params())
        return new

    def remove_labels_from_input(self, example_in):
        if not self.do_input_labels:
            return

        d_labels = self.labels.get_d_labels_input()
        example_in['input'] = example_in['input'][..., d_labels:]

    def set_params(self, params, override=True):
        synonyms = {'compute_pose_vel': 'is_velocity'}
        for k, v in params.items():
            if k in synonyms:
                k = synonyms[k]
            if override or (not hasattr(self, k)) or (getattr(self, k) is None):
                setattr(self, k, v)

    @staticmethod
    def get_default_params():

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
        default_params = FlyExample.get_default_params()
        params = {k: getattr(self, k) for k in default_params.keys()}
        return params

    @staticmethod
    def get_params_from_dataset(dataset):
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
        params = self.get_params()
        params = PoseLabels.flyexample_to_poselabels_params(params)

        return params

    def get_observationinputs_params(self):
        params = self.get_params()
        params = ObservationInputs.flyexample_to_observationinput_params(params)
        return params

    @property
    def ntimepoints(self):
        # number of time points
        return self.labels.ntimepoints

    @property
    def szrest(self):
        return self.labels.szrest

    def get_labels(self):
        return self.labels

    def get_inputs(self):
        return self.inputs

    def get_metadata(self, makecopy=True):
        if makecopy:
            return copy.deepcopy(self.metadata)
        else:
            return self.metadata

    def get_raw_example(self, format='standard', makecopy=True):
        example = self.labels.get_raw_labels(format=format, makecopy=makecopy)
        example['input'] = self.inputs.get_raw_inputs(makecopy=makecopy)
        example['metadata'] = self.get_metadata(makecopy=makecopy)
        return example

    def get_input_labels(self):
        if self.do_input_labels == False:
            return None
        else:
            return self.labels.get_input_labels()

    def get_n_input_labels(self):
        return len(self.labels.idx_nextcossin_to_multi)

    def get_train_example(self, do_add_noise=False):

        # to do: add noise
        metadata = self.get_train_metadata()
        input_labels = self.get_input_labels()

        train_inputs = self.inputs.get_train_inputs(input_labels=input_labels,
                                                    labels=self.labels,
                                                    do_add_noise=do_add_noise)
        train_labels = self.labels.get_train_labels(added_noise=train_inputs['eta'])

        flatten = self.flatten_labels or self.flatten_obs
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
        starttoff = self.starttoff
        metadata = self.get_metadata()
        if metadata is None:
            return None
        metadata = copy.deepcopy(metadata)
        metadata['t0'] += starttoff
        metadata['frame0'] += starttoff
        return metadata

    @staticmethod
    def split_zscore_params(zscore_params):
        if zscore_params is not None:
            zscore_params_input = {'mu_input': zscore_params['mu_input'], 'sig_input': zscore_params['sig_input']}
            zscore_params_labels = {'mu_labels': zscore_params['mu_labels'], 'sig_labels': zscore_params['sig_labels']}
        else:
            zscore_params_input = None
            zscore_params_labels = None
        return zscore_params_input, zscore_params_labels

    def set_zscore_params(self, zscore_params):
        zscore_params_input, zscore_params_labels = FlyExample.split_zscore_params(zscore_params)
        self.inputs.set_zscore_params(zscore_params_input)
        self.labels.set_zscore_params(zscore_params_labels)


class PoseLabels:
    def __init__(self, example_in=None,
                 Xkp=None, scale=None, metadata=None,
                 dozscore=False, dodiscretize=False,
                 dataset=None, **kwargs):

        # different representations of labels:
        # labels_raw -- representation used for training/prediction
        # store in this format so that it is efficient for training
        # this contains the follow:
        # continuous: (sz) x d_output_continuous
        # discrete: (sz) x d_output_discrete x nbins
        # todiscretize: (sz) x d_output_discrete
        # stacked: (sz) x ntypes x d_output_flatten
        # these will be z-scored if zscore_params is not None
        #
        # full_labels_discreteidx: indices of

        self.set_params(kwargs)
        if dataset is not None:
            self.set_params(self.get_params_from_dataset(dataset), override=False)
        default_params = PoseLabels.get_default_params()
        self.set_params(default_params, override=False)

        # default_params = self.get_default_params()
        # self.set_params(default_params,override=False)

        # copy over labels_in
        self.label_keys = {}
        self.labels_raw = {}
        self.pre_sz = None
        self.metadata = metadata
        self.categories = None
        self.init_pose = None

        if (self.discretize_params is not None) and ('bin_edges' in self.discretize_params) \
                and (self.discretize_params['bin_edges'] is not None):
            self.discretize_nbins = self.discretize_params['bin_edges'].shape[-1] - 1
        else:
            self.discretize_nbins = 0

        # to do: include flattening

        if example_in is not None:
            self.set_raw_example(example_in, dozscore=dozscore, dodiscretize=dodiscretize)
        elif Xkp is not None:
            self.set_keypoints(Xkp, scale)

        if 'continuous' in self.labels_raw:
            assert self.d_multicontinuous == self.labels_raw['continuous'].shape[-1]
        if self.is_discretized() and 'discrete' in self.labels_raw:
            assert self.d_multidiscrete == self.labels_raw['discrete'].shape[-2]

        return

    def __str__(self):
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

    def set_prediction(self, predin, ts=None, zscored=True, use_todiscretize=False, nsamples=1):

        # convert to ndarray if torch tensors
        pred = {k: v.numpy() if type(v) is torch.Tensor else v for k, v in predin.items()}
        
        if ts is None:
            ts = slice(self.starttoff,None)
        
        # if discretized, this will sample from discrete        
        multi = self.raw_labels_to_multi(pred, use_todiscretize=use_todiscretize, nsamples=nsamples, zscored=zscored, collapse_samples=True, ts=ts)
        
        # if discretized, send through the discretized values, otherwise sample will be discretized
        # since binning is soft, this will be slightly different than the original discrete
        if self.is_discretized():
            multi_discrete = pred['discrete']
        else:
            multi_discrete = None
        self.set_multi(multi,multi_discrete=multi_discrete,zscored=zscored,ts=ts)
        
        # if self.is_continuous():
        #     if 'continuous' in pred:
        #         self.labels_raw['continuous'][..., ts, :] = pred['continuous']
        #     elif 'labels' in pred:
        #         self.labels_raw['continuous'][..., ts, :] = pred['labels']
        #     else:
        #         raise ValueError('pred must contain continuous or labels')
        # if self.is_discretized():
        #     if 'discrete' in pred:
        #         self.labels_raw['discrete'][..., ts, :, :] = pred['discrete']
        #     elif 'labels_discrete' in pred:
        #         self.labels_raw['discrete'][..., ts, :, :] = pred['labels_discrete']
        #     else:
        #         raise ValueError('pred must contain discrete or labels_discrete')
            
        # if 'todiscretize' in self.labels_raw:
        #     if 'labels_todiscretize' in pred:
        #         self.labels_raw['todiscretize'][...,ts,:] = pred['labels_todiscretize']
        #     elif 'todiscretize' in pred:
        #         self.labels_raw['todiscretize'][...,ts,:] = pred['todiscretize']
        #     else:
        #         self.labels_raw['todiscretize'][...,ts,:] = np.nan

        return

    def set_raw_example(self, example_in, dozscore=False, dodiscretize=False):

        if example_in is None:
            self.labels_raw = {}
            self.label_keys = {}
            self.metadata = None
            self.categories = None
            self.pre_sz = None
            self.scale = None
            return

        if 'labels' in example_in:
            labels_in = example_in['labels']
            self.label_keys['continuous'] = 'labels'
        elif 'continuous' in example_in:
            labels_in = example_in['continuous']
            self.label_keys['continuous'] = 'continuous'
        else:
            raise ValueError('labels_in must contain labels or continuous')
        tinit = 0
        if 'labels_init' in example_in and example_in['labels_init'] is not None:
            labels_in = np.concatenate((example_in['labels_init'], labels_in), axis=-2)
            tinit = example_in['labels_init'].shape[-2]
        elif 'continuous_init' in example_in and example_in['continuous_init'] is not None:
            labels_in = np.concatenate((example_in['continuous_init'], labels_in), axis=-2)
            tinit = example_in['continuous_init'].shape[-2]
        self.labels_raw['continuous'] = np.atleast_2d(labels_in)

        if 'labels_discrete' in example_in:
            labels_discrete = example_in['labels_discrete']
            self.label_keys['discrete'] = 'labels_discrete'
        elif 'discrete' in example_in:
            labels_discrete = example_in['discrete']
            self.label_keys['discrete'] = 'discrete'
        else:
            labels_discrete = None
        if labels_discrete is not None:
            labels_discrete = np.atleast_3d(labels_discrete)
            if 'labels_discrete_init' in example_in and example_in['labels_discrete_init'] is not None:
                labels_discrete = np.concatenate((example_in['labels_discrete_init'], labels_discrete), axis=-3)
            elif 'discrete_init' in example_in and example_in['discrete_init'] is not None:
                labels_discrete = np.concatenate((example_in['discrete_init'], labels_discrete), axis=-3)
            self.labels_raw['discrete'] = labels_discrete

        if 'labels_todiscretize' in example_in:
            labels_todiscretize = example_in['labels_todiscretize']
            self.label_keys['todiscretize'] = 'labels_todiscretize'
        elif 'todiscretize' in example_in:
            labels_todiscretize = example_in['todiscretize']
            self.label_keys['todiscretize'] = 'todiscretize'
        else:
            labels_todiscretize = None
        if labels_todiscretize is not None:
            labels_todiscretize = np.atleast_2d(labels_todiscretize)
            if 'labels_todiscretize_init' in example_in and example_in['labels_todiscretize_init'] is not None:
                labels_todiscretize = np.concatenate((example_in['labels_todiscretize_init'], labels_todiscretize),
                                                     axis=-2)
            elif 'todiscretize_init' in example_in and example_in['todiscretize_init'] is not None:
                labels_todiscretize = np.concatenate((example_in['todiscretize_init'], labels_todiscretize), axis=-2)
            self.labels_raw['todiscretize'] = labels_todiscretize

        if self.is_continuous():
            self.pre_sz = self.labels_raw['continuous'].shape[:-2]
        else:
            self.pre_sz = self.labels_raw['discrete'].shape[:-3]

        if 'mask' in example_in:
            self.labels_raw['mask'] = np.atleast_1d(example_in['mask'])
            if tinit > 0:
                self.labels_raw['mask'] = np.concatenate(
                    (np.zeros(self.pre_sz + (tinit,), dtype=bool), self.labels_raw['mask']), axis=-1)

        # if 'labels_stacked' in example_in:
        #   self.labels_raw['stacked'] = example_in['labels_stacked']
        #   self.label_keys['stacked'] = 'labels_stacked'
        # elif 'stacked' in example_in:
        #   self.labels_raw['stacked'] = example_in['stacked']
        #   self.label_keys['stacked'] = 'stacked'
        self.scale = example_in['scale']
        if 'metadata' in example_in:
            self.metadata = example_in['metadata']
        else:
            self.metadata = None

        if 'categories' in example_in:
            self.categories = example_in['categories']

        if dozscore and self.is_zscored():
            self.labels_raw['continuous'] = self.zscore_multi(self.labels_raw['continuous'])
        if dodiscretize and self.is_discretized():
            self.discretize_multi(self.labels_raw)
        
        if 'init_all' in example_in:
          # output of get_train_example/get_train_labels, need to use init_all
          self.init_pose = example_in['init_all']
        elif 'init' in example_in:
            self.init_pose = example_in['init']

    def append_raw(self, pred):
        if 'labels' in pred:
            toappend = np.atleast_2d(pred['labels'])
        elif 'continuous' in pred:
            toappend = np.atleast_2d(pred['continuous'])
        else:
            raise ValueError('pred must contain labels or continuous')
        tappend = toappend.shape[-2]
        self.labels_raw['continuous'].append(toappend, axis=-2)
        if 'discrete' in self.labels_raw:
            if 'labels_discrete' in pred:
                toappend = np.atleast_2d(pred['labels_discrete'])
            elif 'discrete' in pred:
                toappend = np.atleast_2d(pred['discrete'])
            else:
                raise ValueError('pred must contain labels_discrete or discrete')
            assert toappend.shape[-2] == tappend
            self.labels_raw['discrete'].append(toappend, axis=-2)

        if 'todiscretize' in self.labels_raw:
            if 'labels_todiscretize' in pred:
                toappend = np.atleast_2d(pred['labels_todiscretize'])
            elif 'todiscretize' in pred:
                toappend = np.atleast_2d(pred['todiscretize'])
            else:
                toappend = np.zeros(self.pre_sz + (tappend, self.d_multidiscrete), self.dtype)
                toappend[:] = np.nan
            self.labels_raw['todiscretize'].append(toappend, axis=-2)

        return

    def copy(self):
        return self.copy_subindex()

    def copy_subindex(self, idx_pre=None, ts=None):

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
            # hasn't been tested yet...
            ks = ['continuous', 'discrete', 'todiscretize', 'mask']
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
            for k in ks:
                if k not in labels:
                    continue
                if k == 'discrete':
                    labels[k] = labels[k][..., ts, :, :]
                else:
                    labels[k] = labels[k][..., ts, :]
            if (labels['metadata'] is not None) and ('t0' in labels['metadata']):
                labels['metadata']['t0'] += toff

        new = PoseLabels(example_in=labels, init_next=init_next, **self.get_params())
        return new

    def erase_labels(self):
        if self.is_continuous() and 'continuous' in self.labels_raw:
            self.labels_raw['continuous'][..., self.starttoff:, :] = np.nan
        if self.is_discretized():
            if 'discrete' in self.labels_raw:
                self.labels_raw['discrete'][..., self.starttoff:, :, :] = np.nan
            if 'todiscretize' in self.labels_raw:
                self.labels_raw['todiscretize'][..., self.starttoff:, :] = np.nan
        return

    @staticmethod
    def flyexample_to_poselabels_params(params):
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
        params = FlyExample.get_default_params()
        params = PoseLabels.flyexample_to_poselabels_params(params)
        return params

    def get_params_from_dataset(self, dataset):
        params = FlyExample.get_params_from_dataset(dataset)
        params = PoseLabels.flyexample_to_poselabels_params(params)
        return params

    def get_params(self):
        kwlabels = {
            'zscore_params': self.zscore_params,
            'discreteidx': self.idx_nextdiscrete_to_next,
            'tspred_global': self.tspred_global,
            'discrete_tspred': self.discrete_tspred,
            'ntspred_relative': self.ntspred_relative,
            'discretize_params': self.discretize_params,
            'is_velocity': self.is_velocity,
            'simplify_out': self.simplify_out,
            'starttoff': self.starttoff,
            'flatten_labels': self.flatten_labels,
            'dct_m': self.dct_m,
            'idct_m': self.idct_m,
        }
        return kwlabels

    def set_params(self, params, override=True):
        translatedict = {'discreteidx': 'idx_nextdiscrete_to_next'}
        for k, v in params.items():
            if k in translatedict:
                k = translatedict[k]
            if override or (not hasattr(self, k)) or (getattr(self, k) is None):
                setattr(self, k, v)

    @property
    def ntimepoints(self):
        # number of time points
        if len(self.labels_raw) == 0:
            return 0
        if self.is_continuous():
            return self.labels_raw['continuous'].shape[-2]
        else:
            return self.labels_raw['discretized'].shape[-3]

    @property
    def ntimepoints_train(self):
        return self.ntimepoints - self.starttoff

    @property
    def device(self):
        return self.labels_raw['continuous'].device

    @property
    def dtype(self):
        if self.is_continuous():
            return self.labels_raw['continuous'].dtype
        else:
            return self.labels_raw['discretized'].dtype

    @property
    def shape(self):
        return self.pre_sz + (self.ntimepoints, self.get_d_labels_full(),)

    @property
    def d_labels_full(self):
        return self.d_multi

    def is_dct(self):
        return self.ntspred_relative > 1

    def set_init_pose(self, init_pose):
        self.init_pose = init_pose

    def get_init_pose(self, starttoff=None, makecopy=False):
        if starttoff is None:
            init_pose = self.init_pose
        else:
            init_pose = self.init_pose[:, starttoff]
        if makecopy:
            init_pose = init_pose.copy()
        return init_pose

    def get_init_global(self, starttoff=None, makecopy=True):
        init_global0 = self.init_pose[..., self.idx_nextglobal_to_next, :]
        if starttoff is None:
            init_global = init_global0
            if makecopy:
                init_global = init_global.copy()
            else:
                init_global = init_global0
            return init_global
        init_global = init_global0[..., starttoff]
        if makecopy:
            init_global = init_global.copy()
            init_global0 = init_global0.copy()
        return init_global, init_global0

    def get_scale(self, makecopy=True):
        if makecopy:
            return self.scale.copy()
        else:
            return self.scale

    def get_categories(self, makecopy=True):
        if self.categories is None:
            return None
        if makecopy:
            return self.categories.copy()
        else:
            return self.categories

    def get_metadata(self, makecopy=True):
        if makecopy:
            return copy.deepcopy(self.metadata)
        else:
            return self.metadata

    def get_d_labels_input(self):
        return self.d_next_cossin

    # which indices of pose (next frame, global + relative) are global
    @property
    def idx_nextglobal_to_next(self):
        return np.array(featglobal)

    @property
    def d_next_global(self):
        return len(self.idx_nextglobal_to_next)

    # which indices of pose (next frame, global + relative) are global
    @property
    def idx_nextglobal_to_next(self):
        return np.array(featglobal)

    @property
    def d_next_global(self):
        return len(self.idx_nextglobal_to_next)

    # which indices of pose (next frame, global + relative) are relative
    @property
    def idx_nextrelative_to_next(self):
        if self.simplify_out is None:
            return np.nonzero(featrelative)[0]
        else:
            return np.array([])

    @property
    def d_next_relative(self):
        return len(self.idx_nextrelative_to_next)

    @property
    def d_next(self):
        return self.d_next_global + self.d_next_relative

    # which indices are angles
    @property
    def is_angle_next(self):
        return featangle

    # which indices of pose (next frame, global + relative) are continuous
    @property
    def idx_nextcontinuous_to_next(self):
        iscontinuous = np.ones(self.d_next, dtype=bool)
        iscontinuous[self.idx_nextdiscrete_to_next] = False
        return np.nonzero(iscontinuous)[0]

    # we will use a cosine/sine representation for relative pose
    # next_cossin is equivalent to next if velocity is used
    @property
    def idx_nextcossinglobal_to_nextcossin(self):
        return np.arange(self.d_next_global)

    @property
    def d_next_cossin_global(self):
        return len(self.idx_nextcossinglobal_to_nextcossin)

    @property
    def idx_nextglobal_to_nextcossinglobal(self):
        return np.arange(self.d_next_global)

    def get_idx_nextrelative_to_nextcossinrelative(self):
        if self.is_velocity:
            return np.arange(self.d_next_relative), self.d_next_relative
        else:
            return relfeatidx_to_cossinidx(self.idx_nextdiscrete_to_next)

    @property
    def idx_nextrelative_to_nextcossinrelative(self):
        idx, _ = self.get_idx_nextrelative_to_nextcossinrelative()
        return idx

    @property
    def d_next_cossin_relative(self):
        _, d = self.get_idx_nextrelative_to_nextcossinrelative()
        return d

    @property
    def d_next_cossin(self):
        return self.d_next_cossin_relative + self.d_next_cossin_global

    @property
    def idx_nextcossinrelative_to_nextcossin(self):
        return np.setdiff1d(np.arange(self.d_next_cossin), self.idx_nextcossinglobal_to_nextcossin)

    @property
    def idx_next_to_nextcossin(self):
        idx = list(range(self.d_next))
        idx_nextglobal_to_next = self.idx_nextglobal_to_next
        idx_nextglobal_to_nextcossinglobal = self.idx_nextglobal_to_nextcossinglobal
        idx_nextcossinglobal_to_nextcossin = self.idx_nextcossinglobal_to_nextcossin
        idx_nextrelative_to_next = self.idx_nextrelative_to_next
        idx_nextrelative_to_nextcossinrelative = self.idx_nextrelative_to_nextcossinrelative
        idx_nextcossinrelative_to_nextcossin = self.idx_nextcossinrelative_to_nextcossin

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

    # which indices of nextcossin are discrete/continuous
    @property
    def idx_nextcossindiscrete_to_nextcossin(self):
        idx = []
        idx_next_to_nextcossin = self.idx_next_to_nextcossin
        for inext in self.idx_nextdiscrete_to_next:
            inextcossin = idx_next_to_nextcossin[inext]
            idx.append(inextcossin)
        return idx

    @property
    def idx_nextcossincontinuous_to_nextcossin(self):
        idx = []
        idx_next_to_nextcossin = self.idx_next_to_nextcossin
        for inext in self.idx_nextcontinuous_to_next:
            inextcossin = idx_next_to_nextcossin[inext]
            if type(inextcossin) is np.ndarray:
                idx.extend(inextcossin.tolist())
            else:
                idx.append(inextcossin)
        return idx

    @property
    def d_multi_relative(self):
        return self.d_next_cossin_relative * self.ntspred_relative

    @property
    def d_multi_global(self):
        return self.d_next_cossin_global * len(self.tspred_global)

    @property
    def d_multi(self):
        return self.d_multi_global + self.d_multi_relative

    # which multi correspond to nextcossin
    @property
    def idx_nextcossin_to_multi(self):
        assert (np.min(self.tspred_global) == 1)
        return self.feattpred_to_multi([(f, 1) for f in range(self.d_next_cossin)])

    # look up table from multi index to (feat,tpred)
    # d_multi x 2 array
    @property
    def idx_multi_to_multifeattpred(self):
        return self.multi_to_feattpred(np.arange(self.d_multi))

    # look up table from (feat,tpred) to multi index
    # dict
    @property
    def idx_multifeattpred_to_multi(self):
        idx_multifeattpred_to_multi = {}
        for idx, ft in enumerate(self.idx_multi_to_multifeattpred):
            idx_multifeattpred_to_multi[tuple(ft.tolist())] = idx
        return idx_multifeattpred_to_multi

    # which indices of multi correspond to multi_relative and multi_global
    def get_multi_isrelative(self):
        idx_nextcossinrelative_to_nextcossin = self.idx_nextcossinrelative_to_nextcossin
        idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
        isrelative = np.array([ft[0] in idx_nextcossinrelative_to_nextcossin for ft in idx_multi_to_multifeattpred])
        return isrelative

    @property
    def idx_multirelative_to_multi(self):
        isrelative = self.get_multi_isrelative()
        return np.nonzero(isrelative)[0]

    @property
    def idx_multiglobal_to_multi(self):
        isrelative = self.get_multi_isrelative()
        return np.nonzero(isrelative == False)[0]

    # which indices of multi correspond to multi_discrete, multi_continuous
    def get_multi_isdiscrete(self):
        idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
        isdiscrete = (np.isin(idx_multi_to_multifeattpred[:, 0], self.idx_nextcossindiscrete_to_nextcossin) & \
                      (idx_multi_to_multifeattpred[:, 1] == 1)) | \
                     (np.isin(idx_multi_to_multifeattpred[:, 0], self.idx_nextcossinglobal_to_nextcossin) & \
                      np.isin(idx_multi_to_multifeattpred[:, 1], self.discrete_tspred))
        return isdiscrete

    @property
    def idx_multidiscrete_to_multi(self):
        isdiscrete = self.get_multi_isdiscrete()
        return np.nonzero(isdiscrete)[0]

    @property
    def idx_multicontinuous_to_multi(self):
        isdiscrete = self.get_multi_isdiscrete()
        return np.nonzero(isdiscrete == False)[0]

    @property
    def idx_multi_to_multidiscrete(self):
        isdiscrete = self.get_multi_isdiscrete()
        idx = np.zeros(self.d_multi, dtype=int)
        idx[:] = -1
        idx[isdiscrete] = np.arange(np.count_nonzero(isdiscrete))
        return idx

    @property
    def idx_multi_to_multicontinuous(self):
        iscontinuous = self.get_multi_isdiscrete() == False
        idx = np.zeros(self.d_multi, dtype=int)
        idx[:] = -1
        idx[iscontinuous] = np.arange(np.count_nonzero(iscontinuous))
        return idx

    @property
    def d_multidiscrete(self):
        return len(self.idx_multidiscrete_to_multi)

    @property
    def d_multicontinuous(self):
        return len(self.idx_multicontinuous_to_multi)

    def feattpred_to_multi(self, ftidx):
        idx = ravel_label_index(ftidx, ntspred_relative=self.ntspred_relative,
                                tspred_global=self.tspred_global, nrelrep=self.d_next_cossin_relative)
        return idx

    def multi_to_feattpred(self, idx):
        ftidx = unravel_label_index(idx, ntspred_relative=self.ntspred_relative, tspred_global=self.tspred_global,
                                    nrelrep=self.d_next_cossin_relative)
        return ftidx

    def is_zscored(self):
        return self.zscore_params is not None

    def is_discretized(self):
        return self.discretize_params is not None

    def is_continuous(self):
        return 'continuous' in self.labels_raw

    def is_todiscretize(self):
        return self.is_discretized() and ('todiscretize' in self.labels_raw)

    def is_masked(self):
        return 'mask' in self.labels_raw

    def get_raw_labels(self, format='standard', ts=None, makecopy=True):
        labels_out = {}
        for kin in self.labels_raw.keys():
            if format == 'standard':
                kout = kin
            else:
                kout = self.label_keys[kin]
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
        raw_labels = self.get_raw_labels(makecopy=False, **kwargs)
        labels_out = {}
        for k, v in raw_labels.items():
            if type(v) is np.ndarray:
                labels_out[k] = torch.tensor(v)
        return labels_out

    def get_ntokens(self):
        return self.d_multidiscrete + int(self.is_continuous())

    def get_flatten_max_doutput(self):
        return np.max(self.d_multicontinuous, self.discretize_nbins)

    def get_train_labels(self, added_noise=None, namingscheme='standard'):

        # makes a copy
        raw_labels = self.get_raw_labels_tensor_copy()

        # to do: add noise
        assert added_noise is None, 'not implemented'

        rename_dict = {'discrete': 'discrete', 
                       'continuous': 'continuous', 
                       'todiscretize': 'todiscretize',
                       'continuous_init': 'continuous_init',
                       'discrete_init': 'discrete_init', 
                       'todiscretize_init': 'todiscretize_init'}
                        
        if namingscheme == 'train':
          rename_dict['discrete'] = 'labels_discrete'
          rename_dict['continuous'] = 'labels'
          rename_dict['todiscretize'] = 'labels_todiscretize'
          rename_dict['continuous_init'] = 'labels_init'
          rename_dict['discrete_init'] = 'labels_discrete_init'
          rename_dict['todiscretize_init'] = 'labels_todiscretize_init'

        train_labels = {}

        if self.is_discretized():
            train_labels[rename_dict['discrete']] = raw_labels['discrete'][..., self.starttoff:, :, :]
            train_labels[rename_dict['todiscretize']] = raw_labels['todiscretize'][..., self.starttoff:, :]
            train_labels[rename_dict['discrete_init']] = raw_labels['discrete'][..., :self.starttoff, :, :]
            train_labels[rename_dict['todiscretize_init']] = raw_labels['todiscretize'][..., :self.starttoff, :]
        else:
            train_labels[rename_dict['discrete']] = None
            train_labels[rename_dict['todiscretize']] = None
            train_labels[rename_dict['discrete_init']] = None
            train_labels[rename_dict['todiscretize_init']] = None

        train_labels['init_all'] = raw_labels['init']
        train_labels['init'] = raw_labels['init'][..., self.starttoff]
        train_labels['scale'] = raw_labels['scale']
        if 'categories' in raw_labels:
            train_labels['categories'] = raw_labels['categories']
        else:
            train_labels['categories'] = None

        if not self.flatten_labels:
            train_labels[rename_dict['continuous']] = raw_labels['continuous'][..., self.starttoff:, :]
            train_labels[rename_dict['continuous_init']] = raw_labels['continuous'][..., :self.starttoff, :]
            if 'mask' in raw_labels:
                train_labels['mask'] = raw_labels['mask'][..., self.starttoff:]
        else:
            contextl = self.ntimepoints
            dtype = raw_labels['continuous'].dtype
            ntokens = self.get_ntokens()
            flatten_max_doutput = self.get_flatten_max_doutput()
            flatlabels = torch.zeros(self.pre_sz + (contextl, ntokens, flatten_max_doutput), dtype=dtype)
            for i in range(self.d_output_discrete):
                # inputnum = self.flatten_nobs_types+i
                flatlabels[..., i, :self.discretize_nbins] = raw_labels['discrete'][..., i, :]
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
        if not self.is_masked():
            return None
        labels_raw = self.get_raw_labels(format='standard', ts=ts, makecopy=makecopy)
        return labels_raw['mask']

    def unzscore_multi(self, multi):
        if not self.is_zscored():
            return multi
        multi = unzscore(multi, self.zscore_params['mu_labels'], self.zscore_params['sig_labels'])
        return multi

    def zscore_multi(self, multi_unz):
        if not self.is_zscored():
            return multi_unz
        multi = zscore(multi_unz, self.zscore_params['mu_labels'], self.zscore_params['sig_labels'])
        return multi

    def labels_discrete_to_continuous(self, labels_discrete, epsilon=1e-3):
        assert self.is_discretized()
        sz = labels_discrete.shape
        nbins = sz[-1]
        nfeat = sz[-2]
        szrest = sz[:-2]
        n = int(np.prod(np.array(szrest)))
        labels_discrete = labels_discrete.reshape((n, nfeat, nbins))

        # nfeat x nbins
        bin_centers = self.discretize_params['bin_medians']
        s = np.sum(labels_discrete, axis=-1)
        assert np.max(np.abs(1 - s)) < epsilon, 'discrete labels do not sum to 1'
        continuous = np.sum(bin_centers[None, ...] * labels_discrete, axis=-1) / s
        continuous = np.reshape(continuous, szrest + (nfeat,))
        return continuous

    def sample_discrete_labels(self, labels_discrete, nsamples=1):
        assert self.is_discretized()

        sz = labels_discrete.shape
        nbins = sz[-1]
        nfeat = sz[-2]
        szrest = sz[:-2]
        n = int(np.prod(np.array(szrest)))
        labels_discrete = labels_discrete.reshape((n, nfeat, nbins))
        bin_samples = self.discretize_params['bin_samples']
        nsamples_per_bin = bin_samples.shape[0]
        continuous = np.zeros((nsamples,) + szrest + (nfeat,), dtype=labels_discrete.dtype)
        for f in range(nfeat):
            # to do make weighted_sample work with numpy directly
            binnum = weighted_sample(torch.tensor(labels_discrete[:, f, :]), nsamples=nsamples).numpy()
            sample = np.random.randint(low=0, high=nsamples_per_bin, size=(nsamples, n))
            curr = bin_samples[sample, f, binnum].reshape((nsamples,) + szrest)
            continuous[..., f] = curr

        return continuous

    def raw_labels_to_multi(self, labels_raw, use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False, ts=None):
        
        # to do: add flattening support here
        
        # allocate multi
        T = len_wrapper(ts, self.ntimepoints)
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
            multi[..., self.idx_multidiscrete_to_multi] = labels_discrete

        # get continuous
        multi[..., self.idx_multicontinuous_to_multi] = labels_raw['continuous']

        # unzscore
        if zscored == False and self.is_zscored():
            multi = self.unzscore_multi(multi)

        return multi

    def get_multi(self, use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False, ts=None):

        labels_raw = self.get_raw_labels(format='standard', ts=ts, makecopy=False)
        multi = self.raw_labels_to_multi(labels_raw, use_todiscretize=use_todiscretize, nsamples=nsamples, 
                                         zscored=zscored, collapse_samples=collapse_samples, ts=ts)

        return multi

    def get_multi_discrete(self, makecopy=True, ts=None):
        if not self.is_discretized():
            nts = len_wrapper(ts, self.ntimepoints)
            return np.zeros((self.pre_sz + (nts, 0, 0)), dtype=self.dtype)
        labels_raw = self.get_raw_labels(format='standard', ts=ts, makecopy=makecopy)
        return labels_raw['discrete']

    def set_multi(self, multi, multi_discrete=None, zscored=False, ts=None):

        multi = np.atleast_2d(multi)

        # zscore
        if self.is_zscored() and (zscored == False):
            multi = self.zscore_multi(multi)

        labels_raw = self.get_raw_labels(format='standard', makecopy=False)

        if ts is None:
            ts = slice(None)
        elif np.isscalar(ts):
            ts = [ts,]

        # set continuous
        labels_raw['continuous'][...,ts,:] = multi[...,self.idx_multicontinuous_to_multi]

        # set discrete
        if self.is_discretized():
            labels_raw['todiscretize'][...,ts,:] = multi[..., self.idx_multidiscrete_to_multi]
            if multi_discrete is None:
                labels_raw['discrete'][...,ts,:,:] = discretize_labels(labels_raw['todiscretize'][..., ts, :],
                                                                    self.discretize_params['bin_edges'],
                                                                    soften_to_ends=True)
            else:
                labels_raw['discrete'][...,ts,:,:] = multi_discrete

        return

    def multi_to_multiidct(self, multi):
        if not self.is_dct():
            return multi

        multi_idct = np.zeros(multi.shape, dtype=multi.dtype)
        idct_m = self.idct_m.T
        idx_nextcossinrelative_to_nextcossin = self.idx_nextcossinrelative_to_nextcossin
        idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
        for irel in range(self.d_next_cossin_relative):
            i = idx_nextcossinrelative_to_nextcossin[irel]
            idxfeat = np.nonzero((idx_multi_to_multifeattpred[:, 0] == i) & \
                                 (idx_multi_to_multifeattpred[:, 1] > 1))[0]
            # features are in order
            assert np.all(idx_multi_to_multifeattpred[idxfeat, 1] == np.arange(2, self.ntspred_relative + 1))
            multi_dct = multi[..., idxfeat].reshape((-1, self.ntspred_relative - 1))
            multi_idct[..., idxfeat] = (multi_dct @ idct_m).reshape((multi.shape[:-1]) + (self.ntspred_relative - 1,))
        return multi_idct

    def get_idx_mutli_to_futureglobal(self, tspred=None):
        idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
        idx = np.isin(idx_multi_to_multifeattpred[:, 0], self.idx_nextcossinglobal_to_nextcossin)
        if tspred is not None:
            idx = idx & (np.isin(idx_multi_to_multifeattpred[:, 1], tspred))
        return idx

    def multi_to_futureglobal(self, multi, tspred=None):
        idx = self.get_idx_mutli_to_futureglobal(tspred)
        return multi[..., idx]

    def get_future_global(self, tspred=None, **kwargs):
        multi = self.get_multi(**kwargs)
        futureglobalvel = self.multi_to_futureglobal(multi, tspred=tspred)
        if tspred is None:
            ntspred = len(self.tspred_global)
        elif hasattr(tspred, '__len__'):
            ntspred = len(tspred)
        else:
            ntspred = 1
            tspred = [tspred, ]

        futureglobalvel = futureglobalvel.reshape((futureglobalvel.shape[:-1]) + (ntspred, self.d_next_cossin_global))

        return futureglobalvel

    def get_future_global_as_discrete(self, tspred=None, ts=None, **kwargs):
        # TODO: add some checks that global are discrete
        if not self.is_discretized():
            return None
        labels_raw = self.get_raw_labels(format='standard', ts=ts, makecopy=False)
        labels_discrete = np.zeros(self.pre_sz + (self.ntimepoints, self.d_multi, self.discretize_nbins),
                                   dtype=self.dtype)
        labels_discrete[:] = np.nan
        labels_discrete[..., self.idx_multidiscrete_to_multi, :] = labels_raw['discrete']
        idx = self.get_idx_mutli_to_futureglobal(tspred)
        labels_discrete = labels_discrete[..., idx, :]
        if tspred is None:
            ntspred = len(self.tspred_global)
        elif hasattr(tspred, '__len__'):
            ntspred = len(tspred)
        else:
            ntspred = 1

        labels_discrete = labels_discrete.reshape(
            self.pre_sz + (self.ntimepoints, ntspred, self.d_next_cossin_global, self.discretize_nbins))
        return labels_discrete

    def futureglobal_to_futureglobalpos(self, globalpos0, futureglobalvel, **kwargs):
        # futureglobalvel is szrest x T x ntspred x d_next_cossin_global

        szrest = futureglobalvel.shape[:-3]
        n = int(np.prod(szrest))
        T = futureglobalvel.shape[-3]
        ntspred = futureglobalvel.shape[-2]
        futureglobalvel = futureglobalvel.reshape((n, T, ntspred, self.d_next_global))
        globalpos0 = globalpos0[..., :T, :].reshape((n, T, self.d_next_global))
        xorigin0 = np.tile(globalpos0[..., None, :2], (1, 1, ntspred, 1))
        xtheta0 = np.tile(globalpos0[..., None, 2], (1, 1, ntspred))
        xoriginvelrel = futureglobalvel[..., [1, 0]]  # forward=y then sideways=x
        xoriginvel = rotate_2d_points(xoriginvelrel.reshape((n * T * ntspred, 2)),
                                      -xtheta0.reshape(n * T * ntspred)).reshape((n, T, ntspred, 2))
        xorigin = xorigin0 + xoriginvel
        xtheta = modrange(xtheta0 + futureglobalvel[..., 2], -np.pi, np.pi)
        futureglobalpos = np.concatenate((xorigin, xtheta[..., None]), axis=-1)

        return futureglobalpos.reshape(szrest + (T, ntspred, self.d_next_global))

    def get_future_globalpos(self, tspred=None, **kwargs):
        globalpos0 = self.get_next_pose_global(**kwargs)
        futureglobal = self.get_future_global(tspred=tspred, **kwargs)
        futureglobalpos = self.futureglobal_to_futureglobalpos(globalpos0, futureglobal, **kwargs)
        return futureglobalpos

    def get_multi_idct(self, **kwargs):
        multi = self.get_multi(**kwargs)
        return self.multi_to_multiidct(multi)

    def multiidct_to_futurecossinrelative(self, multi_idct, tspred=None):
        if not self.is_dct():
            return np.zeros(self.pre_sz + (self.ntimepoints, 0), dtype=multi_idct.dtype)
        if tspred is None:
            tspred = np.arange(2, self.ntspred_relative + 1)
        elif not hasattr(tspred, '__len__'):
            tspred = [tspred, ]
        ntspred = len(tspred)
        idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
        idxfeat = np.nonzero(np.isin(idx_multi_to_multifeattpred[:, 0], self.idx_nextcossinrelative_to_nextcossin) & \
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
        if not self.is_velocity:
            return futurerel
        relpose0 = self.get_next_pose_relative(**kwargs)
        return futurerel + relpose0[..., :-1, None, :]

    def multi_to_nextcossin(self, multi):
        next_cossin = multi[..., self.idx_nextcossin_to_multi]
        return next_cossin

    def get_nextcossin(self, **kwargs):
        # note that multi_idct is ignored since we don't use the dct representation for the next frame
        multi = self.get_multi(**kwargs)
        return self.multi_to_nextcossin(multi)

    def set_nextcossin(self, nextcossin, **kwargs):
        nextcossin = np.atleast_2d(nextcossin)
        multi = self.get_multi(**kwargs)
        multi[..., self.idx_nextcossin_to_multi] = nextcossin
        self.set_multi(multi, **kwargs)

    def nextcossinglobal_to_nextglobal(self, next_cossinglobal):
        return next_cossinglobal

    def nextcossinrelative_to_nextrelative(self, next_cossin_relative):
        szrest = next_cossin_relative.shape[:-2]
        T = next_cossin_relative.shape[-2]
        n = int(np.prod(szrest))
        next_cossin_relative = next_cossin_relative.reshape((n, T, self.d_next_cossin_relative))
        next_relative = np.zeros((n, T, self.d_next_relative), dtype=next_cossin_relative.dtype)
        idx_nextrelative_to_nextcossinrelative = self.idx_nextrelative_to_nextcossinrelative
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
        next[..., self.idx_nextglobal_to_next] = \
            self.nextcossinglobal_to_nextglobal(next_cossin[..., self.idx_nextcossinglobal_to_nextcossin])
        next[..., self.idx_nextrelative_to_next] = \
            self.nextcossinrelative_to_nextrelative(next_cossin[..., self.idx_nextcossinrelative_to_nextcossin])
        return next

    def next_to_nextcossin(self, next):
        szrest = next.shape[:-1]
        n = np.prod(szrest)
        next_cossin = np.zeros((n, self.d_next_cossin), dtype=next.dtype)
        idx_next_to_nextcossin = self.idx_next_to_nextcossin
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

        idx_next_to_nextcossin = self.idx_next_to_nextcossin

        idx_next_cossin = []
        for i in idx_next:
            ic = idx_next_to_nextcossin[i]
            if type(ic) is np.ndarray:
                idx_next_cossin = idx_next_cossin + ic.tolist()
            else:
                idx_next_cossin.append(ic)

        return idx_next_cossin

    def convert_idx_nextcossin_to_multi(self, idx_nextcossin):
        idx_nextcossin_to_multi = self.idx_nextcossin_to_multi
        idx_multi = idx_nextcossin_to_multi[idx_nextcossin]
        return idx_multi

    def convert_idx_next_to_multi(self, idx_next):
        idx_next_cossin = self.convert_idx_next_to_nextcossin(idx_next)
        idx_multi = self.convert_idx_nextcossin_to_multi(idx_next_cossin)

        return idx_multi

    def convert_idx_next_to_multi_anyt(self, idx_next):
        idx_next_cossin = self.convert_idx_next_to_nextcossin(idx_next)
        idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
        idx_multi_anyt = np.nonzero(np.isin(idx_multi_to_multifeattpred[:, 0], idx_next_cossin))[0]
        ts = idx_multi_to_multifeattpred[idx_multi_anyt, 1]
        return idx_multi_anyt, ts

    def globalvel_to_globalpos(self, globalvel, starttoff=0, init_pose=None):

        n = globalvel.shape[0]
        T = globalvel.shape[1]

        if init_pose is None:
            init_pose = self.init_pose[...,starttoff]
            
        globalpos0 = init_pose[..., self.idx_nextglobal_to_next]
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
            init_pose = self.init_pose[..., starttoff]
        relpose0 = init_pose[..., self.idx_nextrelative_to_next]

        if self.is_velocity:
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
        globalvel = next[..., self.idx_nextglobal_to_next]
        globalpos = self.globalvel_to_globalpos(globalvel, starttoff=starttoff, init_pose=init_pose)

        relrep = next[..., self.idx_nextrelative_to_next]
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
        if self.is_velocity:
            next = np.diff(nextpose, axis=1)
        else:
            idx_nextglobal_to_next = self.idx_nextglobal_to_next
            next = nextpose[..., 1:, :].copy()
            next[..., idx_nextglobal_to_next] = np.diff(nextpose[..., idx_nextglobal_to_next], axis=1)
        next[..., self.is_angle_next] = modrange(next[..., self.is_angle_next], -np.pi, np.pi)
        next = next.reshape(szrest + (T - 1, self.d_next))

        return next, init_pose

    def next_to_nextvelocity(self, next):

        if self.is_velocity:
            return next

        szrest = next.shape[:-2]
        n = int(np.prod(szrest))
        T = next.shape[-2]
        idx_nextrelative_to_next = self.idx_nextrelative_to_next
        velrel = np.zeros((n, T + 1, self.d_next_relative), dtype=next.dtype)
        velrel[:, 0, :] = self.init_pose[idx_nextrelative_to_next]
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
        next_relative = next[..., self.idx_nextrelative_to_next]
        return next_relative

    def next_to_nextglobal(self, next):
        next_global = next[..., self.idx_nextglobal_to_next]
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
        self.pre_sz = nextpose.shape[:-2]
        next, init_pose = self.nextpose_to_next(nextpose)
        self.set_init_pose(init_pose.T)
        self.set_next(next, zscored=False)

    def nextpose_to_nextkeypoints(self, pose):

        if self.scale.ndim == 1:
            nflies = 1
        else:
            nflies = int(np.prod(self.scale.shape[:-1]))

        # input to feat2kp is expected to be an np.ndarray with shape nfeatures x T x nflies
        if nflies == 1:
            szrest = pose.shape[:-1]
            n = int(np.prod(szrest))
            pose = pose.reshape((n, self.d_next)).T
            scale = self.scale
        else:
            szrest = pose.shape[:-2]
            T = pose.shape[-2]
            n = int(np.prod(szrest))
            assert n == nflies
            pose = pose.reshape((n, T, self.d_next)).transpose((1, 2, 0))
            scale = self.scale.reshape((nflies, -1)).T
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
        discretize_idx = self.idx_multidiscrete_to_multi
        example['todiscretize'] = example['continuous'][..., discretize_idx].copy()
        example['discrete'] = discretize_labels(example['todiscretize'], self.discretize_params['bin_edges'],
                                                soften_to_ends=True)
        example['continuous'] = example['continuous'][..., self.idx_multicontinuous_to_multi]
        return

    def set_keypoints(self, Xkp, scale=None):

        if (scale is None) and (self.scale is not None):
            scale = self.scale

        # function for computing features
        example = compute_features(Xkp[..., None], scale_perfly=scale, outtype=np.float32,
                                   simplify_out=self.simplify_out,
                                   dct_m=self.dct_m,
                                   tspred_global=self.tspred_global,
                                   compute_pose_vel=self.is_velocity,
                                   discreteidx=self.idx_nextdiscrete_to_next,
                                   simplify_in='no_sensory')

        self.set_raw_example(example,dozscore=self.is_zscored(),dodiscretize=self.is_discretized())
        
        if self.init_pose is None:
          self.init_pose = kp2feat(Xkp[:,:,:2],scale)[...,0]
        
        return
    
    def get_next_velocity(self, **kwargs):

        next = self.get_next(**kwargs)

        # global will always be velocity
        if self.is_velocity:
            return next

        next_vel = self.next_to_nextvelocity(next)

        return next_vel

    def set_zscore_params(self, zscore_params):
        self.zscore_params = zscore_params
        return

    def add_next_noise(self, eta_next, zscored=False):
        next = self.get_next(zscored=zscored)
        next = next + eta_next
        self.set_next(next, zscored=zscored)

    def get_nextglobal_names(self):
        return ['forward', 'sideways', 'orientation']

    def get_nextrelative_names(self):
        idx_nextrelative_to_next = self.idx_nextrelative_to_next
        return [posenames[i] for i in idx_nextrelative_to_next]

    def get_next_names(self):
        next_names = [None, ] * self.d_next
        next_names_global = self.get_nextglobal_names()
        next_names_relative = self.get_nextrelative_names()
        for i, inext in enumerate(self.idx_nextglobal_to_next):
            next_names[inext] = next_names_global[i]
        for i, inext in enumerate(self.idx_nextrelative_to_next):
            next_names[inext] = next_names_relative[i]
        return next_names

    def get_nextcossin_names(self):
        next_names = self.get_next_names()
        idx_next_to_nextcossin = self.idx_next_to_nextcossin
        next_names_cossin = [None, ] * self.d_next_cossin
        for i, ics in enumerate(idx_next_to_nextcossin):
            if hasattr(ics, '__len__'):
                next_names_cossin[ics[0]] = next_names[i] + '_cos'
                next_names_cossin[ics[1]] = next_names[i] + '_sin'
            else:
                next_names_cossin[ics] = next_names[i]
        return next_names_cossin

    def get_multi_names(self):
        ft = self.idx_multi_to_multifeattpred
        ismulti = (np.max(self.tspred_global) > 1) or (self.ntspred_relative > 1)
        multi_names = [None, ] * self.d_multi
        nextcs_names = self.get_nextcossin_names()
        for i in range(self.d_multi):
            if ismulti:
                multi_names[i] = nextcs_names[ft[i, 0]] + '_' + str(ft[i, 1])
        return multi_names

    def select_featidx_plot(self, ntsplot=None, ntsplot_global=None, ntsplot_relative=None):

        idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
        idx_multifeattpred_to_multi = self.idx_multifeattpred_to_multi
        ntspred_global = len(self.tspred_global)
        if ntsplot_global is None and ntsplot is not None:
            ntsplot_global = ntsplot
        if ntsplot_global is None or (ntsplot >= ntspred_global):
            idxglobal = self.idx_multiglobal_to_multi
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
            for fi, f in enumerate(self.idx_nextcossinglobal_to_nextcossin):
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
            idxrelative = self.idx_multirelative_to_multi
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
            for fi, f in enumerate(self.idx_nextcossinrelative_to_nextcossin):
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
