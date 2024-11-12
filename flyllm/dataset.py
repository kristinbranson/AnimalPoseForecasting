import numpy as np
import copy
import tqdm
import torch

from apf.data import fit_discretize_labels, discretize_labels, weighted_sample
from apf.utils import zscore, unzscore
from apf.models import (  # TODO: dataset should not depend on models
    generate_square_full_mask,
    apply_mask,
    get_output_and_attention_weights,
    pred_apply_fun
)
from flyllm.pose import FlyExample
import logging
LOG = logging.getLogger(__name__)

class FlyMLMDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data: list[dict[str, np.ndarray | dict[str, int]]],
            max_mask_length: int | None = None,
            pmask: float | None = None,
            masktype: str | None = 'block',
            simplify_out: str | None = None,
            simplify_in: str | None = None,
            pdropout_past: float = 0.,
            maskflag: bool | None = None,
            input_labels: bool = True,
            dozscore: bool = False,
            zscore_params: dict[str, np.ndarray] | None = None,
            discreteidx: np.ndarray | None = None,
            discrete_tspred: tuple[int, ...] = (1, ),
            discretize_nbins: int = 50,
            discretize_epsilon: np.ndarray | None = None,
            discretize_params: dict[str: np.ndarray] | None = None,
            flatten_labels: bool = False,
            flatten_obs_idx: dict[str, list[int]] | None = None,
            flatten_do_separate_inputs: bool = False,
            input_noise_sigma: np.ndarray | None = None,
            p_add_input_noise: float = 0,
            dct_ms: tuple[np.ndarray | None, np.ndarray | None] | None = None,
            tspred_global: tuple[int, ...] = (1, ),
            compute_pose_vel: bool = True,
    ) -> None:
        """
        Args
            data: List of dictionaries per batch. Each dictionary contains:
                'input': np.ndarray float32 of size len_context x n_features
                'labels': np.ndarray float32 of size len_context x n_labels
                'init': np.ndarray float32 of size 3 x 2 (n_features x n_buff_frames)
                    TODO: Not sure what this is
                        Two frames of info when predicting velocities for next frame (fwd, side, and ori feats)
                        Sometimes it is the full pose, then its n_pose x 2
                        Sometimes it has only one frame
                'scale': np.ndarray float32 of size n_scales
                'metadata': dict containing:
                    'flynum': int
                    'id': int
                    't0': int
                    'videoidx': int
                    'frame0': int
                'categories': np.ndarray float32 of size n_categories x ?
            max_mask_length:
            pmask:
            masktype:
            simplify_out: One of {'relative', ...}
            simplify_in: One of {'no_sensory', ...}
            pdropout_past:
            maskflag:
            input_labels:
            dozscore:
            zscore_params:
            discreteidx:
            discrete_tspred:
            discretize_nbins:
            discretize_epsilon:
            discretize_params:
            flatten_labels:
            flatten_obs_idx:
            flatten_do_separate_inputs:
            input_noise_sigma:
            p_add_input_noise:
            dct_ms:
            tspred_global:
            compute_pose_vel:
        """
        
        # set mutable defaults
        if zscore_params is None:
            zscore_params = {}
        if discretize_params is None:
            discretize_params = {}

        # copy dicts
        data = [example.copy() for example in data]

        # dtype should be float32
        self.dtype = np.float32

        # parameters for masking, dropout
        self.max_mask_length = max_mask_length
        self.pmask = pmask
        self.masktype = masktype
        self.pdropout_past = pdropout_past
        if maskflag is None:
            maskflag = (masktype is not None) or (pdropout_past > 0.)
        self.maskflag = maskflag
        
        # parameters for feature computation

        # modulation of task to make it easier
        self.simplify_out = simplify_out 
        self.simplify_in = simplify_in

        # which indices of next frame predictions to discretize
        self.discretefeat = discreteidx

        # discrete cosine transform
        self.dct_m = None
        self.idct_m = None
        if dct_ms is not None:
            self.dct_m = dct_ms[0]
            self.idct_m = dct_ms[1]
        self.tspred_global = tspred_global

        # whether to predict relative pose velocities (true) or position (false)
        self.compute_pose_vel = compute_pose_vel

        # whether to input previous frame's labels
        if input_labels:
            assert (masktype is None)
            assert (pdropout_past == 0.)
        self.input_labels = input_labels

        # discretization parameters
        # these will be overwritten during discretizing
        self.discrete_tspred = np.array([])
        self.discreteidx = np.array([])
        self.discretize = False
        self.discretize_nbins = None
        self.discretize_bin_samples = None
        self.discretize_bin_edges = None
        self.discretize_bin_means = None
        self.discretize_bin_medians = None

        # zscoring parameters
        # these will be overwritten during z-scoring
        self.mu_input = None
        self.sig_input = None
        self.mu_labels = None
        self.sig_labels = None
        
        # flatten parameters
        self.flatten_labels = False
        self.flatten_obs_idx = None
        self.flatten_obs = False
        self.flatten_nobs_types = None
        self.flatten_nlabel_types = None
        self.flatten_dinput_pertype = None
        self.flatten_max_dinput = None
        self.flatten_max_doutput = None

        self.input_noise_sigma = input_noise_sigma
        self.p_add_input_noise = p_add_input_noise
        self.set_eval_mode()

        # apply all transforms to data
        
        # zscore
        if dozscore:
            LOG.info('Z-scoring data...')
            data = self.zscore(data, **zscore_params)
            LOG.info('Done.')
        
        # discretize
        if discreteidx is not None:
            LOG.info('Discretizing labels...')
            data = self.discretize_labels(data, discreteidx, discrete_tspred, nbins=discretize_nbins,
                                          bin_epsilon=discretize_epsilon, **discretize_params)
            LOG.info('Done.')

        # flatten -- flattening hasn't been implemented in pose class yet
        self.set_flatten_params(flatten_labels=flatten_labels, flatten_obs_idx=flatten_obs_idx,
                                flatten_do_separate_inputs=flatten_do_separate_inputs)

        # store examples in objects
        LOG.info('Creating FlyExample objects...')
        self.data = []
        for i in tqdm.trange(len(data)):
            example_in = data[i]
            self.data.append(FlyExample(example_in, dataset=self))
            # if i > 1000:
            #     LOG.warning('DEBUGGING: breaking after 100 examples')
            #     break
        LOG.info('Done.')

        self.set_train_mode()

    @property
    def ntimepoints(self):
        # number of time points in training examples
        n = self.data[0].ntimepoints-self.get_start_toff()
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
        l = self.ntimepoints * self.ntokens_per_timepoint
        if (self.flatten_labels or self.flatten_obs) and not self.ismasked():
            l -= 1
        return l

    @property
    def flatten(self):
        return self.flatten_obs or self.flatten_labels

    @property
    def continuous(self):
        return self.data[0].labels.is_continuous()

    @property
    def noutput_tokens_per_timepoint(self):
        if self.flatten_labels and self.discretize:
            return len(self.discreteidx) + int(self.continuous)
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

    @property
    def ntspred_max(self):
        return np.maximum(self.ntspred_relative, np.max(self.tspred_global))

    @property
    def is_zscored(self):
        return self.mu_input is not None

    def set_train_mode(self):
        self.do_add_noise = self.input_noise_sigma is not None and self.p_add_input_noise > 0

    def set_eval_mode(self):
        self.do_add_noise = False

    def set_flatten_params(self, flatten_labels=False, flatten_obs_idx=None, flatten_do_separate_inputs=False):

        # TODO REMOVE THESE
        self.flatten_labels = flatten_labels
        self.flatten_obs_idx = flatten_obs_idx
        if self.flatten_labels:
            if self.flatten_obs_idx is None:
                self.flatten_obs_idx = {'all': [0, self.dfeat]}
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
            self.flatten_dinput_pertype = np.zeros(self.flatten_nobs_types + self.flatten_nlabel_types, dtype=int)
            for i, v in enumerate(self.flatten_obs_idx.values()):
                self.flatten_dinput_pertype[i] = v[1] - v[0]
            if self.flatten_labels and self.discretize:
                self.flatten_dinput_pertype[self.flatten_nobs_types:] = self.discretize_nbins
            if self.d_output_continuous > 0:
                self.flatten_dinput_pertype[-1] = self.d_output_continuous
            self.flatten_max_dinput = np.max(self.flatten_dinput_pertype)
            if self.flatten_do_separate_inputs:
                self.flatten_dinput = np.sum(self.flatten_dinput_pertype)
            else:
                self.flatten_dinput = self.flatten_max_dinput

            self.flatten_input_type_to_range = np.zeros((self.flatten_dinput_pertype.size, 2), dtype=int)

            if self.discretize:
                self.flatten_max_doutput = np.maximum(self.discretize_nbins, self.d_output_continuous)
            else:
                self.flatten_max_doutput = self.d_output_continuous

            if self.flatten_do_separate_inputs:
                cs = np.cumsum(self.flatten_dinput_pertype)
                self.flatten_input_type_to_range[1:, 0] = cs[:-1]
                self.flatten_input_type_to_range[:, 1] = cs
            else:
                self.flatten_input_type_to_range[:, 1] = self.flatten_dinput_pertype

            # label tokens should be:
            # observations (flatten_nobs_types)
            # discrete outputs (d_output_discrete)
            # continuous outputs (<=1)
            self.idx_output_token_discrete = torch.arange(self.flatten_nobs_types,
                                                          self.flatten_nobs_types + self.d_output_discrete, dtype=int)
            if self.d_output_continuous > 0:
                self.idx_output_token_continuous = torch.tensor([self.ntokens_per_timepoint - 1, ])
            else:
                self.idx_output_token_continuous = torch.tensor([])

        return

    def discretize_labels(self, data, discreteidx, discrete_tspred, nbins=50,
                          bin_edges=None, bin_samples=None, bin_epsilon=None,
                          bin_means=None, bin_medians=None, **kwargs):
        """
            discretize_labels(self,discreteidx,discrete_tspred,nbins=50,bin_edges=None,bin_samples=None,bin_epsilon=None,**kwargs)
            For each feature in discreteidx, discretize the labels into nbins bins. For each example in the data,
            labels_discrete is an ndarray of shape T x len(discreteidx) x nbins, where T is the number of time points, and
            indicates whether the label is in each bin, with soft-binning.
            labels_todiscretize is an ndarray of shape T x len(discreteidx) with the original continuous labels.
            labels gets replaced with an ndarray of shape T x len(continuous_idx) with the continuous labels.
            discretize_bin_edges is an ndarray of shape len(discreteidx) x (nbins+1) with the bin edges for each discrete feature.
            discretize_bin_samples is an ndarray of shape nsamples x len(discreteidx) x nbins with samples from each bin
        """

        if not isinstance(discreteidx, np.ndarray):
            discreteidx = np.array(discreteidx)
        if not isinstance(discrete_tspred, np.ndarray):
            discrete_tspred = np.array(discrete_tspred)

        self.discrete_tspred = discrete_tspred

        bin_epsilon_feat = np.array(bin_epsilon)
        assert len(bin_epsilon_feat) <= len(discreteidx)
        if len(bin_epsilon_feat) < len(discreteidx):
            bin_epsilon_feat = np.concatenate((bin_epsilon_feat, np.zeros(len(discreteidx) - len(bin_epsilon_feat))))

        # translate to multi time point representation
        dummyexample = FlyExample(dataset=self)
        discreteidx_next = discreteidx
        bin_epsilon = np.zeros(dummyexample.labels.d_multi)
        bin_epsilon[:] = np.nan
        for i, i_next in enumerate(discreteidx_next):
            # indices for all tspred associated with this feature
            idx_multi_curr, _ = dummyexample.labels.get_multiidx_for_featidx(i_next)
            idx_multi_curr = idx_multi_curr[dummyexample.labels.get_multi_isdiscrete(idx_multi_curr)]
            #idx_multi_curr = idx_multi_curr[np.isin(idx_multi_curr, dummyexample.labels.idx_multidiscrete_to_multi)]
            bin_epsilon[idx_multi_curr] = bin_epsilon_feat[i]

        isdiscrete = np.isnan(bin_epsilon) == False
        self.discreteidx = np.nonzero(isdiscrete)[0]
        self.bin_epsilon = bin_epsilon[self.discreteidx]

        self.discretize_nbins = nbins

        assert ((bin_edges is None) == (bin_samples is None))

        if bin_edges is None:
            if self.sig_labels is not None:
                bin_epsilon = np.array(self.bin_epsilon) / self.sig_labels[self.discreteidx]
            self.discretize_bin_edges, self.discretize_bin_samples, self.discretize_bin_means, self.discretize_bin_medians = \
                fit_discretize_labels(data, self.discreteidx, nbins=nbins, bin_epsilon=bin_epsilon, **kwargs)
        else:
            self.discretize_bin_samples = bin_samples
            self.discretize_bin_edges = bin_edges
            self.discretize_bin_means = bin_means
            self.discretize_bin_medians = bin_medians
            assert nbins == bin_edges.shape[-1] - 1

        for example in tqdm.tqdm(data):
            example['labels_todiscretize'] = example['labels'][:, self.discreteidx]
            example['labels_discrete'] = discretize_labels(example['labels_todiscretize'], self.discretize_bin_edges,
                                                           soften_to_ends=True)
            example['labels'] = example['labels'][:, isdiscrete==False]

        self.discretize = True
        self.discretize_fun = lambda x: discretize_labels(x, self.discretize_bin_edges, soften_to_ends=True)

        return data

    def get_discretize_params(self):

        discretize_params = {
            'bin_edges': self.discretize_bin_edges,
            'bin_samples': self.discretize_bin_samples,
            'bin_means': self.discretize_bin_means,
            'bin_medians': self.discretize_bin_medians,
        }
        return discretize_params

    def get_bin_edges(self, zscored=False):

        if self.discretize == False:
            return

        if zscored or (self.mu_labels is None):
            bin_edges = self.discretize_bin_edges
        else:
            bin_edges = self.unzscore_labels(self.discretize_bin_edges.T, self.discreteidx).T

        return bin_edges

    def get_bin_samples(self, zscored=False):
        if self.discretize == False:
            return

        if zscored or (self.mu_labels is None):
            bin_samples = self.discretize_bin_samples
        else:
            sz = self.discretize_bin_samples.shape
            bin_samples = self.discretize_bin_samples.transpose(0, 2, 1).reshape((sz[0] * sz[2], sz[1]))
            bin_samples = self.unzscore_labels(bin_samples, self.discreteidx)
            bin_samples = bin_samples.reshape((sz[0], sz[2], sz[1])).transpose(0, 2, 1)

        return bin_samples

    def metadata_to_index(self, flynum, t0):
        starttoff = self.get_start_toff()
        for i, d in enumerate(self.data):
            if (d.metadata['t0'] == t0 - starttoff) and (d.metadata['flynum'] == flynum):
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

    def unzscore_labels(self, zlabels, featidx=None):
        if self.mu_labels is None:
            rawlabels = zlabels.copy()
        else:
            if featidx is None:
                rawlabels = unzscore(zlabels, self.mu_labels, self.sig_labels)
            else:
                rawlabels = unzscore(zlabels, self.mu_labels[featidx], self.sig_labels[featidx])
        return rawlabels.astype(self.dtype)

    def zscore(self, data, mu_input=None, sig_input=None, mu_labels=None, sig_labels=None):
        """
        self.zscore(mu_input=None,sig_input=None,mu_labels=None,sig_labels=None)
        zscore the data. input and labels are z-scored for each example in data
        and converted to float32. They are stored in place in the dict for each example
        in the dataset. If mean and standard deviation statistics are input, then
        these statistics are used for z-scoring. Otherwise, means and standard deviations
        are computed from this data.

        Args:
            mu_input (ndarray, dfeat, optional): Pre-computed mean for z-scoring input.
            If None, mu_input is computed as the mean of all the inputs in data.
            Defaults to None.
            sig_input (ndarray, dfeat, optional): Pre-computed standard deviation for
            z-scoring input. If mu_input is None, sig_input is computed as the std of all
            the inputs in data. Defaults to None. Do not set this to None if mu_input
            is not None.
            mu_labels (ndarray, d_output_continuous, optional): Pre-computed mean for z-scoring labels.
            If None, mu_labels is computed as the mean of all the labels in data.
            Defaults to None.
            sig_labels (ndarray, dfeat, optional): Pre-computed standard deviation for
            z-scoring labels. If mu_labels is None, sig_labels is computed as the standard
            deviation of all the labels in data. Defaults to None. Do not set this
            to None if mu_labels is not None.

        No value returned.
        """

        # must happen before discretizing
        assert self.discretize == False, 'z-scoring should happen before discretizing'

        def zscore_helper(f):
            mu = 0.
            sig = 0.
            n = 0.
            for example in data:
                # input is T x dfeat
                n += np.sum(np.isnan(example[f]) == False, axis=0)
                mu += np.nansum(example[f], axis=0)
                sig += np.nansum(example[f] ** 2., axis=0)
            mu = mu / n
            sig = np.sqrt(sig / n - mu ** 2.)
            assert (np.any(np.isnan(mu)) == False)
            assert (np.any(np.isnan(sig)) == False)

            return mu, sig

        if mu_input is None:
            self.mu_input, self.sig_input = zscore_helper('input')
        else:
            self.mu_input = mu_input.copy()
            self.sig_input = sig_input.copy()

        self.mu_input = self.mu_input.astype(self.dtype)
        self.sig_input = self.sig_input.astype(self.dtype)

        if mu_labels is None:
            self.mu_labels, self.sig_labels = zscore_helper('labels')
        else:
            self.mu_labels = mu_labels.copy()
            self.sig_labels = sig_labels.copy()

        self.mu_labels = self.mu_labels.astype(self.dtype)
        self.sig_labels = self.sig_labels.astype(self.dtype)

        for example in data:
            example['input'] = zscore(example['input'],self.mu_input,self.sig_input)
            example['labels'] = zscore(example['labels'],self.mu_labels,self.sig_labels)

        return data

    def get_poselabel_params(self):
        return self.data[0].labels.get_params()

    def get_flyexample_params(self):
        return self.data[0].get_params()

    def get_zscore_params(self):

        zscore_params = {
            'mu_input': self.mu_input,
            'sig_input': self.sig_input,
            'mu_labels': self.mu_labels,
            'sig_labels': self.sig_labels,
        }
        return zscore_params

    def maskblock(self, inl):
        # choose a mask length
        maxl = min(inl - 1, self.max_mask_length)
        l = np.random.randint(1, self.max_mask_length)

        # choose mask start
        t0 = np.random.randint(0, inl - l)
        t1 = t0 + l

        # create mask
        mask = torch.zeros(inl, dtype=bool)
        mask[t0:t1] = True

        return mask

    def masklast(self, inl):
        mask = torch.zeros(inl, dtype=bool)
        mask[-1] = True
        return mask

    def maskind(self, inl, pmask=None):
        if pmask is None:
            pmask = self.pmask
        mask = torch.rand(inl) <= pmask
        if not torch.any(mask):
            imask = np.random.randint(inl)
            mask[imask] = True
        return mask

    def set_masktype(self, masktype):
        self.masktype = masktype

    def mask_input(self, input, masktype='default'):

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
            input = apply_mask(input, mask, self.dfeat)
            maskflagged = True
        if self.pdropout_past > 0:
            dropout_mask = self.maskind(contextl, pmask=self.pdropout_past)
            input = apply_mask(input, dropout_mask, self.dfeat, maskflagged)
            maskflagged = True
        else:
            dropout_mask = None
        if self.maskflag and not maskflagged:
            input = apply_mask(input, None)

        return input, mask, dropout_mask

    def get_start_toff(self):
        if self.ismasked() or (self.input_labels == False) or \
                self.flatten_labels or self.flatten_obs:
            starttoff = 0
        else:
            starttoff = 1
        return starttoff

    def __getitem__(self, idx: int):
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
            where example['labels'][t,:] is the continuous motion from frame t to t+1 and/or
            the pose at frame t+1
            example['labels_discrete'] is a tensor of shape contextl x d_output_discrete x
            discretize_nbins, where example['labels_discrete'][t,i,:] is one-hot encoding of
            discrete motion feature i from frame t to t+1 and/or pose at frame t+1.
            example['init'] is a tensor of shape dglobal, corresponding to the global
            position in frame 0.
            example['mask'] is a tensor of shape contextl indicating which frames are masked.

            For causal LMs:
            example['input'] is a tensor of shape (contextl-1) x (d_input_labels + dfeat).
            if input_labels == True, example['input'][t,:d_input_labels] is the motion from
            frame t to t+1 and/or the pose at frame t+1,
            example['input'][t,d_input_labels:] are the input features for
            frame t+1.
            example['labels'] is a tensor of shape contextl x d_output
            where example['labels'][t,:] is the motion from frame t+1 to t+2 and/or the pose at
            frame t+2.
            example['init'] is a tensor of shape dglobal, corresponding to the global
            position in frame 1.
            example['labels_discrete'] is a tensor of shape contextl x d_output_discrete x
            discretize_nbins, where example['labels_discrete'][t,i,:] is one-hot encoding of
            discrete motion feature i from frame t+1 to t+2 and/or pose at frame t+2.

            For all:
            example['scale'] are the scale features for this fly, used for converting from
            relative pose features to keypoints.
            example['categories'] are the currently unused categories for this sequence.
            example['metadata'] is a dict of metadata about this sequence.
            example['idx'] is the index of this example in the dataset.

        """

        res = self.data[idx].get_train_example()
        res['idx'] = idx
        res['input'], mask, dropout_mask = self.mask_input(res['input'])
        if mask is not None:
            res['mask'] = mask
        if dropout_mask is not None:
            res['dropout_mask'] = dropout_mask

        return res

    def get_example(self, idx: int):
        """
        example = self.get_example(idx)
        Returns dataset example idx, FlyExample object
        """
        return self.data[idx]

    # TODO REMOVE THIS AFTER CHECKING ADDING NOISE

    def add_noise(self, example, input_labels):

        # add noise to the input movement and pose
        # desire is to do movement truemovement(t-1->t)
        # movement noisemovement(t-1->t) = truemovement(t-1->t) + eta(t) actually done
        # resulting in noisepose(t) = truepose(t) + eta(t)[featrelative]
        # output should then be fixmovement(t->t+1) = truemovement(t->t+1) - eta(t)
        # input pose: noise_input_pose(t) = truepose(t) + eta(t)[featrelative]
        # input movement: noise_input_movement(t-1->t) = truemovement(t-1->t) + eta(t)
        # output movement: noise_output_movement(t->t+1) = truemovement(t->t+1) - eta(t)
        # movement(t->t+1) = truemovement(t->t+1)

        example = copy.deepcopy(example)

        T = example.ntimesteps
        d_labels = example.labels.d_next

        # # divide sigma by standard deviation if zscored
        # if self.sig_labels is not None:
        #   input_noise_sigma = self.input_noise_sigma / self.sig_labels

        # additive noise
        eta = np.zeros((T, d_labels))
        do_add_noise = np.random.rand(T) <= self.p_add_input_noise
        eta[do_add_noise, :] = self.input_noise_sigma[None, :] * np.random.randn(np.count_nonzero(do_add_noise),
                                                                                 self.d_output)

        # problem with multiplicative noise is that it is 0 when the movement is 0 -- there should always be some jitter
        # etamult = np.maximum(-self.max_input_noise,np.minimum(self.max_input_noise,self.input_noise_sigma[None,:]*np.random.randn(input.shape[0],self.d_output)))
        # if self.input_labels:
        #   eta = input_labels*etamult
        # else:
        #   labelsprev = torch.zeros((labels.shape[0],nfeatures),dtype=labels.dtype,device=labels.device)
        #   if self.continuous:
        #     labelsprev[1:,self.continuous_idx] = labels[:-1,:]
        #   if self.discretize:
        #     labelsprev[1:,self.discreteidx] = labels_todiscretize[:-1,:]
        #   eta = labelsprev*etamult

        # input pose
        eta_pose = example.labels.next_to_nextpose(eta)
        example.inputs.add_pose_noise(eta_pose, zscored=False)

        # input labels
        if self.input_labels:
            eta_input_labels = example.labels.next_to_input_labels(eta)
            input_labels += eta_input_labels

        # output labels
        example.labels.add_next_noise(-eta, zscored=False)

        return eta

    def __len__(self):
        return len(self.data)
    
    @property
    def d_input_labels(self):
        """
        d_input_labels = self.d_input_labels
        Returns the number of labels concatenated to inputs. 
        """
        if self.input_labels:
            return self.data[0].labels.get_d_labels_input()
        else:
            return 0
        
    @property
    def d_input(self):
        """
        d_input = self.d_input
        Returns the number of features concatenated to inputs. 
        """
        return self.data[0].d_input
    
    @property
    def d_output(self):
        """
        d_output = self.d_output
        Returns the number of output features. 
        """
        return self.data[0].d_labels
            
    @property
    def d_output_discrete(self):
        """
        d_output_discrete = self.d_output_discrete
        Returns the number of discrete output features. 
        """
        return self.data[0].d_labels_discrete
    
    @property
    def d_output_continuous(self):
        """
        d_output_continuous = self.d_output_continuous
        Returns the number of continuous output features. 
        """
        return self.data[0].d_labels_continuous
            
    def get_input_shapes(self):
        return self.data[0].get_train_input_shapes()

    def get_predict_mask(self, masksize=None, device=None):
        if masksize is None:
            masksize = self.contextl

        if device is None:
            device = self.device

        if self.ismasked():
            net_mask = generate_square_full_mask(masksize).to(device)
            is_causal = False
        else:
            net_mask = torch.nn.Transformer.generate_square_subsequent_mask(masksize).to(device)
            is_causal = True

        return net_mask, is_causal

    def predict_open_loop(self, examples_pred, fliespred, scales, Xkp_fill, burnin, model, maxcontextl=np.inf, debug=False,
                          need_weights=False, nsamples=0, labels_true=None):
        """
        predict_open_loop(self,Xkp,fliespred,scales,burnin,model,sensory_params,maxcontextl=np.inf,debug=False)

        Args:
            examples_pred: list of FlyExample objects to be predicted in open loop. labels can be nan 
            for frames/flies to be predicted. Will be overwritten
            #Xkp (ndarray, nkpts x 2 x tpred x nflies): keypoints for all flies for all frames.
            #Can be nan for frames/flies to be predicted. Will be overwritten.
            #fliespred (ndarray, nfliespred): indices of flies to predict
            #scales (ndarray, nscale x nfliespred): scale parameters for the flies to be predicted
            burnin (int): number of frames to use for initialization
            maxcontextl (int, optional): maximum number of frames to use for context. Default np.inf
            debug (bool, optional): whether to fill in from movement computed from Xkp_all

        Example call:
        dataset.predict_open_loop(examples_pred,burnin,model,maxcontextl=config['contextl'],
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

        tpred = examples_pred[0].ntimepoints
        nfliespred = len(examples_pred)
        if need_weights:
            attn_weights = [None, ] * tpred

        if self.ismasked():
            # to do: figure this out for flattened models
            masktype = 'last'
            dummy = np.zeros((1, self.d_output))
            dummy[:] = np.nan
        else:
            masktype = None

        # start predicting motion from frame burnin-1 to burnin = t
        masksizeprev = None
        
        # global position of each fly in the previous frame, so that we don't have to integrate to compute position
        pose_prev = []
        for i in range(nfliespred):
            pose_curr = examples_pred[i].labels.get_next_pose(ts=np.arange(burnin),use_todiscretize=True)[-1]
            pose_prev.append(pose_curr)
        
        for t in tqdm.trange(burnin, tpred): 
            t0 = int(np.maximum(t - maxcontextl, 0))

            for i,fly in enumerate(fliespred):
                # copy frames up to t
                # don't use the init_pose
                # get_next_pose[:,-1] will be nan
                example_pred = examples_pred[i].copy_subindex(ts=np.arange(t0, t+1),needinit=False)
                # inputs will go from t0 through t
                # labels (unused) will go from t0+example_pred.starttoff through t+example_pred.starttoff
                test_example = example_pred.get_train_example()
                xcurr = test_example['input']
                assert not torch.any(torch.isnan(xcurr))
                xcurr, _, _ = self.mask_input(xcurr, masktype)

                if debug:
                    label_pred = labels_true[i].copy_subindex(ts=np.arange(t0, t+1))
                    pred = label_pred.get_train_labels()
                    pred = {k: v.reshape((1,) + v.shape) if type(v) is torch.Tensor else v for k, v in pred.items()}
                    #zmovementout = np.tile(self.zscore_labels(movement_true[t - 1, :, i]).astype(dtype)[None],
                    #                       (nsamples1, 1))
                else:

                    if self.flatten:
                        raise NotImplementedError("Flattening not yet implemented")
                        # not implemented yet
                        # to do: not sure if multiple samples here works

                        zmovementout = np.zeros((nsamples1, self.d_output), dtype=dtype)
                        zmovementout_flattened = np.zeros((self.noutput_tokens_per_timepoint, self.flatten_max_doutput),
                                                          dtype=dtype)

                        for token in range(self.noutput_tokens_per_timepoint):

                            lastidx = xcurr.shape[0] - self.noutput_tokens_per_timepoint
                            masksize = lastidx + token
                            net_mask, is_causal = self.get_predict_mask(masksize=masksize, device=device)

                            with torch.no_grad():
                                predtoken = model(xcurr[None, :lastidx + token, ...].to(device), mask=net_mask,
                                                  is_causal=is_causal)
                            # to-do: integrate with labels object
                            if token < len(self.discreteidx):
                                # sample
                                sampleprob = torch.softmax(predtoken[0, -1, :self.discretize_nbins], dim=-1)
                                binnum = int(weighted_sample(sampleprob, nsamples=nsamples1))

                                # store in input
                                xcurr[lastidx + token, binnum[0]] = 1.
                                zmovementout_flattened[token, binnum[0]] = 1.

                                # convert to continuous
                                nsamples_per_bin = self.discretize_bin_samples.shape[0]
                                sample = int(torch.randint(low=0, high=nsamples_per_bin, size=(nsamples,)))
                                zmovementcurr = self.discretize_bin_samples[sample, token, binnum]

                                # store in output
                                zmovementout[:, self.discreteidx[token]] = zmovementcurr
                            else:  # else token < len(self.discreteidx)
                                # continuous
                                zmovementout[:, self.continuous_idx] = predtoken[0, -1, :len(self.continuous_idx)].cpu()
                                zmovementout_flattened[token, :len(self.continuous_idx)] = zmovementout[
                                    self.continuous_idx, 0]

                    else:  # else flatten

                        masksize = t - t0
                        if masksize != masksizeprev:
                            net_mask, is_causal = self.get_predict_mask(masksize=masksize, device=device)
                            masksizeprev = masksize

                        if need_weights:
                            with torch.no_grad():
                                pred, attn_weights_curr = get_output_and_attention_weights(model,
                                                                                           xcurr[None, ...].to(device),
                                                                                           net_mask)
                            # dimensions correspond to layer, output frame, input frame
                            attn_weights_curr = torch.cat(attn_weights_curr, dim=0).cpu().numpy()
                            if i == 0:
                                attn_weights[t] = np.tile(attn_weights_curr[..., None], (1, 1, 1, nfliespred))
                                attn_weights[t][..., 1:] = np.nan
                            else:
                                attn_weights[t][..., i] = attn_weights_curr
                        else:
                            with torch.no_grad():
                                # predict for all frames
                                # masked: movement from 0->1, ..., t->t+1
                                # causal: movement from 1->2, ..., t->t+1
                                # last prediction: t->t+1
                                pred = model.output(xcurr[None, ...].to(device), mask=net_mask, is_causal=is_causal)
                        # to-do: this is not incorportated into sampling, probably should be
                        if model.model_type == 'TransformerBestState' or model.model_type == 'TransformerState':
                            pred = model.randpred(pred)
                        # z-scored movement from t to t+1

                    # end else flatten
                # end else debug
                        
                pred = pred_apply_fun(pred, lambda x: x[0, -1, ...].cpu().numpy() if type(x) is torch.Tensor else x)
                # set the label for frame t, but not the inputs yet
                examples_pred[i].labels.set_prediction(pred,ts=t)                
                
                # store keypoints predicted for this frame
                Xkpcurr = examples_pred[i].labels.get_next_keypoints(ts=[t,],init_pose=pose_prev[i])
                Xkp_fill[:,:,t+1,fly] = Xkpcurr[-1]


                #globapos_curr = examples_pred[i].labels.get_next_pose_global(ts=[t,],globalpos0=globalpos_prev[i])
                pose_curr = examples_pred[i].labels.get_next_pose(ts=[t,],init_pose=pose_prev[i])
                pose_prev[i] = pose_curr[-1]
                #globalpos_prev[i] = globapos_curr
                  
            # end loop over flies
            
            if t < tpred-1:
                # update observations for the next frame
                for i,fly in enumerate(fliespred):
                    # this is just one frame of inputs, so don't crop the end
                    examples_pred[i].inputs.set_inputs_from_keypoints(Xkp_fill[:,:,[t+1,],:],fly,scale=scales[i],ts=[t+1,],npad=0)

        if need_weights:
            return examples_pred, attn_weights
        else:
            return examples_pred

    def get_next_global_feature_names(self):
        return self.data[0].labels.get_nextglobal_names()

    def get_next_feature_names(self):
        return self.data[0].labels.get_nextcossin_names()

    def get_feature_names(self):
        """
        outnames = self.get_feature_names()

        Returns:
            outnames (list of strings): names of each output motion
        """
        return self.data[0].labels.get_multi_names()

    def get_model_params(self):
        model_params = {
            'zscore_params': self.get_zscore_params(),
            'discretize_params': self.get_discretize_params(),
        }
        return model_params

    # REMOVE AFTER DEBUGGING FLATTENING
    # def unflatten_labels(self, labels_flattened):
        
    #     assert self.flatten_labels
    #     sz = labels_flattened.shape
    #     newsz = sz[:-2] + (self.ntimepoints, self.ntokens_per_timepoint, self.flatten_max_doutput)
    #     if not self.ismasked():
    #         pad = torch.zeros(sz[:-2] + (1, self.flatten_max_doutput), dtype=labels_flattened.dtype,
    #                           device=labels_flattened.device)
    #         labels_flattened = torch.cat((pad, labels_flattened), dim=-2)
    #     labels_flattened = labels_flattened.reshape(newsz)
    #     if self.d_output_continuous > 0:
    #         labels_continuous = labels_flattened[..., -1, :self.d_output_continuous]
    #     else:
    #         labels_continuous = Noneclass FlyTestDataset(F)

    #     if self.discretize:
    #         labels_discrete = labels_flattened[..., self.flatten_nobs_types:, :self.discretize_nbins]
    #         if self.continuous:
    #             labels_discrete = labels_discrete[..., :-1, :]
    #     else:
    #         labels_discrete = None
    #     return labels_continuous, labels_discrete

    # def apply_flatten_input(self, input):

    #     if type(input) == np.ndarray:
    #         input = torch.Tensor(input)

    #     if self.flatten_obs == False:
    #         return input

    #     # input is of size ...,contextl,d_input
    #     sz = input.shape[:-2]
    #     contextl = input.shape[-2]
    #     newinput = torch.zeros(sz + (contextl, self.flatten_nobs_types, self.flatten_max_dinput), dtype=input.dtype)

    #     for i, v in enumerate(self.flatten_obs_idx.values()):
    #         newinput[..., i, :self.flatten_dinput_pertype[i]] = input[..., v[0]:v[1]]
    #     return newinput

    # def unflatten_input(self, input_flattened):
    #     assert self.flatten_obs
    #     sz = input_flattened.shape
    #     if not self.ismasked():
    #         pad = torch.zeros(sz[:-2] + (1, self.flatten_dinput), dtype=input_flattened.dtype,
    #                           device=input_flattened.device)
    #         input_flattened = torch.cat((input_flattened, pad), dim=-2)
    #     resz = sz[:-2] + (self.ntimepoints, self.ntokens_per_timepoint, self.flatten_dinput)
    #     input_flattened = input_flattened.reshape(resz)
    #     newsz = sz[:-2] + (self.ntimepoints, self.dfeat)
    #     newinput = torch.zeros(newsz, dtype=input_flattened.dtype)
    #     for i, v in enumerate(self.flatten_obs_idx.values()):
    #         newinput[..., :, v[0]:v[1]] = input_flattened[..., i,
    #                                       self.flatten_input_type_to_range[i, 0]:self.flatten_input_type_to_range[i, 1]]
    #     return newinput

class FlyTestDataset(FlyMLMDataset):
    def __init__(self,data: list[dict[str, np.ndarray | dict[str, int]]],
        contextl: int, need_labels: bool = False, need_metadata: bool = False, 
        need_init: bool = False, make_copy: bool = False, **kwargs):

        super().__init__(data,**kwargs)

        self._contextl = contextl
        self.n_examples_per_id = np.array([self.data[i].ntimepoints-self.contextl+1 for i in range(len(self.data))])
        self.start_example_per_id = np.r_[0,np.cumsum(self.n_examples_per_id)]
        self.need_labels = need_labels
        self.need_metadata = need_metadata
        self.make_copy = make_copy
        self.need_init = need_init

        self.set_eval_mode()

        return

    @property
    def contextl(self):
        return self._contextl

    def __len__(self):
        return self.start_example_per_id[-1]
    
    def get_example(self, idx: int, lastonly: bool = False):
        """
        example = self.get_example(idx)
        Returns dataset example idx, FlyExample object
        """
        id = np.searchsorted(self.start_example_per_id,idx,side='right')-1
        idx1 = idx-self.start_example_per_id[id]
        if lastonly:
            return self.data[id].copy_subindex(ts=[idx1+self.contextl-2,idx1+self.contextl-1])
        else:
            return self.data[id].copy_subindex(ts=np.arange(idx1,idx1+self.contextl))
    
    def __getitem__(self, idx: int):
        id = np.searchsorted(self.start_example_per_id,idx,side='right')-1
        idx1 = idx-self.start_example_per_id[id]
        res = self.data[id].get_train_example(ts=np.arange(idx1,idx1+self.contextl),needinit=self.need_init,
                                              needlabels=self.need_labels,needmetadata=self.need_metadata,makecopy=self.make_copy)
        res['idx'] = idx
        return res

    def create_data_from_pred(self,all_pred,labelidx):
        """
        pred_data, true_data = self.create_data_from_pred(all_pred,labelidx)
        Inputs:
        all_pred: predictions. If a dict, then each key is a prediction type and each value is a numpy array of predictions of
        size (pre_sz x) npred x doutput. If a numpy array, then it is of size npred x doutput.
        labelidx: ndarray indices of the labels corresponding to predictions, of size npred.
        Returns:
        pred_data: list of FlyExample objects with the predictions, one example per id
        true_data: list of FlyExample objects with the true labels, one example per id
        """
        if type(labelidx) is torch.Tensor:
            labelidx = labelidx.cpu().numpy()

        labelids = np.searchsorted(self.start_example_per_id,labelidx,side='right')-1
        labelts = labelidx - self.start_example_per_id[labelids] + self.contextl-1
        unique_ids = np.unique(labelids)
        pred_data = [None,]*(np.max(unique_ids)+1)
        true_data = [None,]*(np.max(unique_ids)+1)


        for id in unique_ids:
            idxcurr = np.nonzero(labelids == id)[0]
            tscurr = labelts[idxcurr]
            mint = np.min(tscurr)
            maxt = np.max(tscurr)

            true_example = self.data[id].copy_subindex(ts=np.arange(mint,maxt+1))
            true_data[id] = true_example
            pred_example = true_example.copy()
            
            pred_example.labels.erase_labels()
            if isinstance(all_pred,dict):
                curr_pred = {k:v[idxcurr] for k,v in all_pred.items()}
            else:
                curr_pred = all_pred[idxcurr]
            pred_example.labels.set_prediction(curr_pred,ts=labelts[idxcurr]-mint)
            pred_data[id] = pred_example

        return pred_data, true_data
