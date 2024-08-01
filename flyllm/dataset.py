import numpy as np
from torch.utils.data import Dataset
import copy
import tqdm
import torch

from flyllm.config import featrelative, featglobal, featorigin, feattheta, nrelative, nglobal, nfeatures
from flyllm.features import (
    feat2kp,
    relfeatidx_to_cossinidx, relpose_cos_sin_to_angle,
    ravel_label_index, unravel_label_index,
    split_features,
    zscore, unzscore,
    get_sensory_feature_shapes,
)
from flyllm.data import fit_discretize_labels, discretize_labels, weighted_sample, labels_discrete_to_continuous
from flyllm.utils import rotate_2d_points, compute_npad
from flyllm.models import (  # TODO: dataset should not depend on models
    generate_square_full_mask,
    apply_mask,
    unpack_input,
    get_output_and_attention_weights,
    pred_apply_fun
)
from flyllm.pose import FlyExample, PoseLabels, ObservationInputs


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
        self.dtype = data[0]['input'].dtype
        # number of outputs
        self.d_output = data[0]['labels'].shape[-1]
        self.d_output_continuous = self.d_output
        self.d_output_discrete = 0
        self.d_input = data[0]['input'].shape[-1]

        # number of inputs
        self.dfeat = data[0]['input'].shape[-1]

        self.max_mask_length = max_mask_length
        self.pmask = pmask
        self.masktype = masktype
        self.pdropout_past = pdropout_past
        self.simplify_out = simplify_out  # modulation of task to make it easier
        self.simplify_in = simplify_in
        if maskflag is None:
            maskflag = (masktype is not None) or (pdropout_past > 0.)
        self.maskflag = maskflag

        # TODO REMOVE THESE
        # features used for representing relative pose
        if compute_pose_vel:
            self.nrelrep = nrelative
            self.featrelative = featrelative.copy()
        else:
            self.relfeat_to_cossin_map, self.nrelrep = relfeatidx_to_cossinidx(discreteidx)
            self.featrelative = np.zeros(nglobal + self.nrelrep, dtype=bool)
            self.featrelative[nglobal:] = True

        self.discretefeat = discreteidx

        self.dct_m = None
        self.idct_m = None
        if dct_ms is not None:
            self.dct_m = dct_ms[0]
            self.idct_m = dct_ms[1]
        self.tspred_global = tspred_global

        # TODO REMOVE THESE
        # indices of labels corresponding to the next frame if multiple frames are predicted
        tnext = np.min(self.tspred_global)
        self.nextframeidx_global = self.ravel_label_index([(f, tnext) for f in featglobal])
        if self.simplify_out is None:
            self.nextframeidx_relative = self.ravel_label_index([(i, 1) for i in np.nonzero(self.featrelative)[0]])
        else:
            self.nextframeidx_relative = np.array([])
        self.nextframeidx = np.r_[self.nextframeidx_global, self.nextframeidx_relative]
        if self.dct_m is not None:
            dct_tau = self.dct_m.shape[0]
            # not sure if t+1 should be t+2 here -- didn't add 1 when updating code to make t = 1 mean next frame for relative features
            self.idxdct_relative = np.stack(
                [self.ravel_label_index([(i, t + 1) for i in np.nonzero(self.featrelative)[0]]) for t in
                 range(dct_tau)])
        self.d_output_nextframe = len(self.nextframeidx)

        # whether to predict relative pose velocities (true) or position (false)
        self.compute_pose_vel = compute_pose_vel

        if input_labels:
            assert (masktype is None)
            assert (pdropout_past == 0.)

        self.input_labels = input_labels
        # TODO REMOVE THESE
        if self.input_labels:
            self.d_input_labels = self.d_output_nextframe
        else:
            self.d_input_labels = 0

        # which outputs to discretize, which to keep continuous
        # TODO REMOVE THESE
        self.discreteidx = np.array([])

        self.discrete_tspred = np.array([1, ])
        self.discretize = False

        # TODO REMOVE THESE
        self.continuous_idx = np.arange(self.d_output)

        self.discretize_nbins = None
        self.discretize_bin_samples = None
        self.discretize_bin_edges = None
        self.discretize_bin_means = None
        self.discretize_bin_medians = None

        self.mu_input = None
        self.sig_input = None
        self.mu_labels = None
        self.sig_labels = None

        self.dtype = np.float32

        # TODO REMOTE IDX
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
        if dozscore:
            print('Z-scoring data...')
            data = self.zscore(data, **zscore_params)
            print('Done.')

        if discreteidx is not None:
            print('Discretizing labels...')
            data = self.discretize_labels(data, discreteidx, discrete_tspred, nbins=discretize_nbins,
                                          bin_epsilon=discretize_epsilon, **discretize_params)
            print('Done.')

        self.set_flatten_params(flatten_labels=flatten_labels, flatten_obs_idx=flatten_obs_idx,
                                flatten_do_separate_inputs=flatten_do_separate_inputs)

        # store examples in objects
        self.data = []
        for example_in in data:
            self.data.append(FlyExample(example_in, dataset=self))

        self.set_train_mode()

    @property
    def ntimepoints(self):
        # number of time points
        n = self.data[0].ntimepoints
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
        l = self.ntimepoints * self.ntokens_per_timepoint
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

    # TODO REMOVE THESE
    def ravel_label_index(self, ftidx):

        idx = ravel_label_index(ftidx, dct_m=self.dct_m, tspred_global=self.tspred_global, nrelrep=self.nrelrep)
        return idx

    # TODO REMOVE THESE
    def unravel_label_index(self, idx):

        ftidx = unravel_label_index(idx, dct_m=self.dct_m, tspred_global=self.tspred_global, nrelrep=self.nrelrep)
        return ftidx

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

        # translate to multi representation
        dummyexample = FlyExample(dataset=self)
        discreteidx_next = discreteidx
        bin_epsilon = np.zeros(dummyexample.labels.d_multi)
        bin_epsilon[:] = np.nan
        for i, i_next in enumerate(discreteidx_next):
            idx_multi_curr, _ = dummyexample.labels.convert_idx_next_to_multi_anyt(i_next)
            idx_multi_curr = idx_multi_curr[np.isin(idx_multi_curr, dummyexample.labels.idx_multidiscrete_to_multi)]
            bin_epsilon[idx_multi_curr] = bin_epsilon_feat[i]

        self.discreteidx = np.nonzero(np.isnan(bin_epsilon) == False)[0]
        self.bin_epsilon = bin_epsilon[self.discreteidx]

        self.discretize_nbins = nbins
        self.continuous_idx = np.ones(self.d_output, dtype=bool)
        self.continuous_idx[self.discreteidx] = False
        self.continuous_idx = np.nonzero(self.continuous_idx)[0]
        self.d_output_continuous = len(self.continuous_idx)
        self.d_output_discrete = len(self.discreteidx)

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

        for example in data:
            example['labels_todiscretize'] = example['labels'][:, self.discreteidx]
            example['labels_discrete'] = discretize_labels(example['labels_todiscretize'], self.discretize_bin_edges,
                                                           soften_to_ends=True)
            example['labels'] = example['labels'][:, self.continuous_idx]

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

    def remove_labels_from_input(self, input):
        if self.hasmaskflag():
            return input[..., self.d_input_labels:-1]
        else:
            return input[..., self.d_input_labels:]

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
            example['input'] = self.zscore_input(example['input'])
            example['labels'] = self.zscore_labels(example['labels'])

        return data

    def get_poselabel_params(self):
        return self.data[0].labels.get_params()

    def get_flyexample_params(self):
        return self.data[0].get_params()

    def get_zscore_params(self):

        zscore_params = {
            'mu_input': self.mu_input.copy(),
            'sig_input': self.sig_input.copy(),
            'mu_labels': self.mu_labels.copy(),
            'sig_labels': self.sig_labels.copy(),
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

    def zscore_input(self, rawinput):
        if self.mu_input is None:
            input = rawinput.copy()
        else:
            input = (rawinput - self.mu_input) / self.sig_input
        return input.astype(self.dtype)

    def zscore_nextframe_labels(self, rawlabels):
        if self.mu_labels is None:
            labels = rawlabels.copy()
        else:
            # if rawlabels.shape[-1] > self.d_output_continuous:
            #   labels = rawlabels.copy()
            #   labels[...,self.continuous_idx] = (rawlabels[...,self.continuous_idx]-self.mu_labels)/self.sig_labels
            # else:
            labels = (rawlabels - self.mu_labels[self.nextframeidx]) / self.sig_labels[self.nextframeidx]
        return labels.astype(self.dtype)

    def zscore_labels(self, rawlabels):
        if self.mu_labels is None:
            labels = rawlabels.copy()
        else:
            # if rawlabels.shape[-1] > self.d_output_continuous:
            #   labels = rawlabels.copy()
            #   labels[...,self.continuous_idx] = (rawlabels[...,self.continuous_idx]-self.mu_labels)/self.sig_labels
            # else:
            labels = (rawlabels - self.mu_labels) / self.sig_labels
        return labels.astype(self.dtype)

    def unzscore_nextframe_labels(self, zlabels):
        if self.mu_labels is None:
            rawlabels = zlabels.copy()
        else:
            # if zlabels.shape[-1] > self.d_output_continuous:
            #   rawlabels = zlabels.copy()
            #   rawlabels[...,self.continuous_idx] = unzscore(zlabels[...,self.continuous_idx],self.mu_labels,self.sig_labels)
            # else:
            rawlabels = unzscore(zlabels, self.mu_labels[self.nextframeidx], self.sig_labels[self.nextframeidx])
        return rawlabels.astype(self.dtype)

    def unzscore_labels(self, zlabels, featidx=None):
        if self.mu_labels is None:
            rawlabels = zlabels.copy()
        else:
            # if zlabels.shape[-1] > self.d_output_continuous:
            #   rawlabels = zlabels.copy()
            #   rawlabels[...,self.continuous_idx] = unzscore(zlabels[...,self.continuous_idx],self.mu_labels,self.sig_labels)
            # else:
            if featidx is None:
                rawlabels = unzscore(zlabels, self.mu_labels, self.sig_labels)
            else:
                rawlabels = unzscore(zlabels, self.mu_labels[featidx], self.sig_labels[featidx])
        return rawlabels.astype(self.dtype)

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

    def get_input_shapes(self):
        idx, sz = get_sensory_feature_shapes(self.simplify_in)
        if self.input_labels:
            for k, v in idx.items():
                idx[k] = [x + self.d_input_labels for x in v]
            idx['labels'] = [0, self.d_input_labels]
            sz['labels'] = (self.d_input_labels,)
        return idx, sz

    def unpack_input(self, input, dim=-1):

        idx, sz = self.get_input_shapes()
        res = unpack_input(input, idx, sz, dim=dim)

        return res

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

        """

        res = self.data[idx].get_train_example()
        res['input'], mask, dropout_mask = self.mask_input(res['input'])
        if mask is not None:
            res['mask'] = mask
        if dropout_mask is not None:
            res['dropout_mask'] = dropout_mask

        return res

        # TODO REMOVE THIS AFTER CHECKING FLATTENING
        datacurr = copy.deepcopy(self.data[idx])

        if self.input_labels:
            # should we use all future predictions, or just the next time point?
            input_labels = datacurr.labels.get_input_labels()
        else:
            input_labels = None

        # add_noise
        # to do: make this work with objects
        if self.do_add_noise:
            eta, datacurr = self.add_noise(datacurr, input_labels)
        if self.input_labels:
            input_labels = torch.tensor(input_labels)
        labels = datacurr.get_labels()

        # whether we start with predicting the 0th or the 1th frame in the input sequence
        starttoff = self.get_start_toff()

        init = torch.tensor(labels.get_init_pose(starttoff))
        scale = torch.tensor(labels.get_scale())
        categories = torch.tensor(labels.get_categories())
        metadata = datacurr.get_metadata(makecopy=True)
        metadata['t0'] += starttoff
        metadata['frame0'] += starttoff

        raw_labels = labels.get_raw_labels_tensor_copy(format='input')
        res = {'input': None, 'labels': None, 'labels_discrete': None,
               'labels_todiscretize': None,
               'init': init, 'scale': scale, 'categories': categories,
               'metadata': metadata}

        res['labels'] = raw_labels['labels'][starttoff:, :]
        if self.discretize:
            res['labels_discrete'] = raw_labels['labels_discrete'][starttoff:, :, :]
            res['labels_todiscretize'] = raw_labels['labels_todiscretize'][starttoff:, :]

        input = torch.tensor(datacurr.get_inputs().get_raw_inputs())
        nin = input.shape[-1]
        contextl = input.shape[0]
        input, mask, dropout_mask = self.mask_input(input)

        if self.flatten:
            ntypes = self.ntokens_per_timepoint
            # newl = contextl*ntypes
            newlabels = torch.zeros((contextl, ntypes, self.flatten_max_doutput), dtype=input.dtype)
            newinput = torch.zeros((contextl, ntypes, self.flatten_dinput), dtype=input.dtype)
            newmask = torch.zeros((contextl, ntypes), dtype=bool)
            # offidx = np.arange(contextl)*ntypes
            if self.flatten_obs:
                for i, v in enumerate(self.flatten_obs_idx.values()):
                    newinput[:, i,
                    self.flatten_input_type_to_range[i, 0]:self.flatten_input_type_to_range[i, 1]] = input[:, v[0]:v[1]]
                    newmask[:, i] = False
            else:
                newinput[:, 0, :self.flatten_dinput_pertype[0]] = input
            if self.discretize:
                if self.flatten_labels:
                    for i in range(self.d_output_discrete):
                        inputnum = self.flatten_nobs_types + i
                        newlabels[:, inputnum, :self.discretize_nbins] = raw_labels['labels_discrete'][:, i, :]
                        newinput[:, inputnum,
                        self.flatten_input_type_to_range[inputnum, 0]:self.flatten_input_type_to_range[inputnum, 1]] = \
                        raw_labels['labels_discrete'][:, i, :]
                        if mask is None:
                            newmask[:, self.flatten_nobs_types + i] = True
                        else:
                            newmask[:, self.flatten_nobs_types + i] = mask.clone()
                    if self.continuous:
                        inputnum = -1
                        newlabels[:, -1, :labels.shape[-1]] = raw_labels['labels']
                        newinput[:, -1,
                        self.flatten_input_type_to_range[inputnum, 0]:self.flatten_input_type_to_range[inputnum, 1]] = \
                        raw_labels['labels']
                        if mask is None:
                            newmask[:, -1] = True
                        else:
                            newmask[:, -1] = mask.clone()
                else:
                    newinput[:, -1, :self.d_output] = raw_labels['labels']
            newlabels = newlabels.reshape((contextl * ntypes, self.flatten_max_doutput))
            newinput = newinput.reshape((contextl * ntypes, self.flatten_dinput))
            newmask = newmask.reshape(contextl * ntypes)
            if not self.ismasked():
                newlabels = newlabels[1:, :]
                newinput = newinput[:-1, :]
                newmask = newmask[1:]

            res['input'] = newinput
            res['input_stacked'] = input
            res['mask_flattened'] = newmask
            res['labels'] = newlabels
            res['labels_stacked'] = labels
        else:
            if self.input_labels:
                input = torch.cat((input_labels[:-starttoff, :], input[starttoff:, :]), dim=-1)
            res['input'] = input

        if mask is not None:
            res['mask'] = mask
        if dropout_mask is not None:
            res['dropout_mask'] = dropout_mask
        return res

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

    # TODO REMOVE ALL OF THESE
    def get_global_movement_idx(self):
        idxglobal = self.ravel_label_index(np.stack(np.meshgrid(featglobal, self.tspred_global), axis=-1))
        return idxglobal

    def get_global_movement(self, movement):
        idxglobal = self.get_global_movement_idx()
        movement_global = movement[..., idxglobal]
        return movement_global

    def set_global_movement(self, movement_global, movement):
        idxglobal = self.get_global_movement_idx()
        movement[..., idxglobal] = movement_global
        return movement

    def get_global_movement_discrete(self, movement_discrete):
        if not self.discretize:
            return None
        idxglobal = self.get_global_movement_idx()
        movement_global_discrete = np.zeros(
            movement_discrete.shape[:-2] + idxglobal.shape + movement_discrete.shape[-1:], dtype=self.dtype)
        movement_global_discrete[:] = np.nan
        for i in range(idxglobal.shape[0]):
            for j in range(idxglobal.shape[1]):
                idx = idxglobal[i, j]
                didx = np.nonzero(self.discreteidx == idx)[0]
                if len(didx) == 0:
                    continue
                movement_global_discrete[..., i, j, :] = movement_discrete[..., didx[0], :]
        return movement_global_discrete

    def get_next_relative_movement(self, movement):
        movement_next_relative = movement[..., self.nextframeidx_relative]
        return movement_next_relative

    def get_relative_movement_dct(self, movements, iszscored=False):
        movements_dct = movements[..., self.idxdct_relative]
        if not iszscored and self.mu_labels is not None:
            movements_dct = unzscore(movements_dct, self.mu_labels[self.idxdct_relative],
                                     self.sig_labels[self.idxdct_relative])
        movements_relative = self.idct_m @ movements_dct
        return movements_relative

    def get_next_relative_movement_dct(self, movements, iszscored=True, dozscore=True):
        if self.simplify_out == 'global':
            return movements[..., []]

        if type(movements) is np.ndarray:
            movements = torch.as_tensor(movements)

        movements_dct = movements[..., self.idxdct_relative]
        if not iszscored and self.mu_labels is not None:
            mu = torch.as_tensor(self.mu_labels[self.idxdct_relative]).to(dtype=movements.dtype,
                                                                          device=movements.device)
            sig = torch.as_tensor(self.sig_labels[self.idxdct_relative]).to(dtype=movements.dtype,
                                                                            device=movements.device)
            movements_dct = unzscore(movements_dct, mu, sig)

        idct_m0 = torch.as_tensor(self.idct_m[[0, ], :]).to(dtype=movements.dtype, device=movements.device)
        dctfeat = movements[..., self.idxdct_relative]
        movements_next_relative = torch.matmult(idct_m0, dctfeat)

        if dozscore:
            movements_next_relative = zscore(movements_next_relative, self.mu_labels[self.nextframeidx_relative],
                                             self.sig_labels[self.nextframeidx_relative])

        return movements_next_relative

    def compare_dct_to_next_relative(self, movements):
        movements_next_relative_dct = self.get_next_relative_movement_dct(movements, iszscored=True, dozscore=True)
        movements_next_relative0 = movements[..., self.nextframeidx_relative]
        err = movements_next_relative_dct - movements_next_relative0
        return err

    def get_next_movements(self, movements=None, example=None, iszscored=False, use_dct=False, **kwargs):
        """
        get_next_movements(movements=None,example=None,iszscored=False,use_dct=False,**kwargs)
        extracts the next frame movements/pose from the input, ignoring predictions for frames further
        into the future.
        Inputs:
        movements: ... x d_output ndarray of movements. Required if example is None. Default: None.
        example: dict holding training/test example. Required if movements is None. Default: None.
        iszscored: whether movements are z-scored. Default: False.
        use_dct: whether to use DCT to extract relative pose features. Default: False.
        Extra args are fed into get_full_labels if movements is None
        Outputs:
        movements_next: ... x d_output ndarray of movements/pose for the next frame.
        """
        if movements is None:
            movements = self.get_full_labels(example=example, **kwargs)
            iszscored = True

        if torch.is_tensor(movements):
            movements = movements.numpy()

        if iszscored and self.mu_labels is not None:
            movements = unzscore(movements, self.mu_labels, self.sig_labels)

        movements_next_global = movements[..., self.nextframeidx_global]
        if self.simplify_out is None:
            if use_dct and self.dct_m is not None:
                dctfeat = movements[..., self.idxdct_relative]
                movements_next_relative = self.idct_m[[0, ], :] @ dctfeat
            else:
                movements_next_relative = movements[..., self.nextframeidx_relative]
            movements_next = np.concatenate((movements_next_global, movements_next_relative), axis=-1)
        else:
            movements_next = movements_next_global
        return movements_next

    def get_init_pose(self, example=None, input0=None, global0=None, zscored=False):
        if example is not None:
            if input0 is None:
                input = self.get_full_inputs(example=example)
                input0 = input[..., 0, :]
            if global0 is None:
                global0 = example['init']

        istorch = torch.is_tensor(input0)

        if (self.mu_input is not None) and (zscored == False):
            input0 = unzscore(input0, self.mu_input, self.sig_input)

        input0 = split_features(input0, simplify=self.simplify_in)
        relative0 = input0['pose']

        if istorch:
            pose0 = torch.zeros(nfeatures, dtype=relative0.dtype, device=relative0.device)
        else:
            pose0 = np.zeros(nfeatures, dtype=relative0.dtype)
        pose0[featglobal] = global0
        pose0[featrelative] = relative0

        return pose0

    def get_Xfeat(self, input0=None, global0=None, movements=None, example=None, use_dct=False, **kwargs):
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
                input0 = input[..., 0, :]
            if global0 is None:
                global0 = example['init']
            if movements is None:
                movements = self.get_full_labels(example=example, **kwargs)

        szrest = movements.shape[:-1]
        n = np.prod(np.array(szrest))

        if torch.is_tensor(input0):
            input0 = input0.numpy()
        if torch.is_tensor(global0):
            global0 = global0.numpy()
        if torch.is_tensor(movements):
            movements = movements.numpy()

        if self.mu_input is not None:
            input0 = unzscore(input0, self.mu_input, self.sig_input)
            movements = unzscore(movements, self.mu_labels, self.sig_labels)

        # get movements/pose for next frame prediction
        movements_next = self.get_next_movements(movements=movements, iszscored=False, use_dct=use_dct)

        if not self.compute_pose_vel:
            movements_next = self.convert_cos_sin_to_angle(movements_next)

        input0 = split_features(input0, simplify=self.simplify_in)
        Xorigin0 = global0[..., :2]
        Xtheta0 = global0[..., 2]
        thetavel = movements_next[..., feattheta]

        Xtheta = np.cumsum(np.concatenate((Xtheta0[..., None], thetavel), axis=-1), axis=-1)
        Xoriginvelrel = movements_next[..., [featorigin[1], featorigin[0]]]
        Xoriginvel = rotate_2d_points(Xoriginvelrel.reshape((n, 2)), -Xtheta[..., :-1].reshape(n)).reshape(
            szrest + (2,))
        Xorigin = np.cumsum(np.concatenate((Xorigin0[..., None, :], Xoriginvel), axis=-2), axis=-2)
        Xfeat = np.zeros(szrest[:-1] + (szrest[-1] + 1, nfeatures), dtype=self.dtype)
        Xfeat[..., featorigin] = Xorigin
        Xfeat[..., feattheta] = Xtheta

        if self.simplify_out == 'global':
            Xfeat[..., featrelative] = np.tile(input0['pose'], szrest[:-1] + (szrest[-1] + 1, 1))
        else:
            Xfeatpose = np.concatenate((input0['pose'][..., None, :], movements_next[..., featrelative]), axis=-2)
            if self.compute_pose_vel:
                Xfeatpose = np.cumsum(Xfeatpose, axis=-2)
            Xfeat[..., featrelative] = Xfeatpose

        return Xfeat

    def get_Xkp(self, example, pred=None, **kwargs):
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
            movements = self.get_full_pred(pred, **kwargs)
        else:
            movements = None
        Xfeat = self.get_Xfeat(example=example, movements=movements, **kwargs)
        Xkp = self.feat2kp(Xfeat, scale)
        return Xkp

    def get_Xkp0(self, input0=None, global0=None, movements=None, scale=None, example=None):
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

        Xfeat = self.get_Xfeat(input0=input0, global0=global0, movements=movements, example=example)
        Xkp = self.feat2kp(Xfeat, scale)
        return Xkp

    def feat2kp(self, Xfeat, scale):
        """
        Xkp = self.feat2kp(Xfeat)

        Args:
            Xfeat (ndarray, T x nfeatures): full pose features for each frame
            scale (ndarray, dscale): scale features

        Returns:
            Xkp (ndarray, nkeypoints x 2 x T+1 x 1): keypoints for each frame
        """
        Xkp = feat2kp(Xfeat.T[..., None], scale[..., None])
        return Xkp

    def construct_input(self, obs, movement=None):

        # to do: merge this code with getitem so that we don't have to duplicate
        dtype = obs.dtype

        if self.input_labels:
            assert (movement is not None)

        if self.flatten:
            xcurr = np.zeros((obs.shape[0], self.ntokens_per_timepoint, self.flatten_dinput), dtype=dtype)

            if self.flatten_obs:
                for i, v in enumerate(self.flatten_obs_idx.values()):
                    xcurr[:, i, self.flatten_input_type_to_range[i, 0]:self.flatten_input_type_to_range[i, 1]] = obs[:,
                                                                                                                 v[0]:v[
                                                                                                                     1]]
            else:
                xcurr[:, 0, :self.flatten_dinput_pertype[0]] = obs

            if self.input_labels:
                # movement not set for last time points, will be 0s
                if self.flatten_labels:
                    for i in range(movement.shape[1]):
                        if i < len(self.discreteidx):
                            dmovement = self.discretize_nbins
                        else:
                            dmovement = len(self.continuous_idx)
                        inputnum = self.flatten_nobs_types + i
                        xcurr[:-1, inputnum,
                        self.flatten_input_type_to_range[inputnum, 0]:self.flatten_input_type_to_range[
                            inputnum, 1]] = movement[:, i, :dmovement]
                else:
                    inputnum = self.flatten_nobs_types
                    xcurr[:-1, inputnum, self.flatten_input_type_to_range[inputnum, 0]:self.flatten_input_type_to_range[
                        inputnum, 1]] = movement
            xcurr = np.reshape(xcurr, (xcurr.shape[0] * xcurr.shape[1], xcurr.shape[2]))

        else:
            if self.input_labels:
                xcurr = np.concatenate((movement, obs[1:, ...]), axis=-1)
            else:
                xcurr = obs

        return xcurr

    def get_movement_npad(self):
        npad = compute_npad(self.tspred_global, self.dct_m)
        return npad

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

    def get_movement_names_global(self):
        return self.data[0].labels.get_nextglobal_names()

    def get_movement_names(self):
        return self.data[0].labels.get_nextcossin_names()

    def get_outnames(self):
        """
        outnames = self.get_outnames()

        Returns:
            outnames (list of strings): names of each output motion
        """
        return self.data[0].labels.get_multi_names()

    # TODO REMOVE THIS
    def parse_label_fields(self, example):

        labels_discrete = None
        labels_todiscretize = None
        labels_stacked = None

        # get labels_continuous, labels_discrete from example
        if isinstance(example, dict):
            if 'labels' in example:
                labels_continuous = example['labels']
            elif 'continuous' in example:
                labels_continuous = example['continuous']  # prediction
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
            labels_continuous, labels_discrete = self.unflatten_labels(labels_continuous)

        return labels_continuous, labels_discrete, labels_todiscretize, labels_stacked

    def unflatten_labels(self, labels_flattened):
        assert self.flatten_labels
        sz = labels_flattened.shape
        newsz = sz[:-2] + (self.ntimepoints, self.ntokens_per_timepoint, self.flatten_max_doutput)
        if not self.ismasked():
            pad = torch.zeros(sz[:-2] + (1, self.flatten_max_doutput), dtype=labels_flattened.dtype,
                              device=labels_flattened.device)
            labels_flattened = torch.cat((pad, labels_flattened), dim=-2)
        labels_flattened = labels_flattened.reshape(newsz)
        if self.d_output_continuous > 0:
            labels_continuous = labels_flattened[..., -1, :self.d_output_continuous]
        else:
            labels_continuous = None
        if self.discretize:
            labels_discrete = labels_flattened[..., self.flatten_nobs_types:, :self.discretize_nbins]
            if self.continuous:
                labels_discrete = labels_discrete[..., :-1, :]
        else:
            labels_discrete = None
        return labels_continuous, labels_discrete

    def apply_flatten_input(self, input):

        if type(input) == np.ndarray:
            input = torch.Tensor(input)

        if self.flatten_obs == False:
            return input

        # input is of size ...,contextl,d_input
        sz = input.shape[:-2]
        contextl = input.shape[-2]
        newinput = torch.zeros(sz + (contextl, self.flatten_nobs_types, self.flatten_max_dinput), dtype=input.dtype)

        for i, v in enumerate(self.flatten_obs_idx.values()):
            newinput[..., i, :self.flatten_dinput_pertype[i]] = input[..., v[0]:v[1]]
        return newinput

    def unflatten_input(self, input_flattened):
        assert self.flatten_obs
        sz = input_flattened.shape
        if not self.ismasked():
            pad = torch.zeros(sz[:-2] + (1, self.flatten_dinput), dtype=input_flattened.dtype,
                              device=input_flattened.device)
            input_flattened = torch.cat((input_flattened, pad), dim=-2)
        resz = sz[:-2] + (self.ntimepoints, self.ntokens_per_timepoint, self.flatten_dinput)
        input_flattened = input_flattened.reshape(resz)
        newsz = sz[:-2] + (self.ntimepoints, self.dfeat)
        newinput = torch.zeros(newsz, dtype=input_flattened.dtype)
        for i, v in enumerate(self.flatten_obs_idx.values()):
            newinput[..., :, v[0]:v[1]] = input_flattened[..., i,
                                          self.flatten_input_type_to_range[i, 0]:self.flatten_input_type_to_range[i, 1]]
        return newinput

    def get_full_inputs(self, example=None, idx=None, use_stacked=False):
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

    def get_continuous_discrete_labels(self, example):

        # get labels_continuous, labels_discrete from example
        labels_continuous, labels_discrete, _, _ = self.parse_label_fields(example)
        return labels_continuous, labels_discrete

    def get_continuous_labels(self, example):

        labels_continuous, _ = self.get_continuous_discrete_labels(example)
        return labels_continuous

    def get_discrete_labels(self, example):
        _, labels_discrete = self.get_continuous_discrete_labels(example)

        return labels_discrete

    def get_full_pred(self, pred, **kwargs):
        return self.get_full_labels(example=pred, ispred=True, **kwargs)

    def convert_cos_sin_to_angle(self, movements_in):
        # relpose_cos_sin = WORKING HERE
        if self.compute_pose_vel:
            return movements_in.copy()
        relpose_cos_sin = movements_in[..., -self.nrelrep:]
        relpose_angle = relpose_cos_sin_to_angle(relpose_cos_sin, discreteidx=self.discretefeat)
        return np.concatenate((movements_in[..., :-self.nrelrep], relpose_angle), axis=-1)

    def get_full_labels(self, example=None, idx=None, use_todiscretize=False, sample=False, use_stacked=False,
                        ispred=False, nsamples=0):

        if self.discretize and sample:
            return self.sample_full_labels(example=example, idx=idx, nsamples=nsamples)

        if example is None:
            example = self[idx]

        # get labels_continuous, labels_discrete from example
        labels_continuous, labels_discrete, labels_todiscretize, labels_stacked = \
            self.parse_label_fields(example)

        if self.flatten_labels:
            if use_stacked and labels_stacked is not None:
                labels_continuous, labels_discrete = self.unflatten_labels(labels_stacked)

        if self.discretize:
            # should be ... x d_output_discrete x discretize_nbins
            sz = labels_discrete.shape
            newsz = sz[:-2] + (self.d_output,)
            labels = torch.zeros(newsz, dtype=labels_discrete.dtype)
            if self.d_output_continuous > 0:
                labels[..., self.continuous_idx] = labels_continuous
            if use_todiscretize and (labels_todiscretize is not None):
                labels[..., self.discreteidx] = labels_todiscretize
            else:
                labels[..., self.discreteidx] = labels_discrete_to_continuous(labels_discrete,
                                                                              torch.tensor(self.discretize_bin_edges))
        else:
            labels = labels_continuous.clone()

        return labels

    def sample_full_labels(self, example=None, idx=None, nsamples=0):
        if example is None:
            example = self[idx]

        nsamples1 = nsamples
        if nsamples1 == 0:
            nsamples1 = 1

        # get labels_continuous, labels_discrete from example
        labels_continuous, labels_discrete, _, _ = self.parse_label_fields(example)

        if not self.discretize:
            return labels_continuous

        # should be ... x d_output_continuous
        sz = labels_discrete.shape[:-2]
        dtype = labels_discrete.dtype
        newsz = (nsamples1,) + sz + (self.d_output,)
        labels = torch.zeros(newsz, dtype=dtype)
        if self.continuous:
            labels[..., self.continuous_idx] = labels_continuous

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
            binnum = weighted_sample(labels_discrete[..., i, :].reshape((n, nbins)), nsamples=nsamples)
            sample = torch.randint(low=0, high=nsamples_per_bin, size=(nsamples, n))
            labelscurr = torch.Tensor(self.discretize_bin_samples[sample, i, binnum].reshape((nsamples,) + szrest))
            labels[..., self.discreteidx[i]] = labelscurr

        if nsamples == 0:
            labels = labels[0]

        return labels
