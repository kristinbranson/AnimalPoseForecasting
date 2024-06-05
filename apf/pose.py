import numpy as np
import torch

from apf.config import featglobal, featrelative, featangle
from apf.features import relfeatidx_to_cossinidx, ravel_label_index, unravel_label_index, zscore, unzscore, feat2kp
from apf.data import weighted_sample
from apf.utils import modrange


class PoseLabels:
    def __init__(self, example_in=None, init_pose=None, simplify_out=None,
                 discrete_idx=[], tspred_global=[1, ],
                 ntspred_relative=1,
                 zscore_params=None, discretize_params=None,
                 is_velocity=True, flatten_labels=False):

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
        # full_labels_discrete_idx: indices of

        self.labels_raw = {}
        self.simplify_out = simplify_out
        self.tspred_global = np.array(tspred_global)
        self.ntspred_relative = ntspred_relative
        self.zscore_params = zscore_params
        self.discretize_params = discretize_params
        self.is_velocity = is_velocity
        self.flatten_labels = flatten_labels
        self.init_pose = init_pose

        # copy over labels_in
        self.label_keys = {}
        self.labels_raw = {}
        self.pre_sz = None
        self.metadata = None
        self.categories = None
        if example_in is not None:
            if 'labels' in example_in:
                self.labels_raw['continuous'] = example_in['labels']
                self.label_keys['continuous'] = 'labels'
            elif 'continuous' in example_in:
                self.labels_raw['continuous'] = example_in['continuous']
                self.label_keys['continuous'] = 'continuous'
            else:
                raise ValueError('labels_in must contain labels or continuous')
            if 'labels_discrete' in example_in:
                self.labels_raw['discrete'] = example_in['labels_discrete']
                self.label_keys['discrete'] = 'labels_discrete'
            elif 'discrete' in example_in:
                self.labels_raw['discrete'] = example_in['discrete']
                self.label_keys['discrete'] = 'discrete'
            if 'labels_todiscretize' in example_in:
                self.labels_raw['todiscretize'] = example_in['labels_todiscretize']
                self.label_keys['todiscretize'] = 'labels_todiscretize'
            elif 'todiscretize' in example_in:
                self.labels_raw['todiscretize'] = example_in['todiscretize']
                self.label_keys['todiscretize'] = 'todiscretize'
            if 'labels_stacked' in example_in:
                self.labels_raw['stacked'] = example_in['labels_stacked']
                self.label_keys['stacked'] = 'labels_stacked'
            elif 'stacked' in example_in:
                self.labels_raw['stacked'] = example_in['stacked']
                self.label_keys['stacked'] = 'stacked'
            self.scale = example_in['scale']
            if 'metadata' in example_in:
                self.metadata = example_in['metadata']
            if 'categories' in example_in:
                self.categories = example_in['categories']
            self.pre_sz = self.labels_raw['continuous'].shape[:-1]

        self.set_indices(discrete_idx)

        self.init_pose = init_pose

        return

    def set_indices(self, discrete_idx):

        self.discrete_idx_in = discrete_idx

        # which indices of pose (next frame, global + relative) are global
        self.idx_nextglobal_to_next = np.array(featglobal)
        self.d_next_global = len(self.idx_nextglobal_to_next)

        # which indices of pose (next frame, global + relative) are relative
        if self.simplify_out is None:
            self.idx_nextrelative_to_next = np.nonzero(featrelative)[0]
        else:
            self.idx_nextrelative_to_next = np.array([])
        self.d_next_relative = len(self.idx_nextrelative_to_next)

        self.d_next = self.d_next_global + self.d_next_relative

        # which indices are angles
        self.is_angle_next = featangle

        # which indices of pose (next frame, global + relative) are discrete
        self.idx_nextdiscrete_to_next = np.array(discrete_idx)

        # which indices of pose (next frame, global + relative) are continuous
        self.idx_nextcontinuous_to_next = np.setdiff1d(np.arange(self.d_next), self.idx_nextdiscrete_to_next)

        # we will use a cosine/sine representation for relative pose
        # next_cossin is equivalent to next if velocity is used
        self.idx_next_to_nextcossin = list(range(self.d_next))
        self.idx_nextcossinglobal_to_nextcossin = np.arange(self.d_next_global)
        self.idx_nextglobal_to_nextcossinglobal = np.arange(self.d_next_global)
        self.d_next_cossin_global = len(self.idx_nextcossinglobal_to_nextcossin)
        if self.is_velocity:
            self.idx_nextrelative_to_nextcossinrelative = list(range(self.d_next_relative))
            self.d_next_cossin_relative = self.d_next_relative
        else:
            self.idx_nextrelative_to_nextcossinrelative, self.d_next_cossin_relative = \
                relfeatidx_to_cossinidx(discrete_idx)
        self.d_next_cossin = self.d_next_cossin_relative + self.d_next_cossin_global
        self.idx_nextcossinrelative_to_nextcossin = \
            np.setdiff1d(np.arange(self.d_next_cossin), self.idx_nextcossinglobal_to_nextcossin)
        self.idx_next_to_nextcossin = list(range(self.d_next))
        for inextglobal in range(self.d_next_global):
            inext = self.idx_nextglobal_to_next[inextglobal]
            inextcossinglobal = self.idx_nextglobal_to_nextcossinglobal[inextglobal]
            inextcossin = self.idx_nextcossinglobal_to_nextcossin[inextcossinglobal]
            self.idx_next_to_nextcossin[inext] = inextcossin
        for inextrel in range(self.d_next_relative):
            inext = self.idx_nextrelative_to_next[inextrel]
            inextcossinrelative = self.idx_nextrelative_to_nextcossinrelative[inextrel]
            # if type(inextcossinrelative) is list:
            #   inextcossin = [self.idx_nextcossinrelative_to_nextcossin[x] for x in inextcossinrelative]
            # else:
            inextcossin = self.idx_nextcossinrelative_to_nextcossin[inextcossinrelative]
            self.idx_next_to_nextcossin[inext] = inextcossin

        # which indices of nextcossin are discrete/continuous
        self.idx_nextcossindiscrete_to_nextcossin = []
        for inext in self.idx_nextdiscrete_to_next:
            inextcossin = self.idx_next_to_nextcossin[inext]
            self.idx_nextcossindiscrete_to_nextcossin.append(inextcossin)
        self.idx_nextcossincontinuous_to_nextcossin = []
        for inext in self.idx_nextcontinuous_to_next:
            inextcossin = self.idx_next_to_nextcossin[inext]
            if type(inextcossin) is np.ndarray:
                self.idx_nextcossincontinuous_to_nextcossin.extend(inextcossin.tolist())
            else:
                self.idx_nextcossincontinuous_to_nextcossin.append(inextcossin)

        # which multi correspond to nextcossin
        self.d_multi_relative = self.d_next_cossin_relative * self.ntspred_relative
        self.d_multi_global = self.d_next_cossin_global * len(self.tspred_global)
        self.d_multi = self.d_multi_global + self.d_multi_relative
        assert (np.min(self.tspred_global) == 1)
        self.idx_nextcossin_to_multi = self.feattpred_to_multi([(f, 1) for f in range(self.d_next_cossin)])

        # look up table from multi index to (feat,tpred)
        # d_multi x 2 array
        self.idx_multi_to_multifeattpred = self.multi_to_feattpred(np.arange(self.d_multi))

        # look up table from (feat,tpred) to multi index
        # dict
        self.idx_multifeattpred_to_multi = {}
        for idx, ft in enumerate(self.idx_multi_to_multifeattpred):
            self.idx_multifeattpred_to_multi[tuple(ft.tolist())] = idx

        # which indices of multi correspond to multi_relative and multi_global
        isrelative = np.array(
            [ft[0] in self.idx_nextcossinrelative_to_nextcossin for ft in self.idx_multi_to_multifeattpred])
        self.idx_multirelative_to_multi = np.nonzero(isrelative)[0]
        self.idx_multiglobal_to_multi = np.nonzero(isrelative == False)[0]

        # which indices of multi correspond to multi_discrete, multi_continuous
        isdiscrete = np.array(
            [ft[0] in self.idx_nextcossindiscrete_to_nextcossin for ft in self.idx_multi_to_multifeattpred])
        self.idx_multidiscrete_to_multi = np.nonzero(isdiscrete)[0]
        self.idx_multicontinuous_to_multi = np.nonzero(isdiscrete == False)[0]
        self.d_multidiscrete = len(self.idx_multidiscrete_to_multi)
        self.d_multicontinuous = len(self.idx_multicontinuous_to_multi)

        assert self.d_multicontinuous == self.labels_raw['continuous'].shape[-1]
        if self.is_discretized():
            assert self.d_multidiscrete == self.labels_raw['discrete'].shape[-2]

        # to do: include dct

        # to do: include flattening

        return

    def feattpred_to_multi(self, ftidx):
        idx = ravel_label_index(ftidx, dct_tau=self.ntspred_relative,
                                tspred_global=self.tspred_global, nrelrep=self.d_next_cossin_relative)
        return idx

    def multi_to_feattpred(self, idx):
        ftidx = unravel_label_index(idx, dct_tau=self.ntspred_relative, tspred_global=self.tspred_global,
                                    nrelrep=self.d_next_cossin_relative)
        return ftidx

    def is_zscored(self):
        return self.zscore_params is not None

    def is_discretized(self):
        return self.discretize_params is not None

    def get_labels_raw(self, format='standard'):
        if format == 'standard':
            return self.labels_raw
        # format = 'input'
        labels_out = {}
        for k, v in self.label_keys.items():
            labels_out[v] = self.labels_raw[k]
        return labels_out

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
        labels_discrete = torch.reshape(labels_discrete, (n, nfeat, nbins))

        # nfeat x nbins
        bin_centers = torch.tensor(self.discretize_params['bin_medians']).to(device=labels_discrete.device)
        s = torch.sum(labels_discrete, dim=-1)
        assert torch.max(torch.abs(1 - s)) < epsilon, 'discrete labels do not sum to 1'
        continuous = torch.sum(bin_centers[None, ...] * labels_discrete, dim=-1) / s
        continuous = torch.reshape(continuous, szrest + (nfeat,))
        return continuous

    def sample_discrete_labels(self, labels_discrete, nsamples=1):
        assert self.is_discretized()

        sz = labels_discrete.shape
        nbins = sz[-1]
        nfeat = sz[-2]
        szrest = sz[:-2]
        n = int(np.prod(np.array(szrest)))
        labels_discrete = torch.reshape(labels_discrete, (n, nfeat, nbins))
        bin_samples = self.discretize_params['bin_samples']
        nsamples_per_bin = bin_samples.shape[0]
        continuous = torch.zeros((nsamples,) + szrest + (nfeat,), dtype=labels_discrete.dtype,
                                 device=labels_discrete.device)
        for f in range(nfeat):
            binnum = weighted_sample(labels_discrete[:, f, :], nsamples=nsamples)
            sample = torch.randint(low=0, high=nsamples_per_bin, size=(nsamples, n))
            curr = torch.Tensor(bin_samples[sample, f, binnum].reshape((nsamples,) + szrest))
            continuous[..., f] = curr

        return curr

    def get_multi(self, use_todiscretize=False, nsamples=0, zscored=False, collapse_samples=False):

        labels_raw = self.get_labels_raw(format='standard')

        # to do: add flattening support here

        # allocate multi
        multisz = self.pre_sz + (self.d_multi,)
        if nsamples > 0:
            multisz = (nsamples,) + multisz
        multi = torch.zeros(multisz, dtype=labels_raw['continuous'].dtype,
                            device=labels_raw['continuous'].device)
        multi[:] = torch.nan

        if self.is_discretized():
            if use_todiscretize:
                assert 'todiscretize' in self.labels_raw
                # shape is pre_sz x d_multi_discrete
                labels_discrete = labels_raw['todiscretize']
            elif nsamples > 0:
                # shape is nsamples x pre_sz x d_multi_discrete
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

    def multi_to_nextcossin(self, multi):
        next_cossin = multi[..., self.idx_nextcossin_to_multi]
        return next_cossin

    def get_nextcossin(self, **kwargs):
        multi = self.get_multi(**kwargs)
        return self.multi_to_nextcossin(multi)

    def nextcossin_to_next(self, next_cossin):
        szrest = next_cossin.shape[:-1]
        n = np.prod(szrest)
        next = torch.zeros(n, self.d_next, dtype=next_cossin.dtype, device=next_cossin.device)
        for inext in range(self.d_next):
            inextcossin = self.idx_next_to_nextcossin[inext]
            if type(inextcossin) is np.ndarray:
                next[..., inext] = torch.atan2(next_cossin[..., inextcossin[1]], next_cossin[..., inextcossin[0]])
            else:
                next[..., inext] = next_cossin[..., inextcossin]
        return next

    def get_next(self, **kwargs):
        next_cossin = self.get_nextcossin(**kwargs)
        return self.nextcossin_to_next(next_cossin)

    def next_to_nextpose(self, next):

        szrest = next.shape[:-2]
        n = int(np.prod(szrest))
        T = next.shape[-2]
        next = next.reshape((n, T, self.d_next))
        pose = torch.zeros((n, T + 1, self.d_next), dtype=next.dtype, device=next.device)
        pose[:, 0, :] = self.init_pose
        pose[:, 1:, :] = next
        if self.is_velocity:
            pose = torch.cumsum(pose, dim=1)
        else:
            pose[..., self.idx_nextglobal_to_next] = torch.cumsum(pose[..., self.idx_nextglobal_to_next], dim=1)
        pose[..., self.is_angle_next] = modrange(pose[..., self.is_angle_next], -np.pi, np.pi)

        pose = pose.reshape(szrest + (T + 1, self.d_next))

        return pose

    def next_to_nextvelocity(self, next):

        if self.is_velocity:
            return next

        szrest = next.shape[:-2]
        n = int(np.prod(szrest))
        T = next.shape[-2]
        velrel = torch.zeros((n, T + 1, self.d_next_relative), dtype=next.dtype, device=next.device)
        velrel[:, 0, :] = self.init_pose[self.idx_nextrelative_to_next]
        velrel[:, 1:, :] = next[..., self.idx_nextrelative_to_next].reshape((n, T, self.d_next_relative))
        velrel[:, :-1, :] = torch.diff(velrel, dim=1)
        velrel = velrel[:, :-1, :]
        velrel = velrel.reshape(szrest + (T, self.d_next_relative))
        vel = next.clone()
        vel[..., self.idx_nextrelative_to_next] = velrel

        return vel

    def get_next_pose(self, **kwargs):

        next = self.get_next(**kwargs)
        # global will always be velocity, still need to do an integration
        next_pose = self.next_to_nextpose(next)
        return next_pose

    def nextpose_to_nextkeypoints(self, pose):

        # input to feat2kp is expected to be an np.ndarray with shape nfeatures x T x nflies
        szrest = pose.shape[:-1]
        n = int(np.prod(szrest))
        # todo: make a version of this that works with torch tensors and without transposing
        nppose = pose.reshape((n, self.d_next)).T.numpy(force=True)
        npkp = feat2kp(nppose, self.scale.numpy(force=True))
        npkp = npkp[..., 0].transpose((2, 0, 1))
        kp = torch.tensor(npkp, dtype=pose.dtype, device=pose.device)
        kp = kp.reshape(szrest + kp.shape[-2:])
        return kp

    def get_next_keypoints(self, **kwargs):

        next_pose = self.get_next_pose(**kwargs)
        next_keypoints = self.nextpose_to_nextkeypoints(next_pose)
        return next_keypoints

    def get_next_velocity(self, **kwargs):

        next = self.get_next(**kwargs)

        # global will always be velocity
        if self.is_velocity:
            return next

        next_vel = self.next_to_nextvelocity(next)

        return next_vel
