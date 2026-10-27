"""Functions for running flyllm experiments

    See example usage in notebooks/agent_fly.py
"""
import copy
import numpy as np
import logging
from dataclasses import dataclass, field


from apf.io import load_and_filter_data
from apf.dataset import (
    Zscore, Discretize, Data, Dataset, Operation, Fusion, Subset, Roll, Identity,
    Velocity, GlobalVelocity, apply_opers_from_data, apply_opers_from_data_params
)
from apf.data import debug_less_data

from flyllm.config import featrelative, keypointnames, featangle
from flyllm.features import (
    kp2feat, compute_sensory_wrapper, compute_scale_perfly, compute_noise_params,
    feat2kp, body_centric_kp, rotate_2d_points
)
import tqdm
import apf.utils as utils

LOG = logging.getLogger(__name__)

@dataclass
class Sensory(Operation):
    """ Computes sensory features for the flies.

    NOTE: this operation is not invertible.
    Attributes: 
        idxinfo: Keeps track of which dimensions of the sensory output correspond to what (e.g. wall, otherflies, ...)        
    """
    
    localattrs = ['idxinfo']
    idxinfo: dict | None = None
    
    def apply(self, Xkp: np.ndarray, isdata: np.ndarray | None = None) -> np.ndarray:
        """ Computes sensory features from keypoints.

        Args:
            Xkp: (x, y) position of the agents' keypoints, (n_agents, n_frames, 2, n_keypoints) float
                array, or (n_frames, 2, n_keypoints) for a single agent, which is then alone in the arena:
                its other-fly features take their maximum-distance value.
            isdata: indicates whether there is data for a given frame or agent, only used to speed up computation,
                (n_frames, n_agents) bool array, or (n_frames,) for a single agent

        Returns:
            sensory_data: (n_agents,  n_frames, n_sensory_features) float array, or (n_frames,
                n_sensory_features) for a single agent
        """
        ismultiagent = Xkp.ndim == 4
        # Single agent: add an agent axis (to Xkp, and to isdata as an (n_frames, 1) column), compute,
        # remove it at the end.
        if not ismultiagent:
            Xkp = Xkp[None, ...]                                 # (1, n_frames, 2, n_keypoints)
            if isdata is not None:
                isdata = np.asarray(isdata).reshape((-1, 1))
        feats = []
        for flyid in range(Xkp.shape[0]):
            if isdata is not None:
                isdatacurr = isdata[:, flyid]
            else:
                isdatacurr = None
            feat, idxinfo = compute_sensory_wrapper(Xkp.T, flyid, returnidx=True, isdata=isdatacurr)
            feats.append(feat.T)
        self.idxinfo = idxinfo
        sensory_data = np.array(feats)                           # (n_agents, n_frames, n_sensory_features)
        if not ismultiagent:
            sensory_data = sensory_data[0]
        return sensory_data

    def invert(self, sensory: np.ndarray) -> None:
        LOG.error(f"Operation {self} is not invertible")
        return None


@dataclass
class SensoryRawKP(Operation):

    def __init__(self):
        self.name = 'sensory'

    def apply(self, Xkp: np.ndarray, isdata: np.ndarray | None = None) -> np.ndarray:
        """ Flattened keypoints of all 'other' flies, for each fly.

        Args:
            Xkp: (x, y) pixel position of the agents, (n_agents,  n_frames, n_keypoints, 2) float array
            isdata: indicates whether there is data for a given frame or agent, only used to speed up computation, (n_frames, n_agents) bool array

        Returns:
            sensory_data: (n_agents,  n_frames, (n_agents-1)*n_keypoints*2) float array
        """
        n_agents, n_frames, d, n_keypoints = Xkp.shape
        sensory_data = np.zeros((n_agents, n_frames, (n_agents-1)*n_keypoints*d))
        for agent_id in range(Xkp.shape[0]):
            other_agents = np.setdiff1d(np.arange(n_agents), np.array([agent_id]))
            sensory_data[agent_id] = Xkp[other_agents].transpose([1, 0, 2, 3]).reshape(n_frames, -1)
        sensory_data[np.isnan(sensory_data)] = 0
        if isdata is not None:
            sensory_data[~isdata.T] = np.nan
        return sensory_data

    def invert(self, sensory: np.ndarray) -> None:
        LOG.error(f"Operation {self} is not invertible")
        return None

@dataclass
class Pose(Operation):
    """ Computes fly pose from keypoints.
    Attributes:
        scale_perfly: Scale of each unique individual in the data. (n_scales, n_individuals) float array
    """
    localattrs = ['scale_perfly']
    scale_perfly: np.ndarray | None = None

    @staticmethod
    def _single_agent_per_frame(value, n_frames: int):
        """ Puts a single agent's per-frame argument (flyid or isdata) in (n_frames, 1) form.

        kp2feat and feat2kp take these as (n_frames, n_agents), with agents on the last axis.

        Args:
            value: None, a scalar (the same for every frame, e.g. one fly's identity), or an
                (n_frames,) array.
            n_frames: number of frames.

        Returns:
            None, or an (n_frames, 1) array.
        """
        if value is None:
            return None
        value = np.asarray(value)
        if value.ndim == 0:
            return np.full((n_frames, 1), value)
        return value.reshape((n_frames, 1))

    def apply(self, Xkp: np.ndarray, scale_perfly: np.ndarray | None = None, flyid: np.ndarray | None = None, isdata: np.ndarray | None = None) -> np.ndarray:
        """ Computes pose features from keypoints.

        Args:
            Xkp: (x, y) position of the agents' keypoints, (n_agents, n_frames, 2, n_keypoints) float
                array, or (n_frames, 2, n_keypoints) for a single agent.
            scale_perfly: Scale of each unique individual in the data. (n_scales, n_individuals) float array
            flyid: Identity of the fly at each frame, (n_frames, n_agents) int array; for a single agent,
                an (n_frames,) int array or one int.
            isdata: Whether each frame has data, (n_frames, n_agents) bool array; for a single agent,
                (n_frames,).

        Returns:
            pose: (n_agents, n_frames, n_pose_features) float array, or (n_frames, n_pose_features) for
                a single agent.
        """
        if scale_perfly is not None:
            self.scale_perfly = scale_perfly
        ismultiagent = Xkp.ndim == 4
        # Single agent: add an agent axis (to Xkp, and to flyid and isdata as (n_frames, 1)
        # columns), compute, remove it at the end.
        if not ismultiagent:
            Xkp = Xkp[None, ...]                                 # (1, n_frames, 2, n_keypoints)
            flyid = self._single_agent_per_frame(flyid, Xkp.shape[1])
            isdata = self._single_agent_per_frame(isdata, Xkp.shape[1])
        pose = kp2feat(Xkp=Xkp.T, scale_perfly=scale_perfly, flyid=flyid, isdata=isdata).T   # (n_agents, n_frames, n_pose_features)
        if not ismultiagent:
            pose = pose[0]
        return pose

    def invert(self, pose: np.ndarray, flyid: np.ndarray | int = None):
        """ Computes keypoints from pose features.

        Args:
            pose: (n_agents, n_frames, n_pose_features) float array, or (n_frames, n_pose_features)
                for a single agent.
            flyid: Identity of the fly at each frame, (n_frames, n_agents) int array; for a single agent,
                an (n_frames,) int array or one int.

        Returns:
            Xkp: (x, y) position of the agents' keypoints, (n_agents, n_frames, 2, n_keypoints) float
                array, or (n_frames, 2, n_keypoints) for a single agent.
        """
        ismultiagent = pose.ndim == 3
        # Single agent: add an agent axis (to pose, and to flyid as an (n_frames, 1) column),
        # compute, remove it at the end.
        if not ismultiagent:
            pose = pose[None, ...]                               # (1, n_frames, n_pose_features)
            flyid = self._single_agent_per_frame(flyid, pose.shape[1])
        Xkp = feat2kp(pose.T, scale_perfly=self.scale_perfly, flyid=flyid).T   # (n_agents, n_frames, 2, n_keypoints)
        if not ismultiagent:
            Xkp = Xkp[0]
        return Xkp


@dataclass
class BodyCentricKP(Operation):

    def __init__(self):
        self.name = 'pose'

    def apply(self, Xkp):
        Xn, fthorax, thorax_theta = body_centric_kp(Xkp.T)
        n_kp, dim, n_frames, n_flies = Xn.shape
        return np.concatenate([fthorax, thorax_theta[None, :, :], Xn.reshape((-1, n_frames, n_flies))], axis=0).T

    def invert(self, body_centric: np.ndarray):
        body_centricT = body_centric.T
        fthorax = body_centricT[:2]
        thorax_theta = body_centricT[2:3]
        _, n_frames, n_flies = body_centricT.shape
        kpts = body_centricT[3:].reshape((-1, 2, n_frames, n_flies))
        return (rotate_2d_points(kpts, -thorax_theta) + fthorax).T


class RawKP(Operation):
    # Returns the flattened keypoints

    def __init__(self):
        self.name = 'pose'
        self.d_kp = None

    def apply(self, Xkp):
        n_flies, n_frames, d_kp, n_keypoints = Xkp.shape
        self.d_kp = d_kp
        return Xkp.reshape(n_flies, n_frames, -1)

    def invert(self, Xkp_flat: np.ndarray):
        d_feat = Xkp_flat.shape[-1]
        new_shape = list(Xkp_flat.shape) + [1]
        new_shape[-2] = self.d_kp
        new_shape[-1] = d_feat//self.d_kp
        return Xkp_flat.reshape(new_shape)


class VelocityAndNextPose(Operation):
    """ Combines global and local velocity.

    Attributes:
        featrelative: Bool vector indicating whether pose index is a relative feature.
    """

    localattrs = ['global_inds', 'local_inds', 'fusion']

    def __init__(self, featrelative: np.ndarray):
        # Keep track of global and local indices for the pose features
        self.global_inds = np.where(~featrelative)[0]
        self.local_inds = np.where(featrelative)[0]
        # Use Fusion to apply global vs local operations to the relevant feature dimensions
        # We apply the roll operation with -1 to the relative features, so that they are the pose of the
        # next frame.
        self.fusion = Fusion(
            operations=[GlobalVelocity(tspred=[1]), Roll(dt=-1)],
            indices_per_op=[self.global_inds, self.local_inds]
        )
        self.name = 'velocity'
        super().__post_init__()

    def apply(self, pose: np.ndarray, isstart: np.ndarray | None = None):
        """ Compute global and local velocity from pose.

        Args:
            pose: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
            isstart: Indicates whether a new fly track starts at a given frame for an agent.
                (n_frames, n_agents) bool array or (n_frames,) bool array

        Returns:
            velocity: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
        """
        return self.fusion.apply(pose, kwargs_per_op=[{'isstart': isstart}, {}])

    def invert(self, velocity: np.ndarray, x0: np.ndarray | None = None):
        """ Compute pose from pose velocity and an initial pose.

        Args:
            velocity: Delta pose (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
            x0: Initial pose (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array

        Returns:
            pose: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
        """
        if x0 is not None:
            # if has a value for every frame, just take the first frame
            if x0.ndim == velocity.ndim:
                x0 = x0[0]
            kwargs_per_op = [{'x0': x0[..., self.global_inds]}, {}]
        else:
            kwargs_per_op = None
        return self.fusion.invert(velocity, kwargs_per_op)


def load_data(
        config: dict,
        filename: str,
        debug: bool = False,
        output_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Loads the data and computes scale per fly.

    Args:
        config: Config for loading the data
        filename: Name of file to read the data from (e.g. 'intrainfile', 'invalfile')
        debug: Whether to use less data for debugging

    Returns:
        X: (x, y) keypoint positions, (n_keypoints, 2, n_frames, n_agents) float array
        flyids: Identity of fly individuals (n_frames, n_agents) int array
        isstart: indicates whether a new track starts at a give frame for each fly, (n_frames, n_agents) bool array
        isdata: indicates whether data should be used, (n_frames, n_agents) bool array
            Data can be unused because it is invalid (nans) or because it was filtered on fly type.
        scale_perfly: Scale of each unique individual in the data. (n_scales, n_individuals) float array
        useoutputmask: whether each frame's output may be used for the loss, (n_frames, n_agents) bool
            array. Frames are excluded when the fly is not in config['output_categories'].
    """
    data, scale_perfly = load_and_filter_data(
        filename,
        config,
        compute_scale_per_agent=compute_scale_perfly,
        compute_noise_params=compute_noise_params,
        keypointnames=keypointnames,
        debug=debug,
        n_frames_per_video=15000,
        max_n_videos=5
    )

    # Remove all NaN agents (sometimes the last one is a dummy)
    Xkp = data['X']
    valid = np.sum(~np.isnan(Xkp[0, 0]), axis=-2) > 0

    # TODO: Remove this hack
    if config.get('use_raw_kp', False):
    # if False: # For the purpose of comparing models, I want to set all data to invalid where there are 11 flies
        print('ELIMINATING 11th FLY!')
        vidids = np.unique(data['videoidx'])
        for vidid in vidids:
            inds = np.where(data['videoidx'][:, 0]==vidid)[0]
            counts = (~np.isnan(data['X'][7, 0, inds])).sum(-1)
            if counts.max() > 10:
                data['isdata'][inds, :] = False
        if len(valid) > 10:
            valid[10:] = False

    Xkp = Xkp[..., valid]
    flyids = data['ids'][..., valid]
    isstart = data['isstart'][..., valid]
    isdata = data['isdata'][..., valid]
    labels = data['y'][..., valid]
    categories = data['categories']
    useoutputmask = data['useoutputmask'][..., valid]

    if output_labels:
        return Xkp, flyids, isstart, isdata, scale_perfly, useoutputmask, labels, categories
    return Xkp, flyids, isstart, isdata, scale_perfly, useoutputmask


def make_dataset(
        config: dict,
        filename: str,
        ref_dataset: Dataset | None = None,
        return_all: bool = False,
        debug: bool = True,
        indata: dict | None = None
) -> Dataset | tuple[Dataset, np.ndarray, Data, Data, Data, Data]:
    """ Creates a dataset from config, for a given file name and optionally using a reference dataset.

    TODO: Currently this doesn't support different ways of computing the features,
      should assert that the config matches what is done here

    Args:
        config: Config for loading the data
        filename: Name of file to read the data from (e.g. 'intrainfile', 'invalfile')
        ref_dataset: Dataset to copy postprocessing operations from.
            When loading validation/test set, provide training set.
        return_all: Whether to return intermediate variables in addition to Dataset (see returns)
        debug: Whether to use less data for debugging

    Returns:
        dataset: Dataset for flyllm experiment.
        [
        flyids: Identity of fly individuals (n_frames, n_agents) int array
        track: Keypoint data
        pose: Pose data
        velocity: Velocity data
        sensory: Sensory data
        ]
    """
    
    # Load data
    if indata is None:
        Xkp, flyids, isstart, isdata, scale_perfly, useoutputmask, labels, categories = load_data(config, config[filename], debug=debug, output_labels=True)
    else:
        Xkp = indata['Xkp']
        flyids = indata['flyids']
        isstart = indata['isstart']
        isdata = indata['isdata']
        useoutputmask = indata['useoutputmask']
        scale_perfly = indata['scale_perfly']

    # Compute features
    LOG.info('Computing input and label features for dataset...')
    track = Data('keypoints', Xkp.T, [])
    print(track.array.shape)

    LOG.info('Computing sensory features...')
    if config.get('use_raw_kp', False):
        sensory = SensoryRawKP()(track, isdata=isdata)
    else:
        sensory = Sensory()(track, isdata=isdata)

    if config.get('body_centric_kp', False):
        LOG.info('Computing body_centric_kp features...')
        pose = BodyCentricKP()(track)
        pose.array[~isdata.T] = np.nan
        is_relative = np.ones(pose.array.shape[-1], bool)
        is_angle = np.zeros(pose.array.shape[-1], bool)
        is_relative[:3] = 0
        is_angle[2] = 1
    elif config.get('use_raw_kp', False):
        LOG.info('Computing raw kp features (flatten)...')
        pose = RawKP()(track)
        pose.array[~isdata.T] = np.nan
        is_relative = np.ones(pose.array.shape[-1], bool)
        is_angle = np.zeros(pose.array.shape[-1], bool)
    else:
        LOG.info('Computing pose features...')
        pose = Pose()(track, scale_perfly=scale_perfly, flyid=flyids, isdata=isdata)
        is_relative = featrelative
        is_angle = featangle

    assert not np.any(np.isnan(sensory.array[isdata.T])) and np.all(
        np.isnan(sensory.array[~isdata.T])), "Sensory features should be nan iff isdata == False"
    assert not np.any(np.isnan(pose.array[isdata.T])) and np.all(
        np.isnan(pose.array[~isdata.T])), "Pose features should be nan iff isdata == False"

    print(pose.array.shape)

    LOG.info('Computing velocity features...')
    if config.get('global_velocity_only', False):
        # global_ids = np.where(is_relative == 0)[0]
        # local_ids = np.where(is_relative == 1)[0]
        # vel_op = Fusion([GlobalVelocity(tspred=config['tspred_global']), Identity()], [global_ids, local_ids])
        # vel_op.name = 'velocity'  # override the name for ease of simulation
        # velocity = vel_op(pose, kwargs_per_op=[{'isstart': isstart}, {}])
        velocity = VelocityAndNextPose(featrelative=is_relative)(pose, isstart=isstart)
    else:
        if sum(is_relative) == len(is_relative):
            from apf.dataset import LocalVelocity
            vel_op = LocalVelocity(is_angle=is_angle)
            vel_op.name = 'velocity'
            velocity = vel_op(pose, isstart=isstart)
        else:
            velocity = Velocity(featrelative=is_relative, featangle=is_angle)(pose, isstart=isstart)

    tspred_global = config['tspred_global']
    aux_tspred = [dt for dt in tspred_global if dt > 1]
    if len(aux_tspred) > 0:
        LOG.warning('!!!Auxiliary prediction is currently commented out!!!')
        auxiliary = GlobalVelocity(tspred=aux_tspred)(pose, isstart=isstart)
    else:
        auxiliary = None

    metadata = {'labels': { 'velocity': {'pose': flyids.T, 'velocity': pose.array} },
                'inputs': { 'pose': {'pose': flyids.T},
                            'velocity': {'pose': flyids.T} }}

    LOG.info('Assembling dataset...')

    args = {
        'context_length': config['contextl'],
        'isstart': isstart,
        'metadata': metadata,
        'useoutputmask': useoutputmask
    }

    # Assemble the dataset
    if ref_dataset is not None:
        dataset = Dataset(
            inputs=apply_opers_from_data(ref_dataset.inputs, {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
            labels=apply_opers_from_data(ref_dataset.labels, {'velocity': velocity}), #, 'auxiliary': auxiliary}),
            **args
        )
    elif 'dataset_params' in config and config['dataset_params'] is not None and \
        ('inputs' in config['dataset_params']) and ('labels' in config['dataset_params']):
        dataset = Dataset(
            inputs=apply_opers_from_data_params(config['dataset_params']['inputs'], {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
            labels=apply_opers_from_data_params(config['dataset_params']['labels'], {'velocity': velocity}), #, 'auxiliary': auxiliary}),
            **args
        )
    else:
        # Need to zscore before binning, otherwise bin_epsilon values need to be divided by zscore stds
        zscored_velocity = Zscore()(velocity)

        discreteidx = config['discreteidx']

        if discreteidx is None or len(discreteidx) == 0:
            label_velocity = zscored_velocity
        else:
            continuousidx = np.setdiff1d(np.arange(velocity.array.shape[-1]), discreteidx)
            indices_per_op = [discreteidx, continuousidx]

            zsig = zscored_velocity.operations[-1].std
            bin_config = {'nbins': config['discretize_nbins'],
                          'bin_epsilon': config['discretize_epsilon'] / zsig[discreteidx]}

            if 'bin_edges_absolute' in config and config['bin_edges_absolute'] is not None and len(config['bin_edges_absolute']) > 0:
                # check that all discreteidx are present
                assert all(idx in config['bin_edges_absolute'] for idx in discreteidx), "Not all discreteidx are present in bin_edges_absolute"
                bin_config['bin_edges'] = np.vstack([config['bin_edges_absolute'][featidx]/zsig[featidx] for featidx in discreteidx]) # nfeat x nbins + 1

            label_velocity = Fusion([Discretize(**bin_config), Identity()], indices_per_op)(zscored_velocity)

        if config.get('global_velocity_only', False):
            # We can skip the current pose here because the rolled by 1 velocity contains the current relative pose.
            inputs={
                'velocity': Zscore()(Roll(dt=1)(velocity)),
                'sensory': Zscore()(sensory),
            }
        else:
            inputs={
                'velocity': Zscore()(Roll(dt=1)(velocity)),
                'pose': Zscore()(Subset(is_relative)(pose)),
                'sensory': Zscore()(sensory),
            }

        dataset = Dataset(
            inputs=inputs,
            labels={
                'velocity': label_velocity,
                # 'velocity': Fusion([Discretize(**bin_config), Zscore()], indices_per_op)(velocity),
                # 'auxiliary': Discretize()(auxiliary)
            },
            **args
        )
    dataset_params = dataset.get_params()
    if return_all:
        return dataset, flyids, track, pose, velocity, sensory, dataset_params, isdata, isstart, useoutputmask, labels, categories
    else:
        return dataset
