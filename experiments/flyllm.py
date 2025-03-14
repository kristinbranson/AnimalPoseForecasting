"""Functions for running flyllm experiments

    See example usage in notebooks/agent_fly.py
"""

import numpy as np
import logging

from apf.io import load_and_filter_data
from apf.dataset import Zscore, Discretize, Data, Dataset, Operation, Fusion, Subset, Roll, Identity, apply_opers_from_data
from apf.utils import modrange, rotate_2d_points, set_invalid_ends
from apf.data import debug_less_data

from flyllm.config import featrelative, keypointnames, featangle
from flyllm.features import (
    kp2feat, compute_sensory_wrapper, compute_global_velocity,
    compute_relpose_velocity, compute_scale_perfly, compute_noise_params, feat2kp
)

LOG = logging.getLogger(__name__)


class Sensory(Operation):
    """ Computes sensory features for the flies.

    NOTE: this operation is not invertible.
    """
    def __init__(self):
        # Keeps track of which dimensions of the sensory output correspond to what (e.g. wall, otherflies, ...)
        self.idxinfo = None

    def apply(self, Xkp: np.ndarray) -> np.ndarray:
        """ Computes sensory features from keypoints.

        Args:
            Xkp: (x, y) pixel position of the agents, (n_agents,  n_frames, n_keypoints, 2) float array

        Returns:
            sensory_data: (n_agents,  n_frames, n_sensory_features) float array
        """
        feats = []
        for flyid in range(Xkp.shape[0]):
            feat, idxinfo = compute_sensory_wrapper(Xkp.T, flyid, returnidx=True)
            feats.append(feat.T)
        self.idxinfo = idxinfo
        return np.array(feats)

    def invert(self, sensory: np.ndarray) -> None:
        LOG.error(f"Operation {self} is not invertible")
        return None


class Pose(Operation):
    """ Computes fly pose from keypoints.
    """
    def __init__(self):
        # Keep track of scale_perfly, needed for inverting operation.
        self.scale_perfly = None

    def apply(self, Xkp: np.ndarray, scale_perfly: np.ndarray = None, flyid: np.ndarray = None):
        """ Computes pose features from keypoints.

        Args:
            Xkp: (x, y) pixel position of the agents, (n_agents,  n_frames, n_keypoints, 2) float array
            scale_perfly: Scale of each unique individual in the data. (n_individuals, n_scales) float array
            flyid: Identity of fly corresponding to Xkp, (n_frames, n_agents) int array

        Returns:
            pose: (n_agents,  n_frames, n_pose_features) float array
        """
        if scale_perfly is not None:
            self.scale_perfly = scale_perfly
        return kp2feat(Xkp=Xkp.T, scale_perfly=scale_perfly, flyid=flyid).T

    def invert(self, pose: np.ndarray, flyid: np.ndarray | int = None):
        """ Computes keypoints from pose features.

        Args:
            pose:  (n_agents,  n_frames, n_pose_features) float array
            flyid: Identity of fly corresponding to pose.
                If pose has a single agent, flyid is an int.
                If it has multiple agents, it as a (n_frames, n_agents) int array.

        Returns:
            Xkp: (x, y) pixel position of the agents, (n_agents,  n_frames, n_keypoints, 2) float array
        """
        return feat2kp(pose.T, scale_perfly=self.scale_perfly, flyid=flyid).T


class LocalVelocity(Operation):
    """ Computes the relative pose movement from t to t + 1.
    """
    def apply(self, pose: np.ndarray, isstart: np.ndarray | None = None) -> np.ndarray:
        """ Compute velocity from pose.

        Args:
            pose: (n_agents,  n_frames, n_pose_features) float array
            isstart: Indicates whether a new fly track starts at a given frame for an agent.
                (n_frames, n_agents) bool array

        Returns:
            velocity: (n_agents,  n_frames, n_pose_features) float array
        """
        pose_velocity = np.moveaxis(compute_relpose_velocity(pose.T), 2, 0)
        if isstart is not None:
            # Set pose deltas that cross individuals to NaN.
            set_invalid_ends(pose_velocity, isstart, dt=1)
        pose_velocity = pose_velocity[0]
        return pose_velocity.T

    def invert(self, velocity: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        """ Compute pose from pose velocity and an initial pose.

        Args:
            velocity: Delta pose (n_agents,  n_frames, n_pose_features) float array
            x0: Initial pose (n_agents,  n_frames, n_pose_features) float array

        Returns:
            pose: (n_agents,  n_frames, n_pose_features) float array
        """
        # Note: here we are assuming dt=1
        if x0 is None:
            n_agents, _, n_features = velocity.shape
            x0 = np.zeros((n_agents, n_features))
        velocity = np.concatenate([x0[:, None, :], velocity], axis=1)[:, :-1, :]
        pose = np.cumsum(velocity, axis=1)
        is_angle = np.where(featangle[featrelative])[0]
        for i in is_angle:
            pose[..., i] = modrange(pose[..., i], -np.pi, np.pi)
        return pose


class GlobalVelocity(Operation):
    """ Computes the global movement of a fly,  dfwd, dside, dtehta, from its (x, y, theta) position.

    Args:
        tspred: A list of dt for which global movement (from t to t+dt) will be computed for.
    """
    def __init__(self, tspred: list[int]):
        self.tspred = tspred

    def apply(self, position: np.ndarray, isstart: np.ndarray | None = None) -> np.ndarray:
        """ Compute global movement from position.

        Args:
            position: (n_agents,  n_frames, 3) float array
            isstart: Indicates whether a new fly track starts at a given frame for an agent.
                (n_frames, n_agents) bool array

        Returns:
            velocity: Global pose velocity, flattened over tspred.
                (n_agents,  n_frames, 3 * len(self.tspred)) float array
        """
        Xorigin = position[..., :2].T
        Xtheta = position[..., 2].T
        _, n_frames, n_flies = Xorigin.shape
        dXoriginrel, dtheta = compute_global_velocity(Xorigin, Xtheta, self.tspred)
        movement_global = np.concatenate((dXoriginrel[:, [1, 0]], dtheta[:, None, :, :]), axis=1)
        if isstart is not None:
            for movement, dt in zip(movement_global, self.tspred):
                set_invalid_ends(movement, isstart, dt=dt)
        movement_global = movement_global.reshape((-1, n_frames, n_flies))

        return movement_global.T

    def invert(self, velocity: np.ndarray, x0: np.ndarray | None = None):
        """ Compute position from global movement and an initial position.

        NOTE: This assumes velocity is only given for dt=1

        Args:
            velocity: Global movmement (n_agents,  n_frames, 3) float array
            x0: Initial pose (n_agents,  n_frames, n_pose_features) float array

        Returns:
            pose: (n_agents,  n_frames, n_pose_features) float array
        """
        if x0 is None:
            n_agents, _, n_dim = velocity.shape
            x0 = np.zeros((n_agents, n_dim))

        d_theta = np.concatenate([x0[:, None, 2], velocity[:, :, 2]], axis=1)
        theta = modrange(np.cumsum(d_theta, axis=1), -np.pi, np.pi)[:, :-1]

        d_pos_rel = velocity[..., [1, 0]]
        d_pos = rotate_2d_points(d_pos_rel.transpose((1, 2, 0)), -theta.T).transpose((2, 0, 1))
        d_pos = np.concatenate([x0[:, None, :2], d_pos], axis=1)
        pos = np.cumsum(d_pos, axis=1)[:, :-1, :]
        return np.concatenate([pos, theta[:, :, None]], axis=-1)


class Velocity(Operation):
    """ Combines global and local velocity.
    """
    def __init__(self):
        # Keep track of global and local indices for the pose features
        self.global_inds = np.where(~featrelative)[0]
        self.local_inds = np.where(featrelative)[0]
        # Use Fusion to apply global vs local operations to the relevant feature dimensions
        self.fusion = Fusion([GlobalVelocity(tspred=[1]), LocalVelocity()], [self.global_inds, self.local_inds])

    def apply(self, pose: np.ndarray, isstart: np.ndarray | None = None):
        """ Compute global and local velocity from pose.

        Args:
            pose: (n_agents,  n_frames, n_pose_features) float array
            isstart: Indicates whether a new fly track starts at a given frame for an agent.
                (n_frames, n_agents) bool array

        Returns:
            velocity: (n_agents,  n_frames, n_pose_features) float array
        """
        return self.fusion.apply(pose, kwargs_per_op={'isstart': isstart})

    def invert(self, velocity: np.ndarray, x0: np.ndarray | None = None):
        """ Compute pose from pose velocity and an initial pose.

        Args:
            velocity: Delta pose (n_agents,  n_frames, n_pose_features) float array
            x0: Initial pose (n_agents,  n_frames, n_pose_features) float array

        Returns:
            pose: (n_agents,  n_frames, n_pose_features) float array
        """
        if x0 is not None:
            kwargs_per_op = [{'x0': x0[..., self.global_inds]}, {'x0': x0[..., self.local_inds]}]
        else:
            kwargs_per_op = None
        return self.fusion.invert(velocity, kwargs_per_op)


def load_data(
        config: dict,
        filename: str,
        debug: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Loads the data and computes scale per fly.

    Args:
        config: Config for loading the data
        filename: Name of file to read the data from (e.g. 'intrainfile', 'invalfile')
        debug: Whether to use less data for debugging

    Returns:
        X: (2, n_keypoints, n_frames, n_agents) float array
        flyids: Identity of fly individuals (n_frames, n_agents) int array
        isstart: indicates whether a new track starts at a give frame for each fly, (n_frames, n_agents) bool array
        isdata: indicates whether data should be used, (n_frames, n_agents) bool array
            Data can be unused because it is invalid (nans) or because it was filtered on fly type.
        scale_perfly: Scale of each unique individual in the data. (n_individuals, n_scales) float array
    """
    data, scale_perfly = load_and_filter_data(
        filename,
        config,
        compute_scale_per_agent=compute_scale_perfly,
        compute_noise_params=compute_noise_params,
        keypointnames=keypointnames,
    )

    if debug:
        debug_less_data(data, n_frames_per_video=45000, max_n_videos=2)

    # Remove all NaN agents (sometimes the last one is a dummy)
    Xkp = data['X']
    valid = np.sum(~np.isnan(Xkp[0, 0]), axis=-2) > 0
    Xkp = Xkp[..., valid]
    flyids = data['ids'][..., valid]
    isstart = data['isstart'][..., valid]
    isdata = data['isdata'][..., valid]

    return Xkp, flyids, isstart, isdata, scale_perfly


def make_dataset(
        config: dict,
        filename: str,
        ref_dataset: Dataset | None = None,
        return_all: bool = False,
        debug: bool = True
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
    Xkp, flyids, isstart, isdata, scale_perfly = load_data(config, config[filename], debug=debug)

    # Compute features
    track = Data('keypoints', Xkp.T, [])
    sensory = Sensory()(track)
    pose = Pose()(track, scale_perfly=scale_perfly, flyid=flyids)
    pose.array[isdata.T == 0] = np.nan
    sensory.array[isdata.T == 0] = np.nan
    velocity = Velocity()(pose, isstart=isstart)

    tspred_global = config['tspred_global']
    aux_tspred = [dt for dt in tspred_global if dt > 1]
    if len(aux_tspred) > 0:
        auxiliary = GlobalVelocity(tspred=aux_tspred)(pose, isstart=isstart)
    else:
        auxiliary = None

    # Assemble the dataset
    if ref_dataset is None:
        # velocity = OddRoot(5)(velocity)
        bin_config = {'nbins': config['discretize_nbins'], 'bin_epsilon': config['discretize_epsilon']}

        # Need to zscore before binning, otherwise bin_epsilon values need to be divided by zscore stds
        zscored_velocity = Zscore()(velocity)
        discreteidx = config['discreteidx']
        bin_config['bin_epsilon'] /= zscored_velocity.operations[-1].std[discreteidx]
        continuousidx = np.setdiff1d(np.arange(velocity.array.shape[-1]), discreteidx)
        indices_per_op = [discreteidx, continuousidx]
        dataset = Dataset(
            inputs = {
                'velocity': Zscore()(Roll(dt=1)(velocity)),
                'pose': Zscore()(Subset(featrelative)(pose)),
                'sensory': Zscore()(sensory),
            },
            labels = {
                'velocity': Fusion([Discretize(**bin_config), Identity()], indices_per_op)(zscored_velocity),
                # 'velocity': Fusion([Discretize(**bin_config), Zscore()], indices_per_op)(velocity),
                # 'auxiliary': Discretize()(auxiliary)
            },
            context_length=config['contextl'],
            isstart=isstart,
        )
    else:
        dataset = Dataset(
            inputs = apply_opers_from_data(ref_dataset.inputs, {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
            labels = apply_opers_from_data(ref_dataset.labels, {'velocity': velocity}), #, 'auxiliary': auxiliary}),
            context_length=config['contextl'],
            isstart=isstart,
        )
    if return_all:
        return dataset, flyids, track, pose, velocity, sensory
    else:
        return dataset
