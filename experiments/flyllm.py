"""Functions for running flyllm experiments

    See example usage in notebooks/agent_fly.py
"""

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
    kp2feat, compute_sensory_wrapper, compute_scale_perfly, compute_noise_params, feat2kp
)

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


@dataclass
class Pose(Operation):
    """ Computes fly pose from keypoints.
    Attributes:
        scale_perfly: Scale of each unique individual in the data. (n_individuals, n_scales) float array
    """
    localattrs = ['scale_perfly']
    scale_perfly: np.ndarray | None = None

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
    velocity = Velocity(featrelative=featrelative, featangle=featangle)(pose, isstart=isstart)

    tspred_global = config['tspred_global']
    aux_tspred = [dt for dt in tspred_global if dt > 1]
    if len(aux_tspred) > 0:
        LOG.warning('!!!Auxiliary prediction is currently commented out!!!')
        auxiliary = GlobalVelocity(tspred=aux_tspred)(pose, isstart=isstart)
    else:
        auxiliary = None

    # Assemble the dataset
    if ref_dataset is not None:
        dataset = Dataset(
            inputs=apply_opers_from_data(ref_dataset.inputs, {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
            labels=apply_opers_from_data(ref_dataset.labels, {'velocity': velocity}), #, 'auxiliary': auxiliary}),
            context_length=config['contextl'],
            isstart=isstart)
    elif 'dataset_params' in config and config['dataset_params'] is not None and \
        ('inputs' in config['dataset_params']) and ('labels' in config['dataset_params']):
        dataset = Dataset(
            inputs=apply_opers_from_data_params(config['dataset_params']['inputs'], {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
            labels=apply_opers_from_data_params(config['dataset_params']['labels'], {'velocity': velocity}), #, 'auxiliary': auxiliary}),
            context_length=config['contextl'],
            isstart=isstart)
    else:
        # velocity = OddRoot(5)(velocity)

        discreteidx = config['discreteidx']
        continuousidx = np.setdiff1d(np.arange(velocity.array.shape[-1]), discreteidx)
        indices_per_op = [discreteidx, continuousidx]
    

        # Need to zscore before binning, otherwise bin_epsilon values need to be divided by zscore stds
        zscored_velocity = Zscore()(velocity)
        bin_config = {'nbins': config['discretize_nbins'],
                      'bin_epsilon': config['discretize_epsilon'] / zscored_velocity.operations[-1].std[discreteidx]}

        dataset = Dataset(
            inputs={
                'velocity': Zscore()(Roll(dt=1)(velocity)),
                'pose': Zscore()(Subset(featrelative)(pose)),
                'sensory': Zscore()(sensory),
            },
            labels={
                'velocity': Fusion([Discretize(**bin_config), Identity()], indices_per_op)(zscored_velocity),
                # 'velocity': Fusion([Discretize(**bin_config), Zscore()], indices_per_op)(velocity),
                # 'auxiliary': Discretize()(auxiliary)
            },
            context_length=config['contextl'],
            isstart=isstart,
        )
    dataset_params = dataset.get_params()
    if return_all:
        return dataset, flyids, track, pose, velocity, sensory, dataset_params
    else:
        return dataset
