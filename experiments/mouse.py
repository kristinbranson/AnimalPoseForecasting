"""Functions for running mouse experiments, on the MABe22 mouse triplets.

The counterpart of experiments/flyllm.py.  Pose is body-centric keypoints (mousellm.features), the inputs
are the previous movement, the pose and the other mice's sensory features, and the labels are the next
movement: forward, sideways and turn of the body frame, discretized, and the change in the body-frame
keypoints, continuous.
"""
import logging
from dataclasses import dataclass

import numpy as np

from apf.dataset import (
    Zscore, Discretize, Data, Dataset, Operation, Fusion, Subset, Roll, Identity, Velocity,
    apply_opers_from_data, apply_opers_from_data_params
)
from apf.io import load_and_filter_data
from mousellm.config import featangle, featrelative, keypointnames
from mousellm.features import body_centric_kp_to_pose, compute_sensory, get_sensory_feature_idx, pose_to_kp

LOG = logging.getLogger(__name__)


@dataclass
class Sensory(Operation):
    """ Computes the mice's sensory features: vision and touch of the other mice.

    NOTE: this operation is not invertible.
    Attributes:
        idxinfo: Which dimensions of the sensory output are which kind of feature.
    """

    localattrs = ['idxinfo']
    idxinfo: dict | None = None

    def apply(self, Xkp: np.ndarray, isdata: np.ndarray | None = None) -> np.ndarray:
        """ Computes sensory features from keypoints.

        Args:
            Xkp: (x, y) positions of the agents' keypoints, (n_agents, n_frames, 2, n_keypoints) float array.
            isdata: whether each agent has data at each frame, (n_frames, n_agents) bool array.  Other frames'
                features are NaN.  None for all frames.

        Returns:
            sensory_data: (n_agents, n_frames, n_sensory_features) float array
        """
        self.idxinfo = get_sensory_feature_idx()
        sensory_data = None
        for mousenum in range(Xkp.shape[0]):
            feat = compute_sensory(Xkp, mousenum, None if isdata is None else isdata[:, mousenum])
            if sensory_data is None:
                sensory_data = np.empty((Xkp.shape[0],) + feat.shape, dtype=feat.dtype)
            sensory_data[mousenum] = feat
        return sensory_data

    def invert(self, sensory: np.ndarray) -> None:
        LOG.error(f"Operation {self} is not invertible")
        return None


@dataclass
class BodyCentricKP(Operation):
    """ Computes pose features, body-centric keypoints (mousellm.config.posenames), from keypoints. """

    def __init__(self):
        self.name = 'pose'

    def apply(self, Xkp: np.ndarray) -> np.ndarray:
        """ Xkp: (n_agents, n_frames, 2, n_keypoints) -> (n_agents, n_frames, n_pose_features) """
        return body_centric_kp_to_pose(Xkp)

    def invert(self, pose: np.ndarray, ids: np.ndarray | None = None) -> np.ndarray:
        """ pose: (..., n_pose_features) -> (..., 2, n_keypoints)

        ids, the track ids, are unused.  The dataset metadata passes them to the pose operation, since
        experiments.flyllm.Pose needs them. """
        return pose_to_kp(pose)


def load_data(
        config: dict,
        filename: str,
        debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """ Loads a data file made by mousellm/convert_mabe22.py, filtered and flip-augmented as config says.

    Args:
        config: Config for loading the data
        filename: Name of the file to read the data from
        debug: Whether to use less data for debugging

    Returns:
        X: (x, y) keypoint positions, (n_keypoints, 2, n_frames, n_agents) float array
        ids: track ids, (n_frames, n_agents) int array, -1 where untracked
        isstart: whether a new track starts at each frame for each mouse, (n_frames, n_agents) bool array
        isdata: whether data should be used, (n_frames, n_agents) bool array
        useoutputmask: whether each frame's output may be used for the loss, (n_frames, n_agents) bool array
        labels: (n_categories, n_frames, n_agents) category labels; the mouse data has none
        categories: category names
    """
    # No scale per agent: body-centric keypoints don't use one, and computing it for every track is slow.
    data, _ = load_and_filter_data(
        filename,
        config,
        keypointnames=keypointnames,
        debug=debug,
        n_frames_per_video=15000,
        max_n_videos=5
    )
    return data['X'], data['ids'], data['isstart'], data['isdata'], data['useoutputmask'], data['y'], \
        data['categories']


def make_dataset(
        config: dict,
        filename: str,
        ref_dataset: Dataset | None = None,
        return_all: bool = False,
        debug: bool = True,
) -> Dataset | tuple:
    """ Creates a dataset from config, for a given file name and optionally using a reference dataset.

    Args:
        config: Config for loading the data
        filename: Key of the config entry naming the file to read the data from (e.g. 'intrainfile',
            'invalfile')
        ref_dataset: Dataset to copy postprocessing operations from.
            When loading validation/test set, provide training set.
        return_all: Whether to return intermediate variables in addition to Dataset (see returns)
        debug: Whether to use less data for debugging

    Returns:
        dataset: Dataset for mouse experiment.
        [
        ids: Track ids (n_frames, n_agents) int array
        track: Keypoint data
        pose: Pose data
        velocity: Velocity data
        sensory: Sensory data
        dataset_params: The dataset's operations' parameters
        isdata, isstart, useoutputmask: (n_frames, n_agents) bool arrays
        labels, categories: category labels and names
        ]
        The same as experiments.flyllm.make_dataset returns, so flyllm.prepare can use either.
    """
    Xkp, ids, isstart, isdata, useoutputmask, labels, categories = load_data(config, config[filename], debug=debug)

    LOG.info('Computing input and label features for dataset...')
    track = Data('keypoints', Xkp.T, [])

    LOG.info('Computing sensory features...')
    sensory = Sensory()(track, isdata=isdata)

    LOG.info('Computing body-centric keypoint pose features...')
    pose = BodyCentricKP()(track)

    assert not np.any(np.isnan(sensory.array[isdata.T])) and np.all(
        np.isnan(sensory.array[~isdata.T])), "Sensory features should be nan iff isdata == False"
    assert not np.any(np.isnan(pose.array[isdata.T])) and np.all(
        np.isnan(pose.array[~isdata.T])), "Pose features should be nan iff isdata == False"

    LOG.info('Computing velocity features...')
    velocity = Velocity(featrelative=featrelative, featangle=featangle)(pose, isstart=isstart)

    metadata = {'labels': {'velocity': {'pose': ids.T, 'velocity': pose.array}},
                'inputs': {'pose': {'pose': ids.T},
                           'velocity': {'pose': ids.T}}}

    LOG.info('Assembling dataset...')

    args = {
        'context_length': config['contextl'],
        'isstart': isstart,
        'metadata': metadata,
        'useoutputmask': useoutputmask
    }

    if ref_dataset is not None:
        dataset = Dataset(
            inputs=apply_opers_from_data(ref_dataset.inputs, {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
            labels=apply_opers_from_data(ref_dataset.labels, {'velocity': velocity}),
            **args
        )
    elif 'dataset_params' in config and config['dataset_params'] is not None and \
            ('inputs' in config['dataset_params']) and ('labels' in config['dataset_params']):
        dataset = Dataset(
            inputs=apply_opers_from_data_params(config['dataset_params']['inputs'],
                                                {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
            labels=apply_opers_from_data_params(config['dataset_params']['labels'], {'velocity': velocity}),
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

            if config.get('bin_edges_absolute'):
                assert all(idx in config['bin_edges_absolute'] for idx in discreteidx), \
                    "Not all discreteidx are present in bin_edges_absolute"
                bin_config['bin_edges'] = np.vstack([config['bin_edges_absolute'][featidx] / zsig[featidx]
                                                     for featidx in discreteidx])  # nfeat x nbins + 1

            label_velocity = Fusion([Discretize(**bin_config), Identity()], indices_per_op)(zscored_velocity)

        inputs = {
            'velocity': Zscore()(Roll(dt=1)(velocity)),
            'pose': Zscore()(Subset(featrelative)(pose)),
            'sensory': Zscore()(sensory),
        }

        dataset = Dataset(
            inputs=inputs,
            labels={'velocity': label_velocity},
            **args
        )
    dataset_params = dataset.get_params()
    if return_all:
        return dataset, ids, track, pose, velocity, sensory, dataset_params, isdata, isstart, useoutputmask, \
            labels, categories
    else:
        return dataset
