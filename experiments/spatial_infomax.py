import numpy as np
import logging

from apf.io import load_raw_npz_data
from apf.dataset import Dataset, Data, Zscore, Discretize, Operation, Roll, Velocity, apply_opers_from_data

from spatial_infomax.utils.data_loader import HeightMap, load_data
from spatial_infomax.utils.models import compute_whisker

LOG = logging.getLogger(__name__)


def create_npz(session_ids: list[int]) -> dict:
    """ Generates data for the given session_ids in a format that can be saved with np.savez and read by load_npz_data.

    #   Example:
    #   train_data = create_npz(np.arange(5))
    #   val_data = create_npz([5])

    Note: By default this uses sample_spacing=5 and a smoothing filter
    """
    # Per frame data
    X = []
    ids = []
    frames = []
    y = []

    # Per trial data
    sessionids = []
    trialids = []
    targets = []
    distractors = []
    agent_id = 0

    for ses_idx in session_ids:
        session = load_data(ses_idx=ses_idx)
        trials = session.all_trials()
        for trial_idx in range(len(trials)):
            trial = trials[trial_idx]
            trial = trial.transform()

            n_frames = trial.x.shape[-1]

            X_trial = np.vstack([trial.x, trial.y, trial.z, trial.euler])
            X.append(X_trial)
            ids.append(np.ones(n_frames, int) * agent_id)
            frames.append(np.arange(n_frames))
            y.append(np.ones((1, n_frames), int))

            sessionids.append(ses_idx)
            trialids.append(trial_idx)
            targets.append(trial.target)
            distractors.append(trial.dist)

            agent_id += 1

    return {
        'X': np.concatenate(X, axis=1)[..., None],
        'ids': np.concatenate(ids)[..., None],
        'frames': np.concatenate(frames)[..., None],
        'y': np.concatenate(y, axis=1)[..., None],
        'featnames': ['head_x', 'head_y', 'head_z', 'head_theta'],
        'categories': ['dummy'],
        'sessionids': np.array(sessionids),
        'trialids': np.array(trialids),
        'targets': np.array(targets),
        'distractors': np.array(distractors),
    }


def load_npz_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Loads data from .npz files and computes observation.

    Args:
        filepath: Full path to .npz data

    Returns:
        position: Global position (x, y, theta) of agent at each frame, (3, n_frames, n_agents) float
        isstart: Indicates whether a frame is the start of a sequence for an agent, (n_frames, n_agents) bool
        mouseids: Identity of different mice (or same mouse but different trials).
    """
    # load the raw data
    data = load_raw_npz_data(filepath)

    # extract position
    featnames = list(data['featnames'])
    feat_ids = np.array([featnames.index(name) for name in ['head_x', 'head_y', 'head_theta']])
    position = data['X'][feat_ids]

    return position, data['isstart'], data['ids']


class Sensory(Operation):
    """ Computes whisker values for the mouse.

    NOTE: this operation is not invertible.
    """
    def __init__(self):
        self.heightmap = HeightMap()

    def apply(self, position: np.ndarray) -> np.ndarray:
        """ Computes whisker values from position and heightmap.

        Args:
            position: (x, y, theta) of agent at each frame, (n_agents, n_frames, 3) float

        Returns:
            sensory_data: (n_agents,  n_frames, n_whisker_angles) float array
        """
        positionT = position.T
        _, n_frames, n_agents = positionT.shape
        wh_vals, center_height, end_heights = compute_whisker(
            heightmap=self.heightmap,
            center=positionT[:2].reshape((2, n_frames * n_agents)),
            theta=positionT[2].reshape((n_frames * n_agents))
        )

        # TODO: Add bw distance to edge for all the whiskers (instead of the current binary representation)

        all_heights = np.concatenate([center_height[None, :], end_heights], axis=0)
        return all_heights.reshape((-1, n_frames, n_agents)).T

        return wh_vals.reshape((-1, n_frames, n_agents)).T

    def invert(self, sensory: np.ndarray) -> None:
        LOG.error(f"Operation {self} is not invertible")
        return None


def make_dataset(
        config: dict,
        filename: str,
        ref_dataset: Dataset | None = None,
        return_all: bool = False,
) -> Dataset | tuple[Dataset, np.ndarray, Data, Data, Data]:
    """ Creates a dataset from config, for a given file name and optionally using a reference dataset.

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
        pose: Pose data
        velocity: Velocity data
        sensory: Sensory data
        ]
    """
    # Load the data
    X, isstart, mouseids = load_npz_data(config[filename])

    # Compute features
    pose = Data(name='pose', array=X.T, operations=[])
    sensory = Sensory()(pose)
    velocity = Velocity(featrelative=np.zeros(X.shape[0], bool))(pose, isstart=isstart)

    # Assemble the dataset
    if ref_dataset is None:
        # TODO: determine good bin config values
        bin_config = {'nbins': config['discretize_nbins']}
        dataset = Dataset(
            inputs={
                'velocity': Zscore()(Roll(dt=1)(velocity)),
                'sensory': Zscore()(sensory),
            },
            labels={
                'velocity': Discretize(**bin_config)(velocity),
            },
            isstart=isstart,
            context_length=config['contextl'],
        )
    else:
        dataset = Dataset(
            inputs=apply_opers_from_data(ref_dataset.inputs, {'velocity': velocity, 'sensory': sensory}),
            labels=apply_opers_from_data(ref_dataset.labels, {'velocity': velocity}), #, 'auxiliary': auxiliary}),
            context_length=config['contextl'],
            isstart=isstart,
        )
    if return_all:
        return dataset, mouseids, pose, velocity, sensory
    else:
        return dataset
