import importlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from apf.dataset import Zscore, Discretize, Data, Dataset
from apf.simulation import compare_gt_motion_pred_sim
from apf.io import load_raw_npz_data
from apf.training import to_dataloader, init_model, train
from apf.utils import function_args_from_config
from apf.models import TransformerModel

from flyllm.features import compute_global_velocity

from spatial_infomax.utils.data_loader import HeightMap, load_data
from spatial_infomax.utils.models import compute_whisker


def create_npz(session_ids: list[int]) -> dict:
    """ Generates data for the given session_ids in a format that can be saved with np.savez and read by load_npz_data.

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
        observation: Whisker values for agent at each frame, (n_whisker_angles, n_frames, n_agents) float
        isstart: Indicates whether a frame is the start of a sequence for an agent, (n_frames, n_agents) bool
    """
    # load the raw data
    data = load_raw_npz_data(filepath)

    # extract position
    featnames = list(data['featnames'])
    feat_ids = np.array([featnames.index(name) for name in ['head_x', 'head_y', 'head_theta']])
    position = data['X'][feat_ids]

    # compute observation
    observation = compute_observation(position)

    return position, observation, data['isstart']


def compute_observation(position: np.ndarray) -> np.ndarray:
    """ Computes observation for each agent at each frame.

    Args:
        position: Global position (x, y, theta) of agent at each frame, (3, n_frames, n_agents) float

    Returns
        observation: Whisker values for agent at each frame, (n_whisker_angles, n_frames, n_agents) float
    """
    heightmap = HeightMap()

    _, n_frames, n_agents = position.shape
    wh_vals, _, _ = compute_whisker(
        heightmap=heightmap,
        center=position[:2].reshape((2, n_frames * n_agents)),
        theta=position[2].reshape((n_frames * n_agents))
    )
    return wh_vals.reshape((-1, n_frames, n_agents))


def set_invalid_ends(data: np.ndarray, isstart: np.ndarray, dt: int) -> None:
    """ Sets last dt frames at the end of a continuous sequence to be NaN.

    Args:
        data: Data that was computed using dt, e.g. future motion prediction. (n_features, n_frames, n_agents) float
        isstart: Indicates whether a frame is the start of a sequence for an agent, (n_frames, n_agents) bool
        dt: number of frames to set as invalid.
    """
    n_agents = data.shape[-1]
    for i in range(n_agents):
        starts = np.where(isstart[:, i] == 1)[0]
        invalids = np.unique(np.concatenate([starts - i - 1 for i in range(dt)]))
        data[..., invalids, i] = np.nan


def compute_global_movement(position: np.ndarray, dt: int, isstart: np.ndarray | None = None) -> np.ndarray:
    """ Computes global movement of each agent at each timestep for the given time step, dt.

    NOTE: If user wishes to predict various time steps into the future this function should be
        called separately for each dt.

    Args:
        position: Global position (x, y, theta) of agent at each frame, (3, n_frames, n_agents) float
        dt: Movement to be computed for this many time steps ahead.
        isstart: Indicates whether a frame is the start of a sequence for an agent, (n_frames, n_agents) bool

    Returns:
        movement_global: Global displacement (dside, dforward, dtheta) for dt, (3, n_frames, n_agents) float
    """
    # Compute global velocity
    dXoriginrel, dtheta = compute_global_velocity(
        Xorigin=position[:2],
        Xtheta=position[2],
        tspred_global=[dt],
    )

    # concatenate the global (dforward, dsideways, dorientation)
    movement_global = np.concatenate((dXoriginrel[:, [1, 0], :, :], dtheta[:, None, :, :]), axis=1)

    # reshape it (flatten over tspred)
    ntspred_global, nglobal, n_frames, n_agents = movement_global.shape
    movement_global = movement_global.reshape((ntspred_global * nglobal, n_frames, n_agents))

    if isstart is not None:
        set_invalid_ends(movement_global, isstart, dt)

    return movement_global


def experiment(config: dict) -> None:
    # Load the data
    #   data was created with
    #   train_data = create_npz(np.arange(5))
    #   val_data = create_npz([5])
    position, observation, isstart = load_npz_data(config['intrainfile'])

    # Compute features
    tspred = 1
    global_motion = compute_global_movement(position, dt=tspred, isstart=isstart)

    # Wrap into data class with relevant data operations
    motion_zscore = Zscore()
    # Continuous output
    # motion_data_labels = Data(raw=global_motion.T, operation=motion_zscore)
    # Discrete output
    motion_data_labels = Data(raw=global_motion.T, operation=Discretize())
    # TODO: Move the roll logic into Data or Dataset?
    motion_data_input = Data(raw=np.roll(global_motion.T, shift=tspred, axis=1), operation=motion_zscore)
    observation_data = Data(raw=observation.T, operation=Zscore())

    # Wrap into dataset
    train_dataset = Dataset(
        inputs=[motion_data_input, observation_data],
        labels=[motion_data_labels],
        isstart=isstart,
        context_length=config['contextl'],
    )

    # Now do the same for validation data and use the dataset operations from the training data
    val_position, val_observation, val_sessions = load_npz_data(config['invalfile'])
    val_global_motion = compute_global_movement(val_position, dt=tspred)
    val_motion_data_labels = Data(raw=val_global_motion.T, operation=motion_data_labels.operation)
    val_motion_data_input = Data(raw=np.roll(val_global_motion.T, shift=tspred, axis=1),
                                 operation=motion_data_input.operation)
    val_observation_data = Data(raw=val_observation.T, operation=observation_data.operation)
    val_dataset = Dataset(
        inputs=[val_motion_data_input, val_observation_data],
        labels=[val_motion_data_labels],
        isstart=isstart,
        context_length=config['contextl'],
    )

    # Map to dataloaders
    device = torch.device(config['device'])
    batch_size = config['batch_size']
    train_dataloader = to_dataloader(train_dataset, device, batch_size, shuffle=True)
    val_dataloader = to_dataloader(val_dataset, device, batch_size, shuffle=False)

    # Initialize the model
    model_args = function_args_from_config(config, TransformerModel)
    model = init_model(train_dataset, model_args).to(device)

    # Train the model
    train_args = function_args_from_config(config, train)
    model, best_model, loss_epoch = train(
        train_dataloader,
        val_dataloader,
        model,
        **train_args,
    )

    # Plot the losses
    idx = np.argmin(loss_epoch['val'])
    print((idx, loss_epoch['val'][idx]))
    plt.figure()
    plt.plot(loss_epoch['train'])
    plt.plot(loss_epoch['val'])
    plt.plot(idx, loss_epoch['val'][idx], '.g')
    plt.show()

    # Simulate and compare with ground truth trajectory
    compare_gt_motion_pred_sim(
        model=model,
        motion_operation=val_motion_data_input.operation,
        motion_labels_operation=val_motion_data_labels.operation,
        observation_operation=val_observation_data.operation,
        compute_observation=compute_observation,
        dataset=val_dataset,
        position=val_position,
        agent_id=0,
        t0=200,
        track_len=1000,
        burn_in=55,
        max_contextl=64,
        noise_factor=0,
        bg_img=HeightMap().map,
    )
