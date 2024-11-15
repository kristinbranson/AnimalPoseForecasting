import numpy as  np
import torch
import matplotlib.pyplot as plt
from typing import Callable

from apf.utils import rotate_2d_points
from apf.dataset import Operation, Dataset
from apf.models import TransformerModel


def apply_motion(
        x: float, y: float, theta: float, d_fwd: float, d_side: float, d_theta: float
) -> tuple[float, float, float]:
    """ Updates current 2d pose with a delta pose.

    Args:
        x, y: 2d position
        theta: Orientation in radians
        d_fwd, d_side: Delta to position at zero-orientation
        d_theta: Delta to orientation

    Returns:
        x_new, y_new, theta_new: Updated 2d pose
    """
    dx, dy = rotate_2d_points(np.array([d_side, d_fwd])[None, :], -theta).flat
    return x + dx, y + dy, theta + d_theta


def get_motion_track(
    labels_operation: Operation,
    motion_labels: torch.Tensor,
    gt_track: np.ndarray,
) -> np.ndarray:
    """ Uses initial position of gt track and iteratively applies motion at each frame to obtain a new track.

    Args:
        motion_labels: (n_frames, n_motion_features) labels corresponding to delta pose at each frame.
        labels_operation: Operation that has been applied to motion for training (e.g. zscore or binning).
        gt_track: (3, n_frames) ground truth 2d pose at each frame.

    Returns:
        motion_track: (3, n_frames) 2d pose at each frame by derived from per frame motion and initial pose.
    """
    gt_motion = labels_operation.inverse(
        motion_labels.detach().cpu().numpy()[None, ...])[0, ...]
    n_frames = gt_motion.shape[0]
    track = np.zeros((3, n_frames))
    track[:, 0] = gt_track[:, 0]
    x, y, theta = track[:, 0]
    for i in range(n_frames-1):
        dx, dy, dtheta = gt_motion[i, :]
        x, y, theta = apply_motion(x, y, theta, dx, dy, dtheta)
        track[:, i + 1] = [x, y, theta]

    return track


def get_pred_motion_track(
    model: TransformerModel,
    labels_operation: Operation,
    gt_input: torch.Tensor,
    gt_track: np.ndarray,
) -> np.ndarray:
    """ Given ground truth inputs, makes a motion prediction and returns the resulting motion track.

    Args:
        model: Transformer model used for predicting future motion.
        labels_operation: Operation that was applied to motion data for training (e.g. zscore or binning).
        gt_input: (batch_size, n_frames, d_input) ground truth input to the model.
        gt_track: (3, n_frames) ground truth 2d pose at each frame.

    Returns:
        motion_track: (3, n_frames) 2d pose at each frame by derived from per frame predicted motion and initial pose.
    """
    model.eval()
    device = next(model.parameters()).device
    n_frames = gt_track.shape[-1]
    train_src_mask = torch.nn.Transformer.generate_square_subsequent_mask(n_frames, device=device)
    valpred = model.output(gt_input[None, ...], mask=train_src_mask, is_causal=False)
    key = 'discrete' if model.is_mixed else 'continuous'
    pred_motion_processed = valpred[key][0]
    return get_motion_track(
        motion_labels=pred_motion_processed,
        labels_operation=labels_operation,
        gt_track=gt_track,
    )


def simulate(
    model: TransformerModel,
    motion_operation: Operation,
    motion_labels_operation: Operation,
    observation_operation: Operation,
    compute_observation: Callable,
    gt_input: torch.Tensor,
    gt_track: np.ndarray,
    burn_in: int = 30,
    track_len: int = 1000,
    max_contextl: int | None = 64,
    noise_factor: float = 0,
) -> np.ndarray:
    """ Simulates an agent given model and some initialization.

    TODO: compare with Kristin's sliding window prediction in test_dataset

    Args:
        model: Transformer model used for predicting future motion.
        motion_operation: Operation to be applied to motion input data (e.g. zscore).
        observation_operation: Operation to be applied to observation input data (e.g. zscore).
        motion_labels_operation: Operation that was applied to motion label data for training (e.g. zscore or binning).
        compute_observation: Function for computing observation given current state
        gt_input: (batch_size, n_frames, d_input) ground truth input to the model. Used for burning in the simulation.
        gt_track: (3, n_frames) ground truth 2d pose at each frame. Used for first burn_in frames.
        burn_in: How many frames to apply the model to gt_input before simulating.
        track_len: How long the total track should be after simulation.
        max_contextl: Max number of frames to feed as input to the model. If None, uses the full history.
        noise_factor: How much noise to add to prediction (only used for continuous output)
            (helpful for getting non-deterministic tracks for continuous motion output)

    Returns:
        sim_track: (3, track_len) 2d pose at each frame based on open loop simulation.
    """
    if burn_in > 0:
        assert gt_track is not None and gt_input is not None

    # Initialize model input
    device = next(model.parameters()).device
    model_input = torch.zeros((1, track_len, gt_input.shape[-1]))
    curr_frame = burn_in
    model_input[0, :curr_frame, :] = gt_input[:curr_frame]
    model_input = model_input.to(device)

    # Initialize track
    track = np.zeros((3, track_len))
    track[:, :curr_frame] = gt_track[:, :curr_frame]
    track[:, curr_frame:] = 0

    # Currently only handling all discrete or all continuous
    key = 'discrete' if model.is_mixed else 'continuous'

    model.eval()
    while curr_frame < track_len:
        # Make a motion prediction
        frame0 = 0
        if max_contextl is not None:
            frame0 = max(0, curr_frame - max_contextl)
        pred = model.output(model_input[:, frame0:curr_frame, :])[key]
        pred_np = pred.detach().cpu().numpy()[0, -1, :]

        # Map prediction to raw data format
        if noise_factor > 0 and not model.is_mixed:
            pred_np += np.random.randn(pred_np.shape[-1]) * noise_factor
        pred_np = motion_labels_operation.inverse(pred_np[None, None, ...])[0, ...]

        # Update position of mouse
        d_fwd, d_side, d_theta = pred_np.flat
        x_prev, y_prev, theta_prev = track[:, curr_frame - 1]
        x, y, theta = apply_motion(x_prev, y_prev, theta_prev, d_fwd, d_side, d_theta)

        # Compute observation
        observation = compute_observation(np.array([x, y, theta])[:, None, None])

        # Assemble the next input
        next_input = np.array([d_fwd, d_side, d_theta] + list(observation.flat))
        next_input[:3] = motion_operation.apply(next_input[None, None, :3]).flat
        next_input[3:] = observation_operation.apply(next_input[None, None, 3:]).flat

        model_input[0, curr_frame, :] = torch.from_numpy(next_input)
        track[:, curr_frame] = [x, y, theta]

        curr_frame += 1

    return track


def compare_gt_motion_pred_sim(
        model: TransformerModel,
        motion_operation: Operation,
        motion_labels_operation: Operation,
        observation_operation: Operation,
        compute_observation: Callable,
        dataset: Dataset,
        position: np.ndarray,
        agent_id: int = 0,
        t0: int = 200,
        track_len: int = 1000,
        burn_in: int = 35,
        max_contextl: int | None = 64,
        noise_factor: float = 0,
        bg_img: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Compares ground truth track with tracks obtained from: gt motion, predicted motion, simualtion.

    Args:
        model: Transformer model used for predicting future motion.
        motion_operation: Operation to be applied to motion input data (e.g. zscore).
        observation_operation: Operation to be applied to observation input data (e.g. zscore).
        motion_labels_operation: Operation that was applied to motion label data for training (e.g. zscore or binning).
        compute_observation: Function for computing observation given current state
        dataset: Dataset, used to extract ground truth inputs and labels from.
        position: Ground truth position data.
        agent_id: Agent from position to use.
        t0: First frame from position to use.
        burn_in: How many frames to apply the model to gt_input before simulating.
        track_len: How long the total track should be after simulation.
        max_contextl: Max number of frames to feed as input to the model. If None, uses the full history.
        noise_factor: How much noise to add to prediction (only used for continuous output)
            (helpful for getting non-deterministic tracks for continuous motion output)
        bg_img: Image to display trajectories on.

    Returns:
        gt_track: (3, track_len) ground truth 2d pose track with (x, y, theta) for each frame.
        motion_track: (3, track_len) track obtained from iteratively applying ground truth motion labels.
        pred_track: (3, track_len) track obtained from motion prediction on top of ground truth input data.
        sim_track: (3, track_len) track obtained from open loop simulation.
    """
    gt_track = position[:, t0:t0 + track_len, agent_id]
    gt_chunk = dataset.get_chunk(start_frame=t0, duration=track_len, agent_id=agent_id)
    if dataset.n_bins is None:
        gt_labels = gt_chunk['labels']
    else:
        gt_labels = gt_chunk['labels_discrete'].reshape((-1, dataset.d_output_discrete, dataset.n_bins))

    motion_track = get_motion_track(
        motion_labels=gt_labels,
        labels_operation=motion_labels_operation,
        gt_track=gt_track,
    )

    pred_track = get_pred_motion_track(
        model=model,
        labels_operation=motion_labels_operation,
        gt_input=gt_chunk['input'],
        gt_track=gt_track,
    )

    sim_track = simulate(
        model=model,
        motion_operation=motion_operation,
        motion_labels_operation=motion_labels_operation,
        observation_operation=observation_operation,
        compute_observation=compute_observation,
        gt_input=gt_chunk['input'],
        gt_track=gt_track,
        burn_in=burn_in,
        track_len=track_len,
        max_contextl=max_contextl,
        noise_factor=noise_factor,
    )

    plt.figure()
    if bg_img is not None:
        plt.imshow(bg_img, cmap='gray')
    for track in [gt_track, motion_track, pred_track, sim_track]:
        x, y, theta = track
        plt.plot(x, y, '.', linewidth=2, markersize=2)
    plt.legend(['gt', 'gt motion', 'pred motion', 'simulation'])
    plt.show()

    return gt_track, motion_track, pred_track, sim_track
