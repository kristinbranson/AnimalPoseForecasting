import copy
import numpy as  np
import torch
import matplotlib.pyplot as plt
from typing import Callable

from apf.dataset import get_post_operations, get_operation, apply_inverse_operations, apply_opers_from_data
from apf.utils import rotate_2d_points, modrange
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
    return x + dx, y + dy, modrange(theta + d_theta, -np.pi, np.pi)


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
    dataset,
    model,
    track, # TODO: embed this in dataset?
    pose, # TODO: embed this in dataset?
    flyids, # TODO: embed this in dataset?
    track_len: int = 1000,
    burn_in: int = 200,
    max_contextl: int = 512,
    agent_id: int = 0,
    start_frame: int = 1000,
):
    # Extract ground truth
    gt_chunk = dataset.get_chunk(start_frame=start_frame, duration=track_len, agent_id=agent_id)
    gt_input = gt_chunk['input']

    gt_track = track.array[:, start_frame:start_frame + track_len]
    gt_pose = pose.array[:, start_frame:start_frame + track_len]
    flyid = np.unique(flyids[start_frame:start_frame + track_len, agent_id])
    assert len(flyid) == 1, f"Too many flyids: {flyid}"
    flyid = flyid[0]

    # Initialize model input
    device = next(model.parameters()).device
    model_input = torch.zeros((1, track_len, gt_input.shape[-1]))
    curr_frame = burn_in
    model_input[0, :curr_frame, :] = torch.from_numpy(gt_input[:curr_frame])
    model_input = model_input.to(device)

    # Initialize track
    pred_track = copy.deepcopy(gt_track)
    pred_track[agent_id, curr_frame:] = np.nan
    pred_pose = copy.deepcopy(gt_pose)
    pred_pose[agent_id, curr_frame:] = np.nan

    # Simulate
    model.eval()
    velocities = []
    while curr_frame < track_len:
        # Make a motion prediction
        frame0 = 0
        if max_contextl is not None:
            frame0 = max(0, curr_frame - max_contextl)

        # Apply model to previous frames
        pred = model.output(model_input[:, frame0:curr_frame, :])
        # pred = {
        #         'continuous': torch.from_numpy(gt_chunk['labels'][None, frame0:curr_frame, :]),
        #         'discrete': torch.from_numpy(gt_chunk['labels_discrete'][None, frame0:curr_frame, :])
        #        }

        # Extract velocity from the prediction
        proc_velocity = dataset.split_output_to_labels(pred)['velocity'][:, -1:, :]
        # proc_velocity = pred['continuous'][:, -1].detach().cpu().numpy()

        # Invert preprocessing operations to obtain raw velocity
        velocity_operations = dataset.labels['velocity'].operations
        preproc_opers = get_post_operations(velocity_operations, 'velocity')
        velocity = apply_inverse_operations(proc_velocity, preproc_opers)

        velocities.append(velocity[0, 0])

        # Apply velocity to current pose (TODO: make this the inverse velocity operation)
        curr_pose = pred_pose[agent_id, curr_frame-1]
        velocity_op = get_operation(velocity_operations, 'velocity')
        velocity = np.concatenate([velocity, np.zeros_like(velocity)], axis=1)
        new_pose = velocity_op.inverse(velocity, x0=curr_pose[None, :])[:, -1:, :]
        pred_pose[agent_id, curr_frame] = new_pose

        if np.isnan(new_pose).sum() > 0:
            print(np.isnan(new_pose[0]))
            break

        # Map pose to keypoints
        pose_op = get_operation(velocity_operations, 'pose')
        keypoints = pose_op.inverse(pred_pose[agent_id, curr_frame, :], flyid)
        pred_track[agent_id, curr_frame] = keypoints

        # Compute sensory information
        sensory_op = get_operation(dataset.inputs['sensory'].operations, 'sensory')
        sensory = sensory_op.apply(pred_track[:, curr_frame:curr_frame+1])

        # Assemble inputs and apply the same preproc operations to the data as the training data
        inputs = {'sensory': sensory[agent_id],
                  'pose': pred_pose[agent_id, curr_frame],
                  'velocity': velocity[0, :1, :]}
        inputs_proc = apply_opers_from_data(dataset.inputs, inputs)
        curr_in = np.concatenate(list(inputs_proc.values()), axis=-1)
        model_input[:, curr_frame, :] = torch.from_numpy(curr_in.astype(np.float32)).to(device)

        curr_frame += 1

    # n = 1000
    # x, y = gt_pose[agent_id, :n, :2].T
    # plt.figure(figsize=(7, 7))
    # plot_arena()
    # plt.plot(x, y, '-g', linewidth=5)
    # x, y = pred_pose[agent_id, :n, :2].T
    # plt.plot(x, y, '-r', linewidth=2)
    # plt.axis('equal')
    # plt.show()
    return pred_track


# def simulate_old(
#     model: TransformerModel,
#     motion_operation: Operation,
#     motion_labels_operation: Operation,
#     observation_operation: Operation,
#     compute_observation: Callable,
#     gt_input: torch.Tensor,
#     gt_track: np.ndarray,
#     burn_in: int = 30,
#     track_len: int = 1000,
#     max_contextl: int | None = 64,
#     noise_factor: float = 0,
# ) -> np.ndarray:
#     """ Simulates an agent given model and some initialization.
#
#     TODO: compare with Kristin's sliding window prediction in test_dataset
#
#     Args:
#         model: Transformer model used for predicting future motion.
#         motion_operation: Operation to be applied to motion input data (e.g. zscore).
#         observation_operation: Operation to be applied to observation input data (e.g. zscore).
#         motion_labels_operation: Operation that was applied to motion label data for training (e.g. zscore or binning).
#         compute_observation: Function for computing observation given current state
#         gt_input: (batch_size, n_frames, d_input) ground truth input to the model. Used for burning in the simulation.
#         gt_track: (3, n_frames) ground truth 2d pose at each frame. Used for first burn_in frames.
#         burn_in: How many frames to apply the model to gt_input before simulating.
#         track_len: How long the total track should be after simulation.
#         max_contextl: Max number of frames to feed as input to the model. If None, uses the full history.
#         noise_factor: How much noise to add to prediction (only used for continuous output)
#             (helpful for getting non-deterministic tracks for continuous motion output)
#
#     Returns:
#         sim_track: (3, track_len) 2d pose at each frame based on open loop simulation.
#     """
#     if burn_in > 0:
#         assert gt_track is not None and gt_input is not None
#
#     # Initialize model input
#     device = next(model.parameters()).device
#     model_input = torch.zeros((1, track_len, gt_input.shape[-1]))
#     curr_frame = burn_in
#     model_input[0, :curr_frame, :] = gt_input[:curr_frame]
#     model_input = model_input.to(device)
#
#     # Initialize track
#     track = np.zeros((3, track_len))
#     track[:, :curr_frame] = gt_track[:, :curr_frame]
#     track[:, curr_frame:] = 0
#
#     # Currently only handling all discrete or all continuous
#     key = 'discrete' if model.is_mixed else 'continuous'
#
#     model.eval()
#     while curr_frame < track_len:
#         # Make a motion prediction
#         frame0 = 0
#         if max_contextl is not None:
#             frame0 = max(0, curr_frame - max_contextl)
#         pred = model.output(model_input[:, frame0:curr_frame, :])[key]
#         pred_np = pred.detach().cpu().numpy()[0, -1, :]
#
#         # Map prediction to raw data format
#         if noise_factor > 0 and not model.is_mixed:
#             pred_np += np.random.randn(pred_np.shape[-1]) * noise_factor
#         pred_np = motion_labels_operation.inverse(pred_np[None, None, ...])[0, ...]
#
#         # Update position of mouse
#         d_fwd, d_side, d_theta = pred_np.flat
#         x_prev, y_prev, theta_prev = track[:, curr_frame - 1]
#         x, y, theta = apply_motion(x_prev, y_prev, theta_prev, d_fwd, d_side, d_theta)
#
#         # Compute observation
#         observation = compute_observation(np.array([x, y, theta])[:, None, None])
#
#         # Assemble the next input
#         next_input = np.array([d_fwd, d_side, d_theta] + list(observation.flat))
#         next_input[:3] = motion_operation.apply(next_input[None, None, :3]).flat
#         next_input[3:] = observation_operation.apply(next_input[None, None, 3:]).flat
#
#         model_input[0, curr_frame, :] = torch.from_numpy(next_input)
#         track[:, curr_frame] = [x, y, theta]
#
#         curr_frame += 1
#
#     return track


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
