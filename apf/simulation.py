import copy
import numpy as  np
import torch
import logging

from apf.dataset import Dataset, Data, get_post_operations, get_operation, apply_inverse_operations, apply_opers_from_data
from apf.models import TransformerModel

LOG = logging.getLogger(__name__)


def simulate(
    dataset: Dataset,
    model: TransformerModel,
    track: Data,  # TODO: embed this in dataset?
    pose: Data,  # TODO: embed this in dataset?
    identities: np.ndarray,  # TODO: embed this in dataset?
    track_len: int = 1000,
    burn_in: int = 200,
    max_contextl: int = 512,
    agent_idx: int = 0,
    start_frame: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """ Simulates an agent given model and some initialization.

    Args:
        dataset: Dataset that defines which operations need to be applied to the data, also used for burn-in.
        model: Transformer model used for predicting future motion.
        track: Track array corresponding to dataset (used for ground truth comparison). TODO: Can we make this optional?
        pose: Pose array corresponding to dataset (used for ground truth comparison). TODO: Can we make this optional?
        identities: Maps frame and agent id to a unique individual id. Corresponds to dataset.
        track_len: How long the total track should be after simulation (including burnin).
        burn_in: How many frames to use for the initialization.
        max_contextl: Max number of frames to feed as input to the model. If None, uses the full history.
        agent_idx: Which agent to use for initialization.
        start_frame: Which start frame to use for the initialization.

    Returns:
        gt_track: ground truth 2d track at each frame. Used for first burn_in frames.
        pred_track: 2d track at each frame based on open loop simulation.
    """
    # Extract ground truth
    gt_chunk = dataset.get_chunk(start_frame=start_frame, duration=track_len, agent_id=agent_idx)
    gt_input = gt_chunk['input']

    gt_track = track.array[:, start_frame:start_frame + track_len]
    gt_pose = pose.array[:, start_frame:start_frame + track_len]
    agent_identity = np.unique(identities[start_frame:start_frame + track_len, agent_idx])
    assert len(agent_identity) == 1, f"Too many individual ids: {agent_identity}"
    agent_identity = agent_identity[0]

    # Initialize model input
    device = next(model.parameters()).device
    model_input = torch.zeros((1, track_len, gt_input.shape[-1]))
    curr_frame = burn_in
    model_input[0, :curr_frame, :] = torch.from_numpy(gt_input[:curr_frame])
    model_input = model_input.to(device)

    # Initialize track
    pred_track = copy.deepcopy(gt_track)
    pred_track[agent_idx, curr_frame + 1:] = np.nan
    pred_pose = copy.deepcopy(gt_pose)
    pred_pose[agent_idx, curr_frame + 1:] = np.nan

    masksizeprev = 0
    model.eval()
    while curr_frame < track_len:
        # Make a motion prediction
        frame0 = 0
        if max_contextl is not None:
            frame0 = max(0, curr_frame - max_contextl)

        masksize = curr_frame - frame0
        if masksize != masksizeprev:
            mask = torch.nn.Transformer.generate_square_subsequent_mask(masksize, device=device)
            masksizeprev = masksize

        # Apply model to previous frames
        with torch.no_grad():
            pred = model.output(model_input[:, frame0:curr_frame, :], mask=mask, is_causal=True)

        # Extract velocity from the prediction
        proc_velocity = dataset.split_output_by_names(pred)['velocity'][:, -1:, :]

        # Invert preprocessing operations to obtain raw velocity
        velocity_operations = dataset.labels['velocity'].operations
        preproc_opers = get_post_operations(velocity_operations, 'velocity')
        velocity = apply_inverse_operations(proc_velocity, preproc_opers)

        # Apply velocity to current pose
        curr_pose = pred_pose[agent_idx, curr_frame - 1]
        velocity_op = get_operation(velocity_operations, 'velocity')
        velocity = np.concatenate([velocity, np.zeros_like(velocity)], axis=1)
        new_pose = velocity_op.invert(velocity, x0=curr_pose[None, :])[:, -1:, :]
        pred_pose[agent_idx, curr_frame] = new_pose

        if np.isnan(new_pose).sum() > 0:
            LOG.error(f"Predicted pose contains a NaN, aborting at frame {curr_frame}.")
            break

        # Map pose to keypoints
        pose_op = get_operation(velocity_operations, 'pose')
        if pose_op is None:
            keypoints = pred_pose[agent_idx, curr_frame, :]
        else:
            keypoints = pose_op.invert(pred_pose[agent_idx, curr_frame, :], agent_identity)
        pred_track[agent_idx, curr_frame] = keypoints

        # Compute sensory information
        sensory_op = get_operation(dataset.inputs['sensory'].operations, 'sensory')
        sensory = sensory_op.apply(pred_track[:, curr_frame:curr_frame+1])

        # Assemble inputs and apply the same preproc operations to the data as the training data
        inputs = {'velocity': velocity[:1, :1, :],
                  'pose': pred_pose[agent_idx, curr_frame],
                  'sensory': sensory[agent_idx]
                  }
        inputs_proc = apply_opers_from_data(dataset.inputs, inputs)
        curr_in = np.concatenate(list(inputs_proc.values()), axis=-1)
        model_input[:, curr_frame, :] = torch.from_numpy(curr_in.astype(np.float32)).to(device)

        curr_frame += 1

    return gt_track, pred_track
