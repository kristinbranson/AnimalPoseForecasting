import copy
import numpy as  np
import torch
import logging

from apf.dataset import Dataset, Data, get_post_operations, get_operation, apply_inverse_operations, apply_opers_from_data
from apf.models import TransformerModel
from tqdm.notebook import tqdm, trange

LOG = logging.getLogger(__name__)

from apf.dataset import Fusion
from experiments.flyllm import BodyCentricKP, RawKP

from flyllm.config import featangle
from apf.utils import modrange


def simulate(
    dataset: Dataset,
    model: TransformerModel,
    track: Data,  # TODO: embed this in dataset?
    pose: Data,  # TODO: embed this in dataset?
    identities: np.ndarray,  # TODO: embed this in dataset?
    track_len: int = 1000,
    burn_in: int = 200,
    max_contextl: int = 512,
    agent_ids: list = None,
    start_frame: int = 1000,
    transformed_cache: np.ndarray = None,
    sanity_check_gt_labels: bool = False,
    sanity_check_gt_inputs: bool = False,
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
        agent_ids: Which agents to simulate. If None, simulates all the agents.
        start_frame: Which start frame to use for the initialization.
        transformed_cache: A cache for the model hidden state output, shape n_agents, n_frames, n_layers, d_model.

    Returns:
        gt_track: ground truth 2d track at each frame. Used for first burn_in frames.
        pred_track: 2d track at each frame based on open loop simulation.
    """
    if agent_ids is None:
        n_agents = track.array.shape[0]
        agent_ids = np.arange(n_agents)
    n_sim_agents = len(agent_ids)

    # Extract ground truth
    gt_input = []
    gt_labels = []
    gt_labels_discrete = []
    agent_identities = []
    for agent_idx in agent_ids:
        # Get data input chunk for this agent
        gt_chunk = dataset.get_chunk(start_frame=start_frame, duration=track_len, agent_id=agent_idx)
        gt_input.append(gt_chunk['input'])
        if 'labels' in gt_chunk:
            gt_labels.append(gt_chunk['labels'])
        if 'labels_discrete' in gt_chunk:
            gt_labels_discrete.append(gt_chunk['labels_discrete'])
        # Get agent identity (to be used for inverting pose which requires agent scale)
        agent_identity = np.unique(identities[start_frame:start_frame + track_len, agent_idx])
        assert len(agent_identity) == 1, f"Too many individual ids: {agent_identity}"
        agent_identities.append(agent_identity[0])
    gt_input = np.array(gt_input)
    gt_labels = np.array(gt_labels)
    gt_labels_discrete = np.array(gt_labels_discrete)
    gt_track = track.array[:, start_frame:start_frame + track_len]
    gt_pose = pose.array[:, start_frame:start_frame + track_len]

    # Initialize model input
    device = next(model.parameters()).device
    model_input = torch.zeros((n_sim_agents, track_len, gt_input.shape[-1]))
    curr_frame = burn_in
    model_input[:, :curr_frame, :] = torch.from_numpy(gt_input[:, :curr_frame])
    model_input = model_input.to(device)

    # Initialize tracks (set the future to nan for all agents to be simulated)
    pred_track = copy.deepcopy(gt_track)
    pred_track[agent_ids, curr_frame + 1:] = np.nan
    pred_pose = copy.deepcopy(gt_pose)
    pred_pose[agent_ids, curr_frame + 1:] = np.nan

    return_hidden = transformed_cache is not None

    masksizeprev = 0
    model.eval()
    for curr_frame in trange(curr_frame, track_len, desc='Simulating', leave=False):
        # Make a motion prediction
        frame0 = 0
        if max_contextl is not None:
            frame0 = max(0, curr_frame - max_contextl)

        masksize = curr_frame - frame0
        if masksize != masksizeprev:
            mask = torch.nn.Transformer.generate_square_subsequent_mask(masksize, device=device)
            masksizeprev = masksize

        # Apply model to previous frames
        if sanity_check_gt_labels:
            pred = {}
            if 'labels' in gt_chunk:
                pred['continuous'] = torch.from_numpy(gt_labels[:, frame0:curr_frame]).to(device)
            if 'labels_discrete' in gt_chunk:
                pred['discrete'] = torch.from_numpy(gt_labels_discrete[:, frame0:curr_frame]).to(device)
        else:
            with torch.no_grad():
                if sanity_check_gt_inputs:
                    curr_input = torch.from_numpy(gt_input[:, frame0:curr_frame, :]).to(device)
                else:
                    curr_input = model_input[:, frame0:curr_frame, :]
                pred = model.output(curr_input, mask=mask, is_causal=True,
                                    return_hidden=return_hidden)
                # pred['continuous'] = torch.from_numpy(gt_labels[:, frame0:curr_frame]).to(device)
                # pred['discrete'] = torch.from_numpy(gt_labels_discrete[:, frame0:curr_frame]).to(device)

        if return_hidden:
            for layer_id, out in enumerate(pred['hidden']):
                transformed_cache[layer_id, agent_ids, start_frame+curr_frame-1, :] = out[:, -1]

        # Extract velocity from the prediction
        proc_velocity = dataset.split_output_by_names(pred)['velocity'][:, -1:, :]

        # Invert preprocessing operations to obtain raw velocity
        velocity_operations = dataset.labels['velocity'].operations
        preproc_opers = get_post_operations(velocity_operations, 'velocity')
        velocity = apply_inverse_operations(proc_velocity, preproc_opers)

        # Apply velocity to current pose
        curr_pose = pred_pose[agent_ids, curr_frame - 1]
        velocity_op = get_operation(velocity_operations, 'velocity')
        velocity = np.concatenate([velocity, np.zeros_like(velocity)], axis=1)

        # if isinstance(velocity_op, Fusion):
        #     # Handle the different type of velocity where predicted pose is not velocity but pose itself
        #     # Note: A smarter thing would have been to write a new operation where this is handled,
        #     #       rather than use Fusion
        #     global_ids = velocity_op.indices_per_op[0]
        #     local_ids = velocity_op.indices_per_op[1]
        #     velocity[:, -1, local_ids] = velocity[:, 0, local_ids]
        #     velocity[:, 0, local_ids] = curr_pose[..., local_ids]
        #     # print(velocity)
        #     new_pose = velocity_op.invert(velocity,
        #                                   [{'x0': curr_pose[..., global_ids]}, {}])[:, -1, :]
        #     #new_pose[..., featangle] = modrange(new_pose[..., featangle], -np.pi, np.pi)
        #     # print()
        #     # print(gt_pose[agent_ids, curr_frame])
        #     # print()
        #     # print(new_pose)
        # else:
        new_pose = velocity_op.invert(velocity, x0=curr_pose)[:, -1, :]
        pred_pose[agent_ids, curr_frame] = new_pose
        
        if np.isnan(new_pose).sum() > 0:
            LOG.error(f"Predicted pose contains a NaN, aborting at frame {curr_frame}.")
            break

        # Map pose to keypoints
        pose_op = get_operation(velocity_operations, 'pose')
        if pose_op is None:
            keypoints = pred_pose[agent_ids, curr_frame, :]
        else:
            if isinstance(pose_op, BodyCentricKP) or isinstance(pose_op, RawKP):
                keypoints = pose_op.invert(pred_pose[agent_ids, curr_frame:curr_frame+1, :])[:, 0]
            else:
                keypoints = pose_op.invert(pred_pose[agent_ids, curr_frame, :], agent_identities)
        pred_track[agent_ids, curr_frame] = keypoints

        # Compute sensory information
        sensory_op = get_operation(dataset.inputs['sensory'].operations, 'sensory')
        sensory = sensory_op.apply(pred_track[:, curr_frame:curr_frame+1])

        # Assemble inputs and apply the same preproc operations to the data as the training data
        inputs = {'velocity': velocity[:, :1, :],
                  'pose': pred_pose[agent_ids, None, curr_frame],
                  'sensory': sensory[agent_ids]
                  }
        inputs_proc = apply_opers_from_data(dataset.inputs, inputs)
        curr_in = np.concatenate(list(inputs_proc.values()), axis=-1)
        # return gt_input[:, curr_frame], curr_in
        model_input[:, curr_frame, :] = torch.from_numpy(curr_in[:, 0, :].astype(np.float32)).to(device)

    return gt_track, pred_track
