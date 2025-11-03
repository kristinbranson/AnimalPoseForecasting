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
    agent_ids: list[int] | None = None,
    start_frame: int = 1000,
    debug_with_gt: bool = False,
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
    agent_identities = []
    for agent_idx in agent_ids:
        # Get data input chunk for this agent
        gt_chunk = dataset.get_chunk(start_frame=start_frame, duration=track_len, agent_id=agent_idx)
        gt_input.append(gt_chunk['input'])
        gt_labels.append(gt_chunk['labels'])
        # Get agent identity (to be used for inverting pose which requires agent scale)
        agent_identity = np.unique(identities[start_frame:start_frame + track_len, agent_idx])
        assert len(agent_identity) == 1, f"Too many individual ids: {agent_identity}"
        agent_identities.append(agent_identity[0])
    gt_input = np.array(gt_input)
    gt_labels = torch.from_numpy(np.array(gt_labels))
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

    # print(pred_pose[agent_ids, curr_frame-1, :])

    new_experiment = not hasattr(model, 'output')

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

        if debug_with_gt:
            pred = {'continuous': gt_labels[:, curr_frame:curr_frame+1, :]}
        else:
            # Apply model to previous frames
            with torch.no_grad():
                if hasattr(model, 'output'):
                    pred = model.output(model_input[:, frame0:curr_frame, :], mask=mask, is_causal=True)
                else:
                    pred = model(model_input[:, frame0:curr_frame, :], mask=mask, is_causal=True)

        # Extract velocity from the prediction
        proc_velocity = dataset.split_output_by_names(pred)['velocity'][:, -1:, :]

        # Invert preprocessing operations to obtain raw velocity
        velocity_operations = dataset.labels['velocity'].operations
        preproc_opers = get_post_operations(velocity_operations, 'velocity')
        velocity = apply_inverse_operations(proc_velocity, preproc_opers)

        # Apply velocity to current pose
        curr_pose = pred_pose[agent_ids, curr_frame - 1]
        velocity_op = get_operation(velocity_operations, 'velocity')
        # Note: setting velocity to zero in the last dimension fails for the new fusion velocity where the local
        # keypoints use the identity operation.
        # TODO: See whether filling in with velocity will work for the old data
        #   (really, this should be handled in the velocity operations themselves)
        velocity = np.concatenate([velocity, velocity], axis=1)
        new_pose = velocity_op.invert(velocity, x0=curr_pose)[:, -1, :]
        pred_pose[agent_ids, curr_frame] = new_pose

        if np.isnan(new_pose).sum() > 0:
            LOG.error(f"Predicted pose contains a NaN, aborting at frame {curr_frame}.")
            break

        # Map pose to keypoints
        if new_experiment:
            pose_op = get_operation(velocity_operations, 'bodycentrickp')
            keypoints = pose_op.invert(pred_pose[agent_ids, curr_frame:curr_frame+1, :])[:, 0]
        else:
            pose_op = get_operation(velocity_operations, 'pose')
            if pose_op is None:
                keypoints = pred_pose[agent_ids, curr_frame, :]
            else:
                keypoints = pose_op.invert(pred_pose[agent_ids, curr_frame, :], agent_identities)
        pred_track[agent_ids, curr_frame] = keypoints

        # print("proc_velocity")
        # print(proc_velocity)
        # print('velocity')
        # print(velocity)
        # print('curr_pose')
        # print(curr_pose)
        # print('new_pose')
        # print(new_pose)
        # print('keypoints')
        # print(keypoints)

        # Compute sensory information
        sensory_op = get_operation(dataset.inputs['sensory'].operations, 'sensory')
        sensory = sensory_op.apply(pred_track[:, curr_frame:curr_frame+1])

        # Assemble inputs and apply the same preproc operations to the data as the training data
        if new_experiment:
            inputs = {'velocity': velocity[:, :1, :],
                      'sensory': sensory[agent_ids]
                      }
        else:
            inputs = {'velocity': velocity[:, :1, :],
                      'pose': pred_pose[agent_ids, None, curr_frame],
                      'sensory': sensory[agent_ids]
                      }
        inputs_proc = apply_opers_from_data(dataset.inputs, inputs)
        curr_in = np.concatenate(list(inputs_proc.values()), axis=-1)
        model_input[:, curr_frame, :] = torch.from_numpy(curr_in[:, 0, :].astype(np.float32)).to(device)

        curr_frame += 1

    return gt_track, pred_track


def simulate_new(
    dataset: Dataset,
    model: TransformerModel,
    track: Data,  # TODO: embed this in dataset?
    pose: Data,  # TODO: embed this in dataset?
    identities: np.ndarray,  # TODO: embed this in dataset?
    track_len: int = 1000,
    burn_in: int = 200,
    max_contextl: int = 512,
    agent_ids: list[int] | None = None,
    start_frame: int = 1000,
    debug_with_gt: bool = False,
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
    agent_identities = []
    for agent_idx in agent_ids:
        # Get data input chunk for this agent
        gt_chunk = dataset.get_chunk(start_frame=start_frame, duration=track_len, agent_id=agent_idx)
        gt_input.append(gt_chunk['input'])
        gt_labels.append(gt_chunk['labels'])
        # Get agent identity (to be used for inverting pose which requires agent scale)
        agent_identity = np.unique(identities[start_frame:start_frame + track_len, agent_idx])
        assert len(agent_identity) == 1, f"Too many individual ids: {agent_identity}"
        agent_identities.append(agent_identity[0])
    gt_input = np.array(gt_input)
    gt_labels = torch.from_numpy(np.array(gt_labels))
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

        if debug_with_gt:
            pred = {'continuous': gt_labels[:, curr_frame:curr_frame+1, :]}
        else:
            # Apply model to previous frames
            with torch.no_grad():
                pred = model(model_input[:, frame0:curr_frame, :], mask=mask, is_causal=True)

        # Extract velocity from the prediction
        proc_pose = dataset.split_output_by_names(pred)['bodycentrickp'][:, -1:, :]

        # Invert preprocessing operations to obtain raw velocity
        pose_operations = dataset.labels['bodycentrickp'].operations
        post_process_opers = get_post_operations(pose_operations, 'bodycentrickp')
        bodycentric_oper = get_operation(pose_operations, 'bodycentrickp')

        new_pose = apply_inverse_operations(proc_pose, post_process_opers)
        keypoints = bodycentric_oper.invert(new_pose)

        if np.isnan(new_pose).sum() > 0:
            LOG.error(f"Predicted pose contains a NaN, aborting at frame {curr_frame}.")
            break

        pred_pose[agent_ids, curr_frame] = new_pose[:, 0]
        pred_track[agent_ids, curr_frame] = keypoints[:, 0]

        # Compute sensory information
        sensory_op = get_operation(dataset.inputs['sensory'].operations, 'sensory')
        sensory = sensory_op.apply(pred_track[:, curr_frame:curr_frame+1])

        # Assemble inputs and apply the same preproc operations to the data as the training data
        inputs = {'bodycentrickp': new_pose,
                  'sensory': sensory[agent_ids]
                  }
        inputs_proc = apply_opers_from_data(dataset.inputs, inputs)
        curr_in = np.concatenate(list(inputs_proc.values()), axis=-1)
        model_input[:, curr_frame, :] = torch.from_numpy(curr_in[:, 0, :].astype(np.float32)).to(device)

        curr_frame += 1

    return gt_track, pred_track


def simulate_new2(
    dataset: Dataset,
    model: TransformerModel,
    track: Data,  # TODO: embed this in dataset?
    pose: Data,  # TODO: embed this in dataset?       # NOTE: Here this is global pos
    identities: np.ndarray,  # TODO: embed this in dataset?
    track_len: int = 1000,
    burn_in: int = 200,
    max_contextl: int = 512,
    agent_ids: list[int] | None = None,
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
        agent_ids: Which agents to simulate. If None, simulates all the agents.
        start_frame: Which start frame to use for the initialization.

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
    agent_identities = []
    for agent_idx in agent_ids:
        # Get data input chunk for this agent
        gt_chunk = dataset.get_chunk(start_frame=start_frame, duration=track_len, agent_id=agent_idx)
        gt_input.append(gt_chunk['input'])
        # Get agent identity (to be used for inverting pose which requires agent scale)
        agent_identity = np.unique(identities[start_frame:start_frame + track_len, agent_idx])
        assert len(agent_identity) == 1, f"Too many individual ids: {agent_identity}"
        agent_identities.append(agent_identity[0])
    gt_input = np.array(gt_input)
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
        named_pred = dataset.split_output_by_names(pred)
        proc_global_vel = named_pred['global_vel'][:, -1, :]
        proc_local_kpts = named_pred['local_kpts'][:, -1, :]

        # Invert preprocessing operations to obtain raw velocity
        global_vel_operations = dataset.labels['global_vel'].operations
        post_process_opers = get_post_operations(global_vel_operations, 'globalvelocity')
        global_vel_oper = get_operation(global_vel_operations, 'globalvelocity')

        global_vel = apply_inverse_operations(proc_global_vel, post_process_opers)
        global_vel = np.concatenate([global_vel, np.zeros_like(global_vel)], axis=1)
        global_pos = global_vel_oper.invert(global_vel, x0=pred_pose[agent_ids, curr_frame-1])[:, -1, :]

        # Invert preprocessing operations to obtain raw local keypoints
        local_kpts_operations =  dataset.labels['local_kpts'].operations
        post_process_opers = get_post_operations(local_kpts_operations, 'split')
        local_kpts = apply_inverse_operations(local_kpts_operations, post_process_opers)
        split_oper = get_operation(post_process_opers, 'split')
        body_centric_oper = get_operation(local_kpts_operations, 'bodycentrickp')

        body_centric_kp = split_oper.invert({'local_kpts': local_kpts, 'global_pos': global_pos})
        keypoints = body_centric_oper.invert(body_centric_kp)

        if np.isnan(body_centric_kp).sum() > 0:
            LOG.error(f"Predicted pose contains a NaN, aborting at frame {curr_frame}.")
            break

        pred_pose[agent_ids, curr_frame] = global_pos
        pred_track[agent_ids, curr_frame] = keypoints

        # Compute sensory information
        sensory_op = get_operation(dataset.inputs['sensory'].operations, 'sensory')
        sensory = sensory_op.apply(pred_track[:, curr_frame:curr_frame+1])

        # Assemble inputs and apply the same preproc operations to the data as the training data
        # TODO: Ensure that the dimensions of all the data is as expected (especially velocity)
        inputs = {'local_kpts': local_kpts,
                  'global_vel': global_vel,
                  'sensory': sensory[agent_ids]
                  }
        inputs_proc = apply_opers_from_data(dataset.inputs, inputs)
        curr_in = np.concatenate(list(inputs_proc.values()), axis=-1)
        model_input[:, curr_frame, :] = torch.from_numpy(curr_in[:, 0, :].astype(np.float32)).to(device)

        curr_frame += 1

    return gt_track, pred_track
