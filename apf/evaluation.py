
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from tqdm.notebook import tqdm
from scipy.stats import entropy
from scipy import ndimage

from apf.io import load_config_from_model_file, load_model
from apf.simulation import simulate
from apf.dataset import Dataset, Fusion, apply_opers_from_data
from apf.models import initialize_model, MLP
from apf.plotting import fig_to_numpy, arrays_to_video, convert_video_to_mp4
from experiments.flyllm import Sensory, Pose, Velocity
from flyllm.prepare import init_datasets, init_raw_data
from flyllm.config import (read_config, featrelative, featangle, skeleton_edges,
                           keypointnames, PXPERMM, ARENA_RADIUS_PX, ARENA_RADIUS_MM)
from flyllm.features import body_centric_kp
from flyllm.plotting import plot_arena #plot_frame
from experiments.flyllm import BodyCentricKP, RawKP


def get_multi_sessions(dataset, min_duration=None):
    key = list(dataset.inputs.keys())[0]
    n_flies, n_frames, _ = dataset.inputs[key].array.shape
    session_frames = np.zeros((n_flies, n_frames), int)
    for i, session in enumerate(dataset.sessions):
        session_frames[session.agent_id, session.start_frame:session.start_frame + session.duration] = i #i+1

    # Find continuous chunks of frames where all valid session ids are same per fly
    diffs = np.zeros_like(session_frames)
    diffs[:, 1:] = np.abs(np.diff(session_frames, axis=1))
    start_frames = np.where(diffs.max(0) > 0)[0]

    sessions = []
    for i in range(len(start_frames) - 1):
        start_frame = start_frames[i]
        end_frame = start_frames[i + 1]
        duration = end_frame - start_frame
        if min_duration is not None and duration < min_duration:
            continue
        agent_ids = np.where(session_frames[:, start_frame] > 0)[0]
        if len(agent_ids) == 0:
            continue
        sessions.append((start_frame, duration, agent_ids))
    return sessions



def plot_frame(track, frame_id, agent_ids, kp=7, show_spotlight=True, is_female=None):
    plot_arena()
    x, y = track[:, frame_id, :, :].transpose([1, 2, 0])  # n_keypoints x n_agents
    if is_female is not None:
        female = is_female[agent_ids, frame_id] == 1
        f_inds = agent_ids[np.where(female)[0]]
        m_inds = agent_ids[np.where(~female)[0]]
    for edge in skeleton_edges:
        # Plot all agents in grey (wings in cyan)
        lw = 0.8
        alpha = 1
        linestyle = '-'
        if 0 in edge or 1 in edge:
            # lw = 1.5
            # alpha = 0.5
            linestyle = '--'
        plt.plot(x[edge, :], y[edge, :], linestyle, color=.3*np.ones(3), linewidth=lw, alpha=alpha)
        # Plot specified agents in black and with a dot
        if is_female is None:
            plt.plot(x[edge][:, agent_ids], y[edge][:, agent_ids], linestyle + 'k', linewidth=lw, alpha=alpha)
        else:
            plt.plot(x[edge][:, m_inds], y[edge][:, m_inds], linestyle + 'k', linewidth=lw, alpha=alpha)
    # plt.plot(x[kp:kp+1, agent_ids], y[kp:kp+1, agent_ids], '.', markersize=8)
    if show_spotlight:
        if is_female is None:
            plt.plot(x[kp:kp+1, agent_ids], y[kp:kp+1, agent_ids], 'o', markeredgecolor='none', markersize=60, alpha=0.2)
        else:
            plt.scatter(x[kp, f_inds], y[kp, f_inds], c=np.arange(len(f_inds)), s=3500, alpha=0.2, cmap='spring', clim=[0, len(f_inds)], edgecolors='none')
            plt.scatter(x[kp, m_inds], y[kp, m_inds], c=np.arange(len(m_inds)), s=3500, alpha=0.2, cmap='winter', clim=[0, len(m_inds)], edgecolors='none')


def make_dataset_video(dataset, track, resultsdir, video_name, is_female=None):
    sessions = get_multi_sessions(dataset)
    n_show = 10
    indices = np.arange(80, len(sessions), len(sessions) // n_show)

    rgbs = []
    for idx in tqdm(indices):
        first_frame, duration, agent_ids = sessions[idx]
        for frame_id in tqdm(range(first_frame, first_frame + duration), leave=False):
            fig = plt.figure(figsize=(10, 10))
            plot_frame(track.array, frame_id, agent_ids, is_female=is_female)
            plt.axis('equal')
            plt.axis('off')
            plt.text(-ARENA_RADIUS_MM+1, ARENA_RADIUS_MM-2,
                     f"session: {idx}, frame: {frame_id - first_frame}", fontsize=12)
            fig.tight_layout()
            # plt.show()
            # break

            rgbs.append(fig_to_numpy(fig))
            plt.close()
            del(fig)

    output_path = os.path.join(resultsdir, f'{video_name}.avi')
    arrays_to_video(rgbs, output_path, fps=30, codec='png ', color_format='rgb')
    convert_video_to_mp4(output_path)


import glob
def load_sim_track(gt_track, sim_savedir, contextl):
    """
    Parameters
        gt_track: gt_track.array is a float32 of shape (n_flies, n_frames, 2, n_keypoints)
        sim_savedir: Directory where tracklets simulated from gt_track are stored
        contextl: Context length that was used for the simulation

    Returns
         sim_track:
         sim_frame:
    """
    sim_track = copy.deepcopy(gt_track)
    sim_frame = np.zeros(sim_track.array.shape[:2])

    num_agents = sim_frame.shape[0]

    files = glob.glob(os.path.join(sim_savedir, '*.npy'))
    for i in tqdm(range(len(files))):
        filepath = files[i]

        # Extract info about the tracklet from the filepath
        idx0 = filepath.index('startsimframe_')
        idx1 = filepath.index('_agentid_')
        idx2 = filepath.index('_simlen_')
        idx3 = filepath.index('.npy')
        startsimframe = int(filepath[idx0 + len('startsimframe_') : idx1])
        agents = [int(val) for val in filepath[idx1+len('_agentid_') : idx2].split('_')]
        simlen = int(filepath[idx2+len('_simlen_') : idx3])

        # Handle the case where the provided track ignores the 11th fly
        # (hack related to the Keypoints model)
        if max(agents) >= num_agents:
            continue

        # Load simulated tracklet
        tracklet = np.load(filepath)

        # Ensure that the first contextl of frames corresponds to the gt_track (the simulation prompt)
        assert (np.allclose(gt_track.array[agents, startsimframe-contextl:startsimframe], tracklet[:, :contextl])), \
            f"{gt_track.array[agents, startsimframe-contextl:startsimframe].shape} vs {tracklet[:, :contextl].shape}"

        sim_track.array[agents, startsimframe:startsimframe+simlen] = tracklet[:, contextl:]
        sim_frame[agents, startsimframe:startsimframe+simlen] = (np.arange(simlen)+1)[None, :]

    return sim_track, sim_frame


def simulate_tracks(model, train_dataset, train_data, savedir, sim_len, stride=None, simulate_single=False, first_sim_frame=None, sim_transformed_path=None):
    """
    Parameters
        model:
        train_dataset:
        train_data:
        savedir:
        sim_len:
        stride:
        simulate_single:
        first_sim_frame: First frame to simulate within vaid session. If None, contextl will be used.

    Returns

    """
    track = train_data['track']
    pose = train_data['pose']
    flyids = train_data['flyids']

    if stride is None:
        stride = sim_len

    contextl = train_dataset.context_length
    if first_sim_frame is None:
        first_sim_frame = contextl

    if simulate_single:
        sessions = []
        for session in train_dataset.sessions:
            if session.duration < sim_len + first_sim_frame + 1:
                continue
            sessions.append((session.start_frame, session.duration, [session.agent_id]))
    else:
        # key = list(train_dataset.inputs.keys())[0]
        # n_flies, n_frames, _ = train_dataset.inputs[key].array.shape
        # session_frames = np.zeros((n_flies, n_frames), int)
        # for i, session in enumerate(train_dataset.sessions):
        #     session_frames[session.agent_id, session.start_frame:session.start_frame + session.duration] = i
        #
        # # Find continuous chunks of frames where all valid session ids are same per fly
        # diffs = np.zeros_like(session_frames)
        # diffs[:, 1:] = np.abs(np.diff(session_frames, axis=1))
        # start_frames = np.where(diffs.max(0) > 0)[0]
        #
        # for i in range(len(start_frames) - 1):
        #     start_frame = start_frames[i]
        #     end_frame = start_frames[i + 1]
        #     duration = end_frame - start_frame
        #     if duration < sim_len + first_sim_frame + 1:
        #         continue
        #     agent_ids = np.where(session_frames[:, start_frame] > 0)[0]
        #     sessions.append((start_frame, duration, agent_ids))

        sessions = get_multi_sessions(train_dataset, min_duration=sim_len + first_sim_frame + 1)

    if sim_transformed_path is not None:
        d_model = model.d_model
        n_layers = len(model.transformer_encoder.layers)
        n_agents, n_frames = train_dataset.inputs['velocity'].array.shape[:2]
        # NOTE: This is very dangerous, only do it like this if an overwrite flag is set
        mode = 'r+' if os.path.exists(sim_transformed_path) else 'w+'
        transformed = np.memmap(sim_transformed_path, dtype=np.float32, mode=mode,
                                shape=(n_layers, n_agents, n_frames, d_model))
    else:
        transformed = None

    # Simulate
    sim_track = copy.deepcopy(train_data['track'])
    sim_frame = np.zeros(sim_track.array.shape[:2])
    for i, session in tqdm(enumerate(sessions)):
        session_start, session_duration, agent_ids = session
        # start_frames = np.arange(session_start, session_start + session_duration - track_len, stride)
        start_sim_frames = np.arange(session_start + first_sim_frame, session_start + session_duration - sim_len, stride)

        agent_str = str(agent_ids[0])
        for a in range(1, len(agent_ids)):
            agent_str = agent_str + f"_{agent_ids[a]}"

        for start_sim_frame in tqdm(start_sim_frames, leave=False):
            save_str = f"session_{i}_startsimframe_{start_sim_frame}_agentid_{agent_str}_simlen_{sim_len}.npy"
            save_path = os.path.join(savedir, save_str)

            if os.path.exists(save_path):
                pred_track = np.load(save_path)
            else:
                print(f"Warning: {save_path} does not exist")
                continue
                gt_track, pred_track = simulate(
                    dataset=train_dataset,
                    model=model,
                    track=track,
                    pose=pose,
                    identities=flyids,
                    track_len=sim_len + contextl,
                    burn_in=contextl,
                    max_contextl=contextl,
                    agent_ids=agent_ids,
                    start_frame=start_sim_frame - contextl,
                    transformed_cache=transformed,
                )
                pred_track = pred_track[agent_ids]
                np.save(save_path, pred_track)

            sim_track.array[agent_ids, start_sim_frame:start_sim_frame + sim_len] = pred_track[:, contextl:]
            sim_frame[agent_ids, start_sim_frame:start_sim_frame + sim_len] = (np.arange(sim_len) + 1)[None, :]

    return sim_track, sim_frame, transformed


def create_dataset_from_ref_data(ref_dataset, ref_data, sim_track, sim_frame):
    # Valid date only where flies are being simulated
    isdata = copy.deepcopy(ref_data['isdata'])
    isdata[sim_frame.T < 1] = False

    # Start of each data is the first frame of simulation
    isstart = sim_frame.T == 1

    # Copy flyids and scale_perfly from original track
    flyids = copy.deepcopy(ref_data['flyids'])


    # Use all outputs of valid data
    useoutputmask = np.ones(isdata.shape, dtype=bool)

    # Compute features for simulated data
    sensory_op = ref_dataset.inputs['sensory'].operations[0]
    sensory = sensory_op(sim_track, isdata=isdata)

    pose_op = ref_data['pose'].operations[0]
    if isinstance(pose_op, BodyCentricKP) or isinstance(pose_op, RawKP):
        pose = pose_op(sim_track)
        pose.array[~isdata.T] = np.nan
    else:
        scale_perfly = copy.deepcopy(ref_data['pose'].operations[0].scale_perfly)
        pose = pose_op(sim_track, scale_perfly=scale_perfly, flyid=flyids, isdata=isdata)

    assert not np.any(np.isnan(sensory.array[isdata.T])) and np.all(
        np.isnan(sensory.array[~isdata.T])), "Sensory features should be nan iff isdata == False"
    assert not np.any(np.isnan(pose.array[isdata.T])) and np.all(
        np.isnan(pose.array[~isdata.T])), "Pose features should be nan iff isdata == False"

    vel_op = ref_dataset.labels['velocity'].operations[1]
    if isinstance(vel_op, Fusion):
        velocity = vel_op(pose, kwargs_per_op=[{'isstart': isstart}, {}])
    else:
        velocity = vel_op(pose, isstart=isstart)

    metadata = {'labels': {'velocity': {'pose': flyids.T, 'velocity': pose.array}},
                'inputs': {'pose': {'pose': flyids.T},
                           'velocity': {'pose': flyids.T}}}
    args = {
        'context_length': ref_dataset.context_length,
        'isstart': isstart,
        'metadata': metadata,
        'useoutputmask': useoutputmask
    }

    # Assemble the dataset
    sim_dataset = Dataset(
        inputs=apply_opers_from_data(ref_dataset.inputs, {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
        labels=apply_opers_from_data(ref_dataset.labels, {'velocity': velocity}),
        **args
    )

    sim_data = {'track': sim_track,
                'pose': pose,
                'sensory': sensory,
                'velocity': velocity,
                'isstart': isstart,
                'isdata': isdata,
                'sim_frame': sim_frame}

    return sim_dataset, sim_data


def plot_classifier_stats(stats, label, pdf=None):

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'{label}', fontsize=16, fontweight='bold')
    plt.subplot(1, 3, 1)
    plt.plot(stats['losses'])
    plt.title('Losses')
    plt.subplot(1, 3, 2)
    plt.plot(stats['accuracies'])
    plt.title('Accuracies')
    plt.subplot(1, 3, 3)
    precision, recall, thresholds = stats['PR_curve']
    valprecision, valrecall, valthresholds = stats['val_PR_curve']
    idx = np.argmin(np.abs(thresholds-0.5))
    val_idx = np.argmin(np.abs(valthresholds-0.5))
    plt.plot(recall, precision, '.', markersize=2)
    plt.plot(recall[idx], precision[idx], '.k')
    plt.plot(valrecall, valprecision, '.', markersize=2)
    plt.plot(valrecall[val_idx], valprecision[val_idx], '.k')
    fig.tight_layout()
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
        plt.close()


def plot_distributions(distributions, posenames=None, title=None, pdf=None, symlog=False):
    fig = plt.figure(figsize=(15, 10))
    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    for i, (idx, (bin_edges, count, sim_count, kl_div)) in enumerate(distributions.items()):
        ax = plt.subplot(6, 5, i + 1)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        plt.plot(bin_centers, count)
        plt.plot(bin_centers, sim_count)
        plt.yticks([])
        plt.ylabel(f"KL = {kl_div:.3f}")
        if posenames is not None:
            plt.title(posenames[idx])
        if symlog:
            ax.set_xscale('symlog', linthresh=0.01)
    fig.tight_layout()
    return fig


def symlog(x, linthresh=1.0, linscale=1.0, base=10):
    """
    Apply symmetric log (symlog) transformation to data.

    Parameters:
        x        : array-like, input data (can include negatives and zero)
        linthresh: float, range within which the plot is linear (default 1.0)
        linscale : float, scales the linear range relative to the log range (default 1.0)
        base     : float, logarithm base (default 10)

    Returns:
        numpy array of transformed values
    """
    x = np.asarray(x, dtype=float)
    log_base = np.log(base)

    # Scale factor for the linear region
    linear_scale = linscale / log_base

    # Transformation
    abs_x = np.abs(x)
    log_part = np.sign(x) * (
        linear_scale + np.log(abs_x / linthresh) / log_base
    )
    linear_part = x * (linear_scale / linthresh)

    return np.where(abs_x > linthresh, log_part, linear_part)


def compute_feature_distributions(train_feat, sim_feat, use_log=False, n_bins=1000, feat_ids=None):

    # Remove nans
    train_feat = train_feat[~np.isnan(train_feat).max(1)]
    sim_feat = sim_feat[~np.isnan(sim_feat).max(1)]

    if feat_ids is None:
        feat_ids = np.arange(train_feat.shape[-1])

    distributions = {}
    for i, idx in enumerate(feat_ids):
        dat = train_feat[:, idx]
        dat = dat[~np.isnan(dat)]
        if use_log:
            # eps = 0.001
            # dat = np.log(np.abs(dat) + eps)
            dat = symlog(dat, linthresh=0.01, linscale=1)
        count, bin_edges = np.histogram(dat, n_bins)
        dat = sim_feat[:, idx]
        dat = dat[~np.isnan(dat)]
        if use_log:
            # dat = np.log(np.abs(dat) + eps)
            dat = symlog(dat, linthresh=0.01, linscale=1)
        sim_count, _ = np.histogram(dat, bin_edges)
        eps = 0.1
        kl_div = entropy(count + eps, sim_count + eps)
        distributions[idx] = (bin_edges, count, sim_count, kl_div)

    return distributions


def detect_walk(track, sessions, PXPERMM, half_width, kp=7, dur_thresh=20, log_vel_thresh=0):
    walking = np.zeros((track.array.shape[:2]))
    n_bouts = 0
    for session in sessions:
        agent_id = session.agent_id
        f0 = session.start_frame
        dur = session.duration
        frames = np.arange(f0, f0+dur)
        x, y = track.array[agent_id, frames, :, kp].T * PXPERMM + half_width
        vel = np.zeros(len(x))
        vel[1:] = (np.diff(x) ** 2 + np.diff(y) ** 2) ** 0.5
        log_vel = np.log(np.abs(vel) + 0.00001).flatten()
        labeled_array, num_components = ndimage.label(log_vel > log_vel_thresh)

        component_durs = np.zeros(num_components)
        component_maxs = np.zeros(num_components)
        for i in range(num_components):
            component_durs[i] = (labeled_array == i + 1).sum()
            component_maxs[i] = log_vel[labeled_array == i + 1].max()
        valid_components = np.where(component_durs > dur_thresh)[0] + 1
        n_bouts += len(valid_components)
        for i, c in enumerate(valid_components):
            curr_valid = labeled_array == c
            walking[agent_id, frames[curr_valid]] = i + 1
        walking[agent_id, 0] = 0

    return walking


def get_bouts(predictions, flies, frames, action_id, min_dur=40):
    valid = np.where(predictions[:, action_id] == 1)[0]
    flies = flies[valid]
    frames = frames[valid]

    framediff = np.ones(len(frames)) * 2
    flydiff = np.ones(len(flies))

    framediff[1:] = np.diff(frames)
    flydiff[1:] = np.diff(flies)

    first_frame = np.where((framediff > 1) + (np.abs(flydiff) > 0))[0]
    bouts = [[first_frame[i], first_frame[i + 1] - 1] for i in range(len(first_frame) - 1)]
    bouts.append([first_frame[-1], len(frames) - 1])
    bouts = np.array(bouts)
    durs = np.array([bout[1] - bout[0] + 1 for bout in bouts])
    bouts = bouts[durs >= min_dur]

    return frames[bouts], flies[bouts[:, 0]]


def select_bouts(classifications, all_actions, sim_frame, action, data_type, n_examples, min_dist=1, max_dist=None, min_dur=40):
    action_id = all_actions.index(action)
    classific = classifications[data_type]
    predictions = classific['logits'] > 0.5
    frames = classific['frames']
    flies = classific['flies']
    bouts = get_bouts(predictions, flies, frames, action_id, min_dur=min_dur)

    # get distsnce to prompt
    n_bouts = len(bouts[0])
    dists_from_prompt = np.zeros(n_bouts, int)
    for bout_idx in range(n_bouts):
        f_start, f_end = bouts[0][bout_idx]
        agent_id = bouts[1][bout_idx]
        dists_from_prompt[bout_idx] = sim_frame[agent_id, f_start]

    if max_dist is None:
        max_dist = dists_from_prompt.max()
    valid = np.logical_and(dists_from_prompt > min_dist, dists_from_prompt <= max_dist)
    indices = np.random.permutation(np.where(valid)[0])[:n_examples]

    return bouts, indices, dists_from_prompt


def plot_locomotions(track_data, classifications, all_actions, sim_frame, action, data_type, min_dist=1, max_dist=None, plot_with_features=False):
    if action == 'walking':
        keypoint_ids = [i for i, name in enumerate(keypointnames) if 'leg_tip' in name]
    elif action == 'perframe_wingext':
        keypoint_ids = [i for i, name in enumerate(keypointnames) if 'wing' in name]

    if plot_with_features:
        n_examples = 10
        rows = n_examples
        cols = 3
        width = 15
        height = width / cols * rows
    else:
        rows = 5
        cols = 5
        n_examples = rows * cols
        width = height = 15

    bouts, indices, dists_from_prompt = select_bouts(
        classifications, all_actions, sim_frame, action, data_type, n_examples, min_dist, max_dist)

    fig = plt.figure(figsize=(width, height))
    for i, bout_idx in enumerate(indices):

        f_start, f_end = bouts[0][bout_idx]
        agent_id = bouts[1][bout_idx]
        dist_from_prompt = int(dists_from_prompt[bout_idx])

        # Find the keypoints for this section
        kpts = track_data.array[agent_id, f_start:f_end]

        # Map them to body centric
        Xn, fthorax, thorax_theta = body_centric_kp(kpts.T)

        if plot_with_features:
            ax = plt.subplot(rows, cols, i * 3 + 1)
        else:
            ax = plt.subplot(rows, cols, i + 1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)  # Default is usually 1.0 or 1.5

        x, y = Xn[:, :, 0, 0].T
        for edge in skeleton_edges:
            plt.plot(x[edge], y[edge], '-k')
        for kp_id in keypoint_ids:
            x, y = Xn[kp_id, :, :, 0]
            line = plt.plot(x, y, '-', linewidth=1)[0]
            clr = line.get_color()
            plt.plot(x[0], y[0], '.', color=clr, markersize=5)

        if data_type == 'train_new':
            plt.title(f'bout #{bout_idx}, dur = {f_end - f_start}', fontsize=10)
        else:
            plt.title(f'bout #{bout_idx}, dur = {f_end - f_start}, dtp = {dist_from_prompt}', fontsize=10)
        plt.axis('equal')
        plt.axis([-3.7, 3.6, -3.5, 2])
        plt.xticks([])
        plt.yticks([])

        if plot_with_features:
            plt.axis('off')
            for kp_id in keypoint_ids:
                x, y = Xn[kp_id, :, :, 0]
                plt.subplot(rows, cols, i * 3 + 2)
                plt.plot(x, np.arange(len(x)), '-', linewidth=1)
                plt.subplot(rows, cols, i * 3 + 3)
                plt.plot(y, '-', linewidth=1)
            ax = plt.subplot(rows, cols, i * 3 + 2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)  # Default is usually 1.0 or 1.5
            plt.xlabel('fly left-to-right')
            plt.ylabel('time')
            ax = plt.subplot(rows, cols, i * 3 + 3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)  # Default is usually 1.0 or 1.5
            plt.ylabel('fly back-to-front')
            plt.xlabel('time')

    fig.tight_layout()


def shift_and_scale(x, y, pixpermm=PXPERMM, half_width=ARENA_RADIUS_PX, multiplier=1):
    x = x * multiplier * pixpermm + half_width
    y = y * pixpermm + half_width
    return x, y


def get_range(x_path, y_path, width=ARENA_RADIUS_PX*2, height=ARENA_RADIUS_PX*2, radius=200, buf=50):
    radius = np.array(
        [(x_path.max() - x_path.min()) // 2 + buf, (y_path.max() - y_path.min()) // 2 + buf, radius]).max()
    midx = x_path.mean()
    midy = y_path.mean()
    maxx = midx + radius
    minx = midx - radius
    maxy = midy + radius
    miny = midy - radius
    if minx < 0:
        maxx -= minx
        minx = 0
    if miny < 0:
        maxy -= miny
        miny = 0
    if maxx > width:
        diff = maxx - width
        minx -= diff
        maxx -= diff
    if maxy > height:
        diff = maxy - height
        miny -= diff
        maxy -= diff
    return [minx, maxx, maxy, miny]


def video_of_action_bouts(track_data, classifications, all_actions, sim_frame, action, data_type, output_path,
                          is_female, n_samples=10, min_dist=1, max_dist=None, bg_img=None, overwrite=False):

    output_path_mp4 = output_path.replace('avi', 'mp4')
    if os.path.exists(output_path_mp4):
        if overwrite:
            os.remove(output_path_mp4)
        else:
            print(f"{output_path_mp4} already exists and will not be overwritten")
            return

    bouts, indices, dists_from_prompt = select_bouts(
        classifications, all_actions, sim_frame, action, data_type, n_samples, min_dist, max_dist)

    # Plot the frames of a random bout
    rgbs = []
    for sample_i in tqdm(range(len(indices))):
        idx = indices[sample_i]
        f_start, f_end = bouts[0][idx]
        fly_id = bouts[1][idx]

        x, y = track_data.array[fly_id, f_start:f_end+1, :, 7].T
        x_path, y_path = shift_and_scale(x, y)

        axlim = get_range(x_path, y_path, width=np.inf, height=np.inf, radius=200, buf=50)

        for f in range(f_start, f_end + 1):
            fig = plt.figure(figsize=(10, 10))
            if bg_img is not None:
                plt.imshow(bg_img, cmap='gray')

            for i in range(track_data.array.shape[0]):
                x, y = track_data.array[i, f]
                x, y = shift_and_scale(x, y)
                female = is_female[i, f_start]
                for edge in skeleton_edges:
                    if edge[1] in [0, 1]:
                        plt.plot(x[edge], y[edge], '-c', linewidth=2, alpha=0.5)
                    else:
                        if female:
                            plt.plot(x[edge], y[edge], '-', color=0.6*np.ones(3), linewidth=1)
                        else:
                            plt.plot(x[edge], y[edge], '-k', linewidth=1)

            f_idx = f - f_start
            plt.plot(x_path[:f_idx + 1], y_path[:f_idx + 1], '.r', markersize=5, alpha=0.5)
            plt.plot(x_path[f_idx], y_path[f_idx], '.r', markersize=7)
            plt.axis('off')
            plt.axis(axlim)
            # plt.clim([0, 1])
            if data_type.startswith('train'):
                plt.title(f"{action} bout #{idx}")
            else:
                plt.title(f"{action} bout #{idx}, dtp = {dists_from_prompt[idx]}")
            # plt.show()
            # return

            rgbs.append(fig_to_numpy(fig))
            plt.close()
            del (fig)

        # Add a blank frame
        if sample_i < len(indices) - 1:
            for d in [.8, .5, .3, 0, .3, .5, .8]:
                rgb = np.ones_like(rgbs[-1]) * d
                rgbs.append(rgb)

    arrays_to_video(
        frames=rgbs,
        output_path=output_path,
        fps=30.0,
        color_format="rgb",
        codec="png ",  # note: the gap after png is necessary
    )
    convert_video_to_mp4(output_path)


def get_closest_fly(pred_trks, agent_id, kp):
    endpoints = [trk[agent_id, -1, :, kp] for trk in pred_trks]
    endpoints = np.array(endpoints)

    other_ids = np.concatenate([np.arange(0, agent_id), np.arange(agent_id + 1, 10)])
    other_flies = pred_trks[-1][other_ids, -1, :, kp]

    headings = np.array([trk[agent_id, -1, :, 7] - trk[agent_id, -1, :, 8] for trk in pred_trks])
    headings_norm = np.linalg.norm(headings, axis=1)
    headings = headings / headings_norm[:, None]
    # Note: this becomes a bit annoying because of the vectorization I'm doing

    dist_to_others = np.zeros((other_flies.shape[0], endpoints.shape[0]))
    angle_to_others = np.zeros_like(dist_to_others)
    for i, flyid in enumerate(other_flies):
        vec_to_others = other_flies[i] - endpoints
        dist_to_others[i, :] = np.linalg.norm(vec_to_others, axis=1)
        vec_to_others = vec_to_others / dist_to_others[i, :][:, None]
        dots = np.einsum("ij,ij->i", headings, vec_to_others)
        angles = np.arccos(np.clip(dots, -1, 1))
        # dist_to_others[i, angles > angle_thresh] = dist_to_others.max()
        angle_to_others[i, :] = angles

    distance = dist_to_others + 2*angle_to_others

    closest_id = np.argmin(distance, axis=0)
    dist_to_closest = np.min(distance, axis=0)

    closest_fly = np.zeros(endpoints.shape[0], int)
    for i, idx in enumerate(np.unique(closest_id)):
        inds = np.where(np.logical_and(closest_id == idx, dist_to_closest < 5))[0]
        closest_fly[inds] = other_ids[idx] + 1
    return closest_fly


# def plot_qualitative(train_dataset, train_data, raw_data, model, start_frame, track_len, agent_ids):
#     track = train_data['track']
#     pose = train_data['pose']
#     flyids = train_data['flyids']
#
#     contextl = train_dataset.context_length
#
#     # simulate
#     # start_frame = 45004
#     # agent_ids = [0, 1, 4, 6]
#     # agent_ids = [0]
#     # track_len = contextl_repeats * contextl
#
#     t0 = time.time()
#     n_trials = 5
#     pred_trks = []
#     for i in range(n_trials):
#         gt_track, pred_track = simulate(
#             dataset=train_dataset,
#             model=model,
#             track=track,
#             pose=pose,
#             identities=flyids,
#             track_len=track_len,
#             burn_in=contextl,
#             max_contextl=contextl,
#             agent_ids=agent_ids,
#             start_frame=start_frame,
#         )
#         pred_trks.append(pred_track)
#     time.time() - t0
#
#     first_frame = contextl
#     last_frame = None
#
#     video_idx = raw_data['videoidx'][start_frame + first_frame].astype(int)[0]
#     video_frame = raw_data['frames'][start_frame + first_frame][0]
#     moviepath = os.path.join(str(raw_data['expdirs'][video_idx]), 'movie.ufmf')
#     movie = Movie(moviepath)
#     frame, val = movie.get_frame(video_frame)
#
#     first_img = movie.get_frame(raw_data['frames'][start_frame][0])[0]
#     last_img = movie.get_frame(raw_data['frames'][start_frame + track_len][0])[0]
#
#     plt.figure(figsize=(10, 10))
#
#     frame = frame.astype(float) / 255
#     last_img = last_img.astype(float) / 255
#
#     plt.imshow(frame + 0.3 * last_img, cmap='gray')
#     plt.imshow(frame, cmap='gray')
#
#     kp = 7
#
#     half_width = frame.shape[0] / 2
#     clr_ct = 1
#     def get_unique_colors(N, cmap_name='jet'):
#         cmap = plt.get_cmap(cmap_name)
#         indices = np.linspace(0, 1, N)
#         colors = cmap(indices)
#         return colors
#
#     clrs = get_unique_colors(10)
#     gray = 0.5 * np.ones(3)
#     for agent_id in range(gt_track.shape[0]):
#         if agent_id in agent_ids:
#             closest_fly = get_closest_fly(pred_trks, agent_id, kp)
#             for i, pred_track in enumerate(pred_trks):
#                 x, y = pred_track[agent_id, first_frame:last_frame, :, kp].T * PXPERMM + half_width
#                 clr_ct = closest_fly[i]
#                 if clr_ct == 0:
#                     plt.plot(x, y, '-', color=gray, markersize=1, linewidth=1, alpha=0.5, zorder=1)
#                     plt.plot(x[-1], y[-1], '.', color=gray, markersize=4, alpha=0.5, zorder=1)
#                 else:
#                     plt.plot(x, y, '-', color=clrs[clr_ct - 1], markersize=1, linewidth=1, alpha=0.5, zorder=10)
#                     plt.plot(x[-1], y[-1], '.', color=clrs[clr_ct - 1], markersize=4, alpha=0.5, zorder=10)
#                 clr_ct += 1
#
#             x, y = gt_track[agent_id, first_frame:last_frame, :, kp].T * PXPERMM + half_width
#             plt.plot(x, y, '-w', markersize=1, linewidth=3, zorder=11)
#             plt.plot(x[-1], y[-1], '.w', markersize=4, zorder=11)
#             plt.plot(x, y, '-', color=clrs[agent_id], markersize=1, linewidth=1, zorder=12)
#             plt.plot(x[-1], y[-1], '.', color=clrs[agent_id], markersize=2, zorder=12)
#
#     plt.axis('equal')
#     plt.axis('off')
#     plt.show()
#
#     # PLOT LOCOMOTION OF LEGS
#     pred = pred_trks[3]
#
#     agent = 0
#     x, y = track.array[agent, :, :, kp].T * PXPERMM + half_width
#
#     vel = (np.diff(x) ** 2 + np.diff(y) ** 2) ** 0.5
#     log_vel = np.log(np.abs(vel) + 0.00001).flatten()
#     # log_vel[np.isinf(np.abs(log_vel))
#
#     plt.figure()
#     valid = np.where(~np.isnan(log_vel))[0]
#     count, bins = np.histogram(log_vel[valid], bins=100)
#     binc = (bins[1:] + bins[:-1]) / 2
#     plt.plot(binc, count, '-')
#     plt.plot([-1.25, -1.25], [0, count.max() * 1.01], '-k')
#     plt.plot([0, 0], [0, count.max() * 1.01], '-k')
#     plt.show()
#
#     x, y = pred[agent, contextl:, :, kp].T * PXPERMM + half_width
#     vel = (np.diff(x) ** 2 + np.diff(y) ** 2) ** 0.5
#     log_vel = np.log(np.abs(vel) + 0.00001).flatten()
#
#     dur_thresh = 20
#     labeled_array, num_components = ndimage.label(log_vel > 0)
#     component_durs = np.zeros(num_components)
#     component_maxs = np.zeros(num_components)
#     for i in range(num_components):
#         component_durs[i] = (labeled_array == i + 1).sum()
#         component_maxs[i] = log_vel[labeled_array == i + 1].max()
#     valid_components = np.where(component_durs > dur_thresh)[0] + 1
#     walking = np.zeros(len(vel))
#     for i, c in enumerate(valid_components):
#         walking[labeled_array == c] = i + 1
#     walking[0] = 0
#
#     plt.figure(figsize=(7, 2))
#     plt.scatter(np.arange(len(vel)), log_vel, c=log_vel, s=1, cmap='jet')
#     plt.plot([0, len(vel)], [-1.25, -1.25], '-k')
#     plt.plot([0, len(vel)], [0, 0], '-k')
#     # plt.show()
#
#     # plt.figure(figsize=(7, 2))
#     plt.plot((walking > 0) * 2, '-', linewidth=1)
#     plt.show()
#
#     plt.figure()
#     plt.imshow(frame, cmap='gray')
#     plt.scatter(x[1:], y[1:], c=walking, s=2, cmap='jet')
#     plt.show()
#
#     # Pick a window from one of the walking bouts
#
#     agent_id = agent
#
#     legids = np.array([idx for idx, name in enumerate(keypointnames) if 'leg_tip' in name])
#     legnames = [keypointnames[idx] for idx in legids]
#
#     body_centric = body_centric_kp(pred[:, contextl:].T)[0].T
#
#     bout_id = 7
#     frames = np.where(walking == bout_id)[0]
#     # frames = frames[:15]
#
#     # agent_id=0
#     legpos = body_centric[agent_id, frames, :][..., legids]
#
#     frame_id = contextl + 0  # frames[0]
#     video_frame = raw_data['frames'][start_frame + frame_id][0]
#     frame, val = movie.get_frame(video_frame)
#     frame = frame.astype(float) / 255
#     kpt = pred[agent_id, frame_id, :, :].T * PXPERMM + half_width
#     rotmat = get_rotmat(kpt)
#     rot_img = rotate_image(frame, rotmat)
#     rot_kpt = rotate_keypoints(kpt, rotmat)
#     buf = 70
#     cropbox = get_cropbox(rot_kpt, buf)
#     crop_img, out_of_bounds = crop_image(rot_img, cropbox, fill_value=0)
#     # crop_kpts = crop_keypoints(rot_kpt, cropbox)
#
#
#     plt.figure(figsize=(17, 5))
#     plt.subplot(1, 3, 3)
#     plt.imshow(crop_img, cmap='gray')
#     # image of the fly at the first frame
#     for i in range(legpos.shape[-1]):
#         x, y = legpos[:, :, i].T * PXPERMM + crop_img.shape[0] / 2
#
#         plt.subplot(1, 3, 1)
#         plt.plot(x, np.arange(len(x)))
#         plt.subplot(1, 3, 2)
#         plt.plot(y)
#         plt.subplot(1, 3, 3)
#         plt.plot(x, crop_img.shape[0] - y, '.-')
#         # plt.scatter(x, crop_img.shape[0]-y, c=np.arange(len(x)), s=5, cmap='jet')
#     plt.subplot(1, 3, 1)
#     plt.xlabel('fly left-to-right')
#     plt.ylabel('time')
#     plt.subplot(1, 3, 2)
#     plt.ylabel('fly back-to-front')
#     plt.xlabel('time')
#     # plt.legend(legnames, loc='upper left', bbox_to_anchor=(1.05, 1))
#     plt.subplot(1, 3, 3)
#     plt.axis('equal')
#     plt.axis('off')
#     plt.show()
#
#
# def eval_simulation(configfile, modelfile, savedir, sim_len, first_sim_frame=None, stride=None):
#     # Load config
#     print('Loading config')
#     config = read_config(configfile)
#     load_config_from_model_file(loadmodelfile=modelfile, config=config)
#
#     # Load dataset
#     print('Loading dataset')
#     res = init_datasets(config, needvaldata=True, res={'config': config})
#     train_dataset = res['train_dataset']
#     train_data = res['train_data']
#     contextl = train_dataset.context_length
#     val_dataset = res['val_dataset']
#     val_data = res['val_data']
#
#     # Load raw data, to get the action labels
#     raw_res = init_raw_data(config, needtraindata=True, needvaldata=True)
#     raw_data = raw_res['data']
#     raw_val_data = raw_res['valdata']
#     raw_data['y'][:, ~raw_data['isdata']] = np.nan
#     raw_val_data['y'][:, ~raw_val_data['isdata']] = np.nan
#     assert (raw_data['X'].shape[-2] == train_dataset.inputs['velocity'].array.shape[1])
#
#     # Load model
#     device = torch.device(config['device'])
#     model, criterion = initialize_model(config, train_dataset, device)
#     _ = load_model(modelfile, model, device)
#
#     # Prepare cache folder
#     cache_foldername = 'hidden_cache'
#     if not os.path.exists(cache_foldername):
#         os.makedirs(cache_foldername)
#     configname = os.path.split(configfile)[-1].replace('.json', '')
#     modelname = os.path.split(modelfile)[-1].replace('.pth', '')
#     sim_transformed_path = os.path.join(cache_foldername, f"sim__{configname}__{modelname}.npy")
#
#     # Run/load the simulation and create a dataset from it
#     print('Loading simulated data')
#     sim_track, sim_frame, sim_transformed = simulate_tracks(
#         model, train_dataset, train_data, savedir, sim_len, stride=stride, simulate_single=False,
#         first_sim_frame=first_sim_frame, sim_transformed_path=sim_transformed_path
#     )
#     print('Creating simulated dataset')
#     sim_dataset, sim_data = create_dataset_from_ref_data(train_dataset, train_data, sim_track, sim_frame)
#
#     # Create and save hidden states for train, val, and sim datasets
#     print('Computing and caching hidden states')
#     transformed_data = {'sim': sim_transformed}
#     for dataname, dataset in zip(['train', 'val'], [train_dataset, val_dataset]):
#         print(f'-- {dataname}')
#         transformed_path = os.path.join(cache_foldername, f"{dataname}__{configname}__{modelname}.npy")
#         transformed_data[dataname] = transform_dataset(dataset, model, transformed_path, batch_size=128)
#
#     # ====== QUANTITATIVE EVALUATION ========
#     pdf = PdfPages(os.path.join(savedir, f"{modelname}_report.pdf"))
#
#     # Get frame chunks of simulated frames
#     valid_frames = {
#         'first': np.logical_and(sim_frame > 0, sim_frame <= contextl),
#         'last': sim_frame > sim_len - contextl,
#         'all': sim_frame > 0,
#     }
#
#     # FEATURE DISTRIBUTIONS
#     print('Plotting feature distributions')
#     for frame_label, frames in valid_frames.items():
#         plot_feature_distributions(train_data, sim_data, frames, frame_label, data_name='pose', use_log=False, pdf=pdf)
#         plot_feature_distributions(train_data, sim_data, frames, frame_label, data_name='velocity', use_log=True, pdf=pdf)
#
#     # ACTION DISTRIBUTIONS
#     classifiers = {}
#     try:
#         print('Training per-frame action classifiers')
#         classifications = {}
#
#         actions = [name for name in raw_data['categories']]
#         # actions = [name for name in raw_data['categories'] if name.startswith('perframe_')]
#         frame_counts = np.zeros((len(actions), 2))
#         for i in range(len(actions)):
#             idx = raw_data['categories'].index(actions[i])
#             frame_counts[i, 0] = (raw_data['y'][idx] == 1).sum()
#             frame_counts[i, 1] = (raw_data['y'][idx] == 0).sum()
#         _, n_frames, n_flies = raw_data['y'].shape
#         total_frames = n_frames * n_flies
#         actions = [actions[i] for i in range(len(actions)) if 500 < frame_counts[i, 0] < total_frames]
#         # actions = [actions[i] for i in range(len(actions)) if frame_counts[i].sum() > 1000]
#
#         print(actions)
#         if len(actions) == 0:
#             tmp = 0/1
#
#         # Train classifiers
#         overwrite = True
#         for action in actions:
#             print(f'Classifier for {action}')
#             classifier_path = f'from_hidden_classifier_{action}.pth'
#             action_idx = raw_data['categories'].index(action)
#             val_action_idx = raw_val_data['categories'].index(action)
#             if os.path.exists(classifier_path) and not overwrite:
#                 print('- loading from existing')
#                 classifier = MLP(input_size=model.d_model, layer_size=[128])
#                 # classifier.load_state_dict(torch.load(classifier_path, weights_only=True))
#                 checkpoint = torch.load(classifier_path, weights_only=False)
#                 classifier.load_state_dict(checkpoint['model_state_dict'])
#                 classifier.stats = checkpoint['stats']
#                 classifier.eval()
#             else:
#                 print('- training')
#                 classifier = train_from_hidden(transformed_data['train'], transformed_data['val'],
#                                                raw_data['y'][action_idx].T.astype(np.float32),
#                                                raw_val_data['y'][val_action_idx].T.astype(np.float32),
#                                                device, pdf, action)
#                 classifier.eval()
#                 # torch.save(classifier.state_dict(), classifier_path)
#                 checkpoint = {
#                     'model_state_dict': classifier.state_dict(),
#                     'stats': classifier.stats,
#                 }
#                 torch.save(checkpoint, classifier_path)
#             classifiers[action] = classifier
#
#         # Apply classifiers to real and sim data
#         summarytable = np.zeros((2, len(actions)))
#         for dat_i, (name, dataset) in enumerate({'train': train_dataset, 'sim': sim_dataset}.items()):
#             print(f'Running classifiers for {name}')
#             flies, frames = np.where(valid_frames['all'])
#
#             # Apply classifiers
#             logits = np.ones((len(frames), len(actions))) * np.nan
#             batch_size = 64
#             for b in tqdm(range(0, len(frames), batch_size)):
#                 batch_hidden = torch.from_numpy(transformed_data[name][flies[b:b+batch_size], frames[b:b+batch_size]]).to(device)
#                 for act_i, action in enumerate(actions):
#                    logits[b:b+batch_size, act_i] = classifiers[action](batch_hidden).detach().cpu().numpy()[:, 0]
#
#             summarytable[dat_i] = (logits > 0.5).sum(0)
#             classifications[name] = {'logits': logits, 'frames': frames, 'flies': flies}
#         plot_array_as_table(
#             data=summarytable.T,
#             row_labels=actions,
#             col_labels=['real', 'sim'],
#             title='Real vs sim label statistics', pdf=pdf
#         )
#
#         fig = plt.figure(figsize=(8, 4))
#         fractions = summarytable / logits.shape[0]
#         for i in range(len(actions)):
#             plt.plot([i, i], fractions[:, i], '--k', linewidth=1)
#         plt.plot(fractions[0, :], '.', markersize=10, label='real')
#         plt.plot(fractions[1, :], '.', markersize=8, label='sim')
#         plt.xticks(np.arange(len(actions)), actions, rotation=90)
#         plt.ylabel('fraction of frames')
#         plt.title('Perframe action classification')
#         plt.legend()
#         plt.ylim([0, 1])
#         fig.tight_layout()
#         if pdf is None:
#             plt.show()
#         else:
#             pdf.savefig(fig)
#             plt.close()
#
#         tmp = 1/0
#
#     except Exception as e:
#         print(e)
#         return train_data, val_data, sim_data, train_dataset, val_dataset, sim_dataset, transformed_data, valid_frames, sim_frame, raw_data, raw_val_data, classifiers, pdf, e
#
#     # TRUE-vs-SIM CLASSIFICATION ACCURACY
#     print('Training true-vs-sim classifiers')
#     try:
#         class_results = np.zeros((len(valid_frames), 2))
#         n_epochs = 3
#         for i, (frame_label, frames) in enumerate(valid_frames.items()):
#             train_feat = train_dataset.inputs['velocity'].array[frames]
#             sim_feat = sim_dataset.inputs['velocity'].array[frames]
#
#             # print(f"Frames used: {frame_label}")
#             accuracy = classify_real_fake(train_feat, sim_feat, n_epochs=n_epochs)
#             control_accuracy = classify_real_fake(train_feat, sim_feat, n_epochs=n_epochs*2, control=True)
#             class_results[i, 0] = accuracy
#             class_results[i, 1] = control_accuracy
#         plot_array_as_table(
#             data=class_results,
#             row_labels=list(valid_frames.keys()),
#             col_labels=['accuracy', 'control accuracy'],
#             title='Real vs fake classification accuracy', pdf=pdf
#         )
#         pdf.close()
#     except Exception as e:
#         pdf.close()
#
#     # ====== QUALITATIVE ========
#     # MULTI TRAJECTORIES (CLUSTERED BY WHERE IT ENDS UP)
#
#     # PATTERNS IN AGENTS FRAME, DURING CERTAIN ACTIONS (LIKE WALKING)
#
#
#     return train_data, sim_data, train_dataset, sim_dataset, valid_frames, sim_frame, raw_data, classifications


"""
Would be nice to have this code split:


simulate_tracks
create_sim_dataset

train_model
apply_model

train_perframe_classifier (lightweight, wraps train_model)
train_realsim_classifier  (lightweight, wraps train_model)

make_multiple_track_preds
cluster_tracks (by fly it ends up next to)
plot_track_distribution
make_video_from_track
 - makes a video for specific frames and fly/flies
 - can be used to visualize entire video, action bouts, certain simulated trajectories, ...

walk_detector
plot_locomotion
    - plots specified keypoints in fly's reference frame, both on top of an image and the distributions
    - can test this on detected walks and detected wing extensions

evaluate_simulation
    - runs analysis up to a certain sim_frame count
    - feature distribution, action distribution, real-sim accuracy
    - plots clusters of short term simulations
    - plots locomotion for specific actions and keypoints for real-vs-sim actions
    - videos of real-vs-sim actions
    - video of full simulations of all flies (along with some feature or classification score)




Plan of attack:
- basically, paper_figures.ipynb is everything in the evaluate_simulation function + more
- take all functionality that I used in that notebook and move to functions here
- make that notebook as clean and minimal as possible
- replace code in evaluate_simulation with the code we are now using in the notebook 
- clean up and consolidate functions
"""
