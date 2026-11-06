"""Simulating from the test set, one independently resumable file pair per chunk."""
import copy
import os

import numpy as np

from apf.eval_pipeline.context import RunContext, job_items
from apf.eval_pipeline.params import atomic_path
from apf.simulation import simulate


def enumerate_sessions(dataset, sim_len, first_sim_frame):
    """Returns (start_frame, duration, agent_ids) of periods where the same set of agent sessions is active."""
    key = list(dataset.inputs.keys())[0]
    n_agents, n_frames, _ = dataset.inputs[key].array.shape
    session_frames = np.full((n_agents, n_frames), -1)  # -1 is no session, 0 is a session
    for i, session in enumerate(dataset.sessions):
        session_frames[session.agent_id, session.start_frame:session.start_frame + session.duration] = i
    changes = np.where(np.any(session_frames[:, 1:] != session_frames[:, :-1], axis=0))[0] + 1
    boundaries = np.concatenate([[0], changes, [n_frames]])

    sessions = []
    for start_frame, end_frame in zip(boundaries[:-1], boundaries[1:]):
        duration = end_frame - start_frame
        if duration < sim_len + first_sim_frame + 1:
            continue
        agent_ids = np.where(session_frames[:, start_frame] >= 0)[0]
        if len(agent_ids) > 0:
            sessions.append((int(start_frame), int(duration), agent_ids))
    return sessions


def enumerate_chunks(dataset, sim_len, first_sim_frame, stride=None, max_chunks=None):
    """Returns a list of {'start': first simulated frame, 'agents': agent ids, 'file': file name}."""
    stride = sim_len if stride is None else stride
    chunks = []
    for i, (start, duration, agent_ids) in enumerate(enumerate_sessions(dataset, sim_len, first_sim_frame)):
        agent_str = '_'.join(str(agent) for agent in agent_ids)
        for start_sim_frame in np.arange(start + first_sim_frame, start + duration - sim_len, stride):
            chunks.append({
                'start': int(start_sim_frame),
                'agents': [int(agent) for agent in agent_ids],
                'file': f"session_{i}_startsimframe_{start_sim_frame}_agentid_{agent_str}_simlen_{sim_len}.npy",
            })
    return chunks if max_chunks is None else chunks[:max_chunks]


def _save_npy(path, array):
    tmp = atomic_path(path)
    np.save(tmp, array)
    os.replace(tmp, path)


def run_simulate(ctx: RunContext, shard: int):
    """Simulates the chunks of this job.

    The hidden states have to be recorded here. Each chunk's context is ground truth, whereas a stitched track has
    the previous chunk's simulation in those frames.
    """
    params = ctx.params
    paths = ctx.paths
    dataset = ctx.test_dataset
    data = ctx.test_data
    model = ctx.load_model(dataset)
    contextl = dataset.context_length
    track_len = params.sim_len + contextl

    for _, chunk in job_items(ctx.manifest['chunks'], shard, params.n_simulate_jobs):
        track_path = os.path.join(paths.sim_track_dir, chunk['file'])
        hidden_path = os.path.join(paths.sim_hidden_dir, chunk['file'])
        if os.path.exists(track_path) and os.path.exists(hidden_path):
            continue
        agent_ids = np.array(chunk['agents'])
        hidden = np.full((len(agent_ids), track_len, model.d_model), np.nan, np.float32)
        _, pred_track = simulate(
            dataset=dataset, model=model, track=data['track'], pose=data['pose'], identities=data['flyids'],
            track_len=track_len, burn_in=contextl, max_contextl=contextl,
            agent_ids=agent_ids, start_frame=chunk['start'] - contextl,
            hidden_layer=params.probe_layer, hidden_out=hidden,
        )
        _save_npy(hidden_path, hidden[:, contextl:].astype(params.feature_dtype))
        _save_npy(track_path, pred_track[agent_ids])


def run_stitch(ctx: RunContext, shard: int):
    """Writes all chunks into a copy of the test track. Frames of a simulation that aborted are not marked valid."""
    params = ctx.params
    paths = ctx.paths
    contextl = ctx.test_dataset.context_length
    sim_track = copy.deepcopy(ctx.test_data['track'].array)
    sim_frame = np.zeros(sim_track.shape[:2])
    for chunk in ctx.manifest['chunks']:
        agents = np.array(chunk['agents'])
        pred_track = np.load(os.path.join(paths.sim_track_dir, chunk['file']))[:, contextl:]
        start = chunk['start']
        sim_track[agents, start:start + params.sim_len] = pred_track
        finite = ~np.isnan(pred_track).any(axis=(2, 3))
        sim_frame[agents, start:start + params.sim_len] = (np.arange(params.sim_len) + 1)[None] * finite
    _save_npy(paths.sim_track, sim_track)
    _save_npy(paths.sim_frame, sim_frame)
