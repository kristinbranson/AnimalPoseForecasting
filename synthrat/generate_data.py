"""Generate synthetic rat trajectories with RatInABox, for training APF models on.

This is the data-generation half of the synthrat experiment, kept separate from the
training and evaluation half: nothing here imports the model, and nothing in
`experiments/synthrat.py` runs an episode. The two meet at the trajectory pickle this
module writes, which a training config names as its `intrainfilestr` / `invalfilestr`.

An episode is a rollout of RatInABox's value-function-guided policy: the agent moves up
the gradient of a `ValueNeuron` built over place-cell inputs, from a random start
position until it reaches the reward or runs out of time. Each episode records the
agent's track and the hidden state of the value function, plus, when a sensory
configuration is given, the firing rates of the sensory populations the model observes.

The state of the environment, agent, place cells and value function is not pickled
directly -- RatInABox objects do not survive a round trip -- but captured as plain dicts
by the `get_*_info` functions here and rebuilt by the `rehydrate_*` functions in
`synthrat.sensory`.

Requires RatInABox to be installed (`import ratinabox`); the fork this was developed
against is kristinbranson/RatInABox on the `speedup` branch.

Command line:

    python -m synthrat.generate_data --state data/ratinabox_rl_state_20260423.pkl \\
        --out-dir data --timestamp 20260423 \\
        --train-episodes 10000 --val-episodes 1000 --workers 32

writes `ratinabox_rl_traindata_<timestamp>.pkl` and `ratinabox_rl_valdata_<timestamp>.pkl`
into the output directory. With `--sensory-config <cell_config.json>` the sensory
populations are rebuilt from that file first and the state pickle is rewritten, so later
runs generate against the new configuration.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import multiprocessing as mp

import numpy as np
import ratinabox.utils
import tqdm.auto as tqdm

import apf.utils

from synthrat.sensory import (compute_sensory, head_direction_from_orientation, init_sensory,
                              orientation_from_head_direction, rehydrate_agent, rehydrate_data,
                              rehydrate_env, rehydrate_placecells, rehydrate_value_neuron)

# Set in each worker process by _init_worker and read by _run_one_episode, so the
# rebuilt RatInABox objects are made once per worker rather than per episode.
_WORKER_STATE = None

# Names of the episode sets written per run, and the seed each is generated with by
# default, so train and validation episodes never coincide.
TRAIN_SPLIT, VALIDATION_SPLIT = 'train', 'val'
DEFAULT_SEEDS = {TRAIN_SPLIT: 0, VALIDATION_SPLIT: 1}


def get_info(obj):
    """Return a dict snapshot of a RatInABox object's config.

    For each key in `obj.params`, prefer the live attribute value on `obj`
    (so post-construction edits like `Env.walls.append(...)` or
    `Inputs.place_cell_centres[-4:] = ...` are captured) and fall back to
    the params value otherwise. Arrays are `.copy()`'d to avoid aliasing —
    later in-place edits on `obj` won't silently mutate the returned dict.

    Does NOT capture attributes that aren't keys of `obj.params` (e.g.
    custom notebook attributes, derived attributes, or attributes under a
    different name like `place_cell_widths` vs. params key `widths`). Use
    the type-specific wrappers (`get_env_info`, `get_placecell_info`, etc.)
    to add those.
    """
    info = dict(obj.params)
    for k in info.keys():
        if hasattr(obj, k):
            v = getattr(obj, k)
            if callable(v):
                # live attr is a lambda/closure (e.g. FeedForwardLayer wraps
                # activation_function dict into a lambda); keep the params-level
                # spec so the snapshot stays picklable and rehydration re-wraps.
                continue
            info[k] = v.copy() if hasattr(v, "copy") else v
    return info


def get_env_info(Env):
    """Snapshot of an Environment's config plus derived geometry attributes.

    Captures everything `get_info` would (dimensionality, scale, aspect,
    boundary, holes, walls, boundary_conditions, ...) plus `extent` and
    `is_rectangular`, which are computed during `Environment.__init__` from
    the other params but are not themselves params keys.

    Note: `Env.walls` includes the boundary walls (auto-added for a solid
    rectangular env) plus any walls added via `Env.add_wall(...)`. It is
    captured as a copy, so later `Env.walls[-1] = ...` edits won't affect
    the returned dict.
    """
    env_info = get_info(Env)
    if hasattr(Env,'extent'):
        env_info['extent'] = np.asarray(Env.extent).copy()
    if hasattr(Env,'is_rectangular'):
        env_info['is_rectangular'] = Env.is_rectangular
    # Environment.__init__ rewrites self.objects from a list-of-positions
    # (the params form) to {"objects": array, "object_types": array}. get_info
    # captures that post-init dict, which is NOT what the constructor accepts.
    # Serialize back to the list-of-positions form so rehydration works.
    if isinstance(Env.objects, dict) and "objects" in Env.objects:
        pos = np.asarray(Env.objects["objects"])
        env_info["objects"] = pos.tolist() if pos.size else []
    return env_info


def get_agent_info(Ag):
    """Snapshot of an Agent's config plus the notebook's custom extras.

    Captures everything `get_info` would (dt, speed_mean, speed_std, motion
    model params, wall-repel params, etc.) and additionally `exploit_explore_ratio`,
    a custom attribute set by the notebook (not a standard Agent param).
    """
    agent_info = get_info(Ag)
    if hasattr(Ag,'exploit_explore_ratio'):
        agent_info['exploit_explore_ratio'] = Ag.exploit_explore_ratio
    return agent_info


def get_placecell_info(pc):
    """Snapshot of a PlaceCells population's config plus the live per-cell widths.

    Captures everything `get_info` would (n, description, wall_geometry,
    place_cell_centres, min_fr/max_fr, noise params, ...) and then
    **overrides** `widths` with a copy of the live `pc.place_cell_widths`
    array. This matters because `pc.widths` and `pc.place_cell_widths` are
    two separate arrays: `widths` is a mirror of the construction input
    (scalar or array), while `place_cell_widths` is the length-n array
    actually used by `get_state` for firing-rate computation and the one
    affected by in-place edits like `Inputs.place_cell_widths[-4:] = 0.2`.

    If `pc` has a custom `episode_end_time` attribute (set on the reward
    neuron in the notebook), that is also captured.
    """
    pc_info = get_info(pc)
    if hasattr(pc,'place_cell_widths'):
        pc_info['widths'] = np.asarray(pc.place_cell_widths).copy()   # (n,)
    if hasattr(pc,'episode_end_time'):
        pc_info['episode_end_time'] = pc.episode_end_time
    return pc_info


def get_value_neuron_info(ValNeur):
    """Snapshot a ValueNeuron's config plus its learned per-input-layer weights.

    Captures:
      - the construction params (tau, tau_e, eta, L2, activation_function, n, ...)
      - per-input-layer weight arrays: `inputs[layer_name]["w"]` (copied)
      - the notebook-custom `max_value` attribute, if present

    Does NOT save the `input_layers` list itself (those are live Neurons
    objects). On rehydration you must pass in the reconstructed input
    layers explicitly.
    """
    info = get_info(ValNeur)
    # input_layers is a list of live Neurons — drop it, rehydrate caller
    # will supply it.
    info.pop("input_layers", None)
    # save per-layer weights under a new key
    info["inputs"] = {
        name: {
            "w": np.asarray(entry["w"]).copy(),
            "n": entry["n"],
        }
        for name, entry in ValNeur.inputs.items()
    }
    if hasattr(ValNeur, "max_value"):
        info["max_value"] = ValNeur.max_value
    return info


def get_sensory_info(Sensory):
    """Snapshot a dict of sensory Neurons populations.

    For each population, captures the config via `get_info` plus any custom
    attributes.

    Returns
    -------
    sensory_info : dict mapping population name -> info dict.
    """
    sensory_info = {}
    for k,neuron in Sensory.items():
        info = get_info(neuron)
        info["cls_name"] = type(neuron).__name__                                                                                                      
        info["n"] = neuron.n                           # live, not params["n"]
                                                                                                                                                        
        # Vector cells: save per-cell realized tuning                                                                                                 
        for attr in ("tuning_distances", "tuning_angles",                                                                                             
                    "sigma_distances", "sigma_angles"):                                                                                              
            if hasattr(neuron, attr):                                                                                                                 
                info[attr] = np.asarray(getattr(neuron, attr)).copy()
                                                                                                                                                        
        # HDC / VelocityCells: save preferred angles + tunings
        for attr in ("preferred_angles", "angular_tunings"):                                                                                          
            if hasattr(neuron, attr):                                                                                                                 
                info[attr] = np.asarray(getattr(neuron, attr)).copy()
        # FOV cell arrangement could be a function
        if hasattr(neuron,'cell_arrangement') and callable(neuron.cell_arrangement):
            info['cell_arrangement'] = neuron.cell_arrangement.__name__
                                                                                                                                                        
        # SpeedCell: save Agent-derived scale
        if hasattr(neuron, "one_sigma_speed"):                                                                                                        
            info["one_sigma_speed"] = neuron.one_sigma_speed
                                                                          
        sensory_info[k] = info
    return sensory_info


def get_steep_ascent(ValueNeuron, pos):
    """This function will be used for policy improvement. Calculates direction steepest ascent (gradient) of the value function and returns a drift velocity in this direction. Returns None when the local gradient is exceedingly low"""
    V = ValueNeuron.get_state(evaluate_at=None, pos=pos)[0][0] #query the firing rate at the given position
    if V <= 0.05*ValueNeuron.max_value:
        return None # if the value function is too low it is unreliable, return None
    else:  # calculate gradient locally
        V_plusdx = ValueNeuron.get_state(evaluate_at=None, pos=pos + np.array([1e-3, 0]))[0][0]
        V_plusdy = ValueNeuron.get_state(evaluate_at=None, pos=pos + np.array([0, 1e-3]))[0][0]
        gradV = np.array([V_plusdx - V, V_plusdy - V])
        norm = np.linalg.norm(gradV)
        gradV = gradV / norm
        return gradV


def do_episode(ref_ValNeur, ValNeur, Ag, Inputs, Reward, max_t=60, min_t=0):
    """
    Runs an "episode" of the agent moving around the environment. The agents policy is guided by the value function of the ref_ValNeur (approximately equivalent the epislon greedy). Meanwhile the value function of valNeur is being trained on the (greedy) policy.
    
    ref_ValNeur: the fixed reference value function used for getting the drift velocity
    ValNeur: the value function being trained
    Ag: the agent
    Inputs: the input features
    Reward: the reward neuron
    train: whether to train the value function or not
    max_t: the maximum time the episode can run for before timeout
    """
    
    #save start time and position
    Ag.episode_data["start_time"].append(Ag.t)
    Ag.episode_data["start_pos"].append(Ag.pos)

    #resets to zero the eligibility trace and the td error ready for a new episode 
    ValNeur.reset() 

    while True:
        #get greedy direction of steepest ascent of the value function
        gradV = get_steep_ascent(ref_ValNeur, Ag.pos)
        if gradV is None: drift_velocity = None #if None, the agent will just randomly explore
        else: drift_velocity = 3 * Ag.speed_mean * gradV
        # you can ignore this (force agent to travel towards reward when v nearby) helps stability.
        if (Ag.pos[0] > 0.8) and (Ag.pos[1] < 0.4):
            dir_to_reward = Reward.place_cell_centres[0] - Ag.pos
            drift_velocity = (
                3 * Ag.speed_mean * (dir_to_reward / np.linalg.norm(dir_to_reward))
            )

        # move the agent
        Ag.update(
            drift_velocity=drift_velocity,
            drift_to_random_strength_ratio=Ag.exploit_explore_ratio,
        )
        # update inputs and train weights
        Inputs.update()
        Reward.update()
        ValNeur.update()

        t_elapsed = Ag.t - Ag.episode_data["start_time"][-1]

        # end episode when at some random moment when reward is high OR after timeout
        if (t_elapsed >= min_t) and (np.random.uniform() < Ag.dt * Reward.firingrate / Reward.episode_end_time):
            Ag.exploit_explore_ratio *= 1.1  # policy gets greedier if it was successful
            Ag.episode_data["success_or_failure"].append(1)
            break
        if t_elapsed >= max_t:  # timeout
            Ag.episode_data["success_or_failure"].append(0)
            break
    Ag.episode_data["end_time"].append(Ag.t)
    Ag.episode_data["end_pos"].append(Ag.pos)
    Ag.exploit_explore_ratio = max(0.1, min(1, Ag.exploit_explore_ratio)) #keep between 0.1 and 1
    Ag.velocity = np.random.uniform(-0.1, 0.1, size=(2,))
    return


def collect_episode(Ag,ValNeur,Reward,framerate=10):
    """Return per-timestep data for the most recent episode.

    Slices the continuous `.history` dicts of Agent, ValueNeuron, and Reward
    for the time window `[last_start + dt, now]` (i.e. the episode that just
    finished), optionally downsampled to `framerate` Hz. Caller is
    responsible for collecting the returns across episodes (e.g. into lists
    or a dict of lists).

    Parameters
    ----------
    Ag : ratinabox Agent
        Must have `episode_data["start_time"]` populated (set by `do_episode`).
    ValNeur : ValueNeuron
    Reward : PlaceCells (the reward neuron)
    framerate : float
        Target samples/second for the slice. Controls the stride used to
        downsample the full-rate (1/Ag.dt Hz) history.

    Returns
    -------
    track_curr : dict with keys 'pos' (T, 2), 'head_direction' (T, 2),
                 'vel' (T, 2).
    hidden_curr : dict with keys 'val_firingrate' (T,), 'reward_firingrate' (T,).
    """
    t_start = Ag.episode_data["start_time"][-1] + Ag.dt
    t_end = Ag.history["t"][-1]
    slc = Ag.get_history_slice(t_start=t_start, t_end=t_end, framerate=framerate)

    history_data = Ag.get_history_arrays() # gets history dataframe as dictionary of arrays (only recomputing arrays from lists if necessary)
    keys_state = ['pos','head_direction','vel']
    track_curr = {}
    for k in keys_state:
        track_curr[k] = history_data[k][slc]
    
    hidden_curr = {}
    hidden_curr['val_firingrate'] = np.asarray(ValNeur.history['firingrate'][slc]).ravel()
    hidden_curr['reward_firingrate'] = np.asarray(Reward.history['firingrate'][slc]).ravel()
    return track_curr, hidden_curr


def _reset_episode_state(Ag, Inputs, Reward, ValNeur):
    """Clear per-episode accumulator state AND reset Ag dynamics so each
    episode starts from an independent initial condition.

    History lists are cleared so memory stays bounded across thousands of
    episodes (history is write-only during update() — nothing reads it back
    to compute dynamics).

    `Ag.velocity`, `Ag.head_direction`, and the various `prev_*` / derived
    attributes are re-initialised here (via `initialise_position_and_velocity`
    plus the same post-init fixups `Agent.__init__` does) so that episodes
    don't inherit state from each other — important both for conceptual
    independence and for parallel reproducibility, since chunk assignment
    across workers would otherwise let initial velocity drift between runs.
    """
    Ag.reset_history()
    Inputs.reset_history()
    Reward.reset_history()
    ValNeur.reset_history()
    Ag.episode_data = {
        "start_time": [], "end_time": [], "start_pos": [], "end_pos": [],
        "success_or_failure": [],
    }
    Ag._history_arrays = {}
    Ag._last_history_array_cache_time = None
    # Reset dynamics — consumes np.random, so per-episode seed determines it.
    Ag.initialise_position_and_velocity()
    Ag.prev_pos = Ag.pos.copy()
    Ag.measured_velocity = Ag.velocity.copy()
    Ag.prev_measured_velocity = Ag.measured_velocity.copy()
    Ag.measured_rotational_velocity = 0
    Ag.head_direction = Ag.velocity / np.linalg.norm(Ag.velocity)
    Ag.distance_to_closest_wall = np.inf
    Ag.distance_travelled = 0.0
    Ag.prev_t = 0.0
    Ag.t = 0.0


def _init_worker(env_info, ag_info, inputs_info, reward_info, valneur_info,
                 startpos_lo, startpos_hi, exploit_explore_ratio,
                 episode_end_time, max_t, framerate):
    """Pool initializer: rehydrate live objects once per worker and stash
    them plus the invariant episode args in a module-level cache."""
    global _WORKER_STATE
    Env = rehydrate_env(env_info)
    Ag = rehydrate_agent(Env, ag_info)
    Inputs = rehydrate_placecells(Ag, inputs_info)
    Reward = rehydrate_placecells(Ag, reward_info)
    ValNeur = rehydrate_value_neuron(Ag, valneur_info, input_layers=[Inputs])
    Reward.episode_end_time = episode_end_time
    _WORKER_STATE = dict(
        Ag=Ag, Inputs=Inputs, Reward=Reward, ValNeur=ValNeur,
        startpos_lo=np.asarray(startpos_lo, dtype=float),
        startpos_hi=np.asarray(startpos_hi, dtype=float),
        exploit_explore_ratio=exploit_explore_ratio,
        max_t=max_t, framerate=framerate,
    )


def _run_one_episode(seed):
    """Worker task: reseed RNG, reset per-episode state, run one episode."""
    np.random.seed(int(seed))
    s = _WORKER_STATE
    Ag, Inputs, Reward, ValNeur = s['Ag'], s['Inputs'], s['Reward'], s['ValNeur']
    _reset_episode_state(Ag, Inputs, Reward, ValNeur)
    Ag.exploit_explore_ratio = s['exploit_explore_ratio']
    Ag.pos = np.random.uniform(s['startpos_lo'], s['startpos_hi'], size=(2,))
    do_episode(ValNeur, ValNeur, Ag, Inputs, Reward, max_t=s['max_t'])
    return collect_episode(Ag, ValNeur, Reward, framerate=s['framerate'])


def generate_episodes(
    Ag, Env, Inputs, Reward, ValNeur,
    Sensory=None,
    nepisodes=100,
    startpos_range=((0.05, 0.05), (0.7, 0.7)),
    exploit_explore_ratio=1.0,
    episode_end_time=3.0,
    max_t=60,
    framerate=None,
    progress_bar=True,
    nworkers=1,
    seed=None,
):
    """Run a batch of **evaluation** rollouts and collect per-episode data.

    Uses `ValNeur` as its own reference value function (equivalent to
    `ref_ValNeur is ValNeur`). For training rollouts, call `do_episode`
    directly with a separate `ref_ValNeur`.

    Each episode:
      1. Per-episode state is cleared on all objects (`Ag.history`,
         `Ag.episode_data`, `Inputs/Reward/ValNeur.history`) so memory stays
         bounded even at nepisodes in the 10k range.
      2. `Ag.exploit_explore_ratio` is pinned to `exploit_explore_ratio`.
      3. `Ag.pos` is sampled uniformly from `startpos_range`.
      4. `do_episode(ValNeur, ValNeur, Ag, Inputs, Reward, max_t=max_t)`.
      5. `collect_episode` slices out (track_curr, hidden_curr).

    Any of Ag, Env, Inputs, Reward, ValNeur may be passed as a live object
    or as the corresponding info dict (from `get_*_info`). When `ValNeur` is
    a dict, it is rehydrated with `input_layers=[Inputs]` — for
    multi-input-layer ValueNeurons, rehydrate manually before calling.

    Parameters
    ----------
    Ag, Env, Inputs, Reward, ValNeur : RatInABox object or info dict
    Sensory : dict of Neurons, cell_config dict, or None
        If non-None, `compute_sensory` is called on each track and included
        in the return. Only supported when `nworkers <= 1`; parallel mode
        raises if Sensory is not None — compute sensory offline from `track`
        using the same cell_config.
    nepisodes : int
    startpos_range : ((xmin, ymin), (xmax, ymax))
    exploit_explore_ratio : float
        Re-pinned before every episode so the policy doesn't drift.
    episode_end_time : float
        Set on Reward; larger -> slower probabilistic termination at reward.
    max_t : float
        Hard timeout per episode, seconds.
    framerate : float or None
        Downsampling rate for `collect_episode`. None -> keep every step.
    progress_bar : bool
    nworkers : int
        If > 1, episodes run in parallel via a `multiprocessing.Pool`. Each
        worker rehydrates once from info dicts (produced here from the live
        objects) then runs a chunk of episodes; results come back in episode
        order via `pool.imap`. Cross-OS since `do_episode` lives in this
        module and is importable by name.
    seed : int or None
        Base RNG seed. In parallel mode, per-episode seeds are derived via
        `np.random.SeedSequence(seed).generate_state(nepisodes)`; a given
        (seed, nepisodes) reproduces the same episodes regardless of
        `nworkers`. Serial mode uses the current np.random state directly
        when seed is None. Parallel and serial are NOT bitwise-equivalent.

    Returns
    -------
    {"track": [...], "hidden": [...], "sensory": [...] or None}
    """
    if nworkers > 1 and Sensory is not None:
        raise ValueError(
            "parallel mode (nworkers > 1) requires Sensory=None. Compute "
            "sensory offline from the returned `track` via compute_sensory."
        )



    startpos_lo = np.asarray(startpos_range[0], dtype=float)
    startpos_hi = np.asarray(startpos_range[1], dtype=float)

    if nworkers <= 1:
        
        # Rehydrate any info-dict arguments. Order matters: each later object
        # may reference those rehydrated above it.
        if isinstance(Env, dict):
            Env = rehydrate_env(Env)
        if isinstance(Ag, dict):
            Ag = rehydrate_agent(Env, Ag)
        if isinstance(Inputs, dict):
            Inputs = rehydrate_placecells(Ag, Inputs)
        if isinstance(Reward, dict):
            Reward = rehydrate_placecells(Ag, Reward)
        if isinstance(ValNeur, dict):
            ValNeur = rehydrate_value_neuron(Ag, ValNeur, input_layers=[Inputs])
        # A live Sensory is also a dict -- population name -> Neurons -- so only a dict
        # whose values are themselves config dicts is treated as a cell configuration.
        if isinstance(Sensory, dict) and all(isinstance(cfg, dict) for cfg in Sensory.values()):
            Sensory = init_sensory(Ag, Env, Sensory)

        Reward.episode_end_time = episode_end_time
        if framerate is None:
            framerate = 1.0 / Ag.dt
        
        iterator = range(nepisodes)
        if progress_bar:
            iterator = tqdm.tqdm(iterator)

        track, hidden, sensory = [], [], []
        for _ in iterator:
            _reset_episode_state(Ag, Inputs, Reward, ValNeur)
            Ag.exploit_explore_ratio = exploit_explore_ratio
            Ag.pos = np.random.uniform(startpos_lo, startpos_hi, size=(2,))
            do_episode(ValNeur, ValNeur, Ag, Inputs, Reward, max_t=max_t)
            track_curr, hidden_curr = collect_episode(
                Ag, ValNeur, Reward, framerate=framerate
            )
            track.append(track_curr)
            hidden.append(hidden_curr)
            if Sensory is not None:
                sensory.append(compute_sensory(track_curr, Sensory))

        return {
            "track": track,
            "hidden": hidden,
            "sensory": sensory if Sensory is not None else None,
        }

    # Parallel path. Sensory is guaranteed None here.
    if isinstance(Env, dict):
        env_info = Env
    else:
        env_info = get_env_info(Env)
    if isinstance(Ag, dict):
        ag_info = Ag
    else:
        ag_info = get_agent_info(Ag)
    if isinstance(Inputs, dict):
        inputs_info = Inputs
    else:
        inputs_info = get_placecell_info(Inputs)
    if isinstance(Reward, dict):
        reward_info = Reward
    else:
        reward_info = get_placecell_info(Reward)
    if isinstance(ValNeur, dict):
        valneur_info = ValNeur
    else:
        valneur_info = get_value_neuron_info(ValNeur)

    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    seeds = [int(s) for s in np.random.SeedSequence(seed).generate_state(nepisodes)]

    ctx = mp.get_context()
    initargs = (
        env_info, ag_info, inputs_info, reward_info, valneur_info,
        startpos_lo.tolist(), startpos_hi.tolist(), exploit_explore_ratio,
        episode_end_time, max_t, framerate,
    )

    track, hidden = [], []
    with ctx.Pool(nworkers, initializer=_init_worker, initargs=initargs) as pool:
        chunksize = max(1, nepisodes // (nworkers * 20))
        it = pool.imap(_run_one_episode, seeds, chunksize=chunksize)
        if progress_bar:
            it = tqdm.tqdm(it, total=nepisodes)
        for track_curr, hidden_curr in it:
            track.append(track_curr)
            hidden.append(hidden_curr)

    return {"track": track, "hidden": hidden, "sensory": None}


def compute_velocity(episode):
    """
    forward_vel, sideways_vel, orientation_vel = compute_velocity(episode)
    Given an episode dict (as returned by `collect_episode`), compute the forward 
    velocity, sideways velocity, and orientation velocity between each step. 
    Forward and sideways are defined relative to the head direction of the agent, 
    which is given by episode['head_direction']. Orientation velocity is the 
    change in head direction angle between steps.
    
    Inputs:
    episode['pos'] : (T, 2)
    episode['head_direction'] : (T, 2) unit vectors 
    
    Returns
    -------
    forward_vel : array (T-1,)
    sideways_vel : array (T-1,)
    orientation_vel : array (T-1,)
    """
    pos = episode['pos']
    headdir_unitvec = episode['head_direction'] # [dx,dy]
    vel_global = np.diff(pos, axis=0)
    forward_vel = np.sum(vel_global * headdir_unitvec[:-1], axis=1)
    orthogonal_unitvec = np.stack([headdir_unitvec[:-1,1], -headdir_unitvec[:-1,0]], axis=1)
    sideways_vel = np.sum(vel_global * orthogonal_unitvec, axis=1)
    
    orientation = ratinabox.utils.get_angle(episode['head_direction'],is_array=True) # arctan2(headdir_unitvec[:,1], headdir_unitvec[:,0])
    orientation_vel = apf.utils.modrange(np.diff(orientation, axis=0),-np.pi,np.pi)
    
    return forward_vel, sideways_vel, orientation_vel


def run_policy_from_burn_in(rehydrated: dict, burn_in_pose: np.ndarray, n_frames: int,
                            n_samples: int, base_seed: int,
                            exploit_explore_ratio: float = 1.0) -> list:
    """Continue a given trajectory under the value-function policy, several times.

    The answer to "what would the rat itself have done from here?", for comparing a
    model's rollouts against the policy that produced the training data. Each sample
    replays `burn_in_pose` exactly -- so the model and the policy start from identical
    state, including the value function's own history -- and then runs free.

    The replay is done by importing the burn-in as a trajectory and stepping the agent
    along it, updating the place cells and value neuron each frame, rather than by
    teleporting the agent, so the value function sees the same input sequence it would
    have seen had it walked there.

    Args:
        rehydrated: live objects from rehydrate_policy(): 'Ag', 'Inputs', 'Reward',
            'ValNeur'.
        burn_in_pose: (n_burn_in, 3) float (x, y, theta) of the frames to replay, in the
            environment's units and radians, with theta in the fly convention (heading - pi/2,
            synthrat.sensory.ORIENTATION_OFFSET) used by the dataset poses.
        n_frames: frames to run after the burn-in.
        n_samples: independent continuations to run.
        base_seed: seed of the first sample; sample k uses base_seed + k.
        exploit_explore_ratio: how strongly the agent follows the value gradient.

    Returns:
        A list of n_samples (n_burn_in + T, 3) float arrays of (x, y, theta), theta in the
        same convention as burn_in_pose, each beginning with burn_in_pose so every sample and the ground truth align frame for
        frame.

    Side effects:
        Steps the agent and populations in `rehydrated`; the reward's episode_end_time is
        restored afterwards, but their state is left where the last sample finished.
    """
    agent = rehydrated['Ag']
    inputs, reward, value_neuron = (rehydrated['Inputs'], rehydrated['Reward'],
                                    rehydrated['ValNeur'])

    burn_in_pose = np.asarray(burn_in_pose, dtype=float)
    n_burn_in = burn_in_pose.shape[0]
    burn_in_positions = burn_in_pose[:, :2]
    burn_in_head_direction = head_direction_from_orientation(burn_in_pose[:, 2])   # (n_burn_in, 2)

    # One repeated final frame, so the last burn-in update stays inside the imported
    # trajectory's time range and does not wrap around to its start.
    times = np.arange(n_burn_in + 1) * agent.dt
    positions = np.vstack([burn_in_positions, burn_in_positions[-1:]])

    # An episode normally ends once the agent has sat on the reward long enough; here it
    # must run for a fixed number of frames instead, so that is disabled and restored.
    saved_episode_end_time = reward.episode_end_time
    reward.episode_end_time = float('inf')
    max_t = n_frames * agent.dt          # seconds after the burn-in

    samples = []
    try:
        for sample in range(n_samples):
            np.random.seed(int(base_seed + sample))
            _reset_episode_state(agent, inputs, reward, value_neuron)
            agent.exploit_explore_ratio = exploit_explore_ratio

            agent.import_trajectory(times=times, positions=positions)
            agent.pos = burn_in_positions[0].copy()
            agent.prev_pos = agent.pos.copy()
            agent.head_direction = burn_in_head_direction[0].astype(float).copy()
            first_step = burn_in_positions[1] - burn_in_positions[0]
            agent.velocity = (first_step / agent.dt).astype(float).copy()
            agent.measured_velocity = agent.velocity.copy()
            for frame in range(n_burn_in - 1):
                agent.update(dt=agent.dt)
                # Override what the import computed with the recorded pose, so the replay
                # matches the ground truth rather than an interpolation of it.
                agent.head_direction = burn_in_head_direction[frame + 1]
                step = burn_in_positions[frame + 1] - burn_in_positions[frame]
                agent.velocity = (step / agent.dt).astype(float)
                agent.measured_velocity = agent.velocity.copy()
                inputs.update()
                reward.update()
                value_neuron.update()

            agent.use_imported_trajectory = False
            _ = do_episode(value_neuron, value_neuron, agent, inputs, reward,
                           max_t=max_t, min_t=max_t)

            track, _ = collect_episode(agent, value_neuron, reward,
                                       framerate=1.0 / agent.dt)
            theta = orientation_from_head_direction(track['head_direction'])
            pose = np.concatenate([track['pos'], theta[:, None]], axis=1)   # (T, 3)
            samples.append(np.vstack([burn_in_pose, pose]))
    finally:
        reward.episode_end_time = saved_episode_end_time
        agent.use_imported_trajectory = False
    return samples


def load_state(path: str) -> dict:
    """Read a saved RatInABox state pickle.

    Args:
        path: the state pickle, holding 'env_info', 'agent_info', the place-cell and
            value-neuron snapshots, and optionally 'sensory_info'/'sensory_config'.

    Returns:
        The state dict, as saved.
    """
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def rehydrate_policy(state: dict) -> dict:
    """Rebuild every live RatInABox object an episode needs from a state dict.

    `synthrat.sensory.rehydrate_data` rebuilds the environment, agent and sensory
    populations; the policy itself -- the place-cell inputs, the reward population and
    the value neuron over them -- is rebuilt here, since only generation needs it.

    Args:
        state: a state dict from load_state().

    Returns:
        dict with 'Env', 'Ag', 'Sensory' (possibly None), 'Inputs', 'Reward', 'ValNeur'.
    """
    rehydrated = rehydrate_data(state)
    agent = rehydrated['Ag']
    rehydrated['Inputs'] = rehydrate_placecells(agent, state['inputs_placecell_info'])
    rehydrated['Reward'] = rehydrate_placecells(agent, state['reward_placecell_info'])
    rehydrated['ValNeur'] = rehydrate_value_neuron(agent, state['value_neuron_info'],
                                                   input_layers=[rehydrated['Inputs']])
    return rehydrated


def apply_sensory_config(state: dict, cell_config: dict, agent, environment) -> dict:
    """Rebuild the sensory populations from a cell configuration and record them.

    Args:
        state: the state dict, updated in place with 'sensory_config' and 'sensory_info'.
        cell_config: population name -> its RatInABox configuration, as
            synthrat.sensory.init_sensory takes.
        agent, environment: the rehydrated RatInABox Agent and Environment.

    Returns:
        The rebuilt sensory populations, as init_sensory returns them.
    """
    sensory = init_sensory(agent, environment, cell_config)
    state['sensory_config'] = cell_config
    state['sensory_info'] = get_sensory_info(sensory)
    return sensory


def generate_split(state: dict, rehydrated: dict, n_episodes: int, seed: int,
                   out_path: str, **episode_kwargs) -> dict:
    """Generate one set of episodes and write it beside the state it came from.

    The written pickle is the state dict plus this split's 'track' and 'hidden', which is
    what `experiments.synthrat.make_dataset` reads. Sensory firing rates are not computed
    here: they are the expensive part, they depend on the sensory configuration rather
    than on the episodes, and `make_dataset` computes and caches them at training time.

    Args:
        state: the state dict the episodes are generated from.
        rehydrated: the live objects from rehydrate_policy().
        n_episodes: episodes to run.
        seed: base random seed; train and validation splits must differ.
        out_path: pickle to write.
        episode_kwargs: passed through to generate_episodes (nworkers,
            exploit_explore_ratio, episode_end_time, max_t, framerate).

    Returns:
        The episodes dict from generate_episodes().

    Side effects:
        Writes out_path.
    """
    episodes = generate_episodes(
        rehydrated['Ag'], rehydrated['Env'], rehydrated['Inputs'], rehydrated['Reward'],
        rehydrated['ValNeur'],
        nepisodes=n_episodes, seed=seed, **episode_kwargs)
    written = dict(state)
    written['track'] = episodes['track']
    written['hidden'] = episodes['hidden']
    with open(out_path, 'wb') as handle:
        pickle.dump(written, handle)
    print(f"wrote {len(episodes['track'])} episodes to {out_path}", flush=True)
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--state', required=True,
                        help="state pickle holding the environment, agent and policy")
    parser.add_argument('--out-dir', default=None,
                        help="where the episode pickles go (default: the state's directory)")
    parser.add_argument('--timestamp', default=None,
                        help="suffix of the written file names (default: from the state's)")
    parser.add_argument('--train-episodes', type=int, default=10000,
                        help="episodes in the training split (0 to skip it)")
    parser.add_argument('--val-episodes', type=int, default=1000,
                        help="episodes in the validation split (0 to skip it)")
    parser.add_argument('--workers', type=int, default=1,
                        help="worker processes; episodes are independent")
    parser.add_argument('--seed-train', type=int, default=DEFAULT_SEEDS[TRAIN_SPLIT],
                        help="base seed for the training split")
    parser.add_argument('--seed-val', type=int, default=DEFAULT_SEEDS[VALIDATION_SPLIT],
                        help="base seed for the validation split")
    parser.add_argument('--exploit-explore-ratio', type=float, default=1.0,
                        help="how strongly the policy follows the value gradient")
    parser.add_argument('--episode-end-time', type=float, default=3.0,
                        help="seconds of reward contact that end an episode")
    parser.add_argument('--max-t', type=float, default=60,
                        help="seconds before an episode is cut off")
    parser.add_argument('--framerate', type=float, default=None,
                        help="frames per second the track is sampled at")
    parser.add_argument('--sensory-config', default=None,
                        help="JSON of sensory populations to rebuild before generating; "
                             "the state pickle is rewritten with them")
    args = parser.parse_args()

    state = load_state(args.state)
    out_dir = args.out_dir if args.out_dir is not None else os.path.dirname(args.state)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = args.timestamp
    if timestamp is None:
        # e.g. ratinabox_rl_state_20260423.pkl -> 20260423
        timestamp = os.path.splitext(os.path.basename(args.state))[0].rsplit('_', 1)[-1]

    rehydrated = rehydrate_policy(state)
    if args.sensory_config is not None:
        with open(args.sensory_config) as handle:
            cell_config = json.load(handle)
        rehydrated['Sensory'] = apply_sensory_config(state, cell_config, rehydrated['Ag'],
                                                     rehydrated['Env'])
        with open(args.state, 'wb') as handle:
            pickle.dump(state, handle)
        print(f"rebuilt sensory populations {sorted(cell_config)} and rewrote {args.state}",
              flush=True)

    episode_kwargs = {'nworkers': args.workers,
                      'exploit_explore_ratio': args.exploit_explore_ratio,
                      'episode_end_time': args.episode_end_time,
                      'max_t': args.max_t, 'framerate': args.framerate}
    for split, n_episodes, seed in ((TRAIN_SPLIT, args.train_episodes, args.seed_train),
                                    (VALIDATION_SPLIT, args.val_episodes, args.seed_val)):
        if n_episodes <= 0:
            continue
        print(f"generating {n_episodes} {split} episodes...", flush=True)
        generate_split(state, rehydrated, n_episodes, seed,
                       os.path.join(out_dir, f"ratinabox_rl_{split}data_{timestamp}.pkl"),
                       **episode_kwargs)


if __name__ == '__main__':
    main()
