"""Utilities for saving, rebuilding, extending, and plotting rollouts of a
RatInABox reinforcement-learning agent.

This module bundles a few things that work together but are independent:

1. Per-episode rollout capture
    `collect_episode` slices the continuous `.history` dicts of the live
    Agent, ValueNeuron, and Reward objects for the most recent episode and
    returns `(track_curr, hidden_curr)` — two dicts of per-timestep arrays.
    Run it right after each `do_episode(...)` call and collect the returns
    yourself (e.g. into lists or a dict of lists).
    `generate_episodes` wraps this in a full eval-rollout batch: it samples
    random start positions, calls `do_episode` + `collect_episode` for each
    of `nepisodes` iterations, pins `exploit_explore_ratio` each step, and
    returns parallel lists `(track, hidden)`. It hardcodes `train=False`
    and uses `ValNeur` as its own reference — intended for pure evaluation.

2. Static-config snapshots
    `get_info`, `get_agent_info`, `get_env_info`, `get_placecell_info`
    return dicts capturing the construction params of a RatInABox object
    plus the instance attributes that can be mutated after init (e.g.
    `Env.walls`, `pc.place_cell_widths`, custom notebook attributes like
    `Ag.exploit_explore_ratio` or `Reward.episode_end_time`). Arrays are
    copied so later in-place edits don't silently mutate your saved dicts.

3. Object rehydration
    `rehydrate_env`, `rehydrate_agent`, `rehydrate_placecells`,
    `rehydrate_value_neuron`, and `rehydrate_sensory` rebuild an
    Environment, Agent, PlaceCells population, ValueNeuron, or full
    dict of sensory populations from the dicts produced by the
    corresponding `get_*_info` helpers. Each filters the input dict to
    valid params keys (ignoring derived attributes like `Env.extent` or
    custom attributes like `Ag.exploit_explore_ratio`), then restores
    mutated state, learned weights, and custom attributes post-construction.
    `rehydrate_value_neuron` also requires the live input-layer Neurons
    objects. `rehydrate_sensory` overwrites per-cell tuning attributes
    after construction so that `cell_arrangement="random"` populations
    are restored exactly (bypassing the constructor's fresh random draws).

4. Offline sensory-neuron firing rates
    `init_sensory` takes an Agent + Environment (either live objects or the
    corresponding info dicts) and a `cell_config` describing desired sensory
    populations (BoundaryVectorCells, FieldOfViewBVCs, HeadDirectionCells,
    VelocityCells, SpeedCell, etc.). It instantiates each population once
    (so randomly sampled tuning parameters stay fixed across episodes) and
    returns a `Sensory` dict mapping population name -> Neurons.
    `compute_sensory` then applies those populations to a saved trajectory
    (`track_curr` with `pos`, `head_direction`, and optionally `vel`) to
    produce a dict `{name: (T, n_cells)}`. Pass the SAME Ag used in
    `init_sensory` — the Sensory neurons hold an internal reference to it.
    Vectorizes position-dependent allocentric cells; loops over T for the rest.

5. Hand-built polar cell grid
    `rect_polar_grid` is a `cell_arrangement` callable for vector cells
    that lays out a rectangular n_rings × n_angles grid in polar (d, theta)
    coordinates. Pass it as `"cell_arrangement": rect_polar_grid` in a
    cell_config entry when you want a firing-rate output that reshapes
    cleanly to `(T, n_rings, n_angles)`.

6. Plotting
    `plot_episode` draws one saved trajectory on top of the Env, with the
    current agent position and head-direction marker.
    `visualize_sensory` takes a trajectory + its computed sensory firing
    rates + a timestep `t`, and produces a composite figure: trajectory
    and agent state (position, heading, velocity) on the left, one polar
    subplot per sensory population on the right showing firing rates at
    time t against each cell's preferred tuning direction.

Typical use from a notebook
---------------------------
    import apf_ratinabox as apf

    # 1) snapshot static config once, up front
    env_info     = apf.get_env_info(Env)
    agent_info   = apf.get_agent_info(Ag)
    inputs_info  = apf.get_placecell_info(Inputs)
    reward_info  = apf.get_placecell_info(Reward)

    # 2) run a batch of evaluation rollouts
    track, hidden = apf.generate_episodes(
        Ag, Env, Inputs, Reward, ValNeur,
        do_episode, nepisodes=100,
    )
    # or, one episode at a time:
    #   do_episode(ref_ValNeur, ValNeur, Ag, Inputs, Reward, train=False)
    #   track_curr, hidden_curr = apf.collect_episode(Ag, ValNeur, Reward)

    # 3) compute sensory neurons offline from the saved trajectories
    cell_config = {
        "bvc_ego": {"type": "BoundaryVectorCells", "n": 16,
                    "reference_frame": "egocentric"},
        "hdc":     {"type": "HeadDirectionCells", "n": 10,
                    "angular_spread_degrees": 30},
        "speed":   {"type": "SpeedCell"},
    }
    # init_sensory takes a live (or info-dict) Ag/Env and returns the
    # Neurons populations. compute_sensory recovers the Agent from Sensory
    # internally, so no need to pass it separately.
    Sensory = apf.init_sensory(agent_info, env_info, cell_config)
    sensory = [apf.compute_sensory(tc, Sensory) for tc in track]
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp


# Make the sibling ratinabox/ checkout (synthrat/ratinabox/ratinabox/) importable
# without installing it into site-packages, so `ratinabox` is only reachable via
# this module rather than globally from AnimalPoseForecasting.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ratinabox"))

import ratinabox
import ratinabox.utils

# NB: `ratinabox/__init__.py` does `from .Neurons import *`, `.Environment
# import *`, `.Agent import *`, and `from . import contribs` (whose own
# __init__ stars ValueNeuron in). After that, at the top level:
#   ratinabox.Environment, ratinabox.Agent, ratinabox.PlaceCells,
#   ratinabox.BoundaryVectorCells, ..., ratinabox.contribs.ValueNeuron
# are all the CLASSES (the submodule bindings were shadowed by the stars).
# So we access classes directly under `ratinabox.`, not `ratinabox.Neurons.`.

_CELL_TYPES = {
    "BoundaryVectorCells": ratinabox.BoundaryVectorCells,
    "ObjectVectorCells":   ratinabox.ObjectVectorCells,
    "FieldOfViewBVCs":     ratinabox.FieldOfViewBVCs,
    "FieldOfViewOVCs":     ratinabox.FieldOfViewOVCs,
    "HeadDirectionCells":  ratinabox.HeadDirectionCells,
    "VelocityCells":       ratinabox.VelocityCells,
    "SpeedCell":           ratinabox.SpeedCell,
}

CELL_VECTOR_CLASSES = {"BoundaryVectorCells", "ObjectVectorCells", "FieldOfViewBVCs", "FieldOfViewOVCs"}


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

def rehydrate_env(env_info):
    """Reconstruct an `Environment` from the static info dict saved by
    your `get_env_info`.

    Forwards any key that is a valid `Environment` params key; ignores the
    rest (e.g. derived attributes like `extent`, `is_rectangular`). Walls
    are set by direct overwrite after construction so the saved geometry
    (including in-place edits like the shortcut demo in the notebook) is
    preserved exactly.
    """
    valid_keys = set(ratinabox.Environment.default_params.keys())
    params = {k: env_info[k] for k in valid_keys if k in env_info}
    # Don't pass walls through the constructor — we'll overwrite after.
    params.pop("walls", None)
    # Back-compat: older snapshots may have captured the post-init dict form
    # of objects ({"objects": array, "object_types": array}). Convert to the
    # list-of-positions form the constructor expects.
    if isinstance(params.get("objects"), dict):
        pos = params["objects"].get("objects")
        if pos is not None and hasattr(pos, "tolist"):
            pos = pos.tolist()
        params["objects"] = pos or []

    Env = ratinabox.Environment(params=params)

    if "walls" in env_info and env_info["walls"] is not None:
        Env.walls = np.asarray(env_info["walls"], dtype=float).copy()

    return Env


def rehydrate_agent(Env, agent_info):
    """Reconstruct an `Agent` from the dict saved by `get_agent_info`.

    Forwards any key that is a valid `Agent` params key; ignores the rest
    (e.g. your custom `exploit_explore_ratio`). After construction, restores
    any custom attributes from `agent_info` that aren't real Agent params.

    Parameters
    ----------
    Env : ratinabox Environment
        The (already rehydrated) env this agent belongs to.
    agent_info : dict
        Produced by `get_agent_info`. Expected to include standard Agent
        params (dt, speed_mean, speed_std, coherence times, wall_repel_*,
        thigmotaxis, ...) and optionally `exploit_explore_ratio`.

    Returns
    -------
    Ag : the reconstructed Agent.
    """
    valid_keys = set(ratinabox.Agent.default_params.keys())
    params = {k: v for k, v in agent_info.items() if k in valid_keys}
    Ag = ratinabox.Agent(Env, params=params)

    # Restore custom (non-params) attributes:
    if "exploit_explore_ratio" in agent_info:
        Ag.exploit_explore_ratio = agent_info["exploit_explore_ratio"]

    return Ag


def rehydrate_placecells(Ag, pc_info):
    """Reconstruct a `PlaceCells` population from the dict saved by
    `get_placecell_info`.

    Forwards any key that is a valid PlaceCells params key (walking the
    full class inheritance via `ratinabox.utils.collect_all_params`), then
    restores `place_cell_widths` exactly (bypassing the `widths * ones(n)`
    derivation in `__init__`) and any custom attributes like
    `episode_end_time`.

    Parameters
    ----------
    Ag : ratinabox Agent
        The agent this population attaches to.
    pc_info : dict
        Produced by `get_placecell_info`. `widths` is expected to hold the
        live per-cell widths array (as `get_placecell_info` stores it).

    Returns
    -------
    pc : the reconstructed PlaceCells.
    """
    valid_keys = set(ratinabox.utils.collect_all_params(
        ratinabox.PlaceCells
    ).keys())
    params = {k: v for k, v in pc_info.items() if k in valid_keys}
    pc = ratinabox.PlaceCells(Ag, params=params)

    # Ensure per-cell widths exactly match the saved array (not just
    # element-wise equal via `widths * np.ones(n)` inside __init__).
    if "widths" in pc_info:
        pc.place_cell_widths = np.asarray(pc_info["widths"]).copy()

    # Restore custom (non-params) attributes:
    if "episode_end_time" in pc_info:
        pc.episode_end_time = pc_info["episode_end_time"]

    return pc


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


def rehydrate_value_neuron(Ag, valneur_info, input_layers):
    """Reconstruct a `ValueNeuron` from a `get_value_neuron_info` snapshot.

    Parameters
    ----------
    Ag : ratinabox Agent
        The (already rehydrated or live) agent this ValueNeuron attaches to.
    valneur_info : dict
        Produced by `get_value_neuron_info`.
    input_layers : list of ratinabox Neurons
        The input populations this ValueNeuron sums over. Must be live
        objects (e.g. your rehydrated Inputs). Order and names should
        match the snapshot's `valneur_info["inputs"]` dict keys; weights
        are restored by name.

    Returns
    -------
    ValNeur : the reconstructed ValueNeuron with learned weights restored.
    """
    VN_cls = ratinabox.contribs.ValueNeuron
    valid_keys = set(ratinabox.utils.collect_all_params(VN_cls).keys())
    params = {k: v for k, v in valneur_info.items()
              if k in valid_keys and k != "input_layers"}
    params["input_layers"] = input_layers
    ValNeur = VN_cls(Ag, params=params)

    # Restore learned weights by layer name.
    for name, saved in valneur_info.get("inputs", {}).items():
        if name in ValNeur.inputs:
            ValNeur.inputs[name]["w"] = np.asarray(saved["w"]).copy()

    if "max_value" in valneur_info:
        ValNeur.max_value = valneur_info["max_value"]

    return ValNeur


def rehydrate_data(data):
    res = {}
    res['Env'] = rehydrate_env(data['env_info'])
    res['Ag'] = rehydrate_agent(res['Env'], data['agent_info'])
    if "inputs_placecell_info" in data:
        res['Inputs'] = rehydrate_placecells(res['Ag'], data['inputs_placecell_info'])
    if "reward_placecell_info" in data:
        res['Reward'] = rehydrate_placecells(res['Ag'], data['reward_placecell_info'])
    if "value_neuron_info" in data:
        assert "inputs_placecell_info" in data, "rehydrating ValueNeuron requires the place cell info for its input layers"
        res['ValNeur'] = rehydrate_value_neuron(res['Ag'], data['value_neuron_info'], input_layers=[res['Inputs']])
    if "sensory_info" in data:
        res['Sensory'] = rehydrate_sensory(res['Ag'], data['sensory_info'])
    return res


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
                                                                                                                                                        
        # SpeedCell: save Agent-derived scale
        if hasattr(neuron, "one_sigma_speed"):                                                                                                        
            info["one_sigma_speed"] = neuron.one_sigma_speed
                                                                          
        sensory_info[k] = info
    return sensory_info


def rehydrate_sensory(Ag, sensory_info):
    """Inverse of `get_sensory_info`: rebuild a `Sensory` dict from the
    saved snapshot, attached to the given Agent.

    For each population:
      1. Look up the class via `info["cls_name"]`.
      2. Filter the info dict to valid params keys for that class, using
         `ratinabox.utils.collect_all_params(cls)` to walk the inheritance.
      3. Instantiate the Neurons population.
      4. Overwrite any saved per-cell attributes (tuning_distances,
         tuning_angles, sigma_distances, sigma_angles, preferred_angles,
         angular_tunings, one_sigma_speed, n) so the rehydrated population
         reproduces the saved state exactly — even when the original
         used `cell_arrangement="random"` which would otherwise re-draw
         tuning on each construction.

    Parameters
    ----------
    Ag : ratinabox Agent
        The (already rehydrated or live) agent the populations attach to.
        Must match the Agent used at `init_sensory` time (SpeedCell and
        VelocityCells read `Ag.speed_mean + Ag.speed_std` at construction,
        though we overwrite `one_sigma_speed` afterwards if it was saved).
    sensory_info : dict {name: info_dict}
        Produced by `get_sensory_info`.

    Returns
    -------
    Sensory : dict {name: Neurons} — shape-compatible with `init_sensory`'s
        return, so it can be passed to `compute_sensory` directly.
    """
    Sensory = {}
    # attributes that, when present in the saved info, override the
    # constructor-time values (bypasses random sampling, broadcast logic, etc.)
    _override_attrs = (
        "tuning_distances", "tuning_angles",
        "sigma_distances",  "sigma_angles",
        "preferred_angles", "angular_tunings",
        "one_sigma_speed",
        "n",
    )

    for name, info in sensory_info.items():
        cls_name = info["cls_name"]
        if cls_name not in _CELL_TYPES:
            raise ValueError(
                f"unknown cell type {cls_name!r}; "
                f"available: {list(_CELL_TYPES)}"
            )
        cls = _CELL_TYPES[cls_name]
        valid_keys = set(ratinabox.utils.collect_all_params(cls).keys())
        params = {k: v for k, v in info.items() if k in valid_keys}
        neuron = cls(Ag, params=params)

        # Overwrite per-cell realized attributes so the population is
        # exactly what we saved (in particular, replaces any random draws
        # that the constructor just took).
        for attr in _override_attrs:
            if attr in info:
                setattr(neuron, attr, info[attr].copy()
                        if hasattr(info[attr], "copy") else info[attr])

        Sensory[name] = neuron

    return Sensory


# ---------------- 
# copied from ratinabox.reinforcement learning example
# ---------------- 

# a function which takes a value neuron and a position and returns the direction of steepest ascent of the value function at that position
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

def do_episode(ref_ValNeur, ValNeur, Ag, Inputs, Reward, max_t=60):
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

        # end episode when at some random moment when reward is high OR after timeout
        if np.random.uniform() < Ag.dt * Reward.firingrate / Reward.episode_end_time:
            Ag.exploit_explore_ratio *= 1.1  # policy gets greedier if it was successful
            Ag.episode_data["success_or_failure"].append(1)
            break
        if (Ag.t - Ag.episode_data["start_time"][-1]) > max_t:  # timeout
            Ag.episode_data["success_or_failure"].append(0)
            break
    Ag.episode_data["end_time"].append(Ag.t)
    Ag.episode_data["end_pos"].append(Ag.pos)
    Ag.exploit_explore_ratio = max(0.1, min(1, Ag.exploit_explore_ratio)) #keep between 0.1 and 1
    Ag.velocity = np.random.uniform(-0.1, 0.1, size=(2,))
    return

# ----------------
# parallel helpers
# ----------------

_WORKER_STATE = None


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
        if isinstance(Sensory, dict):
            Sensory = init_sensory(Ag, Env, Sensory)

        Reward.episode_end_time = episode_end_time
        if framerate is None:
            framerate = 1.0 / Ag.dt
        
        iterator = range(nepisodes)
        if progress_bar:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator)
            except ImportError:
                pass

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
            try:
                from tqdm.auto import tqdm
                it = tqdm(it, total=nepisodes)
            except ImportError:
                pass
        for track_curr, hidden_curr in it:
            track.append(track_curr)
            hidden.append(hidden_curr)

    return {"track": track, "hidden": hidden, "sensory": None}


def init_sensory(Ag, Env, cell_config):
    """Instantiate a dict of sensory Neurons populations attached to `Ag`/`Env`.

    Populations are constructed **once** here so their randomly sampled tuning
    parameters (BVC preferred angles/distances, HDC preferred directions, etc.)
    stay fixed across subsequent `compute_sensory` calls on different episodes.

    Parameters
    ----------
    Ag : ratinabox Agent or agent_info dict
        If a dict, rehydrated via `rehydrate_agent(Env, ...)`. VelocityCells
        and SpeedCell read `Ag.speed_mean + Ag.speed_std` for their speed
        tuning, so this should match the agent that produced the rollouts.
    Env : ratinabox Environment or env_info dict
        If a dict, rehydrated via `rehydrate_env`. Boundary/object vector
        cells use `Env.walls` / `Env.objects` for their firing-rate calcs.
    cell_config : dict
        Maps population name -> dict with key 'type' (one of the keys of
        `_CELL_TYPES`) and any params to pass to that Neurons class.
        Example:
            {"bvc_ego": {"type": "BoundaryVectorCells", "n": 16,
                         "reference_frame": "egocentric"},
             "hdc":     {"type": "HeadDirectionCells", "n": 10},
             "speed":   {"type": "SpeedCell"}}

    Returns
    -------
    Sensory : dict mapping population name -> live Neurons instance. Each
        neuron holds an internal reference to `Ag`/`Env`, so pass the SAME
        `Ag` to `compute_sensory` later (VelocityCells reads `Ag.velocity`
        directly).
    """
    
    if isinstance(Env, dict):
        Env = rehydrate_env(Env)
    if isinstance(Ag, dict):
        Ag = rehydrate_agent(Env, Ag)

    Sensory = {}
    for name, cfg in cell_config.items():
        cls_name = cfg["type"]
        if cls_name not in _CELL_TYPES:
            raise ValueError(
                f"unknown cell type {cls_name!r}; "
                f"available: {list(_CELL_TYPES)}"
            )
        cls = _CELL_TYPES[cls_name]
        params = {}
        for k, v in cfg.items():
            if k == "type":
                continue
            elif k == 'cell_arrangement' and v == 'rect_polar_grid':
                params[k] = rect_polar_grid
            else:
                params[k] = v
        
        Sensory[name] = cls(Ag, params=params)

    return Sensory


def compute_sensory(track_curr, Sensory):
    """Compute firing rates along one trajectory for all sensory populations.

    Parameters
    ----------
    track_curr : dict
        One episode's trajectory, as returned by `collect_episode`.
        Required keys:
            'pos'            : array (T, 2)
            'head_direction' : array (T, 2)   (unit vectors)
        Optional (needed for VelocityCells and SpeedCell):
            'vel'            : array (T, 2)
    Sensory : dict
        Produced by `init_sensory`. Maps population name -> live Neurons.
        The Agent used by VelocityCells (for the per-step `Ag.velocity` write)
        is recovered internally from `Sensory`'s first entry, guaranteeing
        consistency with the Agent the populations were attached to.

    Returns
    -------
    dict {name: (T, n_cells) firing-rate array}.
    """
    pos      = np.asarray(track_curr["pos"])
    head_dir = np.asarray(track_curr["head_direction"])
    vel      = np.asarray(track_curr["vel"]) if "vel" in track_curr else None
    T = pos.shape[0]

    # All Sensory neurons were attached to the same Ag at init time; recover it.
    Ag = next(iter(Sensory.values())).Agent

    out = {}
    for name, neuron in Sensory.items():
        out[name] = _firingrate_over_trajectory(
            neuron, Ag, pos, head_dir, vel, T
        )
    return out


def _firingrate_over_trajectory(neuron, Ag, pos, head_dir, vel, T):
    """Compute `(T, n_cells)` firing rates for one neuron population.

    Internal dispatch used by `compute_sensory`. Vectorizes over the full
    trajectory when the cell's `get_state` supports an array-valued `pos`
    and a single broadcast-compatible head direction (allocentric vector
    cells); otherwise loops over timesteps.

    Parameters
    ----------
    neuron : ratinabox Neurons instance
    Ag : the dummy Agent that `neuron` was attached to — used to set
        `Ag.velocity` per step for VelocityCells (whose `get_state` reads
        `self.Agent.velocity` directly for the speed scale).
    pos : (T, 2) array
    head_dir : (T, 2) array of unit vectors
    vel : (T, 2) array or None (required only for Velocity/Speed cells)
    T : int, number of timesteps
    """
    cls_name = type(neuron).__name__

    # BVCs / OVCs / FieldOfView*: pos is vectorized over T; egocentric variants
    # only accept a single head_direction per call, so we loop for those.
    if cls_name in (
        "BoundaryVectorCells", "ObjectVectorCells",
        "FieldOfViewBVCs", "FieldOfViewOVCs",
    ):
        is_ego = getattr(neuron, "reference_frame", "allocentric") == "egocentric"
        if not is_ego:
            fr = neuron.get_state(evaluate_at=None, pos=pos)  # (n_cells, T)
            return np.asarray(fr).T                           # (T, n_cells)

        out = np.empty((T, neuron.n))
        for t in range(T):
            fr = neuron.get_state(
                evaluate_at=None,
                pos=pos[t:t + 1],
                head_direction=head_dir[t],
            )
            out[t] = np.asarray(fr).ravel()
        return out

    if cls_name == "HeadDirectionCells":
        out = np.empty((T, neuron.n))
        for t in range(T):
            fr = neuron.get_state(
                evaluate_at=None,
                head_direction=head_dir[t],
            )
            out[t] = np.asarray(fr).ravel()
        return out

    if cls_name == "VelocityCells":
        if vel is None:
            raise ValueError("VelocityCells require track_curr['vel']")
        # VelocityCells.get_state reads self.Agent.velocity directly to get the
        # speed scale, so we must set Ag.velocity on each step as well as
        # passing it via kwargs.
        out = np.empty((T, neuron.n))
        for t in range(T):
            Ag.velocity = np.asarray(vel[t], dtype=float)
            fr = neuron.get_state(evaluate_at=None, velocity=vel[t])
            out[t] = np.asarray(fr).ravel()
        return out

    if cls_name == "SpeedCell":
        if vel is None:
            raise ValueError("SpeedCell requires track_curr['vel']")
        out = np.empty((T, neuron.n))
        for t in range(T):
            fr = neuron.get_state(evaluate_at=None, vel=vel[t])
            out[t] = np.asarray(fr).ravel()
        return out

    raise TypeError(f"unsupported cell class: {cls_name}")

def rect_polar_grid(distance_range, angle_range, n_rings, n_angles, **_):
    """`cell_arrangement` callable for a rectangular polar grid of vector cells.

    Pass this as `"cell_arrangement": rect_polar_grid` in a vector-cell
    config (with extra keys `n_rings` and `n_angles`). Unlike RatInABox's
    built-in `uniform_manifold` / `diverging_manifold`, which produce
    variable cells-per-ring, this lays out a fixed n_rings × n_angles
    grid so the resulting firing-rate vector reshapes cleanly to
    `(T, n_rings, n_angles)`.

    Receptive-field widths are set to half the inter-cell spacing in both
    radial and angular directions (rough tiling with ~1σ overlap).

    Parameters
    ----------
    distance_range : (d_min, d_max)
        Ring radii, in metres.
    angle_range : (θ_min, θ_max)
        Bearings in degrees, matching RatInABox's convention: `angle_range=[0, θ]`
        means a symmetric ±θ FoV (total angular extent `2θ`). `θ_min` is
        typically 0 (cells start at straight ahead); setting `θ_min > 0`
        creates a blind spot directly ahead.
    n_rings : number of concentric rings
    n_angles : number of angular bins per ring (tiled symmetrically across ±θ_max)

    Returns
    -------
    (mu_d, mu_theta, sigma_d, sigma_theta) : four length-(n_rings*n_angles) arrays
        Preferred distance (m), preferred bearing (rad), radial std (m),
        angular std (rad). Flattened in ring-major order (ring 0 first,
        so `reshape(n_rings, n_angles)` recovers the grid).
    """
    r = np.linspace(distance_range[0], distance_range[1], n_rings)
    # Symmetric about heading to match RatInABox's built-in manifold convention.
    theta_max = np.deg2rad(angle_range[1])
    t = np.linspace(-theta_max, theta_max, n_angles)
    dd, tt = np.meshgrid(r, t, indexing="ij")
    mu_d, mu_theta = dd.ravel(), tt.ravel()
    d_step = (distance_range[1] - distance_range[0]) / max(n_rings - 1, 1)
    t_step = (2 * theta_max) / max(n_angles - 1, 1)
    sigma_d     = np.full_like(mu_d, d_step / 2)
    sigma_theta = np.full_like(mu_d, t_step / 2)
    return mu_d, mu_theta, sigma_d, sigma_theta

def plot_episode(track_curr,Env,axcurr=None,**kwargs):
    """Plot one episode's trajectory on top of the environment.

    Draws the path as semi-transparent dots, the final agent position as a
    solid dot, and a triangle marker rotated to show the final head direction.

    Parameters
    ----------
    track_curr : dict
        Must contain 'pos' (T, 2) and 'head_direction' (T, 2).
    Env : ratinabox Environment
        Drawn in the background via `Env.plot_environment`.
    axcurr : matplotlib Axes, optional
        If None, uses `plt.gca()`.
    **kwargs :
        alpha       : float (default 0.7) — transparency of trajectory dots
        point_size  : float (default 15)  — size of trajectory dots
        agent_color : str   (default "r") — color of final position + heading

    Returns
    -------
    (htraj, hagent, hd) : PathCollection handles for the trajectory, the
        final-position dot, and the head-direction triangle, in that order.
    """
    alpha = kwargs.get("alpha", 0.7) #transparency of trajectory
    point_size = kwargs.get("point_size", 15) #size of trajectory points
    agent_color = kwargs.get("agent_color", "r") #color of the agent if show_agent is True
    
    if axcurr is None:
        axcurr = plt.gca()
    fig = axcurr.figure
    trajectory = track_curr['pos']
    head_direction = track_curr['head_direction']
    _, _ = Env.plot_environment(fig=fig, ax=axcurr, autosave=False)
    htraj = axcurr.scatter(
        trajectory[:-1, 0],
        trajectory[:-1, 1],
        s=point_size,
        linewidth=0,
        alpha=alpha,
    )
    hagent = axcurr.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        s=40,
        c=agent_color,
        linewidth=0,
        marker="o",
    )
    rotated_agent_marker = matplotlib.markers.MarkerStyle(marker=[(-1,0),(1,0),(0,4)]) # a triangle
    rotated_agent_marker._transform = rotated_agent_marker.get_transform().rotate_deg(-ratinabox.utils.get_bearing(head_direction[-1])*180/np.pi)
    hd = axcurr.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        s=200,
        alpha=1,
        c=agent_color,
        linewidth=0,
        marker=rotated_agent_marker,
    )
    return htraj, hagent, hd

def plot_agent_heading(axcurr, pos, head_dir, triheight=5, triwidth=4,
                       s=600, c='r', edgecolor='r', linewidth=0, zorder=5):
    """Draw a rotated triangle marker indicating the agent's heading.

    The triangle is defined with its base along y=0 and tip at (0, triheight),
    then rotated by `-get_bearing(head_dir)` degrees (compass convention)
    so the tip points in the heading direction.

    Parameters
    ----------
    axcurr : matplotlib Axes (Cartesian)
    pos : (2,) world-frame position where the triangle is placed.
    head_dir : (2,) unit vector in world frame.
    triheight, triwidth : float
        Dimensions of the unit-marker triangle before scatter scales by `s`.
        Larger `triheight` / smaller `triwidth` → more arrow-like.
    s : float, scatter marker area in points².
    c, edgecolor, linewidth : standard matplotlib scatter styling.
    zorder : float, draw order.
    """
    hd_marker = matplotlib.markers.MarkerStyle(marker=[(-triwidth/2, 0), (triwidth/2, 0), (0, triheight)])
    hd_marker._transform = hd_marker.get_transform().rotate_deg(
        -ratinabox.utils.get_bearing(head_dir) * 180 / np.pi
    )
    axcurr.scatter(pos[0], pos[1],
                    s=s, c=c,
                    edgecolor=edgecolor, linewidth=linewidth,
                    marker=hd_marker, zorder=zorder)


def plot_agent_state(axcurr, pos, head_dir, vel, t):
    """Draw the agent's position, heading, and velocity at timestep `t`.

    Rendered elements (in draw order):
      - heading triangle (orange C1) via `plot_agent_heading`
      - position dot (black)
      - velocity arrow (green C2) via `quiver` — only if `vel` is provided

    Parameters
    ----------
    axcurr : matplotlib Axes (Cartesian)
    pos : (T, 2) array of positions over the episode.
    head_dir : (T, 2) array of unit heading vectors.
    vel : (T, 2) array of velocities, or None.
    t : int, index into the arrays above.
    """
    plot_agent_heading(axcurr, pos[t], head_dir[t], triheight=5, triwidth=4,
                       s=2000, c='C1', edgecolor='C1', linewidth=1.5, zorder=4)
    axcurr.scatter(pos[t, 0], pos[t, 1],
                    s=80, c="k", linewidth=0, zorder=5)

    if vel is not None:
        axcurr.quiver(pos[t, 0], pos[t, 1], vel[t, 0], vel[t, 1],
                        angles="xy", scale_units="xy", scale=1,
                        color="C2", width=0.005, zorder=6)
        
def plot_head_direction(axcurr, neuron, fr, head_dir=None, head_bearing=None):
    """Plot an HDC/VelocityCells population on a polar axis as von-Mises bumps.

    Each cell is drawn as a smooth curve peaking at its preferred angle, with
    peak height = firing rate `fr[j]` and angular width set by
    `neuron.angular_tunings[j]` (via κ = 1/σ²). An orange (C1) radial line
    marks the current heading direction so you can see how well-aligned the
    population response is with the agent's actual heading.

    Convention: the polar axis is set to the **compass convention** (0° at
    North/top, angles increasing clockwise), matching the Cartesian trajectory
    panel's +y=up orientation. Because `preferred_angles` are math-convention
    (CCW from East) while `get_bearing` is already compass (CW from North),
    the former are converted via `π/2 - angle` and the latter passes through
    unchanged.

    Parameters
    ----------
    axcurr : matplotlib polar Axes (must have projection='polar').
    neuron : live HeadDirectionCells or VelocityCells instance.
    fr : (n,) firing rates at the timestep of interest.
    head_dir : (2,) unit vector OR
    head_bearing : float (radians, compass convention).
        Exactly one of these must be provided. If `head_bearing` is given it
        is used directly, otherwise `get_bearing(head_dir)` is called.
    """
    if head_bearing is None:
        head_bearing = ratinabox.utils.get_bearing(head_dir)  # radians
    
    preferred = np.asarray(neuron.preferred_angles)   # (n,) radians, math (CCW from E)
    tunings   = np.asarray(neuron.angular_tunings)    # (n,) radians
    # Compass convention: 0° at N (top), increasing clockwise — so the
    # polar panel visually aligns with the Cartesian trajectory plot
    # (up = +y = N).
    #   - preferred_angles are math-convention (CCW from E); convert
    #     to compass via  plot_angle = π/2 - angle.
    #   - head_bearing (from ratinabox.utils.get_bearing) is ALREADY
    #     compass-convention (CW from N); pass it through unchanged.
    axcurr.set_theta_zero_location("N")
    axcurr.set_theta_direction(-1)
    preferred_plot    = np.pi / 2 - preferred
    head_bearing_plot = head_bearing
    # Plot each cell as a von-Mises bump: peak at preferred_angle,
    # peak height = firing rate, angular width set by angular_tunings.
    theta_grid = np.linspace(-np.pi, np.pi, 360)
    for j in range(neuron.n):
        kappa = 1.0 / max(tunings[j]**2, 1e-6)
        bump = fr[j] * np.exp(kappa * (np.cos(theta_grid - preferred_plot[j]) - 1))
        axcurr.plot(theta_grid, bump,
                        color='C0', linewidth=2)
    r_max = max(fr.max(), 1e-6)
    axcurr.plot([head_bearing_plot, head_bearing_plot],[0, r_max * 1.05],color="C1", linewidth=2)

def get_ax_lims(pos, Env, Sensory):
    """Compute x/y axis limits that show the environment plus all sensory
    ellipses for the trajectory range.

    Takes the max (tuning_distance + sigma_distance) across any vector-cell
    populations in `Sensory` as `reach` — the farthest any ellipse can
    extend from the agent. Then returns limits that union the environment
    extent with the trajectory's bounding box padded by `reach`.

    Parameters
    ----------
    pos : array of positions. Any shape with the last axis of length 2
        (e.g. (T, 2) for one episode, or (..., 2) more generally); the
        bounding box is computed over all dimensions except the last.
    Env : ratinabox Environment — used for `Env.extent = [xmin, xmax, ymin, ymax]`.
    Sensory : dict {name: Neurons} — scanned for vector-cell populations
        via their class name being in `CELL_VECTOR_CLASSES`.

    Returns
    -------
    (xmin, xmax), (ymin, ymax) : tuples of float suitable for
        `ax.set_xlim(...)` / `ax.set_ylim(...)`.
    """
    extent = np.asarray(Env.extent)
    reach = 0
    for neuron in Sensory.values():
        cls_name = type(neuron).__name__
        if cls_name in CELL_VECTOR_CLASSES:
            reachcurr = float(np.max(np.asarray(neuron.tuning_distances) + np.asarray(neuron.sigma_distances)))
            reach = max(reach, reachcurr)
            
    xmin = min(extent[0], np.min(pos[...,0]) - reach)
    xmax = max(extent[1], np.max(pos[...,0]) + reach)
    ymin = min(extent[2], np.min(pos[...,1]) - reach)
    ymax = max(extent[3], np.max(pos[...,1]) + reach)

    return (xmin,xmax), (ymin, ymax)

def visualize_sensory(track_curr, sensory_curr, t, Env, Sensory, fig=None, ax=None,
                      ax_xlim=None, ax_ylim=None):
    """Snapshot visualization at one timestep: trajectory + agent state +
    firing rates of each sensory population.

    Left panel: environment with trajectory up to time t, current agent
    position (red dot), heading (rotated triangle), and velocity (green
    arrow). Vector-cell populations (BVC/OVC/FieldOfView*) are overlaid
    on this panel using RatInABox's `Neurons.display_vector_cells`, which
    draws each cell as an ellipse at its preferred (d, θ) with fill alpha
    scaled by firing rate and size equal to the 1σ receptive-field extent.
    Multiple vector populations overlay on the same axis using each
    population's `.color` attribute.

    Right column: one polar subplot per non-vector population:
      - HeadDirectionCells: polar bar at preferred angles; red line = heading.
      - VelocityCells: same as HDC (bars scale by speed).
      - SpeedCell: horizontal bar, length = firing rate.
      - Any other class: fallback bar plot vs cell index.

    Because RatInABox's `display_vector_cells` reads from
    `neuron.Agent.history` and `neuron.history`, this function populates
    a single-timestep history on each vector-cell neuron before calling it.
    That mutates the live neurons' history dicts — if you also train/update
    these same Sensory objects, the history will be the one-timestep stub
    until the next `update()` appends to it.

    Parameters
    ----------
    track_curr : dict with 'pos' (T, 2), 'head_direction' (T, 2), and
        (optionally) 'vel' (T, 2).
    sensory_curr : dict {name: (T, n_cells)} from `compute_sensory`.
    t : int, timestep to visualize.
    Env : ratinabox Environment (live or rehydrated).
    Sensory : dict {name: Neurons} from `init_sensory`.
    fig : matplotlib Figure, optional. If None, a new figure is created.

    Returns
    -------
    fig : the matplotlib Figure.
    """
    pos      = np.asarray(track_curr["pos"])
    head_dir = np.asarray(track_curr["head_direction"])
    vel      = np.asarray(track_curr["vel"]) if "vel" in track_curr else None

    # Split populations into "vector" (overlaid on trajectory) vs "other"
    # (separate right-column subplot).
    n_cell_types = len(Sensory)
    nax = 1 + n_cell_types  # trajectory + one subplot per population
    if ax is None:
        if fig is None:
            fig,ax1 = plt.subplots(1, nax, figsize=(5*nax,5))
        else:
            ax1 = fig.subplots(1, nax)
        ax = {}
        ax['traj'] = ax1[0]
        for i, name in enumerate(Sensory.keys()):
            ax[name] = ax1[i+1]

    if ax_xlim is None or ax_ylim is None:
        ax_xlim, ax_ylim = get_ax_lims(pos[t], Env, Sensory)

    def plot_env(axcurr):
        Env.plot_environment(fig=fig, ax=axcurr, autosave=False)
        axcurr.set_xlim(ax_xlim)
        axcurr.set_ylim(ax_ylim)
        axcurr.set_aspect('equal')
        
    # --- trajectory + agent state at t ---
    plot_env(ax['traj'])
    ax['traj'].scatter(pos[:t + 1, 0], pos[:t + 1, 1],
                    s=12, alpha=0.5, linewidth=0, c="C0", zorder=2)
    plot_agent_state(ax['traj'], pos, head_dir, vel, t)
    ax['traj'].set_title(f"t = {t}", fontsize=16)

    # not sure what this is for
    # Fake a single-timestep history so display_vector_cells has what
    # it needs. `t_id` inside will be 0.
    ag = next(iter(Sensory.values())).Agent
    ag.history["t"]              = [float(t)]
    ag.history["pos"]            = [pos[t].tolist()]
    ag.history["head_direction"] = [head_dir[t].tolist()]

    for name, neuron in Sensory.items():
        fr = np.asarray(sensory_curr[name])[t]    # (n_cells,)
        cls_name = type(neuron).__name__
        if cls_name in CELL_VECTOR_CLASSES:
            neuron.history["t"]          = [float(t)]
            neuron.history["firingrate"] = [fr.tolist()]
            plot_env(ax[name])
            plot_agent_state(ax[name], pos, head_dir, vel, t)
            neuron.display_vector_cells(fig=fig, ax=ax[name], t=float(t))
        elif cls_name in ("HeadDirectionCells", "VelocityCells"):
            if ax[name].name != "polar":
                axpos = ax[name].get_position()
                ax[name].remove()
                ax[name] = fig.add_axes(axpos, projection='polar')
            plot_head_direction(ax[name],neuron,fr,head_dir=head_dir[t])
        elif cls_name == "SpeedCell":
            ax[name].plot(np.asarray(sensory_curr[name]),'-',color='C0',lw=2)
            ax[name].plot(t, fr, 'o',color='C1',ms=16)
        
        ax[name].set_title(name, fontsize=16)

    fig.tight_layout()