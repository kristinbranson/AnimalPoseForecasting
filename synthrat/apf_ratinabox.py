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
import pickle
import sys
import inspect
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
from dataclasses import dataclass, field

import apf.utils
import apf.dataset

import ratinabox
import ratinabox.utils

import logging
LOG = logging.getLogger(__name__)

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
    try:
        if "sensory_info" in data:
                res['Sensory'] = rehydrate_sensory(res['Ag'], data['sensory_info'])
    except Exception as e:
        print(f"Error rehydrating sensory info: {e}")
        res['Sensory'] = None
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
        # FOV cell arrangement could be a function
        if hasattr(neuron,'cell_arrangement') and callable(neuron.cell_arrangement):
            info['cell_arrangement'] = neuron.cell_arrangement.__name__
                                                                                                                                                        
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

        # If cell_arrangement is a callable (e.g. rect_polar_grid), its named
        # kwargs (n_angles, spatial_resolution, beta, ...) aren't in any
        # class's default_params and would get filtered out below. Resolve
        # the callable now and add its parameter names to valid_keys.
        ca = info.get('cell_arrangement')
        if isinstance(ca, str) and ca in globals() and callable(globals()[ca]):
            ca = globals()[ca]
        if callable(ca):
            sig = inspect.signature(ca)
            valid_keys.update(
                p for p, param in sig.parameters.items()
                if param.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                      inspect.Parameter.VAR_KEYWORD)
            )

        params = {k: v for k, v in info.items() if k in valid_keys}

        # rehydrate rect_polar_grid cell arrangement if needed
        if 'cell_arrangement' in params and params['cell_arrangement'] in globals():
            params['cell_arrangement'] = globals()[params['cell_arrangement']]
            
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

def compute_sensory(track_curr, Sensory=None, info=None, **kwargs):
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

    if info is not None:
        # need to rehydrate
        Env = rehydrate_env(info['env_info'])
        Ag = rehydrate_agent(Env, info['agent_info'])
        Sensory = rehydrate_sensory(Ag, info['sensory_info']) 

    # All Sensory neurons were attached to the same Ag at init time; recover it.
    Ag = next(iter(Sensory.values())).Agent

    out = {}
    for name, neuron in Sensory.items():
        out[name] = _firingrate_over_trajectory(
            neuron, Ag, pos, head_dir, vel, T
        )
    return out

def get_rect_polar_grid_shape(neuron):
    """
    For rect_polar_grid FieldOfViewCells, return the shape of the firing-rate array per step,
    i.e. (n_rings, n_per_ring). n_rings is derived from the unique tuning_distances; n_per_ring is derived from n // n_rings.
    """

    n_rings = len(np.unique(neuron.tuning_distances))
    n_per_ring = neuron.n // n_rings
    return n_rings, n_per_ring

def get_feature_names(neuron, prefix=None):
    """Return a list of feature (cell) names for one Neurons population.

    Length matches `neuron.n` and ordering matches the columns of the (T, n)
    firing-rate array produced by `_firingrate_over_trajectory`.

    Naming conventions per cell type:
      - BoundaryVectorCells / ObjectVectorCells / FieldOfView*:
          `{prefix}__r={d:.3f}_theta={deg:+.1f}deg`. For rect_polar_grid
          layouts (constant cells per ring) a `_ring{i}_ang{j}` suffix is
          appended so grid indexing is recoverable.
      - HeadDirectionCells / VelocityCells:
          `{prefix}__theta={deg:+.1f}deg`
      - SpeedCell:
          `{prefix}__speed`

    Parameters
    ----------
    neuron : ratinabox Neurons instance
    prefix : str or None
        Prepended (with `__`) to each name. Defaults to `neuron.name`.
    """
    cls_name = type(neuron).__name__
    if prefix is None:
        prefix = neuron.name

    if cls_name in ("BoundaryVectorCells", "ObjectVectorCells",
                    "FieldOfViewBVCs", "FieldOfViewOVCs"):
        d = np.asarray(neuron.tuning_distances)
        theta_deg = np.degrees(np.asarray(neuron.tuning_angles))
        unique_r = np.unique(d)
        n_rings = len(unique_r)
        if n_rings > 0 and neuron.n % n_rings == 0:
            n_per_ring = neuron.n // n_rings
            ring_idx = np.searchsorted(unique_r, d)
            ang_idx = np.tile(np.arange(n_per_ring), n_rings)[:neuron.n]
            return [
                f"{prefix}__r={d[i]:.3f}_theta={theta_deg[i]:+.1f}deg"
                f"_ring{ring_idx[i]}_ang{ang_idx[i]}"
                for i in range(neuron.n)
            ]
        return [
            f"{prefix}__r={d[i]:.3f}_theta={theta_deg[i]:+.1f}deg"
            for i in range(neuron.n)
        ]

    if cls_name in ("HeadDirectionCells", "VelocityCells"):
        theta_deg = np.degrees(np.asarray(neuron.preferred_angles))
        return [f"{prefix}__theta={theta_deg[i]:+.1f}deg" for i in range(neuron.n)]

    if cls_name == "SpeedCell":
        return [f"{prefix}__speed"]

    raise TypeError(f"unsupported cell class: {cls_name}")

def get_all_feature_names(sensory_or_info):
    """Concatenate per-population feature names into one flat list.

    Accepts either:
      - a live `Sensory` dict (population name -> Neurons), or
      - a full info bundle (dict with 'env_info', 'agent_info', 'sensory_info'
        keys, as stored on the `Sensory` Operation). In that case the
        populations are rehydrated first via rehydrate_env / rehydrate_agent
        / rehydrate_sensory.

    Order matches the iteration order of the resulting `Sensory` dict (matching
    how `compute_sensory` returns its result). Population name is used as the
    prefix for each cell's feature name.
    """
    if isinstance(sensory_or_info, dict) and 'sensory_info' in sensory_or_info:
        info = sensory_or_info
        Env = rehydrate_env(info['env_info'])
        Ag = rehydrate_agent(Env, info['agent_info'])
        Sensory = rehydrate_sensory(Ag, info['sensory_info'])
    else:
        Sensory = sensory_or_info

    names = []
    for pop_name, neuron in Sensory.items():
        names.extend(get_feature_names(neuron, prefix=pop_name))
    return names

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

    # BVCs / OVCs / FieldOfView*: pos is vectorized over T. Egocentric variants
    # use the patched get_state(is_array=True) to also vectorize head_direction.
    # For very large N_pos we additionally chunk + run those chunks across a
    # multiprocessing pool (numpy ufuncs are single-threaded on their own).
    # Sweet spot from bench_sensory: chunk_size=50k, n_workers=8.
    if cls_name in (
        "BoundaryVectorCells", "ObjectVectorCells",
        "FieldOfViewBVCs", "FieldOfViewOVCs",
    ):
        is_ego = getattr(neuron, "reference_frame", "allocentric") == "egocentric"
        if not is_ego:
            fr = neuron.get_state(evaluate_at=None, pos=pos,
                                  chunk_size=50000, n_workers=8,
                                  parallel_threshold=200000)
            return np.asarray(fr).T                           # (T, n_cells)

        fr = neuron.get_state(
            evaluate_at=None,
            pos=pos,
            head_direction=head_dir,
            is_array=True,
            chunk_size=50000,
            n_workers=8,
            parallel_threshold=200000,
        )                                                     # (n_cells, T)
        return np.asarray(fr).T                               # (T, n_cells)

    if cls_name == "HeadDirectionCells":
        fr = neuron.get_state(
            evaluate_at=None,
            head_direction=head_dir,
            is_array=True,
        )                                                     # (n, T)
        return np.asarray(fr).T                               # (T, n)

    if cls_name == "VelocityCells":
        if vel is None:
            raise ValueError("VelocityCells require track_curr['vel']")
        fr = neuron.get_state(
            evaluate_at=None,
            velocity=vel,
            is_array=True,
        )                                                     # (n, T)
        return np.asarray(fr).T                               # (T, n)

    if cls_name == "SpeedCell":
        if vel is None:
            raise ValueError("SpeedCell requires track_curr['vel']")
        fr = neuron.get_state(evaluate_at=None, vel=vel, is_array=True)  # (1, T)
        return np.asarray(fr).T                                          # (T, 1)

    raise TypeError(f"unsupported cell class: {cls_name}")

def rect_polar_grid(distance_range, angle_range, n_angles,
                    spatial_resolution, beta=5, **_):
    """`cell_arrangement` callable for a polar grid of vector cells with
    diverging-manifold ring spacing and a fixed number of cells per ring.

    Ring radii follow RatInABox's diverging_manifold rule (Hartley model
    + just-touching radially):

        resolution(r) = xi + r/beta,   xi chosen so resolution(d_min) = spatial_resolution
        r_{k+1}       = (2*r_k + resolution_k + xi) / (2 - 1/beta)

    The innermost ring is at d_min (clamped to >= 0.01); the loop stops
    when r_{k+1} would exceed d_max (so the outermost ring is at most
    d_max but typically a bit less). n_rings is derived; recover via
    `len(mu_d) // n_angles`.

    Each ring is given exactly n_angles cells, evenly tiled across
    [-theta_max, +theta_max] at the midpoints of equal-width angular
    bins, so the firing-rate vector reshapes cleanly to
    `(T, n_rings, n_angles)`.

    In display_vector_cells (where the ellipse "width" axis is rotated
    to point radially outward, despite the variable names):
        - radial diameter      = sigma_angles * r = resolution(r)  -> ring neighbors just touch
        - tangential diameter  = sigma_distances  = r * dtheta     -> in-ring neighbors just touch

    Cells are not square in general (resolution(r) != r * dtheta), since
    rings follow Hartley but n_angles is fixed.

    Side effect: sigma_distances and sigma_angles are also used by
    `get_state` as Gaussian / von Mises widths for the firing rate. With
    this swap of axis roles, the *radial* firing field has width
    r * dtheta and the *angular* one has width resolution(r) / r. Looks
    right on the plot but physically idiosyncratic.

    Parameters
    ----------
    distance_range : (d_min, d_max)
        Innermost ring at d_min (clamped to >= 0.01); outermost ring is
        the largest one whose successor would exceed d_max.
    angle_range : (theta_min, theta_max) in degrees
        theta_min is currently ignored; cells tile symmetrically across
        [-theta_max, +theta_max].
    n_angles : int
        Cells per ring (fixed across rings).
    spatial_resolution : float
        Radial cell size at the innermost ring, in metres.
    beta : float, default 5
        Hartley growth parameter; smaller -> faster cell growth -> fewer rings.

    Returns
    -------
    (mu_d, mu_theta, sigma_d, sigma_theta) : four 1-D arrays of length
        n_rings * n_angles, flattened in ring-major order.
    """
    theta_max = np.deg2rad(angle_range[1])
    dtheta = (2 * theta_max) / n_angles
    t = np.linspace(-theta_max + dtheta / 2, theta_max - dtheta / 2, n_angles)

    radii, resolutions = [], []
    r = max(0.01, distance_range[0])
    xi = spatial_resolution - r / beta
    while r < distance_range[1]:
        resolution = xi + r / beta
        radii.append(r)
        resolutions.append(resolution)
        r = (2 * r + resolution + xi) / (2 - 1 / beta)
    radii = np.array(radii)
    resolutions = np.array(resolutions)

    dd, tt = np.meshgrid(radii, t, indexing="ij")
    mu_d, mu_theta = dd.ravel(), tt.ravel()
    sigma_d     = (dd * dtheta).ravel()                 # tangential, ring-touching
    sigma_theta = (resolutions[:, None] / dd).ravel()   # radial, Hartley-touching
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
    if isinstance(track_curr, dict):
        trajectory = track_curr['pos']
        head_direction = track_curr['head_direction']
    elif hasattr(track_curr,'shape'):
        trajectory = track_curr[..., :2]
        # last dim is an angle (radians); convert to a unit direction vector
        # so downstream code (get_bearing) sees a consistent (T, 2) shape.
        angle = track_curr[..., 2]
        head_direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    else:
        raise ValueError("track_curr must be a dict with 'pos' and 'head_direction' keys, or an array with shape (..., 3) where the last dimension is (x, y, head_direction).")
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
        
def plot_head_direction(axcurr, neuron, fr, head_dir=None, head_bearing=None,
                        min_height=0.1):
    """Plot an HDC/VelocityCells population on a polar axis as von-Mises bumps.

    Each cell is drawn as a smooth curve peaking at its preferred angle, with
    peak height = max(firing rate `fr[j]`, `min_height`) and angular width set
    by `neuron.angular_tunings[j]` (via κ = 1/σ²). The `min_height` floor
    keeps every cell visible even when its firing rate is zero; cells with
    non-zero firing rate appear larger in proportion. An orange (C1) radial
    line marks the current heading direction so you can see how well-aligned
    the population response is with the agent's actual heading.

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
    min_height : float, default 0.1
        Minimum peak height for each cell's bump, so silent cells stay
        visible. Active cells with fr[j] > min_height plot at fr[j].
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
    # peak height = max(firing rate, min_height), angular width set by angular_tunings.
    theta_grid = np.linspace(-np.pi, np.pi, 360)
    for j in range(neuron.n):
        kappa = 1.0 / max(tunings[j]**2, 1e-6)
        height = max(min_height, fr[j])
        bump = height * np.exp(kappa * (np.cos(theta_grid - preferred_plot[j]) - 1))
        axcurr.plot(theta_grid, bump,
                        color='C0', linewidth=2)
    r_max = max(fr.max(), min_height)
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
    # Accept either a dict {'pos','head_direction'[,'vel']} or an array of
    # shape (T, 3) where the last dim is (x, y, theta).
    if isinstance(track_curr, dict):
        pos      = np.asarray(track_curr["pos"])
        head_dir = np.asarray(track_curr["head_direction"])
        vel      = np.asarray(track_curr["vel"]) if "vel" in track_curr else None
    else:
        track_curr = np.asarray(track_curr)
        pos = track_curr[..., :2]
        angle = track_curr[..., 2]
        head_dir = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        vel = None

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
    else:
        if fig is None:
            fig = ax['traj'].figure

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
    
@dataclass
class Sensory(apf.dataset.Operation):
    """ Computes sensory neuron firing rates

    NOTE: this operation is not invertible.
    Attributes: 
        idxinfo: Keeps track of which dimensions of the sensory output correspond to what cell type
    """
    
    localattrs = ['idxinfo','feature_names','sensory_info']
    idxinfo: dict | None = None
    feature_names: list | None = None
    # dict containing 'env_info', 'agent_info', and 'sensory_info' needed to rehydrate the Sensory neurons
    info: dict | None = None 
    
    def apply(self, X: np.ndarray, info: dict, isdata: np.ndarray | None = None) -> np.ndarray:
        """ Computes sensory features from keypoints.

        Args:
            X: (x, y, orientation, vel_x, vel_y) position of the agents, (n_agents,  n_frames, 5) float array
            isdata: indicates whether there is data for a given frame or agent, only used to speed up computation, 
            (n_frames, n_agents) bool array

        Returns:
            sensory_data: (n_agents,  n_frames, n_sensory_features) float array
        """
        self.info = info
        self.idxinfo = {}
        sensory_data = None
        for ratid in range(X.shape[0]):
            if isdata is not None:
                isdatacurr = isdata[ratid]
                head_direction = np.stack([np.cos(X[ratid,isdatacurr,2]), np.sin(X[ratid,isdatacurr,2])], axis=-1)
                track = {
                    'pos': X[ratid,isdatacurr,0:2],
                    'head_direction': head_direction
                }
            else:
                isdatacurr = None
                head_direction = np.stack([np.cos(X[ratid,:,2]), np.sin(X[ratid,:,2])], axis=-1)
                track = {
                    'pos': X[ratid,:,0:2],
                    'head_direction': head_direction
                }
            featscurr = compute_sensory(track,info=self.info)
            if ratid == 0:
                # store idxinfo
                idxoff = 0
                idxinfo = {}
                for k,v in featscurr.items():
                    ncurr = v.shape[1]
                    idxinfo[k] = (idxoff, idxoff + ncurr)
                    idxoff += ncurr
                sensory_data = np.full((X.shape[0],X.shape[1],idxoff),np.nan)
            sensory_data_curr = np.concatenate(list(featscurr.values()), axis=1)
            if isdatacurr is not None:
                sensory_data[ratid,isdatacurr,:] = sensory_data_curr
            else:
                sensory_data[ratid,:,:] = sensory_data_curr
        self.idxinfo = idxinfo
        self.feature_names = get_all_feature_names(self.info)
        return sensory_data, {'isdata': isdata}

    def invert(self, sensory: np.ndarray) -> None:
        LOG.error(f"Operation {self} is not invertible")
        return None
    
    def __str__(self):
        s = f"Operation {self.name} of class Sensory with idxinfo keys:\n"
        if self.idxinfo is not None:
            for key in self.idxinfo:
                s += f"  {key}: {self.idxinfo[key]}\n"
        else:
            s += "  idxinfo is None\n"
        return s[:-1]
    
    def update_feature_names(self, input_feature_names):
        return self.feature_names
    
    def invert_feature_names(self, input_feature_names):
        LOG.error(f"Operation {self.name} is not invertible")
        return None

        
def make_dataset(
        config: dict,
        filename: str,
        ref_dataset: apf.dataset.Dataset | None = None,
        return_all: bool = False,
        debug: bool = True,
        data: dict | None = None,
        cached_sensory_array: np.ndarray | None = None,
) -> apf.dataset.Dataset | tuple[apf.dataset.Dataset, np.ndarray, apf.dataset.Data, apf.dataset.Data, apf.dataset.Data, apf.dataset.Data]:
    """ Creates a dataset from config, for a given file name and optionally using a reference dataset.

    Args:
        config: Config for loading the data
        filename: Name of file to read the data from (e.g. 'intrainfile', 'invalfile')
        ref_dataset: Dataset to copy postprocessing operations from.
            When loading validation/test set, provide training set.
        return_all: Whether to return intermediate variables in addition to Dataset (see returns)
        debug: Whether to use less data for debugging
        data: Optionally provide pre-loaded data dict to avoid re-loading from file.
        cached_sensory_array: Optional precomputed sensory firing-rate array of
            shape (n_agents, n_total_frames, n_sensory_features). If provided,
            Sensory.apply is skipped and this array is wrapped in a Data with
            the matching Sensory operation. The shape must match what
            Sensory.apply would have produced for the given inputs (otherwise
            downstream Zscore/Discretize/etc. shapes won't line up).
        data dict: {
            'track': [{
                'pos': (T,2) array of x,y positions
                'head_direction': (T,2) array of unit vectors
            }, ...] list of episodes
            'hidden': [{
                firingrate_key: (T, n_cells) array of firing rates for each hidden neuron type
            }, ...] list of episodes
            'env_info': dict of environment setup
            'agent_info': dict of agent setup
            'sensory_info': {
                sensory_key: dict of sensory cell setup 
            }
        }
    

    Returns:
        dataset: Dataset for flyllm experiment.
        [
          track: x,y,orientation
          velocity: Egocentric velocity data
          sensory: Sensory data
        ]
    """
    
    # Load data
    if data is None:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

    dt = data['agent_info']['dt']

    if debug:
        n_episodes = 5
        data['track'] = data['track'][:n_episodes]
        data['hidden'] = data['hidden'][:n_episodes]

    info = {
        'env_info': data['env_info'],
        'agent_info': data['agent_info'],
        'sensory_info': data['sensory_info']
    }

    # create track from X which is (n_agents, n_frames, 5):
    # x, y, orientation, vel_x, vel_y

    # concatenate all episodes together and create isstart arrays to keep track of 
    # episode boundaries
    episode_lengths = [len(ep['pos']) for ep in data['track']]
    ntotal_frames = np.sum(episode_lengths)
    isstart = np.zeros(ntotal_frames, dtype=bool)
    isstart[0] = True
    isstart[np.cumsum(episode_lengths)[:-1]] = True
    pos = np.concatenate([ep['pos'] for ep in data['track']], axis=0)
    head_direction = np.concatenate([ep['head_direction'] for ep in data['track']], axis=0)
    orientation = apf.utils.modrange(ratinabox.utils.get_angle(head_direction,is_array=True),-np.pi, np.pi)
    # vel can be computed from np.diff(pos,axis=0)/dt, except for first time point
    vel = np.concatenate([ep['vel'] for ep in data['track']], axis=0)
    X = np.concatenate([pos,orientation[:,None]],axis=-1)
    
    pose = apf.dataset.Data('pos',X[None])

    # position_velocity = apf.dataset.Data('pos_vel', X[None,...], [], feature_names=['x', 'y', 'orientation', 'vel_x', 'vel_y'])

    if cached_sensory_array is not None:
        # Skip the expensive Sensory.apply call by wrapping the cached firing-
        # rate array in a Data with the same Sensory operation metadata.
        # `isstart` here matches the un-broadcast (1-axis-less) form expected by
        # Sensory.apply's invertdata; it gets broadcast to the dataset's
        # convention below.
        sensory_op = Sensory()
        sensory_op.info = info
        sensory = apf.dataset.Data('sensory', cached_sensory_array,
                                   operations=[sensory_op],
                                   invertdata=[{'isdata': isstart}])
    else:
        sensory = Sensory()(pose, info=info)
        
    # reshape to be (n_agents, n_frames, ...)
    isstart = isstart[None,:]
    
    velocity = apf.dataset.GlobalVelocity(tspred=[1,])(pose,isstart=isstart)

    args = {
        'context_length': config['contextl'],
        'isstart': isstart
    }

    # Assemble the dataset
    if ref_dataset is not None:
        dataset = apf.dataset.Dataset(
            inputs=apf.dataset.apply_opers_from_data(ref_dataset.inputs, {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
            labels=apf.dataset.apply_opers_from_data(ref_dataset.labels, {'velocity': velocity}), #, 'auxiliary': auxiliary}),
            **args
        )
    elif 'dataset_params' in config and config['dataset_params'] is not None and \
        ('inputs' in config['dataset_params']) and ('labels' in config['dataset_params']):
        dataset = apf.dataset.Dataset(
            inputs=apf.dataset.apply_opers_from_data_params(config['dataset_params']['inputs'], {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
            labels=apf.dataset.apply_opers_from_data_params(config['dataset_params']['labels'], {'velocity': velocity}), #, 'auxiliary': auxiliary}),
            **args
        )
    else:
        # velocity = OddRoot(5)(velocity)

        # discretize everything
        discreteidx = config['discreteidx']    

        # Need to zscore before binning, otherwise bin_epsilon values need to be divided by zscore stds
        zscored_velocity = apf.dataset.Zscore()(velocity)

        zsig = zscored_velocity.operations[-1].std
        bin_config = {'nbins': config['discretize_nbins'],
                      'bin_epsilon': config['discretize_epsilon'] / zsig[discreteidx]}

        if 'bin_edges_absolute' in config and config['bin_edges_absolute'] is not None and len(config['bin_edges_absolute']) > 0:
            # check that all discreteidx are present
            assert all(idx in config['bin_edges_absolute'] for idx in discreteidx), "Not all discreteidx are present in bin_edges_absolute"
            bin_config['bin_edges'] = np.vstack([config['bin_edges_absolute'][featidx]/zsig[featidx] for featidx in discreteidx]) # nfeat x nbins + 1

        dataset = apf.dataset.Dataset(
            inputs={
                'velocity': apf.dataset.Zscore()(apf.dataset.Roll(dt=1)(velocity)),
                'sensory': apf.dataset.Zscore()(sensory),
            },
            labels={
                'velocity': apf.dataset.Discretize(**bin_config)(zscored_velocity),
            },
            **args
        )
    dataset_params = dataset.get_params()
    if return_all:
        return dataset, info, pose, velocity, sensory, dataset_params, isstart
    else:
        return dataset, info

def debug_plot_sample(example_in, ratinabox_info, nplot=3, pred=None, fig=None, ax=None):
    """Visualize nplot random batch elements via visualize_sensory.

    Layout: nplot rows × (1 trajectory + n_sensory) columns. The trajectory
    column shows the inverted pose plus vector-cell ellipses; subsequent
    columns show one panel per non-vector sensory population.

    Parameters
    ----------
    example_in : dict from dataset.item_to_data — must have 'labels'/'velocity'
        and 'inputs'/'sensory' Data objects.
    ratinabox_info : dict with 'Env' and 'Sensory' (live populations).
    dataset : (unused, accepted for caller compatibility)
    nplot : number of batch elements to draw.
    pred : (unused for now; reserved for overlaying model predictions).
    fig, ax : if both None, a new figure is created and returned.
        If provided, ax must be the list-of-dicts structure that this function
        previously returned (one dict per row, mapping 'traj' / population
        names to Axes); the panels are cleared and redrawn so subsequent
        update calls reuse the same figure.

    Returns
    -------
    fig, ax : matplotlib Figure and the per-row list of Axes-dicts.
    """
    invert_args = {'discretize': {'use_todiscretize': True}}
    pose_true = apf.dataset.apply_inverse_operations(example_in['labels']['velocity'], extraargs=invert_args)
    sensory_op = apf.dataset.get_operation(example_in['inputs']['sensory'].operations, 'sensory')
    fr_all = example_in['inputs']['sensory'].array      # (B, T, total_cells)

    Sensory = ratinabox_info['Sensory']
    pop_names = list(Sensory.keys())
    n_cols = 1 + len(pop_names)
    t = pose_true.shape[1] - 1
    batch_size = pose_true.shape[0]
    samples_plot = np.random.choice(batch_size, size=min(nplot, batch_size), replace=False)
    nrows = len(samples_plot)

    # --- Figure / axes setup ----------------------------------------------
    if fig is None and ax is None:
        fig, ax_grid = plt.subplots(nrows, n_cols, figsize=(5 * n_cols, 5 * nrows),
                                    squeeze=False)
        ax = []
        for i in range(nrows):
            ax_row = {'traj': ax_grid[i, 0]}
            for j, name in enumerate(pop_names):
                ax_row[name] = ax_grid[i, 1 + j]
            ax.append(ax_row)
    else:
        # Reuse prior axes; clear them so the redraw doesn't pile up.
        if fig is None:
            fig = ax[0]['traj'].figure
        for ax_row in ax:
            for axcurr in ax_row.values():
                axcurr.clear()

    # --- Draw each sample --------------------------------------------------
    for ax_row, sample_idx in zip(ax, samples_plot):
        sensory_curr = {name: fr_all[sample_idx, ..., start:end]
                        for name, (start, end) in sensory_op.idxinfo.items()}
        visualize_sensory(pose_true[sample_idx], sensory_curr, t,
                          ratinabox_info['Env'], Sensory,
                          fig=fig, ax=ax_row)

        # Overlay the predicted trajectory + final state on the trajectory
        # panel, in a different color, so it's directly comparable to the
        # ground truth drawn by visualize_sensory.
        if pred is not None:
            pose_p = pred[sample_idx]                           # (T, 3)
            ax_row['traj'].scatter(pose_p[:t + 1, 0], pose_p[:t + 1, 1],
                                   s=12, alpha=0.5, linewidth=0,
                                   c='C3', zorder=3)
            ax_row['traj'].scatter(pose_p[t, 0], pose_p[t, 1],
                                   s=80, c='C3', linewidth=0, zorder=6)

    return fig, ax


def initialize_debug_plots(dataset, dataloader, ratinabox_info, name='', nplot=3):

    example_batch = next(iter(dataloader))
    example = dataset.item_to_data(apf.utils.convert_torch_to_numpy(example_batch))

    # plot to visualize input features
    figsample, axsample = debug_plot_sample(example, ratinabox_info, nplot=nplot)

    axsample[0]['traj'].set_title(name)
    figsample.tight_layout()

    hdebug = {
        'figsample': figsample,
        'axsample': axsample,
        'example': example
    }

    return hdebug


def update_debug_plots(hdebug, config, model, dataset, ratinabox_info, example, pred,
                       criterion=None, name='', nplot=3):
    if config['modelstatetype'] == 'prob':
        pred1 = model.maxpred({k: v.detach() for k, v in pred.items()})
    elif config['modelstatetype'] == 'best':
        pred1 = model.randpred(pred.detach())
    else:
        if isinstance(pred, dict):
            pred1 = {k: v.detach().cpu() for k, v in pred.items()}
        else:
            pred1 = pred.detach().cpu()
    # `example` from the training loop is a raw batched torch dict
    # ({'input', 'labels', 'labels_discrete', 'metadata'}). Convert to the
    # Data-dict form ({'inputs', 'labels'} of Data objects) that
    # debug_plot_sample expects.
    example_data = dataset.item_to_data(apf.utils.convert_torch_to_numpy(example))

    # Build a "prediction example" that uses pred1 in place of the discrete
    # labels (with the example's metadata so invertdata is intact), then run
    # the same inversion to recover a (B, T, 3) predicted pose array. We use
    # do_sampling=False so the inverted bins become the deterministic weighted
    # average of bin centers.
    pred_item = {'metadata': example['metadata']}
    if isinstance(pred1, dict):
        if 'labels_discrete' in pred1:
            pred_item['labels_discrete'] = pred1['labels_discrete']
        elif 'discrete' in pred1:
            pred_item['labels_discrete'] = pred1['discrete']
        if 'labels' in pred1:
            pred_item['labels'] = pred1['labels']
        elif 'continuous' in pred1:
            pred_item['labels'] = pred1['continuous']
    else:
        pred_item['labels_discrete'] = pred1
    pred_data = dataset.item_to_data(apf.utils.convert_torch_to_numpy(pred_item))
    pose_pred = apf.dataset.apply_inverse_operations(
        pred_data['labels']['velocity'],
        extraargs={'discretize': {'do_sampling': False}})

    debug_plot_sample(example_data, ratinabox_info, nplot=nplot,
                      fig=hdebug['figsample'], ax=hdebug['axsample'], pred=pose_pred)
    hdebug['axsample'][0]['traj'].set_title(name)
    hdebug['figsample'].tight_layout()


def initialize_loss_plots(loss_epoch):
    nax = len(loss_epoch) // 2
    assert (nax >= 1) and (nax <= 3)
    hloss = {}

    hloss['fig'], hloss['ax'] = plt.subplots(nax, 1)
    if nax == 1:
        hloss['ax'] = [hloss['ax'], ]

    hloss['train'], = hloss['ax'][0].plot(loss_epoch['train'].cpu(), '.-', label='Train')
    hloss['val'], = hloss['ax'][0].plot(loss_epoch['val'].cpu(), '.-', label='Val')

    if 'train_continuous' in loss_epoch:
        hloss['train_continuous'], = hloss['ax'][1].plot(loss_epoch['train_continuous'].cpu(), '.-',
                                                         label='Train continuous')
    if 'train_discrete' in loss_epoch:
        hloss['train_discrete'], = hloss['ax'][2].plot(loss_epoch['train_discrete'].cpu(), '.-', label='Train discrete')
    if 'val_continuous' in loss_epoch:
        hloss['val_continuous'], = hloss['ax'][1].plot(loss_epoch['val_continuous'].cpu(), '.-', label='Val continuous')
    if 'val_discrete' in loss_epoch:
        hloss['val_discrete'], = hloss['ax'][2].plot(loss_epoch['val_discrete'].cpu(), '.-', label='Val discrete')

    hloss['ax'][-1].set_xlabel('Epoch')
    hloss['ax'][-1].set_ylabel('Loss')
    for l in hloss['ax']:
        l.legend()
    return hloss


def update_loss_plots(hloss, loss_epoch):
    hloss['train'].set_ydata(loss_epoch['train'].cpu())
    hloss['val'].set_ydata(loss_epoch['val'].cpu())
    if 'train_continuous' in loss_epoch:
        hloss['train_continuous'].set_ydata(loss_epoch['train_continuous'].cpu())
    if 'train_discrete' in loss_epoch:
        hloss['train_discrete'].set_ydata(loss_epoch['train_discrete'].cpu())
    if 'val_continuous' in loss_epoch:
        hloss['val_continuous'].set_ydata(loss_epoch['val_continuous'].cpu())
    if 'val_discrete' in loss_epoch:
        hloss['val_discrete'].set_ydata(loss_epoch['val_discrete'].cpu())
    for l in hloss['ax']:
        l.relim()
        l.autoscale()
