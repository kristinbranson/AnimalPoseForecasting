"""Sensory-neuron firing rates for the RatInABox synthetic rat.

This module rebuilds the RatInABox objects saved alongside a trajectory dataset and
computes the firing rates of the sensory cell populations the model observes.

1. Object rehydration
    `rehydrate_env`, `rehydrate_agent`, and `rehydrate_sensory` rebuild an
    Environment, Agent, and dict of sensory populations from the info dicts stored
    in a saved trajectory pkl. Each filters the input dict to valid params keys
    (ignoring derived attributes like `Env.extent`), then restores mutated state and
    custom attributes post-construction. `rehydrate_sensory` overwrites per-cell
    tuning attributes after construction so that `cell_arrangement="random"`
    populations are restored exactly, bypassing the constructor's fresh random draws.
    `rehydrate_data` bundles all three.

2. Offline sensory firing rates
    `init_sensory` takes an Agent + Environment and a `cell_config` describing the
    desired populations (BoundaryVectorCells, FieldOfViewBVCs, HeadDirectionCells,
    VelocityCells, SpeedCell, ...). It instantiates each population once, so randomly
    sampled tuning parameters stay fixed across episodes. `compute_sensory` then
    applies those populations to a saved trajectory to produce `{name: (T, n_cells)}`.

3. Hand-built polar cell grid
    `rect_polar_grid` is a `cell_arrangement` callable for vector cells that lays out
    a rectangular n_rings x n_angles grid in polar (d, theta) coordinates, so the
    firing-rate output reshapes cleanly to `(T, n_rings, n_angles)`.

4. Feature naming
    `get_feature_names` / `get_all_feature_names` produce a human-readable name per
    cell, ordered to match the columns of the firing-rate arrays.

REQUIRES a patched RatInABox -- see the note in experiments/synthrat.py.
"""

import inspect
import numpy as np

import ratinabox
import ratinabox.utils

# NB: `ratinabox/__init__.py` does `from .Neurons import *`, `.Environment
# import *`, `.Agent import *`, and `from . import contribs`. After that, at the
# top level:
#   ratinabox.Environment, ratinabox.Agent, ratinabox.PlaceCells,
#   ratinabox.BoundaryVectorCells, ...
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

def rehydrate_env(env_info):
    """Reconstruct an `Environment` from the `env_info` dict saved alongside
    the trajectories.

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
    """Reconstruct an `Agent` from the saved `agent_info` dict.

    Forwards any key that is a valid `Agent` params key; ignores the rest
    (e.g. your custom `exploit_explore_ratio`). After construction, restores
    any custom attributes from `agent_info` that aren't real Agent params.

    Parameters
    ----------
    Env : ratinabox Environment
        The (already rehydrated) env this agent belongs to.
    agent_info : dict
        The saved `agent_info`. Expected to include standard Agent
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
def rehydrate_sensory(Ag, sensory_info):
    """Rebuild a `Sensory` dict from the saved `sensory_info` snapshot,
    attached to the given Agent.

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
        The saved `sensory_info`, one entry per population.

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


def rehydrate_data(data):
    """Rebuild the live RatInABox objects needed to compute and plot sensory data.

    Parameters
    ----------
    data : dict with 'env_info', 'agent_info', and (optionally) 'sensory_info',
        as saved in a trajectory pkl or returned by `make_dataset`.

    Returns
    -------
    dict with keys 'Env', 'Ag', and 'Sensory' (the last is None if rehydration
    fails or no sensory_info was saved).
    """
    res = {}
    res['Env'] = rehydrate_env(data['env_info'])
    res['Ag'] = rehydrate_agent(res['Env'], data['agent_info'])
    res['Sensory'] = None
    if "sensory_info" in data:
        try:
            res['Sensory'] = rehydrate_sensory(res['Ag'], data['sensory_info'])
        except Exception as e:
            print(f"Error rehydrating sensory info: {e}")
    return res


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
        One episode's trajectory, i.e. one entry of the saved 'track' list.
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
