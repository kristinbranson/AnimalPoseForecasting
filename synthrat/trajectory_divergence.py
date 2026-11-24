"""Distribution-divergence metrics between two sets of 2D agent trajectories.

Built to compare futures drawn from the same burn-in: K model rollouts
(`experiments.synthrat.simulate(..., n_samples=K)`) vs. K RatInABox rollouts
(`synthrat.generate_data.run_policy_from_burn_in`). Each set is a float array
(K, T, 3) of (x m, y m, theta rad) per frame; frames [0, burn_in) are the
shared burn-in and frames [burn_in, T) are the sampled future.

Metrics, per horizon h (number of future frames considered):
    js_<kinematic>   Jensen-Shannon divergence (bits, in [0, 1]) between
                     histograms of per-step kinematics over the first h
                     future steps, pooled over samples.
    w1_<kinematic>   Wasserstein-1 distance between the same per-step
                     kinematic samples (m/s or rad/s).
    w1_position      Exact Wasserstein-1 distance (m) between the K positions
                     at future frame h of each set.
    js_occupancy     JS divergence (bits) between 2D histograms of all
                     positions in the first h future frames.
Kinematics are forward and sideways velocity in the agent's heading frame
(m/s) and angular velocity of the heading (rad/s), from
`synthrat.generate_data.compute_velocity`.

All functions are pure (no I/O). Rows containing NaN (a model rollout that
aborted) are dropped before a metric is computed.
"""

import numpy as np
import scipy.optimize
import scipy.spatial.distance
import scipy.stats

from synthrat.generate_data import compute_velocity
from synthrat.sensory import head_direction_from_orientation

KINEMATIC_NAMES = ('forward', 'sideways', 'angular')
KINEMATIC_UNITS = {'forward': 'm/s', 'sideways': 'm/s', 'angular': 'rad/s'}
METRIC_NAMES = (
    tuple(f'js_{name}' for name in KINEMATIC_NAMES)
    + tuple(f'w1_{name}' for name in KINEMATIC_NAMES)
    + ('w1_position', 'js_occupancy')
)
METRIC_UNITS = {
    **{f'js_{name}': 'bits' for name in KINEMATIC_NAMES},
    **{f'w1_{name}': KINEMATIC_UNITS[name] for name in KINEMATIC_NAMES},
    'w1_position': 'm',
    'js_occupancy': 'bits',
}


def pose_to_kinematics(pose, dt):
    """Per-step kinematics of one trajectory.

    Parameters
    ----------
    pose : (T, 3) float array of (x m, y m, theta rad).
    dt : float, seconds per frame.

    Returns
    -------
    dict {name: (T-1,) float array} for name in KINEMATIC_NAMES. Step i goes
    from frame i to frame i+1. 'forward' and 'sideways' are the displacement
    projected onto the heading at frame i and its right-hand perpendicular,
    in m/s; 'angular' is the heading change wrapped to [-pi, pi), in rad/s.
    """
    pose = np.asarray(pose, dtype=float)
    # compute_velocity works on a RatInABox episode, whose head_direction is the physical
    # heading, while a pose stores orientation in the fly convention (heading minus
    # ORIENTATION_OFFSET; see synthrat.sensory). Converting here rather than taking
    # (cos theta, sin theta) directly is what keeps 'forward' along the heading.
    episode = {
        'pos': pose[:, :2],                                                  # (T, 2)
        'head_direction': head_direction_from_orientation(pose[:, 2]),       # (T, 2)
    }
    forward, sideways, angular = compute_velocity(episode)    # each (T-1,), per frame
    return {'forward': forward / dt, 'sideways': sideways / dt, 'angular': angular / dt}


def window_kinematics(poses, burn_in, horizon, dt):
    """Kinematics of the first `horizon` future steps, pooled over samples.

    The future steps are the ones ending at frames burn_in, ..., burn_in +
    horizon - 1, i.e. step indices burn_in - 1 ... burn_in + horizon - 2
    (the first one starts at the last burn-in frame).

    Parameters
    ----------
    poses : (K, T, 3) float array, T >= burn_in + horizon.
    burn_in : int, number of shared burn-in frames.
    horizon : int, number of future frames.
    dt : float, seconds per frame.

    Returns
    -------
    dict {name: (K * horizon,) float array}; samples containing any NaN in the
    window are dropped.
    """
    window = poses[:, burn_in - 1:burn_in + horizon]                        # (K, horizon + 1, 3)
    window = window[~np.isnan(window).any(axis=(1, 2))]                     # (K_valid, horizon + 1, 3)
    per_sample = [pose_to_kinematics(p, dt) for p in window]
    return {name: np.concatenate([k[name] for k in per_sample]) if per_sample else np.zeros(0)
            for name in KINEMATIC_NAMES}


def js_divergence(samples_a, samples_b, edges):
    """Jensen-Shannon divergence between histograms of two sample sets.

    Parameters
    ----------
    samples_a : (n_a,) or (n_a, d) float array.
    samples_b : (n_b,) or (n_b, d) float array.
    edges : list of d increasing 1D arrays, bin edges per dimension. An
        underflow and an overflow bin are added in every dimension, so mass
        outside the edges still counts.

    Returns
    -------
    float, JS divergence in bits, in [0, 1] (0 = same histogram, 1 = disjoint
    support). NaN if either set has no non-NaN rows.
    """
    samples_a = np.asarray(samples_a, dtype=float)
    samples_b = np.asarray(samples_b, dtype=float)
    if samples_a.ndim == 1:
        samples_a = samples_a[:, None]
        samples_b = samples_b[:, None]
    samples_a = samples_a[~np.isnan(samples_a).any(axis=1)]                 # (n_a_valid, d)
    samples_b = samples_b[~np.isnan(samples_b).any(axis=1)]                 # (n_b_valid, d)
    if len(samples_a) == 0 or len(samples_b) == 0:
        return np.nan
    full_edges = [np.concatenate([[-np.inf], e, [np.inf]]) for e in edges]
    counts_a, _ = np.histogramdd(samples_a, bins=full_edges)
    counts_b, _ = np.histogramdd(samples_b, bins=full_edges)
    # scipy returns the JS *distance* (square root of the divergence)
    distance = scipy.spatial.distance.jensenshannon(counts_a.ravel(), counts_b.ravel(), base=2)
    return float(distance ** 2)


def wasserstein_2d_equal_n(points_a, points_b):
    """Exact Wasserstein-1 distance between two equal-size 2D point clouds.

    Solves the optimal one-to-one matching under Euclidean cost. NaN rows are
    dropped; if the sets then differ in size, the larger one is truncated to
    the smaller (rows are exchangeable i.i.d. samples).

    Parameters
    ----------
    points_a : (n, 2) float array, metres.
    points_b : (n, 2) float array, metres.

    Returns
    -------
    float, mean matched distance in metres. NaN if either set is empty.
    """
    points_a = points_a[~np.isnan(points_a).any(axis=1)]
    points_b = points_b[~np.isnan(points_b).any(axis=1)]
    n = min(len(points_a), len(points_b))
    if n == 0:
        return np.nan
    cost = scipy.spatial.distance.cdist(points_a[:n], points_b[:n])        # (n, n) metres
    rows, cols = scipy.optimize.linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


def make_kinematic_edges(kinematics, n_bins=25, percentile_range=(0.5, 99.5)):
    """Histogram bin edges for each kinematic quantity, from reference data.

    Parameters
    ----------
    kinematics : dict {name: 1D float array}, e.g. pooled ground-truth
        kinematics from `pose_to_kinematics`.
    n_bins : int, number of equal-width bins between the percentiles.
    percentile_range : (low, high) percentiles that set the outer edges;
        values outside land in the under/overflow bins of `js_divergence`.

    Returns
    -------
    dict {name: (n_bins + 1,) float array of edges}.
    """
    edges = {}
    for name, values in kinematics.items():
        low, high = np.nanpercentile(values, percentile_range)
        edges[name] = np.linspace(low, high, n_bins + 1)
    return edges


def make_occupancy_edges(extent, bin_size):
    """2D histogram bin edges covering the arena.

    Parameters
    ----------
    extent : (4,) array (x_min, x_max, y_min, y_max) in metres (RatInABox
        `Env.extent`).
    bin_size : float, bin width in metres.

    Returns
    -------
    [x_edges, y_edges], each a 1D float array; positions outside the arena go
    to the under/overflow bins of `js_divergence`.
    """
    x_min, x_max, y_min, y_max = extent
    n_x = int(np.ceil((x_max - x_min) / bin_size))
    n_y = int(np.ceil((y_max - y_min) / bin_size))
    return [np.linspace(x_min, x_min + n_x * bin_size, n_x + 1),
            np.linspace(y_min, y_min + n_y * bin_size, n_y + 1)]


def count_wall_violations(poses, walls, extent):
    """Count steps that cross a wall and frames outside the arena.

    RatInABox agents cannot pass through walls; model rollouts are not
    constrained, so this measures how often they do.

    Parameters
    ----------
    poses : (K, T, 3) float array of (x m, y m, theta rad).
    walls : (n_walls, 2, 2) float array, wall segments [[x0, y0], [x1, y1]]
        (RatInABox `Env.walls`).
    extent : (4,) array (x_min, x_max, y_min, y_max) (RatInABox `Env.extent`).

    Returns
    -------
    dict with
        'crossings': (K,) int array, steps whose segment strictly crosses a
            wall (touching or sliding along a wall does not count).
        'outside': (K,) int array, frames outside the extent.
    NaN frames are ignored.
    """
    xy = poses[..., :2]                                                     # (K, T, 2)
    start = xy[:, :-1, None, :]                                             # (K, T-1, 1, 2)
    step = (xy[:, 1:] - xy[:, :-1])[:, :, None, :]                          # (K, T-1, 1, 2)
    wall_start = walls[None, None, :, 0, :]                                 # (1, 1, W, 2)
    wall_vec = (walls[:, 1, :] - walls[:, 0, :])[None, None]                # (1, 1, W, 2)

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    denom = cross(step, wall_vec)                                           # (K, T-1, W)
    offset = wall_start - start                                             # (K, T-1, W, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_step = cross(offset, wall_vec) / denom                            # position along the step
        t_wall = cross(offset, step) / denom                                # position along the wall
    crosses = (denom != 0) & (t_step > 0) & (t_step < 1) & (t_wall >= 0) & (t_wall <= 1)
    crossings = crosses.any(axis=-1).sum(axis=-1)                           # (K,)

    x_min, x_max, y_min, y_max = extent
    with np.errstate(invalid='ignore'):
        outside = ((xy[..., 0] < x_min) | (xy[..., 0] > x_max)
                   | (xy[..., 1] < y_min) | (xy[..., 1] > y_max))           # (K, T)
    return {'crossings': crossings.astype(int), 'outside': outside.sum(axis=-1).astype(int)}


def compare_future_sets(poses_a, poses_b, burn_in, horizons, dt, kinematic_edges, occupancy_edges):
    """All divergence metrics between two sets of futures, at several horizons.

    Parameters
    ----------
    poses_a, poses_b : (K_a, T, 3) and (K_b, T, 3) float arrays of
        (x m, y m, theta rad); T >= burn_in + max(horizons).
    burn_in : int, number of shared burn-in frames at the start of each row.
    horizons : sequence of int, numbers of future frames.
    dt : float, seconds per frame.
    kinematic_edges : dict {name: 1D edges} from `make_kinematic_edges`.
    occupancy_edges : [x_edges, y_edges] from `make_occupancy_edges`.

    Returns
    -------
    dict {metric name: (len(horizons),) float array} for every name in
    METRIC_NAMES; see the module docstring for definitions and units.
    """
    out = {name: np.full(len(horizons), np.nan) for name in METRIC_NAMES}
    for i, horizon in enumerate(horizons):
        kin_a = window_kinematics(poses_a, burn_in, horizon, dt)
        kin_b = window_kinematics(poses_b, burn_in, horizon, dt)
        for name in KINEMATIC_NAMES:
            out[f'js_{name}'][i] = js_divergence(kin_a[name], kin_b[name], [kinematic_edges[name]])
            if len(kin_a[name]) > 0 and len(kin_b[name]) > 0:
                out[f'w1_{name}'][i] = scipy.stats.wasserstein_distance(kin_a[name], kin_b[name])
        frame = burn_in + horizon - 1                                       # last future frame
        out['w1_position'][i] = wasserstein_2d_equal_n(poses_a[:, frame, :2], poses_b[:, frame, :2])
        occupancy_a = poses_a[:, burn_in:burn_in + horizon, :2].reshape(-1, 2)  # (K_a * horizon, 2)
        occupancy_b = poses_b[:, burn_in:burn_in + horizon, :2].reshape(-1, 2)  # (K_b * horizon, 2)
        out['js_occupancy'][i] = js_divergence(occupancy_a, occupancy_b, occupancy_edges)
    return out
