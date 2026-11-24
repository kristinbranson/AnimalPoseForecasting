"""Sanity checks for synthrat/trajectory_divergence.py on hand-built inputs.

Run with `python tests/synthrat/test_trajectory_divergence.py`, as for the other tests in
this repository; plain asserts rather than pytest, which is not in the environment.
"""

import numpy as np

from synthrat import trajectory_divergence as td
from synthrat.sensory import ORIENTATION_OFFSET

# Relative tolerance for comparisons of a metric against its exact expected value, matching
# what pytest.approx uses by default.
RELATIVE_TOLERANCE = 1e-6

DT = 0.05  # seconds per frame, as in the synthrat data
BURN_IN = 4
HORIZON = 6
# The synthrat arena: unit square plus an internal wall from (0.8, 0) to (0.8, 0.8).
EXTENT = np.array([0.0, 1.0, 0.0, 1.0])
WALLS = np.array([
    [[0, 0], [1, 0]], [[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [0, 0]],
    [[0.8, 0], [0.8, 0.8]],
], dtype=float)


def straight_line(start, heading, speed, n_frames):
    """(n_frames, 3) pose moving at `speed` m/s along `heading`, facing that way.

    The pose's third column is the orientation in the fly convention the synthrat data
    uses, heading minus ORIENTATION_OFFSET, not the heading itself.
    """
    t = np.arange(n_frames) * DT
    xy = np.asarray(start) + speed * t[:, None] * np.array([np.cos(heading), np.sin(heading)])
    orientation = heading - ORIENTATION_OFFSET
    return np.concatenate([xy, np.full((n_frames, 1), orientation)], axis=1)


def test_js_identical_is_zero():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=500)
    edges = [np.linspace(-3, 3, 26)]
    np.testing.assert_allclose(td.js_divergence(samples, samples, edges), 0.0, atol=1e-12)


def test_js_disjoint_is_one_bit():
    """Distributions with no overlap are 1 bit apart, in one and in two dimensions."""
    for dims in (1, 2):
        rng = np.random.default_rng(0)
        samples_a = rng.uniform(0.0, 0.4, size=(300, dims))
        samples_b = rng.uniform(0.6, 1.0, size=(300, dims))
        edges = [np.linspace(0, 1, 11)] * dims
        np.testing.assert_allclose(td.js_divergence(samples_a, samples_b, edges), 1.0,
                                   rtol=RELATIVE_TOLERANCE, err_msg=f'dims={dims}')


def test_js_counts_mass_outside_edges():
    # all of b lies above the last edge -> overflow bin, disjoint from a
    samples_a = np.full(50, 0.5)
    samples_b = np.full(50, 5.0)
    np.testing.assert_allclose(td.js_divergence(samples_a, samples_b, [np.linspace(0, 1, 11)]),
                               1.0, rtol=RELATIVE_TOLERANCE)


def test_wasserstein_2d_identical_and_shifted():
    rng = np.random.default_rng(0)
    points = rng.uniform(size=(20, 2))
    np.testing.assert_allclose(td.wasserstein_2d_equal_n(points, points), 0.0, atol=1e-12)
    shift = np.array([0.3, 0.4])  # length 0.5
    shuffled = rng.permutation(points) + shift
    np.testing.assert_allclose(td.wasserstein_2d_equal_n(points, shuffled), 0.5,
                               rtol=RELATIVE_TOLERANCE)


def test_kinematics_straight_line():
    speed = 0.1
    pose = straight_line([0.2, 0.3], heading=np.pi / 4, speed=speed, n_frames=10)
    kinematics = td.pose_to_kinematics(pose, DT)
    np.testing.assert_allclose(kinematics['forward'], speed)
    np.testing.assert_allclose(kinematics['sideways'], 0.0, atol=1e-12)
    np.testing.assert_allclose(kinematics['angular'], 0.0, atol=1e-12)


def test_kinematics_turning_in_place():
    angular_speed = 1.0  # rad/s
    theta = np.arange(10) * angular_speed * DT
    pose = np.stack([np.full(10, 0.5), np.full(10, 0.5), theta], axis=1)
    # ratinabox.utils.get_angle adds 1e-6 to x before arctan2, so angles are off by ~1e-6 relative
    np.testing.assert_allclose(td.pose_to_kinematics(pose, DT)['angular'], angular_speed, rtol=1e-5)


def test_wall_crossing_and_outside():
    n_frames = 2
    through_wall = straight_line([0.75, 0.5], heading=0.0, speed=0.1 / DT, n_frames=n_frames)  # (0.75 -> 0.85, 0.5)
    above_wall = straight_line([0.75, 0.9], heading=0.0, speed=0.1 / DT, n_frames=n_frames)    # passes above wall end
    leaving = straight_line([0.95, 0.5], heading=0.0, speed=0.1 / DT, n_frames=n_frames)       # exits right side
    poses = np.stack([through_wall, above_wall, leaving])                    # (3, 2, 3)
    violations = td.count_wall_violations(poses, WALLS, EXTENT)
    np.testing.assert_array_equal(violations['crossings'], [1, 0, 1])
    np.testing.assert_array_equal(violations['outside'], [0, 0, 1])


def test_compare_future_sets_self_and_shifted():
    rng = np.random.default_rng(0)
    n_samples = 8
    n_frames = BURN_IN + HORIZON
    poses = np.stack([
        straight_line(rng.uniform(0.2, 0.4, size=2), heading=rng.uniform(-np.pi, np.pi),
                      speed=rng.uniform(0.05, 0.2), n_frames=n_frames)
        for _ in range(n_samples)
    ])                                                                       # (K, T, 3)
    kinematic_edges = {name: np.linspace(-1, 1, 26) for name in td.KINEMATIC_NAMES}
    occupancy_edges = td.make_occupancy_edges(EXTENT, 0.1)
    horizons = [2, HORIZON]

    same = td.compare_future_sets(poses, poses, BURN_IN, horizons, DT, kinematic_edges, occupancy_edges)
    for name in td.METRIC_NAMES:
        np.testing.assert_allclose(same[name], 0.0, atol=1e-12, err_msg=name)

    shifted = poses.copy()
    shifted[:, BURN_IN:, 0] += 0.5                                           # move the future 0.5 m in x
    different = td.compare_future_sets(poses, shifted, BURN_IN, horizons, DT, kinematic_edges, occupancy_edges)
    np.testing.assert_allclose(different['w1_position'], 0.5)
    assert np.all(different['js_occupancy'] > 0.5)


if __name__ == '__main__':
    import sys

    failures = 0
    for name, test in sorted(dict(globals()).items()):
        if not name.startswith('test_') or not callable(test):
            continue
        try:
            test()
            print(f'PASS {name}')
        except Exception as error:                                   # noqa: BLE001 - report and continue
            failures += 1
            print(f'FAIL {name}: {type(error).__name__}: {error}')
    print(f'{failures} failed')
    sys.exit(1 if failures else 0)
