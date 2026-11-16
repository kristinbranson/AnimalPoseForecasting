"""Checks of the mouse pipeline: mousellm/features.py, mousellm/convert_mabe22.py, experiments/mouse.py.

Run from the repository root with
    python -m pytest tests/test_mouse.py
or without pytest as
    python tests/test_mouse.py
The tests use made-up keypoints, so need no data, trained model or GPU.
"""
import testing_utils  # puts this checkout first on sys.path; must precede the imports below

import numpy as np

from apf.data import flip_agents, flip_arena_bounds, get_flip_idx
from experiments.mouse import BodyCentricKP, Sensory
from mousellm import arena, convert_mabe22
from mousellm.config import (
    SENSORY_PARAMS, keypointnames, kphead, kporigin, kpheading_back, kpheading_front, posenames
)
from mousellm.features import body_centric_kp, compute_wall_vision, get_sensory_feature_idx

N_KEYPOINTS = len(keypointnames)
# An arena so big that the walls are far from every mouse in the tests of the other sensory features
BIG_ARENA_BOUNDS = np.array([-1000., 1000., -1000., 1000.])


def sensory_in_big_arena(Xkp, **kwargs):
    """ Sensory features of Xkp, (n_agents, n_frames, 2, n_keypoints), in BIG_ARENA_BOUNDS. """
    return Sensory()(Xkp, arena_bounds=np.tile(BIG_ARENA_BOUNDS, (Xkp.shape[1], 1)), **kwargs)


def straight_mouse(x0, y0, heading, length=6.):
    """ (2, n_keypoints) keypoints of a mouse at (x0, y0) facing heading, with all its keypoints on its
    midline, nose ahead and tail behind, spread over length cm, plus a little left-right offset for the
    paired keypoints so that no two coincide. """
    along = np.linspace(length / 2, -length / 2, N_KEYPOINTS)
    across = np.array([.3 if 'left' in n else -.3 if 'right' in n else 0. for n in keypointnames])
    # put the center keypoint exactly at (x0, y0)
    along = along - along[kporigin]
    c, s = np.cos(heading), np.sin(heading)
    return np.array([x0 + c * along - s * across, y0 + s * along + c * across])


def test_pose_round_trip():
    """ BodyCentricKP.invert undoes BodyCentricKP.apply. """
    rng = np.random.default_rng(0)
    Xkp = rng.normal(size=(3, 50, 2, N_KEYPOINTS)) * 5
    op = BodyCentricKP()
    pose = op.apply(Xkp)
    assert pose.shape == (3, 50, len(posenames))
    assert np.allclose(op.invert(pose), Xkp)


def test_body_frame_points_along_y():
    """ In the body frame the center keypoint is at the origin and the tail base to neck vector points along
    +y, and the orientation is the heading minus pi/2, as for the flies. """
    heading = .7
    Xkp = straight_mouse(3., -2., heading)[None, None]
    Xn, origin, theta = body_centric_kp(Xkp)
    assert np.allclose(origin[0, 0], [3., -2.])
    assert np.allclose(Xn[0, 0, :, kporigin], 0.)
    v = Xn[0, 0, :, kpheading_front] - Xn[0, 0, :, kpheading_back]
    assert abs(v[0]) < 1e-9 and v[1] > 0
    assert np.isclose(theta[0, 0], heading - np.pi / 2)


def test_flip_swaps_left_and_right():
    """ get_flip_idx pairs each left keypoint with its right one, for flip augmentation. """
    flipidx = get_flip_idx(keypointnames)
    for i, name in enumerate(keypointnames):
        expected = name.replace('left', 'RIGHT').replace('right', 'left').replace('RIGHT', 'right')
        assert keypointnames[flipidx[i]] == expected


def test_vision_sees_a_mouse_ahead():
    """ A mouse straight ahead lights up the bins around straight ahead, and no others; an untracked mouse
    is not seen. """
    a = straight_mouse(0., 0., 0.)
    ahead = straight_mouse(15., 0., np.pi / 2)
    untracked = np.full_like(a, np.nan)
    Xkp = np.stack([a, ahead, untracked])[:, None]                       # (3 mice, 1 frame, 2, n_keypoints)
    sensory = sensory_in_big_arena(Xkp)
    s = get_sensory_feature_idx()['othermice_vision']
    vision = sensory[0, 0, s[0]:s[1]]
    n_oma = SENSORY_PARAMS['n_oma']
    lit = np.nonzero(vision > 0)[0]
    # straight ahead is angle 0, bin n_oma // 2
    assert len(lit) > 0 and np.all(np.abs(lit - n_oma // 2) <= 3), lit
    assert np.all(vision[lit] < 1)


def test_vision_wraps_behind():
    """ A mouse straight behind, which spans the angle -pi = pi, lights up bins at both ends. """
    a = straight_mouse(0., 0., 0.)
    behind = straight_mouse(-15., 0., np.pi / 2)
    sensory = sensory_in_big_arena(np.stack([a, behind])[:, None])
    s = get_sensory_feature_idx()['othermice_vision']
    lit = np.nonzero(sensory[0, 0, s[0]:s[1]] > 0)[0]
    n_oma = SENSORY_PARAMS['n_oma']
    assert 0 in lit and n_oma - 1 in lit and len(lit) < n_oma // 4, lit


def test_touch_is_zero_far_away_and_high_close():
    """ Touch is 0 when the other mouse is well over a body length away, and near 1 for touching. """
    a = straight_mouse(0., 0., 0.)
    s = get_sensory_feature_idx()['othermice_touch']
    far = sensory_in_big_arena(np.stack([a, straight_mouse(0., 30., 0.)])[:, None])[0, 0, s[0]:s[1]]
    assert np.all(far == 0)
    near = sensory_in_big_arena(np.stack([a, straight_mouse(0., .5, 0.)])[:, None])[0, 0, s[0]:s[1]]
    assert near.max() > .8


def test_sensory_nan_iff_no_data():
    """ Frames without data get NaN sensory features, and the rest none. """
    rng = np.random.default_rng(1)
    Xkp = np.stack([np.stack([straight_mouse(*rng.uniform(-20, 20, 2), rng.uniform(-np.pi, np.pi))
                              for _ in range(10)]) for _ in range(3)])    # (3, 10, 2, n_keypoints)
    isdata = rng.uniform(size=(10, 3)) > .3
    sensory = sensory_in_big_arena(Xkp, isdata=isdata)
    assert np.array_equal(np.isnan(sensory).any(-1), ~isdata.T)
    assert not np.isnan(sensory[isdata.T]).any()


def wall_vision(Xkp, arena_bounds):
    """ The wall_vision sensory features of one mouse, Xkp (2, n_keypoints), alone in arena_bounds (4,). """
    sensory = Sensory()(Xkp[None, None], arena_bounds=arena_bounds[None])
    s = get_sensory_feature_idx()['wall_vision']
    return sensory[0, 0, s[0]:s[1]]


def nearness(d):
    """ The vision feature for distance d. """
    return 1. - np.minimum(1., SENSORY_PARAMS['wall_vision_mult'] * d ** SENSORY_PARAMS['wall_vision_exp'])


def brute_force_wall_vision(head, theta, arena_bounds, n_rays_per_bin=400):
    """ compute_wall_vision for one frame, by casting n_rays_per_bin rays across each bin from the head moved
    inside the arena, and taking the shortest. """
    x0, x1, y0, y1 = arena_bounds
    h = np.clip(head, [x0, y0], [x1, y1])
    n_oma = SENSORY_PARAMS['n_oma']
    step = 2. * np.pi / n_oma
    out = []
    for b in range(n_oma):
        angle = theta - np.pi + (b + np.linspace(0., 1., n_rays_per_bin)) * step
        dx, dy = np.cos(angle), np.sin(angle)
        with np.errstate(divide='ignore', invalid='ignore'):
            tx = np.where(dx > 0, (x1 - h[0]) / dx, np.where(dx < 0, (x0 - h[0]) / dx, np.inf))
            ty = np.where(dy > 0, (y1 - h[1]) / dy, np.where(dy < 0, (y0 - h[1]) / dy, np.inf))
        out.append(np.minimum(tx, ty).min())
    return nearness(np.array(out))


def test_wall_vision_ahead_and_behind():
    """ A mouse facing +x in a square arena sees the +x wall straight ahead, in the bin starting at straight
    ahead, at the head's perpendicular distance to it, and the -x wall straight behind, in the first bin. """
    bounds = np.array([-26., 26., -26., 26.])
    a = straight_mouse(0., 0., 0.)
    head = a[:, kphead].mean(axis=-1)
    v = wall_vision(a, bounds)
    n_oma = SENSORY_PARAMS['n_oma']
    assert np.isclose(v[n_oma // 2], nearness(26. - head[0]))
    assert np.isclose(v[0], nearness(26. + head[0]))
    assert np.all((v > 0) & (v < 1))


def test_wall_vision_matches_brute_force():
    """ compute_wall_vision matches casting many rays across each bin, for heads anywhere, including near
    and past the walls, facing any way. """
    rng = np.random.default_rng(3)
    bounds = np.array([-20., 30., -24., 27.])
    n = 40
    head = rng.uniform(-28., 34., (n, 2))
    theta = rng.uniform(-np.pi, np.pi, n)
    v = compute_wall_vision(head, theta, np.tile(bounds, (n, 1)))
    expected = np.stack([brute_force_wall_vision(head[i], theta[i], bounds) for i in range(n)], axis=1)
    assert np.abs(v - expected).max() < 1e-3


def test_wall_vision_is_one_toward_a_wall_the_head_is_past():
    """ A head past the +x wall sees it at distance 0, so with nearness 1, in every bin with a direction
    within 90 degrees of +x, and less near in the others. """
    bounds = np.array([-26., 26., -26., 26.])
    v = compute_wall_vision(np.array([[28., 0.]]), np.array([0.]), bounds[None])[:, 0]
    n_oma = SENSORY_PARAMS['n_oma']
    ahead = np.arange(n_oma // 4, 3 * n_oma // 4)                       # bins from -90 to +90 degrees
    assert np.allclose(v[ahead], 1.) and np.all(np.delete(v, ahead) < 1.)


def test_flipped_mouse_in_flipped_arena_mirrors_wall_vision():
    """ Flipping a mouse, as flip augmentation does, along with its arena reverses its wall vision bins. """
    bounds = np.array([-20., 30., -24., 27.])
    a = straight_mouse(4., -9., .8)
    flipped = flip_agents(a.T[:, :, None], keypointnames)[:, :, 0].T   # flip_agents takes (n_kp, 2, T)
    flipped_bounds = flip_arena_bounds(bounds[None])[0]
    assert np.allclose(flipped_bounds, [-30., 20., -24., 27.])
    v, vf = wall_vision(a, bounds), wall_vision(flipped, flipped_bounds)
    assert np.allclose(vf, v[::-1], atol=1e-6)


def test_placements_and_assignment():
    """ estimate_arena_bounds groups video floor boxes into placements, and assigns sequences without videos
    to the placement whose walls their keypoints run along. """
    rng = np.random.default_rng(3)
    placement_bounds = np.array([[-26., 26., -26., 26.], [-23., 29., -21., 31.], [-28., 24., -30., 22.]])
    n_per = 3 * arena.MIN_PLACEMENT_N_SEQS
    true_placement = np.repeat(np.arange(3), 2 * n_per)
    hists, video_bounds = [], []
    for i, p in enumerate(true_placement):
        x0, x1, y0, y1 = placement_bounds[p]
        # keypoints mostly near the walls, some anywhere inside
        n = 2000
        X = np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])
        wall = rng.integers(4, size=n // 2)
        X[0, :n // 2] = np.where(wall == 0, x0 + rng.uniform(0, 2, n // 2), np.where(wall == 1, x1 - rng.uniform(0, 2, n // 2), X[0, :n // 2]))
        X[1, :n // 2] = np.where(wall == 2, y0 + rng.uniform(0, 2, n // 2), np.where(wall == 3, y1 - rng.uniform(0, 2, n // 2), X[1, :n // 2]))
        hists.append(arena.keypoint_histograms(X[None, :, :, None]))
        # every other sequence has a video, whose floor box is measured with a little noise
        video_bounds.append(placement_bounds[p] + rng.normal(scale=.1, size=4) if i % 2 == 0 else np.full(4, np.nan))
    out = arena.estimate_arena_bounds(np.array(hists), np.array(video_bounds))
    assert len(out['placement_bounds']) == 3
    # placements are numbered in their own order; map them to the true ones by their arenas
    found = np.array([np.argmin(np.abs(placement_bounds - b).sum(1)) for b in out['placement_bounds']])
    assert np.allclose(out['placement_bounds'], placement_bounds[found], atol=.2)
    assert np.mean(found[out['placement']] == true_placement) > .95
    assert np.allclose(out['arena_bounds'], out['placement_bounds'][out['placement']])


def test_short_gaps_are_filled():
    """ Gaps of up to MAX_GAP_N_FRAMES frames are interpolated, and longer ones, ones at the ends, and ones
    across too big a jump are left. """
    n = convert_mabe22.MAX_GAP_N_FRAMES
    x = np.tile(np.arange(30.)[:, None], (1, 2))
    x[3:3 + n] = np.nan                                 # short: filled
    x[12:13 + n] = np.nan                               # one frame too long: left
    x[:2] = np.nan                                      # at the start: left
    expected = np.isnan(x)
    expected[3:3 + n] = False
    filled = x.copy()
    assert convert_mabe22.fill_short_gaps_bang(filled) == n
    assert np.array_equal(np.isnan(filled), expected)
    assert np.allclose(filled[3:3 + n, 0], np.arange(3, 3 + n))
    jump = np.zeros((10, 2))
    jump[5:] = convert_mabe22.MAX_STEP_PX + 1
    jump[4] = np.nan
    assert convert_mabe22.fill_short_gaps_bang(jump) == 0


def test_clean_sequence_tracks():
    """ clean_sequence converts (y, x) pixels to (x, y) cm, marks (0, 0) keypoints missing, and starts a new
    track after an untracked frame, at a jump and at a flip. """
    n_frames = 20
    xy = np.zeros((n_frames, 1, N_KEYPOINTS, 2))
    xy[..., 0] = 200 + np.arange(N_KEYPOINTS)                           # x
    xy[..., 1] = 250.                                                   # y
    xy[15:, :, :, 0] += convert_mabe22.MAX_STEP_PX + 5                  # a jump at frame 15
    kp = np.round(xy[..., ::-1]).astype(np.int32)                      # (y, x)
    kp[6:12] = 0                                                       # untracked, too long to fill
    X, ids, stats = convert_mabe22.clean_sequence(kp, 512, first_id=7)
    assert X.shape == (N_KEYPOINTS, 2, n_frames, 1)
    tracked = ids[:, 0] >= 0
    assert np.array_equal(np.isnan(X).any(axis=(0, 1))[:, 0], ~tracked)
    assert np.array_equal(np.unique(ids[tracked, 0]), [7, 8, 9])
    assert ids[5, 0] == 7 and ids[11, 0] == -1 and ids[12, 0] == 8 and ids[14, 0] == 8 and ids[15, 0] == 9
    assert stats['n_jumps'] == 1
    # a head-tail flip at frame 17 starts another track
    flipped = kp.copy()
    flipped[17:] = flipped[17:, :, ::-1]
    _, ids, stats = convert_mabe22.clean_sequence(flipped, 512, first_id=7)
    assert ids[16, 0] == 9 and ids[17, 0] == 10 and stats['n_jumps'] == 2
    from mousellm.config import CM_PER_PX
    # cm about the frame's center
    assert np.allclose(X[:, 0, 0, 0], (200 + np.arange(N_KEYPOINTS) - 256) * CM_PER_PX)
    assert np.allclose(X[:, 1, 0, 0], (250 - 256) * CM_PER_PX)
    # the same sequence in the 800-pixel frame gives the same cm
    X800, _, _ = convert_mabe22.clean_sequence(np.round(kp * 800 / 512).astype(np.int32), 800, first_id=7)
    assert np.allclose(X800, X, atol=CM_PER_PX, equal_nan=True)


def test_animate_pose_writes_a_video():
    """ mousellm.simulation.animate_pose saves one video frame per track frame, skipping NaN keypoints. """
    import os
    import subprocess
    import tempfile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mousellm.simulation import animate_pose
    n_frames = 5
    X = np.array([[straight_mouse(-10 + t + 8 * m, 5 * m, .3 * m) for t in range(n_frames)] for m in range(3)])
    X_sim = X.copy()
    X_sim[2, 3:] = np.nan                                              # an untracked mouse
    with tempfile.TemporaryDirectory() as dir_path:
        out_file_path = os.path.join(dir_path, 'anim.mp4')
        animate_pose({'Real': X, 'Simulated': X_sim}, burn_in=2, out_file_path_maybe=out_file_path)
        plt.close('all')
        n_video_frames = subprocess.run(
            ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',
             'stream=nb_read_frames', '-of', 'csv=p=0', out_file_path], capture_output=True, text=True).stdout
        assert int(n_video_frames) == n_frames


if __name__ == '__main__':
    testing_utils.run_as_script(globals())
