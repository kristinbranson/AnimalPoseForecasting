"""Checks of the mouse pipeline: mousellm/features.py, mousellm/convert_mabe22.py, experiments/mouse.py.

Run from the repository root with
    python -m pytest tests/test_mouse.py
or without pytest as
    python tests/test_mouse.py
The tests use made-up keypoints, so need no data, trained model or GPU.
"""
import testing_utils  # puts this checkout first on sys.path; must precede the imports below

import numpy as np

from apf.data import get_flip_idx
from experiments.mouse import BodyCentricKP, Sensory
from mousellm import convert_mabe22
from mousellm.config import SENSORY_PARAMS, keypointnames, kporigin, kpheading_back, kpheading_front, posenames
from mousellm.features import body_centric_kp, get_sensory_feature_idx

N_KEYPOINTS = len(keypointnames)


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
    sensory = Sensory()(Xkp)
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
    sensory = Sensory()(np.stack([a, behind])[:, None])
    s = get_sensory_feature_idx()['othermice_vision']
    lit = np.nonzero(sensory[0, 0, s[0]:s[1]] > 0)[0]
    n_oma = SENSORY_PARAMS['n_oma']
    assert 0 in lit and n_oma - 1 in lit and len(lit) < n_oma // 4, lit


def test_touch_is_zero_far_away_and_high_close():
    """ Touch is 0 when the other mouse is well over a body length away, and near 1 for touching. """
    a = straight_mouse(0., 0., 0.)
    s = get_sensory_feature_idx()['othermice_touch']
    far = Sensory()(np.stack([a, straight_mouse(0., 30., 0.)])[:, None])[0, 0, s[0]:s[1]]
    assert np.all(far == 0)
    near = Sensory()(np.stack([a, straight_mouse(0., .5, 0.)])[:, None])[0, 0, s[0]:s[1]]
    assert near.max() > .8


def test_sensory_nan_iff_no_data():
    """ Frames without data get NaN sensory features, and the rest none. """
    rng = np.random.default_rng(1)
    Xkp = np.stack([np.stack([straight_mouse(*rng.uniform(-20, 20, 2), rng.uniform(-np.pi, np.pi))
                              for _ in range(10)]) for _ in range(3)])    # (3, 10, 2, n_keypoints)
    isdata = rng.uniform(size=(10, 3)) > .3
    sensory = Sensory()(Xkp, isdata=isdata)
    assert np.array_equal(np.isnan(sensory).any(-1), ~isdata.T)
    assert not np.isnan(sensory[isdata.T]).any()


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
    from mousellm.config import ARENA_CENTER_PX, CM_PER_PX
    assert np.allclose(X[:, 0, 0, 0], (200 + np.arange(N_KEYPOINTS) - ARENA_CENTER_PX[0]) * CM_PER_PX)
    assert np.allclose(X[:, 1, 0, 0], (250 - ARENA_CENTER_PX[1]) * CM_PER_PX)
    # the same sequence in the 800-pixel frame gives the same cm
    X800, _, _ = convert_mabe22.clean_sequence(np.round(kp * 800 / 512).astype(np.int32), 800, first_id=7)
    assert np.allclose(X800, X, atol=CM_PER_PX, equal_nan=True)


if __name__ == '__main__':
    testing_utils.run_as_script(globals())
