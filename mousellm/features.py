"""Pose and sensory features of the MABe22 mice, computed from keypoints.

Keypoint arrays here are (n_agents, n_frames, 2, n_keypoints), the layout of the 'keypoints' Data in
experiments/mouse.py, with positions in cm.  There are no wall features: the arena's position and size
vary between videos and aren't known for the pose-only data (see mousellm/convert_mabe22.py).
"""
import warnings

import numpy as np

from apf.utils import modrange, rotate_2d_points
from mousellm.config import (
    SENSORY_PARAMS, kporigin, kpheading_back, kpheading_front, kprelative,
    kphead, kpvision_other, kptouch, kptouch_other,
)

# compute_sensory computes this many frames at a time
SENSORY_CHUNK_N_FRAMES = 100000


def body_centric_kp(Xkp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centers keypoints on the center keypoint and rotates them so that the mouse points along +y.

    Args:
        Xkp: (n_agents, n_frames, 2, n_keypoints) keypoint positions.

    Returns:
        Xn: (n_agents, n_frames, 2, n_keypoints) keypoints in the mouse's frame.
        origin: (n_agents, n_frames, 2) position of the center keypoint.
        theta: (n_agents, n_frames) orientation, the angle of the tail base to neck vector minus pi/2, as
            for the flies.
    """
    origin = Xkp[..., kporigin]
    vec = Xkp[..., kpheading_front] - Xkp[..., kpheading_back]
    theta = modrange(np.arctan2(vec[..., 1], vec[..., 0]) - np.pi / 2, -np.pi, np.pi)
    # rotate_2d_points takes points on axis 0 and (x, y) on axis 1
    Xc = np.moveaxis(Xkp - origin[..., None], -1, 0)                   # (n_keypoints, n_agents, n_frames, 2)
    Xn = rotate_2d_points(np.moveaxis(Xc, -1, 1), theta)               # (n_keypoints, 2, n_agents, n_frames)
    return np.moveaxis(Xn, (0, 1), (-1, -2)), origin, theta


def body_centric_kp_to_pose(Xkp: np.ndarray) -> np.ndarray:
    """Pose features from keypoints: origin x and y, orientation, then the body-frame x and y of every
    keypoint but the origin one (mousellm.config.posenames).

    Args:
        Xkp: (n_agents, n_frames, 2, n_keypoints) keypoint positions.

    Returns:
        pose: (n_agents, n_frames, n_pose_features)
    """
    Xn, origin, theta = body_centric_kp(Xkp)
    # (n_agents, n_frames, n_relative_keypoints, 2), flattened to x, y per keypoint
    relative = np.swapaxes(Xn[..., kprelative], -1, -2).reshape(Xn.shape[:2] + (-1,))
    return np.concatenate([origin, theta[..., None], relative], axis=-1)


def pose_to_kp(pose: np.ndarray) -> np.ndarray:
    """Inverse of body_centric_kp_to_pose.

    Args:
        pose: (..., n_pose_features)

    Returns:
        Xkp: (..., 2, n_keypoints) keypoint positions.
    """
    origin = pose[..., :2]
    theta = pose[..., 2]
    relative = pose[..., 3:].reshape(pose.shape[:-1] + (len(kprelative), 2))
    Xn = np.zeros(pose.shape[:-1] + (len(kprelative) + 1, 2), dtype=pose.dtype)
    Xn[..., kprelative, :] = relative
    # rotate by -theta: rotate_2d_points takes points on axis 0 and (x, y) on axis 1
    Xr = rotate_2d_points(np.moveaxis(Xn, (-2, -1), (0, 1)), -theta)
    return np.moveaxis(Xr, (0, 1), (-1, -2)) + origin[..., None]


def compute_othermice_vision(head: np.ndarray, theta: np.ndarray, Xvision_other: np.ndarray) -> np.ndarray:
    """How near the other mice look in each direction, from one mouse's head.

    Each other mouse covers the direction bins between the leftmost and rightmost of its vision keypoints,
    as seen from the head, and its distance there is the distance to its nearest vision keypoint.  The
    feature is 1 - min(1, mult * d ** exp), where d is the nearest other mouse's distance in that bin: 0
    for nothing in view, and nearer 1 for nearer mice.

    Args:
        head: (n_frames, 2) head position.
        theta: (n_frames,) heading angle, the direction the mouse faces.
        Xvision_other: (n_frames, 2, n_vision_keypoints, n_other) vision keypoints of the other mice.  An
            untracked mouse is NaN and is out of view.

    Returns:
        vision: (n_oma, n_frames)
    """
    n_oma = SENSORY_PARAMS['n_oma']
    step = 2. * np.pi / n_oma
    d = Xvision_other - head[:, :, None, None]                          # (n_frames, 2, n_kp, n_other)
    dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)                         # (n_frames, n_kp, n_other)
    angle = np.arctan2(d[:, 1], d[:, 0]) - theta[:, None, None]
    # angles relative to the direction of the mouse's nearest keypoint, so that a mouse spanning the
    # direction behind (-pi = pi) doesn't wrap
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mind = np.nanmin(dist, axis=1)                                  # (n_frames, n_other)
        inearest = np.nanargmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
    anglenearest = np.take_along_axis(angle, inearest[:, None], axis=1)
    rel = modrange(angle - anglenearest, -np.pi, np.pi)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        lo = modrange(anglenearest[:, 0] + np.nanmin(rel, axis=1), -np.pi, np.pi)
        hi = modrange(anglenearest[:, 0] + np.nanmax(rel, axis=1), -np.pi, np.pi)
    blo = np.floor((lo + np.pi) / step)
    bhi = np.floor((hi + np.pi) / step)
    # bin b is covered if it's within (bhi - blo) mod n_oma bins after blo
    bins = np.arange(n_oma)[:, None, None]
    covered = np.mod(bins - blo[None], n_oma) <= np.mod(bhi - blo, n_oma)[None]
    covered &= ~np.isnan(mind)[None]
    nearest = np.where(covered, mind[None], np.inf).min(axis=-1)          # (n_oma, n_frames)
    return 1. - np.minimum(1., SENSORY_PARAMS['othermice_vision_mult'] *
                           nearest ** SENSORY_PARAMS['othermice_vision_exp'])


def compute_othermice_touch(Xtouch: np.ndarray, Xtouch_other: np.ndarray) -> np.ndarray:
    """How near each of one mouse's touch keypoints is to each touch keypoint of the nearest other mouse.

    The feature is 1 - min(1, mult * d ** exp), with d the distance: 0 beyond about a body length.

    Args:
        Xtouch: (n_frames, 2, n_touch) the mouse's touch keypoints.
        Xtouch_other: (n_frames, 2, n_touch_other, n_other) the other mice's touch keypoints.  An untracked
            mouse is NaN and is out of reach.

    Returns:
        touch: (n_touch * n_touch_other, n_frames), touch keypoint major.
    """
    d = Xtouch[:, :, :, None, None] - Xtouch_other[:, :, None]          # (n_frames, 2, n_touch, n_touch_other, n_other)
    dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    dist = np.where(np.isnan(dist), np.inf, dist).min(axis=-1)          # (n_frames, n_touch, n_touch_other)
    touch = 1. - np.minimum(1., SENSORY_PARAMS['othermice_touch_mult'] * dist ** SENSORY_PARAMS['othermice_touch_exp'])
    return touch.reshape((touch.shape[0], -1)).T


def get_sensory_feature_idx() -> dict[str, list[int]]:
    """ The [start, end) feature indices of each kind of sensory feature compute_sensory returns. """
    n_vision = SENSORY_PARAMS['n_oma']
    n_touch = len(kptouch) * len(kptouch_other)
    return {'othermice_vision': [0, n_vision], 'othermice_touch': [n_vision, n_vision + n_touch]}


def compute_sensory(Xkp: np.ndarray, mousenum: int, isdata: np.ndarray | None = None) -> np.ndarray:
    """Sensory features of one mouse: vision of the other mice, then touch of the other mice.

    Args:
        Xkp: (n_agents, n_frames, 2, n_keypoints) keypoint positions of all the mice.
        mousenum: index of the mouse to compute the features of.
        isdata: (n_frames,) whether the mouse has data at each frame.  Other frames are NaN.  None for all.

    Returns:
        sensory: (n_frames, n_sensory_features), with the features in get_sensory_feature_idx order.
    """
    n_frames = Xkp.shape[1]
    idx = get_sensory_feature_idx()
    n_sensory = max(v[1] for v in idx.values())
    sensory = np.full((n_frames, n_sensory), np.nan)
    if isdata is None:
        isdata = np.ones(n_frames, dtype=bool)
    if not np.any(isdata):
        return sensory
    sv = idx['othermice_vision']
    st = idx['othermice_touch']
    frames = np.nonzero(isdata)[0]
    # in chunks of frames, since the intermediate arrays take a few kB per frame
    for i0 in range(0, len(frames), SENSORY_CHUNK_N_FRAMES):
        framescurr = frames[i0:i0 + SENSORY_CHUNK_N_FRAMES]
        X = Xkp[:, framescurr]                                          # (n_agents, n, 2, n_keypoints)
        main = X[mousenum]
        other = np.delete(X, mousenum, axis=0)
        if other.shape[0] == 0:
            # a mouse alone gets one untracked companion
            other = np.full((1,) + main.shape, np.nan)
        other = np.moveaxis(other, 0, -1)                               # (n, 2, n_keypoints, n_other)
        _, _, theta = body_centric_kp(main[None])
        head = main[..., kphead].mean(axis=-1)
        vision = compute_othermice_vision(head, theta[0] + np.pi / 2, other[:, :, kpvision_other])
        touch = compute_othermice_touch(main[..., kptouch], other[:, :, kptouch_other])
        sensory[framescurr, sv[0]:sv[1]] = vision.T
        sensory[framescurr, st[0]:st[1]] = touch.T
    return sensory
