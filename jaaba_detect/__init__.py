"""jaaba_detect -- a MATLAB-free Python port of JAABADetect.m.

Apply a trained JAABA behavior classifier (exported to a plain .classifier.mat) to
tracking data and get per-frame, per-animal behavior scores. Three entry points:

  from jaaba_detect import jaaba_detect            # an experiment directory
  from jaaba_detect import jaaba_detect_from_X     # MABe X<split>_v3.mat arrays
  from jaaba_detect import jaaba_detect_from_track # flyllm/APF keypoint tracks

See README.md and notebooks/jaaba_detect_example.py for usage.
"""
from .detect import jaaba_detect, jaaba_detect_traj, save_scores
from .mabe_adapter import (
    jaaba_detect_from_X,
    jaaba_detect_from_track,
    trajectories_from_X,
    trajectories_from_track,
    load_X_video,
    track_to_apt,
    load_ellipse_coeffs,
    flyllm_keypoint_names,
    APT_LANDMARK_NAMES,
    FLYLLM_KEYPOINT_NAMES,
    FLYLLM_TO_APT_NAME,
    DEFAULT_PXPERMM,
    DEFAULT_FPS,
)
from . import jab_io, trx_io

__all__ = [
    "jaaba_detect", "jaaba_detect_traj", "save_scores",
    "jaaba_detect_from_X", "jaaba_detect_from_track",
    "trajectories_from_X", "trajectories_from_track",
    "load_X_video", "track_to_apt", "load_ellipse_coeffs",
    "flyllm_keypoint_names",
    "APT_LANDMARK_NAMES", "FLYLLM_KEYPOINT_NAMES", "FLYLLM_TO_APT_NAME",
    "DEFAULT_PXPERMM", "DEFAULT_FPS",
    "jab_io", "trx_io",
]
