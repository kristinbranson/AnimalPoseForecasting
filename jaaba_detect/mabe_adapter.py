"""Run a trained JAABA classifier on MABe-style registered arrays (the
`X<split>_v3.mat` ground-truth data, or flyllm/APF simulated tracks) instead of an
experiment directory.

The MABe `X` array is per-frame, per-fly, mm-registered pose + 21 APT keypoints.
It is *exactly* the experiment's `apt.trk` keypoints and FlyTracker ellipse divided
by `pxpermm` (verified to ~1e-14 mm against registered_trx + apt.trk). JAABA's
per-frame features are all invariant to translation/rotation of the world frame, so
to reproduce JAABADetect we only need the correct *scale*: we rebuild a pixel frame
by multiplying the mm coordinates by `pxpermm` and set the trajectory's `pxpermm`
accordingly. (Raw APT social `dist`/`ddist` features live in pixels, so pxpermm=1
would get them wrong by a factor of pxpermm -- hence the rescale.)

Column conventions (from ScriptReorganizeRetrack20250101.m / ScriptJAABAClassifify):
  pose:     x_mm, y_mm, cos_ori, sin_ori, maj_ax_mm, min_ax_mm
  a_mm = maj_ax_mm / 4, b_mm = min_ax_mm / 4   (JAABA quarter-axis convention)
  theta_mm = atan2(sin_ori, cos_ori)
  keypoints: <landmark>_x_mm / <landmark>_y_mm, mapped to APT landmark index by name.
"""
from __future__ import annotations

import os

import numpy as np

# Work both as a package submodule (jaaba_detect.mabe_adapter, e.g. from a notebook)
# and as a standalone import (with jaaba_detect on sys.path).
try:
    from . import jab_io, trx_io
    from .detect import jaaba_detect_traj
except ImportError:
    import jab_io, trx_io
    from detect import jaaba_detect_traj

# APT landmark order (1..21), from landmark_names in the MABe reorg scripts. The
# per-frame APT features index keypoints in THIS order.
APT_LANDMARK_NAMES = [
    "ant_head", "right_eye", "left_eye", "left_thorax", "right_thorax",
    "pos_notum", "pos_abdomen", "right_mid_fe", "right_mid_fetib", "left_mid_fe",
    "left_mid_fetib", "right_front_tar", "right_mid_tar", "right_back_tar",
    "left_back_tar", "left_mid_tar", "left_front_tar", "right_mid_wing",
    "right_outer_wing", "left_mid_wing", "left_outer_wing",
]

# FlyBubble rig constant. The mm-registered X arrays do not carry pxpermm; this is
# the near-constant value for the social-FlyBubble experiments. For exact
# reproduction of JAABADetect on one experiment, pass that experiment's pxpermm.
DEFAULT_PXPERMM = 18.9
DEFAULT_FPS = 150.0


def _norm_names(Xnames):
    if isinstance(Xnames, (list, tuple)):
        return [str(x).strip() for x in Xnames]
    return [str(x).strip() for x in np.asarray(Xnames).ravel()]


def _column_index(Xnames):
    """Return (pose_cols dict, kp_cols list of (xcol,ycol) in APT order)."""
    names = _norm_names(Xnames)
    col = {n: i for i, n in enumerate(names)}
    need = ["x_mm", "y_mm", "cos_ori", "sin_ori", "maj_ax_mm", "min_ax_mm"]
    missing = [n for n in need if n not in col]
    if missing:
        raise KeyError(f"X is missing pose columns {missing}")
    pose = {n: col[n] for n in need}
    kp = []
    for lm in APT_LANDMARK_NAMES:
        cx, cy = f"{lm}_x_mm", f"{lm}_y_mm"
        if cx in col and cy in col:
            kp.append((col[cx], col[cy]))
        else:
            kp.append(None)          # keypoint absent (non-APT classifier still fine)
    return pose, kp


def _fly_from_slice(pose_mm, kpts_mm, firstframe, pxpermm, fps):
    """Build a trx_io.Fly for one contiguous tracklet.

    pose_mm: dict with x_mm,y_mm,theta_mm,a_mm,b_mm arrays (length n, mm).
    kpts_mm: (n, npts, 2) mm keypoints or None.
    A pixel frame is reconstructed by scaling mm by pxpermm (pure scale; the world
    origin/rotation are irrelevant to every JAABA per-frame feature).
    """
    n = pose_mm["x_mm"].size
    P = float(pxpermm)
    x_mm = pose_mm["x_mm"]; y_mm = pose_mm["y_mm"]
    theta_mm = pose_mm["theta_mm"]; a_mm = pose_mm["a_mm"]; b_mm = pose_mm["b_mm"]
    kpts_px = None if kpts_mm is None else kpts_mm * P
    return trx_io.Fly(
        firstframe=int(firstframe), endframe=int(firstframe + n - 1), nframes=int(n),
        off=int(1 - firstframe), fps=float(fps), pxpermm=P,
        dt=np.full(max(n - 1, 0), 1.0 / fps),
        x=x_mm * P, y=y_mm * P, theta=theta_mm, a=a_mm * P, b=b_mm * P,
        x_mm=x_mm, y_mm=y_mm, theta_mm=theta_mm, a_mm=a_mm, b_mm=b_mm,
        sex="", kpts=kpts_px,
    )


def _tracklet_runs(valid, ids, col):
    """Yield (start_idx, stop_idx) half-open runs of one tracklet in a fly column.

    A run breaks on an invalid (all-NaN) frame or, if `ids` is given, on any change
    of id -- so packed columns that hold several tracklets split correctly.
    """
    n = valid.size
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i + 1
        while j < n and valid[j] and (ids is None or ids[j, col] == ids[i, col]):
            j += 1
        yield i, j
        i = j


def trajectories_from_X(X, Xnames, *, pxpermm=DEFAULT_PXPERMM, fps=DEFAULT_FPS,
                        ids=None, first_frame=1, min_frames=1):
    """Build a trx_io.Trajectories from one video's MABe X array.

    X: (nframes, nflies, nfeat) mm-registered. NaN marks frames a fly is absent.
    Xnames: the nfeat feature names (columns).
    ids: optional (nframes, nflies) tracklet ids used to split packed columns into
         separate flies; if None, each column's contiguous non-NaN run is one fly.
    first_frame: absolute frame number of X[0] (1-indexed; JAABA convention).
    Returns a Trajectories with one Fly per tracklet.
    """
    X = np.asarray(X, float)
    if X.ndim != 3:
        raise ValueError(f"X must be (nframes, nflies, nfeat); got {X.shape}")
    nframes, nflies, _ = X.shape
    pose, kp = _column_index(Xnames)
    have_kp = all(c is not None for c in kp)

    flies = []
    for fcol in range(nflies):
        sl = X[:, fcol, :]                              # (nframes, nfeat)
        valid = np.isfinite(sl[:, pose["x_mm"]])
        if not valid.any():
            continue
        for i, j in _tracklet_runs(valid, ids, fcol):
            if j - i < min_frames:
                continue
            seg = sl[i:j]
            cos_o = seg[:, pose["cos_ori"]]; sin_o = seg[:, pose["sin_ori"]]
            pose_mm = dict(
                x_mm=seg[:, pose["x_mm"]].copy(),
                y_mm=seg[:, pose["y_mm"]].copy(),
                theta_mm=np.arctan2(sin_o, cos_o),
                a_mm=seg[:, pose["maj_ax_mm"]] / 4.0,
                b_mm=seg[:, pose["min_ax_mm"]] / 4.0,
            )
            kpts_mm = None
            if have_kp:
                kpts_mm = np.stack(
                    [np.stack([seg[:, cx], seg[:, cy]], axis=1) for (cx, cy) in kp],
                    axis=1)                             # (n, npts, 2)
            flies.append(_fly_from_slice(pose_mm, kpts_mm, first_frame + i, pxpermm, fps))
    return trx_io.Trajectories(flies)


def jaaba_detect_from_X(X, Xnames, classifier, *, pxpermm=DEFAULT_PXPERMM,
                        fps=DEFAULT_FPS, ids=None, first_frame=1, roi=None,
                        verbose=True):
    """Apply a JAABA classifier to one video's MABe X array.

    classifier: path to an exported .classifier.mat OR a loaded jab_io.Classifier.
    Returns the same dict as detect.jaaba_detect (per-fly scores/postprocessed/...).
    """
    traj = trajectories_from_X(X, Xnames, pxpermm=pxpermm, fps=fps, ids=ids,
                               first_frame=first_frame)
    return jaaba_detect_traj(traj, classifier, roi=roi, verbose=verbose)


# --------------------------------------------------------------------------
# flyllm / APF simulated tracks (keypoints only)
# --------------------------------------------------------------------------
# flyllm / APF simulated tracks (keypoints only)
# --------------------------------------------------------------------------
# A flyllm `track` is (nagents, nframes, nkpts, 2) mm keypoints. Its keypoint order
# is flyllm.config.keypointnames, which we read at runtime (authoritative) -- we do
# NOT assume an order; the map to APT landmarks is done BY NAME. flyllm has 19
# keypoints: no outer-wing landmarks and no FlyTracker ellipse, both of which the
# JAABA classifiers use, so this path:
#   - reuses flyllm's mid-wing points for the outer-wing APT landmarks (approx), and
#   - reconstructs the ellipse pose from the keypoints (see _ellipse_from_kpts):
#     center = head/tail midpoint, theta = head->tail axis, a_mm/b_mm from body
#     extent (validated on NorpA to ~0.05mm / ~0.01rad / ~0.007mm vs FlyTracker).
#
# APT landmark name -> the flyllm keypoint name that supplies it. The right/left
# outer wings are approximated by the corresponding mid wing.
_APT_TO_FLYLLM_NAME = {
    "ant_head": "antennae_midpoint", "right_eye": "right_eye", "left_eye": "left_eye",
    "left_thorax": "left_front_thorax", "right_thorax": "right_front_thorax",
    "pos_notum": "base_thorax", "pos_abdomen": "tip_abdomen",
    "right_mid_fe": "right_middle_femur_base",
    "right_mid_fetib": "right_middle_femur_tibia_joint",
    "left_mid_fe": "left_middle_femur_base",
    "left_mid_fetib": "left_middle_femur_tibia_joint",
    "right_front_tar": "right_front_leg_tip", "right_mid_tar": "right_middle_leg_tip",
    "right_back_tar": "right_back_leg_tip", "left_back_tar": "left_back_leg_tip",
    "left_mid_tar": "left_middle_leg_tip", "left_front_tar": "left_front_leg_tip",
    "right_mid_wing": "wing_right", "right_outer_wing": "wing_right",
    "left_mid_wing": "wing_left", "left_outer_wing": "wing_left",
}
# Inverse (mid wings only): flyllm keypoint name -> its APT (mid) landmark name.
# Handy for building a flyllm-order track from APT/X data.
FLYLLM_TO_APT_NAME = {fl: apt for apt, fl in _APT_TO_FLYLLM_NAME.items()
                      if not apt.endswith("outer_wing")}
# Fallback flyllm keypoint order (== flyllm.config.keypointnames), used only when
# flyllm is not importable. The mapping is by name, so this order matters solely as
# the assumed layout of a track passed without explicit kpt_names.
FLYLLM_KEYPOINT_NAMES = [
    "wing_left", "wing_right", "antennae_midpoint", "right_eye", "left_eye",
    "left_front_thorax", "right_front_thorax", "base_thorax", "tip_abdomen",
    "right_middle_femur_base", "right_middle_femur_tibia_joint",
    "left_middle_femur_base", "left_middle_femur_tibia_joint",
    "right_front_leg_tip", "right_middle_leg_tip", "right_back_leg_tip",
    "left_back_leg_tip", "left_middle_leg_tip", "left_front_leg_tip",
]


def flyllm_keypoint_names():
    """The flyllm track's keypoint order. Returns flyllm.config.keypointnames when
    flyllm is importable (authoritative), else the built-in FLYLLM_KEYPOINT_NAMES."""
    try:
        from flyllm.config import keypointnames
        return list(keypointnames)
    except Exception:
        return list(FLYLLM_KEYPOINT_NAMES)
# body-extent -> ellipse-axis linear fits (a_mm from head-tail distance, b_mm from
# thorax width), quarter-axis convention (JAABA). Coefficients live in
# ellipse_from_keypoints.json (also destined for APF configs); loaded at import
# with these literals as a fallback. Fit by least-squares over 12 flybubble
# experiments (~31k fly-frames, 6 genotype families / rigs A-D / eras 2019-2023),
# validated on 12 disjoint held-out experiments (a_mm ~0.008mm, b_mm ~0.009mm --
# ~1% of a_mm, ~3.5% of b_mm). See test/fit_ellipse_coeffs.py.
ELLIPSE_COEFF_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "ellipse_from_keypoints.json")
_A_FIT_DEFAULT = (0.2386, 0.0556)     # a_mm = m*head_tail_dist + c
_B_FIT_DEFAULT = (0.3445, 0.0008)     # b_mm = m*thorax_width  + c


def load_ellipse_coeffs(path=ELLIPSE_COEFF_JSON):
    """Return (a_fit, b_fit) = ((slope,intercept),(slope,intercept)) from the JSON,
    falling back to the built-in defaults if it is missing/unreadable."""
    try:
        import json
        with open(path) as fh:
            e = json.load(fh)["ellipse_from_keypoints"]
        return ((e["a_mm"]["slope"], e["a_mm"]["intercept"]),
                (e["b_mm"]["slope"], e["b_mm"]["intercept"]))
    except (OSError, KeyError, ValueError):
        return _A_FIT_DEFAULT, _B_FIT_DEFAULT


_A_FIT, _B_FIT = load_ellipse_coeffs()


def track_to_apt(track, kpt_names=None):
    """Reorder a keypoint track to APT-21 landmark order, matching keypoints BY NAME.

    track: (nagents, nframes, nkpts, 2) mm. kpt_names: the nkpts keypoint names; if
    None, the flyllm order (flyllm.config.keypointnames) is assumed. Names may be APT
    landmark names or flyllm keypoint names. Returns (nagents, nframes, 21, 2); the
    two outer-wing landmarks reuse the mid-wing points.
    """
    track = np.asarray(track, float)
    names = flyllm_keypoint_names() if kpt_names is None else list(kpt_names)
    if track.shape[2] != len(names):
        raise ValueError(f"track has {track.shape[2]} keypoints but {len(names)} names")
    pos = {n: i for i, n in enumerate(names)}
    idx = []
    for apt_name in APT_LANDMARK_NAMES:
        src = apt_name if apt_name in pos else _APT_TO_FLYLLM_NAME.get(apt_name)
        if src is None or src not in pos:
            raise KeyError(f"no keypoint for APT landmark {apt_name!r}; names={names}")
        idx.append(pos[src])
    return track[:, :, idx, :]


def _ellipse_from_kpts(apt_kpts, a_fit=_A_FIT, b_fit=_B_FIT):
    """Derive (x_mm,y_mm,theta_mm,a_mm,b_mm) from APT-21 mm keypoints (n,21,2)."""
    head = apt_kpts[:, 0, :]          # ant_head (APT 1)
    tail = apt_kpts[:, 6, :]          # pos_abdomen (APT 7)
    lth = apt_kpts[:, 3, :]           # left_thorax (APT 4)
    rth = apt_kpts[:, 4, :]           # right_thorax (APT 5)
    ctr = 0.5 * (head + tail)
    d = head - tail
    theta = np.arctan2(d[:, 1], d[:, 0])
    htd = np.hypot(d[:, 0], d[:, 1])
    thw = np.hypot(lth[:, 0] - rth[:, 0], lth[:, 1] - rth[:, 1])
    a_mm = a_fit[0] * htd + a_fit[1]
    b_mm = b_fit[0] * thw + b_fit[1]
    return dict(x_mm=ctr[:, 0], y_mm=ctr[:, 1], theta_mm=theta, a_mm=a_mm, b_mm=b_mm)


def trajectories_from_track(track, *, kpt_names=None, pxpermm=DEFAULT_PXPERMM,
                            fps=DEFAULT_FPS, first_frame=1, min_frames=1,
                            a_fit=_A_FIT, b_fit=_B_FIT):
    """Build a Trajectories from a flyllm/APF keypoint track.

    track: (nagents, nframes, nkpts, 2) mm keypoints. Missing frames are NaN. The
    ellipse pose is reconstructed from the keypoints (see module notes). Each
    agent's contiguous non-NaN run becomes one fly.
    """
    apt = track_to_apt(track, kpt_names)               # (nagents,nframes,21,2)
    nagents, nframes = apt.shape[:2]
    flies = []
    for ag in range(nagents):
        k = apt[ag]                                    # (nframes,21,2)
        valid = np.isfinite(k[:, 0, 0]) & np.isfinite(k[:, 6, 0])
        if not valid.any():
            continue
        for i, j in _tracklet_runs(valid, None, 0):
            if j - i < min_frames:
                continue
            seg = k[i:j]                               # (n,21,2)
            pose_mm = _ellipse_from_kpts(seg, a_fit, b_fit)
            flies.append(_fly_from_slice(pose_mm, seg, first_frame + i, pxpermm, fps))
    return trx_io.Trajectories(flies)


def jaaba_detect_from_track(track, classifier, *, kpt_names=None,
                            pxpermm=DEFAULT_PXPERMM, fps=DEFAULT_FPS,
                            first_frame=1, roi=None, verbose=True,
                            a_fit=_A_FIT, b_fit=_B_FIT):
    """Apply a JAABA classifier to a flyllm/APF simulated (or GT) keypoint track.

    NOTE: flyllm has no outer-wing landmarks and no FlyTracker ellipse, so this
    reconstructs both from the keypoints (approximate). For exact reproduction use
    jaaba_detect_from_X on the registered X arrays.
    """
    traj = trajectories_from_track(track, kpt_names=kpt_names, pxpermm=pxpermm,
                                   fps=fps, first_frame=first_frame,
                                   a_fit=a_fit, b_fit=b_fit)
    return jaaba_detect_traj(traj, classifier, roi=roi, verbose=verbose)


def load_X_video(matfile, videoidx):
    """Load one video's slice from an X<split>_v3.mat.

    Returns (Xvid (nframes,nflies,nfeat), Xnames, ids_vid (nframes,nflies),
    frames_vid). Rows are ordered by within-video frame number.
    """
    try:
        from . import matio
    except ImportError:
        import sys
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import matio
    d, _ = matio.loadmat(matfile)
    X = np.asarray(d["X"], float)
    Xnames = _norm_names(d["Xnames"])
    vid = np.asarray(d["videoidx"]).ravel().astype(int)
    fr = np.asarray(d["frames"]).ravel().astype(int)
    ids = np.asarray(d["ids"], float) if "ids" in d else None
    m = vid == int(videoidx)
    order = np.argsort(fr[m], kind="stable")
    Xv = X[m][order]
    idv = ids[m][order] if ids is not None else None
    frv = fr[m][order]
    return Xv, Xnames, idv, frv
