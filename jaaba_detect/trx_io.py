"""Read tracking data for JAABA inference: the registered ellipse trx and the
APT keypoint trk.

- Ellipse pose (x,y,theta,a,b and _mm variants, dt, pxpermm, frame bookkeeping)
  comes from the jab's trxfilename (e.g. registered_trx.mat), a MATLAB v7.3 file.
- Keypoints come from the APT trk (e.g. apttrk.mat / apt.trk): per target an
  [npts, 2, nframes] array. Keypoints are verified to share the registered trx's
  pixel frame (their centroid tracks the ellipse centroid).

JAABA frame convention: frames are 1-indexed; a fly's data for absolute frame t
lives at array index t + off, off = 1 - firstframe. nflies, per-fly frame ranges,
npts, pxpermm all come from the data -- nothing is hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np


# ---------------------------------------------------------------- h5py helpers
def _ref_vec(f, ds, i):
    """Dereference struct-array field `ds` (shape (nflies,1) of refs) element i -> 1-D array."""
    return np.array(f[ds[i, 0]]).ravel()


def _ref_scalar(f, ds, i):
    return float(np.array(f[ds[i, 0]]).ravel()[0])


@dataclass
class Fly:
    firstframe: int
    endframe: int
    nframes: int
    off: int              # 1 - firstframe (add to a 1-indexed frame to get array index)
    fps: float
    pxpermm: float
    dt: np.ndarray        # (nframes-1,)
    # ellipse pose, pixels
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    a: np.ndarray
    b: np.ndarray
    # ellipse pose, mm
    x_mm: np.ndarray
    y_mm: np.ndarray
    theta_mm: np.ndarray
    a_mm: np.ndarray
    b_mm: np.ndarray
    sex: str = ""
    kpts: np.ndarray = None   # (nframes, npts, 2), pixels, aligned to this fly's frames

    @property
    def npts(self):
        return None if self.kpts is None else self.kpts.shape[1]


class Trajectories:
    def __init__(self, flies: list[Fly]):
        self.flies = flies

    @property
    def nflies(self):
        return len(self.flies)

    def __getitem__(self, i) -> Fly:
        return self.flies[i]

    # ---- overlapping absolute-frame range for a pair of flies
    def common_frames(self, i, j):
        t0 = max(self.flies[i].firstframe, self.flies[j].firstframe)
        t1 = min(self.flies[i].endframe, self.flies[j].endframe)
        return t0, t1


ELLIPSE_FIELDS = ["x", "y", "theta", "a", "b",
                  "x_mm", "y_mm", "theta_mm", "a_mm", "b_mm"]


def read_trx(trxfile: str) -> Trajectories:
    """Read a JAABA (registered) trx .mat. Handles both v7.3 (HDF5) and older
    (v5/v7, scipy) formats. Returns Trajectories without kpts."""
    try:
        return _read_trx_hdf5(trxfile)
    except OSError:
        return _read_trx_scipy(trxfile)


def _read_trx_hdf5(trxfile: str) -> Trajectories:
    f = h5py.File(trxfile, "r")
    trx = f["trx"]
    nflies = trx["x"].shape[0]
    flies = []
    for i in range(nflies):
        vals = {fld: _ref_vec(f, trx[fld], i) for fld in ELLIPSE_FIELDS}
        firstframe = int(round(_ref_scalar(f, trx["firstframe"], i)))
        endframe = int(round(_ref_scalar(f, trx["endframe"], i)))
        nframes = int(round(_ref_scalar(f, trx["nframes"], i)))
        off = int(round(_ref_scalar(f, trx["off"], i))) if "off" in trx else (1 - firstframe)
        fps = _ref_scalar(f, trx["fps"], i) if "fps" in trx else np.nan
        pxpermm = _ref_scalar(f, trx["pxpermm"], i) if "pxpermm" in trx else np.nan
        dt = _ref_vec(f, trx["dt"], i) if "dt" in trx else np.full(nframes - 1, 1.0 / fps)
        flies.append(Fly(
            firstframe=firstframe, endframe=endframe, nframes=nframes, off=off,
            fps=fps, pxpermm=pxpermm, dt=dt, sex="", **vals,
        ))
    f.close()
    return Trajectories(flies)


def _read_trx_scipy(trxfile: str) -> Trajectories:
    import scipy.io as sio
    m = sio.loadmat(trxfile, squeeze_me=False, struct_as_record=False)
    trx = m["trx"].ravel()          # (nflies,) of mat_struct
    flies = []
    for t in trx:
        def v(fld):
            return np.asarray(getattr(t, fld)).ravel().astype(float)

        def s(fld, default=np.nan):
            return float(np.asarray(getattr(t, fld)).ravel()[0]) if hasattr(t, fld) else default

        firstframe = int(round(s("firstframe")))
        nframes = int(round(s("nframes")))
        fps = s("fps")
        off = int(round(s("off"))) if hasattr(t, "off") else (1 - firstframe)
        dt = v("dt") if hasattr(t, "dt") else np.full(nframes - 1, 1.0 / fps)
        flies.append(Fly(
            firstframe=firstframe, endframe=int(round(s("endframe"))), nframes=nframes,
            off=off, fps=fps, pxpermm=s("pxpermm"), dt=dt, sex="",
            **{fld: v(fld) for fld in ELLIPSE_FIELDS},
        ))
    return Trajectories(flies)


def read_apt_kpts(trkfile: str):
    """Read APT keypoints from a trk .mat (v7.3). Returns (list_of_kpts, startframes, endframes).

    Each list element is (nframes_i, npts, 2) for target i. Handles the
    dense-per-target cell form (pTrk = {ntargets} cell of [npts,2,nframes]).
    """
    f = h5py.File(trkfile, "r")
    startframes = np.array(f["startframes"]).ravel().astype(int)
    endframes = np.array(f["endframes"]).ravel().astype(int)
    pTrk = f["pTrk"]
    kpts_list = []
    if isinstance(pTrk, h5py.Dataset) and pTrk.dtype == object:
        refs = np.array(pTrk).ravel()
        for r in refs:
            a = np.array(f[r])          # h5py reads MATLAB [npts,2,nframes] as (nframes,2,npts)
            # -> (nframes, npts, 2)
            a = np.transpose(a, (0, 2, 1))
            kpts_list.append(a)
    else:
        # dense array form [npts,2,nframes,ntargets] -> h5py (ntargets,nframes,2,npts)
        a = np.array(pTrk)
        a = np.transpose(a, (0, 1, 3, 2))   # (ntargets,nframes,npts,2)
        kpts_list = [a[t] for t in range(a.shape[0])]
    f.close()
    return kpts_list, startframes, endframes


def load_experiment(trxfile: str, trkfile: str | None = None,
                    check_frame_consistency: bool = True) -> Trajectories:
    """Load ellipse trx (+ APT keypoints if given), attaching kpts to each fly
    aligned to that fly's absolute frame range."""
    traj = read_trx(trxfile)
    if trkfile is None:
        return traj
    kpts_list, sf, ef = read_apt_kpts(trkfile)
    assert len(kpts_list) == traj.nflies, \
        f"trk targets ({len(kpts_list)}) != trx flies ({traj.nflies})"
    for i, fly in enumerate(traj.flies):
        kp = kpts_list[i]                       # (nframes_trk, npts, 2), starts at sf[i]
        # align to the trx fly's [firstframe, endframe]
        lo = fly.firstframe - int(sf[i])
        hi = lo + fly.nframes
        if lo < 0 or hi > kp.shape[0]:
            # frame ranges disagree; clip/pad with nan to the trx length
            aligned = np.full((fly.nframes, kp.shape[1], 2), np.nan)
            src0 = max(0, lo)
            dst0 = max(0, -lo)
            n = min(kp.shape[0] - src0, fly.nframes - dst0)
            if n > 0:
                aligned[dst0:dst0 + n] = kp[src0:src0 + n]
            fly.kpts = aligned
        else:
            fly.kpts = kp[lo:hi]
    if check_frame_consistency:
        _check_kpt_frame(traj)
    return traj


def _check_kpt_frame(traj: Trajectories, tol_px: float = 30.0):
    """Sanity check: keypoint centroid should track the ellipse centroid (same frame)."""
    fly = traj.flies[0]
    if fly.kpts is None:
        return
    good = np.isfinite(fly.kpts).all(axis=(1, 2))
    if not good.any():
        return
    t = np.argmax(good)
    cx = np.nanmean(fly.kpts[t, :, 0])
    cy = np.nanmean(fly.kpts[t, :, 1])
    d = np.hypot(cx - fly.x[t], cy - fly.y[t])
    if d > tol_px:
        import warnings
        warnings.warn(
            f"keypoint centroid ({cx:.1f},{cy:.1f}) far from ellipse centroid "
            f"({fly.x[t]:.1f},{fly.y[t]:.1f}); d={d:.1f}px -- frames may not match")


if __name__ == "__main__":
    import sys
    trxfile = sys.argv[1]
    trkfile = sys.argv[2] if len(sys.argv) > 2 else None
    traj = load_experiment(trxfile, trkfile)
    print(f"nflies={traj.nflies}")
    for i, fly in enumerate(traj.flies[:3]):
        print(f" fly{i}: frames {fly.firstframe}..{fly.endframe} n={fly.nframes} "
              f"off={fly.off} pxpermm={fly.pxpermm:.3f} npts={fly.npts}")
        print(f"        x[:3]={fly.x[:3]}  theta[:3]={fly.theta[:3]}")
        if fly.kpts is not None:
            print(f"        kpts shape={fly.kpts.shape}  kp0 centroid="
                  f"({np.nanmean(fly.kpts[0,:,0]):.1f},{np.nanmean(fly.kpts[0,:,1]):.1f})")
