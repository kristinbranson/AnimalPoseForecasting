"""APT keypoint per-frame features (compute_apt.m / compute_apt_social.m).

Feature name grammar (after stripping the leading 'apt_'):
    view{v}_{family}_{comp}_{idx...}
families: body, global (single-fly); pair, triad (single-fly, multi-keypoint);
          social, socialpair (inter-fly). Trailing numbers are 1-indexed keypoints;
for pair/socialpair the LAST number is p2; for triad the order is p1_p2_p3 (vertex p2).

Coordinate frames (must match exactly):
  - body, pair: keypoints rotated into the fly's body frame via convert_to_body
                (alpha = theta - pi/2; the fly heads along +b_y).
  - global, triad: raw lab-frame pixel coordinates.
  - social/socialpair: lab-frame; partner is the nearest landmark of the nearest
    other fly (per frame). social/socialpair 'dist' is in PIXELS (no /pxpermm) and
    'ddist' is a plain diff (no /dt) -- deliberate JAABA quirks, preserved here.

Each feature returns one vector per fly at its natural length (nframes or nframes-1,
matching compute_apt.m); detect.py aligns these to the absolute frame grid.
"""
from __future__ import annotations

import numpy as np


def modrange(x, lo=-np.pi, hi=np.pi):
    return np.mod(x - lo, hi - lo) + lo


def convert_to_body(x, y, trx_x, trx_y, theta):
    """Rotate lab-frame (x,y) into the fly's body frame (convert_to_body in compute_apt.m)."""
    a = theta - np.pi / 2
    dx = x - trx_x
    dy = y - trx_y
    bx = dx * np.cos(a) + dy * np.sin(a)
    by = -dx * np.sin(a) + dy * np.cos(a)
    return bx, by


# ---- elementary comps (compute_apt.m subfunctions), operating on 1-D arrays ----
def _velmag(x, y, dt, pxpermm):
    if x.size == 1:
        return np.array([0.0]) / pxpermm
    dx = np.diff(x)
    dy = np.diff(y)
    return (np.sqrt(dx * dx + dy * dy) / dt) / pxpermm


def _dist_center(x, y, x2, y2, pxpermm):
    return np.sqrt((x - x2) ** 2 + (y - y2) ** 2) / pxpermm


def _ddist_center(x, y, x2, y2, dt, pxpermm):
    if x.size > 1:
        dist = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
        return (np.diff(dist) / dt) / pxpermm
    return np.array([0.0]) / pxpermm


def _relative(x1, y1, x2, y2, dt, comp, pxpermm):
    if comp == "x":
        return (x1 - x2) / pxpermm
    if comp == "y":
        return (y1 - y2) / pxpermm
    if comp == "dx":
        return (np.diff(x1 - x2) / dt) / pxpermm
    if comp == "dy":
        return (np.diff(y1 - y2) / dt) / pxpermm
    if comp == "cos":
        d = x1 - x2
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        out = np.divide(d, length, out=np.zeros_like(d), where=length != 0)
        return out
    if comp == "sin":
        d = y1 - y2
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        out = np.divide(d, length, out=np.zeros_like(d), where=length != 0)
        return out
    if comp == "dtheta":
        if x1.size > 2:
            th = np.arctan2(y1 - y2, x1 - x2)
            return modrange(np.diff(th)) / dt
        return np.array([])
    raise ValueError(f"relative comp {comp!r}")


def _polyarea(xs, ys):
    return 0.5 * np.abs(
        sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i]
            for i in range(len(xs))))


def _pair_fn(x1, y1, x2, y2, dt, comp, pxpermm):
    if comp in ("x", "y", "dx", "dy", "sin", "cos", "dtheta"):
        return _relative(x1, y1, x2, y2, dt, comp, pxpermm)
    if comp == "velmag":
        return _velmag(x1 - x2, y1 - y2, dt, pxpermm)
    if comp == "dist":
        return _dist_center(x1, y1, x2, y2, pxpermm)
    if comp == "ddist":
        return _ddist_center(x1, y1, x2, y2, dt, pxpermm)
    if comp == "areaswept":
        n = x1.size
        area = np.zeros(n)
        for i in range(n - 1):
            area[i] = _polyarea([x1[i], x1[i + 1], x2[i + 1], x2[i]],
                                [y1[i], y1[i + 1], y2[i + 1], y2[i]])
        return area / pxpermm / pxpermm
    raise ValueError(f"pair comp {comp!r}")


def _triad_fn(x1, x2, x3, y1, y2, y3, dt, comp, pxpermm):
    theta1 = np.arctan2(y1 - y2, x1 - x2)
    theta2 = np.arctan2(y3 - y2, x3 - x2)
    theta = modrange(theta1 - theta2)
    if comp == "cos":
        return np.cos(theta)
    if comp == "sin":
        return np.sin(theta)
    if comp == "dangle":
        return np.diff(theta) / dt
    if comp in ("area", "darea"):
        n = x1.size
        area = np.array([_polyarea([x1[i], x2[i], x3[i]], [y1[i], y2[i], y3[i]])
                         for i in range(n)])
        area = area / pxpermm / pxpermm
        return area if comp == "area" else np.diff(area)
    if comp == "dlen":
        len1 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        len2 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        return (len1 - len2) / pxpermm
    raise ValueError(f"triad comp {comp!r}")


# ------------------------------------------------------------------ dispatch
def _kp(fly, part):
    """Raw lab-frame (x,y) of 1-indexed keypoint `part` for a fly, shape (nframes,)."""
    return fly.kpts[:, part - 1, 0].astype(float), fly.kpts[:, part - 1, 1].astype(float)


def _parse(fn):
    assert fn.startswith("apt_")
    parts = fn[len("apt_"):].split("_")
    view = int(parts[0].replace("view", ""))
    family = parts[1]
    comp = parts[2]
    idx = [int(p) for p in parts[3:]]
    return view, family, comp, idx


class AptFeatures:
    """Compute apt_view* per-frame features for all flies of a Trajectories object."""

    def __init__(self, traj, roi=None):
        self.traj = traj
        # roi[i] groups flies; default: all flies in one ROI (the FlyBubble arena)
        self.roi = np.zeros(traj.nflies, int) if roi is None else np.asarray(roi)

    # -- single-fly families -------------------------------------------------
    def _single_fly(self, fly, family, comp, idx):
        x = fly.x.astype(float)
        y = fly.y.astype(float)
        theta = fly.theta.astype(float)
        dt = fly.dt.astype(float)
        pxpermm = fly.pxpermm

        if family == "global":
            px, py = _kp(fly, idx[-1])
            if comp == "velmag":
                return _velmag(px, py, dt, pxpermm)
            raise NotImplementedError(f"global comp {comp!r}")

        if family == "body":
            px, py = _kp(fly, idx[-1])
            bx, by = convert_to_body(px, py, x, y, theta)
            if comp in ("x", "y", "dx", "dy", "cos", "sin", "dtheta"):
                return _relative(bx, by, np.zeros_like(bx), np.zeros_like(by), dt, comp, pxpermm)
            if comp == "distcenter":
                return _dist_center(bx, by, np.zeros_like(bx), np.zeros_like(by), pxpermm)
            if comp == "ddistcenter":
                return _ddist_center(bx, by, np.zeros_like(bx), np.zeros_like(by), dt, pxpermm)
            if comp == "velmag":
                return _velmag(bx, by, dt, pxpermm)
            raise NotImplementedError(f"body comp {comp!r}")

        if family == "pair":
            p1, p2 = idx[-2], idx[-1]
            x1, y1 = _kp(fly, p1)
            x2, y2 = _kp(fly, p2)
            bx1, by1 = convert_to_body(x1, y1, x, y, theta)
            bx2, by2 = convert_to_body(x2, y2, x, y, theta)
            return _pair_fn(bx1, by1, bx2, by2, dt, comp, pxpermm)

        if family == "triad":
            p1, p2, p3 = idx[-3], idx[-2], idx[-1]
            x1, y1 = _kp(fly, p1)
            x2, y2 = _kp(fly, p2)
            x3, y3 = _kp(fly, p3)
            return _triad_fn(x1, x2, x3, y1, y2, y3, dt, comp, pxpermm)

        raise NotImplementedError(f"family {family!r}")

    # -- inter-fly families --------------------------------------------------
    def _dapt_pair(self, fly1, fly2, pt1, pt2):
        """dapt_pair: (dist, dxy) 1..nframes1 for fly1's pt1 vs fly2's pt2 (list)."""
        f1, f2 = self.traj[fly1], self.traj[fly2]
        n1 = f1.nframes
        dist = np.full(n1, np.nan)
        dxy = np.full((2, n1), np.nan)
        t0 = max(f1.firstframe, f2.firstframe)
        t1 = min(f1.endframe, f2.endframe)
        if t1 < t0:
            return dist, dxy
        i0 = t0 + f1.off - 1  # 0-indexed
        i1 = t1 + f1.off - 1
        j0 = t0 + f2.off - 1
        j1 = t1 + f2.off - 1
        pt2 = np.atleast_1d(pt2)
        # a1: fly1 pt1 over overlap -> (2, nov); a2: fly2 pt2(list) over overlap -> (npt2, 2, nov)
        a1 = np.stack([f1.kpts[i0:i1 + 1, pt1 - 1, 0], f1.kpts[i0:i1 + 1, pt1 - 1, 1]], axis=0)
        a2 = np.stack([f2.kpts[j0:j1 + 1, pt2 - 1, 0], f2.kpts[j0:j1 + 1, pt2 - 1, 1]], axis=-1)
        # a2 shape (nov, npt2, 2) -> (npt2, 2, nov)
        a2 = np.transpose(a2, (1, 2, 0))
        dd = a2 - a1[None, :, :]                # (npt2, 2, nov)
        d = np.sqrt(np.sum(dd ** 2, axis=1))    # (npt2, nov)
        ix = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=0)  # (nov,)
        nov = i1 - i0 + 1
        dmin = d[ix, np.arange(nov)]
        dist[i0:i1 + 1] = dmin
        dxy[:, i0:i1 + 1] = dd[ix, :, np.arange(nov)].T
        return dist, dxy

    def _distclosest(self, fly1, pt1, pt2):
        """Min over same-ROI other flies of the (nearest-landmark) distance/vector."""
        f1 = self.traj[fly1]
        n1 = f1.nframes
        cand = [j for j in range(self.traj.nflies)
                if self.roi[j] == self.roi[fly1] and j != fly1]
        if not cand:
            return np.full(n1, np.nan), np.full((2, n1), np.nan)
        dclose = np.full((len(cand), n1), np.nan)
        dxyc = np.full((len(cand), 2, n1), np.nan)
        for k, fly2 in enumerate(cand):
            dcur, dxycur = self._dapt_pair(fly1, fly2, pt1, pt2)
            dclose[k] = dcur
            dxyc[k] = dxycur
        allnan = np.all(np.isnan(dclose), axis=0)
        closei = np.nanargmin(np.where(np.isnan(dclose), np.inf, dclose), axis=0)
        dmin = dclose[closei, np.arange(n1)]
        dmin[allnan] = np.nan
        dxy = dxyc[closei, :, np.arange(n1)].T  # (2, n1)
        dxy[:, allnan] = np.nan
        return dmin, dxy

    def _inter_fly(self, fly1, family, comp, idx):
        pt1 = idx[0]
        if family == "socialpair":
            pt2 = [idx[1]]
        else:
            pt2 = list(range(1, (self.traj[fly1].npts) + 1))
        dist, dxy = self._distclosest(fly1, pt1, pt2)
        if comp == "dist":
            return dist                       # pixels, NOT /pxpermm (JAABA quirk)
        if comp == "ddist":
            return np.diff(dist)              # plain diff, NO /dt (JAABA quirk)
        if comp in ("sin", "cos"):
            theta = self.traj[fly1].theta.astype(float)
            xyangle = np.arctan2(dxy[1, :], dxy[0, :])
            dangle = xyangle - theta
            return np.sin(dangle) if comp == "sin" else np.cos(dangle)
        raise NotImplementedError(f"social comp {comp!r}")

    # -- public --------------------------------------------------------------
    def compute(self, fn):
        """Return a list of per-fly vectors (natural length) for feature name `fn`."""
        _, family, comp, idx = _parse(fn)
        out = []
        for i in range(self.traj.nflies):
            if family in ("social", "socialpair"):
                out.append(self._inter_fly(i, family, comp, idx))
            else:
                out.append(self._single_fly(self.traj[i], family, comp, idx))
        return out
