"""Ellipse (registered-trx) per-frame features (compute_*.m in JAABA/perframe).

Pose is x_mm,y_mm,theta_mm,a_mm,b_mm (a_mm,b_mm are QUARTER axes: nose =
x_mm + 2*a_mm*cos(theta_mm)). "Closest fly" features evaluate a per-frame quantity
against the nearest other fly under a metric (center / nose2ell / nose2tail /
anglesub), aligned across flies via off (=1-firstframe).

Implemented: b_mm, ecc, dv_tail, dcenter, dnose2tail (single/simple), plus the
nose2ell + anglesub + angle-range families. perframe_params (fov,
max_dnose2ell_anglerange) come from the classifier, nothing hard-coded.
"""
from __future__ import annotations

import numpy as np


def modrange(x, lo=-np.pi, hi=np.pi):
    return np.mod(x - lo, hi - lo) + lo


# ============================ ellipse geometry ============================
def ellipsedist_hack(xc, yc, a, b, theta, u, v, npoints=20):
    """Distance from points (u,v) to ellipse (xc,yc,a,b,theta), and the closest
    ellipse-parameter angle. All inputs are (n,) arrays; returns (d, thetai)."""
    ang = np.linspace(0, 2 * np.pi, npoints)          # (npoints,)
    ex = a[:, None] * np.cos(ang)[None, :]
    ey = b[:, None] * np.sin(ang)[None, :]
    X = np.cos(theta)[:, None] * ex - np.sin(theta)[:, None] * ey + xc[:, None]
    Y = np.sin(theta)[:, None] * ex + np.cos(theta)[:, None] * ey + yc[:, None]
    # match dist2.m exactly: |P|^2 + |q|^2 - 2 P.q, clamped at 0 (same FP as MATLAB argmin)
    P2 = X ** 2 + Y ** 2                               # (n, npoints)
    q2 = u ** 2 + v ** 2                               # (n,)
    d2 = P2 + q2[:, None] - 2 * (X * u[:, None] + Y * v[:, None])
    d2 = np.maximum(d2, 0.0)
    i = np.argmin(d2, axis=1)
    n = len(u)
    d = np.sqrt(d2[np.arange(n), i])
    return d, ang[i]


def anglesubtended(x1, y1, a1, b1, theta1, x2, y2, a2, b2, theta2, fov):
    """Visual angle fly2 subtends in fly1's view (anglesubtended.m), vectorized over frames.
    a1,b1,a2,b2 are FULL axes (callers pass 2*a_mm etc.). Returns (n,)."""
    EPS = 1e-5
    n = np.asarray(x1).shape[0]
    out = np.zeros(n)

    # eye of fly1 in fly2's frame
    c1 = x1 + a1 * np.cos(theta1)
    d1 = y1 + a1 * np.sin(theta1)
    c1 = c1 - x2
    d1 = d1 - y2
    c = c1 * np.cos(theta2) + d1 * np.sin(theta2)
    d = d1 * np.cos(theta2) - c1 * np.sin(theta2)

    # checkinborder
    A = c ** 2 / a2 ** 2 + d ** 2 / b2 ** 2
    isonborder = np.abs(A - 1) < EPS
    isinborder = (~isonborder) & (A < 1)
    tiny = np.maximum(a2, b2) < EPS

    # tangent points (computetangentpoints), for the "outside" frames
    Aq = b2 ** 2 * c ** 2 + a2 ** 2 * d ** 2
    Bq = -2 * a2 * b2 ** 2 * c
    Cq = a2 ** 2 * (b2 ** 2 - d ** 2)
    D = Bq ** 2 - 4 * Aq * Cq
    D = np.sqrt(np.maximum(D, 0))

    # 6 candidate phi and their cost
    with np.errstate(divide="ignore", invalid="ignore"):
        cosphi_p = np.clip((-Bq + D) / (2 * Aq), -1, 1)
        cosphi_m = np.clip((-Bq - D) / (2 * Aq), -1, 1)
    phi = np.stack([
        np.zeros(n), np.full(n, np.pi),
        np.arccos(cosphi_p), -np.arccos(cosphi_p),
        np.arccos(cosphi_m), -np.arccos(cosphi_m),
    ], axis=1)                                          # (n,6)
    cost = np.empty((n, 6))
    cost[:, 0] = np.where(np.abs(c - a2) < EPS, 0.0, np.inf)
    cost[:, 1] = np.where(np.abs(c + a2) < EPS, 0.0, np.inf)
    for k in (2, 3, 4, 5):
        sp = np.sin(phi[:, k]); cp = np.cos(phi[:, k])
        cost[:, k] = np.abs((b2 * sp - d) * (-a2 * sp) - (a2 * cp - c) * (b2 * cp))
    order = np.argsort(cost, axis=1, kind="stable")
    phi1 = np.take_along_axis(phi, order[:, 0:1], 1)[:, 0]
    phi2 = np.take_along_axis(phi, order[:, 1:2], 1)[:, 0]

    xa = a2 * np.cos(phi1); ya = b2 * np.sin(phi1)
    xb = a2 * np.cos(phi2); yb = b2 * np.sin(phi2)
    psi1 = np.arctan2(ya - d, xa - c)
    psi2 = np.arctan2(yb - d, xb - c)
    psi0 = np.arctan2(-d, -c)
    dth = theta2 - theta1
    psi0 = psi0 + dth; psi1 = psi1 + dth; psi2 = psi2 + dth
    psi1 = np.mod(psi1 + np.pi, 2 * np.pi) - np.pi
    psi0 = psi1 + np.mod(psi0 - psi1, 2 * np.pi)
    psi2 = psi1 + np.mod(psi2 - psi1, 2 * np.pi)
    swap = psi2 < psi0
    if np.any(swap):
        t = psi1[swap].copy()
        psi1[swap] = psi2[swap]
        psi2[swap] = t
        psi1[swap] = np.mod(psi1[swap] + np.pi, 2 * np.pi) - np.pi
        psi2[swap] = psi1[swap] + np.mod(psi2[swap] - psi1[swap], 2 * np.pi)

    dpsi = _limitbyfov(psi1, psi2, fov)

    out = np.where(tiny, EPS, dpsi)
    out = np.where(isonborder, min(np.pi, fov), out)
    out = np.where(isinborder, fov, out)
    return out


def _limitbyfov(psi1, psi2, fov):
    fov1 = -fov / 2
    fov2 = fov1 + fov
    psi1 = fov1 + np.mod(psi1 - fov1, 2 * np.pi)
    psi2 = fov1 + np.mod(psi2 - fov1, 2 * np.pi)
    dpsi = np.empty_like(psi1)
    c1 = (fov2 <= psi1) & (psi1 <= psi2)
    c2 = (fov2 <= psi2) & (psi2 <= psi1)
    c3 = (psi1 <= fov2) & (fov2 <= psi2)
    c4 = (psi1 <= psi2) & (psi2 <= fov2)
    c5 = (psi2 <= fov2) & (fov2 <= psi1)
    dpsi[:] = (psi2 - fov1) + (fov2 - psi1)           # else branch
    dpsi = np.where(c5, psi2 - fov1, dpsi)
    dpsi = np.where(c4, psi2 - psi1, dpsi)
    dpsi = np.where(c3, fov2 - psi1, dpsi)
    dpsi = np.where(c2, np.mod(fov1 - psi1, 2 * np.pi) + np.mod(fov2 - psi2, 2 * np.pi), dpsi)
    dpsi = np.where(c1, 0.0, dpsi)
    return dpsi


# ============================ features ============================
class EllipseFeatures:
    def __init__(self, traj, roi=None, fov=np.pi, max_dnose2ell_anglerange=127.0,
                 nbodylengths_near=2.5):
        self.traj = traj
        self.roi = np.zeros(traj.nflies, int) if roi is None else np.asarray(roi)
        self.fov = fov
        self.max_dnose2ell_anglerange = max_dnose2ell_anglerange
        self.nbodylengths_near = nbodylengths_near

    def _same_roi(self, i):
        return [j for j in range(self.traj.nflies) if self.roi[j] == self.roi[i]]

    def _overlap(self, i, j):
        f1, f2 = self.traj[i], self.traj[j]
        t0 = max(f1.firstframe, f2.firstframe)
        t1 = min(f1.endframe, f2.endframe)
        if t1 < t0:
            return None
        return t0, t1, t0 + f1.off - 1, t1 + f1.off - 1, t0 + f2.off - 1, t1 + f2.off - 1

    # ---- pairwise distance primitives ----
    def _dcenter_pair(self, i, j):
        f1, f2 = self.traj[i], self.traj[j]
        d = np.full(f1.nframes, np.nan)
        ov = self._overlap(i, j)
        if ov is None:
            return d
        _, _, i0, i1, j0, j1 = ov
        dx = f2.x_mm[j0:j1 + 1] - f1.x_mm[i0:i1 + 1]
        dy = f2.y_mm[j0:j1 + 1] - f1.y_mm[i0:i1 + 1]
        d[i0:i1 + 1] = np.sqrt(dx * dx + dy * dy)
        return d

    def _dnose2tail_pair(self, i, j):
        f1, f2 = self.traj[i], self.traj[j]
        d = np.full(f1.nframes, np.nan)
        ov = self._overlap(i, j)
        if ov is None:
            return d
        _, _, i0, i1, j0, j1 = ov
        xnose = f1.x_mm[i0:i1 + 1] + 2 * f1.a_mm[i0:i1 + 1] * np.cos(f1.theta_mm[i0:i1 + 1])
        ynose = f1.y_mm[i0:i1 + 1] + 2 * f1.a_mm[i0:i1 + 1] * np.sin(f1.theta_mm[i0:i1 + 1])
        xtail = f2.x_mm[j0:j1 + 1] - 2 * f2.a_mm[j0:j1 + 1] * np.cos(f2.theta_mm[j0:j1 + 1])
        ytail = f2.y_mm[j0:j1 + 1] - 2 * f2.a_mm[j0:j1 + 1] * np.sin(f2.theta_mm[j0:j1 + 1])
        d[i0:i1 + 1] = np.sqrt((xtail - xnose) ** 2 + (ytail - ynose) ** 2)
        return d

    def _dnose2ell_pair(self, i, j, idx0=None):
        """dnose2ell_pair: nose(i) to ellipse(j) distance + angle. idx0 = 0-based fly1
        indices to evaluate (None = full overlap)."""
        f1, f2 = self.traj[i], self.traj[j]
        n1 = f1.nframes
        d = np.full(n1, np.nan)
        angle = np.full(n1, np.nan)
        ov = self._overlap(i, j)
        if ov is None:
            return d, angle
        t0, t1, i0, i1, j0, j1 = ov
        xnose = f1.x_mm + 2 * f1.a_mm * np.cos(f1.theta_mm)
        ynose = f1.y_mm + 2 * f1.a_mm * np.sin(f1.theta_mm)
        if idx0 is None:
            ii = np.arange(i0, i1 + 1)
        else:
            ii = np.asarray(idx0, int)
            ii = ii[(ii >= i0) & (ii <= i1)]
        if ii.size == 0:
            return d, angle
        jj = ii + (j0 - i0)                    # shift to fly2 indices (= off2-off1)
        dd, thetai = ellipsedist_hack(
            f2.x_mm[jj], f2.y_mm[jj], 2 * f2.a_mm[jj], 2 * f2.b_mm[jj], f2.theta_mm[jj],
            xnose[ii], ynose[ii], 20)
        d[ii] = dd
        angle[ii] = thetai
        return d, angle - np.pi

    def _dnose2center_pair(self, i, j):
        f1, f2 = self.traj[i], self.traj[j]
        n1 = f1.nframes
        d = np.full(n1, np.nan)
        anglefrom = np.full(n1, np.nan)
        ov = self._overlap(i, j)
        if ov is None:
            return d, anglefrom, None
        _, _, i0, i1, j0, j1 = ov
        xnose = f1.x_mm[i0:i1 + 1] + 2 * f1.a_mm[i0:i1 + 1] * np.cos(f1.theta_mm[i0:i1 + 1])
        ynose = f1.y_mm[i0:i1 + 1] + 2 * f1.a_mm[i0:i1 + 1] * np.sin(f1.theta_mm[i0:i1 + 1])
        dx = f2.x_mm[j0:j1 + 1] - xnose
        dy = f2.y_mm[j0:j1 + 1] - ynose
        d[i0:i1 + 1] = np.sqrt(dx * dx + dy * dy)
        anglefrom[i0:i1 + 1] = modrange(np.arctan2(dy, dx) - f1.theta_mm[i0:i1 + 1])
        return d, anglefrom, (i0, i1, j0, j1)

    def _anglesub_pair(self, i, j):
        f1, f2 = self.traj[i], self.traj[j]
        a = np.full(f1.nframes, np.nan)
        ov = self._overlap(i, j)
        if ov is None:
            return a
        _, _, i0, i1, j0, j1 = ov
        a[i0:i1 + 1] = anglesubtended(
            f1.x_mm[i0:i1 + 1], f1.y_mm[i0:i1 + 1], 2 * f1.a_mm[i0:i1 + 1],
            2 * f1.b_mm[i0:i1 + 1], f1.theta_mm[i0:i1 + 1],
            f2.x_mm[j0:j1 + 1], f2.y_mm[j0:j1 + 1], 2 * f2.a_mm[j0:j1 + 1],
            2 * f2.b_mm[j0:j1 + 1], f2.theta_mm[j0:j1 + 1], self.fov)
        return a

    # ---- closest-fly reducers (return mind/closest arrays over same-ROI others) ----
    def _closest_min(self, i, pairfn):
        f1 = self.traj[i]
        cand = [j for j in self._same_roi(i) if j != i]
        if not cand:
            return np.full(f1.nframes, np.nan), np.full(f1.nframes, -1)
        D = np.full((len(cand), f1.nframes), np.inf)
        for k, j in enumerate(cand):
            dj = pairfn(i, j)
            D[k] = np.where(np.isnan(dj), np.inf, dj)
        ck = np.argmin(D, axis=0)
        mind = D[ck, np.arange(f1.nframes)]
        allinf = np.all(~np.isfinite(D), axis=0)
        mind[allinf] = np.nan
        closest = np.array([cand[k] for k in ck]); closest[allinf] = -1
        return mind, closest

    def _closestfly_nose2ell(self, i):
        f1 = self.traj[i]
        cand = [j for j in self._same_roi(i) if j != i]
        n1 = f1.nframes
        mind = np.full(n1, np.inf)
        angle = np.full(n1, np.nan)
        closest = np.full(n1, -1)
        # bounds
        mindupper = np.full(n1, np.inf)
        dlower = {}
        for j in cand:
            dc, _, ov = self._dnose2center_pair(i, j)
            if ov is None:
                dlower[j] = np.full(n1, np.nan)
                continue
            i0, i1, j0, j1 = ov
            up = np.full(n1, np.inf)
            up[i0:i1 + 1] = dc[i0:i1 + 1] + 2 * self.traj[j].a_mm[j0:j1 + 1]
            mindupper = np.minimum(mindupper, up)
            lo = np.full(n1, np.nan)
            lo[i0:i1 + 1] = dc[i0:i1 + 1] - 2 * self.traj[j].a_mm[j0:j1 + 1]
            dlower[j] = lo
        for j in cand:
            istry = np.where(mindupper >= dlower[j])[0]  # 0-based fly1 indices
            dcurr, anglecurr = self._dnose2ell_pair(i, j, idx0=istry)
            idx = dcurr < mind
            mind[idx] = dcurr[idx]
            angle[idx] = anglecurr[idx]
            closest[idx] = j
        bad = ~np.isfinite(mind)
        mind[bad] = np.nan
        closest[bad] = -1
        return mind, angle, closest

    def _closestfly_anglesub(self, i):
        f1 = self.traj[i]
        cand = [j for j in self._same_roi(i) if j != i]
        n1 = f1.nframes
        if not cand:
            return np.full(n1, np.nan), np.full(n1, -1)
        A = np.full((len(cand), n1), np.nan)
        for k, j in enumerate(cand):
            A[k] = self._anglesub_pair(i, j)
        Amask = np.where(np.isnan(A), -np.inf, A)
        ck = np.argmax(Amask, axis=0)
        maxa = A[ck, np.arange(n1)]
        allnan = np.all(np.isnan(A), axis=0)
        maxa[allnan] = np.nan
        closest = np.array([cand[k] for k in ck]); closest[allnan] = -1
        return maxa, closest

    # =================== public compute ===================
    def compute(self, fn):
        traj = self.traj
        N = traj.nflies
        # ---- base trx fields ----
        if fn in ("a_mm", "b_mm"):
            return [getattr(traj[i], fn).astype(float).copy() for i in range(N)]
        if fn == "ecc":
            return [traj[i].b_mm / traj[i].a_mm for i in range(N)]
        if fn == "area":
            return [self._area(traj[i]) for i in range(N)]
        # ---- single-fly derivatives (diff/dt) ----
        DIFF = {"da": "a_mm", "db": "b_mm"}
        if fn in DIFF:
            return [self._ddt(getattr(traj[i], DIFF[fn]), traj[i].dt) for i in range(N)]
        if fn == "darea":
            return [self._ddt(self._area(traj[i]), traj[i].dt) for i in range(N)]
        if fn == "decc":
            return [self._ddt(traj[i].b_mm / traj[i].a_mm, traj[i].dt) for i in range(N)]
        if fn == "dtheta":
            return [self._dtheta(traj[i]) for i in range(N)]
        if fn == "absdtheta":
            return [np.abs(self._dtheta(traj[i])) for i in range(N)]
        if fn == "phi":
            return [self._phi(traj[i]) for i in range(N)]
        if fn == "dphi":
            return [self._ddt(self._phi(traj[i]), traj[i].dt, wrap=True) for i in range(N)]
        if fn == "yaw":
            return [modrange(self._phi(traj[i]) - traj[i].theta_mm) for i in range(N)]
        if fn == "absyaw":
            return [np.abs(modrange(self._phi(traj[i]) - traj[i].theta_mm)) for i in range(N)]
        if fn == "phisideways":
            return [np.abs(modrange(self._phi(traj[i]) - traj[i].theta_mm, -np.pi / 2, np.pi / 2))
                    for i in range(N)]
        # ---- point velocities ----
        if fn == "velmag_ctr":
            return [self._velmag_pts(traj[i].x_mm, traj[i].y_mm, traj[i].dt) for i in range(N)]
        if fn == "velmag_nose":
            return [self._velmag_pt(traj[i], +1) for i in range(N)]
        if fn == "velmag_tail":
            return [self._velmag_pt(traj[i], -1) for i in range(N)]
        if fn == "dv_tail":
            return [self._dv_tail(traj[i]) for i in range(N)]
        if fn == "dv_ctr":
            return [self._dv_ctr(traj[i]) for i in range(N)]
        if fn == "du_ctr":
            return [self._du_pts(traj[i].x_mm, traj[i].y_mm, traj[i]) for i in range(N)]
        if fn == "du_tail":
            return [self._du_tail(traj[i]) for i in range(N)]
        # ---- center-of-rotation family ----
        if fn == "corfrac_maj":
            return [self._center_of_rotation2(traj[i])[0] for i in range(N)]
        if fn == "corfrac_min":
            return [self._center_of_rotation2(traj[i])[1] for i in range(N)]
        if fn in ("velmag", "du_cor", "dv_cor", "absdv_cor", "flipdv_cor"):
            return [self._cor_vel(traj[i], fn) for i in range(N)]
        # ---- inter-animal ----
        if fn == "dcenter":
            return [self._closest_min(i, self._dcenter_pair)[0] for i in range(N)]
        if fn == "ddcenter":
            return [self._ddt(self._closest_min(i, self._dcenter_pair)[0], traj[i].dt) for i in range(N)]
        if fn == "dnose2tail":
            return [self._closest_min(i, self._dnose2tail_pair)[0] for i in range(N)]
        if fn == "dnose2ell":
            return [self._closestfly_nose2ell(i)[0] for i in range(N)]
        if fn == "dell2nose":
            return [self._closest_min(i, self._dell2nose_pair)[0] for i in range(N)]
        if fn == "anglesub":
            return [self._closestfly_anglesub(i)[0] for i in range(N)]
        if fn == "danglesub":
            return [self._ddt(self._closestfly_anglesub(i)[0], traj[i].dt) for i in range(N)]
        if fn == "angleonclosestfly":
            return [self._closestfly_nose2ell(i)[1] for i in range(N)]
        if fn.startswith("nflies_close"):
            rest = fn[len("nflies_close"):]
            nb = float(rest[1:]) if rest.startswith("_") else self.nbodylengths_near
            return [self._nflies_close(i, nb) for i in range(N)]
        if fn.startswith("closestfly_"):
            typ = fn[len("closestfly_"):]
            out = []
            for i in range(N):
                c = self._closest_by_type(i, typ).astype(float)
                out.append(np.where(c < 0, np.nan, c + 1.0))  # 1-based fly id, NaN if none
            return out
        # closest-fly features: <base>_<type>, type in {anglesub,nose2ell,center,nose2tail}
        for base in ("veltoward", "absanglefrom1to2", "anglefrom1to2",
                     "absthetadiff", "absphidiff", "magveldiff"):
            for typ in ("anglesub", "nose2ell", "center", "nose2tail"):
                if fn == f"{base}_{typ}":
                    return [self._closest_fly_feature(i, base, typ) for i in range(N)]
        if fn.startswith("dnose2ell_angle_"):
            lohi = fn[len("dnose2ell_angle_"):]
            a, b = lohi.split("to")
            lo = -float(a[3:]) if a.startswith("min") else float(a)
            hi = -float(b[3:]) if b.startswith("min") else float(b)
            rng = np.array([lo, hi]) * np.pi / 180.0
            return [self._dnose2ell_anglerange(i, rng) for i in range(traj.nflies)]
        raise NotImplementedError(f"ellipse feature {fn!r} not implemented")

    # ---- closest-fly-type dispatch ----
    def _closest_by_type(self, i, typ):
        if typ == "anglesub":
            return self._closestfly_anglesub(i)[1]
        if typ == "nose2ell":
            return self._closestfly_nose2ell(i)[2]
        if typ == "center":
            return self._closest_min(i, self._dcenter_pair)[1]
        if typ == "nose2tail":
            return self._closest_min(i, self._dnose2tail_pair)[1]
        if typ == "ell2nose":
            return self._closest_min(i, self._dell2nose_pair)[1]
        raise ValueError(f"closest-fly type {typ!r}")

    def _closest_fly_feature(self, i, base, typ):
        closest = self._closest_by_type(i, typ)
        if base == "veltoward":
            return self._veltoward(i, closest)
        if base == "magveldiff":
            return self._magveldiff(i, closest)
        if base == "anglefrom1to2":
            return self._anglefrom1to2(i, closest)
        if base == "absanglefrom1to2":
            return np.abs(self._anglefrom1to2(i, closest))
        if base == "absthetadiff":
            return self._absthetadiff(i, closest)
        if base == "absphidiff":
            return self._absphidiff(i, closest)
        raise NotImplementedError(f"closest-fly base {base!r} not implemented")

    # ---- individual feature bodies (follow compute_*.m) ----
    @staticmethod
    def _dv_tail(fly):
        if fly.nframes < 2:
            return np.array([])
        x_mm, y_mm = fly.x_mm.astype(float), fly.y_mm.astype(float)
        theta = fly.theta.astype(float)
        theta_mm = fly.theta_mm.astype(float)
        a_mm, dt = fly.a_mm.astype(float), fly.dt.astype(float)
        tailx = x_mm + 2 * np.cos(-theta) * a_mm
        taily = y_mm + 2 * np.sin(-theta) * a_mm
        dx = np.diff(tailx); dy = np.diff(taily)
        return dx * np.cos(theta_mm[:-1] + np.pi / 2) + (dy * np.sin(theta_mm[:-1] + np.pi / 2)) / dt

    @staticmethod
    def _dv_ctr(fly):
        if fly.nframes < 2:
            return np.array([])
        x_mm, y_mm = fly.x_mm.astype(float), fly.y_mm.astype(float)
        theta_mm, dt = fly.theta_mm.astype(float), fly.dt.astype(float)
        dx = np.diff(x_mm); dy = np.diff(y_mm)
        return (dx * np.cos(theta_mm[:-1] + np.pi / 2) + dy * np.sin(theta_mm[:-1] + np.pi / 2)) / dt

    # ---- more single-fly helpers ----
    @staticmethod
    def _area(fly):
        return (2 * fly.a_mm.astype(float)) * (2 * fly.b_mm.astype(float)) * np.pi

    @staticmethod
    def _ddt(v, dt, wrap=False):
        v = np.asarray(v, float)
        if v.size < 2:
            return np.array([])
        d = np.diff(v)
        if wrap:
            d = modrange(d)
        return d / dt

    def _dtheta(self, fly):
        if fly.nframes <= 1:
            return np.array([])
        return modrange(np.diff(fly.theta_mm.astype(float))) / fly.dt

    @staticmethod
    def _velmag_pts(x, y, dt):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if x.size < 2:
            return np.array([])
        return np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2) / dt

    def _velmag_pt(self, fly, sgn):
        # sgn=+1 nose (y+2sin*a), sgn=-1 tail (y-2sin*a); x uses cos(theta) either way
        th = fly.theta.astype(float); a = fly.a_mm.astype(float)  # NOTE: .theta not theta_mm
        px = fly.x_mm + 2 * np.cos(th) * a
        py = fly.y_mm + sgn * 2 * np.sin(th) * a
        return self._velmag_pts(px, py, fly.dt)

    def _du_pts(self, x, y, fly):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if x.size < 2:
            return np.array([])
        tm = fly.theta_mm[:-1].astype(float)
        return (np.diff(x) * np.cos(tm) + np.diff(y) * np.sin(tm)) / fly.dt

    def _du_tail(self, fly):
        th = fly.theta.astype(float); a = fly.a_mm.astype(float)
        return self._du_pts(fly.x_mm + 2 * np.cos(th) * a, fly.y_mm - 2 * np.sin(th) * a, fly)

    # ---- center of rotation ----
    def _center_of_rotation2(self, fly):
        f = fly
        if f.nframes < 2:
            return np.zeros((2, 0))
        cost = np.cos(f.theta_mm); sint = np.sin(f.theta_mm)
        a = f.a_mm.astype(float); b = f.b_mm.astype(float)
        dacost = 2 * np.diff(a * cost); dbcost = 2 * np.diff(b * cost)
        dasint = 2 * np.diff(a * sint); dbsint = 2 * np.diff(b * sint)
        Z = dacost * dbcost + dbsint * dasint
        with np.errstate(divide="ignore", invalid="ignore"):
            m11 = dbcost / Z; m21 = -dasint / Z; m13 = dbsint / Z; m22 = dacost / Z
        dx = np.diff(f.x_mm); dy = np.diff(f.y_mm)
        rfrac = np.vstack([-(m11 * dx + m13 * dy), -(m21 * dx + m22 * dy)])
        rfrac[np.isnan(rfrac)] = 0
        idx = np.where(np.sum(rfrac ** 2, axis=0) > 1)[0]
        if idx.size:
            psi = np.linspace(0, 2 * np.pi, 100)
            cp = np.cos(psi)[:, None]; sp = np.sin(psi)[:, None]

            def ept(ii):
                xx = f.x_mm[ii][None, :] + (2 * a[ii] * cost[ii])[None, :] * cp - (2 * b[ii] * sint[ii])[None, :] * sp
                yy = f.y_mm[ii][None, :] + (2 * a[ii] * sint[ii])[None, :] * cp + (2 * b[ii] * cost[ii])[None, :] * sp
                return xx, yy
            x1, y1 = ept(idx); x2, y2 = ept(idx + 1)
            j = np.argmin((x1 - x2) ** 2 + (y1 - y2) ** 2, axis=0)
            rfrac[0, idx] = np.cos(psi)[j]; rfrac[1, idx] = np.sin(psi)[j]
        return rfrac

    def _rfrac2center(self, fly, rfrac):
        f = fly; rm, rn = rfrac[0], rfrac[1]
        a = f.a_mm.astype(float); b = f.b_mm.astype(float)
        c = np.cos(f.theta_mm); s = np.sin(f.theta_mm)
        x1 = f.x_mm[:-1] + rm * a[:-1] * 2 * c[:-1] - rn * b[:-1] * 2 * s[:-1]
        y1 = f.y_mm[:-1] + rm * a[:-1] * 2 * s[:-1] + rn * b[:-1] * 2 * c[:-1]
        x2 = f.x_mm[1:] + rm * a[1:] * 2 * c[1:] - rn * b[1:] * 2 * s[1:]
        y2 = f.y_mm[1:] + rm * a[1:] * 2 * s[1:] + rn * b[1:] * 2 * c[1:]
        return x1, y1, x2, y2

    def _cor_vel(self, fly, fn):
        if fly.nframes < 2:
            return np.array([])
        x1, y1, x2, y2 = self._rfrac2center(fly, self._center_of_rotation2(fly))
        dxc = x2 - x1; dyc = y2 - y1
        dt = fly.dt.astype(float); tm = fly.theta_mm[:-1].astype(float)
        if fn == "velmag":
            vm = np.sqrt(dxc ** 2 + dyc ** 2) / dt
            bad = np.isnan(dxc)
            if bad.any():
                vm[bad] = self._velmag_pts(fly.x_mm, fly.y_mm, fly.dt)[bad]
            return vm
        if fn == "du_cor":
            return (dxc * np.cos(tm) + dyc * np.sin(tm)) / dt
        dv = (dxc * np.cos(tm + np.pi / 2) + dyc * np.sin(tm + np.pi / 2)) / dt
        if fn == "dv_cor":
            return dv
        if fn == "absdv_cor":
            return np.abs(dv)
        if fn == "flipdv_cor":
            return dv * np.sign(self._dtheta(fly))
        raise ValueError(fn)

    # ---- more inter-animal ----
    def _dell2nose_pair(self, i, j):
        f1, f2 = self.traj[i], self.traj[j]
        d = np.full(f1.nframes, np.nan)
        ov = self._overlap(i, j)
        if ov is None:
            return d
        t0, t1, i0, i1, j0, j1 = ov
        xnose = f2.x_mm + 2 * f2.a_mm * np.cos(f2.theta_mm)
        ynose = f2.y_mm + 2 * f2.a_mm * np.sin(f2.theta_mm)
        ii = np.arange(i0, i1 + 1); jj = ii + (j0 - i0)
        dd, _ = ellipsedist_hack(f1.x_mm[ii], f1.y_mm[ii], 2 * f1.a_mm[ii], 2 * f1.b_mm[ii],
                                 f1.theta_mm[ii], xnose[jj], ynose[jj], 20)
        d[ii] = dd
        return d

    def _isclose_pair(self, i, j, nb):
        f1, f2 = self.traj[i], self.traj[j]
        isc = np.zeros(f1.nframes, bool)
        ov = self._overlap(i, j)
        if ov is None:
            return isc
        t0, t1, i0, i1, j0, j1 = ov
        dx = f2.x_mm[j0:j1 + 1] - f1.x_mm[i0:i1 + 1]
        dy = f2.y_mm[j0:j1 + 1] - f1.y_mm[i0:i1 + 1]
        z = np.sqrt(dx * dx + dy * dy) / (4 * f1.a_mm[i0:i1 + 1])
        isc[i0:i1 + 1] = z <= nb
        return isc

    def _nflies_close(self, i, nb):
        f1 = self.traj[i]
        out = np.zeros(f1.nframes)
        for j in self._same_roi(i):
            if j == i:
                continue
            out += self._isclose_pair(i, j, nb).astype(float)
        return out

    def _magveldiff(self, i, closest):
        f1 = self.traj[i]
        out = np.zeros(f1.nframes)
        maxidx = -1
        dx1 = np.diff(f1.x_mm); dy1 = np.diff(f1.y_mm)
        for j in self._same_roi(i):
            if j == i:
                continue
            f2 = self.traj[j]
            idx = np.where(closest[:-1] == j)[0]
            off = f1.firstframe - f2.firstframe
            idx = idx[idx + off != f2.nframes - 1]
            if idx.size == 0:
                continue
            dx2 = f2.x_mm[off + idx + 1] - f2.x_mm[off + idx]
            dy2 = f2.y_mm[off + idx + 1] - f2.y_mm[off + idx]
            out[idx] = np.sqrt((dx1[idx] - dx2) ** 2 + (dy1[idx] - dy2) ** 2)
            maxidx = max(maxidx, int(idx.max()))
        return self._matlab_trim(out, maxidx)

    def _phi(self, fly):
        if fly.nframes < 2:
            return fly.theta_mm.astype(float).copy()
        y, x = fly.y_mm.astype(float), fly.x_mm.astype(float)
        dy1 = np.concatenate([[y[1] - y[0]], (y[2:] - y[:-2]) / 2, [y[-1] - y[-2]]])
        dx1 = np.concatenate([[x[1] - x[0]], (x[2:] - x[:-2]) / 2, [x[-1] - x[-2]]])
        out = np.arctan2(dy1, dx1)
        bad = (dy1 == 0) & (dx1 == 0)
        out[bad] = fly.theta_mm[bad]
        return out

    @staticmethod
    def _matlab_trim(out, maxidx):
        # MATLAB `data{i}(idx)=...` sizes the vector to the largest assigned index.
        return out[:maxidx + 1] if maxidx >= 0 else np.zeros(0)

    def _veltoward(self, i, closest):
        f1 = self.traj[i]
        out = np.zeros(f1.nframes)
        maxidx = -1
        dx1 = np.diff(f1.x_mm); dy1 = np.diff(f1.y_mm)
        for j in self._same_roi(i):
            if j == i:
                continue
            f2 = self.traj[j]
            idx = np.where(closest[:-1] == j)[0]
            off = f1.firstframe - f2.firstframe
            idx = idx[idx + off != f2.nframes - 1]      # drop last frame of fly2 (0-based)
            if idx.size == 0:
                continue
            dx2 = f2.x_mm[off + idx] - f1.x_mm[idx]
            dy2 = f2.y_mm[off + idx] - f1.y_mm[idx]
            dz = np.sqrt(dx2 ** 2 + dy2 ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                ux = np.where(dz == 0, 0.0, dx2 / dz)
                uy = np.where(dz == 0, 0.0, dy2 / dz)
            out[idx] = dx1[idx] * ux + dy1[idx] * uy
            maxidx = max(maxidx, int(idx.max()))
        return self._matlab_trim(out, maxidx)

    def _anglefrom1to2(self, i, closest):
        f1 = self.traj[i]
        out = np.zeros(f1.nframes)
        maxidx = -1
        xnose = f1.x_mm + 2 * f1.a_mm * np.cos(f1.theta_mm)
        ynose = f1.y_mm + 2 * f1.a_mm * np.sin(f1.theta_mm)
        for j in self._same_roi(i):
            if j == i:
                continue
            f2 = self.traj[j]
            idx = np.where(closest == j)[0]
            if idx.size == 0:
                continue
            off = f1.firstframe - f2.firstframe
            dx2 = f2.x_mm[off + idx] - xnose[idx]
            dy2 = f2.y_mm[off + idx] - ynose[idx]
            out[idx] = modrange(np.arctan2(dy2, dx2) - f1.theta_mm[idx])
            maxidx = max(maxidx, int(idx.max()))
        return self._matlab_trim(out, maxidx)

    def _absthetadiff(self, i, closest):
        f1 = self.traj[i]
        out = np.zeros(f1.nframes)
        maxidx = -1
        for j in self._same_roi(i):
            if j == i:
                continue
            f2 = self.traj[j]
            idx = np.where(closest == j)[0]
            if idx.size == 0:
                continue
            off = f1.firstframe - f2.firstframe
            out[idx] = np.abs(modrange(f2.theta_mm[off + idx] - f1.theta_mm[idx]))
            maxidx = max(maxidx, int(idx.max()))
        return self._matlab_trim(out, maxidx)

    def _absphidiff(self, i, closest):
        f1 = self.traj[i]
        phi1 = self._phi(f1)
        out = np.zeros(f1.nframes)
        maxidx = -1
        for j in self._same_roi(i):
            if j == i:
                continue
            f2 = self.traj[j]
            phi2f = self._phi(f2)
            idx = np.where(closest[:-1] == j)[0]
            off = f1.firstframe - f2.firstframe
            idx = idx[idx + off != f2.nframes - 1]
            if idx.size == 0:
                continue
            out[idx] = np.abs(modrange(phi2f[off + idx] - phi1[idx]))
            maxidx = max(maxidx, int(idx.max()))
        return self._matlab_trim(out, maxidx)

    def _dnose2ell_anglerange(self, i, anglerange):
        f1 = self.traj[i]
        n1 = f1.nframes
        MAXVALUE = self.max_dnose2ell_anglerange
        logmax = np.log(MAXVALUE)
        issmooth = not np.isinf(MAXVALUE)
        a1 = anglerange[0]
        a2 = modrange(anglerange[1], a1, a1 + 2 * np.pi)
        cand = [j for j in self._same_roi(i) if j != i]
        mindupper = np.full(n1, np.inf)
        dlower = {j: np.full(n1, np.nan) for j in cand}
        weights = {j: np.full(n1, np.nan) for j in cand}
        for j in cand:
            dc, af, ov = self._dnose2center_pair(i, j)
            if ov is None:
                continue
            i0, i1, j0, j1 = ov
            afr = modrange(af, a1, a1 + 2 * np.pi)
            aj = 2 * self.traj[j].a_mm
            if issmooth:
                u = np.zeros(n1)
                outrange = ~((afr >= a1) & (afr <= a2))
                u_val = np.minimum(np.abs(modrange(afr - a1)), np.abs(modrange(afr - a2))) / np.pi
                u = np.where(outrange, u_val, 0.0)
                w = np.exp(logmax * u)
                seg = slice(i0, i1 + 1)
                mindupper[seg] = np.minimum(
                    mindupper[seg], w[seg] * (dc[seg] + aj[j0:j1 + 1]))
                dlower[j][seg] = w[seg] * (dc[seg] - aj[j0:j1 + 1])
                weights[j][seg] = w[seg]
            else:
                inr = (afr >= a1) & (afr <= a2)
                idx = np.where(inr)[0]
                mindupper[idx] = np.minimum(mindupper[idx], dc[idx] + aj[idx - i0 + j0])
                dlower[j][idx] = dc[idx] - aj[idx - i0 + j0]
        D = np.full((max(len(cand), 1), n1), np.nan)
        for k, j in enumerate(cand):
            istry = np.where(dlower[j] <= mindupper)[0]
            dj = self._dnose2ell_anglerange_pair(i, j, anglerange, istry)
            D[k] = weights[j] * dj if issmooth else dj
        if cand:
            Dm = np.where(np.isnan(D), np.inf, D)
            mind = np.min(Dm, axis=0)
            mind[~np.isfinite(mind)] = np.nan
        else:
            mind = np.full(n1, np.nan)
        mind[np.isnan(mind)] = MAXVALUE
        return mind

    def _dnose2ell_anglerange_pair(self, i, j, anglerange, istry):
        f1, f2 = self.traj[i], self.traj[j]
        n1 = f1.nframes
        d = np.full(n1, np.nan)
        ov = self._overlap(i, j)
        if ov is None or np.size(istry) == 0:
            return d
        t0, t1, i0, i1, j0, j1 = ov
        xnose = f1.x_mm + 2 * f1.a_mm * np.cos(f1.theta_mm)
        ynose = f1.y_mm + 2 * f1.a_mm * np.sin(f1.theta_mm)
        ii = np.asarray(istry, int)
        ii = ii[(ii >= i0) & (ii <= i1)]
        if ii.size == 0:
            return d
        jj = ii + (j0 - i0)
        dd, _ = ellipsedist_hack(
            f2.x_mm[jj], f2.y_mm[jj], 2 * f2.a_mm[jj], 2 * f2.b_mm[jj], f2.theta_mm[jj],
            xnose[ii], ynose[ii], 20)
        d[ii] = dd
        return d
