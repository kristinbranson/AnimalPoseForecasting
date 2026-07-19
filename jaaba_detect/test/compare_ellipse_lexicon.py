"""Validate the FULL trx/ellipse per-frame lexicon vs MATLAB's cached values.

Usage: compare_ellipse_lexicon.py <trxfile> <perframedir>
Compares every trx-based per-frame feature the user listed, computed by
perframe_ellipse.EllipseFeatures, against JAABADetect's cached perframe/*.mat.
"""
import sys, os
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.dirname(PKG))
import trx_io, matio
from perframe_ellipse import EllipseFeatures

FEATURES = """a_mm absanglefrom1to2_nose2ell absdtheta absdv_cor absphidiff_anglesub
absphidiff_nose2ell absthetadiff_anglesub absthetadiff_nose2ell anglefrom1to2_anglesub
anglefrom1to2_nose2ell angleonclosestfly anglesub area b_mm closestfly_anglesub
closestfly_center closestfly_ell2nose closestfly_nose2ell corfrac_maj corfrac_min
danglesub darea da db dcenter ddcenter decc dell2nose dnose2ell_angle_30tomin30
dnose2ell_angle_min20to20 dnose2ell_angle_min30to30 dnose2ell dnose2tail dphi dtheta
du_cor du_ctr du_tail dv_cor dv_ctr dv_tail ecc flipdv_cor magveldiff_anglesub
magveldiff_nose2ell nflies_close phi phisideways velmag velmag_ctr velmag_nose
velmag_tail veltoward_anglesub veltoward_nose2ell yaw absyaw""".split()

# raw-angle features with the +-2pi ellipse-seam degeneracy -> compare circularly
CIRCULAR = {"angleonclosestfly"}
# angle-RATE features (modrange(diff(computed_angle))/dt): +-2pi/dt wrap ambiguity at
# a direction reversal (phi is atan2-based, so 1-ULP differences flip the wrap)
RATE_CIRCULAR = {"dphi"}


def main(trxfile, perframedir):
    traj = trx_io.load_experiment(trxfile, None)
    ell = EllipseFeatures(traj, fov=np.pi, max_dnose2ell_anglerange=127.0, nbodylengths_near=2.5)
    worst = 0.0; worst_where = None; nfail = 0; nmiss = 0
    rows = []
    for fn in FEATURES:
        f = os.path.join(perframedir, fn + ".mat")
        if not os.path.exists(f):
            rows.append((fn, "MISSING", 0)); nmiss += 1; continue
        gt = matio.loadmat(f)[0]["data"]
        try:
            py = ell.compute(fn)
        except Exception as e:
            rows.append((fn, f"ERROR:{e}", 0)); nfail += 1; continue
        w = 0.0; nb = 0
        for i in range(traj.nflies):
            g = np.asarray(gt[i], float).ravel()
            p = np.asarray(py[i], float).ravel()
            n = min(g.size, p.size)
            if g.size != p.size:
                nb += abs(g.size - p.size)
            g, p = g[:n], p[:n]
            both = ~np.isnan(g) & ~np.isnan(p)
            diff = g[both] - p[both]
            if fn in CIRCULAR:
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
            elif fn in RATE_CIRCULAR:
                dt = np.asarray(traj[i].dt, float).ravel()[:n][both]
                diff = (diff * dt + np.pi) % (2 * np.pi) - np.pi   # compare (rate*dt) mod 2pi
            d = np.abs(diff)
            if d.size:
                w = max(w, float(d.max())); nb += int((d > 1e-4).sum())
        rows.append((fn, f"{w:.3e}", nb))
        if w > worst:
            worst, worst_where = w, fn
        if w > 1e-4 or nb > 0:
            nfail += 1
    rows.sort(key=lambda r: -(float(r[1]) if r[1][0].isdigit() else 1e18))
    print(f"{'feature':32s} {'worst|d|':>12s} {'bad':>6s}")
    for name, w, nb in rows:
        flag = "  <--" if (not w[0].isdigit()) or float(w) > 1e-4 or nb > 0 else ""
        print(f"{name:32s} {w:>12s} {nb:>6d}{flag}")
    print(f"\n{len(FEATURES)} features; worst {worst:.3e} ({worst_where}); missing {nmiss}; failing {nfail}")
    print("PASS" if nfail == 0 and nmiss == 0 else "REVIEW")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
