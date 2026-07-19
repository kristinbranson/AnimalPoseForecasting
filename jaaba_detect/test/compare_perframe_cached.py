"""Compare Python per-frame features against JAABADetect's cached perframe/*.mat
on the real 21-landmark experiment, for every feature the classifier uses."""
import sys, os
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.dirname(PKG))
import jab_io, trx_io, matio
from perframe_apt import AptFeatures
from perframe_ellipse import EllipseFeatures


def main(expdir, clf_mat, perframedir):
    clf = jab_io.load_classifier(clf_mat)
    trk = os.path.join(expdir, "apt.trk")
    if not os.path.exists(trk):
        trk = os.path.join(expdir, "apttrk.mat")
    traj = trx_io.load_experiment(os.path.join(expdir, "registered_trx.mat"), trk,
                                  check_frame_consistency=False)
    apt = AptFeatures(traj)
    ell = EllipseFeatures(traj, fov=clf.apt["fov"], max_dnose2ell_anglerange=clf.apt["max_dnose2ell_anglerange"])

    rows = []
    for pff in sorted(clf.pff_names()):
        f = os.path.join(perframedir, pff + ".mat")
        if not os.path.exists(f):
            rows.append((pff, "MISSING", 0)); continue
        data, _ = matio.loadmat(f)
        gt = data["data"]
        while isinstance(gt, dict):
            gt = gt.get("data", gt)
        py = apt.compute(pff) if pff.startswith("apt_") else ell.compute(pff)
        worst = 0.0; nbad = 0
        for i in range(traj.nflies):
            g = np.asarray(gt[i], float).ravel()
            p = np.asarray(py[i], float).ravel()
            n = min(g.size, p.size)
            g, p = g[:n], p[:n]
            both = ~np.isnan(g) & ~np.isnan(p)
            diff = g[both] - p[both]
            if pff == "angleonclosestfly":
                diff = (diff + np.pi) % (2 * np.pi) - np.pi   # circular
            d = np.abs(diff)
            if d.size:
                worst = max(worst, float(d.max())); nbad += int((d > 1e-4).sum())
        rows.append((pff, f"{worst:.3e}", nbad))

    rows.sort(key=lambda r: -(float(r[1]) if r[1] not in ("MISSING",) else 1e9))
    print(f"{'feature':40s} {'worst|d|':>12s} {'frames>1e-4':>12s}")
    for name, w, nbad in rows:
        flag = "  <-- DIFFERS" if (w == "MISSING" or (w != "MISSING" and float(w) > 1e-4)) else ""
        print(f"{name:40s} {w:>12s} {nbad:>12d}{flag}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
