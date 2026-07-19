"""Compare the nose2ell/anglesub ellipse features against ellipse_ref2.m."""
import sys, os
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trx_io
import perframe_ellipse


def _cellvec(v, i):
    a = np.asarray(v, dtype=object).ravel()
    return np.asarray(a[i], dtype=float).ravel()


def main(gtfile, trxfile):
    m = sio.loadmat(gtfile, squeeze_me=True, struct_as_record=False)
    names = [str(s) for s in np.atleast_1d(m["feat_names"]).ravel()]
    traj = trx_io.load_experiment(trxfile, None)
    ell = perframe_ellipse.EllipseFeatures(traj, fov=np.pi, max_dnose2ell_anglerange=127.0)
    worst = 0.0; worst_where = None; nfail = 0
    for fi, fn in enumerate(names):
        gt = m[f"f{fi + 1}"]
        py = ell.compute(fn)
        for i in range(traj.nflies):
            g = _cellvec(gt, i)
            p = np.asarray(py[i], dtype=float).ravel()
            if g.size != p.size:
                print(f"  LEN {fn} fly{i}: {g.size} vs {p.size}"); nfail += 1; continue
            ng, npn = np.isnan(g), np.isnan(p)
            if not np.array_equal(ng, npn):
                nd = int(np.sum(ng != npn))
                print(f"  NAN {fn} fly{i}: {nd} frames differ"); nfail += 1
            both = ~ng & ~npn
            if both.any():
                diff = g[both] - p[both]
                # angleonclosestfly is a raw ellipse-parameter angle with an inherent
                # +-2pi seam degeneracy in JAABA itself; compare circularly.
                if fn == "angleonclosestfly":
                    diff = (diff + np.pi) % (2 * np.pi) - np.pi
                d = np.abs(diff)
                mx = float(d.max()); nbad = int((d > 1e-6).sum())
                if mx > worst: worst, worst_where = mx, f"{fn} fly{i}"
                if mx > 1e-6:
                    print(f"  MISMATCH {fn} fly{i}: max|d|={mx:.3e} ({nbad}/{both.sum()} frames)")
                    nfail += 1
    print(f"\n{len(names)} features x {traj.nflies} flies")
    print(f"worst abs diff: {worst:.3e}  ({worst_where})")
    print("PASS" if worst < 1e-6 and nfail == 0 else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
