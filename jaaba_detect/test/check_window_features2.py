"""Validate the ADDED window stats (change/harmonic/diff_neighbor_*/zscore) vs MATLAB."""
import sys, os
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import window_features as wf

STATS = ["change", "harmonic", "diff_neighbor_mean", "diff_neighbor_min",
         "diff_neighbor_max", "zscore_neighbors"]


def _s(v):
    a = np.asarray(v).ravel()
    return str(a[0]) if a.size else ""


def main(gtfile):
    m = sio.loadmat(gtfile, squeeze_me=True, struct_as_record=False)
    xkeys = sorted(k for k in m if k.startswith("x") and not k.startswith("__"))
    worst = 0.0; worst_desc = None; nchk = 0; nfail = 0
    for xk in xkeys:
        xs = m[xk]
        x = np.asarray(xs.x, float).ravel()
        relb = wf.relative_bins(x)
        for st in STATS:
            blk = getattr(xs, st)
            Y = np.atleast_2d(np.asarray(blk.Y, float))
            trans = [_s(t) for t in np.atleast_1d(blk.trans)]
            radius = np.atleast_1d(np.asarray(blk.radius, float)).astype(int)
            offset = np.atleast_1d(np.asarray(blk.offset, float)).astype(int)
            cwr = np.atleast_1d(np.asarray(blk.cwr, float))
            nh = np.atleast_1d(np.asarray(blk.nh, float))
            for k in range(Y.shape[0]):
                extra = {}
                if np.isfinite(cwr[k]):
                    extra["change_window_radius"] = int(cwr[k])
                if np.isfinite(nh[k]):
                    extra["num_harmonic"] = int(nh[k])
                py = wf.window_feature(x, st, trans[k], int(radius[k]), int(offset[k]),
                                       relbins=relb, extra=extra)
                gt = Y[k]
                ng, npn = np.isnan(gt), np.isnan(py)
                if not np.array_equal(ng, npn):
                    print(f"  NaN-pattern [{xk}/{st}/{trans[k]}/r{radius[k]}/o{offset[k]}] "
                          f"differs at {int((ng!=npn).sum())}")
                    nfail += 1
                both = ~ng & ~npn & np.isfinite(gt) & np.isfinite(py)
                if both.any():
                    mx = float(np.abs(gt[both] - py[both]).max())
                    if mx > worst:
                        worst, worst_desc = mx, f"{xk}/{st}/{trans[k]}/r{radius[k]}/o{offset[k]}"
                    if mx > 1e-6:
                        print(f"  MISMATCH [{xk}/{st}/{trans[k]}/r{radius[k]}/o{offset[k]}"
                              f"{'/cwr'+str(int(cwr[k])) if np.isfinite(cwr[k]) else ''}"
                              f"{'/nh'+str(int(nh[k])) if np.isfinite(nh[k]) else ''}] max|d|={mx:.3e}")
                        nfail += 1
                nchk += 1
    print(f"\nchecked {nchk} window features (added stats)")
    print(f"worst abs diff: {worst:.3e}  ({worst_desc})")
    print("PASS" if worst < 1e-6 and nfail == 0 else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1])
