"""Validate prctile and hist window stats vs MATLAB (gen_window_gt3.m output)."""
import sys, os
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import window_features as wf


def _s(v):
    a = np.asarray(v).ravel()
    return str(a[0]) if a.size else ""


def main(gtfile):
    m = sio.loadmat(gtfile, squeeze_me=True, struct_as_record=False)
    xkeys = sorted(k for k in m if k.startswith("x") and not k.startswith("__"))
    worst = 0.0; worst_d = None; nchk = 0; nfail = 0
    for xk in xkeys:
        xs = m[xk]
        x = np.asarray(xs.x, float).ravel()
        relb = wf.relative_bins(x)
        for st in ("prctile", "hist"):
            blk = getattr(xs, st)
            Y = np.atleast_2d(np.asarray(blk.Y, float))
            trans = [_s(t) for t in np.atleast_1d(blk.trans)]
            radius = np.atleast_1d(np.asarray(blk.radius, float)).astype(int)
            offset = np.atleast_1d(np.asarray(blk.offset, float)).astype(int)
            prc = np.atleast_1d(np.asarray(blk.prc, float))
            he = np.atleast_2d(np.asarray(blk.he, float))
            if he.shape[0] != 2:
                he = he.T
            full_edges = np.asarray(getattr(xs, "hist_edges_full", []), float).ravel() if st == "hist" else None
            nrow = min(Y.shape[0], len(trans), radius.size, offset.size)
            for k in range(nrow):
                extra = {}
                if np.isfinite(prc[k]):
                    extra["prctile"] = float(prc[k])
                if st == "hist" and np.isfinite(he[0, k]):
                    extra["hist_edges"] = (he[0, k], he[1, k])
                    if full_edges is not None and full_edges.size:
                        # 1-based bin index = position of this bin's lower edge in the full array
                        bi = int(np.argmin(np.abs(full_edges - he[0, k]))) + 1
                        extra["hist_edges_full"] = tuple(full_edges.tolist())
                        extra["hist_bin"] = bi
                py = wf.window_feature(x, st, trans[k], int(radius[k]), int(offset[k]),
                                       relbins=relb, extra=extra)
                gt = Y[k]
                both = ~np.isnan(gt) & ~np.isnan(py)
                if not np.array_equal(np.isnan(gt), np.isnan(py)):
                    print(f"  NaN [{xk}/{st}/{trans[k]}/r{radius[k]}/o{offset[k]}]"); nfail += 1
                if both.any():
                    mx = float(np.abs(gt[both] - py[both]).max())
                    if mx > worst:
                        worst, worst_d = mx, f"{xk}/{st}/{trans[k]}/r{radius[k]}/o{offset[k]}"
                    if mx > 1e-6:
                        print(f"  MISMATCH [{xk}/{st}/{trans[k]}/r{radius[k]}/o{offset[k]}] {mx:.3e}"); nfail += 1
                nchk += 1
    print(f"\nchecked {nchk} prctile/hist features; worst {worst:.3e} ({worst_d})")
    print("PASS" if worst < 1e-6 and nfail == 0 else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1])
