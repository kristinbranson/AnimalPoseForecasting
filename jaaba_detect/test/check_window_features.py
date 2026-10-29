"""Compare window_features.py against JAABA ground truth (window_gt.mat)."""
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
    xkeys = [k for k in m if k.startswith("x") and not k.startswith("__")]
    worst = 0.0
    worst_desc = None
    n_checked = 0
    n_bin_mismatch = 0
    for xk in sorted(xkeys):
        xs = m[xk]
        x = np.asarray(xs.x, float).ravel()
        gt_bins = np.asarray(xs.relativeBins, float).ravel()
        py_bins = wf.relative_bins(x)
        if not np.allclose(gt_bins, py_bins, atol=1e-9, rtol=1e-9, equal_nan=True):
            n_bin_mismatch += 1
            print(f"  [{xk}] relative bins mismatch max|d|={np.nanmax(np.abs(gt_bins-py_bins)):.2e}")
        for st in ["mean", "min", "max", "std"]:
            blk = getattr(xs, st)
            Y = np.atleast_2d(np.asarray(blk.Y, float))          # (nfeat, N)
            trans = blk.trans
            trans = [trans] if isinstance(trans, str) else [_s(t) for t in np.atleast_1d(trans)]
            radius = np.atleast_1d(np.asarray(blk.radius, float)).astype(int)
            offset = np.atleast_1d(np.asarray(blk.offset, float)).astype(int)
            for k in range(Y.shape[0]):
                gt = Y[k]
                py = wf.window_feature(x, st, trans[k], int(radius[k]), int(offset[k]),
                                       relbins=py_bins)
                # compare ignoring NaN positions (must match NaN pattern too)
                nan_gt = np.isnan(gt); nan_py = np.isnan(py)
                if not np.array_equal(nan_gt, nan_py):
                    # inf vs nan can differ; report
                    d_nan = np.sum(nan_gt != nan_py)
                    print(f"  [{xk}/{st}/{trans[k]}/r{radius[k]}/o{offset[k]}] NaN-pattern differs at {d_nan} frames")
                both = ~nan_gt & ~nan_py
                if both.any():
                    # handle -inf equality
                    fin = both & np.isfinite(gt) & np.isfinite(py)
                    d = np.abs(gt[fin] - py[fin])
                    mx = d.max() if d.size else 0.0
                    inf_ok = np.array_equal(np.sign(gt[both & ~np.isfinite(gt)]),
                                            np.sign(py[both & ~np.isfinite(py)])) \
                        if (both & ~np.isfinite(gt)).any() or (both & ~np.isfinite(py)).any() else True
                    if mx > worst:
                        worst = mx; worst_desc = f"{xk}/{st}/{trans[k]}/r{radius[k]}/o{offset[k]}"
                    if mx > 1e-6 or not inf_ok:
                        print(f"  MISMATCH [{xk}/{st}/{trans[k]}/r{radius[k]}/o{offset[k]}] "
                              f"max|d|={mx:.3e} inf_ok={inf_ok}")
                n_checked += 1
    print(f"\nchecked {n_checked} window features across {len(xkeys)} vectors")
    print(f"relative-bin mismatches: {n_bin_mismatch}")
    print(f"worst finite abs diff: {worst:.3e}  ({worst_desc})")
    print("PASS" if worst < 1e-6 and n_bin_mismatch == 0 else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1])
