"""Compare perframe_apt.py against the MATLAB reference (apt_ref.m output)."""
import sys, os
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trx_io
import perframe_apt


def _cellvec(v, i):
    a = np.asarray(v, dtype=object).ravel()
    return np.asarray(a[i], dtype=float).ravel()


def main(gtfile, trxfile, trkfile):
    m = sio.loadmat(gtfile, squeeze_me=True, struct_as_record=False)
    names = [str(s) for s in np.atleast_1d(m["feat_names"]).ravel()]
    traj = trx_io.load_experiment(trxfile, trkfile, check_frame_consistency=False)
    apt = perframe_apt.AptFeatures(traj)

    worst = 0.0
    worst_where = None
    nfail = 0
    for fi, fn in enumerate(names):
        gt = m[f"f{fi + 1}"]            # cell {1 x nflies}
        py = apt.compute(fn)
        for i in range(traj.nflies):
            g = _cellvec(gt, i)
            p = np.asarray(py[i], dtype=float).ravel()
            if g.size != p.size:
                print(f"  LEN MISMATCH {fn} fly{i}: matlab {g.size} vs py {p.size}")
                nfail += 1
                continue
            nan_g = np.isnan(g); nan_p = np.isnan(p)
            if not np.array_equal(nan_g, nan_p):
                print(f"  NAN-PATTERN {fn} fly{i}: {np.sum(nan_g != nan_p)} frames differ")
                nfail += 1
            both = ~nan_g & ~nan_p
            if both.any():
                d = np.abs(g[both] - p[both])
                mx = float(d.max())
                if mx > worst:
                    worst = mx; worst_where = f"{fn} fly{i}"
                if mx > 1e-6:
                    print(f"  MISMATCH {fn} fly{i}: max|d|={mx:.3e}")
                    nfail += 1
    print(f"\n{len(names)} features x {traj.nflies} flies")
    print(f"worst abs diff: {worst:.3e}  ({worst_where})")
    print("PASS" if worst < 1e-6 and nfail == 0 else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
