"""Compare Python window features against JAABA's, on real data.

The existing check_window_features*.py validate the window statistics on synthetic
vectors. This one checks the features a trained classifier actually uses, computed
from a real experiment's per-frame data, which additionally exercises the `relative`
percentile bins (taken over the whole trajectory) and the descriptor -> column
bookkeeping that builds the design matrix.

Ground truth comes from dump_window_features.m. Columns are matched by a canonical
key string rather than by position, so a reordering on either side shows up as a
missing key instead of a silent misalignment.

Usage: check_window_features_expdir.py <window_dump.mat> <expdir> <classifier.mat>
"""
import os
import sys

import numpy as np
import scipy.io as sio

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.dirname(PKG))
import detect
import jab_io
import trx_io
import window_features as wf
from perframe_apt import AptFeatures
from perframe_ellipse import EllipseFeatures

# JAABA stores window data as single (JLabelData: x_curr_all{j} = single(x_curr)), so
# agreement is bounded by single precision, not by double.
SINGLE_EPS = np.finfo(np.float32).eps


def _g(v):
    """Format a number the way MATLAB's %g does, for building the match key."""
    return f"{float(v):g}"


def desc_key(d):
    """Canonical key for a window-feature descriptor, matching name2key in the .m."""
    extra = ",".join(sorted(f"{k}={_g(v)}" for k, v in d.extra))
    return f"{d.pff}|{d.stat}|{d.trans}|{_g(d.radius)}|{_g(d.offset)}|{extra}"


def main(dump_mat, expdir, clf_mat):
    dump = sio.loadmat(dump_mat, squeeze_me=True, struct_as_record=False)
    gt = np.asarray(dump["X"], dtype=np.float64)          # (nframes, nfeat)
    keys = [str(k) for k in np.asarray(dump["keys"]).ravel()]
    fly = int(np.asarray(dump["fly"]).ravel()[0]) - 1     # MATLAB 1-based -> 0-based
    nframes, nfeat = gt.shape
    print(f"ground truth: {nframes} frames x {nfeat} features, fly {fly + 1}")

    clf = jab_io.load_classifier(clf_mat)
    by_key = {desc_key(d): d for d in clf.unique_descs}
    missing = [k for k in keys if k not in by_key]
    if missing:
        print(f"  {len(missing)} dumped keys absent from the classifier, e.g. {missing[:3]}")

    # load tracking exactly as detect.jaaba_detect does, so both see the same trk file
    trxpath = detect._find(expdir, clf.trxfilename) or detect._find(expdir, "registered_trx.mat")
    trkpath = detect._find(expdir, clf.trkfilename) or detect._find(expdir, "apttrk.mat")
    print(f"tracking: {os.path.basename(trxpath)} + {os.path.basename(trkpath)}")
    traj = trx_io.load_experiment(trxpath, trkpath, check_frame_consistency=True)
    apt = AptFeatures(traj)
    ell = EllipseFeatures(traj, fov=clf.apt["fov"],
                          max_dnose2ell_anglerange=clf.apt["max_dnose2ell_anglerange"],
                          nbodylengths_near=clf.apt["nbodylengths_near"])

    # group the dumped keys by per-frame feature so each vector is computed once and
    # its relative bins are shared, as in detect.jaaba_detect_traj
    by_pff = {}
    for col, k in enumerate(keys):
        d = by_key.get(k)
        if d is not None:
            by_pff.setdefault(d.pff, []).append((col, d))

    rows = []
    where = {}          # key -> (first bad frames, last bad frames), 1-based
    for pff, items in by_pff.items():
        vec = np.asarray(detect.compute_perframe(pff, apt, ell)[fly], float).ravel()
        relb = wf.relative_bins(vec) if any(d.trans == "relative" for _, d in items) else None
        for col, d in items:
            y = wf.window_feature(vec, d.stat, d.trans, d.radius, d.offset,
                                  relbins=relb, extra=dict(d.extra))
            py = np.full(nframes, np.nan)
            L = min(len(y), nframes)
            py[:L] = y[:L]
            g = gt[:, col]

            nan_g, nan_p = np.isnan(g), np.isnan(py)
            n_nan_diff = int((nan_g != nan_p).sum())
            both = ~nan_g & ~nan_p & np.isfinite(g) & np.isfinite(py)
            if both.any():
                absd = np.abs(g[both] - py[both])
                scale = np.maximum(np.abs(g[both]), np.abs(py[both]))
                reld = absd / np.maximum(scale, np.finfo(float).tiny)
                worst_abs = float(absd.max())
                worst_rel = float(reld.max())
                # frames beyond what single-precision storage can explain
                n_bad = int((reld > 64 * SINGLE_EPS).sum())
            else:
                worst_abs = worst_rel = 0.0
                n_bad = 0
            rows.append((keys[col], worst_abs, worst_rel, n_bad, n_nan_diff, int(both.sum())))
            if n_bad or n_nan_diff:
                bad_frames = np.flatnonzero(nan_g != nan_p)
                if both.any():
                    rel_all = np.zeros(nframes)
                    rel_all[both] = reld
                    bad_frames = np.union1d(bad_frames, np.flatnonzero(rel_all > 64 * SINGLE_EPS))
                where[keys[col]] = (bad_frames[:3] + 1).tolist(), (bad_frames[-3:] + 1).tolist()

    rows.sort(key=lambda r: -r[2])
    differ = [r for r in rows if r[3] or r[4]]
    clean = [r for r in rows if not (r[3] or r[4])]
    print(f"\n{len(rows)} window features compared; {len(clean)} match to single precision, "
          f"{len(differ)} differ\n")
    header = f"{'window feature':<66} {'worst|d|':>10} {'worst rel':>10} {'bad':>7} {'nanΔ':>6}"
    if differ:
        print("differing (bad = frames beyond single precision, nanΔ = frames NaN on one side only):")
        print(header)
        for k, wa, wr, nb, nn, _ in differ:
            first, last = where[k]
            print(f"{k:<66} {wa:>10.3e} {wr:>10.3e} {nb:>7d} {nn:>6d}   frames {first} .. {last}")
    print(f"\nclosest of the matching features:")
    print(header)
    for k, wa, wr, nb, nn, _ in clean[:5]:
        print(f"{k:<66} {wa:>10.3e} {wr:>10.3e} {nb:>7d} {nn:>6d}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
