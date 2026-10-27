"""jaaba_detect: apply a trained JAABA classifier (exported .classifier.mat) to an
experiment's tracking data and produce per-frame behavior scores -- a MATLAB-free
Python port of JAABADetect.m.

Pipeline per fly: per-frame features -> window features (only the descriptors the
classifier references) -> boosted-stump score -> hysteresis + small-bout removal.
Everything is driven by the Classifier object; no classifier-specific constants.
"""
from __future__ import annotations

import os

import numpy as np

# Work both as a package submodule (jaaba_detect.detect, e.g. from a notebook) and
# as a standalone script (python detect.py, with jaaba_detect on sys.path).
try:
    from . import jab_io, trx_io, classify
    from . import window_features as wf
    from .perframe_apt import AptFeatures
    from .perframe_ellipse import EllipseFeatures
except ImportError:
    import jab_io, trx_io, classify
    import window_features as wf
    from perframe_apt import AptFeatures
    from perframe_ellipse import EllipseFeatures


def _find(expdir, name):
    p = os.path.join(expdir, name)
    return p if os.path.exists(p) else None


def compute_perframe(pff, apt: AptFeatures, ell: EllipseFeatures):
    """Return a list (per fly) of the natural-length per-frame feature vector `pff`."""
    if pff.startswith("apt_"):
        return apt.compute(pff)
    return ell.compute(pff)


def jaaba_detect(expdir, classifier_mat, trkfile=None, roi=None, verbose=True):
    """Run inference. Returns dict: scores/postprocessed/t0s/t1s per fly + metadata."""
    clf = jab_io.load_classifier(classifier_mat)

    trxpath = _find(expdir, clf.trxfilename) or _find(expdir, "registered_trx.mat")
    if trxpath is None:
        raise FileNotFoundError(f"trx file {clf.trxfilename!r} not found in {expdir}")
    if trkfile is None:
        # jab names an apt.trk; real experiments often store apttrk.mat
        trkfile = _find(expdir, clf.trkfilename) or _find(expdir, "apttrk.mat")
    needs_apt = any(p.startswith("apt_") for p in clf.pff_names())
    if needs_apt and trkfile is None:
        raise FileNotFoundError(f"APT trk file needed but not found in {expdir}")

    traj = trx_io.load_experiment(trxpath, trkfile if needs_apt else None,
                                  check_frame_consistency=needs_apt)
    return jaaba_detect_traj(traj, clf, roi=roi, verbose=verbose)


def jaaba_detect_traj(traj, clf, roi=None, verbose=True):
    """Core inference on an already-built Trajectories.

    Shared by jaaba_detect() (experiment dir) and the array adapters (MABe X mat /
    flyllm track), so classifier loading lives here in one place. `clf` is a
    jab_io.Classifier OR a path to an exported .classifier.mat; `traj` a
    trx_io.Trajectories whose flies carry x_mm/y_mm/theta_mm/a_mm/b_mm/dt/pxpermm
    (+ kpts in APT landmark order for APT features). Returns the same dict as
    jaaba_detect().
    """
    if not isinstance(clf, jab_io.Classifier):
        clf = jab_io.load_classifier(clf)
    apt = AptFeatures(traj, roi=roi)
    fov = clf.apt.get("fov", np.pi) or np.pi
    maxar = clf.apt.get("max_dnose2ell_anglerange", 127.0) or 127.0
    nbl = clf.apt.get("nbodylengths_near", 2.5) or 2.5
    ell = EllipseFeatures(traj, roi=roi, fov=fov, max_dnose2ell_anglerange=maxar,
                          nbodylengths_near=nbl)

    # descriptor -> unique column
    col_of = {d.key(): k for k, d in enumerate(clf.unique_descs)}
    by_pff = clf.descs_by_pff()

    if verbose:
        print(f"[jaaba_detect] {clf.behavior!r}: {clf.nstumps} stumps, "
              f"{len(clf.unique_descs)} window features, {len(clf.pff_names())} per-frame features, "
              f"{traj.nflies} flies")

    # compute per-frame features once (each returns a per-fly list)
    pff_vecs = {}
    for pff in clf.pff_names():
        pff_vecs[pff] = compute_perframe(pff, apt, ell)
        if verbose:
            print(f"    pff {pff}")

    nunique = len(clf.unique_descs)
    scores, postp, t0s, t1s, tstart, tend = [], [], [], [], [], []
    for i in range(traj.nflies):
        n = traj[i].nframes
        X = np.full((n, nunique), np.nan)
        for pff, dlist in by_pff.items():
            vec = np.asarray(pff_vecs[pff][i], dtype=float).ravel()
            relb = wf.relative_bins(vec) if any(d.trans == "relative" for d in dlist) else None
            for d in dlist:
                col = col_of[d.key()]
                y = wf.window_feature(vec, d.stat, d.trans, d.radius, d.offset,
                                      relbins=relb, extra=dict(d.extra))
                L = min(len(y), n)
                X[:L, col] = y[:L]
        s = classify.boost_classify(X, clf)
        if n < 3:
            s[:] = -1.0
        pp = classify.post_process(s, clf.pp, clf.score_norm)
        a, b = classify.bout_intervals(pp)
        scores.append(s); postp.append(pp)
        t0s.append(a + traj[i].firstframe - 1)   # -> absolute frames
        t1s.append(b + traj[i].firstframe - 1)
        tstart.append(traj[i].firstframe); tend.append(traj[i].endframe)

    return dict(behavior=clf.behavior, scorefilename=clf.scorefilename,
                score_norm=clf.score_norm, scores=scores, postprocessed=postp,
                t0s=t0s, t1s=t1s, tStart=tstart, tEnd=tend)


def save_scores(result, outpath):
    """Write a JAABA-style allScores .mat (scipy v7)."""
    import scipy.io as sio
    nflies = len(result["scores"])

    def cell(vecs):
        out = np.empty((1, nflies), dtype=object)
        for i, v in enumerate(vecs):
            out[0, i] = np.asarray(v, float).reshape(1, -1)
        return out

    allScores = dict(
        scores=cell(result["scores"]),
        postprocessed=cell(result["postprocessed"]),
        tStart=np.array(result["tStart"]).reshape(1, -1).astype(float),
        tEnd=np.array(result["tEnd"]).reshape(1, -1).astype(float),
        t0s=cell(result["t0s"]),
        t1s=cell(result["t1s"]),
        scoreNorm=float(result["score_norm"]),
    )
    sio.savemat(outpath, {"allScores": allScores, "behaviorName": result["behavior"]})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("expdir")
    ap.add_argument("classifier_mat")
    ap.add_argument("--trk", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = jaaba_detect(args.expdir, args.classifier_mat, trkfile=args.trk)
    for i in range(len(res["scores"])):
        s = res["scores"][i]
        print(f"  fly{i}: score mean={np.nanmean(s):.3f} min={np.nanmin(s):.3f} "
              f"max={np.nanmax(s):.3f}  bouts={len(res['t0s'][i])}")
    if args.out:
        save_scores(res, args.out)
        print("wrote", args.out)
