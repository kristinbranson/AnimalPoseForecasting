"""Validate the MABe-X adapter against the (already JAABADetect-validated) real
experiment path, on NorpA_JRC_SS36564_RigB_20210903T081150 which is video 1 of
Xtest1_v3.mat and also a real experiment with registered_trx + apt.trk.

For each classifier: run detect on the real experiment (pixels from apt.trk) and
jaaba_detect_from_X on the X slice (mm rescaled by pxpermm), match flies, compare
raw scores frame-by-frame. A tiny max-diff proves the adapter reconstructs the same
features JAABA uses.
"""
import os, sys
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.dirname(PKG))
import detect, jab_io, mabe_adapter

EXP = "/groups/branson/bransonlab/flybubble_social/NorpA_JRC_SS36564_RigB_20210903T081150"
XMAT = "/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/Xtest1_v3.mat"
VIDEOIDX = 1
PXPERMM = 18.901264068893322   # NorpA registered_trx pxpermm (exact reproduction)

# classifiers are not bundled; they live with the data (MABe2022 dir here)
DATADIR = os.path.dirname(XMAT)
CLFS = [
    ("chase",     os.path.join(DATADIR, "chase_apt.classifier.mat")),
    ("wingext",   os.path.join(DATADIR, "wingextension_apt.classifier.mat")),
    ("courtship", os.path.join(DATADIR, "courtship_v2pt3_apt.classifier.mat")),
]


def match_flies(real, xres):
    """Match adapter flies to real flies by firstframe + mean start position."""
    # use each result's tStart and the first score index isn't positional; match on
    # firstframe (tStart) and the per-fly mean score is not geometric -> match on
    # tStart then on order. All NorpA flies share frames, so match by index of the
    # fly whose scores best-correlate.
    nr = len(real["scores"]); nx = len(xres["scores"])
    used = set(); pairs = []
    for xi in range(nx):
        xs = np.asarray(xres["scores"][xi], float)
        best, bestc = -1, -np.inf
        for ri in range(nr):
            if ri in used:
                continue
            rs = np.asarray(real["scores"][ri], float)
            n = min(xs.size, rs.size)
            a, b = xs[:n], rs[:n]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 10:
                continue
            c = -np.mean(np.abs(a[m] - b[m]))   # closest by mean abs diff
            if c > bestc:
                bestc, best = c, ri
        if best >= 0:
            used.add(best); pairs.append((xi, best))
    return pairs


def main():
    print(f"loading X video {VIDEOIDX} ...")
    Xv, Xnames, ids, frv = mabe_adapter.load_X_video(XMAT, VIDEOIDX)
    print(f"  X slice: {Xv.shape}  frames {frv.min()}..{frv.max()}")
    for name, cm in CLFS:
        clf = jab_io.load_classifier(cm)
        print(f"\n===== {name}  ({clf.behavior!r}) =====")
        real = detect.jaaba_detect(EXP, cm, verbose=False)
        xres = mabe_adapter.jaaba_detect_from_X(Xv, Xnames, clf, pxpermm=PXPERMM,
                                                ids=ids, first_frame=1, verbose=False)
        print(f"  real flies={len(real['scores'])}  adapter flies={len(xres['scores'])}")
        pairs = match_flies(real, xres)
        worst = 0.0; tot = 0; signdiff = 0; ppdiff = 0
        for xi, ri in pairs:
            xs = np.asarray(xres["scores"][xi], float)
            rs = np.asarray(real["scores"][ri], float)
            n = min(xs.size, rs.size); xs, rs = xs[:n], rs[:n]
            m = np.isfinite(xs) & np.isfinite(rs)
            d = np.abs(xs[m] - rs[m])
            worst = max(worst, float(d.max()) if d.size else 0.0)
            tot += int(m.sum())
            signdiff += int((np.sign(xs[m]) != np.sign(rs[m])).sum())
            xp = np.asarray(xres["postprocessed"][xi], float)[:n]
            rp = np.asarray(real["postprocessed"][ri], float)[:n]
            mp = np.isfinite(xp) & np.isfinite(rp)
            ppdiff += int(((xp[mp] > 0) != (rp[mp] > 0)).sum())
        print(f"  {len(pairs)} flies matched, {tot} frames: worst|score d|={worst:.3e}  "
              f"sign-diff={signdiff}  postproc-diff={ppdiff}  "
              f"({100*(1-ppdiff/max(tot,1)):.4f}% agree)")


if __name__ == "__main__":
    main()
