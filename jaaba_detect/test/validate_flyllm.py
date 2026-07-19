"""Quantify the flyllm/simulated-path approximation cost.

Build a flyllm-style keypoint track (19 keypoints, no outer wings, no ellipse)
from the NorpA slice of Xtest1_v3.mat, run jaaba_detect_from_track (which
approximates outer wings by mid-wings and reconstructs the ellipse from
keypoints), and compare its scores to jaaba_detect_from_X (the exact registered
arrays). The difference is purely the flyllm-representation approximation.
"""
import os, sys
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.dirname(PKG))
import jab_io, mabe_adapter as M

XMAT = "/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/Xtest1_v3.mat"
VIDEOIDX = 1
PXPERMM = 18.901264068893322

# classifiers are not bundled; they live with the data (MABe2022 dir here)
DATADIR = os.path.dirname(XMAT)
CLFS = [
    ("chase",     os.path.join(DATADIR, "chase_apt.classifier.mat")),
    ("wingext",   os.path.join(DATADIR, "wingextension_apt.classifier.mat")),
    ("courtship", os.path.join(DATADIR, "courtship_v2pt3_apt.classifier.mat")),
]


def build_flyllm_track(Xv, Xnames):
    """Build a flyllm-order keypoint track from the X columns, using flyllm's own
    keypoint order + the adapter's name map (so this exercises the real by-name path
    and would catch a left/right wing swap)."""
    col = {n: i for i, n in enumerate(Xnames)}
    names = M.flyllm_keypoint_names()
    nf, nfl, _ = Xv.shape
    track = np.full((nfl, nf, len(names), 2), np.nan)
    for k, name in enumerate(names):
        ap = M.FLYLLM_TO_APT_NAME[name]
        track[:, :, k, 0] = Xv[:, :, col[f"{ap}_x_mm"]].T
        track[:, :, k, 1] = Xv[:, :, col[f"{ap}_y_mm"]].T
    return track


def main():
    Xv, Xnames, ids, frv = M.load_X_video(XMAT, VIDEOIDX)
    track = build_flyllm_track(Xv, Xnames)
    print(f"X slice {Xv.shape}, flyllm track {track.shape}")
    for name, cm in CLFS:
        clf = jab_io.load_classifier(cm)
        print(f"\n===== {name} ({clf.behavior!r}) =====")
        exact = M.jaaba_detect_from_X(Xv, Xnames, clf, pxpermm=PXPERMM, ids=ids, verbose=False)
        approx = M.jaaba_detect_from_track(track, clf, pxpermm=PXPERMM, verbose=False)
        ne, na = len(exact["scores"]), len(approx["scores"])
        worst = 0.0; tot = 0; sd = 0; pd = 0
        for i in range(min(ne, na)):
            xs = np.asarray(approx["scores"][i], float)
            rs = np.asarray(exact["scores"][i], float)
            n = min(xs.size, rs.size); xs, rs = xs[:n], rs[:n]
            m = np.isfinite(xs) & np.isfinite(rs)
            d = np.abs(xs[m] - rs[m])
            worst = max(worst, float(d.max()) if d.size else 0.0)
            tot += int(m.sum()); sd += int((np.sign(xs[m]) != np.sign(rs[m])).sum())
            xp = np.asarray(approx["postprocessed"][i], float)[:n]
            rp = np.asarray(exact["postprocessed"][i], float)[:n]
            mp = np.isfinite(xp) & np.isfinite(rp)
            pd += int(((xp[mp] > 0) != (rp[mp] > 0)).sum())
        print(f"  {min(ne,na)} flies, {tot} frames: worst|score d|={worst:.3e}  "
              f"sign-diff={sd} ({100*sd/max(tot,1):.3f}%)  "
              f"behavior-diff={pd} ({100*(1-pd/max(tot,1)):.4f}% agree)")


if __name__ == "__main__":
    main()
