"""End-to-end validation: Python detect.py scores vs real JAABADetect scores.

Usage: check_integration.py <expdir> <classifier.mat> <jaabadetect_gt.mat>
"""
import sys, os
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.dirname(PKG))   # repo root, for matio
import detect
import matio


def _asfloat(v):
    return np.asarray(v, dtype=float).ravel()


def main(expdir, clf_mat, gt_mat):
    print("Running Python detect.py ...")
    res = detect.jaaba_detect(expdir, clf_mat, verbose=False)

    data, _ = matio.loadmat(gt_mat)         # handles v7.3 -> dict
    A = data["allScores"]
    while isinstance(A, list):              # allScores wrapped in a 1-elem cell
        A = A[0]
    gt_scores = A["scores"]                 # list of per-fly arrays
    gt_post = A.get("postprocessed")
    nflies = len(res["scores"])

    def _cell(v, i):
        return _asfloat(v[i])

    print(f"{nflies} flies; scoreNorm py={res['score_norm']:.4f} "
          f"matlab={float(np.asarray(A['scoreNorm']).ravel()[0]):.4f}\n")
    worst = 0.0
    worst_fly = None
    tot_frames = 0
    tot_scorediff = 0
    tot_postdiff = 0
    for i in range(nflies):
        ps = res["scores"][i]
        gs = _cell(gt_scores, i)
        n = min(ps.size, gs.size)
        ps, gs = ps[:n], gs[:n]
        both = ~np.isnan(ps) & ~np.isnan(gs)
        d = np.abs(ps[both] - gs[both])
        mx = float(d.max()) if d.size else 0.0
        nbad = int((d > 1e-4).sum())
        if mx > worst:
            worst, worst_fly = mx, i
        # postprocessed / sign agreement
        signdiff = int((np.sign(ps[both]) != np.sign(gs[both])).sum())
        pp = res["postprocessed"][i][:n]
        if gt_post is not None:
            gp = _cell(gt_post, i)[:n]
            postdiff = int(((pp > 0) != (gp > 0)).sum())
        else:
            postdiff = -1
        tot_frames += int(both.sum())
        tot_scorediff += nbad
        tot_postdiff += max(0, postdiff)
        print(f" fly{i}: n={n} max|score d|={mx:.3e}  frames>1e-4={nbad}  "
              f"sign-diff={signdiff}  postproc-diff={postdiff}")

    print(f"\nworst score abs diff: {worst:.3e} (fly {worst_fly})")
    print(f"total frames>1e-4: {tot_scorediff}/{tot_frames}   total postproc-diff: {tot_postdiff}/{tot_frames}")
    ok = worst < 1e-3 and tot_postdiff < 0.001 * max(tot_frames, 1)
    print("PASS" if ok else "REVIEW")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
