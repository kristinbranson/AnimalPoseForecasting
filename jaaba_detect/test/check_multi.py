"""Compare Python detect.py vs real JAABADetect for several classifiers at once.

Usage: check_multi.py <expdir> <multi_gt.mat> <clf1.mat> [<clf2.mat> ...]
multi_gt.mat holds allScores_k + behavior_k for each jab (run_jaabadetect_multi.m).
Each Python classifier is matched to the ground truth by behavior name.
"""
import sys, os
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.dirname(PKG))
import detect, jab_io, matio


def _cell(v, i):
    return np.asarray(v[i], float).ravel()


def main(expdir, gt_mat, clf_mats):
    data, _ = matio.loadmat(gt_mat)
    # collect ground-truth per behavior
    gt = {}
    k = 1
    while f"allScores_{k}" in data:
        A = data[f"allScores_{k}"]
        while isinstance(A, list):
            A = A[0]
        beh = data[f"behavior_{k}"]
        beh = beh if isinstance(beh, str) else str(np.asarray(beh).ravel()[0])
        gt[beh] = A
        k += 1
    print("ground-truth behaviors:", list(gt.keys()))

    for cm in clf_mats:
        clf = jab_io.load_classifier(cm)
        beh = clf.behavior
        print(f"\n===== {os.path.basename(cm)}  (behavior={beh!r}) =====")
        if beh not in gt:
            print(f"  no ground truth for behavior {beh!r}; have {list(gt.keys())}"); continue
        A = gt[beh]
        res = detect.jaaba_detect(expdir, cm, verbose=False)
        gs_all = A["scores"]
        gp_all = A.get("postprocessed")
        nflies = len(res["scores"])
        tot_f = tot_sd = tot_pd = 0
        worst = 0.0
        for i in range(nflies):
            ps = res["scores"][i]; gs = _cell(gs_all, i)
            n = min(ps.size, gs.size); ps, gs = ps[:n], gs[:n]
            both = ~np.isnan(ps) & ~np.isnan(gs)
            d = np.abs(ps[both] - gs[both])
            mx = float(d.max()) if d.size else 0.0
            worst = max(worst, mx)
            nbad = int((d > 1e-4).sum())
            sd = int((np.sign(ps[both]) != np.sign(gs[both])).sum())
            if gp_all is not None:
                gp = _cell(gp_all, i)[:n]
                pd = int(((res["postprocessed"][i][:n] > 0) != (gp > 0)).sum())
            else:
                pd = sd
            tot_f += int(both.sum()); tot_sd += sd; tot_pd += max(0, pd)
        print(f"  scoreNorm py={res['score_norm']:.4f} matlab={float(np.asarray(A['scoreNorm']).ravel()[0]):.4f}")
        print(f"  {nflies} flies, {tot_f} frames: worst|score d|={worst:.3e}  "
              f"sign-diff={tot_sd}  postproc(behavior)-diff={tot_pd}  ({100*(1-tot_pd/max(tot_f,1)):.4f}% agree)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
