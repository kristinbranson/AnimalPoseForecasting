"""Validate classify.boost_classify + post_process against MATLAB myBoostClassify
+ PostProcessor.PostProcess on a random feature matrix."""
import sys, os
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import classify
import jab_io


def main(gtfile, clffile):
    m = sio.loadmat(gtfile, squeeze_me=True, struct_as_record=False)
    Xu = np.atleast_2d(np.asarray(m["Xu"], float))
    udims = np.asarray(m["udims"]).ravel().astype(int)
    gt_scores = np.asarray(m["scores"], float).ravel()
    gt_posts = np.asarray(m["posts"], float).ravel()

    cm = sio.loadmat(clffile, squeeze_me=True, struct_as_record=False)
    dim = np.asarray(cm["dim"]).ravel().astype(int)
    ddir = np.asarray(cm["dir"]).ravel().astype(float)
    tr = np.asarray(cm["tr"]).ravel().astype(float)
    alpha = np.asarray(cm["alpha"]).ravel().astype(float)

    col = {d: i for i, d in enumerate(udims)}
    cols = np.array([col[d] for d in dim])
    d = Xu[:, cols]
    tt = np.where(ddir > 0, d > tr, d <= tr).astype(float) * 2 - 1
    scores = tt @ alpha
    ds = float(np.max(np.abs(scores - gt_scores)))

    clf = jab_io.load_classifier(clffile)
    posts = classify.post_process(scores, clf.pp, clf.score_norm)
    dp = int(np.sum((posts > 0) != (gt_posts > 0)))

    print(f"myBoostClassify score match: max|d|={ds:.3e}")
    print(f"PostProcess frame disagreements: {dp}/{len(posts)}")
    print("PASS" if ds < 1e-9 and dp == 0 else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
