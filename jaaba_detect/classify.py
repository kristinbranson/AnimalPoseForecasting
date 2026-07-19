"""Boosted-stump scoring + post-processing (myBoostClassify.m, PostProcessor.m).

The classifier is a sum of decision stumps; post-processing is hysteresis (or a
boxcar filter) followed by small-bout removal. All thresholds/blen/scoreNorm come
from the Classifier object, nothing hard-coded.
"""
from __future__ import annotations

import numpy as np


def boost_classify(X_unique: np.ndarray, clf) -> np.ndarray:
    """myBoostClassify: scores[t] = sum_j alpha_j * (+1 if stump j fires else -1).

    X_unique is (N, n_unique_descs); clf.stump_col maps each stump to its column.
    """
    d = X_unique[:, clf.stump_col]                       # (N, nstumps)
    fire = np.where(clf.dir > 0, d > clf.tr, d <= clf.tr)  # bool (N, nstumps)
    tt = fire.astype(float) * 2.0 - 1.0
    return tt @ clf.alpha                                 # (N,)


# ------------------------------------------------------------------ postproc
def _imfill_1d(bw: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """imfill(bw, seeds) for 1-D: flood-fill False-runs of bw that contain a seed."""
    out = bw.copy()
    n = bw.size
    i = 0
    while i < n:
        if not bw[i]:
            j = i
            while j < n and not bw[j]:
                j += 1
            if seed[i:j].any():
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def apply_hysteresis(curs: np.ndarray, hi_val: float, lo_val: float,
                     score_norm: float) -> np.ndarray:
    """PostProcessor.ApplyHysteresis -> bool per-frame."""
    n = curs.size
    if n == 0:
        return np.zeros(0, bool)
    hthresh = curs > hi_val * score_norm
    lthresh = curs > 0
    if hthresh.any():
        pos = _imfill_1d(~lthresh, hthresh) & lthresh
        compute_neg = True
    else:
        pos = np.zeros(n, bool)
        compute_neg = False
    hthresh2 = curs < lo_val * score_norm
    lthresh2 = curs < hi_val * score_norm
    if hthresh2.any() and compute_neg:
        neg = _imfill_1d(~lthresh2, hthresh2) & lthresh2
    else:
        neg = np.ones(n, bool)
    return pos | ~neg


def apply_filtering(curs: np.ndarray, filt_size: int) -> np.ndarray:
    """PostProcessor.ApplyFiltering -> bool per-frame."""
    if curs.size == 0:
        return np.zeros(0, bool)
    filts = np.convolve(curs, np.ones(int(filt_size)), "same")
    return filts > 0


def remove_small_bouts(posts_bool: np.ndarray, blen: int) -> np.ndarray:
    """PostProcessor.RemoveSmallBouts. Input/output as int {0,1} (matches JAABA flips)."""
    posts = posts_bool.astype(int)
    if blen > 1 and posts.size > 0:
        if posts.size <= blen:
            posts[:] = 1 if (posts > 0).sum() > posts.size / 2 else -1
            return posts
        while True:
            tp = np.concatenate([posts, [1 - posts[-1]]])
            changes = np.where(tp[:-1] != tp[1:])[0]
            ends = np.concatenate([[0], changes + 1])
            blens = np.diff(ends)
            smallbout = int(np.argmin(blens))
            if blens[smallbout] >= blen:
                break
            s, e = ends[smallbout], ends[smallbout + 1]
            posts[s:e] = 1 - posts[s:e]
    return posts


def post_process(curs: np.ndarray, pp: dict, score_norm: float) -> np.ndarray:
    """PostProcessor.PostProcess. Returns int {0,1} per-frame (before ->±1 encoding)."""
    if not pp or not pp.get("method"):
        return curs
    if pp["method"].lower() == "hysteresis":
        posts = apply_hysteresis(curs, pp["hyst_hi"], pp["hyst_lo"], score_norm)
    else:
        posts = apply_filtering(curs, pp["filt_size"])
    return remove_small_bouts(posts, int(pp["blen"]))


def bout_intervals(postprocessed: np.ndarray):
    """t0s/t1s (1-indexed, JAABA convention) for runs where postprocessed>0.

    Returns arrays of bout starts t0s and ends t1s such that frames t0s[k]..t1s[k]-1
    are the behavior bout (get_interval_ends semantics).
    """
    b = np.asarray(postprocessed) > 0
    if b.size == 0:
        return np.array([], int), np.array([], int)
    d = np.diff(np.concatenate([[0], b.astype(int), [0]]))
    t0s = np.where(d == 1)[0] + 1   # 1-indexed start
    t1s = np.where(d == -1)[0] + 1  # 1-indexed end (exclusive, = last+1)
    return t0s, t1s
