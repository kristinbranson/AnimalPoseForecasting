"""Refit the keypoint->ellipse-axis linear coefficients on a large, stratified
sample of flybubble_social experiments, and test generalization on a disjoint
held-out set.

Predictors (mm): head-tail distance ||ant_head - pos_abdomen||, thorax width
||left_thorax - right_thorax||, computed from apt.trk keypoints (pixels) / pxpermm.
Targets (mm, quarter-axis): registered_trx a_mm, b_mm (FlyTracker ellipse).

Model:  a_mm = ma * head_tail_dist + ca ;  b_mm = mb * thorax_width + cb
"""
import os, sys
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
import trx_io

EXPLIST = "/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/MABE_explist20250104.txt"

# APT-order landmark indices (0-based) used as predictors
I_HEAD, I_TAIL, I_LTH, I_RTH = 0, 6, 3, 4     # ant_head, pos_abdomen, left/right thorax
FRAME_STRIDE = 200                             # subsample frames per fly
# physical sanity bounds (mm) to drop tracking/ellipse outliers
A_LO, A_HI = 0.3, 1.5
B_LO, B_HI = 0.1, 0.6
HTD_LO, HTD_HI = 1.0, 6.0
THW_LO, THW_HI = 0.3, 3.0


def family(name):
    return name.split("_")[0]


def select(nfit_per_fam=2, ntest_per_fam=2):
    """Stratified, deterministic fit/test split over genotype families."""
    exps = [l.strip() for l in open(EXPLIST) if l.strip()]
    byfam = {}
    for e in exps:
        byfam.setdefault(family(os.path.basename(e)), []).append(e)
    fit, test = [], []
    for fam, lst in sorted(byfam.items()):
        lst = sorted(lst)
        # spread picks across the sorted list (rig/date variety)
        n = len(lst)
        pick = [lst[int(round(k * (n - 1) / max(1, (nfit_per_fam + ntest_per_fam) - 1)))]
                for k in range(nfit_per_fam + ntest_per_fam)]
        pick = list(dict.fromkeys(pick))          # dedup keep order
        fit += pick[:nfit_per_fam]
        test += pick[nfit_per_fam:nfit_per_fam + ntest_per_fam]
    return fit, test


def _apt_targets(f):
    """Yield (i, dataset) for each target in an open apt.trk h5py file, where
    dataset[frame] indexes (2, npts) or (npts, 2). Returns refs + startframes."""
    import h5py
    sfr = np.array(f["startframes"]).ravel().astype(int)
    efr = np.array(f["endframes"]).ravel().astype(int)
    pTrk = f["pTrk"]
    refs = np.array(pTrk).ravel() if (isinstance(pTrk, h5py.Dataset)
                                      and pTrk.dtype == object) else None
    return refs, sfr, efr, pTrk


def collect(exp):
    """Return (htd, a_mm, thw, b_mm) mm arrays for one experiment, or None.

    apt.trk is read STRIDED (only the sampled frames) to avoid loading the whole
    150MB keypoint array.
    """
    import h5py
    trxf = os.path.join(exp, "registered_trx.mat")
    trkf = os.path.join(exp, "apt.trk")
    if not (os.path.exists(trxf) and os.path.exists(trkf)):
        return None
    try:
        traj = trx_io.read_trx(trxf)
        f = h5py.File(trkf, "r")
    except Exception as ex:
        print(f"    SKIP {os.path.basename(exp)}: {ex}")
        return None
    try:
        refs, sfr, efr, pTrk = _apt_targets(f)
        ntrk = sfr.size
        if ntrk != traj.nflies:
            print(f"    SKIP {os.path.basename(exp)}: {ntrk} trk vs {traj.nflies} trx")
            return None
        HTD, A, THW, B = [], [], [], []
        for i, fly in enumerate(traj.flies):
            P = fly.pxpermm
            t0 = max(fly.firstframe, int(sfr[i]))
            t1 = min(fly.endframe, int(efr[i]))
            if t1 <= t0:
                continue
            tt = np.arange(t0, t1 + 1, FRAME_STRIDE)
            ti = tt - fly.firstframe                     # trx index
            ki = tt - int(sfr[i])                        # apt index
            ds = f[refs[i]] if refs is not None else pTrk[..., i]
            # ds shape: MATLAB [npts,2,nframes] -> h5py (nframes,2,npts)
            sub = np.asarray(ds[ki, :, :])               # (m, 2, npts) strided read
            xy = np.transpose(sub, (0, 2, 1))            # (m, npts, 2)
            a = np.asarray(fly.a_mm)[ti]; b = np.asarray(fly.b_mm)[ti]
            head = xy[:, I_HEAD]; tail = xy[:, I_TAIL]
            lth = xy[:, I_LTH]; rth = xy[:, I_RTH]
            htd = np.hypot(head[:, 0] - tail[:, 0], head[:, 1] - tail[:, 1]) / P
            thw = np.hypot(lth[:, 0] - rth[:, 0], lth[:, 1] - rth[:, 1]) / P
            HTD.append(htd); A.append(a); THW.append(thw); B.append(b)
    finally:
        f.close()
    if not HTD:
        return None
    return (np.concatenate(HTD), np.concatenate(A),
            np.concatenate(THW), np.concatenate(B))


def pool(exps, label):
    HTD, A, THW, B = [], [], [], []
    n_ok = 0
    for e in exps:
        r = collect(e)
        if r is None:
            continue
        htd, a, thw, b = r
        HTD.append(htd); A.append(a); THW.append(thw); B.append(b)
        n_ok += 1
        print(f"    {label}: {os.path.basename(e):55s} n={a.size}")
    HTD = np.concatenate(HTD); A = np.concatenate(A)
    THW = np.concatenate(THW); B = np.concatenate(B)
    print(f"  {label}: {n_ok} experiments, {A.size} fly-frames pooled")
    return HTD, A, THW, B


def clean(x, y, xlo, xhi, ylo, yhi):
    m = np.isfinite(x) & np.isfinite(y) & (x > xlo) & (x < xhi) & (y > ylo) & (y < yhi)
    return x[m], y[m]


def linfit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (m, c), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return m, c


def resid_stats(x, y, m, c):
    r = np.abs(y - (m * x + c))
    return dict(mean_abs=round(float(r.mean()), 4),
                p95=round(float(np.percentile(r, 95)), 4),
                rms=round(float(np.sqrt(np.mean(r ** 2))), 4))


def report(name, x, y, m, c):
    s = resid_stats(x, y, m, c)
    print(f"  {name}: slope={m:.4f} intercept={c:.4f}  "
          f"resid mean|.|={s['mean_abs']:.4f} p95={s['p95']:.4f} rms={s['rms']:.4f}  "
          f"(target mean {y.mean():.3f}, n={y.size})")


def write_json(ma, ca, mb, cb, nfit, xat, yat, xbt, ybt, nfit_exps, ntest_exps,
               out=os.path.join(PKG, "ellipse_from_keypoints.json")):
    import json
    spec = {
      "ellipse_from_keypoints": {
        "description": "Reconstruct a FlyTracker-style body ellipse (x,y,theta,a,b) from APT/flyllm keypoints, for applying JAABA classifiers (or other ellipse-based features) to keypoint-only data such as flyllm/APF simulated tracks. All quantities in mm.",
        "units": "mm",
        "axis_convention": "quarter-axis (JAABA/Ctrax): full major axis length = 4*a_mm, full minor = 4*b_mm",
        "center": {"method": "midpoint", "keypoints_apt": ["ant_head", "pos_abdomen"],
                   "keypoints_flyllm": ["antennae_midpoint", "tip_abdomen"]},
        "orientation": {"method": "atan2(head - tail)", "keypoints_apt": ["ant_head", "pos_abdomen"],
                        "keypoints_flyllm": ["antennae_midpoint", "tip_abdomen"]},
        "a_mm": {"model": "linear", "predictor": "distance(head, tail)",
                 "keypoints_apt": ["ant_head", "pos_abdomen"],
                 "keypoints_flyllm": ["antennae_midpoint", "tip_abdomen"],
                 "slope": round(float(ma), 4), "intercept": round(float(ca), 4)},
        "b_mm": {"model": "linear", "predictor": "distance(left_thorax, right_thorax)",
                 "keypoints_apt": ["left_thorax", "right_thorax"],
                 "keypoints_flyllm": ["left_front_thorax", "right_front_thorax"],
                 "slope": round(float(mb), 4), "intercept": round(float(cb), 4)},
        "fit": {"dataset": "flybubble_social", "explist": os.path.basename(EXPLIST),
                "script": "jaaba_detect/test/fit_ellipse_coeffs.py",
                "method": "least-squares, stratified over 6 genotype families / rigs A-D / eras 2019-2023",
                "frame_stride": FRAME_STRIDE, "n_experiments": nfit_exps, "n_fly_frames": int(nfit),
                "heldout": {"n_experiments": ntest_exps, "n_fly_frames": int(yat.size),
                            "note": "disjoint experiments, not used in the fit",
                            "residual_mm": {"a_mm": resid_stats(xat, yat, ma, ca),
                                            "b_mm": resid_stats(xbt, ybt, mb, cb)}}},
      }
    }
    with open(out, "w") as fh:
        json.dump(spec, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {out}")


def main():
    fit_exps, test_exps = select()
    print(f"FIT experiments ({len(fit_exps)}):")
    for e in fit_exps: print("   ", os.path.basename(e))
    print(f"TEST experiments ({len(test_exps)}, disjoint):")
    for e in test_exps: print("   ", os.path.basename(e))

    print("\n== collecting FIT ==")
    fH, fA, fW, fB = pool(fit_exps, "fit")
    print("== collecting TEST ==")
    tH, tA, tW, tB = pool(test_exps, "test")

    # fit a_mm ~ head-tail dist
    xa, ya = clean(fH, fA, HTD_LO, HTD_HI, A_LO, A_HI)
    ma, ca = linfit(xa, ya)
    xb, yb = clean(fW, fB, THW_LO, THW_HI, B_LO, B_HI)
    mb, cb = linfit(xb, yb)

    print("\n=== NEW COEFFICIENTS (fit on FIT set) ===")
    report("a_mm ~ head_tail", xa, ya, ma, ca)
    report("b_mm ~ thorax_wid", xb, yb, mb, cb)

    print("\n=== GENERALIZATION on held-out TEST set ===")
    xat, yat = clean(tH, tA, HTD_LO, HTD_HI, A_LO, A_HI)
    xbt, ybt = clean(tW, tB, THW_LO, THW_HI, B_LO, B_HI)
    report("a_mm (new coeffs)", xat, yat, ma, ca)
    report("b_mm (new coeffs)", xbt, ybt, mb, cb)

    print("\n=== old NorpA-only coeffs on TEST set (baseline) ===")
    report("a_mm (old 0.2371,0.0676)", xat, yat, 0.2371, 0.0676)
    report("b_mm (old 0.1486,0.1484)", xbt, ybt, 0.1486, 0.1484)

    print("\n_A_FIT = (%.4f, %.4f)" % (ma, ca))
    print("_B_FIT = (%.4f, %.4f)" % (mb, cb))

    write_json(ma, ca, mb, cb, ya.size, xat, yat, xbt, ybt,
               len(fit_exps), len(test_exps))


if __name__ == "__main__":
    main()
