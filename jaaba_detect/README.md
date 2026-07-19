# jaaba_detect — Python port of JAABADetect.m

Applies a trained JAABA behavior classifier (a `.jab` file) to tracking data and
produces per-frame, per-animal behavior scores — **without MATLAB at inference
time**. It runs three ways: on a full experiment directory, on the MABe registered
arrays (e.g. `Xtest1_v3.mat`), and directly on **flyllm/APF keypoint tracks such as
the ground-truth and simulated tracks from `agent_fly.py`**.

Validated end-to-end against the real `JAABADetect` on three classifiers —
`chase_apt.jab`, `wingextension_apt.jab`, `courtship_v2pt3_apt.jab` (exported mats
ship alongside the code).

Dependencies: numpy, scipy, h5py, hdf5storage (for reading v7.3 `.mat`); opencv/torch
available but unused. No pandas. Runs e.g. in `conda activate APT`.

## The three ways to run

```python
import mabe_adapter as M     # (run from inside jaaba_detect/, or add it to sys.path)

# (A) MABe dataset arrays  -- e.g. one video of Xtest1_v3.mat
Xv, Xnames, ids, _ = M.load_X_video("Xtest1_v3.mat", videoidx=1)
res = M.jaaba_detect_from_X(Xv, Xnames, "chase_apt.classifier.mat", pxpermm=18.9, ids=ids)

# (B) flyllm / APF keypoint tracks -- e.g. agent_fly.py simulate() output (see below)
res = M.jaaba_detect_from_track(pred_track, "chase_apt.classifier.mat")

# (C) a full experiment directory (registered_trx.mat + apt.trk)
import detect
res = detect.jaaba_detect(expdir, "chase_apt.classifier.mat")
```
Every one of these returns the same dict: per-fly `scores` (raw), `postprocessed`
(the behavior label after hysteresis + small-bout removal), `t0s`/`t1s` (bout
intervals, absolute frames), `tStart`/`tEnd`, and `scoreNorm`. `detect.save_scores`
writes a JAABA-style `allScores.mat`.

## Running on the dataset and on `agent_fly.py` simulations

Both the MABe `X` arrays and the flyllm/APF `track` are per-frame, per-fly, mm
coordinates in the registered arena frame. `mabe_adapter` builds JAABA's trajectory
representation from either and runs the exact same feature → classifier pipeline.

### On the dataset (`X<split>_v3.mat`)
The `X` array is `(nframes, nflies, nfeat)` per video with pose
(`x_mm,y_mm,cos_ori,sin_ori,maj_ax_mm,min_ax_mm`) + the 21 APT keypoints. The adapter
maps columns to APT landmark order by name, converts pose (`a_mm=maj_ax_mm/4`,
`theta=atan2(sin_ori,cos_ori)`), and rebuilds a pixel frame at `pxpermm` (JAABA's raw
social `dist`/`ddist` features are in pixels). `pxpermm` is tight across the FlyBubble
dataset (18.86–19.09), so `18.9` is a good default; pass an experiment's own value for
an exact match.

```python
import mabe_adapter as M
Xv, Xnames, ids, _ = M.load_X_video("Xtest1_v3.mat", videoidx=1)   # video 1 = NorpA...T081150
res = M.jaaba_detect_from_X(Xv, Xnames, "courtship_v2pt3_apt.classifier.mat",
                            pxpermm=18.9, ids=ids)
for i, s in enumerate(res["scores"]):
    print(f"fly {i}: {len(res['t0s'][i])} courtship bouts")
```

### On simulations / ground truth from `agent_fly.py`
`apf.simulation.simulate()` returns `gt_track, pred_track` of shape
`(nagents, nframes, nkpts, 2)` — mm keypoints in flyllm order (`[...,0]`=x, `[...,1]`=y).
Feed either straight in:

```python
from apf.simulation import simulate
import mabe_adapter as M

gt_track, pred_track = simulate(dataset=train_dataset, model=model, track=track,
                                pose=pose, identities=flyids, agent_ids=agent_ids, ...)
# gt_track / pred_track: (nagents, nframes, 19, 2) mm keypoints, flyllm order

pred = M.jaaba_detect_from_track(pred_track, "chase_apt.classifier.mat")  # scored simulation
gt   = M.jaaba_detect_from_track(gt_track,   "chase_apt.classifier.mat")  # scored ground truth
```

flyllm tracks carry only 19 keypoints and no FlyTracker ellipse, whereas the
classifiers use the 2 **outer-wing** landmarks and ellipse features (`a_mm`, `ecc`,
`area`, `dnose2ell`, …). The adapter therefore **reconstructs** what's missing from the
keypoints (see next section). Cost of that reconstruction, measured against the exact
dataset path on the same video: **behavior agreement 99.90 % (chase) / 99.95 %
(wingext) / 99.99 % (courtship)**.

If your track uses a different keypoint set, pass `kpt_names=[...]`; the adapter matches
by APT-landmark or flyllm name.

## Ellipse reconstruction from keypoints (`ellipse_from_keypoints.json`)

For keypoint-only inputs the body ellipse is rebuilt as: center = head/abdomen
midpoint, `theta = atan2(head − tail)`, and the quarter-axes from body extent —

```
a_mm = 0.2386 · ‖ant_head − pos_abdomen‖ + 0.0556      (head-tail distance)
b_mm = 0.3445 · ‖left_thorax − right_thorax‖ + 0.0008  (thorax width)
```

The coefficients live in **`ellipse_from_keypoints.json`** (loaded at runtime by
`mabe_adapter`; also structured to drop into an APF config, with both APT and flyllm
keypoint names). They were fit by least squares over 12 flybubble experiments (~31k
fly-frames spanning 6 genotype families / rigs A–D / eras 2019–2023) and **validated on
12 disjoint held-out experiments**: residuals `a_mm` ≈ 0.0075 mm, `b_mm` ≈ 0.0087 mm
(~1 % of `a_mm`, ~3.5 % of `b_mm`). Center/orientation recover to ~0.05 mm / ~0.01 rad.
Refit with `test/fit_ellipse_coeffs.py` (rewrites the JSON); override per call with
`a_fit=(m,c)` / `b_fit=(m,c)`.

## One-time classifier export (MATLAB)

The `.jab` classifier is a MATLAB MCOS object scipy can't read, so each jab is exported
once to a plain `.mat`:

```bash
matlab -nodisplay -batch "addpath('jaaba_detect'); \
  export_classifier('chase_apt.jab','chase_apt.classifier.mat')"
```

**Nothing classifier-specific is hard-coded**: stump count/`dim`/`dir`/`tr`/`alpha`,
which features/stats/transforms/radii/offsets are needed, `scoreNorm`, post-processing
params, APT skeleton geometry, and per-frame params (`fov`, `nbodylengths_near`,
`max_dnose2ell_anglerange`) are all read from the exported classifier at runtime.

## Modules

| file | role |
|------|------|
| `mabe_adapter.py` | run a classifier on MABe `X` arrays (`jaaba_detect_from_X`) or flyllm/APF keypoint tracks (`jaaba_detect_from_track`); keypoint→APT mapping + ellipse reconstruction |
| `detect.py` | run on an experiment dir (`jaaba_detect`) / a prebuilt trajectory (`jaaba_detect_traj`); JAABA-style `allScores.mat` output |
| `jab_io.py` | load the exported classifier; parse `featureNames[dim]` → `(pff, stat, trans, radius, offset, extra)` descriptors |
| `trx_io.py` | read the ellipse trx (v5 **or** v7.3) and APT keypoints (v7.3), frame-aligned by `off` |
| `window_features.py` | all 12 JAABA window stats + none/abs/flip/relative transforms + `(radius,offset)` alignment, matching `*WindowCore.m` |
| `perframe_apt.py` | the 6 APT keypoint families: body, global, pair, triad, social, socialpair |
| `perframe_ellipse.py` | the full trx/ellipse per-frame lexicon (single-fly kinematics incl. center-of-rotation, and all inter-animal features) |
| `classify.py` | `myBoostClassify` + hysteresis/filter + small-bout removal + bout intervals |
| `matio.py` | read v5/v7.3 `.mat` (used by `load_X_video` and the test harnesses) |
| `export_classifier.m` | one-time MATLAB export of a jab's classifier + aptInfo + per-frame params to a plain `.mat` |
| `ellipse_from_keypoints.json` | the keypoint→ellipse reconstruction coefficients |

## Feature coverage

**Window stats — complete.** All 12 JAABA stats, each with every transform JAABA emits:
`mean`, `min`, `max`, `std`, `change`, `harmonic`, `diff_neighbor_{mean,min,max}`,
`zscore_neighbors`, `prctile` (none/abs/flip/relative; flip & relative multiply by
`sign(x)`), `hist` (only `none` — JAABA's abs/flip/relative hist blocks are commented
out in `ComputeHistWindowFeatures.m`). Includes radius-0 windows.

**Trx / ellipse per-frame — complete** (all 56 features JAABA can emit from ellipse
tracking): single-fly kinematics/shape (`a_mm`/`b_mm`/`area`/`ecc`, `da`/`db`/`darea`/
`decc`, `dtheta`/`absdtheta`, `phi`/`dphi`/`yaw`/`absyaw`/`phisideways`, `velmag`/
`velmag_ctr`/`velmag_nose`/`velmag_tail`, `du_ctr`/`du_tail`/`dv_ctr`/`dv_tail`), the
center-of-rotation family (`corfrac_maj`/`corfrac_min`, `velmag`, `du_cor`/`dv_cor`/
`absdv_cor`/`flipdv_cor`), and all inter-animal features (`dcenter`/`ddcenter`,
`dnose2ell`/`dnose2tail`/`dell2nose`, `anglesub`/`danglesub`, `nflies_close`,
`closestfly_*` identities, `dnose2ell_angle_*`, and the closest-fly base features
`veltoward`/`magveldiff`/`absthetadiff`/`absphidiff`/`anglefrom1to2`/`absanglefrom1to2`
× types anglesub/nose2ell/center/nose2tail).

**APT keypoint per-frame** — body/global/pair/triad/social/socialpair, all comps used by
the chase / wingextension / courtship jabs.

### Not implemented (raise `NotImplementedError` — a loud failure, never a silent wrong score)
- APT comps: `dphi`, `dvelmag`, `u1/v1/u2/v2` (pair), and non-`velmag` `global` comps.
- Non-trx per-frame families: wing features, ROI (`dist2roi2`/`angle2roi2`), spacetime/
  HOGHOF (`st_*`), legacy `dapt*2ctr`/`2ell`, and `dist2wall` / arena-landmark features.

Each is a bounded add-and-validate job if a future jab needs it.

## Validation

Component checks against MATLAB (`test/` harnesses run the real JAABA code / arithmetic):

| component | check | worst \|diff\| |
|-----------|-------|:---:|
| classifier extraction | 100 stumps, params, scoreNorm, postproc | exact |
| window features (all 12 stats, all transforms) | vs `Compute*WindowFeatures` | 2e-14 |
| full trx/ellipse lexicon (56 features) | vs JAABADetect's cached `perframe/*.mat` | 1.4e-10 |
| APT per-frame (all 6 families) | vs `compute_apt.m` | 4e-12 |
| classifier core (`myBoostClassify`+`PostProcess`) | on random X | 7e-15 |

### End-to-end vs real JAABADetect
On `BVAgg_JRC_SS36564_RigA_20210923T080748` (9 flies, 527,103 frames), scratch dir
symlinking the inputs so JAABA's output stays out of shared data. `scoreNorm` exact for
all three; residual confined to the ±2π seam frames below:

| classifier | scoreNorm | behavior agreement | sign flips |
|-----------|:---:|---|:---:|
| `chase_apt` | 6.6514 | 99.975 % | 134 |
| `wingextension_apt` | 14.7282 | **100.000 %** | 0 |
| `courtship_v2pt3_apt` | 49.5395 | 99.971 % | 153 |

Reproduce: `test/run_jaabadetect_multi.m` (ground truth) + `test/check_multi.py`.

### Dataset & simulation adapters
`NorpA_JRC_SS36564_RigB_20210903T081150` is both video 1 of `Xtest1_v3.mat` and a real
experiment, so the adapters are checked against the (JAABADetect-validated)
experiment path:

| path | check | behavior agreement |
|------|-------|---|
| `jaaba_detect_from_X` | vs real-experiment `detect` on NorpA | chase 99.9985 %, wingext & courtship **100 %** |
| `jaaba_detect_from_track` | flyllm-style input vs exact `from_X` | chase 99.90 %, wingext 99.95 %, courtship 99.99 % |

Reproduce: `test/validate_adapter.py`, `test/validate_flyllm.py`.

### Known ±2π degeneracies (physically identical, differ only in wrap)
- **`angleonclosestfly`**: from the ellipse-parameter of the nearest sampled ellipse
  point (`linspace(0,2π,20)`); at the seam MATLAB and numpy break a sub-ULP tie
  differently → ±2π at ~0.5 % of frames.
- **`dphi`** = `modrange(diff(phi))/dt`: `phi` is `atan2`-based, so a 1-ULP difference
  flips the wrap by ±2π/dt at a velocity reversal.

Both are inherent to JAABA's own features; the port matches modulo 2π. The residual
end-to-end differences trace to a stump whose `angleonclosestfly` window value sits
exactly on its threshold, where JAABADetect's block-based prediction path differs ~0.007
from the standalone computation the port reproduces — internal to JAABA, not a port error.

### Bugs found and fixed via validation
- `trx_io` must handle mixed formats — `registered_trx.mat` is often MATLAB v5 (scipy)
  while `apt.trk` is v7.3 (h5py).
- "drop-last-frame" closest-fly features (`veltoward`, `absphidiff`, `magveldiff`) must be
  sized to the last assigned frame, as MATLAB's `data{i}(idx)=…` does — not `zeros(nframes)`.
- `velmag_tail` used a truthy `-1` flag (both nose and tail took the nose position).
- `diff_neighbor_mean` applies its transform *after* the difference (flip = `·sign(x)`),
  unlike `min/max`.
- `prctile/flip` and `prctile/relative` multiply by `sign(x)`; the MATLAB param is
  `'prctile'` (singular), which had silently produced 0 features until caught.

## Notes / assumptions
- Frames are 1-indexed (JAABA convention); a fly's data for absolute frame t is at array
  index t + off, off = 1 − firstframe.
- MABe `X` / flyllm tracks are mm in the registered arena frame; the adapter rescales to a
  pixel frame by `pxpermm` (default 18.9) so pixel-unit APT features match training. Every
  APT feature is translation/rotation-invariant, so only the scale matters.
- Packed `X` columns are split into separate flies by `ids`; flyllm agents split on
  contiguous non-NaN runs.
- Social "closest fly" uses all same-ROI other flies; ROI defaults to a single group
  (the FlyBubble arena). Pass `roi=` for multiple arenas.
- Per-frame vectors keep their natural length (nframes or nframes−1); the frame grid is
  NaN-padded before window features, exactly as JAABA does.
