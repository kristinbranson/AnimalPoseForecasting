# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: transformer
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Running JAABA behavior classifiers from APF
#
# `jaaba_detect` is a submodule of APF. Register it once with
# `pip install -e .` (from the APF repo root) -- then `import jaaba_detect` works
# from any notebook, no need to be inside the `jaaba_detect/` directory.
#
# It applies a trained JAABA classifier -- **any** classifier you've exported to a
# plain `.classifier.mat` (via `export_classifier.m`); nothing is bundled -- to
# tracking data and returns per-frame, per-animal behavior scores. Two ways to call
# it here:
#   1. on the **MABe dataset arrays** (`X<split>_v3.mat`), and
#   2. on **`agent_fly.py` tracks** -- the ground-truth and simulated keypoint
#      tracks from `apf.simulation.simulate()`.

# %%
import numpy as np

import jaaba_detect as jd

# Point this at YOUR exported classifier .mat -- any path on your filesystem.
CLASSIFIER = "/path/to/your/chase_apt.classifier.mat"

print("ellipse-from-keypoints coeffs:", jd.load_ellipse_coeffs())

# %% [markdown]
# ## 1. On the dataset (`X<split>_v3.mat`)
#
# `load_X_video` pulls one video out of a split file as `(nframes, nflies, nfeat)`
# mm-registered pose + 21 APT keypoints. `jaaba_detect_from_X` maps the columns to
# APT order, rebuilds the pixel frame at `pxpermm` (18.9 for the FlyBubble), and
# runs the classifier.

# %%
XMAT = "/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/Xtest1_v3.mat"

Xv, Xnames, ids, frames = jd.load_X_video(XMAT, videoidx=1)
print("video 1:", Xv.shape, "(nframes, nflies, nfeat)")

res = jd.jaaba_detect_from_X(Xv, Xnames, CLASSIFIER,
                             pxpermm=18.9, ids=ids, verbose=False)
for i, s in enumerate(res["scores"]):
    print(f"  fly {i}: {len(res['t0s'][i])} chase bouts, "
          f"mean score {np.nanmean(s):+.2f}")

# %% [markdown]
# `res` has, per fly: `scores` (raw), `postprocessed` (the behavior label after
# hysteresis + small-bout removal), `t0s`/`t1s` (bout start/end frames), and
# `scoreNorm`. `jd.save_scores(res, "scores_chase.mat")` writes a JAABA `allScores.mat`.

# %% [markdown]
# ## 2. On `agent_fly.py` tracks (ground truth & simulated)
#
# `simulate()` returns `gt_track, pred_track` of shape `(nagents, nframes, nkpts, 2)`
# -- mm keypoints in flyllm order. Feed either straight into
# `jaaba_detect_from_track`; it reorders the 19 flyllm keypoints to APT order and
# reconstructs the ellipse (a/b/center/orientation) from the keypoints.
#
# Run this after the `simulate(...)` cell in `agent_fly.py` (same kernel/variables):

# %%
# from apf.simulation import simulate
# gt_track, pred_track = simulate(dataset=train_dataset, model=model, track=track,
#                                 pose=pose, identities=flyids, agent_ids=agent_ids,
#                                 track_len=3000 + config['contextl'] + 1,
#                                 burn_in=config['contextl'],
#                                 max_contextl=config['contextl'],
#                                 start_frame=1000)
#
# pred = jd.jaaba_detect_from_track(pred_track, CLASSIFIER)
# gt   = jd.jaaba_detect_from_track(gt_track,   CLASSIFIER)
# print("simulated chase bouts per agent:", [len(t) for t in pred["t0s"]])
# print("ground-truth chase bouts per agent:", [len(t) for t in gt["t0s"]])

# %% [markdown]
# ### Self-contained demo of the track path (no model needed)
#
# To show the `from_track` call runs, build a flyllm-order track out of the dataset
# keypoints we already loaded, then score it the same way `simulate()` output would
# be. We use the flyllm keypoint order and name map from the package -- no need to
# redefine them (the order comes from `flyllm.config.keypointnames`).

# %%
col = {n: i for i, n in enumerate(Xnames)}
names = jd.flyllm_keypoint_names()                 # authoritative flyllm order
nfr, nfl, _ = Xv.shape
track = np.full((nfl, nfr, len(names), 2), np.nan)
for k, name in enumerate(names):
    ap = jd.FLYLLM_TO_APT_NAME[name]               # flyllm kpt -> APT/X column
    track[:, :, k, 0] = Xv[:, :, col[f"{ap}_x_mm"]].T
    track[:, :, k, 1] = Xv[:, :, col[f"{ap}_y_mm"]].T
print("flyllm-order track:", track.shape, "(nagents, nframes, 19, 2)")

sim = jd.jaaba_detect_from_track(track, CLASSIFIER, verbose=False)
print("chase bouts per agent (from track):", [len(t) for t in sim["t0s"]])
