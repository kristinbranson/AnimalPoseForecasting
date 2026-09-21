# jaaba_sim.py: JAABA behavior scores on simulated and real fly tracks

`jaaba_sim.py score` writes one `jaaba_scores_<model>_<classifier set>.npz` per
forecasting model to `/nrs/branson/AnimalPoseForecasting/jaaba_scores/`. This describes
those files as produced with the `r5nowingtip` classifier set.

Each `jaaba_scores_<model>_r5nowingtip.npz` holds per-frame JAABA scores for **chase**,
**wing extension** and **courtship** on one forecasting model's cached open-loop
simulations, and on the real tracks over exactly the same flies and frames. Comparing the
two sides shows how realistic the simulated behavior is, and how that changes as a
simulation runs further from its real prompt.

## Files

In `/nrs/branson/AnimalPoseForecasting/jaaba_scores/`, generated 2026-09-21:

| file | model | simulation windows | prompt frames | flies simulated per window | frames on the frame axis | simulated fly-frames |
|---|---|---|---|---|---|---|
| `jaaba_scores_ref_r5nowingtip.npz` | ref | 1476 | 512 | 3–7 | 810,158 | 3,834,368 |
| `jaaba_scores_alldata_r5nowingtip.npz` | alldata | 5342 | 512 | 9–11 | 2,793,516 | 27,034,112 |
| `jaaba_scores_binall_r5nowingtip.npz` | binall | 1447 | 512 | 3–7 | 810,158 | 3,745,280 |
| `jaaba_scores_bodycentric_r5nowingtip.npz` | bodycentric | 1476 | 512 | 3–7 | 810,158 | 3,834,368 |
| `jaaba_scores_nobin_r5nowingtip.npz` | nobin | 1476 | 512 | 3–7 | 810,158 | 3,834,368 |
| `jaaba_scores_predpose_r5nowingtip.npz` | predpose | 1476 | 512 | 3–7 | 810,158 | 3,834,368 |
| `jaaba_scores_short_r5nowingtip.npz` | short | 1476 | **64** | 3–7 | 810,158 | 3,834,368 |

Every model predicts 512 frames per window.

- **rawkp is missing.** Its model was trained, and its simulations generated, with at
  most 10 flies per arena, via a fly-elimination step in `experiments/flyllm.py` that is
  currently disabled (`if False:`). Loading its ground truth today gives 11 flies and an
  input width (380) that does not match the checkpoint's normalization (342), so loading
  fails.
- alldata's config covers all flies rather than courting males, so its real behavior
  rates are much lower than the other models'; compare it through sim/real ratios.

## What each file holds

Every array is indexed `[agent, frame]`: 11 agent slots, one per fly in the arena, by
every frame of the model's evaluation track (see *Axes* below). `<b>` stands for one of
the three behaviors: `jaaba_chase`, `jaaba_wingext`, `jaaba_courtship`.

The same stretches of data were scored twice:

- **real side (`gt_…`):** the real tracking, every fly real.
- **simulated side (`sim_…`):** the same stretches, with the flies the model simulated
  replaced by the model's predictions. Flies that were not simulated stay real.

Only frames inside a simulation window's predicted stretch are filled in (see *Which
frames are filled in*). Everywhere else, scores are NaN, labels are 0 and masks are
`False`.

| key | type | value at `[agent, frame]` |
|---|---|---|
| `sim_frame` | float32 | How long this fly had been simulated at this frame. `k` = 1–512: the model had been predicting this fly on its own for `k` frames since its real prompt ended (1 = the first predicted frame). `0`: this fly is real here, either because it was not simulated in this window or because the frame is outside every window. |
| `gt_scores_<b>` | float32 | JAABA's score for behavior `<b>` for the **real** fly at this frame. Positive means the classifier says the fly is doing the behavior; larger means more confident. NaN where nothing was scored. |
| `gt_behavior_<b>` | float32 | The real fly's predicted label: `1.0` if the classifier says it is doing `<b>` (score > 0), otherwise `0.0`, including where nothing was scored. |
| `sim_scores_<b>` | float32 | The same score on the **simulated** side. For a simulated fly (`sim_frame > 0`) it scores the model's predicted fly; for a fly that was not simulated it scores the real fly, among simulated neighbors. |
| `sim_behavior_<b>` | float32 | The simulated side's predicted label, as `gt_behavior_<b>`. |
| `gt_scored_<b>`, `sim_scored_<b>` | bool | `True` where a score exists: the frame is inside a window's predicted stretch and this agent slot holds a tracked fly there. Same as `~np.isnan(<side>_scores_<b>)`. In these files the real and simulated masks are identical, and identical across behaviors. |

**Score scale.** Scores are raw JAABA classifier output: the sum, over the classifier's
200 boosted decision stumps, of each stump's ± weight. To put them on JAABA's normalized
display scale, divide by the classifier's `scoreNorm`: 4.948 chase, 12.551 wingext,
33.610 courtship.

**Labels** are exactly `score > 0`. These classifiers' post-processing (hysteresis
thresholds 0, minimum bout 1 frame) changes nothing.

**Example.** Suppose `sim_frame[3, 1000] == 17`, `sim_behavior_jaaba_chase[3, 1000] == 1`
and `gt_behavior_jaaba_chase[3, 1000] == 0`. Then at frame 1000 the fly in slot 3 had been
simulated for 17 frames; the simulated fly is chasing, and the real fly at that moment
was not.

## Which frames are filled in

Each cached simulation window holds one **tracklet** (a contiguous portion of one agent's
trajectory) per simulated fly: a real prompt (512 frames; 64 for `short`), then 512
frames the model predicted open loop. The other flies in the arena stay real
throughout. The windows never overlap in time, so every fly-frame comes from at most one
window.

- **Only the 512 predicted frames of each window are stored**, for every fly in the
  arena. The prompt frames are scored too, as context for JAABA's windowed features, but
  not stored.
- **Within a stored stretch:**
  - the flies the model simulated have `sim_frame` 1–512;
  - the other flies have `sim_frame == 0`. They are real on both sides, but their
    simulated-side scores can still differ from the real side, because their neighbors
    were simulated and JAABA's social features depend on the neighbors.
- **For the real-vs-simulated comparison, select `sim_frame > 0`.** Every simulated
  fly-frame was scored, so no mask is needed. For the flies that were not simulated, use
  a `*_scored_<b>` mask, since `sim_frame == 0` alone cannot tell a real fly inside a
  window from an empty frame.

## Axes: agents and frames

The arrays index the model's evaluation track exactly as `jaaba_sim.py`'s
`load_ground_truth(configfile, modelfile)` builds it: the MABe **test2** split, with the
model config's category filter, 19 keypoints, and **flip augmentation on**. Flip
augmentation appends a mirror-image copy of the data, doubling the frame axis, and about
half of each model's simulation windows lie in that mirrored half.

Fly identities, video names and video frame numbers are **not** stored. To map an
`(agent, frame)` back to a video and fly, rebuild the track with the same config via
`load_ground_truth`.

## How the scores were computed

- **Segments.** For each window, the segment `[start_frame − prompt, start_frame + 512)`
  of every agent is taken from the real track: 1024 frames, or 576 for `short`.
  - **Simulated side:** the simulated agents are replaced by their tracklets from the
    window file, whose prompt frames equal the real prompt (verified for every window).
  - **Real side:** the same segment, all real.
  - Each segment is scored as its own trajectory with all agents together, using the
    Python port of `JAABADetect` in `jaaba_detect`. That port reproduces MATLAB JAABA's
    labels when given the same inputs.
- **Keypoints → JAABA inputs.** Both sides go through the identical path:
  - 19 keypoints (the forecasting models' set). JAABA's outer-wing landmarks (APT 19,
    21) are filled with copies of the mid-wing points (18, 20).
  - The body ellipse is **rebuilt from keypoints**, since the tracks have no tracker
    ellipse:
    - center = midpoint of head (antennae midpoint) and abdomen tip
    - θ = direction from abdomen tip to head
    - a = 0.2386 × head–abdomen-tip distance + 0.0556 mm
    - b = 0.3445 × left–right front-thorax distance + 0.0008 mm (quarter axes)
  - Pixels per mm = 18.9, frame rate = 150 fps.
- **Relative features.** JAABA's `relative` window features take their percentile bins
  from each scored segment. On the simulated side that segment is half real (the
  prompt).

## Classifiers (`r5nowingtip`)

From `/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/`:
`chase_apt_r5_nowingtip`, `wingextension_apt_r5_nowingtip` and
`courtship_v2pt3_apt_r5_nowingtip` (`.jab` trained in the JAABA UI, exported to
`.classifier.mat`).

- trained on **all** labeled data, including test1/test2
- window radius 5
- 200 stumps
- the 52 APT per-frame features that reference the outer wing tips (landmarks 19, 21)
  removed before training, since simulated tracks have no wing tips

## Accuracy caveats

Held-out balanced accuracy on test1+test2, from the matching `*_r5_nowingtip_split`
classifiers, which were trained with test1/test2 held out:

| behavior | MATLAB JAABA (UI) | Python, tracker ellipse | Python, keypoint tracks (19 kp, as here) |
|---|---|---|---|
| chase | 0.863 | 0.863 | 0.823 |
| wingext | 0.915 | 0.918 | 0.837 |
| courtship | 0.962 | 0.961 | 0.831 |

- **Keypoint tracks score lower** because the body ellipse is rebuilt from keypoints.
  Several of the tracker ellipse's properties, notably its minor axis `b`, cannot be
  recovered from keypoints. Real and simulated sides go through the same path, so
  comparisons between them are like for like; absolute rates carry this error.
- **The `social_*` features still use the wing tips in training.** They take the closest
  landmark on another fly among all 21 landmarks. Here the wing tips are mid-wing copies,
  which changes 37 of the 215 per-frame features the classifiers use, on 3–6% of frames.
  Held-out accuracy moves by at most ~0.02 as a result.

## Example: real vs simulated behavior rates

```python
import numpy as np

z = np.load("/nrs/branson/AnimalPoseForecasting/jaaba_scores/jaaba_scores_ref_r5nowingtip.npz")
sim_frame = z["sim_frame"]                                     # (n_agents, n_frames)
for b in ("jaaba_chase", "jaaba_wingext", "jaaba_courtship"):
    ok = (sim_frame > 0) & z[f"gt_scored_{b}"] & z[f"sim_scored_{b}"]   # simulated fly-frames
    real = z[f"gt_behavior_{b}"][ok] > 0
    sim = z[f"sim_behavior_{b}"][ok] > 0
    print(f"{b:<16} real {100 * real.mean():5.2f}%  sim {100 * sim.mean():5.2f}%")
    # by frames since the prompt ended: 1-4, 5-8, 9-16, ..., 257-512
    edges = [1, 5, 9, 17, 33, 65, 129, 257, 513]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (sim_frame >= lo) & (sim_frame < hi)
        print(f"   frames {lo}-{hi - 1}: real {100 * (z[f'gt_behavior_{b}'][m] > 0).mean():5.2f}%"
              f"  sim {100 * (z[f'sim_behavior_{b}'][m] > 0).mean():5.2f}%")
```

Results on simulated fly-frames (simulated rate ÷ real rate):

| model | chase | wingext | courtship |
|---|---|---|---|
| short | 0.75 | 2.43 | 0.94 |
| ref | 0.38 | 1.98 | 0.63 |
| alldata | 0.48 | 3.67 | 0.59 |
| bodycentric | 0.48 | 0.52 | 0.54 |
| predpose | 0.34 | 0.50 | 0.54 |
| binall | 0.43 | 1.40 | 0.43 |
| nobin | 0.17 | 0.49 | 0.52 |

In the first few frames after the prompt, simulated and real rates agree to within a few
percent, and 93–97% of frames get the same label. Chase and courtship then decline the
longer a simulation runs, while real rates stay flat.

## Running jaaba_sim.py

```bash
conda activate transformer312
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting:/groups/branson/home/bransonk/behavioranalysis/code/APF_main
cd /groups/branson/home/bransonk/behavioranalysis/code/APF_main/notebooks

python jaaba_sim.py describe <model>                 # track and window statistics
python jaaba_sim.py score <model> --workers 20        # write jaaba_scores_<model>_<set>.npz
python jaaba_sim.py validate ref --classifiers r5nowingtipsplit --workers 24
```

- **`score`** uses the `r5nowingtip` set by default (`--classifiers` to change it).
- **`validate`** scores the real test1+test2 tracks and reports balanced accuracy against
  the dataset's labels, pooled over both splits. Use the held-out (`*split`) sets for a
  held-out measure. `--all-keypoints` scores with the real outer wing tips (21
  keypoints) instead of the 19 the simulations have.
- **Classifier sets** are `original`, `originalsplit` and `r<N>[nowingtip][split]`
  (only N = 5 exists); see `classifier_set` in `jaaba_sim.py`.
- **Order matters on PYTHONPATH:** Eyrun's tree comes first, so `apf`, `flyllm` and
  `experiments` are the code that built the cached simulations; `jaaba_detect` comes from
  `APF_main`.

Provenance:

- **`jaaba_sim.py` and `jaaba_detect`** are in `APF_main`, on the `jaaba_detect` branch
  (commit 399a31d).
- **The dataset and simulation code** (`apf`, `flyllm`, `experiments`) is imported
  read-only from Eyrun's tree, which is live and uncommitted.
- **The simulations** come from `…/eyjolfsdottire/AnimalPoseForecastingData/train_data/synthetic_test/<config>_<model>/`.
- **Memory:** each model needs ~90–115 GB (alldata ~175 GB), so run at most two at a time
  on a 503 GB machine.
