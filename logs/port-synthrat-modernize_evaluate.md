# Integrating code from synthrat and modernize_evaluate

## Issue: apply_opers_from_data applies all operations to raw arrays

`apf/dataset.py:apply_opers_from_data`

Contract change in `apply_opers_from_data` introduced in `f467eba` (the original apf port on this branch), 
not in the synthrat refactor. It breaks `apf/simulation.py:simulate`, which passes raw arrays that are
already partly processed, so the whole chain is applied to them a second time. For flies this **crashes**
— re-applying `Pose` to data already in pose space fails in `kp2feat`, `IndexError: index 1 is out of
bounds for axis 0 with size 1` — and fly simulation never reaches the one case that would be silently
wrong rather than fatal, the sensory array. The other callers
(`experiments/{flyllm,synthrat,spatial_infomax}.py`) pass `Data` objects and are unaffected.

| | how it decides which operations still need applying |
|---|---|
| `main` | the **dict key**: `get_post_operations(ops, key)` returns the operations after the one *named* like the key |
| `synthrat` | the **data's own history**: the operations after `datas[key].operations[-1]`; data with no operations gets the whole chain |

**Fix**: `apply_opers_from_data` now decides the resumption point per kind of input:

```python
if isinstance(datas[key], (np.ndarray, torch.Tensor)):
    last_applied = key                              # main's convention: the key names it
elif datas[key].operations:
    last_applied = datas[key].operations[-1].name   # what the data records
else:
    last_applied = None                             # nothing applied: the whole chain
```

"Not found" still warns and applies everything, as before. The docstring spells out all
three cases and notes that `apf/simulation.py` relies on the raw-array one.

This makes `test_input_assembly_matches_dataset` and `test_simulate_inputs_match_pipeline` pass. On a
synthetic chain, all four kinds of input reproduce the training-time values exactly: a raw array, a
`Data` whose last operation matches its key, a `Data` whose last operation does not (the
`GlobalVelocity` case), and a `Data` with no operations. Synthrat's `simulate` does not go through
this function; it calls `get_post_operations` directly.

**Side finding, not changed:** `Operation.__call__` raises `ValueError` for a
`torch.Tensor` — it accepts only `ndarray` or `Data` — on `main` too. So the `Tensor` half
of the raw-array branch only works when no operations remain to apply.

## Issue: simulate crashes on v3 data unless flip augmentation is on

`apf/io.py:load_and_filter_data`, `apf/simulation.py:simulate`

The v3 MABe files (`usertrain_v3.npz` etc., the data in current training configs) carry 21 keypoints:
the 19 the code uses, in the same order, plus `right_outer_wing` and `left_outer_wing` appended
(commented out of `flyllm.config.keypointnames`). The loader dropped the extra two only inside the
flip-augmentation branch. With `augment_flip` off, they stayed in the track, and `simulate` failed
writing the 19 keypoints rebuilt from each predicted pose back into it:
`ValueError: shape mismatch: value array of shape (1,9,2,19) could not be broadcast to indexing
result of shape (9,2,21)`. The training configs all set `augment_flip: true`, so this did not show
up there; the default config does not.

**Fix**: the loader drops keypoints beyond `keypointnames` whenever `keypointnames` is given, before
and independent of flip augmentation. This makes `test_loader_drops_extra_keypoints` pass.

## Issue: synthrat's velocity features did not match their names

`experiments/synthrat.py:make_dataset`, `apf/dataset.py:GlobalVelocity`

The fly code measures orientation with the fly facing +y in its own frame:
`flyllm.features.body_centric_kp` takes the angle of the body axis (base of thorax → front of
thorax) and subtracts π/2, so orientation = heading − π/2. `GlobalVelocity` relies on this: it rotates
each frame-to-frame displacement into the animal's frame, where forward is then +y, and reorders the
result as (forward, sideways, turn), the order its feature names `forward_velocity_1`,
`sideways_velocity_1`, `angular_velocity_1` give. Synthrat took orientation directly as the angle of
RatInABox's head-direction vector (orientation = heading), so in its frame forward was +x, and the
reordering put lateral movement in feature 0 (named forward) and forward movement in feature 1 (named
sideways). In the data, feature 1 was positive 95% of the time (median 6.4 mm per frame) and feature 0
centered on zero. The model did not depend on the names, but plots and analyses using them did.

**Fix**: synthrat now uses the fly convention. `synthrat.sensory` holds the conversion in one place,
`ORIENTATION_OFFSET = π/2` with `orientation_from_head_direction` and
`head_direction_from_orientation`, used everywhere synthrat converts: `make_dataset` (head direction →
orientation), `Sensory.apply` (orientation → head direction for the cells), `run_policy_from_burn_in`
(both ways), the trajectory plots in `synthrat/plotting.py`, and the heading arrows in
`notebooks/agent_synthrat.py`. The conversion now uses `arctan2` directly instead of
`ratinabox.utils.get_angle`, which added 1e-6 to the x component and so shifted every orientation by
up to ~1e-6 rad. `compute_velocity` in `synthrat/generate_data.py` still uses `get_angle`, but only
for differences of orientation, which the offset does not affect.

Consequences: the synthrat inputs and labels change (features 0 and 1 trade places, and the lateral
feature's sign follows the fly convention), so **synthrat models trained before this change no longer
match the dataset and need retraining or conversion** (below). The raw data files store RatInABox head-direction vectors and
are unaffected. The sensory caches (`synthrat/data/apf_cache_v*_*.pkl`) stay valid for loading: only
their firing rates are used, and those depend on the heading, not on how orientation is written. They
also store pose and velocity arrays for inspection, which are in the old convention.

This makes `test_velocity_feature_names_match_movement`, `test_inputs_and_labels_time_alignment` and
`test_sensory_matches_recorded_head_direction` pass.

**Guard against old models**: loading one of them would not fail by itself — the dataset is rebuilt
from the current code and the old weights would silently receive inputs in a different order. So the
synthrat config now records `"orientation_convention": "heading_minus_pi_over_2"`
(`synthrat.sensory.ORIENTATION_CONVENTION`), which is saved in every model's checkpoint with the rest
of the training config. `apf.io.load_model` gained a `check_state` argument, a function called on the
loaded checkpoint before anything is restored, and `notebooks/agent_synthrat.py` passes
`experiments.synthrat.check_orientation_convention` both when evaluating and when resuming training.
It raises `ValueError` if the checkpoint's config records no convention or a different one. Every
synthrat model saved before this change records none, so all of them are refused (checked on
`synthrat/models/synthrat_default_20260726T173100_bestepoch100.pth`).
`test_models_from_the_old_orientation_convention_are_refused` passes.

**Converting existing models**: because the change is a fixed rearrangement — new velocity features
(forward, right, turn) = (old 1, −old 0, old 2), firing rates unchanged — an old model can be
converted exactly instead of retrained. `synthrat/convert_orientation_convention.py`
(`python -m synthrat.convert_orientation_convention <old.pth> <new.pth>`) changes only two layers:
the velocity input layer `encoder.encoder_dict.velocity.weight` (2048 × 3: columns reordered
[1, 0, 2], the new second column negated; bias unchanged) and the output layer `decoder.weight` /
`decoder.bias` (75 rows = 3 features × 25 bins: the forward and lateral blocks swapped, the lateral
block's 25 rows reversed, because negating a value mirrors the bins). It rearranges the saved dataset
parameters to match (velocity z-score means and standard deviations, bin edges, centers and
samples), records `orientation_convention` in the config, renames the `Sensory` operation's module
from the AnimalPoseForecasting clone's `synthrat.apf_ratinabox` to `experiments.synthrat` (otherwise a
dataset cannot be rebuilt from the saved parameters in this repository), and drops the optimizer and
scheduler state (Adam's per-weight averages would need the same rearrangement; they are only needed
to resume training, and dropping them shrinks the file from 2.7 to 0.95 GB). The conversion relies
on the new dataset's parameters being the old ones rearranged, including binning being symmetric
under negation; for the real model's training data they agree to 1e-6 (the old `get_angle` offset).

Converted: `AnimalPoseForecasting/notebooks/synthrat_models/synthratdefault_20260430T003318_bestepoch100.pth`
(the full-data model; the July checkpoints in `synthrat/models` are short test runs on a few
episodes) → `synthrat/models/synthratdefault_20260430T003318_bestepoch100_flyorientation.pth`. Checked
end to end on 16 validation chunks: the old model on inputs built by the old code, and the converted
model on the same chunks built by the new code. The velocity inputs equal the old ones rearranged
(to 7e-6) and the firing-rate inputs match (6e-6); the predicted bin probabilities equal the old ones
rearranged to 3.5e-6, against differences up to 0.92 without the rearrangement. The converted file
passes `check_orientation_convention`. `test_old_model_conversion_is_exact`, which checks the
conversion's arithmetic, passes.

## Issue: several operations mishandled single-agent arrays

`apf/dataset.py:Fusion.apply`, `Fusion.invert`, `LocalVelocity.apply`, `GlobalVelocity.apply`,
`Velocity.apply`, `Velocity.invert`, `Zscore.apply`; `experiments/flyllm.py:Pose.apply`,
`Pose.invert`, `Sensory.apply`; `flyllm/features.py:compute_sensory_wrapper`; and the caller
`flyllm/plotting.py`

Every operation accepts either a batch of agents, (n_agents, n_frames, …), or one agent's array,
(n_frames, …). The convention is that each operation handles the single-agent case itself:

- if its data has no agent axis, it adds one, to the data and to the per-agent arguments it uses;
- it computes as for a batch;
- it removes the axis from its result.

The per-agent arguments keep agents on the *last* axis, the opposite of the data arrays. `isstart`,
`isdata` and fly identities are (n_frames, n_agents), so for one agent they become (n_frames, 1)
columns, while the data gains a leading axis, (1, n_frames, …).

Seven operations broke this convention, in different ways. Batches were handled correctly
throughout, and every dataset is built from batches, so no existing dataset or model is affected:
the regression check below finds the fly pipeline bit-identical to `main`. The single-agent path is
used by code that handles one example at a time:

- the debug plots, which decode one chunk's labels;
- simulation, which converts one frame of several flies, an (n_agents, n_pose_features) array, back
  to keypoints with `Pose.invert`.

**`Fusion`** applies different operations to different feature columns. `Velocity` uses it to combine
`GlobalVelocity` and `LocalVelocity`, and the fly labels use it to combine `Discretize` and
`Identity`. Instead of leaving the agent axis to those operations, it added the axis itself,
inconsistently:

- `Fusion.invert` added it to the data and to *every* keyword argument
  (`{k: v[None, ...] for k, v in kwargs.items()}`). So passing `do_sampling=False` through to its
  `Discretize` crashed on one chunk: `TypeError: 'bool' object is not subscriptable`.
- `Fusion.apply` added it to the data but not to the keyword arguments. So `Velocity.apply` on one
  agent with `isstart` gave `GlobalVelocity` a 3-D pose with a 1-D `isstart`, and failed its check
  `isstart.ndim must be position.ndim - 1`.
- Separately, `Fusion.apply` tested `elif ~isinstance(kwargs_per_op, list)`. `~` on a bool is bitwise
  NOT (`~True == -2`), which is always truthy, so a list of per-operation arguments would have been
  wrapped in a second list.

Now `Fusion` passes its data and arguments to each operation in the shape given, and `~` is `not`.

**`LocalVelocity.apply` and `GlobalVelocity.apply`** added the agent axis to `isstart` at the front,
`isstart[None, ...]` → (1, n_frames), instead of as a column.

- `set_invalid_ends` sets to NaN the movement out of the frame before each track start. It reads
  `isstart[:, agent]`, so with the wrong shape it saw only frame 0's flag.
- So for a single agent, every track start after the first was ignored, and the movement across it
  (a jump between unrelated poses) was kept. In a 10-frame example with a track restarting at
  frame 6, NaN went only to frame 9 instead of to frames 5 and 9.
- No code hit this: callers pass batches, and the one route that would have, `Velocity.apply` on one
  agent, crashed first in `Fusion.apply`.

Now they use `isstart[:, None]` → (n_frames, 1).

**`Velocity`** (`GlobalVelocity` and `LocalVelocity` combined by a `Fusion`) had two problems:

- `apply` relied on `Fusion` to add the agent axis.
- `invert` accepted a starting pose `x0` given for every frame, and took the first frame with
  `x0 = x0[0]`. That is right for one agent, (n_frames, n_features) → (n_features,). But for a batch,
  (n_agents, n_frames, n_features), it took the first *agent*'s whole trajectory, so inverting a batch
  of chunks through their stored metadata crashed.

Both now add the axis if it is missing (to the data, `isstart` and `x0`), and remove it at the end.
`invert` takes each agent's first frame with `x0[:, 0]`.

**`Zscore.apply`** subtracted `self.mean[None, None, :]`, a (1, 1, n_features) array. So one agent's
(n_frames, n_features) came back as (1, n_frames, n_features), for every z-scored input and label.
Z-scoring works feature by feature and needs no agent axis. It now subtracts the (n_features,) mean,
which broadcasts over any leading axes.

**`Pose`** (`experiments/flyllm.py`) added an agent axis for one fly but never removed it.
(`kp2feat` and `feat2kp` always work on (…, n_frames, n_flies) and return a fly axis.)

- One fly's (n_frames, 2, n_keypoints) keypoints came back from `apply` as (1, n_frames,
  n_pose_features).
- One fly's pose came back from `invert` as keypoints of shape (1, n_frames, 2, n_keypoints).
- Its docstring said one fly's identity could be a single int, but `feat2kp` could not reshape an int
  to (n_frames, 1).

Now `apply` and `invert` add the axis (to the data, and to the identity and `isdata` as (n_frames, 1)
columns, with an int identity repeated for every frame), compute, and remove it. The keypoint layout
in the docstrings is corrected to (…, 2, n_keypoints).

**The fly `Sensory.apply`** crashed on a single fly (`IndexError`). A fly's sensory input is what it
sees of the walls and of the other flies, so one fly given alone is now treated as a fly alone in the
arena:

- `Sensory.apply` adds and removes the agent axis like the other operations.
- When a fly has no other flies, `flyllm.features.compute_sensory_wrapper` gives it one untracked
  companion with NaN keypoints.
- Untracked flies already count as infinitely far away, so the other-fly vision and touch features
  take their maximum-distance value, 0. Both are `1 − min(1, mult · distance^exp)` with positive
  exponents. (The vision multiplier makes any fly at least an arena diameter away already give 0.)
- The result is the same as for a fly whose companions are all untracked. The wall features are
  unchanged.

The synthrat `Sensory` already handled one agent, since each rat's firing rates depend only on its
own position and heading.

**Callers that relied on `Pose`'s extra axis:**

- `flyllm/plotting.py` took `[0]` of each single example's inverted keypoints. That `[0]` is removed.
- `apf.simulation.simulate` passes one frame of several flies, (n_agents, n_pose_features). It wrote
  the (1, n_agents, 2, n_keypoints) result into its track by broadcasting; it now gets
  (n_agents, 2, n_keypoints) directly.

**Tests these changes make pass:**

- `test_single_agent_matches_batch` (fly): the `Fusion`, `isstart`, `Velocity` and `Pose` changes;
- `test_operations_handle_single_agent` (fly and synthrat: every operation in both pipelines, one
  agent against the batch): the `Zscore` change;
- `test_sensory_of_a_lone_fly`: the `Sensory` change.

**Still not changed:** `flyllm/plotting.py` asks for decoding without sampling with
`{'discretize': {'do_sampling': False}}`.

- `apply_inverse_operations` passes that only to an operation named `discretize`, so it never reaches
  the `Discretize` inside the fly labels' `Fusion`.
- As a result, the "true" keypoints in the debug plots are decoded by random sampling.
- With the `Fusion` fix, passing the argument through `Fusion` instead
  (`{'fusion': {'kwargs_per_op': [...]}}`) now works for one chunk.

## Added unit tests

`tests/test_fly.py` (18 tests) and `tests/test_synthrat.py` (16 tests). Run from the
repository root as `python tests/test_fly.py`, or `python -m pytest tests/test_fly.py tests/test_synthrat.py` where pytest is
installed. Each file takes under 20 s, needs no trained model or GPU, and skips its tests when their
data is not reachable.

**Fly data and features.** The fly tests use `tests/data/small_usertrain_v3.npz`, committed
with the tests: the first 2,000 frames of each of the first 5 videos of `usertrain_v3.npz` (10 fly
slots, 10,000 frames), in the same format, saved compressed (14 MB; `small_testtrain_v3.npz` is the
same extract of `testtrain_v3.npz`, 15 MB). It is built into a training dataset by
`experiments.flyllm.make_dataset` with the default fly config, no filtering by fly type, and flip
augmentation on, as in the training configs (`tests/config_fly_test.json`). Flip augmentation
appends a mirror-image copy of every fly after the original frames, giving 20,000 frames. The dataset
is built once per run and shared by all the fly tests; the whole file runs in about 15 s. The
quantities involved:

- **Keypoints**: the tracked (x, y) positions, in mm, of 19 body points per fly per frame (a fly is
  ~2.8 mm long). The v3 files carry two more, which the loader drops.
- **Pose**: 29 numbers per fly per frame, computed from the keypoints (`kp2feat`). Three are
  *global* — the position of the front of the thorax (x, y) and the body orientation. The other 26
  are *relative*: the body's shape in the fly's own frame (head position, leg and wing distances
  and angles). `feat2kp` converts a pose back to keypoints.
- **Velocity**: the movement from frame t to frame t + 1. For the global features it is the forward
  and sideways movement in the fly's own frame at t, plus the change in orientation; for the relative
  features it is simply their change. It is undefined (NaN) at the last frame of a track, which has
  no next frame.
- **Sensory**: 186 numbers per fly computed from all flies' keypoints at one frame (what the fly
  "sees" of the walls and the other flies). Each frame's sensory depends only on that frame.
- **Model inputs at frame t**: the movement *into* t (from t − 1 to t; the velocity array shifted by
  one frame), the relative pose at t, and the sensory at t. Each is z-scored: the training mean is
  subtracted and the result divided by the training standard deviation.
- **Labels at frame t**: what the model must predict, the movement *out of* t (from t to t + 1),
  z-scored. The 3 global features are then *discretized*: each value becomes a probability over 25
  bins, with the weight split between the bin containing the value and its nearest neighbor in
  proportion to where the value falls. The 26 relative features stay continuous numbers. Decoding a
  discretized value takes either the probability-weighted average of the bin centers (each bin's
  center is the median training value in it) or, as simulation does, a random sample.
- **Operations**: each processing step (keypoints → pose, pose → velocity, rolling, z-scoring,
  discretizing) is stored with its parameters (the z-score means and standard deviations, the bin
  edges), so it can be re-applied to new data or undone.
- **Sessions and chunks**: a *session* is a longest stretch of consecutive frames of one fly in
  which every input and label is defined (`Dataset.sessions`); it ends wherever the fly's track
  ends, the fly is not tracked, or a new video starts. Training examples, *chunks*, are 65-frame
  windows (the context length) cut from sessions without overlap, so a chunk never spans a gap or
  a video boundary. In the test data every fly is tracked through each 2,000-frame video, so each
  session is 1,998 frames (see test 3) and holds 30 chunks: 94 sessions, 2,820 chunks.
- **Reference dataset**: a validation dataset is built by applying the *training* dataset's
  operations, with the training parameters, to new data.
- **Saved parameters**: a saved model stores its dataset's operation parameters (`dataset_params`);
  loading the model rebuilds the operations from them.
- **Simulation** (`apf.simulation.simulate`): starting from real data for a 65-frame *burn-in*, it
  repeatedly gives the model all inputs so far, takes its predicted movement out of the last frame,
  adds it to the last pose to get the next pose, converts that pose to keypoints, computes sensory
  from all flies' keypoints, and assembles the new frame's inputs, processing them with the training
  operations.

Fly data used for testing is created with `make_dataset()`:
```
(dataset, flyids, track, pose, velocity, sensory, _, isdata, isstart, useoutputmask) = \
    fly_experiment.make_dataset(config, 'intrainfile', return_all=True, debug=False)
```

**Synthrat data and features.** The synthrat tests use the first 5 episodes (~1,300 frames,
concatenated) of the synthrat validation file, built with the default synthrat config. There is one
simulated rat. Its pose is just x, y (m) and orientation (rad); its velocity is the 3 global movement
features above; its sensory input is the firing of 128 field-of-view boundary-vector cells and 16
head-direction cells. The inputs are the movement into frame t and the sensory at t, both z-scored.
The labels are the movement out of t, with all 3 features discretized. Chunks are 16 frames. The last
frame of each episode has no movement out of it.

Synthrat data used for testing is created with `make_dataset()`, from the first 5 episodes of the
validation file (`config` from `synthrat.config.read_config()`):
```
episodes = pickle.load(open(config['invalfile'], 'rb'))     # then truncated to 5 episodes
dataset, info, pose, velocity, sensory, _, isstart = synthrat_experiment.make_dataset(
    config, config['invalfile'], return_all=True, debug=False, data=copy.deepcopy(episodes))
```
`dataset.inputs['velocity']` is `pose` through `GlobalVelocity`, `Roll` and `Zscore`;
`dataset.inputs['sensory']` is `Sensory` then `Zscore`; `dataset.labels['velocity']` is
`GlobalVelocity`, `Zscore` and `Discretize`.

**Stand-in model.** No trained model is used. The two simulation tests replace it with
`TrueLabelModel` (`tests/testing_utils.py`), which at each step outputs the true labels from the
data, as if it predicted perfectly, and records the inputs `simulate` gave it. Because simulation
decodes discretized labels by random sampling, the simulated trajectory stays close to the real one
but does not match it; the tests do not depend on it matching. They check how `simulate` builds the
model's inputs, not how good any model's predictions are.

**Tolerances.** "Exact" means agreement to 1e-9; the measured differences are ~1e-14, floating-point
rounding. Where values pass through float32 storage (keypoints, chunks, the model's inputs) the
tolerance is 1e-4, or 1e-2 where recomputing z-scored velocities from rounded keypoints amplifies the
rounding.

### Fly tests (`tests/test_fly.py`)

**1. Pose → keypoints → pose** (`test_pose_keypoint_round_trip`). Inputs: `pose` and `flyids`. Takes
every fly's pose at every frame, converts it to keypoints with `Pose.invert()` which calls
`feat2kp`, and converts those back with `Pose.apply()` which calls `kp2feat`, then compares the
recovered pose with the original `pose`. Both conversions use that fly's own body measurements,
looked up by its identity in `flyids` in the per-fly scale table stored with the `Pose` operation
(see test 2). The data's keypoints (`track`) are not used. Passes if the recovered pose equals the
original `pose` exactly, at every fly and frame (angles compared modulo 2π).

**2. Keypoints → pose → keypoints** (`test_keypoints_survive_pose_round_trip`). Inputs: `track`,
`pose` and `flyids`. `pose` was computed from `track` inside `make_dataset` by `Pose.apply()`, which
calls `kp2feat`. The test converts `pose` back to keypoints with `Pose.invert()`, which calls
`feat2kp`, and measures the distance from each reconstructed keypoint to the corresponding one in
`track`. There is a loss of information from keypoints (38-d) to pose (29-d); in particular the
lengths of various body parts are dropped. Each individual fly has its own entry in a per-fly scale
table, `scale_perfly`: its median thorax width and length, abdomen length, wing length, head width
and head height (`flyllm.features.compute_scale_perfly`). Keypoints → pose uses only the thorax
length, to place the base and the middle of the thorax on the body axis, from which the angles of
the abdomen, back legs, middle femurs and wings are measured. Pose → keypoints uses all six of the
same fly's values to rebuild the points the pose does not store.

How closely a keypoint can come back therefore depends on how the pose encodes it, so each group of keypoints 
has its own limits on the average distance and on the 99th-percentile distance:

| keypoints | how the pose encodes them | measured mean / 99th pct | limits |
|---|---|---|---|
| leg points (10) | an angle and a distance each, measured in that frame | 0 / 0 | 1e-4 / 1e-4 mm |
| antennae, eyes, front corners of the thorax | head base position and angle from the frame, with *that fly's own* median head and thorax widths and head height | 0.013 / 0.066 mm | 0.03 / 0.15 mm |
| base of the thorax | that fly's own median thorax length | 0.025 / 0.28 mm | 0.05 / 0.5 mm |
| abdomen tip | the abdomen's angle in that frame, at that fly's own median abdomen length | 0.080 / 0.63 mm | 0.15 / 1.0 mm |
| wing tips | each wing's angle in that frame, at that fly's own median wing length | 0.095 / 0.98 mm | 0.15 / 1.5 mm |

**3. Velocity → pose** (`test_velocity_round_trip`). Inputs: `velocity`, `pose` and
`dataset.sessions`. Selects contiguous frames based on the sessions in `dataset` — the stretches of
frames in which the fly is continuously tracked and every input and label is defined (see Sessions
and chunks above). For each session, it calls `Velocity.invert()` starting from the real pose at the
session's first frame (`x0=true_pose[:1]`), which adds up the frame-to-frame movements:
`GlobalVelocity.invert()` rotates forward/sideways movement back into arena coordinates and adds up
the orientation changes, and `LocalVelocity.invert()` adds up the changes in the relative features.
This reconstructs the pose at every frame of the session. Passes if the reconstructed pose at each
frame equals `pose` at that frame exactly. This would catch errors in the velocity computation, in
the direction of rotation, or in wrapping angles. Simulation does exactly this to turn predicted
movement into pose.

**4. Movement over several frames** (`test_global_velocity_future_offsets`). Inputs: `pose` (its 3
global features: x, y, orientation) and `isstart`. Calls `GlobalVelocity(tspred=[1, 3, 10]).apply()`
to compute each fly's movement from frame t to t + 1, t + 3 and t + 10 — forward and sideways in the
fly's own frame at t, and the change in orientation — at every frame t; `isstart` makes movements
that would cross into a new track NaN, and those are skipped. For each offset, applies that movement
to the pose at t as a single step with `GlobalVelocity(tspred=[1]).invert()` and compares where it
lands with the real pose at t + offset. Passes if the landing pose equals `pose` at frame t + offset
exactly, for every fly, frame and offset. This mirrors `debug_fly_example`'s check of predictions
several frames into the future. The current config predicts only one frame ahead; this covers the
multi-frame option (`tspred_global`).

**5. Labels → velocity** (`test_labels_invert_to_velocity`). Inputs: `dataset.labels['velocity']`
and `velocity`. Works on the dataset's whole label array — every fly at every frame — not on chunks:
undoing the labels only as far as velocity needs no per-frame information, so no chunk metadata is
involved (compare test 6). The labels were made from `velocity` by the operations `Zscore` and then
`Fusion`, which applies `Discretize` to the 3 global features and `Identity` to the 26 relative
ones. The test undoes them with `apf.dataset.invert_to_named(labels, 'velocity', ...)`, which calls
`Fusion.invert()` — `Discretize.invert(do_sampling=False)`, the probability-weighted average of the
bin centers, and `Identity.invert()` — and then `Zscore.invert()`, and compares the decoded movement
with `velocity`. The 26 continuous features must equal `velocity` exactly. The discretized ones
cannot: a value is recovered only to within roughly a bin. For values at least two bins from either
end, the error must be less than the width of the widest interior bin. The two end bins are excluded
because they are stretched to cover outliers, and at least half of all values must be checked so the
test cannot pass vacuously. This would catch bins being mixed up, wrong z-score parameters, or
discrete and continuous columns being confused.

**6. A chunk's labels → its pose and keypoints** (`test_labels_invert_to_chunk_pose`). Inputs:
`dataset`, `pose` and `track`. Unlike test 5, this works on one chunk in the form a model, the loss
and the plots see it, and undoes the labels further, to pose and keypoints. That needs information
only a chunk carries: the starting pose for adding up movements and the fly's identity for its body
scale. Takes the first training chunk (`dataset.get_chunk()` at
`dataset.chunk_indices[0]`) and converts it with `dataset.item_to_data()`, which attaches the
metadata the dataset stores with each chunk: the true pose and the fly's identity at each of its
frames. Undoes the label processing with `apf.dataset.invert_to_named(labels, 'pose')` —
`Fusion.invert()` (discretized features decoded by sampling), `Zscore.invert()`, then
`Velocity.invert()` starting from the stored pose at the chunk's first frame — and all the way to
keypoints with `apf.dataset.apply_inverse_operations(labels)`, which adds `Pose.invert()` using the
stored identity. This is the path the debug plots in `flyllm/plotting.py` use. Passes if the pose at
the chunk's first frame equals the true pose in `pose` exactly, the 26 relative features equal the
true ones at every frame (to float32 precision), and the keypoints at the first frame equal those
`Pose.invert()` gives for the true pose. The global features after the first frame are not checked,
because each frame's discretization error accumulates as movements are added up; test 5 bounds the
error of each step. This would catch the stored metadata being wrong or shifted relative to the
chunk (shifting it by one frame makes the test fail), or a break anywhere in the chain of inverse
operations.

**7. No undefined targets in training chunks** (`test_chunks_have_defined_targets`). Inputs:
`dataset.chunk_indices` and `velocity`. For every one of the 2,820 training chunks — which `Dataset`
cut from its sessions (`compute_sessions`, `compute_chunk_indices`) when it was built — checks that
`velocity` is defined at each of its 65 frames, i.e. that the movement each frame's label encodes
exists. This would catch the model being trained to predict an undefined movement, such as out of
the last frame of a track. (For flies those frames are also excluded by their continuous labels
being NaN; the synthrat version of this test is the one that depends on `discretize_labels`
marking undefined movement as NaN.)

**8. Features from a window of keypoints = the full dataset's** (`test_window_features_match_dataset`).
Inputs: `config`, `dataset`, `track`, `flyids`, `isstart`, `isdata`, `useoutputmask` and the scale
table. Takes the first 120 frames of the longest session (see test 3) and calls
`make_dataset(config, 'intrainfile', ref_dataset=dataset, indata=...)` with only those frames of
every fly, the window's first frame marked as the start of every track. With a reference dataset,
`make_dataset` computes `Sensory`, `Pose` and `Velocity` from the keypoints, then
`apply_opers_from_data` applies the rest of the reference's operations (`Roll`, `Subset`, `Zscore`,
`Fusion` with `Discretize`) with the reference's z-score parameters and bins — the way validation
data is built. Compares every input and label with `dataset`'s on the same frames: they must be
identical, with NaN in the same places. Two frames differ by design and are checked separately: the
window's first frame has no movement into it (no earlier frame in the window), and its last frame
has no movement out of it. This mirrors `debug_fly_example`'s comparison of an example taken from the
dataset with one built directly from keypoints. It would catch a feature depending on frames it
should not, or the reference dataset's parameters not being reused.

**9. Dataset rebuilt from saved parameters** (`test_dataset_rebuilt_from_saved_params`). Inputs:
`config` and `dataset`. Calls `dataset.get_params()` — each operation's parameters as a dict
(`Operation.to_dict()`), which is what a saved model stores — puts the result in
`config['dataset_params']`, and calls `make_dataset` again on the same data file. With
`dataset_params` set, `make_dataset` applies the operations rebuilt from those dicts with
`apply_opers_from_data_params`. Every input, label and chunk must be identical to `dataset`'s. This
would catch parameters being lost or changed when a model is saved and reloaded, which would make
the reloaded model receive inputs processed differently from those it was trained on.

**10. Inputs and labels line up in time** (`test_inputs_and_labels_time_alignment`). Inputs:
`config`, `dataset`, `pose`, `track` and `flyids`, plus the window's `isstart`, `isdata` and
`useoutputmask`. Takes the same 120-frame window. For that session's fly, replaces one relative pose
feature (the left middle femur base distance) with random values p[t] (within ±20% of the feature's
standard deviation), converts that fly's modified poses to keypoints with `Pose.invert()`, and calls
`make_dataset(..., ref_dataset=dataset, indata=...)` on the modified keypoints. Then undoes the
processing of each input and label with `apf.dataset.invert_to_named`: the pose and velocity inputs
back to before z-scoring (`Zscore.invert()`), and the labels back to raw velocity (`Fusion.invert()`
without sampling, then `Zscore.invert()`). Reading off the modified feature: the pose input at frame
t must be p[t]; the velocity input at t must be p[t] − p[t−1], the movement into t; the label at t
must be p[t+1] − p[t], the movement out of t. Passes if each of the three, read from the dataset,
equals its value computed directly from p (p[t], p[t] − p[t−1], p[t+1] − p[t]) to within 1e-4. The
test also checks that the random values change by more than 100 times that from frame to frame, so a
one-frame shift cannot go unnoticed. This mirrors `debug_fly_example`'s test that writes the frame
number into a feature. It would catch inputs or labels being off by a frame — for example, if the
velocity input were not shifted (`Roll`), the model would be given the very movement it is asked to
predict (making that change makes this test fail).

**11. Batches split back into named inputs and labels** (`test_batches_split_back_to_dataset`).
Inputs: `dataset`. Takes the first batch of 4 chunks from `apf.dataset.DataLoader(dataset)`, in
which all inputs are concatenated into one array and the labels are split into a continuous array
and a flattened discrete array, converts it to numpy (`apf.utils.convert_torch_to_numpy`), and
splits it back into named inputs (velocity, pose, sensory) and labels (velocity) with
`dataset.item_to_data()`, which calls `split_input_by_names()` and `split_output_by_names()`, as the
loss, the plots and simulation do. Each named input and label of batch item i must equal the
dataset's own array for it (e.g. `dataset.inputs['velocity'].array`) for that chunk's fly at that
chunk's frames, to float32 precision. It also rearranges the discrete labels into the shape a model
outputs (one row of bin probabilities per feature) and checks that `split_output_by_names()` gives
exactly the same named labels from them as from the batch, and checks that `dataset.get_chunk()` for
32 frames starting 16 frames into a 65-frame chunk equals frames 16–47 of the 65-frame chunk,
exactly. This would catch errors in the bookkeeping of which columns belong to which input or label,
and which are discrete.

**12. Assembling one frame's inputs the way simulation does**
(`test_input_assembly_matches_dataset`). Inputs: `dataset`, `track`, `pose`, `velocity` and
`isdata`. At 200 frames spread evenly
over the data, including the mirrored half, builds each frame's inputs the way `simulate` builds
them: the movement into the frame from `velocity`, the pose at the frame from `pose`, and sensory
recomputed from that frame's keypoints in `track` alone, with the dataset's `Sensory` operation's
`apply()` (`experiments.flyllm.Sensory`, which calls `compute_sensory_wrapper`). Passes them as raw
arrays to `apply_opers_from_data(dataset.inputs, ...)`, which must apply only the steps not yet
applied: `Roll` and `Zscore` for velocity, `Subset` (the relative features) and `Zscore` for pose,
`Zscore` for sensory. For every fly tracked at that frame (`isdata`), each processed input must
equal the dataset's stored input for that fly and frame (`dataset.inputs[key].array[fly, t]`)
exactly, with NaN in the same places. Flies not tracked at the frame are left out: the dataset sets
all their inputs to NaN, whereas sensory recomputed from one frame gives values for their
other-flies features; `simulate` only feeds the model flies that are tracked. Not every frame is
checked because recomputing sensory one frame at a time, as `simulate` does, takes ~2 ms per frame
(40 s for all 20,000). This catches `apply_opers_from_data` re-applying steps that were already
applied to a raw array, such as keypoints → pose applied to pose features, which crashes in
`kp2feat`.

**13. Simulation's inputs match the training pipeline** (`test_simulate_inputs_match_pipeline`).
Inputs: `config`, `dataset`, `track`, `pose`, `velocity`, `flyids`, `isdata` and `isstart`. Calls
`apf.simulation.simulate` from frame 1000 for every fly tracked throughout the window as a single
identity (9 of the 10): 65 burn-in frames of real data, then 20 simulated frames, with the stand-in
model, which returns the true labels taken from `dataset.get_chunk()`. Checks that no keypoints are
NaN, and that the inputs `simulate` gave the model for the 65 burn-in frames equal the dataset's
inputs for those frames (`dataset.get_chunk()`) exactly. Then runs the training processing on the
keypoints `simulate` produced — the dataset's own `Pose`, `Velocity` and `Sensory` operations, then
`apply_opers_from_data` for the rest (`Roll`, `Subset`, `Zscore`) — and compares with the inputs
`simulate` actually gave the model at each simulated frame. Passes if, at every simulated frame, the
inputs `simulate` gave the model equal these recomputed inputs to within 1e-2: `simulate` stores
keypoints as float32, and recomputing pose and z-scored velocity from the rounded keypoints gives
differences up to ~1e-3, whereas inputs processed the wrong way differ by order 1 or crash. This
would catch any inconsistency between how simulation builds inputs and how the training data was
built.

**14. Feature names** (`test_feature_names_match_dimensions`). Inputs: `dataset`. Calls
`dataset.get_input_names()` and `get_label_names()`, which prefix each input's and label's feature
names — set by each operation's `update_feature_names()` as it was applied — with its key, e.g.
`sensory__…`. These names label plots and analyses. Passes if there is exactly one name per column,
each starting with the key of the input or label it belongs to. This would catch names falling out
of step with the columns.

**15. The loader drops extra keypoints** (`test_loader_drops_extra_keypoints`). Inputs: `config`.
Calls `experiments.flyllm.load_data`, which calls `apf.io.load_and_filter_data`, on the v3 test file
twice, with `augment_flip` off and on, and checks that each time only the 19 keypoints the code uses
remain. The v3 files append two outer wing points, and `simulate` writes the 19 keypoints rebuilt
from each predicted pose back into the loaded arrays, so it crashes if they hold more. The other fly
tests run with flip augmentation on only, so this is the test that checks the loader with flipping
off.

**16. One agent gives the same result as a batch** (`test_single_agent_matches_batch`). Inputs:
`dataset.labels['velocity']`, `pose`, `velocity`, `track`, `flyids` and `isstart`. For agent 0 alone
and for all agents at once (frames 1–99, inside every fly's first session), runs `Fusion.invert` on
the labels with `do_sampling=False` passed to its `Discretize`, `Velocity.apply` with `isstart`,
`Velocity.invert` with a starting pose `x0`, `Pose.apply` with per-frame identities, and
`Pose.invert` with per-frame identities and with one fly's identity as a single int. It also inverts
the whole batch with a pose for every frame as `x0` and compares with inverting each agent alone.
Passes if, for every call, the result for agent 0 alone equals the result of the same call on the
array with all agents, indexed at agent 0 (`result_for_all_agents[0]`), exactly, with NaN in the
same places; for the whole-batch inversion, the batch result equals the stacked single-agent
results. This catches an operation that mishandles the agent axis for one agent: adding it to
arguments that have none (such as `do_sampling=False`, which crashes), adding it to `isstart` as a
row instead of an (n_frames, 1) column (track starts after the first are ignored, so NaN goes in the
wrong places), taking the first agent instead of each agent's first frame from a per-frame `x0`, or
returning an extra leading axis.

**17. Every operation handles one agent** (`test_operations_handle_single_agent`). Inputs:
`dataset`, `pose`, `velocity` and `sensory` (frames 1–99). For each input and label, starting after
the pose, velocity and sensory computations — `Roll` and `Zscore` for the velocity input, `Subset`
and `Zscore` for the pose input, `Zscore` for the sensory input, `Zscore` and `Fusion` for the
labels — applies each operation in turn to all flies and to fly 0 alone, and checks with
`testing_utils.assert_operations_handle_single_agent` that the result for fly 0 alone equals the
result of the same operation on the array with all flies, indexed at fly 0
(`result_for_all_flies[0]`): same shape, NaN in the same places, identical values; and the same for
each operation's `invert` (except `Subset`, which has none), decoding discretized values without
sampling. `Pose` and `Velocity` are covered by `test_single_agent_matches_batch`, with their
arguments. `Sensory` is left out on purpose, since a fly alone sees no other flies (test 18). An
operation that returns an extra leading axis for one agent fails here, e.g. (1, 99, 29) instead of
(99, 29).

**18. Sensory of a lone fly** (`test_sensory_of_a_lone_fly`). Inputs: `track` (frames 1–99) and the
dataset's `Sensory` operation. Computes sensory for fly 0 given alone, (n_frames, 2, n_keypoints),
and checks that it equals sensory computed for a batch containing only fly 0, (1, n_frames, 2,
n_keypoints), indexed at 0, exactly; that its wall features equal fly 0's wall features computed
from all the flies' keypoints, exactly; and that every other-fly vision and touch feature is 0, the
maximum-distance value. It also checks that with the other flies present, fly 0 does see some of
them (vision), so the zero check is not vacuous; touch is left out of that, since flies are rarely
close enough to touch within 99 frames.

### Synthrat tests (`tests/test_synthrat.py`)

**1. Velocity → pose** (`test_global_velocity_round_trip`). Inputs: `pose`, `velocity` and `isstart`
(which marks where each episode starts). For each of the 5 episodes, calls `GlobalVelocity.invert()`
starting from the real pose at the episode's first frame, which adds up the frame-to-frame
movements, rotating forward/sideways movement back into arena coordinates. Passes if the
reconstructed x, y and orientation at each frame equal `pose` at that frame exactly. Same purpose as
fly test 3.

**2. Labels → velocity** (`test_labels_invert_to_velocity`). Inputs: `dataset.labels['velocity']` and
`velocity`. The labels were made from `velocity` by `Zscore` and then `Discretize`, for all three
features. Undoes them with `apf.dataset.invert_to_named(labels, 'globalvelocity',
discretize={'do_sampling': False})` — `Discretize.invert()` without sampling, then
`Zscore.invert()` — and checks each feature against the widest-interior-bin bound, as fly test 5.

**3. No undefined targets in training chunks** (`test_chunks_have_defined_targets`). Inputs:
`dataset.chunk_indices` and `velocity`. As fly test 7, for every 16-frame chunk. Here it matters:
every synthrat label is discretized, so this depends on `discretize_labels` (called by
`Discretize.apply()`) marking an undefined movement as missing (NaN). If it put the movement into a
bin instead, the last frame of an episode could enter a chunk with a made-up target.

**4. Dataset rebuilt from saved parameters** (`test_dataset_rebuilt_from_saved_params`). Inputs:
`config`, `dataset` and the loaded episodes. As fly test 9: `dataset.get_params()` goes into
`config['dataset_params']`, and `make_dataset` on the same episodes applies the operations rebuilt
from it with `apply_opers_from_data_params`. The firing rates are recomputed (the `Sensory`
operation calls `synthrat.sensory.compute_sensory`), and RatInABox's boundary-vector cells are not
bit-for-bit reproducible (differences ~1e-7 in firing rate, ~1e-6 after z-scoring), so the rebuilt
dataset's sensory inputs must equal `dataset`'s to within 1e-4; its other inputs, its labels and its
chunks must equal `dataset`'s exactly.

**5. Dataset built with itself as reference** (`test_reference_dataset_reproduces_dataset`). Inputs:
`config`, `dataset` and the loaded episodes. Calls `make_dataset(..., ref_dataset=dataset)` on the
same episodes, so `apply_opers_from_data` applies the reference's operations with its z-score
parameters and bins — the way validation data is built. Every input and label of the rebuilt dataset
must equal `dataset`'s, and its chunks must be the same: the sensory inputs to within 1e-4, as in
test 4, the rest exactly.

**6. Cached firing rates** (`test_cached_sensory_matches_computed`). Inputs: `config`, `dataset`,
`sensory` and the loaded episodes. Computing firing rates is slow, so `make_dataset` can take a
precomputed sensory array instead (`cached_sensory_array=`). It then builds the `Sensory`
operation's record of which columns belong to which cell population (`idxinfo`, from
`rehydrate_sensory`) and its feature names (`get_all_feature_names`) without computing any firing
rates. Building with `sensory.array` as the cache must give inputs and labels equal to `dataset`'s
exactly, `get_input_names()` equal to `dataset.get_input_names()`, and a `Sensory` operation whose
`idxinfo` equals that of `dataset`'s `Sensory` operation.

**7. Inputs and labels line up in time** (`test_inputs_and_labels_time_alignment`). Inputs:
`config`, `dataset` (as reference) and the loaded episodes' environment, agent and sensory settings.
Builds a synthetic 60-frame trajectory: the rat starts at (0.3, 0.3) m facing 0.3 rad and on each
frame steps 2–8 mm forward and up to 2 mm sideways and turns up to 0.2 rad, all at random. Calls
`make_dataset(..., ref_dataset=dataset, data=...)` on it, which derives orientation from the head
direction (`orientation_from_head_direction`, fly convention) and builds the inputs and labels. From
the pose `make_dataset` built (checked exactly against the constructed trajectory, whose orientation
the test writes out as heading − π/2), computes the true movement
m[t] from each frame to the next with `GlobalVelocity.apply()`. Undoes the velocity input's z-scoring
(`invert_to_named(..., 'roll')`, i.e. `Zscore.invert()`): at frame t it must equal m[t−1] exactly.
Decodes the labels without sampling (`Discretize.invert()`, then `Zscore.invert()`): at frame t they
must equal m[t] to within the widest interior bin. The test also checks that the steps vary enough
that a one-frame shift would fail, so it cannot pass vacuously. Same purpose as fly test 10, for
discretized labels.

**8. Batches split back into named inputs and labels** (`test_batches_split_back_to_dataset`).
Inputs: `dataset`. As fly test 11. Synthrat's labels are all discrete, with no continuous part,
which exercises the branch of `split_output_by_names()` for fully discretized models.

**9. Firing rates from a single frame** (`test_single_frame_sensory_matches_trajectory`). Inputs:
`pose`, `sensory` and the `Sensory` operation in `dataset`. Synthrat's `simulate` computes each new
frame's firing rates from that frame's pose alone, with `Sensory.apply()` (`experiments.synthrat.
Sensory`, which rebuilds the cells from their stored settings and calls
`synthrat.sensory.compute_sensory`). At 200 frames spread evenly over the 1,290, compares firing
rates computed that way with `sensory`, which `make_dataset` computed over the whole trajectory:
each single-frame result must equal `sensory` at that frame to within 1e-6. (Each single-frame
computation rebuilds the RatInABox cells, ~5 ms, so not every frame is checked.) This holds because
both cell types depend only on the current position and heading; it would stop holding if a cell
type that depends on movement were added.

**10. Simulation's inputs match the training pipeline** (`test_simulate_inputs_match_pipeline`).
Inputs: `dataset`, `pose`, `velocity` and `isstart`. Calls `experiments.synthrat.simulate`, as fly
test 13: 16 burn-in frames and 20 simulated frames, starting one frame into an episode (the first
frame of an episode has no movement into it, so its velocity input is undefined). The expected
inputs come from running the dataset's own `GlobalVelocity` and `Sensory` operations on the poses
`simulate` produced, then `apply_opers_from_data` for `Roll` and `Zscore`. At every simulated frame,
the inputs `simulate` gave the model must equal these recomputed inputs to within 1e-4 (the model's
inputs are stored as float32).

**11. Feature names** (`test_feature_names_match_dimensions`). Inputs: `dataset`. As fly test 14.

**12. Velocity features match their names** (`test_velocity_feature_names_match_movement`). Inputs:
`config`, `dataset` (as reference) and the loaded episodes' settings. Builds two synthetic 60-frame
trajectories from (0.3, 0.3) m facing 0.3 rad, with no turning: one stepping 5 mm straight ahead each
frame, one stepping 5 mm straight to the left. Calls `make_dataset(..., ref_dataset=dataset,
data=...)` on each and reads `velocity`, the output of `GlobalVelocity` with its feature names. Passes
if, for the straight-ahead trajectory, `forward_velocity_1` is +5 mm at every frame and the other two
features are 0, and for the sideways one, `sideways_velocity_1` is 5 mm in magnitude (the fly
convention makes leftward movement negative) and the other two are 0. This depends on
`make_dataset` converting RatInABox's head direction to the fly convention, which `GlobalVelocity`
assumes; if orientation were taken as the heading itself, forward movement would land in
`sideways_velocity_1`.

**13. Firing rates match the recorded head direction**
(`test_sensory_matches_recorded_head_direction`). Inputs: `sensory`, `info`, `isstart` and the first
loaded episode. Computes the firing rates directly from the episode's recorded position and
RatInABox head direction with `synthrat.sensory.compute_sensory`, and compares with `sensory`, which
`make_dataset` computed from the pose — that is, after converting the head direction to orientation
and back (`orientation_from_head_direction`, then `head_direction_from_orientation` in
`Sensory.apply`). The dataset's `sensory` for the first episode must equal the directly computed
firing rates to within 1e-6, at every frame. This catches the two conversions disagreeing: if
`Sensory.apply` took the orientation itself as the heading, the cells would see a heading 90° off.

**14. Old models are refused** (`test_models_from_the_old_orientation_convention_are_refused`).
Inputs: `config`. Saves two checkpoints of a tiny stand-in model (`torch.nn.Linear(2, 2)`) with
`apf.io.save_model` to a temporary directory: one with the current `config`, one with
`orientation_convention` removed from it, as in every synthrat model saved before the fly convention
was adopted. Loads each with `apf.io.load_model(..., check_state=check_orientation_convention)`.
Passes if the first loads and the second raises `ValueError`.

**15. Old models convert exactly** (`test_old_model_conversion_is_exact`). Inputs: random stand-in
weights, and `dataset`'s z-score and bin parameters treated as if they were an old model's. Calls
`synthrat.convert_orientation_convention.convert_weights` and `convert_dataset_params`. With old
velocity features (left, forward, turn) and new ones (forward, right, turn) = (old 1, −old 0, old 2):
the converted velocity input layer applied to the rearranged z-scored velocity must equal the old
layer applied to the old one; the converted output layer's bin logits must equal the old ones with
the forward and lateral blocks swapped and the lateral block's bins reversed; the converted z-score
parameters must turn rearranged raw velocities into the rearranged z-scores; and the converted
lateral bin edges must be the old ones negated and reversed, still increasing. This checks the
conversion's arithmetic without needing a trained model.

**16. Every operation handles one agent** (`test_operations_handle_single_agent`). Inputs: `dataset`,
`pose` and `velocity`. Synthrat has one rat, so the batch is two 60-frame stretches from different
episodes, each starting one frame in. As fly test 17, for `Roll` and `Zscore` (velocity input),
`Sensory` and `Zscore` (sensory input, `Sensory` to 1e-6 for RatInABox's rounding), and `Zscore` and
`Discretize` (labels). It also checks `GlobalVelocity.apply` with an `isstart` that marks a track
start partway through, and `GlobalVelocity.invert` with a starting pose `x0`. An operation that
returns an extra leading axis for one agent fails here, e.g. (1, 60, 3) instead of (60, 3).

### Verification

- All 34 tests pass (18 fly, 16 synthrat).
- The tests catch deliberately introduced errors. With the roll on the velocity input disabled (the
  model would see the movement out of each frame), 4 fly and 2 synthrat tests fail. With the stored
  chunk pose shifted by one frame, `test_labels_invert_to_chunk_pose` fails.

### Found while writing the tests, not changed

- **The 2024 small test files are mirrored relative to the v3 data.**
  `/groups/branson/bransonlab/test_data_apf/small_intrainfile.npz` and `small_invalfile.npz` put
  each "left" keypoint on the opposite side of the body axis from the v3 files and from the fly
  hard-coded in `tests/flyllm/test_features.py` (checked with the sign of the cross product of the
  body axis, base of thorax → antennae, with the left − right offset, for the eyes, the front of the
  thorax and the front legs). `feat2kp` follows the v3 convention, so on the 2024 files it puts each
  eye and front thorax corner where its partner is, ~0.7 mm off; on them,
  `test_keypoints_survive_pose_round_trip` fails (head and front of thorax: mean 0.59 mm against a
  limit of 0.03 mm). The unit tests use v3 extracts; `tests/flyllm/` still uses the 2024 files.

**Not verified:** a real fly or mouse simulation with a trained model.

## Regression check against earlier versions of the code

`tests/regression/workloads.py`, `tests/regression/compare_with_ref.py`

The unit tests check that the code agrees with itself: round trips, time alignment, a single agent
against a batch. They cannot tell whether this PR changed what the code computes. For that, the same
seeded workloads were run under an earlier version of the code and under this PR, and every output
was compared. This lives in `tests/regression/` rather than with the unit tests because it runs two
versions of the code, writes ~1.2 GB of temporary outputs, and takes about 30 s with a GPU. Rerun it
by hand when a change should leave results unchanged:

```
python tests/regression/compare_with_ref.py main
python tests/regression/compare_with_ref.py 2b888a2 --skip-fly --synthrat-old-convention
```

**How it works.** `compare_with_ref.py <ref>` copies the code directories (`apf`, `experiments`,
`flyllm`, `synthrat`) of a git ref into a temporary directory with `git archive`. This leaves the
repository and the working tree untouched; the synthrat data, which are not in git, are linked in.
It then runs `workloads.py` twice, each time in a separate process started from one version's root
directory, so that `import apf` and the other imports pick up that version: once with the ref's code
and once with the working tree's. It compares the two sets of outputs and exits with status 1 if
they differ beyond the allowances below. Options:

- `--skip-fly`
- `--skip-synthrat` (synthrat is skipped automatically if the ref has no synthrat code)
- `--synthrat-old-convention` (below)
- `--out-dir` keeps the copied code and outputs instead of deleting them.

**Fly workload** (`workloads.run_fly`). Data: `tests/data/small_usertrain_v3.npz` with
`tests/config_fly_test.json`, as in the unit tests (10 fly slots, 20,000 frames including the flipped
copy). Each version does the following:

1. **Dataset.** Builds the training dataset with `experiments.flyllm.make_dataset`. It saves:
   - the z-scored input arrays (velocity, pose, sensory) and the label array;
   - the chunk start indices (2,820 chunks);
   - the keypoints;
   - the operation parameters (`dataset.get_params()`: z-score means and standard deviations, bin
     edges, bin centers, bin samples and the rest).
2. **Loss.** Creates a model with random initial weights (`apf.models.initialize_model`) and computes
   its training loss (`apf.models.criterion_wrapper`) on the first 10 batches in order, with dropout
   on as in training. It saves the 10 losses, plus the gradient norm of each of the 124 parameter
   tensors (weights and biases) for the first batch. No weights are updated.
3. **Simulation.** Simulates with that model (`apf.simulation.simulate`). Starting at frame 1000, the
   9 flies tracked throughout frames 1000–1164 get their real first 65 frames (the context length),
   and the model predicts the next 100 by sampling from the predicted bins. It saves the simulated
   keypoints. With random weights the motion is meaningless, but every step of the simulation loop
   runs and must give the same numbers: building inputs from keypoints, decoding and sampling the
   predictions, and converting back to keypoints.
4. **Plot.** Draws `flyllm.plotting.debug_plot_pose` for 3 examples of the first batch, and saves the
   coordinates of every line and point in the figure (30 sets).

**Synthrat workload** (`workloads.run_synthrat`). Runs `experiments.synthrat.make_dataset` on the
first 5 validation episodes (`debug=True`) with the default synthrat config: 1,290 frames and 77
chunks. It saves the z-scored velocity and firing-rate inputs, the labels, and the chunk start
indices.

**Random numbers.** Numpy's and torch's generators are reset to the same seed before each step
(dataset, model, simulation, plot). Torch runs in deterministic mode
(`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). So two versions that
compute the same thing give bit-identical outputs.

The seed must be set before the dataset is built, because fitting the bin edges draws random numbers:
`select_bin_edges` breaks ties at random, and the bin samples used for sampling are drawn at random.
In a first run without that seed, the two versions' bin edges differed and so did their losses,
although neither computation had changed.

**Results, fly: `main` vs this PR.** Each item compares `main`'s output with this PR's output for the
same step.

- **Inputs, keypoints, chunk start indices and saved operation parameters:** identical (largest
  absolute difference 0, missing values in the same places).
- **The 10 losses and the 124 gradient norms:** identical.
- **Simulated keypoints and the list of simulated flies:** identical.
- **`debug_plot_pose`:** all 30 sets of coordinates identical.
- **Labels:** identical at every frame whose movement out of the frame is defined. They differ at
  the 12,094 frames (of 200,000) where it is not: the last frame of each track, and frames where the
  fly slot is empty. At those frames, `main`'s `discretize_labels` filled the discretized features
  with a last-bin label, while this PR's marks them missing (NaN; see synthrat test 3). None of these
  frames is a target in a training chunk (fly test 7), which is why the losses agree.

**Results, synthrat: the code before the orientation change vs this PR.** `main` has no synthrat
code, so the comparison is against the last commit before the orientation change (the second command
above). Its fly simulation crashes (the `apply_opers_from_data` issue above), hence `--skip-fly`.

That code used the old orientation convention, so `--synthrat-old-convention` first converts
its outputs into the new one, using the same rearrangement as the model converter:

- velocity (forward, right, turn) = (old 1, −old 0, old 2);
- in the labels, the forward and lateral blocks of 25 bins are swapped and the lateral block is
  reversed.

Comparisons:

- **Chunk start indices:** identical.
- **Velocity inputs, after rearranging:** equal to within 7.6e-6 in z-scored units (allowed: 1e-5).
  The residual comes from the old code's `get_angle` offset of up to 1e-6 rad in orientation. Without
  rearranging, they differ by up to 6.2.
- **Firing-rate inputs:** equal to within 7.8e-6 (allowed: 1e-5), from RatInABox's rounding (its
  field-of-view cells are not bit-for-bit reproducible).
- **Labels, after rearranging:** equal to within 1.0e-4 (allowed: 5e-4). This is larger than the input
  difference because of how soft labels work:
  - a soft label splits its weight between two neighboring bins in proportion to where the value
    falls within the bin;
  - so shifting the value by δ moves δ / (bin width) of the weight;
  - with δ = 7.6e-6 and the narrowest bin 0.036 z-units wide, the largest expected change is 2.1e-4.

  Without rearranging, the labels differ by up to 1 (all the weight in a different bin).
- **Check that the comparison can fail:** without `--synthrat-old-convention`, it reports the
  velocity inputs and labels as different and exits with status 1.

**Limits:**

- `workloads.py` always comes from the working tree and calls into the reference's code, so the
  functions it uses must exist in both versions. One renamed helper is handled
  (`apf.utils.convert_torch_to_numpy`, which is `dict_convert_torch_to_numpy` on `main`). Comparing
  against much older code may need more of these.
- The workloads use random weights, so this checks the computation, not a trained model's behavior.

