# Porting the same-start divergence comparison onto main

The same-start comparison — K model futures and 2K policy futures from an identical burn-in,
compared by kinematic, positional and occupancy divergence — was written in the
`AnimalPoseForecasting` clone on `modernize_evaluate`, against that clone's
`synthrat/apf_ratinabox.py`. This log covers moving it onto a branch from main.

Five files came over. Two are new library-level changes to tracked files; the rest are new:

| file | what it is |
| --- | --- |
| `synthrat/trajectory_divergence.py` | the divergence metrics |
| `synthrat/scripts/samestart_rollouts.py` | draws the futures, one pickle per episode |
| `synthrat/scripts/run_samestart_rollouts.sh` | conda + cluster wrapper |
| `notebooks/evaluate_synthrat.py` | reads the pickles, makes the tables and figures |
| `tests/synthrat/test_trajectory_divergence.py` | metric checks on hand-built inputs |

The two run-only files went into a new `synthrat/scripts/`, whose `__init__.py` states the
rule: a module belongs there if nothing imports it. The wrapper is a shell script, which has
no business in an importable package, and the rollout module is the only one here that is run
rather than called. `generate_data.py` has a command line interface too, but five files import
it, so it stays in `synthrat/`.

The functions they called were rewired to main's layout: `compute_velocity`, `load_state`,
`rehydrate_policy` and `run_policy_from_burn_in` from `synthrat/generate_data.py`,
`rehydrate_env`/`rehydrate_data` from `synthrat/sensory.py`, `make_dataset_cache_wrapper` and
`simulate` from `experiments/synthrat.py`. The clone's `select_device` folded into
`apf.utils.get_cuda_device`, which was already the weaker version of the same decision: see
below.

Because main's `run_policy_from_burn_in` returns exactly `n_frames` of continuation, the
caller's compensation for the old `n_frames - 1` behaviour — asking for one extra frame and
truncating — came out, and the length check became an equality.

## Issue: kinematics computed from a pose treated orientation as the heading

`pose_to_kinematics` builds a RatInABox-style episode from a dataset pose so that
`compute_velocity` can give forward, sideways and angular speed. It built the head direction
straight from the pose's third column:

```python
'head_direction': np.stack([np.cos(pose[:, 2]), np.sin(pose[:, 2])], axis=-1)
```

That is correct in the clone it was written in, where synthrat orientation *is* the heading.
On main, orientation is the heading minus `ORIENTATION_OFFSET`, so this rotates every heading
by 90 degrees: the forward channel would carry the sideways velocity and the sideways channel
the forward velocity, silently, with no shape or range error to notice.

**Fix**: `pose_to_kinematics` converts with `synthrat.sensory.head_direction_from_orientation`.

The test for it encoded the same assumption — its `straight_line` helper stored a pose moving
along its own theta — so it passed before the fix and failed after. The helper now takes a
`heading`, moves along it, and stores `heading - ORIENTATION_OFFSET` as the orientation, which
is how the dataset stores a pose.

**Checked on recorded data, not just the hand-built test**: for 20 validation episodes,
converting the recorded head direction into the dataset convention and back out through
`pose_to_kinematics` reproduces `compute_velocity` to 3.6e-14 on all three components. Mean
|forward| is 0.0063 m per frame against mean |sideways| 0.0014 m, i.e. forward dominates 4.5:1,
which is what the unfixed code would have inverted.

## Issue: two functions for choosing a device, the weaker one already in the library

The ported code brought its own `select_device`, which takes a byte budget and picks CUDA only
if that much is free, raising instead of falling back when CUDA is demanded outright. But
`apf/utils.py` already had

```python
def get_cuda_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

the same decision without the memory check, called three times in `apf/pose.py`. Keeping both
would have left a general GPU helper inside a rollout script, where the next person writing a
cluster job would re-implement it rather than find it.

**Fix**: `get_cuda_device` takes the byte budget as an optional argument and works either way.
`get_cuda_device()` behaves as before, taking CUDA whenever it is available, which is what a
caller that cannot estimate its own footprint needs — anything choosing a device before its
model exists. `get_cuda_device(required_bytes)` adds the free-memory check, and `requested='cuda'`
raises rather than falling back, so a GPU job fails in seconds instead of running for hours on
CPU. It returns a device rather than a (device, reason) pair, and logs its reasoning instead: at
INFO when it takes a device, at WARNING when it passes CUDA over, since that is the case worth
seeing without turning INFO on. `samestart_rollouts.py` calls it and sets `basicConfig(INFO)` so
a job log records the margin.

**Not changed**: the four other places that choose a device — `flyllm/prepare.py:107`,
`notebooks/choose_discretize_bins.py:113`, `notebooks/agent_synthrat.py:94` and `apf/pose.py`'s
three call sites. All of them pick the device before the model is built, so they have no budget
to pass and would gain only a log line; `flyllm/prepare.py` already asserts for a missing GPU.

## Issue: isstart has a different layout on main, so every episode looked zero-length

`samestart_rollouts.py` finds the episode boundaries with

```python
idxstart = np.nonzero(loaded['isstart'])[1]
```

The clone's `make_dataset_cache_wrapper` returns `isstart` as `(1, n_frames_total)`, so index 1
is the frame axis. Main's returns `(n_frames_total, 1)`. Taking index 1 of that gives the column
indices, all zero, so every episode appeared to start and end at frame 0 and the run skipped all
1000 episodes with "too short for burn-in + horizon" — a clean-looking run that did nothing.

**Fix**: `load_model_and_data` now returns `isstart` flattened to `(n_frames_total,)`, asserting
that the data has one agent and that its length matches the pose array, so the bookkeeping cannot
depend on the layout. The caller uses axis 0.

## Issue: load_config_from_model_file cannot read a checkpoint on torch >= 2.6

`apf/io.py:load_config_from_model_file` called `torch.load(loadmodelfile)`. Since torch 2.6 the
default for `weights_only` is `True`, which refuses a checkpoint whose config and
`dataset_params` contain numpy arrays:

```
_pickle.UnpicklingError: Weights only load failed.
  Unsupported global: GLOBAL numpy._core.multiarray._reconstruct
```

Every other live `torch.load` in the repository already passes `weights_only=False`
(`apf/io.py:45`, `apf/io.py:72`, `synthrat/convert_orientation_convention.py:183`), so this one
was simply missed; any caller in test mode hits it, including `flyllm.prepare.init_config`.

**Fix**: `weights_only=False` with `map_location='cpu'`, since only the config is read here.

## The model had to be converted, and the old numbers do not transfer

The trained checkpoint predates the fly orientation convention, so
`experiments.synthrat.check_orientation_convention` refuses it. `synthrat/convert_orientation_convention.py`
converts it exactly, with no retraining:

```
python -m synthrat.convert_orientation_convention \
    <old>/synthratdefault_20260430T003318_bestepoch100.pth \
    <new>/synthratdefault_20260430T003318_bestepoch100_flyconvention.pth
```

The earlier results in `synthrat/data/..._samestart_K20_H80/` were produced in the clone, where
the data, the model and the metric code were all in the old convention. Those numbers are correct
and do not need regenerating. The clone's poses stored the heading itself, so taking
`(cos theta, sin theta)` as the head direction was right *there*, and its forward and sideways
velocities were genuinely forward and sideways — the same quantities the converted code computes
on main. The convention is a porting concern, not an error in the old results.

**Confirmed against the old run**: for episodes 0, 1 and 2, the ported code reproduces the earlier
run's ground truth and policy futures with positions bit-identical (max |dx, dy| = 0.0) and
orientations agreeing to 1.3e-6 rad, which is RatInABox's `get_angle` epsilon (it adds 1e-6 to x
before `arctan2`). The seeds match because both runs seed policy sample k with
`ratinabox_seed + k` and use the same `t0` and burn-in.

This comparison covers the ground truth and the policy futures, which are deterministic given a
seed. It cannot cover the model futures: sampling draws from the global numpy and torch RNGs, and
the draw order depends on the `n_samples` batching. The model side rests on
`test_old_model_conversion_is_exact` and on the converted checkpoint producing finite, diverging,
in-arena trajectories.

## Verification

- `tests/synthrat/test_trajectory_divergence.py`: 8 checks pass. Converted from pytest to plain
  asserts with a `__main__` runner, as in the repository's other tests, since pytest is not
  installed in `transformer312`.
- `tests/test_synthrat.py`: 16 pass, `tests/test_fly.py`: 18 pass, after the `apf/io.py` and
  `experiments/synthrat.py` changes.
- `simulate(..., n_samples=K)` repeats each agent into K rows. At `n_samples=1` the existing
  callers are unaffected, which `test_simulate_inputs_match_pipeline` covers since it drives
  `simulate` with `agent_ids=[0]`.
- Three episodes end to end on CPU (K=2, horizon=8): shapes `(K, 24, 3)` and `(2K, 24, 3)`, every
  sample's burn-in equal to the ground truth to 0.0 (so the replay is exact and the continuation
  length is right), no NaN, samples diverging, all coordinates inside the arena.
