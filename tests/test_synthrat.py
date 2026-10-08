"""Consistency checks for the synthrat pipeline: experiments/synthrat.py, synthrat/, apf/dataset.py.

Each test computes the same quantity two independent ways and checks that they agree, mirroring
obsolete/notebooks/debug_fly_example.py. The tests, the issues they cover and how they were verified
are described in logs/port-synthrat-modernize_evaluate.md.

Run from the repository root with
    python -m pytest tests/test_synthrat.py
or without pytest as
    python tests/test_synthrat.py
No trained model or GPU is needed. Tests skip when the data, or RatInABox, is not available.

Data. synthrat_data() loads the validation file named in synthrat/config_synthrat_default.json, keeps
its first 5 episodes (~1,300 frames, concatenated) and builds the dataset once per run:

    episodes = pickle.load(open(config['invalfile'], 'rb'))     # then truncated to 5 episodes
    dataset, info, pose, velocity, sensory, _, isstart = synthrat_experiment.make_dataset(
        config, config['invalfile'], return_all=True, debug=False, data=copy.deepcopy(episodes))

The quantities the tests refer to:
- pose: x, y (m) and orientation (rad) of the one simulated rat. Orientation follows the fly
  convention, heading - pi/2 (synthrat.sensory.ORIENTATION_OFFSET), converted from RatInABox's
  head-direction vectors.
- velocity: the GlobalVelocity output, the movement from frame t to t + 1 as (forward, sideways,
  turn) in the rat's own frame at t; NaN at the last frame of each episode.
- sensory: firing rates of 128 field-of-view boundary-vector cells and 16 head-direction cells.
- info: the environment, agent and sensory settings, to rebuild the RatInABox objects.
- isstart: True at the first frame of each episode.
- dataset.inputs at frame t: the movement into t (GlobalVelocity, Roll, Zscore) and the sensory at t
  (Sensory, Zscore). dataset.labels at frame t: the movement out of t (GlobalVelocity, Zscore, then
  Discretize for all three features). Chunks are 16 frames.

Tolerances. "Exact" means agreement to 1e-9. RatInABox's boundary-vector cells are not bit-for-bit
reproducible (~1e-7 in firing rate), so recomputed firing rates are compared to 1e-6, or 1e-4 after
z-scoring. The model's inputs are stored as float32 and compared to 1e-4.
"""
import copy
import functools
import os
import pickle
import tempfile
import unittest

import testing_utils  # puts this checkout first on sys.path; must precede the imports below

import numpy as np
import torch

import apf.dataset
import apf.io
from apf.utils import modrange

# Number of validation episodes the dataset is built from.
N_EPISODES = 5

# Round trips through float64 operations only.
EXACT_TOLERANCE = 1e-9
# RatInABox's field-of-view boundary cells are not bit-for-bit reproducible between evaluations
# (measured differences ~1e-7 in firing rate). After z-scoring, which divides by standard
# deviations as small as ~0.04, the differences are ~1e-6.
SENSORY_TOLERANCE = 1e-6
ZSCORED_SENSORY_TOLERANCE = 1e-4
# simulate() stores model inputs as float32.
FLOAT32_TOLERANCE = 1e-4

# Number of frames, spread evenly over the data, at which single-frame sensory is compared against
# whole-trajectory sensory. Each single-frame computation rebuilds the RatInABox cells (~5 ms), so
# not every frame is checked.
N_SENSORY_CHECK_FRAMES = 200
# Frames simulated with the stand-in model, after a burn-in of context_length frames.
N_SIMULATED_FRAMES = 20
# Number of chunks in the batch that is split back into named inputs and labels.
BATCH_SIZE = 4
# Length of each of the two stretches of frames in the single-agent test.
SINGLE_AGENT_CHECK_FRAMES = 60
# The discretized-label check is vacuous if it covers too few values.
MIN_FRACTION_CHECKED = 0.5

# Synthetic episode for the alignment test: random steps along and across the heading (m per
# frame) and random turns (rad per frame), within the central range of each in the synthrat
# data, starting from this position (m) and heading (rad) inside the 1 m x 1 m environment.
ALIGNMENT_FRAMES = 60
ALIGNMENT_SEED = 0
ALONG_HEADING_STEP_RANGE = (0.002, 0.008)
ACROSS_HEADING_STEP_RANGE = (-0.002, 0.002)
TURN_RANGE = (-0.2, 0.2)
ALIGNMENT_START_POSITION = (0.3, 0.3)
ALIGNMENT_START_HEADING = 0.3
# Step (m per frame) of the straight-ahead and straight-sideways trajectories in the feature-name test.
NAME_CHECK_STEP = 0.005


@functools.cache
def synthrat_data() -> dict:
    """Builds a dataset from the first N_EPISODES episodes of the synthrat validation file, once
    per test session.

    Returns:
        dict with
            config: the synthrat config.
            episodes: the loaded file, truncated to N_EPISODES episodes; pass a deep copy to
                make_dataset(data=...), which keeps references into it.
            experiment: the experiments.synthrat module.
            dataset: apf.dataset.Dataset.
            info: dict of env_info, agent_info and sensory_info, to rebuild the RatInABox objects.
            pose: Data, (1, n_frames, 3) float array of x, y (m) and orientation (rad); all
                episodes are concatenated along frames.
            velocity: Data, global movement from frame t to t + 1, (1, n_frames, 3) float
                array; NaN at the last frame of each episode.
            sensory: Data, (1, n_frames, n_sensory_features) float array of firing rates.
            isstart: (n_frames, 1) bool array, True at the first frame of each episode.
    """
    try:
        import experiments.synthrat as synthrat_experiment
        from synthrat.config import read_config
    except ImportError as error:
        raise unittest.SkipTest(f'synthrat needs RatInABox: {error}')
    config = read_config()
    testing_utils.require_file(config['invalfile'], 'synthrat validation data')
    with open(config['invalfile'], 'rb') as f:
        episodes = pickle.load(f)
    episodes = dict(episodes, track=episodes['track'][:N_EPISODES], hidden=episodes['hidden'][:N_EPISODES])
    dataset, info, pose, velocity, sensory, _, isstart = synthrat_experiment.make_dataset(
        config, config['invalfile'], return_all=True, debug=False, data=copy.deepcopy(episodes))
    return dict(config=config, episodes=episodes, experiment=synthrat_experiment, dataset=dataset, info=info,
                pose=pose, velocity=velocity, sensory=sensory, isstart=isstart)


def episode_bounds(isstart: np.ndarray) -> list[tuple[int, int]]:
    """Frame ranges of the concatenated episodes.

    Args:
        isstart: (n_frames, 1) bool array, True at the first frame of each episode.

    Returns:
        list of (first_frame, end_frame) per episode, end exclusive.
    """
    starts = np.nonzero(isstart[:, 0])[0]
    ends = np.append(starts[1:], isstart.shape[0])
    return list(zip(starts, ends))


def assert_datasets_match(actual, expected, description: str) -> None:
    """Asserts two synthrat datasets have the same inputs, labels and chunks.

    Sensory firing rates are compared with ZSCORED_SENSORY_TOLERANCE, because they may have been
    recomputed; everything else must match to EXACT_TOLERANCE.

    Args:
        actual, expected: apf.dataset.Dataset.
        description: how actual was built, for failure messages.
    """
    for group in ['inputs', 'labels']:
        for key, data in getattr(expected, group).items():
            tolerance = ZSCORED_SENSORY_TOLERANCE if key == 'sensory' else EXACT_TOLERANCE
            testing_utils.assert_arrays_match(getattr(actual, group)[key].array, data.array, tolerance,
                                              f"{description}: {group} '{key}'")
    assert np.array_equal(actual.chunk_indices, expected.chunk_indices), f'{description}: chunks differ'


def synthetic_episode(dt: float, along_range: tuple[float, float] = ALONG_HEADING_STEP_RANGE,
                      across_range: tuple[float, float] = ACROSS_HEADING_STEP_RANGE,
                      turn_range: tuple[float, float] = TURN_RANGE) -> tuple[dict, np.ndarray]:
    """A synthetic trajectory: on each frame the rat steps along and across its heading and turns,
    by amounts drawn uniformly from the given ranges (a range (a, a) gives a constant step).

    Args:
        dt: seconds per frame, for the velocity entry.
        along_range: (min, max) step along the heading, m per frame.
        across_range: (min, max) step to the left of the heading, m per frame.
        turn_range: (min, max) turn, rad per frame.

    Returns:
        episode: dict with 'pos' (ALIGNMENT_FRAMES, 2), 'head_direction' (ALIGNMENT_FRAMES, 2)
            unit vectors and 'vel' (ALIGNMENT_FRAMES, 2) in m/s, as in the synthrat data files.
        pose: (1, ALIGNMENT_FRAMES, 3) float array of x, y and orientation, orientation in the fly
            convention (heading - pi/2), as make_dataset should build it.
    """
    rng = np.random.default_rng(ALIGNMENT_SEED)
    n_steps = ALIGNMENT_FRAMES - 1
    along = rng.uniform(*along_range, n_steps)
    across = rng.uniform(*across_range, n_steps)
    heading = ALIGNMENT_START_HEADING + np.concatenate([[0.], np.cumsum(rng.uniform(*turn_range, n_steps))])
    direction = np.stack([np.cos(heading), np.sin(heading)], axis=-1)              # (n_frames, 2)
    perpendicular = np.stack([-np.sin(heading), np.cos(heading)], axis=-1)        # (n_frames, 2)
    steps = along[:, None] * direction[:-1] + across[:, None] * perpendicular[:-1]   # (n_steps, 2)
    position = np.asarray(ALIGNMENT_START_POSITION) + np.concatenate([np.zeros((1, 2)), np.cumsum(steps, axis=0)])
    velocity = np.concatenate([steps, steps[-1:]]) / dt
    episode = {'pos': position, 'head_direction': direction, 'vel': velocity}
    # written out rather than calling synthrat.sensory.orientation_from_head_direction, so the
    # test checks that function
    orientation = modrange(heading - np.pi / 2, -np.pi, np.pi)
    pose = np.concatenate([position, orientation[:, None]], axis=-1)[None]
    return episode, pose


def test_global_velocity_round_trip():
    """Velocity → pose.

    Inputs: `pose`, `velocity` and `isstart` (which marks where each episode starts). For each of
    the 5 episodes, calls `GlobalVelocity.invert()` starting from the real pose at the episode's
    first frame, which adds up the frame-to-frame movements, rotating forward/sideways movement back
    into arena coordinates. Passes if the reconstructed x, y and orientation at each frame equal
    `pose` at that frame exactly. Same purpose as `test_velocity_round_trip` in test_fly.py.
    """
    data = synthrat_data()
    pose, velocity = data['pose'], data['velocity']
    velocity_op = apf.dataset.get_operation(velocity.operations, 'globalvelocity')
    for start, end in episode_bounds(data['isstart']):
        true_pose = pose.array[0, start:end]                   # (n_episode_frames, 3)
        # the last frame's velocity is NaN, but invert only uses movement into frames 1..end-1
        recovered = velocity_op.invert(velocity.array[0, start:end], x0=true_pose[0])
        error = np.abs(recovered - true_pose)
        error[:, 2] = np.abs(modrange(recovered[:, 2] - true_pose[:, 2], -np.pi, np.pi))
        assert error.max() < EXACT_TOLERANCE, f'episode at frame {start}: round trip error {error.max():.3g}'


def test_labels_invert_to_velocity():
    """Labels → velocity.

    Inputs: `dataset.labels['velocity']` and `velocity`. The labels were made from `velocity` by
    `Zscore` and then `Discretize`, for all three features. Undoes them with
    `apf.dataset.invert_to_named(labels, 'globalvelocity', discretize={'do_sampling': False})` —
    `Discretize.invert()` without sampling, then `Zscore.invert()` — and checks each feature against
    the widest-interior-bin bound, as `test_labels_invert_to_velocity` in test_fly.py.
    """
    data = synthrat_data()
    labels, velocity = data['dataset'].labels['velocity'], data['velocity']
    discretize = labels.operations[-1]
    assert isinstance(discretize, apf.dataset.Discretize), f'expected labels to end in Discretize, got {discretize.name}'
    recovered = apf.dataset.invert_to_named(labels, 'globalvelocity', discretize={'do_sampling': False})
    zscore = apf.dataset.get_operation(labels.operations, 'zscore')
    defined = ~np.isnan(velocity.array).any(-1)                # (1, n_frames)
    for feature in range(velocity.array.shape[-1]):
        true_zscored = (velocity.array[..., feature] - zscore.mean[feature]) / zscore.std[feature]
        decoded_zscored = (recovered[..., feature] - zscore.mean[feature]) / zscore.std[feature]
        errors, fraction_checked, max_width = testing_utils.interior_decoding_errors(
            true_zscored[defined], decoded_zscored[defined], discretize.bin_edges[feature])
        assert fraction_checked > MIN_FRACTION_CHECKED, \
            f'feature {feature}: only {fraction_checked:.2f} of values checked'
        assert errors.max() <= max_width, \
            f'feature {feature}: decoding error {errors.max():.3g} > widest interior bin {max_width:.3g}'


def test_chunks_have_defined_targets():
    """No undefined targets in training chunks.

    Inputs: `dataset.chunk_indices` and `velocity`. As `test_chunks_have_defined_targets` in
    test_fly.py, for every 16-frame chunk. Here it matters: every synthrat label is discretized, so
    this depends on `discretize_labels` (called by `Discretize.apply()`) marking an undefined
    movement as missing (NaN). If it put the movement into a bin instead, the last frame of an
    episode could enter a chunk with a made-up target.
    """
    data = synthrat_data()
    testing_utils.assert_chunks_have_defined_targets(data['dataset'], data['velocity'].array)


def test_dataset_rebuilt_from_saved_params():
    """Dataset rebuilt from saved parameters.

    Inputs: `config`, `dataset` and the loaded episodes. As `test_dataset_rebuilt_from_saved_params`
    in test_fly.py: `dataset.get_params()` goes into `config['dataset_params']`, and `make_dataset`
    on the same episodes applies the operations rebuilt from it with `apply_opers_from_data_params`.
    The firing rates are recomputed (the `Sensory` operation calls
    `synthrat.sensory.compute_sensory`), and RatInABox's boundary-vector cells are not bit-for-bit
    reproducible (differences ~1e-7 in firing rate, ~1e-6 after z-scoring), so the rebuilt dataset's
    sensory inputs must equal `dataset`'s to within 1e-4; its other inputs, its labels and its
    chunks must equal `dataset`'s exactly.
    """
    data = synthrat_data()
    config = copy.deepcopy(data['config'])
    config['dataset_params'] = data['dataset'].get_params()
    rebuilt, _ = data['experiment'].make_dataset(config, config['invalfile'], debug=False,
                                                 data=copy.deepcopy(data['episodes']))
    assert_datasets_match(rebuilt, data['dataset'], 'rebuilt from saved params')


def test_reference_dataset_reproduces_dataset():
    """Dataset built with itself as reference.

    Inputs: `config`, `dataset` and the loaded episodes. Calls `make_dataset(...,
    ref_dataset=dataset)` on the same episodes, so `apply_opers_from_data` applies the reference's
    operations with its z-score parameters and bins — the way validation data is built. Every input
    and label of the rebuilt dataset must equal `dataset`'s, and its chunks must be the same: the
    sensory inputs to within 1e-4, as in `test_dataset_rebuilt_from_saved_params`, the rest exactly.
    """
    data = synthrat_data()
    rebuilt, _ = data['experiment'].make_dataset(data['config'], data['config']['invalfile'], debug=False,
                                                 ref_dataset=data['dataset'], data=copy.deepcopy(data['episodes']))
    assert_datasets_match(rebuilt, data['dataset'], 'built with ref_dataset')


def test_cached_sensory_matches_computed():
    """Cached firing rates.

    Inputs: `config`, `dataset`, `sensory` and the loaded episodes. Computing firing rates is slow,
    so `make_dataset` can take a precomputed sensory array instead (`cached_sensory_array=`). It
    then builds the `Sensory` operation's record of which columns belong to which cell population
    (`idxinfo`, from `rehydrate_sensory`) and its feature names (`get_all_feature_names`) without
    computing any firing rates. Building with `sensory.array` as the cache must give inputs and
    labels equal to `dataset`'s exactly, `get_input_names()` equal to `dataset.get_input_names()`,
    and a `Sensory` operation whose `idxinfo` equals that of `dataset`'s `Sensory` operation.
    """
    data = synthrat_data()
    dataset = data['dataset']
    cached, _ = data['experiment'].make_dataset(data['config'], data['config']['invalfile'], debug=False,
                                                data=copy.deepcopy(data['episodes']),
                                                cached_sensory_array=data['sensory'].array)
    for group in ['inputs', 'labels']:
        for key, original in getattr(dataset, group).items():
            testing_utils.assert_arrays_match(getattr(cached, group)[key].array, original.array, EXACT_TOLERANCE,
                                              f"{group} '{key}'")
    assert cached.get_input_names() == dataset.get_input_names(), 'input names differ'
    population_slices = [apf.dataset.get_operation(d.inputs['sensory'].operations, 'sensory').idxinfo
                         for d in (cached, dataset)]
    assert population_slices[0] == population_slices[1], 'sensory populations are split differently'


def test_inputs_and_labels_time_alignment():
    """Inputs and labels line up in time.

    Inputs: `config`, `dataset` (as reference) and the loaded episodes' environment, agent and
    sensory settings. Builds a synthetic 60-frame trajectory: the rat starts at (0.3, 0.3) m facing
    0.3 rad and on each frame steps 2–8 mm forward and up to 2 mm sideways and turns up to 0.2 rad,
    all at random. Calls `make_dataset(..., ref_dataset=dataset, data=...)` on it, which derives
    orientation from the head direction (`orientation_from_head_direction`, fly convention) and
    builds the inputs and labels. From the pose `make_dataset` built (checked exactly against the
    constructed trajectory, whose orientation the test writes out as heading − π/2), computes the
    true movement m[t] from each frame to the next with `GlobalVelocity.apply()`. Undoes the
    velocity input's z-scoring (`invert_to_named(..., 'roll')`, i.e. `Zscore.invert()`): at frame t
    it must equal m[t−1] exactly. Decodes the labels without sampling (`Discretize.invert()`, then
    `Zscore.invert()`): at frame t they must equal m[t] to within the widest interior bin. The test
    also checks that the steps vary enough that a one-frame shift would fail, so it cannot pass
    vacuously. Same purpose as `test_inputs_and_labels_time_alignment` in test_fly.py, for
    discretized labels.
    """
    data = synthrat_data()
    episode, pose = synthetic_episode(data['episodes']['agent_info']['dt'])
    synthetic = dict(data['episodes'], track=[episode], hidden=[{}])
    dataset, _, built_pose, *_ = data['experiment'].make_dataset(
        data['config'], data['config']['invalfile'], debug=False, ref_dataset=data['dataset'], data=synthetic,
        return_all=True)
    difference = built_pose.array - pose
    difference[..., 2] = modrange(difference[..., 2], -np.pi, np.pi)
    assert np.abs(difference).max() < EXACT_TOLERANCE, 'make_dataset did not build the synthetic trajectory'
    movement = apf.dataset.GlobalVelocity(tspred=[1]).apply(built_pose.array)[0, :-1]   # m[t], (n_frames - 1, 3)

    velocity_input = apf.dataset.invert_to_named(dataset.inputs['velocity'], 'roll')[0]   # (n_frames, 3)
    error = np.abs(velocity_input[1:] - movement).max()
    assert error < EXACT_TOLERANCE, f'velocity input at t vs m[t - 1]: error {error:.3g}'

    labels = dataset.labels['velocity']
    decoded = apf.dataset.invert_to_named(labels, 'globalvelocity', discretize={'do_sampling': False})[0, :-1]
    discretize = labels.operations[-1]
    zscore = apf.dataset.get_operation(labels.operations, 'zscore')
    shift_detectable = False
    for feature in range(movement.shape[-1]):
        def zscored(x):
            return (x - zscore.mean[feature]) / zscore.std[feature]
        errors, fraction_checked, max_width = testing_utils.interior_decoding_errors(
            zscored(movement[:, feature]), zscored(decoded[:, feature]), discretize.bin_edges[feature])
        assert fraction_checked > MIN_FRACTION_CHECKED, f'feature {feature}: only {fraction_checked:.2f} checked'
        assert errors.max() <= max_width, f'label at t vs m[t], feature {feature}: error {errors.max():.3g}'
        shifted_errors, _, _ = testing_utils.interior_decoding_errors(
            zscored(movement[1:, feature]), zscored(decoded[:-1, feature]), discretize.bin_edges[feature])
        shift_detectable |= shifted_errors.max() > max_width
    assert shift_detectable, 'the synthetic movement varies too little for a one-frame shift to fail'


def test_batches_split_back_to_dataset():
    """Batches split back into named inputs and labels.

    Inputs: `dataset`. As `test_batches_split_back_to_dataset` in test_fly.py. Synthrat's labels are
    all discrete, with no continuous part, which exercises the branch of `split_output_by_names()`
    for fully discretized models.
    """
    testing_utils.assert_batches_split_back_to_dataset(synthrat_data()['dataset'], BATCH_SIZE)


def test_single_frame_sensory_matches_trajectory():
    """Firing rates from a single frame.

    Inputs: `pose`, `sensory` and the `Sensory` operation in `dataset`. Synthrat's `simulate`
    computes each new frame's firing rates from that frame's pose alone, with `Sensory.apply()`
    (`experiments.synthrat. Sensory`, which rebuilds the cells from their stored settings and calls
    `synthrat.sensory.compute_sensory`). At 200 frames spread evenly over the 1,290, compares firing
    rates computed that way with `sensory`, which `make_dataset` computed over the whole trajectory:
    each single-frame result must equal `sensory` at that frame to within 1e-6. (Each single-frame
    computation rebuilds the RatInABox cells, ~5 ms, so not every frame is checked.) This holds
    because both cell types depend only on the current position and heading; it would stop holding
    if a cell type that depends on movement were added.
    """
    data = synthrat_data()
    pose, sensory = data['pose'], data['sensory']
    sensory_op = apf.dataset.get_operation(data['dataset'].inputs['sensory'].operations, 'sensory')
    frames = np.linspace(0, pose.array.shape[1] - 1, N_SENSORY_CHECK_FRAMES).astype(int)
    for t in frames:
        single_frame = sensory_op.apply(pose.array[:, t:t + 1])[:, 0]   # (1, n_sensory_features)
        error = np.abs(single_frame - sensory.array[:, t]).max()
        assert error < SENSORY_TOLERANCE, f'frame {t}: sensory differs by {error:.3g}'


def test_simulate_inputs_match_pipeline():
    """Simulation's inputs match the training pipeline.

    Inputs: `dataset`, `pose`, `velocity` and `isstart`. Calls `experiments.synthrat.simulate`, as
    fly `test_sensory_matches_recorded_head_direction`: 16 burn-in frames and 20 simulated frames,
    starting one frame into an episode (the first frame of an episode has no movement into it, so
    its velocity input is undefined). The expected inputs come from running the dataset's own
    `GlobalVelocity` and `Sensory` operations on the poses `simulate` produced, then
    `apply_opers_from_data` for `Roll` and `Zscore`. At every simulated frame, the inputs `simulate`
    gave the model must equal these recomputed inputs to within 1e-4 (the model's inputs are stored
    as float32).
    """
    data = synthrat_data()
    dataset, pose = data['dataset'], data['pose']
    burn_in = dataset.context_length
    n_frames = burn_in + N_SIMULATED_FRAMES
    # Start one frame into an episode: the first frame has no movement into it, so its velocity
    # input is undefined and never appears in a training chunk.
    start_frame = next(start + 1 for start, end in episode_bounds(data['isstart']) if end - start > n_frames)
    agent_ids = np.array([0])

    model = testing_utils.TrueLabelModel(testing_utils.true_labels(dataset, start_frame, n_frames, agent_ids))
    _, pred_pose = data['experiment'].simulate(dataset=dataset, model=model, pose=pose, track_len=n_frames,
                                               burn_in=burn_in, max_contextl=None, agent_ids=agent_ids,
                                               start_frame=start_frame, progress_bar=False)
    assert not np.isnan(pred_pose).any(), 'simulation produced NaN poses'
    recorded = model.last_input                                # (1, n_frames - 1, d_input)

    # burn-in frames are copied from the data
    burn_in_inputs = dataset.get_chunk(start_frame, n_frames, 0)['input'][None]
    assert np.array_equal(recorded[:, :burn_in], burn_in_inputs[:, :burn_in]), 'burn-in inputs differ from the data'

    # simulated frames: run the dataset's operations on the simulated pose
    velocity_op = apf.dataset.get_operation(data['velocity'].operations, 'globalvelocity')
    sensory_op = apf.dataset.get_operation(dataset.inputs['sensory'].operations, 'sensory')
    simulated_pose = apf.dataset.Data('pos', pred_pose, [])  # (1, n_frames, 3)
    expected_parts = apf.dataset.apply_opers_from_data(dataset.inputs, {
        'velocity': velocity_op(simulated_pose, isstart=None),
        'sensory': sensory_op(simulated_pose)})
    expected = np.concatenate([expected_parts[key].array for key in dataset.inputs], axis=-1)   # (1, n_frames, d_input)
    offsets = np.cumsum([0] + [d.array.shape[-1] for d in dataset.inputs.values()])
    simulated = slice(burn_in, n_frames - 1)
    for key, start, stop in zip(dataset.inputs, offsets[:-1], offsets[1:]):
        error = np.abs(recorded[:, simulated, start:stop] - expected[:, simulated, start:stop]).max()
        assert error < FLOAT32_TOLERANCE, f"input '{key}': simulate's input differs from the pipeline by {error:.3g}"


def test_feature_names_match_dimensions():
    """Feature names.

    Inputs: `dataset`. As `test_feature_names_match_dimensions` in test_fly.py.
    """
    testing_utils.assert_feature_names_match_dimensions(synthrat_data()['dataset'])


def test_velocity_feature_names_match_movement():
    """Velocity features match their names.

    Inputs: `config`, `dataset` (as reference) and the loaded episodes' settings. Builds two
    synthetic 60-frame trajectories from (0.3, 0.3) m facing 0.3 rad, with no turning: one stepping
    5 mm straight ahead each frame, one stepping 5 mm straight to the left. Calls `make_dataset(...,
    ref_dataset=dataset, data=...)` on each and reads `velocity`, the output of `GlobalVelocity`
    with its feature names. Passes if, for the straight-ahead trajectory, `forward_velocity_1` is +5
    mm at every frame and the other two features are 0, and for the sideways one,
    `sideways_velocity_1` is 5 mm in magnitude (the fly convention makes leftward movement negative)
    and the other two are 0. This depends on `make_dataset` converting RatInABox's head direction to
    the fly convention, which `GlobalVelocity` assumes; if orientation were taken as the heading
    itself, forward movement would land in `sideways_velocity_1`.
    """
    data = synthrat_data()
    dt = data['episodes']['agent_info']['dt']
    for moved, along, across in [('forward_velocity_1', NAME_CHECK_STEP, 0.), ('sideways_velocity_1', 0., NAME_CHECK_STEP)]:
        episode, _ = synthetic_episode(dt, along_range=(along, along), across_range=(across, across), turn_range=(0., 0.))
        synthetic = dict(data['episodes'], track=[episode], hidden=[{}])
        _, _, _, velocity, *_ = data['experiment'].make_dataset(
            data['config'], data['config']['invalfile'], debug=False, ref_dataset=data['dataset'], data=synthetic,
            return_all=True)
        movement = velocity.array[0, :-1]                     # (n_frames - 1, 3); the last frame has none
        for column, name in enumerate(velocity.feature_names):
            # sideways movement to the left is negative in the fly convention, so compare magnitudes
            expected = NAME_CHECK_STEP if name == moved else 0.
            error = np.abs(np.abs(movement[:, column]) - expected).max()
            assert error < EXACT_TOLERANCE, f'stepping {moved}: feature {name} off by {error:.3g}'
        if moved == 'forward_velocity_1':
            assert (movement[:, velocity.feature_names.index(moved)] > 0).all(), 'forward movement is not positive'


def test_sensory_matches_recorded_head_direction():
    """Firing rates match the recorded head direction.

    Inputs: `sensory`, `info`, `isstart` and the first loaded episode. Computes the firing rates
    directly from the episode's recorded position and RatInABox head direction with
    `synthrat.sensory.compute_sensory`, and compares with `sensory`, which `make_dataset` computed
    from the pose — that is, after converting the head direction to orientation and back
    (`orientation_from_head_direction`, then `head_direction_from_orientation` in `Sensory.apply`).
    The dataset's `sensory` for the first episode must equal the directly computed firing rates to
    within 1e-6, at every frame. This catches the two conversions disagreeing: if `Sensory.apply`
    took the orientation itself as the heading, the cells would see a heading 90° off.
    """
    data = synthrat_data()
    from synthrat.sensory import compute_sensory
    start, end = episode_bounds(data['isstart'])[0]
    recorded = data['episodes']['track'][0]
    by_population = compute_sensory({'pos': recorded['pos'], 'head_direction': recorded['head_direction']},
                                    info=data['info'])        # {population: (n_episode_frames, n_cells)}
    direct = np.concatenate(list(by_population.values()), axis=1)   # (n_episode_frames, n_sensory_features)
    testing_utils.assert_arrays_match(data['sensory'].array[0, start:end], direct, SENSORY_TOLERANCE,
                                      'sensory from the pose vs from the recorded head direction')


def test_models_from_the_old_orientation_convention_are_refused():
    """Old models are refused.

    Inputs: `config`. Saves two checkpoints of a tiny stand-in model (`torch.nn.Linear(2, 2)`) with
    `apf.io.save_model` to a temporary directory: one with the current `config`, one with
    `orientation_convention` removed from it, as in every synthrat model saved before the fly
    convention was adopted. Loads each with `apf.io.load_model(...,
    check_state=check_orientation_convention)`. Passes if the first loads and the second raises
    `ValueError`.
    """
    data = synthrat_data()
    check = data['experiment'].check_orientation_convention
    old_config = {key: value for key, value in data['config'].items() if key != 'orientation_convention'}
    with tempfile.TemporaryDirectory() as directory:
        current_file, old_file = os.path.join(directory, 'current.pth'), os.path.join(directory, 'old.pth')
        apf.io.save_model(current_file, torch.nn.Linear(2, 2), config=data['config'])
        apf.io.save_model(old_file, torch.nn.Linear(2, 2), config=old_config)
        apf.io.load_model(current_file, torch.nn.Linear(2, 2), 'cpu', check_state=check)
        try:
            apf.io.load_model(old_file, torch.nn.Linear(2, 2), 'cpu', check_state=check)
        except ValueError:
            return
    raise AssertionError('a model saved without the current orientation convention was loaded')


def test_old_model_conversion_is_exact():
    """Old models convert exactly.

    Inputs: random stand-in weights, and `dataset`'s z-score and bin parameters treated as if they
    were an old model's. Calls `synthrat.convert_orientation_convention.convert_weights` and
    `convert_dataset_params`. With old velocity features (left, forward, turn) and new ones
    (forward, right, turn) = (old 1, −old 0, old 2): the converted velocity input layer applied to
    the rearranged z-scored velocity must equal the old layer applied to the old one; the converted
    output layer's bin logits must equal the old ones with the forward and lateral blocks swapped
    and the lateral block's bins reversed; the converted z-score parameters must turn rearranged raw
    velocities into the rearranged z-scores; and the converted lateral bin edges must be the old
    ones negated and reversed, still increasing. This checks the conversion's arithmetic without
    needing a trained model.
    """
    data = synthrat_data()
    from synthrat import convert_orientation_convention as convert
    rng = np.random.default_rng(0)
    n_bins, d_model, n_frames = 25, 8, 5
    weights = {convert.VELOCITY_ENCODER_WEIGHT: torch.randn(d_model, 3),
               convert.DECODER_WEIGHT: torch.randn(3 * n_bins, d_model),
               convert.DECODER_BIAS: torch.randn(3 * n_bins)}
    converted = convert.convert_weights(weights, n_bins)

    old_input = torch.randn(n_frames, 3)                                  # z-scored velocity, old order
    new_input = torch.stack([old_input[:, 1], -old_input[:, 0], old_input[:, 2]], dim=1)
    encoded_old = old_input @ weights[convert.VELOCITY_ENCODER_WEIGHT].T
    encoded_new = new_input @ converted[convert.VELOCITY_ENCODER_WEIGHT].T
    assert torch.allclose(encoded_new, encoded_old, atol=1e-6), 'velocity input layer does not commute'

    hidden = torch.randn(n_frames, d_model)
    def logits(w):
        return (hidden @ w[convert.DECODER_WEIGHT].T + w[convert.DECODER_BIAS]).reshape(n_frames, 3, n_bins)
    old_logits, new_logits = logits(weights), logits(converted)
    expected = torch.stack([old_logits[:, 1], old_logits[:, 0].flip(-1), old_logits[:, 2]], dim=1)
    assert torch.equal(new_logits, expected), 'output layer not rearranged as expected'

    params = convert.convert_dataset_params(data['dataset'].get_params())
    old_params = data['dataset'].get_params()
    def attributes(p, group, name):
        return next(op['attributes'] for op in p[group]['velocity'] if op['class'] == name)
    raw_old = rng.normal(size=(n_frames, 3))
    raw_new = np.stack([raw_old[:, 1], -raw_old[:, 0], raw_old[:, 2]], axis=1)
    old_z, new_z = attributes(old_params, 'labels', 'Zscore'), attributes(params, 'labels', 'Zscore')
    zscored_old = (raw_old - old_z['mean']) / old_z['std']
    zscored_new = (raw_new - new_z['mean']) / new_z['std']
    assert np.allclose(zscored_new, np.stack([zscored_old[:, 1], -zscored_old[:, 0], zscored_old[:, 2]], axis=1))
    old_edges = np.asarray(attributes(old_params, 'labels', 'Discretize')['bin_edges'])
    new_edges = np.asarray(attributes(params, 'labels', 'Discretize')['bin_edges'])
    assert np.array_equal(new_edges[1], -old_edges[0][::-1]) and np.all(np.diff(new_edges, axis=1) >= 0)


def test_operations_handle_single_agent():
    """Every operation handles one agent.

    Inputs: `dataset`, `pose` and `velocity`. Synthrat has one rat, so the batch is two 60-frame
    stretches from different episodes, each starting one frame in. As
    `test_operations_handle_single_agent` in test_fly.py, for `Roll` and `Zscore` (velocity input),
    `Sensory` and `Zscore` (sensory input, `Sensory` to 1e-6 for RatInABox's rounding), and `Zscore`
    and `Discretize` (labels). It also checks `GlobalVelocity.apply` with an `isstart` that marks a
    track start partway through, and `GlobalVelocity.invert` with a starting pose `x0`. An operation
    that returns an extra leading axis for one agent fails here, e.g. (1, 60, 3) instead of (60, 3).
    """
    data = synthrat_data()
    dataset, pose, velocity = data['dataset'], data['pose'], data['velocity']
    starts = [start + 1 for start, _ in episode_bounds(data['isstart'])[:2]]    # skip each episode's first frame
    pose_batch = np.stack([pose.array[0, s:s + SINGLE_AGENT_CHECK_FRAMES] for s in starts])          # (2, n, 3)
    velocity_batch = np.stack([velocity.array[0, s:s + SINGLE_AGENT_CHECK_FRAMES] for s in starts])  # (2, n, 3)
    chains = [
        ('velocity input', velocity_batch,
         apf.dataset.get_post_operations(dataset.inputs['velocity'].operations, 'globalvelocity')),
        ('sensory input', pose_batch, dataset.inputs['sensory'].operations),
        ('velocity labels', velocity_batch,
         apf.dataset.get_post_operations(dataset.labels['velocity'].operations, 'globalvelocity')),
    ]
    testing_utils.assert_operations_handle_single_agent(chains, tolerances={'Sensory': SENSORY_TOLERANCE})

    global_velocity = apf.dataset.get_operation(velocity.operations, 'globalvelocity')
    isstart = np.zeros((SINGLE_AGENT_CHECK_FRAMES, 2), bool)
    isstart[[0, SINGLE_AGENT_CHECK_FRAMES // 2], :] = True     # a track start partway through
    testing_utils.assert_arrays_match(global_velocity.apply(pose_batch[0], isstart=isstart[:, 0]),
                                      global_velocity.apply(pose_batch, isstart=isstart)[0], 0.,
                                      'GlobalVelocity.apply with isstart')
    testing_utils.assert_arrays_match(global_velocity.invert(velocity_batch[0], x0=pose_batch[0, 0]),
                                      global_velocity.invert(velocity_batch, x0=pose_batch[:, 0])[0], 0.,
                                      'GlobalVelocity.invert with x0')


if __name__ == '__main__':
    testing_utils.run_as_script(globals())
