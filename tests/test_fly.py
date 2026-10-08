"""Consistency checks for the fly pipeline: experiments/flyllm.py, apf/dataset.py, apf/simulation.py.

Each test computes the same quantity two independent ways and checks that they agree, mirroring
obsolete/notebooks/debug_fly_example.py for the Data / Operation pipeline. The tests, the issues they
cover and how they were verified are described in logs/port-synthrat-modernize_evaluate.md.

Run from the repository root with
    python -m pytest tests/test_fly.py
or without pytest as
    python tests/test_fly.py
No trained model or GPU is needed. Tests skip when the data is not reachable.

Data. tests/data/small_usertrain_v3.npz is the first 2,000 frames of each of the first 5 videos of the
v3 MABe training file usertrain_v3.npz (10 fly slots, 10,000 frames), in the same format.
tests/config_fly_test.json points the default fly config at it, with no filtering by fly type and
flip augmentation on, as in the training configs; flip augmentation appends a mirror-image copy of
every fly after the original frames, giving 20,000 frames. fly_data() builds the training dataset
once per run:

    (dataset, flyids, track, pose, velocity, sensory, _, isdata, isstart,
     useoutputmask) = fly_experiment.make_dataset(config, 'intrainfile', return_all=True, debug=False)

The quantities the tests refer to:
- track: keypoints, the tracked (x, y) positions in mm of 19 body points per fly per frame (a fly is
  ~2.8 mm long). The v3 files carry two more, which the loader drops.
- pose: 29 numbers per fly per frame, computed from the keypoints by Pose.apply (kp2feat). Three are
  global: the position of the front of the thorax (x, y) and the body orientation, measured with the
  fly facing +y in its own frame. The other 26 are relative: the body's shape in the fly's own frame.
  Pose.invert (feat2kp) converts a pose back to keypoints. Body measurements the pose does not store
  come from a per-fly scale table, looked up by the fly's identity in flyids.
- velocity: the movement from frame t to t + 1 (the Velocity operation): for the global features,
  forward and sideways movement in the fly's own frame at t and the change in orientation; for the
  relative features, their change. NaN at the last frame of a track.
- sensory: 186 numbers per fly computed from all flies' keypoints at one frame.
- dataset.inputs at frame t: the movement into t (velocity shifted by one frame, Roll), the relative
  pose at t (Subset) and the sensory at t, each z-scored with the training mean and std (Zscore).
- dataset.labels at frame t: the movement out of t, z-scored, with the 3 global features discretized
  into 25 bins (Fusion of Discretize and Identity). Each processing step is an Operation stored with
  its parameters, so it can be re-applied to new data or undone.
- dataset.sessions: the longest stretches of consecutive frames of one fly in which every input and
  label is defined; here each is one fly's track in one video minus its first and last frames (1,998
  frames), 94 in all. Training chunks are 65-frame windows cut from sessions without overlap (2,820).

Tolerances. "Exact" means agreement to 1e-9; measured differences are ~1e-14. Values that pass
through float32 storage (keypoints, chunks, the model's inputs) are compared to 1e-4, or 1e-2 where
recomputing z-scored velocities from rounded keypoints amplifies the rounding.
"""
import copy
import functools

import testing_utils  # puts this checkout first on sys.path; must precede the imports below

import numpy as np

import apf.dataset
import apf.simulation
import experiments.flyllm as fly_experiment
from apf.utils import modrange
from flyllm.config import featangle, featrelative, keypointnames, posenames, read_config

CONFIG_FILE = testing_utils.REPO_DIR / 'tests' / 'config_fly_test.json'

# Keypoints the code uses. Data files may carry extra keypoints after these (the v3 files append
# the two outer wing points), which feat2kp does not produce, so comparisons use only these.
N_CODE_KEYPOINTS = len(keypointnames)

# Round trips through float64 operations only. Measured errors are ~1e-14.
EXACT_TOLERANCE = 1e-9
# Keypoints and chunks are stored as float32; features recomputed from them carry that rounding.
FLOAT32_TOLERANCE = 1e-4
# simulate() stores keypoints in a float32 array, and recomputing pose and z-scored velocity
# from those keypoints amplifies the rounding (small velocity stds). Measured errors are ~1e-3;
# processing an input twice gives errors of order 1 or crashes.
SIMULATE_TOLERANCE = 1e-2
# keypoints -> pose -> keypoints, per group of keypoints: (names, tolerance on the mean distance,
# tolerance on the 99th-percentile distance), in mm (a fly is ~2.8 mm long). Body measurements the
# pose does not store come from the per-fly scale table (scale_perfly), looked up by the fly's
# identity: each individual fly's median thorax width and length, abdomen length, wing length,
# head width and head height over all of its own frames. How exactly a keypoint comes back depends
# on how the pose encodes it:
#   - leg points are stored directly, as an angle and a distance each, so they come back exactly;
#   - the antennae, eyes and front corners of the thorax are placed from the head base position
#     and angle in the frame, using that fly's own median head width, head height and thorax
#     width: measured mean 0.013, 99th pct 0.066;
#   - the base of the thorax, the abdomen tip and the wing tips are placed at that fly's own median
#     thorax, abdomen and wing length; the pose stores only their angles, so a frame where the
#     part looks longer or shorter than that fly's median (a raised or extended wing, a stretched
#     abdomen) comes back at the median. Measured mean / 99th pct: base of thorax 0.025 / 0.28,
#     abdomen tip 0.080 / 0.63, wing tips 0.095 / 0.98.
# A left/right swap of the head points (each landing on its partner) gives a mean of ~0.7.
KEYPOINT_PERCENTILE = 99
KEYPOINT_GROUP_TOLERANCES = {
    'legs': ([name for name in keypointnames if 'leg' in name or 'femur' in name], 1e-4, 1e-4),
    'head and front of thorax': (['antennae_midpoint', 'left_eye', 'right_eye', 'left_front_thorax',
                                  'right_front_thorax'], 0.03, 0.15),
    'base of thorax': (['base_thorax'], 0.05, 0.5),
    'abdomen tip': (['tip_abdomen'], 0.15, 1.0),
    'wing tips': (['wing_left', 'wing_right'], 0.15, 1.5),
}

# Movement over these numbers of frames is checked against the pose that many frames later.
FUTURE_OFFSETS = [1, 3, 10]
# Number of frames recomputed from a window of keypoints, and modified in the alignment test.
WINDOW_FRAMES = 120
# The alignment test replaces this relative pose feature with random values of this amplitude,
# as a fraction of the feature's standard deviation.
ALIGNMENT_FEATURE = 'left_middle_femur_base_dist'
ALIGNMENT_AMPLITUDE = 0.2
ALIGNMENT_SEED = 0
# Number of frames, spread evenly over the data, at which simulate()'s way of assembling inputs is
# checked. It recomputes sensory one frame at a time, as simulate() does (~2 ms per frame), so not
# every frame is checked.
N_INPUT_CHECK_FRAMES = 200
# Window simulated with the stand-in model: it starts here, burns in for contextl frames, then
# simulates this many frames.
SIMULATE_START_FRAME = 1000
N_SIMULATED_FRAMES = 20
# Number of chunks in the batch that is split back into named inputs and labels.
BATCH_SIZE = 4
# The single-agent checks use frames 1 up to this one, inside the first session of every fly.
SINGLE_AGENT_CHECK_END_FRAME = 100
# The discretized-label check is vacuous if it covers too few values.
MIN_FRACTION_CHECKED = 0.5


@functools.cache
def fly_data() -> dict:
    """Builds the training dataset from the small test file, once per test session.

    Returns:
        dict with
            config: the flyllm config.
            dataset: apf.dataset.Dataset.
            flyids: (n_frames, n_agents) int array, identity of each agent at each frame.
            track: Data, keypoints, (n_agents, n_frames, 2, n_keypoints) float32 array.
            pose: Data, pose features, (n_agents, n_frames, n_pose_features) float array.
            velocity: Data, movement from frame t to t + 1, (n_agents, n_frames,
                n_pose_features) float array; NaN where undefined.
            sensory: Data, (n_agents, n_frames, n_sensory_features) float array.
            isdata, isstart, useoutputmask: (n_frames, n_agents) bool arrays.
            scale_perfly: per-fly body scale used by the Pose operation.
    """
    config = read_config(str(CONFIG_FILE))
    # the test config names its data relative to the repository
    for key in ['datadir', 'intrainfile', 'invalfile']:
        config[key] = str(testing_utils.REPO_DIR / config[key])
    testing_utils.require_file(config['intrainfile'], 'fly test data')
    (dataset, flyids, track, pose, velocity, sensory, _, isdata, isstart, useoutputmask) = \
        fly_experiment.make_dataset(config, 'intrainfile', return_all=True, debug=False)
    scale_perfly = apf.dataset.get_operation(pose.operations, 'pose').scale_perfly
    return dict(config=config, dataset=dataset, flyids=flyids, track=track, pose=pose, velocity=velocity,
                sensory=sensory, isdata=isdata, isstart=isstart, useoutputmask=useoutputmask,
                scale_perfly=scale_perfly)


def wrap_angle_differences(difference: np.ndarray) -> np.ndarray:
    """Wraps the angle features of a pose difference into [-pi, pi).

    Args:
        difference: (..., n_pose_features) float array, difference of two poses.

    Returns:
        same shape, with the featangle columns wrapped.
    """
    difference = difference.copy()
    difference[..., featangle] = modrange(difference[..., featangle], -np.pi, np.pi)
    return difference


def longest_session(dataset):
    """Returns the apf.dataset.Session with the most consecutive valid frames."""
    return max(dataset.sessions, key=lambda session: session.duration)


def window_indata(data: dict, window: slice, keypoints: np.ndarray | None = None) -> dict:
    """Packs a window of frames in the form make_dataset(indata=...) expects.

    Args:
        data: output of fly_data().
        window: frames to include.
        keypoints: (n_agents, n_window_frames, 2, n_keypoints) float array to use instead of the
            data's keypoints, or None for the data's.

    Returns:
        dict with Xkp ((n_keypoints, 2, n_window_frames, n_agents)), flyids, isstart, isdata,
        useoutputmask ((n_window_frames, n_agents) each) and scale_perfly. The window's first
        frame starts a new track for every agent.
    """
    if keypoints is None:
        keypoints = data['track'].array[:, window]
    isstart = data['isstart'][window].copy()
    isstart[0] = True
    return {'Xkp': keypoints.T, 'flyids': data['flyids'][window], 'isstart': isstart,
            'isdata': data['isdata'][window], 'useoutputmask': data['useoutputmask'][window],
            'scale_perfly': data['scale_perfly']}


def test_loader_drops_extra_keypoints():
    """The loader drops extra keypoints.

    Inputs: `config`. Calls `experiments.flyllm.load_data`, which calls
    `apf.io.load_and_filter_data`, on the v3 test file twice, with `augment_flip` off and on, and
    checks that each time only the 19 keypoints the code uses remain. The v3 files append two
    outer wing points, and `simulate` writes the 19 keypoints rebuilt from each predicted pose back
    into the loaded arrays, so it crashes if they hold more. The other fly tests run with flip
    augmentation on only, so this is the test that checks the loader with flipping off.
    """
    config = copy.deepcopy(fly_data()['config'])
    for augment_flip in [False, True]:
        config['augment_flip'] = augment_flip
        keypoints = fly_experiment.load_data(config, config['intrainfile'])[0]   # (n_keypoints, 2, n_frames, n_agents)
        assert keypoints.shape[0] == N_CODE_KEYPOINTS, \
            f'augment_flip={augment_flip}: loaded {keypoints.shape[0]} keypoints, expected {N_CODE_KEYPOINTS}'


def test_pose_keypoint_round_trip():
    """Pose → keypoints → pose.

    Inputs: `pose` and `flyids`. Takes every fly's pose at every frame, converts it to keypoints
    with `Pose.invert()` which calls `feat2kp`, and converts those back with `Pose.apply()` which
    calls `kp2feat`, then compares the recovered pose with the original `pose`. Both conversions use
    that fly's own body measurements, looked up by its identity in `flyids` in the per-fly scale
    table stored with the `Pose` operation (see `test_keypoints_survive_pose_round_trip`). The
    data's keypoints (`track`) are not used. Passes if the recovered pose equals the original `pose`
    exactly, at every fly and frame (angles compared modulo 2π).
    """
    data = fly_data()
    pose, flyids = data['pose'], data['flyids']
    pose_op = apf.dataset.get_operation(pose.operations, 'pose')
    original = pose.array                                     # (n_agents, n_frames, n_pose_features)
    keypoints = pose_op.invert(original, flyid=flyids)        # (n_agents, n_frames, 2, n_keypoints)
    recovered = pose_op.apply(keypoints, scale_perfly=pose_op.scale_perfly, flyid=flyids)
    valid = ~np.isnan(original).any(-1)
    assert valid.any(), 'no valid frames to check'
    error = np.abs(wrap_angle_differences(recovered - original))[valid]
    assert error.max() < EXACT_TOLERANCE, f'pose round trip error {error.max():.3g}'


def test_keypoints_survive_pose_round_trip():
    """Keypoints → pose → keypoints.

    Inputs: `track`, `pose` and `flyids`. `pose` was computed from `track` inside `make_dataset` by
    `Pose.apply()`, which calls `kp2feat`. The test converts `pose` back to keypoints with
    `Pose.invert()`, which calls `feat2kp`, and measures the distance from each reconstructed
    keypoint to the corresponding one in `track`. There is a loss of information from keypoints
    (38-d) to pose (29-d); in particular the lengths of various body parts are dropped. Each
    individual fly has its own entry in a per-fly scale table, `scale_perfly`: its median thorax
    width and length, abdomen length, wing length, head width and head height
    (`flyllm.features.compute_scale_perfly`). Keypoints → pose uses only the thorax length, to place
    the base and the middle of the thorax on the body axis, from which the angles of the abdomen,
    back legs, middle femurs and wings are measured. Pose → keypoints uses all six of the same fly's
    values to rebuild the points the pose does not store.

    How closely a keypoint can come back therefore depends on how the pose encodes it, so each group
    of keypoints has its own limits on the average distance and on the 99th-percentile distance:

    | keypoints | how the pose encodes them | measured mean / 99th pct | limits |
    |---|---|---|---|
    | leg points (10) | an angle and a distance each, measured in that frame | 0 / 0 | 1e-4 / 1e-4 mm |
    | antennae, eyes, front corners of the thorax | head base position and angle from the frame, with that fly's own median head and thorax widths and head height | 0.013 / 0.066 mm | 0.03 / 0.15 mm |
    | base of the thorax | that fly's own median thorax length | 0.025 / 0.28 mm | 0.05 / 0.5 mm |
    | abdomen tip | the abdomen's angle in that frame, at that fly's own median abdomen length | 0.080 / 0.63 mm | 0.15 / 1.0 mm |
    | wing tips | each wing's angle in that frame, at that fly's own median wing length | 0.095 / 0.98 mm | 0.15 / 1.5 mm |
    """
    data = fly_data()
    pose, track, flyids = data['pose'], data['track'], data['flyids']
    pose_op = apf.dataset.get_operation(pose.operations, 'pose')
    keypoints = pose_op.invert(pose.array, flyid=flyids)                      # (n_agents, n_frames, 2, n_keypoints)
    true_keypoints = track.array[..., :N_CODE_KEYPOINTS]
    distance = np.linalg.norm(keypoints - true_keypoints, axis=-2)            # (n_agents, n_frames, n_keypoints)
    distance = distance[~np.isnan(distance).any(-1)]                          # (n_valid_fly_frames, n_keypoints)
    assert len(distance) > 0, 'no valid frames to check'
    grouped = [name for names, _, _ in KEYPOINT_GROUP_TOLERANCES.values() for name in names]
    assert sorted(grouped) == sorted(keypointnames), 'every keypoint must be in exactly one group'
    for group, (names, mean_tolerance, percentile_tolerance) in KEYPOINT_GROUP_TOLERANCES.items():
        group_distance = distance[:, [keypointnames.index(name) for name in names]]
        assert group_distance.mean() < mean_tolerance, \
            f'{group}: mean distance {group_distance.mean():.3g} mm >= {mean_tolerance}'
        percentile = np.percentile(group_distance, KEYPOINT_PERCENTILE)
        assert percentile < percentile_tolerance, \
            f'{group}: {KEYPOINT_PERCENTILE}th percentile distance {percentile:.3g} mm >= {percentile_tolerance}'


def test_velocity_round_trip():
    """Velocity → pose.

    Inputs: `velocity`, `pose` and `dataset.sessions`. Selects contiguous frames based on the
    sessions in `dataset` — the stretches of frames in which the fly is continuously tracked and
    every input and label is defined (see the module docstring). For each session, it calls
    `Velocity.invert()` starting from the real pose at the session's first frame
    (`x0=true_pose[:1]`), which adds up the frame-to-frame movements: `GlobalVelocity.invert()`
    rotates forward/sideways movement back into arena coordinates and adds up the orientation
    changes, and `LocalVelocity.invert()` adds up the changes in the relative features. This
    reconstructs the pose at every frame of the session. Passes if the reconstructed pose at each
    frame equals `pose` at that frame exactly. This would catch errors in the velocity computation,
    in the direction of rotation, or in wrapping angles. Simulation does exactly this to turn
    predicted movement into pose.
    """
    data = fly_data()
    pose, velocity = data['pose'], data['velocity']
    velocity_op = apf.dataset.get_operation(velocity.operations, 'velocity')
    for session in data['dataset'].sessions:
        frames = slice(session.start_frame, session.start_frame + session.duration)
        true_pose = pose.array[session.agent_id, frames]       # (n_session_frames, n_pose_features)
        recovered = velocity_op.invert(velocity.array[session.agent_id, frames][None], x0=true_pose[:1])[0]
        error = np.abs(wrap_angle_differences(recovered - true_pose))
        assert error.max() < EXACT_TOLERANCE, \
            f'agent {session.agent_id}, session at frame {session.start_frame}: error {error.max():.3g}'


def test_global_velocity_future_offsets():
    """Movement over several frames.

    Inputs: `pose` (its 3 global features: x, y, orientation) and `isstart`. Calls
    `GlobalVelocity(tspred=[1, 3, 10]).apply()` to compute each fly's movement from frame t to t +
    1, t + 3 and t + 10 — forward and sideways in the fly's own frame at t, and the change in
    orientation — at every frame t; `isstart` makes movements that would cross into a new track NaN,
    and those are skipped. For each offset, applies that movement to the pose at t as a single step
    with `GlobalVelocity(tspred=[1]).invert()` and compares where it lands with the real pose at t +
    offset. Passes if the landing pose equals `pose` at frame t + offset exactly, for every fly,
    frame and offset. This mirrors `debug_fly_example`'s check of predictions several frames into
    the future. The current config predicts only one frame ahead; this covers the multi-frame option
    (`tspred_global`).
    """
    data = fly_data()
    position = data['pose'].array[..., :3]                    # (n_agents, n_frames, 3): x, y, orientation
    movement = apf.dataset.GlobalVelocity(tspred=FUTURE_OFFSETS).apply(position, isstart=data['isstart'])
    single_step = apf.dataset.GlobalVelocity(tspred=[1])
    frames = np.arange(position.shape[1] - max(FUTURE_OFFSETS))
    for i, offset in enumerate(FUTURE_OFFSETS):
        step = movement[:, frames, 3 * i:3 * i + 3]            # (n_agents, n_checked_frames, 3)
        defined = ~np.isnan(step).any(-1)
        step, start = step[defined], position[:, frames][defined]     # (n_defined, 3) each
        # one step from start, then a dummy zero step: frame 1 of the inverse is where it lands
        landed = single_step.invert(np.stack([step, np.zeros_like(step)], axis=1), x0=start)[:, 1]
        target = position[:, frames + offset][defined]
        error = np.abs(landed - target)
        error[:, 2] = np.abs(modrange(landed[:, 2] - target[:, 2], -np.pi, np.pi))
        assert error.max() < EXACT_TOLERANCE, f'offset {offset}: error {error.max():.3g}'


def test_labels_invert_to_velocity():
    """Labels → velocity.

    Inputs: `dataset.labels['velocity']` and `velocity`. Works on the dataset's whole label array —
    every fly at every frame — not on chunks: undoing the labels only as far as velocity needs no
    per-frame information, so no chunk metadata is involved (compare
    `test_labels_invert_to_chunk_pose`). The labels were made from `velocity` by the operations
    `Zscore` and then `Fusion`, which applies `Discretize` to the 3 global features and `Identity`
    to the 26 relative ones. The test undoes them with `apf.dataset.invert_to_named(labels,
    'velocity', ...)`, which calls `Fusion.invert()` — `Discretize.invert(do_sampling=False)`, the
    probability-weighted average of the bin centers, and `Identity.invert()` — and then
    `Zscore.invert()`, and compares the decoded movement with `velocity`. The 26 continuous features
    must equal `velocity` exactly. The discretized ones cannot: a value is recovered only to within
    roughly a bin. For values at least two bins from either end, the error must be less than the
    width of the widest interior bin. The two end bins are excluded because they are stretched to
    cover outliers, and at least half of all values must be checked so the test cannot pass
    vacuously. This would catch bins being mixed up, wrong z-score parameters, or discrete and
    continuous columns being confused.
    """
    data = fly_data()
    labels, velocity = data['dataset'].labels['velocity'], data['velocity']
    fusion = labels.operations[-1]
    assert isinstance(fusion, apf.dataset.Fusion), f'expected the labels to end in a Fusion, got {fusion.name}'
    # Discretize sits inside the Fusion, so its do_sampling goes through Fusion.invert.
    recovered = apf.dataset.invert_to_named(labels, 'velocity', fusion=testing_utils.no_sampling_invert_kwargs(fusion))
    defined = ~np.isnan(velocity.array).any(-1)               # (n_agents, n_frames)

    zscore = apf.dataset.get_operation(labels.operations, 'zscore')
    for op, feature_idx in zip(fusion.operations, fusion.indices_per_op):
        if isinstance(op, apf.dataset.Discretize):
            for bin_idx, feature in enumerate(feature_idx):
                true_zscored = (velocity.array[..., feature] - zscore.mean[feature]) / zscore.std[feature]
                decoded_zscored = (recovered[..., feature] - zscore.mean[feature]) / zscore.std[feature]
                errors, fraction_checked, max_width = testing_utils.interior_decoding_errors(
                    true_zscored[defined], decoded_zscored[defined], op.bin_edges[bin_idx])
                assert fraction_checked > MIN_FRACTION_CHECKED, \
                    f'feature {feature}: only {fraction_checked:.2f} of values checked'
                assert errors.max() <= max_width, \
                    f'feature {feature}: decoding error {errors.max():.3g} > widest interior bin {max_width:.3g}'
        else:
            error = np.abs(recovered[..., feature_idx] - velocity.array[..., feature_idx])[defined]
            assert error.max() < EXACT_TOLERANCE, f'continuous label inversion error {error.max():.3g}'


def test_labels_invert_to_chunk_pose():
    """A chunk's labels → its pose and keypoints.

    Inputs: `dataset`, `pose` and `track`. Unlike `test_labels_invert_to_velocity`, this works on
    one chunk in the form a model, the loss and the plots see it, and undoes the labels further, to
    pose and keypoints. That needs information only a chunk carries: the starting pose for adding up
    movements and the fly's identity for its body scale. Takes the first training chunk
    (`dataset.get_chunk()` at `dataset.chunk_indices[0]`) and converts it with
    `dataset.item_to_data()`, which attaches the metadata the dataset stores with each chunk: the
    true pose and the fly's identity at each of its frames. Undoes the label processing with
    `apf.dataset.invert_to_named(labels, 'pose')` — `Fusion.invert()` (discretized features decoded
    by sampling), `Zscore.invert()`, then `Velocity.invert()` starting from the stored pose at the
    chunk's first frame — and all the way to keypoints with
    `apf.dataset.apply_inverse_operations(labels)`, which adds `Pose.invert()` using the stored
    identity. This is the path the debug plots in `flyllm/plotting.py` use. Passes if the pose at
    the chunk's first frame equals the true pose in `pose` exactly, the 26 relative features equal
    the true ones at every frame (to float32 precision), and the keypoints at the first frame equal
    those `Pose.invert()` gives for the true pose. The global features after the first frame are not
    checked, because each frame's discretization error accumulates as movements are added up;
    `test_labels_invert_to_velocity` bounds the error of each step. This would catch the stored
    metadata being wrong or shifted relative to the chunk (shifting it by one frame makes the test
    fail), or a break anywhere in the chain of inverse operations.
    """
    data = fly_data()
    dataset, pose = data['dataset'], data['pose']
    start_frame, agent = dataset.chunk_indices[0]
    frames = slice(start_frame, start_frame + dataset.context_length)
    labels = dataset.item_to_data(dataset.get_chunk(start_frame, dataset.context_length, agent))['labels']['velocity']

    from_labels = apf.dataset.invert_to_named(labels, 'pose')  # (context_length, n_pose_features)
    true_pose = pose.array[agent, frames]
    error = np.abs(wrap_angle_differences(from_labels - true_pose))
    assert error[0].max() < EXACT_TOLERANCE, f'first-frame pose error {error[0].max():.3g}'
    assert error[:, featrelative].max() < FLOAT32_TOLERANCE, f'relative pose error {error[:, featrelative].max():.3g}'

    keypoints = np.asarray(apf.dataset.apply_inverse_operations(labels)).reshape(
        data['track'].array[agent, frames][..., :N_CODE_KEYPOINTS].shape)    # (context_length, 2, n_keypoints)
    pose_op = apf.dataset.get_operation(pose.operations, 'pose')
    true_keypoints = np.asarray(pose_op.invert(true_pose, flyid=labels.invertdata['pose'])).reshape(keypoints.shape)
    error = np.abs(keypoints[0] - true_keypoints[0]).max()
    assert error < EXACT_TOLERANCE, f'first-frame keypoints differ by {error:.3g}'


def test_chunks_have_defined_targets():
    """No undefined targets in training chunks.

    Inputs: `dataset.chunk_indices` and `velocity`. For every one of the 2,820 training chunks —
    which `Dataset` cut from its sessions (`compute_sessions`, `compute_chunk_indices`) when it was
    built — checks that `velocity` is defined at each of its 65 frames, i.e. that the movement each
    frame's label encodes exists. This would catch the model being trained to predict an undefined
    movement, such as out of the last frame of a track. (For flies those frames are also excluded by
    their continuous labels being NaN; the synthrat version of this test is the one that depends on
    the change to `discretize_labels`.)
    """
    data = fly_data()
    testing_utils.assert_chunks_have_defined_targets(data['dataset'], data['velocity'].array)


def test_window_features_match_dataset():
    """Features from a window of keypoints = the full dataset's.

    Inputs: `config`, `dataset`, `track`, `flyids`, `isstart`, `isdata`, `useoutputmask` and the
    scale table. Takes the first 120 frames of the longest session (see `test_velocity_round_trip`)
    and calls `make_dataset(config, 'intrainfile', ref_dataset=dataset, indata=...)` with only those
    frames of every fly, the window's first frame marked as the start of every track. With a
    reference dataset, `make_dataset` computes `Sensory`, `Pose` and `Velocity` from the keypoints,
    then `apply_opers_from_data` applies the rest of the reference's operations (`Roll`, `Subset`,
    `Zscore`, `Fusion` with `Discretize`) with the reference's z-score parameters and bins — the way
    validation data is built. Compares every input and label with `dataset`'s on the same frames:
    they must be identical, with NaN in the same places. Two frames differ by design and are checked
    separately: the window's first frame has no movement into it (no earlier frame in the window),
    and its last frame has no movement out of it. This mirrors `debug_fly_example`'s comparison of
    an example taken from the dataset with one built directly from keypoints. It would catch a
    feature depending on frames it should not, or the reference dataset's parameters not being
    reused.
    """
    data = fly_data()
    dataset = data['dataset']
    session = longest_session(dataset)
    window = slice(session.start_frame, session.start_frame + min(session.duration, WINDOW_FRAMES))
    from_window = fly_experiment.make_dataset(data['config'], 'intrainfile', ref_dataset=dataset,
                                              indata=window_indata(data, window))
    n_frames = window.stop - window.start
    for key, expected in dataset.inputs.items():
        testing_utils.assert_arrays_match(from_window.inputs[key].array[:, 1:], expected.array[:, window][:, 1:],
                                          EXACT_TOLERANCE, f"input '{key}'")
    for key, expected in dataset.labels.items():
        testing_utils.assert_arrays_match(from_window.labels[key].array[:, :n_frames - 1],
                                          expected.array[:, window][:, :n_frames - 1], EXACT_TOLERANCE, f"label '{key}'")
    agent = session.agent_id
    assert np.isnan(from_window.inputs['velocity'].array[agent, 0]).all(), 'first frame should have no velocity input'
    assert np.isnan(from_window.labels['velocity'].array[agent, -1]).all(), 'last frame should have no target'


def test_dataset_rebuilt_from_saved_params():
    """Dataset rebuilt from saved parameters.

    Inputs: `config` and `dataset`. Calls `dataset.get_params()` — each operation's parameters as a
    dict (`Operation.to_dict()`), which is what a saved model stores — puts the result in
    `config['dataset_params']`, and calls `make_dataset` again on the same data file. With
    `dataset_params` set, `make_dataset` applies the operations rebuilt from those dicts with
    `apply_opers_from_data_params`. Every input, label and chunk must be identical to `dataset`'s.
    This would catch parameters being lost or changed when a model is saved and reloaded, which
    would make the reloaded model receive inputs processed differently from those it was trained on.
    """
    data = fly_data()
    dataset = data['dataset']
    config = copy.deepcopy(data['config'])
    config['dataset_params'] = dataset.get_params()
    rebuilt = fly_experiment.make_dataset(config, 'intrainfile', debug=False)
    for group in ['inputs', 'labels']:
        for key, original in getattr(dataset, group).items():
            testing_utils.assert_arrays_match(getattr(rebuilt, group)[key].array, original.array, EXACT_TOLERANCE,
                                              f"{group} '{key}'")
    assert np.array_equal(rebuilt.chunk_indices, dataset.chunk_indices), 'chunks differ'


def test_inputs_and_labels_time_alignment():
    """Inputs and labels line up in time.

    Inputs: `config`, `dataset`, `pose`, `track` and `flyids`, plus the window's `isstart`, `isdata`
    and `useoutputmask`. Takes the same 120-frame window. For that session's fly, replaces one
    relative pose feature (the left middle femur base distance) with random values p[t] (within ±20%
    of the feature's standard deviation), converts that fly's modified poses to keypoints with
    `Pose.invert()`, and calls `make_dataset(..., ref_dataset=dataset, indata=...)` on the modified
    keypoints. Then undoes the processing of each input and label with
    `apf.dataset.invert_to_named`: the pose and velocity inputs back to before z-scoring
    (`Zscore.invert()`), and the labels back to raw velocity (`Fusion.invert()` without sampling,
    then `Zscore.invert()`). Reading off the modified feature: the pose input at frame t must be
    p[t]; the velocity input at t must be p[t] − p[t−1], the movement into t; the label at t must be
    p[t+1] − p[t], the movement out of t. Passes if each of the three, read from the dataset, equals
    its value computed directly from p (p[t], p[t] − p[t−1], p[t+1] − p[t]) to within 1e-4. The test
    also checks that the random values change by more than 100 times that from frame to frame, so a
    one-frame shift cannot go unnoticed. This mirrors `debug_fly_example`'s test that writes the
    frame number into a feature. It would catch inputs or labels being off by a frame — for example,
    if the velocity input were not shifted (`Roll`), the model would be given the very movement it
    is asked to predict (making that change makes this test fail).
    """
    data = fly_data()
    dataset, pose, flyids = data['dataset'], data['pose'], data['flyids']
    feature = posenames.index(ALIGNMENT_FEATURE)
    assert featrelative[feature] and not featangle[feature], f'{ALIGNMENT_FEATURE} must be a relative, non-angle feature'
    session = longest_session(dataset)
    agent = session.agent_id
    window = slice(session.start_frame, session.start_frame + min(session.duration, WINDOW_FRAMES))

    # replace the feature for one agent and rebuild its keypoints
    rng = np.random.default_rng(ALIGNMENT_SEED)
    modified_pose = pose.array[agent:agent + 1, window].copy()                # (1, n_frames, n_pose_features)
    amplitude = ALIGNMENT_AMPLITUDE * np.nanstd(pose.array[..., feature])
    values = modified_pose[0, 0, feature] + amplitude * rng.uniform(-1, 1, modified_pose.shape[1])   # p[t], (n_frames,)
    modified_pose[0, :, feature] = values
    pose_op = apf.dataset.get_operation(pose.operations, 'pose')
    keypoints = data['track'].array[:, window].copy()                         # (n_agents, n_frames, 2, n_keypoints)
    # any extra keypoints beyond the code's keep their tracked values; nothing uses them
    keypoints[agent, ..., :N_CODE_KEYPOINTS] = pose_op.invert(modified_pose, flyid=flyids[window][:, agent:agent + 1])[0]
    modified = fly_experiment.make_dataset(data['config'], 'intrainfile', ref_dataset=dataset,
                                           indata=window_indata(data, window, keypoints))

    movement = np.diff(values)                                                # movement[t] = p[t+1] - p[t]
    assert np.abs(np.diff(movement)).max() > 100 * FLOAT32_TOLERANCE, 'values vary too little to detect a shift'
    velocity_op = apf.dataset.get_operation(modified.inputs['velocity'].operations, 'velocity')
    # Velocity output: the GlobalVelocity columns, then one column per relative feature
    velocity_column = velocity_op.fusion.dims_per_op[0] + list(velocity_op.local_inds).index(feature)
    pose_column = list(np.nonzero(featrelative)[0]).index(feature)            # Subset keeps the relative features

    pose_input = apf.dataset.invert_to_named(modified.inputs['pose'], 'subset')[agent, :, pose_column]
    velocity_input = apf.dataset.invert_to_named(modified.inputs['velocity'], 'roll')[agent, :, velocity_column]
    fusion = modified.labels['velocity'].operations[-1]
    label = apf.dataset.invert_to_named(modified.labels['velocity'], 'velocity',
                                        fusion=testing_utils.no_sampling_invert_kwargs(fusion))[agent, :, velocity_column]
    for description, actual, expected in [('pose input at t vs p[t]', pose_input, values),
                                          ('velocity input at t vs p[t] - p[t-1]', velocity_input[1:], movement),
                                          ('label at t vs p[t+1] - p[t]', label[:-1], movement)]:
        error = np.abs(actual - expected).max()
        assert error < FLOAT32_TOLERANCE, f'{description}: error {error:.3g}'


def test_batches_split_back_to_dataset():
    """Batches split back into named inputs and labels.

    Inputs: `dataset`. Takes the first batch of 4 chunks from `apf.dataset.DataLoader(dataset)`, in
    which all inputs are concatenated into one array and the labels are split into a continuous
    array and a flattened discrete array, converts it to numpy (`apf.utils.convert_torch_to_numpy`),
    and splits it back into named inputs (velocity, pose, sensory) and labels (velocity) with
    `dataset.item_to_data()`, which calls `split_input_by_names()` and `split_output_by_names()`, as
    the loss, the plots and simulation do. Each named input and label of batch item i must equal the
    dataset's own array for it (e.g. `dataset.inputs['velocity'].array`) for that chunk's fly at
    that chunk's frames, to float32 precision. It also rearranges the discrete labels into the shape
    a model outputs (one row of bin probabilities per feature) and checks that
    `split_output_by_names()` gives exactly the same named labels from them as from the batch, and
    checks that `dataset.get_chunk()` for 32 frames starting 16 frames into a 65-frame chunk equals
    frames 16–47 of the 65-frame chunk, exactly. This would catch errors in the bookkeeping of which
    columns belong to which input or label, and which are discrete.
    """
    testing_utils.assert_batches_split_back_to_dataset(fly_data()['dataset'], BATCH_SIZE)


def test_input_assembly_matches_dataset():
    """Assembling one frame's inputs the way simulation does.

    Inputs: `dataset`, `track`, `pose`, `velocity` and `isdata`. At 200 frames
    spread evenly over the data, including the mirrored half, builds each frame's inputs the way
    `simulate` builds them: the movement into the frame from `velocity`, the pose at the frame from
    `pose`, and sensory recomputed from that frame's keypoints in `track` alone, with the dataset's
    `Sensory` operation's `apply()` (`experiments.flyllm.Sensory`, which calls
    `compute_sensory_wrapper`). Passes them as raw arrays to `apply_opers_from_data(dataset.inputs,
    ...)`, which must apply only the steps not yet applied: `Roll` and `Zscore` for velocity,
    `Subset` (the relative features) and `Zscore` for pose, `Zscore` for sensory. For every fly
    tracked at that frame (`isdata`), each processed input must equal the dataset's stored input for
    that fly and frame (`dataset.inputs[key].array[fly, t]`) exactly, with NaN in the same places.
    Flies not tracked at the frame are left out: the dataset sets all their inputs to NaN, whereas
    sensory recomputed from one frame gives values for their other-flies features; `simulate` only
    feeds the model flies that are tracked. Not every frame is checked because recomputing sensory
    one frame at a time, as `simulate` does, takes ~2 ms per frame (40 s for all 20,000). This
    catches `apply_opers_from_data` re-applying steps that were already applied to a raw array, such
    as keypoints → pose applied to pose features, which crashes in `kp2feat`.
    """
    data = fly_data()
    dataset, track, pose, velocity = data['dataset'], data['track'], data['pose'], data['velocity']
    sensory_op = apf.dataset.get_operation(dataset.inputs['sensory'].operations, 'sensory')
    # frame 0 has no movement into it
    frames = np.linspace(1, pose.array.shape[1] - 1, N_INPUT_CHECK_FRAMES).astype(int)
    for t in frames:
        # simulate() passes the movement into frame t, the pose at t, and sensory recomputed
        # from the keypoints at t, each (n_agents, 1, n_features).
        raw_inputs = {'velocity': velocity.array[:, t - 1:t],
                      'pose': pose.array[:, t:t + 1],
                      'sensory': sensory_op.apply(track.array[:, t:t + 1])}
        processed = apf.dataset.apply_opers_from_data(dataset.inputs, raw_inputs)
        tracked = data['isdata'][t]                                               # (n_agents,)
        for key, expected in dataset.inputs.items():
            assembled = getattr(processed[key], 'array', processed[key])[:, 0]   # (n_agents, n_features)
            testing_utils.assert_arrays_match(assembled[tracked], expected.array[tracked, t], EXACT_TOLERANCE,
                                              f"frame {t}, input '{key}'")


def test_simulate_inputs_match_pipeline():
    """Simulation's inputs match the training pipeline.

    Inputs: `config`, `dataset`, `track`, `pose`, `velocity`, `flyids`, `isdata` and `isstart`.
    Calls `apf.simulation.simulate` from frame 1000 for every fly tracked throughout the window as a
    single identity (9 of the 10): 65 burn-in frames of real data, then 20 simulated frames, with
    the stand-in model, which returns the true labels taken from `dataset.get_chunk()`. Checks that
    no keypoints are NaN, and that the inputs `simulate` gave the model for the 65 burn-in frames
    equal the dataset's inputs for those frames (`dataset.get_chunk()`) exactly. Then runs the
    training processing on the keypoints `simulate` produced — the dataset's own `Pose`, `Velocity`
    and `Sensory` operations, then `apply_opers_from_data` for the rest (`Roll`, `Subset`, `Zscore`)
    — and compares with the inputs `simulate` actually gave the model at each simulated frame.
    Passes if, at every simulated frame, the inputs `simulate` gave the model equal these recomputed
    inputs to within 1e-2: `simulate` stores keypoints as float32, and recomputing pose and z-scored
    velocity from the rounded keypoints gives differences up to ~1e-3, whereas inputs processed the
    wrong way differ by order 1 or crash. This would catch any inconsistency between how simulation
    builds inputs and how the training data was built.
    """
    data = fly_data()
    config, dataset, track, pose, flyids = (data[key] for key in ['config', 'dataset', 'track', 'pose', 'flyids'])
    burn_in = config['contextl']
    n_frames = burn_in + N_SIMULATED_FRAMES
    window = slice(SIMULATE_START_FRAME, SIMULATE_START_FRAME + n_frames)
    # simulate every agent that is tracked throughout the window as a single identity
    agent_ids = np.nonzero(data['isdata'][window].all(0) & ~data['isstart'][window][1:].any(0))[0]
    assert len(agent_ids) > 0, 'no agent is tracked throughout the simulated window'

    model = testing_utils.TrueLabelModel(testing_utils.true_labels(dataset, SIMULATE_START_FRAME, n_frames, agent_ids))
    gt_track, pred_track = apf.simulation.simulate(
        dataset=dataset, model=model, track=track, pose=pose, identities=flyids, track_len=n_frames,
        burn_in=burn_in, max_contextl=None, agent_ids=agent_ids, start_frame=SIMULATE_START_FRAME)
    assert not np.isnan(pred_track[agent_ids]).any(), 'simulation produced NaN keypoints'
    recorded = model.last_input                                # (n_simulated_agents, n_frames - 1, d_input)

    # burn-in frames are copied from the data
    burn_in_inputs = np.stack([dataset.get_chunk(SIMULATE_START_FRAME, n_frames, a)['input'] for a in agent_ids])
    assert np.array_equal(recorded[:, :burn_in], burn_in_inputs[:, :burn_in]), 'burn-in inputs differ from the data'

    # simulated frames: run the dataset's operations on the simulated keypoints
    pose_op = apf.dataset.get_operation(pose.operations, 'pose')
    velocity_op = apf.dataset.get_operation(data['velocity'].operations, 'velocity')
    sensory_op = apf.dataset.get_operation(dataset.inputs['sensory'].operations, 'sensory')
    simulated_track = apf.dataset.Data('keypoints', pred_track, [])   # (n_agents, n_frames, 2, n_keypoints)
    simulated_pose = pose_op(simulated_track, scale_perfly=pose_op.scale_perfly, flyid=flyids[window])
    expected_parts = apf.dataset.apply_opers_from_data(dataset.inputs, {
        'velocity': velocity_op(simulated_pose, isstart=None),
        'pose': simulated_pose,
        'sensory': sensory_op(simulated_track)})
    # (n_simulated_agents, n_frames, d_input)
    expected = np.concatenate([expected_parts[key].array for key in dataset.inputs], axis=-1)[agent_ids]
    # Column range of each input key in the concatenation. (dataset.input_idx splits sensory
    # further, so it is not used here.)
    offsets = np.cumsum([0] + [d.array.shape[-1] for d in dataset.inputs.values()])
    simulated = slice(burn_in, n_frames - 1)
    for key, start, stop in zip(dataset.inputs, offsets[:-1], offsets[1:]):
        error = np.abs(recorded[:, simulated, start:stop] - expected[:, simulated, start:stop]).max()
        assert error < SIMULATE_TOLERANCE, f"input '{key}': simulate's input differs from the pipeline by {error:.3g}"


def test_feature_names_match_dimensions():
    """Feature names.

    Inputs: `dataset`. Calls `dataset.get_input_names()` and `get_label_names()`, which prefix each
    input's and label's feature names — set by each operation's `update_feature_names()` as it was
    applied — with its key, e.g. `sensory__…`. These names label plots and analyses. Passes if there
    is exactly one name per column, each starting with the key of the input or label it belongs to.
    This would catch names falling out of step with the columns.
    """
    testing_utils.assert_feature_names_match_dimensions(fly_data()['dataset'])


def test_single_agent_matches_batch():
    """One agent gives the same result as a batch.

    Inputs: `dataset.labels['velocity']`, `pose`, `velocity`, `track`, `flyids` and `isstart`. For
    agent 0 alone and for all agents at once (frames 1–99, inside every fly's first session), runs
    `Fusion.invert` on the labels with `do_sampling=False` passed to its `Discretize`,
    `Velocity.apply` with `isstart`, `Velocity.invert` with a starting pose `x0`, `Pose.apply` with
    per-frame identities, and `Pose.invert` with per-frame identities and with one fly's identity as
    a single int. It also inverts the whole batch with a pose for every frame as `x0` and compares
    with inverting each agent alone. Passes if, for every call, the result for agent 0 alone equals
    the result of the same call on the array with all agents, indexed at agent 0
    (`result_for_all_agents[0]`), exactly, with NaN in the same places; for the whole-batch
    inversion, the batch result equals the stacked single-agent results. This catches an operation
    that mishandles the agent axis for one agent: adding it to arguments that have none (such as
    `do_sampling=False`, which crashes), adding it to `isstart` as a row instead of an
    (n_frames, 1) column (track starts after the first are ignored, so NaN goes in the wrong
    places), taking the first agent instead of each agent's first frame from a per-frame `x0`, or
    returning an extra leading axis.
    """
    data = fly_data()
    labels, pose, velocity, isstart = data['dataset'].labels['velocity'], data['pose'], data['velocity'], data['isstart']
    track, flyids = data['track'], data['flyids']
    agent = 0
    frames = slice(1, SINGLE_AGENT_CHECK_END_FRAME)        # inside the first session, so x0 and identity are defined
    fusion = labels.operations[-1]
    kwargs = testing_utils.no_sampling_invert_kwargs(fusion)
    velocity_op = apf.dataset.get_operation(velocity.operations, 'velocity')
    pose_op = apf.dataset.get_operation(pose.operations, 'pose')
    one_fly_identity = int(flyids[frames.start, agent])
    batch_inverted = velocity_op.invert(velocity.array[:, frames], x0=pose.array[:, frames])   # per-frame x0
    pairs = {
        'Fusion.invert with do_sampling=False': (fusion.invert(labels.array[agent], **kwargs),
                                                 fusion.invert(labels.array, **kwargs)[agent]),
        'Velocity.apply with isstart': (velocity_op.apply(pose.array[agent], isstart=isstart[:, agent]),
                                        velocity_op.apply(pose.array, isstart=isstart)[agent]),
        'Velocity.invert with x0': (velocity_op.invert(velocity.array[agent, frames], x0=pose.array[agent, frames.start]),
                                    velocity_op.invert(velocity.array[:, frames], x0=pose.array[:, frames.start])[agent]),
        'Velocity.invert with a per-frame x0, every agent of a batch':
            (np.stack([velocity_op.invert(velocity.array[a, frames], x0=pose.array[a, frames])
                       for a in range(velocity.array.shape[0])]), batch_inverted),
        'Pose.apply with per-frame identities': (
            pose_op.apply(track.array[agent, frames], scale_perfly=pose_op.scale_perfly, flyid=flyids[frames, agent]),
            pose_op.apply(track.array[:, frames], scale_perfly=pose_op.scale_perfly, flyid=flyids[frames])[agent]),
        'Pose.invert with per-frame identities': (pose_op.invert(pose.array[agent, frames], flyid=flyids[frames, agent]),
                                                   pose_op.invert(pose.array[:, frames], flyid=flyids[frames])[agent]),
        'Pose.invert with one identity': (pose_op.invert(pose.array[agent, frames], flyid=one_fly_identity),
                                          pose_op.invert(pose.array[:, frames], flyid=flyids[frames])[agent]),
    }
    for description, (single, batched) in pairs.items():
        assert np.array_equal(single, batched, equal_nan=True), f'{description}: single agent differs from batch'


def test_operations_handle_single_agent():
    """Every operation handles one agent.

    Inputs: `dataset`, `pose`, `velocity` and `sensory` (frames 1–99). For each input and label,
    starting after the pose, velocity and sensory computations — `Roll` and `Zscore` for the
    velocity input, `Subset` and `Zscore` for the pose input, `Zscore` for the sensory input,
    `Zscore` and `Fusion` for the labels — applies each operation in turn to all flies and to fly 0
    alone, and checks with `testing_utils.assert_operations_handle_single_agent` that the result for
    fly 0 alone equals the result of the same operation on the array with all flies, indexed at fly
    0 (`result_for_all_flies[0]`): same shape, NaN in the same places, identical values; and the
    same for each operation's `invert` (except `Subset`, which has none), decoding discretized
    values without sampling. `Pose` and `Velocity` are covered by `test_single_agent_matches_batch`,
    with their arguments. `Sensory` is left out on purpose, since a fly alone sees no other flies
    (`test_sensory_of_a_lone_fly`). An operation that returns an extra leading axis for one agent
    fails here, e.g. (1, 99, 29) instead of (99, 29).
    """
    data = fly_data()
    dataset = data['dataset']
    frames = slice(1, SINGLE_AGENT_CHECK_END_FRAME)
    pose, velocity, sensory = (data[key].array[:, frames] for key in ['pose', 'velocity', 'sensory'])
    chains = [
        ('velocity input', velocity, apf.dataset.get_post_operations(dataset.inputs['velocity'].operations, 'velocity')),
        ('pose input', pose, apf.dataset.get_post_operations(dataset.inputs['pose'].operations, 'pose')),
        ('sensory input', sensory, apf.dataset.get_post_operations(dataset.inputs['sensory'].operations, 'sensory')),
        ('velocity labels', velocity, apf.dataset.get_post_operations(dataset.labels['velocity'].operations, 'velocity')),
    ]
    testing_utils.assert_operations_handle_single_agent(chains)


def test_sensory_of_a_lone_fly():
    """Sensory of a lone fly.

    Inputs: `track` (frames 1–99) and the dataset's `Sensory` operation. Computes sensory for fly 0
    given alone, (n_frames, 2, n_keypoints), and checks that it equals sensory computed for a batch
    containing only fly 0, (1, n_frames, 2, n_keypoints), indexed at 0, exactly; that its wall
    features equal fly 0's wall features computed from all the flies' keypoints, exactly; and that
    every other-fly vision and touch feature is 0, the maximum-distance value. It also checks that
    with the other flies present, fly 0 does see some of them (vision), so the zero check is not
    vacuous; touch is left out of that, since flies are rarely close enough to touch within 99
    frames.
    """
    data = fly_data()
    track = data['track'].array[:, 1:SINGLE_AGENT_CHECK_END_FRAME]          # (n_agents, n_frames, 2, n_keypoints)
    sensory_op = apf.dataset.get_operation(data['dataset'].inputs['sensory'].operations, 'sensory')
    agent = 0
    alone = sensory_op.apply(track[agent])                                     # (n_frames, n_sensory_features)
    testing_utils.assert_arrays_match(alone, sensory_op.apply(track[agent:agent + 1])[0], 0.,
                                      'one fly alone vs a batch of only that fly')
    with_others = sensory_op.apply(track)[agent]
    groups = sensory_op.idxinfo                                                # {group: [first column, end column]}
    walls = slice(*groups['wall_touch'])
    testing_utils.assert_arrays_match(alone[:, walls], with_others[:, walls], 0., 'wall features')
    for group in ['otherflies_vision', 'otherflies_touch']:
        values = alone[:, slice(*groups[group])]
        assert np.all(values == 0), f'{group}: a lone fly should see no other flies, got max {np.nanmax(values):.3g}'
    # so the check above is not vacuous: with the others present, the fly sees some of them (touch is
    # left out, since flies are rarely close enough to touch within a short stretch of frames)
    assert np.any(with_others[:, slice(*groups['otherflies_vision'])] > 0), 'no other flies seen even with them present'


if __name__ == '__main__':
    testing_utils.run_as_script(globals())
