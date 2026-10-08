"""Shared helpers for the unit tests in this directory.

Importing this module puts the repository root first on sys.path, so the tests always exercise
this checkout's apf/, experiments/, flyllm/ and synthrat/ rather than another copy that happens
to be importable. Test modules must import it before any of those packages.

The tests run either under pytest, from the repository root:
    python -m pytest tests/test_fly.py tests/test_synthrat.py
or as plain scripts, which needs no pytest:
    python tests/test_fly.py
"""
import pathlib
import sys
import traceback
import unittest

import numpy as np
import torch

REPO_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import apf.dataset  # noqa: E402  (needs the sys.path entry above)
import apf.utils  # noqa: E402


def require_file(path: str | pathlib.Path, description: str) -> None:
    """Skips the calling test when a data file it needs is not reachable.

    pytest reports unittest.SkipTest as a skip; run_as_script does the same.

    Args:
        path: file the test reads.
        description: what the file is, for the skip message.
    """
    if not pathlib.Path(path).exists():
        raise unittest.SkipTest(f"{description} not found: {path}")


class TrueLabelModel(torch.nn.Module):
    """Stand-in for a trained model: predicts the true labels and records what it was given.

    simulate() only calls model.output(input, mask=..., is_causal=...), model.eval() and
    model.parameters(), so this provides exactly those. Given an input covering frames
    [0, n), output returns the labels for frames [0, n) of the simulated window, so the last
    prediction is the true movement out of frame n - 1. simulate() must therefore be run with
    max_contextl=None, so that every input starts at frame 0.

    Attributes:
        labels: true labels for the simulated window, keyed as in a Dataset chunk ('labels'
            and/or 'labels_discrete'), each (n_agents, n_frames, n_label_features) float array.
        last_input: the input of the most recent output call, (n_agents, n_input_frames,
            d_input) float32 array, or None before the first call.
    """

    def __init__(self, labels: dict[str, np.ndarray]):
        """
        Args:
            labels: see the class attribute of the same name.
        """
        super().__init__()
        # simulate() reads the device from the first parameter.
        self.device_anchor = torch.nn.Parameter(torch.zeros(1))
        self.labels = labels
        self.last_input = None

    def output(self, input: torch.Tensor, mask=None, is_causal=None) -> dict[str, torch.Tensor]:
        """Returns the true labels for the frames covered by input, and records input.

        Args:
            input: (n_agents, n_input_frames, d_input) float tensor, frames [0, n_input_frames).
            mask, is_causal: accepted for compatibility with TransformerModel.output, ignored.

        Returns:
            dict with the same keys as self.labels, each (n_agents, n_input_frames,
            n_label_features) float tensor.
        """
        self.last_input = input.detach().cpu().numpy()
        n_input_frames = input.shape[1]
        return {key: torch.from_numpy(value[:, :n_input_frames]) for key, value in self.labels.items()}


def true_labels(dataset, start_frame: int, n_frames: int, agent_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Collects a Dataset's labels over one window for several agents, as TrueLabelModel wants them.

    Args:
        dataset: apf.dataset.Dataset.
        start_frame: first frame of the window.
        n_frames: number of frames in the window.
        agent_ids: (n_agents,) int array of agents to collect.

    Returns:
        dict with the label keys of a chunk ('labels' and/or 'labels_discrete'), each
        (n_agents, n_frames, n_label_features) float32 array.
    """
    chunks = [dataset.get_chunk(start_frame=start_frame, duration=n_frames, agent_id=agent_id)
              for agent_id in agent_ids]
    keys = [key for key in ('labels', 'labels_discrete') if key in chunks[0]]
    return {key: np.stack([chunk[key] for chunk in chunks]) for key in keys}


def interior_decoding_errors(zscored: np.ndarray, decoded: np.ndarray, bin_edges: np.ndarray):
    """Decoding error of discretized labels, for values far enough from the end bins.

    A soft label puts weight on the bin containing the value and one neighbor, and decoding
    takes the weighted average of the bin medians. For a value at least two bins from either
    end, both bins are interior, so the error is at most the widest interior bin. The end bins
    are left out because they stretch to cover outliers.

    Args:
        zscored: true z-scored values of one feature, any shape.
        decoded: the same values decoded from their discretized labels, same shape.
        bin_edges: (n_bins + 1,) bin edges of that feature, z-scored units.

    Returns:
        errors: (n_checked,) absolute decoding errors for the values that are checked.
        fraction_checked: fraction of the non-NaN values that are checked.
        max_interior_width: width of the widest interior bin, the bound on errors.
    """
    checked = (zscored > bin_edges[2]) & (zscored < bin_edges[-3])
    errors = np.abs(decoded - zscored)[checked]
    fraction_checked = checked.sum() / np.count_nonzero(~np.isnan(zscored))
    max_interior_width = np.diff(bin_edges)[1:-1].max()
    return errors, fraction_checked, max_interior_width


def assert_arrays_match(actual: np.ndarray, expected: np.ndarray, tolerance: float, description: str) -> None:
    """Asserts two arrays are NaN in the same places and within tolerance everywhere else.

    Args:
        actual, expected: float arrays of the same shape.
        tolerance: largest allowed absolute difference.
        description: what is being compared, for the failure message.
    """
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f'{description}: shape {actual.shape} != {expected.shape}'
    nan_mismatch = np.isnan(actual) != np.isnan(expected)
    assert not nan_mismatch.any(), f'{description}: NaN in different places ({nan_mismatch.sum()} entries)'
    defined = ~np.isnan(expected)
    if defined.any():
        error = np.abs(actual[defined] - expected[defined]).max()
        assert error <= tolerance, f'{description}: max difference {error:.3g} > {tolerance:.3g}'


def assert_chunks_have_defined_targets(dataset, velocity: np.ndarray) -> None:
    """Asserts that no training chunk contains a frame whose target movement is undefined.

    Args:
        dataset: apf.dataset.Dataset.
        velocity: (n_agents, n_frames, n_velocity_features) float array of movement from each
            frame to the next, the quantity the labels encode; NaN where undefined.
    """
    start_frames, agents = dataset.chunk_indices.T                       # (n_chunks,) each
    frames = start_frames[:, None] + np.arange(dataset.context_length)   # (n_chunks, context_length)
    targets = velocity[agents[:, None], frames]                          # (n_chunks, context_length, n_features)
    n_undefined = np.isnan(targets).any(-1).sum()
    assert n_undefined == 0, f'{n_undefined} chunk frames have an undefined target'


def assert_feature_names_match_dimensions(dataset) -> None:
    """Asserts Dataset.get_input_names / get_label_names give one name per feature dimension,
    each prefixed by its data key.

    Args:
        dataset: apf.dataset.Dataset.
    """
    for names, datas in [(dataset.get_input_names(), dataset.inputs), (dataset.get_label_names(), dataset.labels)]:
        expected_prefixes = [f'{key}__' for key, data in datas.items() for _ in range(data.array.shape[-1])]
        assert len(names) == len(expected_prefixes), f'{len(names)} names for {len(expected_prefixes)} dimensions'
        mismatched = [name for name, prefix in zip(names, expected_prefixes) if not name.startswith(prefix)]
        assert not mismatched, f'names not prefixed by their key: {mismatched[:5]}'


def assert_batches_split_back_to_dataset(dataset, batch_size: int) -> None:
    """Asserts that a DataLoader batch, split back into named inputs and labels, matches the
    dataset's arrays at each chunk's frames, including when the discrete labels are laid out as a
    model predicts them, and that a sub-range of a chunk equals the matching slice.

    Args:
        dataset: apf.dataset.Dataset.
        batch_size: number of chunks in the batch checked; the first batch_size chunks.
    """
    # chunks are stored as float32
    float32_tolerance = 1e-6

    loader = apf.dataset.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = apf.utils.convert_torch_to_numpy(next(iter(loader)))
    split = dataset.item_to_data(batch)
    for i, (start_frame, agent) in enumerate(dataset.chunk_indices[:batch_size]):
        frames = slice(start_frame, start_frame + dataset.context_length)
        for group, datas in [('inputs', dataset.inputs), ('labels', dataset.labels)]:
            for key, data in datas.items():
                assert_arrays_match(split[group][key].array[i], data.array[agent, frames], float32_tolerance,
                                    f"batch item {i}, {group} '{key}'")

    # A model predicts discrete labels unflattened, (..., d_output_discrete, nbins).
    if 'labels_discrete' in batch:
        prediction = {'discrete': batch['labels_discrete'].reshape(
            batch['labels_discrete'].shape[:-1] + (dataset.d_output_discrete, dataset.discretize_nbins))}
        if 'labels' in batch:
            prediction['continuous'] = batch['labels']
        split_prediction = dataset.split_output_by_names(prediction)
        for key in dataset.labels:
            assert np.array_equal(split_prediction[key], split['labels'][key].array, equal_nan=True), \
                f"label '{key}' splits differently from a prediction-shaped batch"

    start_frame, agent = dataset.chunk_indices[0]
    offset, duration = dataset.context_length // 4, dataset.context_length // 2
    chunk = dataset.get_chunk(start_frame, dataset.context_length, agent)
    sub_chunk = dataset.get_chunk(start_frame + offset, duration, agent)
    for key in ['input', 'labels', 'labels_discrete']:
        if key in chunk:
            assert np.array_equal(sub_chunk[key], chunk[key][offset:offset + duration], equal_nan=True), \
                f"sub-chunk '{key}' differs from the slice of the full chunk"


# Operations whose invert is not defined (they log an error and return None).
NON_INVERTIBLE_OPERATIONS = {'Sensory', 'Subset'}


def no_sampling_invert_kwargs(op) -> dict:
    """Keyword arguments for op.invert that decode discretized values deterministically.

    Discretize.invert samples from the bin probabilities by default; do_sampling=False takes the
    probability-weighted average of the bin centers instead. A Fusion passes per-operation arguments
    to the operations it combines.

    Args:
        op: an apf.dataset.Operation.

    Returns:
        {'do_sampling': False} for a Discretize, {'kwargs_per_op': [...]} for a Fusion (with
        do_sampling=False for each Discretize in it), {} otherwise.
    """
    if isinstance(op, apf.dataset.Discretize):
        return {'do_sampling': False}
    if isinstance(op, apf.dataset.Fusion):
        return {'kwargs_per_op': [{'do_sampling': False} if isinstance(inner, apf.dataset.Discretize) else {}
                                  for inner in op.operations]}
    return {}


def assert_operations_handle_single_agent(chains: list, agent: int = 0, tolerances: dict | None = None) -> None:
    """Asserts that every operation gives the same result for one agent as for that agent in a batch.

    Each chain's operations are applied in order to the batch. At every step, the operation is also
    applied to the chosen agent's array alone (no agent axis), and its result must equal the batch
    result for that agent: same shape, NaN in the same places, values within tolerance. The same is
    checked for invert, except for operations in NON_INVERTIBLE_OPERATIONS; discretized values are
    decoded without sampling so that both calls are deterministic.

    Args:
        chains: list of (description, batch array (n_agents, n_frames, ...), list of operations to
            apply in order).
        agent: index of the agent checked alone.
        tolerances: {operation class name: largest allowed difference}; other operations must match
            exactly.
    """
    tolerances = tolerances or {}
    for description, array, operations in chains:
        for op in operations:
            name = type(op).__name__
            label = f'{description}: {name}'
            batch = op.apply(array)
            tolerance = tolerances.get(name, 0.)
            assert_arrays_match(op.apply(array[agent]), batch[agent], tolerance, f'{label}.apply')
            if name not in NON_INVERTIBLE_OPERATIONS:
                kwargs = no_sampling_invert_kwargs(op)
                assert_arrays_match(op.invert(batch[agent], **kwargs), op.invert(batch, **kwargs)[agent],
                                    tolerance, f'{label}.invert')
            array = batch


def run_as_script(namespace: dict) -> None:
    """Runs every test_* function in a module and exits non-zero if any failed.

    Args:
        namespace: the module's globals().
    Side effects:
        Prints one PASS / FAIL / SKIP line per test, with the traceback for failures, then
        exits the process.
    """
    tests = {name: value for name, value in namespace.items() if name.startswith('test_') and callable(value)}
    outcomes = {'passed': 0, 'skipped': 0, 'failed': 0}
    for name, test in tests.items():
        try:
            test()
            outcomes['passed'] += 1
            print(f'PASS {name}')
        except unittest.SkipTest as skip:
            outcomes['skipped'] += 1
            print(f'SKIP {name}: {skip}')
        except Exception:
            outcomes['failed'] += 1
            print(f'FAIL {name}')
            traceback.print_exc()
    print(', '.join(f'{count} {outcome}' for outcome, count in outcomes.items()))
    sys.exit(1 if outcomes['failed'] else 0)
