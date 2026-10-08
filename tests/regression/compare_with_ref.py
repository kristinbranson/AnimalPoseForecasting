"""Compare the current code against an older version of it, by running the same workloads under both.

Not a unit test; run it by hand when a change should leave results unchanged. It extracts the code of
a git ref (e.g. main) into a temporary directory with git archive (the repository is not changed),
runs tests/regression/workloads.py under that version and under the current working tree, each in
its own process, and compares the outputs:
  - fly (tests/data/small_usertrain_v3.npz, tests/config_fly_test.json): every dataset input,
    keypoints, chunk indices, saved operation parameters, the losses and first-batch gradients of a
    seeded random model, a seeded simulation's keypoints, and the coordinates drawn by
    flyllm.plotting.debug_plot_pose must be identical. Labels must be identical at every frame whose
    movement is defined; at undefined frames (no movement out of the frame) they may differ, since
    older code (main before the synthrat branch) gave them a last-bin label instead of NaN.
  - synthrat (first validation episodes): inputs, labels and chunks must be identical, the firing
    rates to within RatInABox's rounding. With --synthrat-old-convention, the reference predates the
    fly orientation convention, so its velocity features are rearranged first (new = (old 1, -old 0,
    old 2), the lateral label bins reversed) and compared within the tolerance the old orientation's
    1e-6 rad offset allows.
Synthrat is skipped if the reference has no synthrat code.

Usage, from the repository root (about 30 s with a GPU; it falls back to the CPU; ~1.2 GB of
temporary outputs, deleted at the end unless --out-dir is given):
    python tests/regression/compare_with_ref.py main
    python tests/regression/compare_with_ref.py <ref> --synthrat-old-convention --skip-fly
Exits with status 1 if anything differs. workloads.py comes from the working tree and imports the
reference's modules, so the functions it calls must exist in both versions.
"""
import argparse
import io
import json
import os
import pathlib
import pickle
import subprocess
import sys
import tarfile
import tempfile

import numpy as np

REPO_DIR = pathlib.Path(__file__).resolve().parents[2]
WORKLOADS = pathlib.Path(__file__).resolve().parent / 'workloads.py'
# the saved operation parameters contain apf objects, so unpickling them needs apf importable
sys.path.insert(0, str(REPO_DIR))

# Code directories extracted from the reference.
CODE_DIRECTORIES = ['apf', 'experiments', 'flyllm', 'synthrat']
# Environment for the workload processes: deterministic cuBLAS, no display.
WORKLOAD_ENVIRONMENT = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'MPLBACKEND': 'Agg'}

# Synthrat firing rates are not bit-for-bit reproducible between RatInABox evaluations (~1e-7, ~1e-6
# after z-scoring).
SYNTHRAT_SENSORY_TOLERANCE = 1e-5
# Against code from before the fly orientation convention: its orientation was shifted by up to ~1e-6
# rad, which makes z-scored velocities differ by up to ~1e-5, and soft labels by that divided by the
# narrowest bin (~0.04 z-units).
SYNTHRAT_OLD_CONVENTION_INPUT_TOLERANCE = 1e-5
SYNTHRAT_OLD_CONVENTION_LABEL_TOLERANCE = 5e-4


def extract_reference(ref: str, destination: pathlib.Path) -> list:
    """Extracts the reference's code directories into destination with git archive.

    Args:
        ref: git ref, e.g. 'main' or a commit hash.
        destination: empty directory to extract into.

    Returns:
        the code directories the reference has.
    """
    listing = subprocess.run(['git', '-C', str(REPO_DIR), 'ls-tree', '--name-only', ref],
                             capture_output=True, text=True, check=True).stdout.split()
    present = [d for d in CODE_DIRECTORIES if d in listing]
    archive = subprocess.run(['git', '-C', str(REPO_DIR), 'archive', ref] + present,
                             capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter='data')
    if 'synthrat' in present:
        # the generated synthrat data is not in git; share this checkout's
        os.symlink(os.path.realpath(REPO_DIR / 'synthrat' / 'data'), destination / 'synthrat' / 'data')
    return present


def write_fly_config(path: pathlib.Path) -> None:
    """Writes tests/config_fly_test.json with its data directory made absolute, so any checkout can read it.

    Args:
        path: where to write the config.
    """
    config = json.loads((REPO_DIR / 'tests' / 'config_fly_test.json').read_text())
    config['datadir'] = str(REPO_DIR / config['datadir'])
    path.write_text(json.dumps(config, indent=4))


def run_workload(checkout: pathlib.Path, arguments: list) -> None:
    """Runs workloads.py with the code of checkout, in its own process.

    Args:
        checkout: root directory of the code to run.
        arguments: arguments for workloads.py.
    """
    environment = dict(os.environ, **WORKLOAD_ENVIRONMENT)
    subprocess.run([sys.executable, str(WORKLOADS)] + arguments, cwd=checkout, env=environment, check=True)


def compare_arrays(name: str, reference, current, tolerance: float = 0.) -> bool:
    """Prints how two arrays compare and returns whether they match.

    Args:
        name: what is compared, for the printout.
        reference, current: the reference version's array and the current version's.
        tolerance: largest allowed absolute difference at entries defined in both; NaN must be in
            the same places.
    """
    reference, current = np.asarray(reference), np.asarray(current)
    if reference.shape != current.shape:
        print(f'  {name:40s} DIFFERENT shapes {reference.shape} vs {current.shape}')
        return False
    if reference.dtype.kind in 'iub':
        same = np.array_equal(reference, current)
        print(f'  {name:40s} {"identical" if same else "DIFFERENT"}')
        return same
    nan_same = np.array_equal(np.isnan(reference), np.isnan(current))
    defined = ~np.isnan(reference) & ~np.isnan(current)
    difference = np.abs(reference[defined] - current[defined]).max() if defined.any() else 0.
    ok = nan_same and difference <= tolerance
    status = 'identical' if (nan_same and difference == 0) else ('within tolerance' if ok else 'DIFFERENT')
    print(f'  {name:40s} {status}: max |difference| {difference:.3g}, NaN in the same places: {nan_same}')
    return ok


def differing_parameters(reference, current, path: str = 'params') -> list:
    """Recursively compares saved operation parameters; returns the paths of entries that differ."""
    if isinstance(reference, dict) and isinstance(current, dict):
        if reference.keys() != current.keys():
            return [f'{path}: keys {sorted(set(reference) ^ set(current))}']
        return [d for k in reference for d in differing_parameters(reference[k], current[k], f'{path}.{k}')]
    if isinstance(reference, (list, tuple)) and isinstance(current, (list, tuple)):
        if len(reference) != len(current):
            return [f'{path}: lengths {len(reference)} vs {len(current)}']
        return [d for i, (a, b) in enumerate(zip(reference, current))
                for d in differing_parameters(a, b, f'{path}[{i}]')]
    if hasattr(reference, '__dict__') and type(reference) is type(current) and not isinstance(reference, np.ndarray):
        return differing_parameters(vars(reference), vars(current), path)
    if isinstance(reference, np.ndarray) or isinstance(current, np.ndarray):
        return [] if np.array_equal(np.asarray(reference), np.asarray(current), equal_nan=True) else [path]
    return [] if reference == current else [path]


def compare_fly(reference_dir: pathlib.Path, current_dir: pathlib.Path) -> bool:
    """Compares the fly outputs of the reference and the current code; returns whether they match."""
    reference, current = np.load(reference_dir / 'fly.npz'), np.load(current_dir / 'fly.npz')
    if sorted(reference.files) != sorted(current.files):
        print('  different outputs saved:', sorted(set(reference.files) ^ set(current.files)))
        return False
    ok = all(compare_arrays(k, reference[k], current[k]) for k in sorted(reference.files)
             if not k.startswith('plot/') and not k.startswith('labels/'))
    plots = sorted(k for k in reference.files if k.startswith('plot/'))
    n_same = sum(np.array_equal(reference[k], current[k], equal_nan=True) for k in plots)
    print(f'  {"debug_plot_pose coordinates":40s} {n_same} of {len(plots)} plotted artists identical')
    ok &= n_same == len(plots)
    with open(reference_dir / 'fly_params.pkl', 'rb') as f:
        reference_params = pickle.load(f)
    with open(current_dir / 'fly_params.pkl', 'rb') as f:
        current_params = pickle.load(f)
    differing = differing_parameters(reference_params, current_params)
    print(f'  {"saved operation parameters":40s} {"identical" if not differing else "DIFFERENT: " + str(differing[:5])}')
    ok &= not differing

    # labels: identical where the movement is defined; at undefined frames older code gave a
    # last-bin label instead of NaN, so differences are allowed there only
    for key in [k for k in reference.files if k.startswith('labels/')]:
        labels_reference, labels_current = reference[key], current[key]        # (n_agents, n_frames, n_label_features)
        undefined = np.isnan(labels_reference).any(-1) & np.isnan(labels_current).any(-1)   # (n_agents, n_frames)
        same_entries = (labels_reference == labels_current) | (np.isnan(labels_reference) & np.isnan(labels_current))
        differing_frames = ~same_entries.all(-1)
        defined_same = np.array_equal(labels_reference[~undefined], labels_current[~undefined])
        only_undefined = bool(np.all(undefined[differing_frames]))
        print(f'  {key + " at defined frames":40s} {"identical" if defined_same else "DIFFERENT"}')
        print(f'  {key + " at undefined frames":40s} {differing_frames.sum()} frames differ; '
              f'all at frames whose movement is undefined: {only_undefined}')
        ok &= defined_same and only_undefined
    return ok


def compare_synthrat(reference_dir: pathlib.Path, current_dir: pathlib.Path, old_convention: bool) -> bool:
    """Compares the synthrat outputs; returns whether they match.

    Args:
        reference_dir, current_dir: directories holding synthrat.npz.
        old_convention: the reference predates the fly orientation convention, so its velocity
            features are rearranged before comparing.
    """
    reference, current = np.load(reference_dir / 'synthrat.npz'), np.load(current_dir / 'synthrat.npz')
    velocity, labels = reference['inputs/velocity'], reference['labels/velocity']
    input_tolerance, label_tolerance = 0., 0.
    if old_convention:
        # the model converter's rearrangement: velocity features reordered and the lateral one
        # negated; in the soft labels, the blocks of bins reordered and a negated feature's bins
        # reversed (negating a value mirrors the bins)
        from synthrat.convert_orientation_convention import OLD_TO_NEW_ORDER, OLD_TO_NEW_SIGN, rearrange_values
        velocity = rearrange_values(velocity, negate=True)
        binned = labels.reshape(labels.shape[:-1] + (len(OLD_TO_NEW_ORDER), -1))     # (..., feature, bin)
        labels = np.stack([binned[..., old, ::-1] if sign < 0 else binned[..., old, :]
                           for old, sign in zip(OLD_TO_NEW_ORDER, OLD_TO_NEW_SIGN)], axis=-2).reshape(labels.shape)
        input_tolerance, label_tolerance = SYNTHRAT_OLD_CONVENTION_INPUT_TOLERANCE, SYNTHRAT_OLD_CONVENTION_LABEL_TOLERANCE
    suffix = ' (rearranged)' if old_convention else ''
    results = [
        compare_arrays('chunk_indices', reference['chunk_indices'], current['chunk_indices']),
        compare_arrays('inputs/velocity' + suffix, velocity, current['inputs/velocity'], input_tolerance),
        compare_arrays('inputs/sensory', reference['inputs/sensory'], current['inputs/sensory'],
                       max(SYNTHRAT_SENSORY_TOLERANCE, input_tolerance)),
        compare_arrays('labels/velocity' + suffix, labels, current['labels/velocity'], label_tolerance),
    ]
    return all(results)


def run_comparison(args: argparse.Namespace, out_dir: pathlib.Path) -> bool:
    """Extracts the reference, runs the workloads under both versions, and compares them.

    Args:
        args: parsed command-line arguments (see main).
        out_dir: empty or new directory for the extracted code and the outputs.

    Returns:
        whether every comparison passed.
    """
    reference_checkout = out_dir / 'reference_code'
    reference_checkout.mkdir(parents=True)
    present = extract_reference(args.ref, reference_checkout)
    print(f'comparing the working tree against {args.ref}; outputs in {out_dir}')
    outputs = {name: out_dir / f'outputs_{name}' for name in ['reference', 'current']}
    for directory in outputs.values():
        directory.mkdir()
    checkouts = {'reference': reference_checkout, 'current': REPO_DIR}

    ok = True
    if not args.skip_fly:
        config = out_dir / 'config_fly.json'
        write_fly_config(config)
        for name in ['reference', 'current']:
            run_workload(checkouts[name], ['fly', str(config), str(outputs[name])])
        print(f'=== fly: {args.ref} vs current')
        ok &= compare_fly(outputs['reference'], outputs['current'])
    if not args.skip_synthrat and 'synthrat' in present:
        for name in ['reference', 'current']:
            run_workload(checkouts[name], ['synthrat', str(outputs[name])])
        print(f'=== synthrat: {args.ref} vs current')
        ok &= compare_synthrat(outputs['reference'], outputs['current'], args.synthrat_old_convention)
    elif not args.skip_synthrat:
        print(f'=== synthrat: skipped, {args.ref} has no synthrat code')
    return ok


def main() -> None:
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('ref', help='git ref of the version to compare against, e.g. main')
    parser.add_argument('--out-dir', help='keep the extracted code and outputs here (default: a temporary '
                                          'directory, deleted at the end)')
    parser.add_argument('--skip-fly', action='store_true', help='skip the fly workloads')
    parser.add_argument('--skip-synthrat', action='store_true', help='skip the synthrat workload')
    parser.add_argument('--synthrat-old-convention', action='store_true',
                        help='the reference predates the fly orientation convention for synthrat')
    args = parser.parse_args()

    if args.out_dir:
        ok = run_comparison(args, pathlib.Path(args.out_dir))
    else:
        # the fly outputs take ~0.6 GB per version, so they are not left behind
        with tempfile.TemporaryDirectory(prefix='apf_regression_') as out_dir:
            ok = run_comparison(args, pathlib.Path(out_dir))
    print('RESULT:', 'no differences' if ok else 'DIFFERENCES FOUND')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
