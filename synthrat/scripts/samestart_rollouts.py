"""Draw same-start futures from the trained synthrat model and from RatInABox.

For each selected episode of a RatInABox data file (default: the val file in
the config), starting from the same burn-in (the `context_length` frames that
begin `context_length` frames into the episode, as in agent_synthrat and
probe_value_neuron), this draws
    - K model futures: `experiments.synthrat.simulate(..., n_samples=K)`, batched in
      one forward pass per frame, on the GPU when one is usable;
    - 2K RatInABox futures: `synthrat.generate_data.run_policy_from_burn_in`, on CPU.
The first K RatInABox futures are the comparison set for the model; the
second K give the RatInABox-vs-RatInABox noise floor
(see notebooks/evaluate_synthrat.py).

Device handling never fails on GPU problems: `--device auto` uses CUDA only
if it initializes and has enough free memory, otherwise CPU. If CUDA runs out
of memory mid-run, the samples are split into smaller forward passes; at one
row per pass it moves the model to CPU and continues.

Output: one pkl per episode, `<outdir>/epi%05d.pkl`, written atomically, with
keys
    'model'      (K, burn_in + horizon, 3) float: (x m, y m, theta rad)
    'ratinabox'  (2K, burn_in + horizon, 3) float, same layout
    'gt'         (burn_in + horizon, 3) float: the recorded episode frames
    plus metadata (episode index, t0, seeds, device, timings).
Episodes whose file already exists are skipped, so an interrupted job can be
rerun with the same arguments.

With `--claim`, any number of workers (local processes and cluster jobs) can
share the same episode range: a worker claims an episode by atomically
creating `epi%05d.pkl.lock` and removes the lock after writing the pkl. A
worker that is killed leaves its lock behind; rerun once without `--claim`
afterwards to fill any such gaps, then delete leftover .lock files.

Usage (repo root on PYTHONPATH, env transformer312):
    python -m synthrat.scripts.samestart_rollouts --episode-start 0 --episode-stop 200
On the Janelia cluster, submit synthrat/scripts/run_samestart_rollouts.sh with the
same arguments.
"""

import argparse
import logging
import os
import pickle
import socket
import time

import numpy as np
import torch

import synthrat
from synthrat.config import DEFAULTCONFIGFILE, read_config
from synthrat.generate_data import load_state, rehydrate_policy, run_policy_from_burn_in
import apf.dataset
import apf.io
from apf.io import load_model
from apf.models import initialize_model
from apf.utils import get_cuda_device
import experiments.synthrat as synthrat_exp

REPO_ROOT = os.path.dirname(synthrat.__path__[0])
DEFAULT_CONFIG_FILE = DEFAULTCONFIGFILE
# A checkpoint trained before the fly orientation convention was adopted has to be converted
# first (synthrat/convert_orientation_convention.py), which is what the suffix marks; models
# are not in the repository, so this default only fits the machine it was trained on.
DEFAULT_MODEL_FILE = os.path.join(REPO_ROOT, 'notebooks', 'synthrat_models',
                                  'synthratdefault_20260430T003318_bestepoch100_flyconvention.pth')
# GPU memory beyond the weights: CUDA context plus activations (K rows x context_length tokens).
GPU_MARGIN_BYTES = int(1e9)
# Seed offset between episodes for the RatInABox samples; must exceed 2K.
RATINABOX_SEED_STRIDE = 10000


def parse_args():
    """Command-line options; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config-file', default=DEFAULT_CONFIG_FILE)
    parser.add_argument('--model-file', default=DEFAULT_MODEL_FILE)
    parser.add_argument('--data-file', default=None,
                        help="raw RatInABox pkl to take burn-ins from (default: config['invalfile'])")
    parser.add_argument('--outdir', default=None,
                        help='default: <datadir>/<model name>_<data name>_samestart_K<K>_H<horizon>')
    parser.add_argument('--episode-start', type=int, default=0)
    parser.add_argument('--episode-stop', type=int, default=None, help='exclusive; default: all episodes')
    parser.add_argument('--n-samples', type=int, default=20, help='K, model futures per episode')
    parser.add_argument('--horizon', type=int, default=80, help='future frames per sample')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--rows-per-call', type=int, default=None,
                        help='samples per simulate() call (default: all K); halved automatically on CUDA OOM')
    parser.add_argument('--claim', action='store_true',
                        help='claim episodes with lock files so several workers can share one range')
    parser.add_argument('--model-seed', type=int, default=0)
    parser.add_argument('--ratinabox-seed', type=int, default=1000,
                        help='episode e, sample k uses seed ratinabox_seed + 10000*e + k (as in agent_synthrat)')
    return parser.parse_args()


def load_model_and_data(config_file, model_file, data_file):
    """Load the trained model on CPU, the dataset, and the RatInABox objects.

    Parameters
    ----------
    config_file : str, synthrat config json.
    model_file : str, trained checkpoint (.pth); its stored config and
        training-set normalization (dataset_params) override the json.
    data_file : str or None, raw RatInABox pkl; None uses config['invalfile'].

    Returns
    -------
    dict with
        'config': dict
        'model': torch Module on CPU, eval mode
        'dataset': apf.dataset.Dataset for data_file, normalized with the
            training-set parameters stored in the checkpoint
        'pose': apf.dataset.Data, array (1, n_frames_total, 3) = (x m, y m, theta rad)
        'isstart': (n_frames_total,) bool, True at the first frame of each episode
        'ratinabox_info': dict of live RatInABox objects, including the
            policy objects needed by run_policy_from_burn_in
        'data_file': str, the data file used

    Raises
    ------
    AssertionError if the checkpoint stores no dataset_params, in which case the data would be
    normalized with its own statistics instead of the training set's.
    ValueError (from experiments.synthrat.check_orientation_convention) if the checkpoint was
    trained under a different orientation convention; synthrat/convert_orientation_convention.py
    converts such a checkpoint.
    """
    device = torch.device('cpu')
    config = read_config(config_file)
    # The checkpoint's own config and its training-set normalization take precedence over the
    # json, so the data is preprocessed exactly as it was for training.
    apf.io.load_config_from_model_file(loadmodelfile=model_file, config=config)
    assert config.get('dataset_params') is not None, \
        'checkpoint has no dataset_params; the data would be normalized with its own statistics'
    config['device'] = str(device)
    np.random.seed(config['numpy_seed'])
    torch.manual_seed(config['torch_seed'])
    if data_file is None:
        data_file = config['invalfile']

    dataset, data, ratinabox_info = synthrat_exp.make_dataset_cache_wrapper(data_file, config)
    # The policy objects are rebuilt from the trajectory file itself rather than from the
    # dataset cache or a separate state file, so the value function replayed here is the one
    # that generated these episodes.
    ratinabox_info = rehydrate_policy(load_state(data_file))

    model, _ = initialize_model(config, dataset, device)
    load_model(model_file, model, device, check_state=synthrat_exp.check_orientation_convention)
    model.eval()

    # make_dataset returns isstart as (n_frames_total, n_agents) while pose is agent-first;
    # flatten it to one frame mask so the episode bookkeeping cannot depend on that layout.
    pose = data['pose']
    n_agents, n_frames_total = pose.array.shape[:2]
    assert n_agents == 1, f'one agent expected in the synthrat data, got {n_agents}'
    isstart = np.asarray(data['isstart']).reshape(-1)                       # (n_frames_total,)
    assert isstart.shape[0] == n_frames_total, \
        f'isstart has {isstart.shape[0]} frames, pose has {n_frames_total}'
    return {'config': config, 'model': model, 'dataset': dataset, 'pose': pose,
            'isstart': isstart, 'ratinabox_info': ratinabox_info, 'data_file': data_file}


def simulate_model_samples(model, dataset, pose, start_frame, burn_in, horizon, n_samples, rows_per_call):
    """Draw `n_samples` model futures from one burn-in, `rows_per_call` samples per simulate() call.

    Parameters
    ----------
    model : torch Module (on any device).
    dataset : apf.dataset.Dataset matching `pose`.
    pose : apf.dataset.Data, array (1, n_frames_total, 3).
    start_frame : int, first burn-in frame in `pose`.
    burn_in : int, number of burn-in frames.
    horizon : int, number of future frames.
    n_samples : int, number of futures.
    rows_per_call : int, maximum samples per simulate() call (GPU memory).

    Returns
    -------
    (n_samples, burn_in + horizon, 3) float array of (x m, y m, theta rad).
    Uses the global numpy and torch RNGs (seed them before calling).
    """
    chunks = []
    for first in range(0, n_samples, rows_per_call):
        n_rows = min(rows_per_call, n_samples - first)
        _, pred_pose = synthrat_exp.simulate(
            dataset, model, pose, track_len=burn_in + horizon, burn_in=burn_in,
            agent_ids=[0], start_frame=start_frame, progress_bar=False, n_samples=n_rows)
        chunks.append(pred_pose)                                            # (n_rows, burn_in + horizon, 3)
    return np.concatenate(chunks, axis=0)


def simulate_with_fallback(state, dataset, pose, start_frame, burn_in, horizon, n_samples, seed):
    """`simulate_model_samples` that recovers from CUDA failures instead of raising.

    On CUDA out-of-memory, halves `state['rows_per_call']` and retries; at one
    row per call, or on any other CUDA runtime error, moves the model to CPU
    and retries. RNGs are reseeded with `seed` before every attempt.

    Parameters
    ----------
    state : dict with 'model', 'device' (torch.device), 'rows_per_call' (int).
        Updated in place when a fallback happens, so later episodes keep it.
    dataset, pose, start_frame, burn_in, horizon, n_samples : as in
        `simulate_model_samples`.
    seed : int, seed for numpy and torch global RNGs.

    Returns
    -------
    (n_samples, burn_in + horizon, 3) float array.
    """
    while True:
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            return simulate_model_samples(state['model'], dataset, pose, start_frame, burn_in, horizon,
                                          n_samples, state['rows_per_call'])
        except torch.cuda.OutOfMemoryError:
            if state['device'].type != 'cuda':
                raise
            torch.cuda.empty_cache()
            if state['rows_per_call'] > 1:
                state['rows_per_call'] = max(1, state['rows_per_call'] // 2)
                print(f'CUDA out of memory; retrying with {state["rows_per_call"]} rows per call', flush=True)
                continue
            reason = 'CUDA out of memory with 1 row per call'
        except RuntimeError as e:
            if state['device'].type != 'cuda' or 'CUDA' not in str(e):
                raise
            reason = f'CUDA error: {e}'
        print(f'{reason}; moving the model to CPU and continuing', flush=True)
        state['model'] = state['model'].to('cpu')
        state['device'] = torch.device('cpu')


def claim_episode(lock_path, out_path):
    """Atomically claim an episode for this worker.

    Parameters
    ----------
    lock_path : str, lock file to create (exclusive create; fails if it exists).
    out_path : str, the episode's output pkl.

    Returns
    -------
    bool, True if this worker now owns the episode (the lock file exists and
    contains this host and pid); False if another worker holds the lock or
    already wrote the output.
    """
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f'{socket.gethostname()} {os.getpid()}\n'.encode())
    os.close(fd)
    # Another worker may have finished this episode (and removed its lock)
    # between the caller's existence check and the claim.
    if os.path.exists(out_path):
        os.remove(lock_path)
        return False
    return True


def main():
    """Load everything, then simulate and save each selected episode."""
    # INFO so a job log records which device was chosen and with how much room to spare.
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    args = parse_args()
    loaded = load_model_and_data(args.config_file, args.model_file, args.data_file)
    config, dataset, pose = loaded['config'], loaded['dataset'], loaded['pose']
    ratinabox_info = loaded['ratinabox_info']
    burn_in = dataset.context_length
    dt = ratinabox_info['Ag'].dt                                             # seconds per frame
    track_len = burn_in + args.horizon
    assert 2 * args.n_samples <= RATINABOX_SEED_STRIDE

    model = loaded['model']
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    device = get_cuda_device(2 * param_bytes + GPU_MARGIN_BYTES, args.device)
    print(f'model parameters: {param_bytes / 1e9:.2f} GB; device: {device}', flush=True)
    state = {'model': model.to(device), 'device': device,
             'rows_per_call': args.rows_per_call or args.n_samples}

    if args.outdir is None:
        model_name = os.path.splitext(os.path.basename(args.model_file))[0]
        data_name = os.path.splitext(os.path.basename(loaded['data_file']))[0]
        args.outdir = os.path.join(config['datadir'],
                                   f'{model_name}_{data_name}_samestart_K{args.n_samples}_H{args.horizon}')
    os.makedirs(args.outdir, exist_ok=True)
    print(f'writing to {args.outdir}', flush=True)

    idxstart = np.nonzero(loaded['isstart'])[0]
    idxstart = np.append(idxstart, loaded['isstart'].shape[0])               # end of the last episode
    n_episodes = len(idxstart) - 1
    episode_stop = n_episodes if args.episode_stop is None else min(args.episode_stop, n_episodes)

    for epi in range(args.episode_start, episode_stop):
        out_path = os.path.join(args.outdir, f'epi{epi:05d}.pkl')
        if os.path.exists(out_path):
            continue
        t0 = int(idxstart[epi]) + burn_in                                   # first burn-in frame
        if t0 + track_len > idxstart[epi + 1]:
            print(f'epi {epi}: too short for burn-in + horizon, skipping', flush=True)
            continue
        lock_path = out_path + '.lock'
        if args.claim and not claim_episode(lock_path, out_path):
            continue

        time_start = time.time()
        model_seed = int(np.random.SeedSequence([args.model_seed, epi]).generate_state(1)[0])
        model_poses = simulate_with_fallback(state, dataset, pose, t0, burn_in, args.horizon,
                                             args.n_samples, model_seed)    # (K, track_len, 3)
        time_model = time.time() - time_start

        time_start = time.time()
        burn_in_pose = pose.array[0, t0:t0 + burn_in]                       # (burn_in, 3)
        ratinabox_seed = args.ratinabox_seed + RATINABOX_SEED_STRIDE * epi
        samples = run_policy_from_burn_in(
            ratinabox_info, burn_in_pose=burn_in_pose, n_frames=args.horizon,
            n_samples=2 * args.n_samples, base_seed=ratinabox_seed)
        assert all(len(s) == track_len for s in samples), [len(s) for s in samples]
        ratinabox_poses = np.stack(samples)                                  # (2K, track_len, 3)
        time_ratinabox = time.time() - time_start

        result = {
            'epi': epi, 't0': t0, 'burn_in': burn_in, 'horizon': args.horizon, 'dt': dt,
            'model': model_poses, 'ratinabox': ratinabox_poses,
            'gt': pose.array[0, t0:t0 + track_len].copy(),
            'model_seed': model_seed, 'ratinabox_seed': ratinabox_seed,
            'model_file': args.model_file, 'data_file': loaded['data_file'],
            'device_used': str(state['device']), 'rows_per_call': state['rows_per_call'],
            'time_model_s': time_model, 'time_ratinabox_s': time_ratinabox,
        }
        tmp_path = out_path + '.tmp'
        with open(tmp_path, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, out_path)
        if args.claim:
            os.remove(lock_path)

        message = (f'epi {epi}: model {time_model:.1f} s on {state["device"]} '
                   f'({state["rows_per_call"]} rows/call), ratinabox {time_ratinabox:.1f} s')
        if state['device'].type == 'cuda':
            message += f', peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB'
        print(message, flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
