"""Run fixed, seeded workloads with the code of the checkout this is run from, and save their outputs.

Not a unit test: tests/regression/compare_with_ref.py runs this once under an older version of the
code and once under the current one, and compares the outputs. Everything is seeded and torch runs in
deterministic mode, so two versions whose code behaves the same produce identical outputs. Seeding
must happen before the dataset is built: fitting the discretization bins draws random numbers
(select_bin_edges breaks ties at random, and the bin samples are drawn at random).

Usage, from the root of the checkout to test (compare_with_ref.py does this):
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python <path>/workloads.py fly <fly_config.json> <out_dir>
    python <path>/workloads.py synthrat <out_dir>

Outputs, in out_dir:
    fly.npz: the dataset's inputs and labels, chunk indices and keypoints; losses of a seeded random
        model on the first batches and its first-batch gradient norms; a simulation's keypoints; the
        coordinates drawn by flyllm.plotting.debug_plot_pose.
    fly_params.pkl: the dataset's operation parameters (Dataset.get_params()).
    synthrat.npz: the synthrat dataset's inputs, labels and chunk indices.
"""
import os
import pickle
import sys

import numpy as np
import torch

# the checkout to test is the working directory
sys.path.insert(0, os.getcwd())

SEED = 0
# Number of batches whose loss is recorded.
N_LOSS_BATCHES = 10
# Simulation: start frame, and frames simulated after a burn-in of contextl frames.
SIMULATE_START_FRAME = 1000
N_SIMULATED_FRAMES = 100
# Number of examples drawn by debug_plot_pose.
N_PLOTTED_EXAMPLES = 3


def seed_everything() -> None:
    """Resets numpy's and torch's random generators to SEED."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def plotted_coordinates(figure) -> list:
    """All line and scatter coordinates drawn in a matplotlib figure, in drawing order.

    Args:
        figure: matplotlib Figure.

    Returns:
        list of (n_points, 2) float arrays.
    """
    coordinates = []
    for ax in figure.axes:
        for line in ax.get_lines():
            coordinates.append(np.column_stack([line.get_xdata(), line.get_ydata()]).astype(float))
        for collection in ax.collections:
            coordinates.append(np.asarray(collection.get_offsets(), dtype=float))
    return coordinates


def run_fly(config_path: str, out_dir: str) -> None:
    """Fly workloads: dataset, losses of a seeded random model, a simulation, and the debug pose plot.

    Args:
        config_path: fly config, e.g. tests/config_fly_test.json with an absolute datadir.
        out_dir: where to write fly.npz and fly_params.pkl.
    """
    import matplotlib
    matplotlib.use('Agg')
    import apf.dataset
    import apf.models
    import apf.simulation
    import apf.utils
    import experiments.flyllm as fly_experiment
    from flyllm.config import read_config
    from flyllm.plotting import debug_plot_pose

    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = read_config(config_path)
    seed_everything()
    (dataset, flyids, track, pose, velocity, sensory, params, isdata, isstart,
     useoutputmask) = fly_experiment.make_dataset(config, 'intrainfile', return_all=True, debug=False)
    arrays = {f'inputs/{k}': v.array for k, v in dataset.inputs.items()}
    arrays.update({f'labels/{k}': v.array for k, v in dataset.labels.items()})
    arrays['chunk_indices'] = dataset.chunk_indices
    arrays['track'] = track.array

    # losses of a seeded random model on the first batches, and the gradient of the first one
    seed_everything()
    model, criterion = apf.models.initialize_model(config, dataset, device)
    loader = apf.dataset.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    mask = torch.nn.Transformer.generate_square_subsequent_mask(dataset.context_length, device=device)
    losses = []
    model.train()
    for i, batch in enumerate(loader):
        if i == N_LOSS_BATCHES:
            break
        pred = model(batch['input'].to(device), mask=mask, is_causal=True)
        loss, _, _ = apf.models.criterion_wrapper(batch, pred, criterion, dataset, config)
        if i == 0:
            loss.backward()
            arrays['first_batch_gradient_norms'] = np.array(
                [p.grad.norm().item() for p in model.parameters() if p.grad is not None])
        losses.append(loss.item())
    arrays['losses'] = np.array(losses)

    # a simulation with the same model, of every fly tracked throughout the window; sampling is seeded
    model.eval()
    burn_in = config['contextl']
    n_frames = burn_in + N_SIMULATED_FRAMES
    window = slice(SIMULATE_START_FRAME, SIMULATE_START_FRAME + n_frames)
    agent_ids = np.nonzero(isdata[window].all(0) & ~isstart[window][1:].any(0))[0]
    seed_everything()
    _, pred_track = apf.simulation.simulate(dataset=dataset, model=model, track=track, pose=pose,
                                            identities=flyids, track_len=n_frames, burn_in=burn_in,
                                            max_contextl=burn_in, agent_ids=agent_ids,
                                            start_frame=SIMULATE_START_FRAME)
    arrays['simulated_track'] = pred_track
    arrays['simulated_agents'] = agent_ids

    # the debug pose plot of a few examples, keypoints inverted from their labels
    # (older code names the conversion helper dict_convert_torch_to_numpy)
    to_numpy = getattr(apf.utils, 'convert_torch_to_numpy', None) or apf.utils.dict_convert_torch_to_numpy
    batch = to_numpy(next(iter(loader)))
    seed_everything()
    _, _, figure = debug_plot_pose(batch, dataset, nsamplesplot=N_PLOTTED_EXAMPLES)
    for i, coordinates in enumerate(plotted_coordinates(figure)):
        arrays[f'plot/{i:03d}'] = coordinates

    np.savez(os.path.join(out_dir, 'fly.npz'), **arrays)
    with open(os.path.join(out_dir, 'fly_params.pkl'), 'wb') as f:
        pickle.dump(dataset.get_params(), f)
    print(f'saved fly outputs to {out_dir}')


def run_synthrat(out_dir: str) -> None:
    """Synthrat workload: the dataset built from the first validation episodes (make_dataset's debug mode).

    Args:
        out_dir: where to write synthrat.npz.
    """
    import experiments.synthrat as synthrat_experiment
    from synthrat.config import read_config

    config = read_config()
    seed_everything()
    dataset, _ = synthrat_experiment.make_dataset(config, config['invalfile'], debug=True)
    arrays = {f'inputs/{k}': v.array for k, v in dataset.inputs.items()}
    arrays.update({f'labels/{k}': v.array for k, v in dataset.labels.items()})
    arrays['chunk_indices'] = dataset.chunk_indices
    np.savez(os.path.join(out_dir, 'synthrat.npz'), **arrays)
    print(f'saved synthrat outputs to {out_dir}')


if __name__ == '__main__':
    if sys.argv[1] == 'fly':
        run_fly(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'synthrat':
        run_synthrat(sys.argv[2])
    else:
        raise ValueError(f'unknown workload {sys.argv[1]!r}; use fly or synthrat')
