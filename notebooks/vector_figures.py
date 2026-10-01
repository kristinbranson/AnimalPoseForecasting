"""Redraw APF result figures as vector graphics (PDF and SVG).

The figures in `experiments/full_evaluation.py` are saved through `save_and_show_fig`,
which always writes a 150 dpi PNG, so every panel taken from them is raster and cannot be
relabelled or rescaled in a vector editor. This module redraws them from the same inputs
and saves real vector files, with text kept as text rather than outlines.

Each figure is one function taking a loaded `context` (see `load_context`) plus its own
arguments and returning a matplotlib figure; `FIGURES` maps a command-line name to it.
Adding another figure means writing one such function and adding it to that dict --
the loading, saving and CLI are shared.

Figures available:
    locomotion  the per-bout locomotion panels: the body-centric skeleton with the leg
                tips' trajectories, then those trajectories against time. Port of
                apf.evaluation.plot_locomotions(..., plot_with_features=True).

Run it with the same environment as jaaba_sim.py (see its module docstring).
"""
from __future__ import annotations

import argparse
import os
import pickle

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Resolved from the tree that produced the results (see jaaba_sim's module docstring).
from apf.dataset import Data
from apf.io import load_and_filter_data
from flyllm.config import skeleton_edges, keypointnames
from flyllm.prepare import read_config, load_config_from_model_file
from flyllm.features import body_centric_kp

from jaaba_sim import EXPERIMENTS, VALIDATION_SPLIT, load_simulated

# Where Eyrun's evaluation writes its per-model results, including the pickle holding the
# action classifications the bouts are enumerated from.
EVALUATION_RESULTS_DIR = "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/results"
# Her two pickles hold the same per-frame classifications and differ only in the summary
# counts, so either enumerates the same bouts.
RESULTS_PICKLE = "action_classification_results_max_dist_512.pkl"

# Cached simulation windows. full_evaluation.py, which produced the result figures,
# simulates the training split and writes to 'synthetic'; the JAABA scoring in
# jaaba_sim.py uses the held-out split under 'synthetic_test'.
SIM_PARENT_DIRS = {
    'train': "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/train_data/synthetic",
    'test': "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/train_data/synthetic_test",
}
# The split those figures are drawn from: full_evaluation's 'train_new' side is the
# training data, not the held-out data.
DEFAULT_SPLIT = 'usertrain'

# Her names for the real and simulated sides, as stored in the pickle.
REAL = 'train_new'
SIMULATED = 'sim_new'

VECTOR_FORMATS = ('pdf', 'svg')

# Bouts shorter than this are not treated as bouts, as in apf.evaluation.get_bouts.
MIN_BOUT_FRAMES = 40

# Axis limits of the skeleton panel, in mm, from plot_locomotions.
SKELETON_AXIS_LIMITS = [-3.7, 3.6, -3.5, 2]
TRAJECTORY_LINEWIDTH = 1
START_MARKER_SIZE = 5
SPINE_LINEWIDTH = 0.5
# Rows of the layout grid per bout. The time-series panels span the middle half of them,
# so they are half as tall as the skeleton panel beside them.
SUBROWS_PER_BOUT = 4
# Space between panels, as a fraction of a panel, for the constrained layout.
PANEL_SPACING = 0.1
# Height of a bout title within its panel, in axes coordinates.
TITLE_HEIGHT = 0.82
# Ticks on the millimetre axis of the time-series panels; ones outside the shared limits
# are dropped by matplotlib.
POSITION_TICKS = (-2, -1, 0, 1, 2)


def configure_vector_text() -> None:
    """Keep text editable in the saved files rather than converted to paths.

    Sets TrueType fonts in PDF and PostScript output and disables matplotlib's SVG font
    conversion, so labels stay selectable text in Illustrator or Inkscape.
    """
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'


def save_vector(figure, out_dir: str, stem: str,
                formats: tuple = VECTOR_FORMATS) -> list[str]:
    """Save one figure in every vector format.

    Args:
        figure: the matplotlib figure.
        out_dir: directory to write into; created if missing.
        stem: file name without extension.
        formats: extensions to write, each a vector format matplotlib supports.

    Returns:
        The paths written.

    Side effects:
        Writes one file per format and closes the figure.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for extension in formats:
        path = os.path.join(out_dir, f"{stem}.{extension}")
        figure.savefig(path, bbox_inches='tight')
        paths.append(path)
        print(f"wrote {path}")
    plt.close(figure)
    return paths


def results_dir(nickname: str) -> str:
    """Directory of one model's evaluation results, named as her get_results_dir does."""
    configfile, modelfile = EXPERIMENTS[nickname]
    config_name = os.path.basename(configfile).removesuffix('.json')
    model_name = os.path.basename(modelfile).removesuffix('.pth')
    return os.path.join(EVALUATION_RESULTS_DIR, f"{config_name}_{model_name}")


def load_track(nickname: str, split: str) -> tuple:
    """Load one split's keypoint track, the array the result figures index into.

    Only the keypoints are built, not the pose, velocity and sensory features
    init_datasets computes, since the figures need nothing else. The result is the same
    array experiments/flyllm.py make_dataset wraps as its 'track': the filtered,
    condensed and flip-augmented keypoints of that split.

    Args:
        nickname: model variant, a key of EXPERIMENTS; its config supplies the category
            filter, the keypoint set and the flip augmentation.
        split: MABe split to load, e.g. 'usertrain' for the figures full_evaluation.py
            draws from its training data.

    Returns:
        (track, contextl): track is an apf.dataset.Data whose .array is
        (n_agents, n_frames, 2, n_keypoints) float32 mm keypoints; contextl is the
        model's prompt length in frames.

    Side effects:
        Reads the split's .npz and the checkpoint; needs tens of GB while loading.
    """
    configfile, modelfile = EXPERIMENTS[nickname]
    config = read_config(configfile)
    load_config_from_model_file(loadmodelfile=modelfile, config=config, weights_only=False)
    for key in ('invalfilestr', 'invalfile'):
        config[key] = config[key].replace(VALIDATION_SPLIT, split)
    data, _ = load_and_filter_data(config['invalfile'], config, compute_scale_per_agent=None,
                                   keypointnames=keypointnames, debug=False,
                                   n_frames_per_video=15000, max_n_videos=5)
    # Drop agent slots that hold no fly anywhere, as experiments/flyllm.py load_data does.
    keypoints = data['X']                                  # (n_kp, 2, n_frames, n_agents)
    valid = np.sum(~np.isnan(keypoints[0, 0]), axis=-2) > 0
    return Data('keypoints', keypoints[..., valid].T, []), int(config['contextl'])


def load_context(nickname: str, *, split: str = DEFAULT_SPLIT, sim_set: str = 'train',
                 need_simulated: bool = True) -> dict:
    """Load everything a figure of one model needs: results, real track, simulated track.

    Args:
        nickname: model variant, a key of EXPERIMENTS.
        split: MABe split the figure is drawn from; the result figures use the training
            split, since full_evaluation.py's 'train_new' side is the training data.
        sim_set: which cached simulations to splice in, a key of SIM_PARENT_DIRS.
        need_simulated: also splice the cached simulation windows into a copy of the real
            track, which the simulated side of a figure needs.

    Returns:
        dict with keys:
            nickname (str), split (str),
            results (dict from her action_classification_results pickle: classifications,
                actions, sim_frame, frame_counts, bout_counts),
            tracks ({REAL: array, SIMULATED: array}, each
                (n_agents, n_frames, 2, n_keypoints) float mm keypoints; SIMULATED is
                absent when need_simulated is False),
            sim_frame ((n_agents, n_frames) float, 0 on real agent-frames and k on the
                k-th simulated frame after a prompt).

    Raises:
        ValueError: if the loaded track does not have the shape the results pickle was
            built against, which would make every bout's frame index wrong.

    Side effects:
        Reads a ~2 GB pickle, the split's .npz and the cached simulation windows.
    """
    picklepath = os.path.join(results_dir(nickname), RESULTS_PICKLE)
    print(f"reading {picklepath}", flush=True)
    with open(picklepath, 'rb') as handle:
        results = pickle.load(handle)

    track, contextl = load_track(nickname, split)
    print(f"track {track.array.shape}, results sim_frame {results['sim_frame'].shape}",
          flush=True)
    if track.array.shape[:2] != results['sim_frame'].shape:
        raise ValueError(
            f"the {split} track is {track.array.shape[:2]} but the results were built "
            f"against {results['sim_frame'].shape}; the figures index into that track, "
            f"so pass the split it was built from")

    tracks = {REAL: track.array}
    sim_frame = results['sim_frame']
    if need_simulated:
        configfile, modelfile = EXPERIMENTS[nickname]
        savedir = os.path.join(
            SIM_PARENT_DIRS[sim_set],
            f"{os.path.basename(configfile).removesuffix('.json')}_"
            f"{os.path.basename(modelfile).removesuffix('.pth')}")
        simulated = load_simulated(track, savedir, contextl)
        tracks[SIMULATED] = simulated['sim_track'].array
        sim_frame = simulated['sim_frame']
    return {'nickname': nickname, 'split': split, 'results': results, 'tracks': tracks,
            'sim_frame': sim_frame}


def enumerate_bouts(results: dict, action: str, data_type: str) -> dict:
    """Enumerate one action's bouts exactly as apf.evaluation.get_bouts does.

    The bout index is what the figure titles call "Bout #", so this is what makes a
    published bout findable again: the enumeration is deterministic given the pickle.

    Args:
        results: the loaded results pickle.
        action: action name, e.g. 'walking'.
        data_type: REAL or SIMULATED.

    Returns:
        dict with 'frames' ((n_bouts, 2) int, first and last frame of each bout),
        'agents' ((n_bouts,) int) and 'distance_to_prompt' ((n_bouts,) int, how many
        frames the simulation had run when the bout started; 0 on the real side).
    """
    action_id = results['actions'].index(action)
    classification = results['classifications'][data_type]
    positive = np.where(classification['logits'][:, action_id] > 0.5)[0]
    frames = classification['frames'][positive]
    agents = classification['flies'][positive]

    # A bout breaks wherever the frame number jumps or the fly changes.
    frame_step = np.full(len(frames), 2)
    frame_step[1:] = np.diff(frames)
    agent_step = np.ones(len(agents))
    agent_step[1:] = np.diff(agents)
    starts = np.where((frame_step > 1) | (np.abs(agent_step) > 0))[0]
    bouts = [[starts[i], starts[i + 1] - 1] for i in range(len(starts) - 1)]
    bouts.append([starts[-1], len(frames) - 1])
    bouts = np.array(bouts)
    bouts = bouts[np.array([stop - start + 1 for start, stop in bouts]) >= MIN_BOUT_FRAMES]

    bout_frames = frames[bouts]
    bout_agents = agents[bouts[:, 0]]
    sim_frame = results['sim_frame']
    distance = np.array([sim_frame[agent, start]
                         for agent, start in zip(bout_agents, bout_frames[:, 0])], int)
    return {'frames': bout_frames, 'agents': bout_agents, 'distance_to_prompt': distance}


def _trace_keypoints(action: str) -> list[int]:
    """Keypoint indices whose trajectories a given action's panels trace."""
    if action == 'walking':
        return [i for i, name in enumerate(keypointnames) if 'leg_tip' in name]
    if action == 'perframe_wingext':
        return [i for i, name in enumerate(keypointnames) if 'wing' in name]
    raise ValueError(f"no keypoints defined for action {action!r}")


def _thin_spines(axes) -> None:
    """Draw thin axis spines, as in plot_locomotions."""
    for spine in axes.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)


def bout_trajectories(context: dict, data_type: str, bout_index: int,
                      action: str) -> dict:
    """Body-centric trajectories of one bout's traced keypoints.

    Args:
        context: from load_context().
        data_type: REAL or SIMULATED.
        bout_index: index into that side's bouts, as enumerate_bouts orders them; this is
            what a figure title calls "Bout #".
        action: action whose bouts are enumerated and whose keypoints are traced.

    Returns:
        dict with 'skeleton' ((n_keypoints, 2) mm keypoints at the bout's first frame),
        'traces' ({keypoint index: (2, n_frames) mm trajectory}), 'n_frames' (int),
        'agent' (int) and 'distance_to_prompt' (int, 0 on the real side).
    """
    enumerated = enumerate_bouts(context['results'], action, data_type)
    first_frame, last_frame = enumerated['frames'][bout_index]
    agent = int(enumerated['agents'][bout_index])
    keypoints = context['tracks'][data_type][agent, first_frame:last_frame]
    # body_centric_kp takes n_keypoints x 2 x n_frames and returns it 4-D.
    centered = body_centric_kp(keypoints.T)[0]              # (n_kp, 2, n_frames, 1)
    return {'skeleton': centered[:, :, 0, 0],
            'traces': {keypoint: centered[keypoint, :, :, 0]
                       for keypoint in _trace_keypoints(action)},
            'n_frames': centered.shape[2], 'agent': agent,
            'distance_to_prompt': int(enumerated['distance_to_prompt'][bout_index])}


def shared_limits(drawn: list[dict], margin: float = 0.05) -> tuple:
    """Axis limits that fit every bout, so all panels are drawn at one scale.

    Args:
        drawn: bout_trajectories() results, one per row.
        margin: fraction of each range added on both sides.

    Returns:
        (time_limits, position_limits): (0, max frames) padded, and one millimetre range
        covering both the left-to-right and the back-to-front coordinate of every bout, so
        the two time-series columns share it.
    """
    n_frames = max(bout['n_frames'] for bout in drawn)
    positions = np.concatenate([trace.ravel() for bout in drawn
                                for trace in bout['traces'].values()])
    span = positions.max() - positions.min()
    return ((-margin * n_frames, n_frames * (1 + margin)),
            (positions.min() - margin * span, positions.max() + margin * span))


def locomotion_bouts(context: dict, bouts: list[tuple], action: str) -> list[dict]:
    """Extract every requested bout's trajectories, the only track data the figure needs.

    Kept separate from the drawing so it can be cached: it is a few hundred kB, against
    the minutes and tens of GB that loading the track costs.
    """
    return [bout_trajectories(context, data_type, bout_index, action)
            for data_type, bout_index in bouts]


def locomotion_figure(context: dict | None, bouts: list[tuple], action: str = 'walking',
                      panel_size: float = 2.2, drawn: list[dict] | None = None) -> object:
    """Draw one row of locomotion panels per requested bout.

    Each row holds the fly's skeleton in body-centric coordinates with every traced
    keypoint's trajectory over the bout, then that trajectory's left-to-right coordinate
    against time, then its back-to-front coordinate against time. Time runs along the x
    axis of both time-series panels, and every panel of every row shares one time scale
    and one millimetre scale, so bouts and coordinates can be compared by eye.

    Args:
        context: from load_context(); may be None when drawn is given.
        bouts: one (data_type, bout_index) per row, data_type being REAL or SIMULATED.
        action: action whose bouts are drawn, 'walking' or 'perframe_wingext'.
        panel_size: height of one row in inches; width is three times this.
        drawn: locomotion_bouts() output, to redraw without loading a track.

    Returns:
        The matplotlib figure.
    """
    if drawn is None:
        drawn = locomotion_bouts(context, bouts, action)
    time_limits, position_limits = shared_limits(drawn)

    n_rows = len(bouts)
    # Each bout row is split into SUBROWS_PER_BOUT so the time-series panels can be half
    # the height of the skeleton beside them, centred vertically in the row.
    figure = plt.figure(figsize=(3 * panel_size * 1.45, panel_size * n_rows),
                        layout='constrained')
    grid = figure.add_gridspec(SUBROWS_PER_BOUT * n_rows, 3,
                               hspace=PANEL_SPACING, wspace=PANEL_SPACING)
    quarter = SUBROWS_PER_BOUT // 4
    axes_grid = []
    for row in range(n_rows):
        top = SUBROWS_PER_BOUT * row
        axes_grid.append((
            figure.add_subplot(grid[top:top + SUBROWS_PER_BOUT, 0]),
            figure.add_subplot(grid[top + quarter:top + 3 * quarter, 1]),
            figure.add_subplot(grid[top + quarter:top + 3 * quarter, 2]),
        ))

    for row, ((data_type, bout_index), bout) in enumerate(zip(bouts, drawn)):
        skeleton_axes, sideways_axes, forward_axes = axes_grid[row]
        x, y = bout['skeleton'].T                           # keypoints at the first frame
        for edge in skeleton_edges:
            skeleton_axes.plot(x[edge], y[edge], '-k')
        frames = np.arange(bout['n_frames'])
        for keypoint, trace in bout['traces'].items():
            x, y = trace                                    # (n_frames,) each, mm
            line = skeleton_axes.plot(x, y, '-', linewidth=TRAJECTORY_LINEWIDTH)[0]
            skeleton_axes.plot(x[0], y[0], '.', color=line.get_color(),
                               markersize=START_MARKER_SIZE)
            sideways_axes.plot(frames, x, '-', linewidth=TRAJECTORY_LINEWIDTH)
            forward_axes.plot(frames, y, '-', linewidth=TRAJECTORY_LINEWIDTH)

        title = f"Bout #{bout_index}"
        if data_type == SIMULATED:
            title += f"\nDTP = {bout['distance_to_prompt']} frames"
        # The skeleton axes spans the whole row while its drawing sits in the middle, so
        # the title is placed just above the drawing rather than at the top of the box.
        skeleton_axes.set_title(title, fontsize=10, y=TITLE_HEIGHT)
        skeleton_axes.axis('equal')
        skeleton_axes.axis(SKELETON_AXIS_LIMITS)
        skeleton_axes.set_xticks([])
        skeleton_axes.set_yticks([])
        skeleton_axes.axis('off')

        sideways_axes.set_ylabel("fly's left-to-right (mm)")
        forward_axes.set_ylabel("fly's back-to-front (mm)")
        for axes in (sideways_axes, forward_axes):
            axes.set_xlim(*time_limits)
            axes.set_ylim(*position_limits)
            axes.set_yticks(POSITION_TICKS)
            _thin_spines(axes)
            if row == n_rows - 1:
                axes.set_xlabel('time (frame)')
        label = 'Real' if data_type == REAL else 'Simulated'
        skeleton_axes.text(-0.1, 0.5, label, transform=skeleton_axes.transAxes,
                           rotation=90, va='center', ha='center')

    return figure


# Figure name on the command line -> the function drawing it.
FIGURES = {'locomotion': locomotion_figure}


def _parse_bout(text: str) -> tuple:
    """Parse a 'real:23296' or 'sim:14071' command-line bout into (data_type, index)."""
    side, _, index = text.partition(':')
    if not index.isdigit() or side not in ('real', 'sim'):
        raise argparse.ArgumentTypeError(f"expected real:<index> or sim:<index>, got {text!r}")
    return (REAL if side == 'real' else SIMULATED), int(index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('figure', choices=sorted(FIGURES))
    parser.add_argument('--model', default='ref', choices=sorted(EXPERIMENTS),
                        help="model variant whose results are drawn (default ref)")
    parser.add_argument('--bouts', nargs='+', type=_parse_bout,
                        default=[_parse_bout('real:23296'), _parse_bout('sim:14071')],
                        metavar='SIDE:INDEX',
                        help="bouts to draw, one row each, e.g. real:23296 sim:14071")
    parser.add_argument('--action', default='walking',
                        choices=['walking', 'perframe_wingext'],
                        help="action whose bouts are drawn (default walking)")
    parser.add_argument('--split', default=DEFAULT_SPLIT,
                        help=f"MABe split the results were built from (default {DEFAULT_SPLIT})")
    parser.add_argument('--sim-set', default='train', choices=sorted(SIM_PARENT_DIRS),
                        help="which cached simulations to splice in (default train)")
    parser.add_argument('--out-dir', default='figures',
                        help="directory for the vector files (default ./figures)")
    parser.add_argument('--name', default=None,
                        help="file name without extension (default <figure>_<model>)")
    parser.add_argument('--formats', nargs='+', default=list(VECTOR_FORMATS),
                        help=f"vector formats to write (default {' '.join(VECTOR_FORMATS)})")
    parser.add_argument('--bout-cache', default=None,
                        help="pickle holding the extracted bouts; written on the first "
                             "run and reused afterwards, so restyling the figure needs "
                             "no track load")
    args = parser.parse_args()

    configure_vector_text()
    cache_key = (args.figure, args.model, args.split, args.sim_set, args.action,
                 tuple(args.bouts))
    drawn = None
    if args.bout_cache is not None and os.path.exists(args.bout_cache):
        with open(args.bout_cache, 'rb') as handle:
            cached = pickle.load(handle)
        if cached['key'] == cache_key:
            drawn = cached['drawn']
            print(f"reusing the bouts cached in {args.bout_cache}")
        else:
            print(f"{args.bout_cache} holds different bouts; rebuilding")

    context = None
    if drawn is None:
        need_simulated = any(data_type == SIMULATED for data_type, _ in args.bouts)
        context = load_context(args.model, split=args.split, sim_set=args.sim_set,
                               need_simulated=need_simulated)
        drawn = locomotion_bouts(context, args.bouts, args.action)
        if args.bout_cache is not None:
            with open(args.bout_cache, 'wb') as handle:
                pickle.dump({'key': cache_key, 'drawn': drawn}, handle)
            print(f"cached the extracted bouts in {args.bout_cache}")

    figure = FIGURES[args.figure](context, args.bouts, action=args.action, drawn=drawn)
    stem = args.name if args.name is not None else f"{args.figure}_{args.model}"
    save_vector(figure, args.out_dir, stem, formats=tuple(args.formats))


if __name__ == '__main__':
    main()
