"""Tables of the fraction of time simulated and real flies spend in each behavior.

Reads the JAABA scores written by `jaaba_sim.py score` and the walking labels written by
`jaaba_sim.py walk`, and draws two tables, one row per forecasting model variant:

    1. the simulated fraction, under a first row holding the reference model's real
       fraction;
    2. every model's own real fraction, which differ slightly because the models'
       simulation windows differ.

Both are drawn in the layout of the behavior-frequency table in the APF paper figures,
with each column scaled to its own range of the viridis colormap.

The fractions cover the frames that table uses: the 3rd to the 64th frame after a
window's prompt ends, on fly-frames where every behavior was scored on both sides. Each
model is measured on its own windows, not on a frame set shared by all models, so that
table 2 shows the real rates each model's numbers were compared against.

Run it with the same environment as jaaba_sim.py (see its module docstring).
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jaaba_sim import DEFAULT_CLASSIFIER_SET, RESULTS_PARENT_DIR

# Model variants in the order the paper figure lists them, with its display names.
MODEL_DISPLAY_NAMES = {
    'ref': 'Reference',
    'short': 'Shorter context',
    'nobin': 'No discretization',
    'binall': 'Discretize all',
    'predpose': 'Static local pose',
    'bodycentric': 'No handcrafted pose',
    'rawkp': 'Keypoints',
}

# Behavior key in the score/walk files -> display name. Walking comes from the walk files,
# the other three from the JAABA score files.
BEHAVIOR_DISPLAY_NAMES = {
    'walking': 'Walking',
    'jaaba_chase': 'Chasing',
    'jaaba_wingext': 'Wing extension',
    'jaaba_courtship': 'Courtship',
}
WALKING = 'walking'

# The window of simulated frames the paper figure measures over: frames 3 to 64 after the
# prompt. The first two predicted frames are left out there, so they are left out here.
FIRST_SIM_FRAME = 3
LAST_SIM_FRAME = 64

# The model whose real fractions head table 1.
REFERENCE_MODEL = 'ref'

# Table layout, in the style of the paper figure.
CELL_DECIMALS = 2
# The real fractions differ between models only in the third decimal, except for rawkp,
# so that table is printed finer.
REAL_CELL_DECIMALS = 3
COLUMN_LABEL_ROTATION = 25
GRID_COLOR = 'white'
GRID_LINEWIDTH = 0.8
# Cell text is white on dark fill, black on light fill.
DARK_FILL_LUMINANCE = 0.55


def score_path(scores_dir: str, nickname: str, classifier_set: str) -> str:
    """Path of one model's JAABA score file."""
    return os.path.join(scores_dir, f"jaaba_scores_{nickname}_{classifier_set}.npz")


def walk_path(scores_dir: str, nickname: str) -> str:
    """Path of one model's walking file."""
    return os.path.join(scores_dir, f"jaaba_walk_{nickname}.npz")


def behavior_fractions(scores_dir: str, nickname: str, classifier_set: str,
                       first_sim_frame: int = FIRST_SIM_FRAME,
                       last_sim_frame: int = LAST_SIM_FRAME) -> dict:
    """Fraction of time one model's real and simulated flies spend in each behavior.

    The frames counted are this model's own simulated fly-frames whose distance to the
    prompt lies in [first_sim_frame, last_sim_frame] and where every behavior was scored
    on both the real and the simulated side, so all behaviors share one denominator.

    Args:
        scores_dir: directory holding jaaba_scores_*.npz and jaaba_walk_*.npz.
        nickname: model variant, a key of MODEL_DISPLAY_NAMES.
        classifier_set: classifier set the scores were written with, e.g. 'r5nowingtip'.
        first_sim_frame, last_sim_frame: inclusive range of sim_frame values to count.

    Returns:
        dict with 'real' and 'sim', each mapping behavior key -> fraction in [0, 1], and
        'n_frames', the number of fly-frames behind every fraction.

    Raises:
        ValueError: if the walking file does not cover the same fly-frames as the scores,
            which would silently mix up the two sources.
    """
    scores = np.load(score_path(scores_dir, nickname, classifier_set))
    walk = np.load(walk_path(scores_dir, nickname))
    n_agents = scores['sim_frame'].shape[0]     # 10 for rawkp, 11 otherwise
    sim_frame = scores['sim_frame']
    # The walk files are written with the reference track's agent count, so trim; their
    # frames must then agree with the scores exactly.
    if not np.array_equal(walk['sim_frame'][:n_agents], sim_frame):
        raise ValueError(f"{walk_path(scores_dir, nickname)} covers different fly-frames "
                         f"than {score_path(scores_dir, nickname, classifier_set)}")

    counted = (sim_frame >= first_sim_frame) & (sim_frame <= last_sim_frame)
    for behavior in BEHAVIOR_DISPLAY_NAMES:
        source = walk if behavior == WALKING else scores
        for side in ('gt', 'sim'):
            counted &= source[f"{side}_scored_{behavior}"][:n_agents]

    fractions = {'real': {}, 'sim': {}, 'n_frames': int(counted.sum())}
    for behavior in BEHAVIOR_DISPLAY_NAMES:
        source = walk if behavior == WALKING else scores
        for side, label in (('gt', 'real'), ('sim', 'sim')):
            positive = source[f"{side}_behavior_{behavior}"][:n_agents][counted] > 0
            fractions[label][behavior] = float(np.mean(positive))
    return fractions


def table_values(fractions_by_model: dict, models: list[str]) -> tuple:
    """Assemble the two tables' values from per-model fractions.

    Args:
        fractions_by_model: nickname -> behavior_fractions() result.
        models: nicknames in table row order.

    Returns:
        (simulated, real), each (n_rows, n_behaviors) float. `simulated` has the reference
        model's real fractions as its first row, then one simulated row per model; `real`
        has one row per model, holding that model's own real fractions.
    """
    behaviors = list(BEHAVIOR_DISPLAY_NAMES)
    real = np.array([[fractions_by_model[model]['real'][b] for b in behaviors]
                     for model in models])
    simulated = np.array([[fractions_by_model[model]['sim'][b] for b in behaviors]
                          for model in models])
    reference_real = [fractions_by_model[REFERENCE_MODEL]['real'][b] for b in behaviors]
    return np.vstack([reference_real, simulated]), real


def table_figure(values: np.ndarray, row_labels: list[str], column_labels: list[str],
                 title: str, savepath: str | None = None,
                 figsize: tuple = (4.2, 3.2), decimals: int = CELL_DECIMALS) -> None:
    """Draw one table as a viridis heatmap with the values printed in the cells.

    Each column is scaled to its own minimum and maximum, because the behaviors differ in
    how common they are; colors are therefore comparable down a column, not across.

    Args:
        values: (n_rows, n_columns) float fractions in [0, 1].
        row_labels: one label per row, drawn on the left.
        column_labels: one label per column, drawn rotated along the top.
        title: figure title.
        savepath: where to write the figure; None only shows it.
        figsize: figure size in inches.
        decimals: decimal places printed in each cell.

    Side effects:
        Writes savepath if given, and closes the figure.
    """
    spans = values.max(axis=0) - values.min(axis=0)      # (n_columns,)
    normalized = (values - values.min(axis=0)) / np.where(spans > 0, spans, 1.0)
    colors = matplotlib.colormaps['viridis'](normalized)  # (n_rows, n_columns, 4) RGBA

    figure, axes = plt.subplots(figsize=figsize)
    axes.imshow(colors, aspect='auto')
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            red, green, blue = colors[row, column, :3]
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            axes.text(column, row, f"{values[row, column]:.{decimals}f}",
                      ha='center', va='center',
                      color='white' if luminance < DARK_FILL_LUMINANCE else 'black')

    axes.set_xticks(range(len(column_labels)), column_labels, rotation=COLUMN_LABEL_ROTATION,
                    ha='left')
    axes.set_yticks(range(len(row_labels)), row_labels)
    axes.xaxis.tick_top()
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    # Thin grid between cells, as in the paper figure.
    axes.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    axes.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    axes.grid(which='minor', color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    axes.tick_params(which='minor', length=0)
    axes.set_title(title, pad=25)
    figure.tight_layout()
    if savepath is not None:
        figure.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"wrote {savepath}")
    plt.close(figure)


def markdown_table(values: np.ndarray, row_labels: list[str], column_labels: list[str],
                   decimals: int = CELL_DECIMALS) -> str:
    """Render one table as a markdown table, for pasting into notes."""
    lines = ["| | " + " | ".join(column_labels) + " |",
             "|---|" + "|".join(["---:"] * len(column_labels)) + "|"]
    for label, row in zip(row_labels, values):
        lines.append(f"| {label} | "
                     + " | ".join(f"{value:.{decimals}f}" for value in row) + " |")
    return "\n".join(lines)


def main(scores_dir: str, classifier_set: str, models: list[str], out_dir: str,
         first_sim_frame: int, last_sim_frame: int) -> None:
    """Build both tables and write them as figures, printing them as markdown.

    Args:
        scores_dir: directory holding jaaba_scores_*.npz and jaaba_walk_*.npz.
        classifier_set: classifier set the scores were written with.
        models: model nicknames, in row order.
        out_dir: directory for the figures.
        first_sim_frame, last_sim_frame: inclusive range of sim_frame values to count.

    Side effects:
        Writes two PDFs to out_dir and prints both tables.
    """
    fractions_by_model = {}
    for nickname in models:
        fractions_by_model[nickname] = behavior_fractions(
            scores_dir, nickname, classifier_set, first_sim_frame, last_sim_frame)
        print(f"{MODEL_DISPLAY_NAMES[nickname]:<22} "
              f"{fractions_by_model[nickname]['n_frames']:>9} fly-frames")

    simulated, real = table_values(fractions_by_model, models)
    column_labels = list(BEHAVIOR_DISPLAY_NAMES.values())
    model_labels = [MODEL_DISPLAY_NAMES[nickname] for nickname in models]
    frame_range = f"frames {first_sim_frame}-{last_sim_frame} after the prompt"

    os.makedirs(out_dir, exist_ok=True)
    table_figure(simulated, ['Real'] + model_labels, column_labels,
                 f"Fraction of time in action, simulated\n({frame_range})",
                 os.path.join(out_dir, 'behavior_fraction_sim.pdf'),
                 figsize=(4.2, 3.6))
    table_figure(real, model_labels, column_labels,
                 f"Fraction of time in action, real\n({frame_range})",
                 os.path.join(out_dir, 'behavior_fraction_real.pdf'),
                 decimals=REAL_CELL_DECIMALS)

    print(f"\nSimulated fraction of time, {frame_range} "
          f"(first row: real, {MODEL_DISPLAY_NAMES[REFERENCE_MODEL]})\n")
    print(markdown_table(simulated, ['Real'] + model_labels, column_labels))
    print(f"\nReal fraction of time on each model's own frames, {frame_range}\n")
    print(markdown_table(real, model_labels, column_labels, decimals=REAL_CELL_DECIMALS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scores-dir', default=RESULTS_PARENT_DIR,
                        help=f"directory of score and walk files (default {RESULTS_PARENT_DIR})")
    parser.add_argument('--classifiers', default=DEFAULT_CLASSIFIER_SET,
                        help=f"classifier set of the score files (default {DEFAULT_CLASSIFIER_SET})")
    parser.add_argument('--models', nargs='+', default=list(MODEL_DISPLAY_NAMES),
                        choices=list(MODEL_DISPLAY_NAMES), metavar='MODEL',
                        help="model variants, in row order")
    parser.add_argument('--out-dir', default=None,
                        help="directory for the figures (default: the scores directory)")
    parser.add_argument('--first-sim-frame', type=int, default=FIRST_SIM_FRAME,
                        help=f"first frame after the prompt to count (default {FIRST_SIM_FRAME})")
    parser.add_argument('--last-sim-frame', type=int, default=LAST_SIM_FRAME,
                        help=f"last frame after the prompt to count (default {LAST_SIM_FRAME})")
    args = parser.parse_args()
    main(args.scores_dir, args.classifiers, args.models,
         args.out_dir if args.out_dir is not None else args.scores_dir,
         args.first_sim_frame, args.last_sim_frame)
