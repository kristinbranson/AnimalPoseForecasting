"""Stacked bars of how often a simulation keeps its prompt's category.

Redraws the confusion matrices in Eyrun's `mabe_results_figures/confusion_matrix_faction_*.pdf`
as one stacked bar per prompt class: black for the fraction of simulated sequences the
probe assigns the prompt's own label, grey for the fraction it cannot decide ('mixed').

Her analysis, in her notebook, labels each simulated sequence twice with a linear probe on
the model's hidden states: the prompt's label comes from the real frames before the
simulation, the simulated stretch's label from the predicted frames. A simulated stretch
counts as class 1 when more than 60% of its frames are positive, class 0 when fewer than
40% are, and 'mixed' in between. The confusion matrix is row-normalised, so each row's
three fractions sum to 1; the fraction left out of these bars is the one assigned the
*other* label.

The numbers are read from her PDFs rather than recomputed, so this stays tied to the
analysis that produced them.

Run it with the same environment as jaaba_sim.py (see its module docstring).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vector_figures import configure_vector_text, save_vector

CONFUSION_DIR = ("/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/"
                 "mabe_results_figures")
CONFUSION_FILE = "confusion_matrix_faction_{category}.pdf"
ACCURACY_FILE = "mean_prompt_accuracy_{category}.pdf"

# Category -> (name of its class 0, name of its class 1), in her notebook's order.
CATEGORY_CLASS_NAMES = {
    'female': ('male', 'female'),
    'courtship': ('other lines', 'courtship lines'),
    'blind': ('not blind', 'blind'),
    '91B01': ('other lines', '91B01'),
}

# Classes are drawn class 1 first, so each category leads with the class it is named for.
CLASS_DRAW_ORDER = (1, 0)

CORRECT_COLOR = 'black'
MIXED_COLOR = '0.65'
# Sequences the simulation gave the other class; drawn only in the accuracy panel.
FLIPPED_COLOR = 'white'
FLIPPED_EDGE_COLOR = 'black'
OUTCOME_NAMES = ('correct', 'mixed', 'flipped')
BAR_HEIGHT = 0.88
# Blank rows between one category's pair of bars and the next.
CATEGORY_GAP = 0.45
# Where a category's name sits, in axes coordinates, left of its class names.
CATEGORY_LABEL_X = -0.58
# Text size in points, as rendered at 100%. Panel a of the assembled MABe figure is 12 pt,
# measured from its cap height, so matching it means drawing these panels at 12 too.
LABEL_FONTSIZE = 8
# Panel size in inches at LABEL_FONTSIZE = 8; grown with the font so text does not crowd.
BASE_FIGSIZE = (4.4, 2.4)

# "0 (0.21)   0.592 0.225 0.183": class, its share of sequences, then the row's fractions
# for predicted 0, predicted 1 and mixed.
_ROW = re.compile(r"^\s*([01])\s*\(([\d.]+)\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
# "Confusion matrix - female (52790)"
_TITLE = re.compile(r"Confusion matrix\s*-\s*(\S+)\s*\((\d+)\)")
# "0 0.989 0.928 0.981": class, then its mean prompt accuracy per simulated label.
_ACCURACY_ROW = re.compile(r"^\s*([01])\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def read_confusion(category: str, confusion_dir: str = CONFUSION_DIR) -> dict:
    """Read one category's confusion matrix out of its PDF.

    Args:
        category: a key of CATEGORY_CLASS_NAMES.
        confusion_dir: directory holding confusion_matrix_faction_<category>.pdf.

    Returns:
        dict with 'matrix' ((2, 3) float, rows are the prompt's class and columns are the
        simulation's label: 0, 1, mixed; each row sums to 1), 'class_share' ((2,) float,
        each class's share of the sequences) and 'n_sequences' (int).

    Raises:
        ValueError: if the PDF does not hold the two expected rows, or a row does not sum
            to 1, which would mean the file is not the matrix this expects.
    """
    path = os.path.join(confusion_dir, CONFUSION_FILE.format(category=category))
    text = subprocess.run(['pdftotext', '-layout', path, '-'],
                          capture_output=True, text=True, check=True).stdout

    matrix, share = np.zeros((2, 3)), np.zeros(2)
    seen = set()
    n_sequences = 0
    for line in text.splitlines():
        title = _TITLE.search(line)
        if title is not None:
            n_sequences = int(title.group(2))
        row = _ROW.match(line)
        if row is not None:
            index = int(row.group(1))
            share[index] = float(row.group(2))
            matrix[index] = [float(row.group(i)) for i in (3, 4, 5)]
            seen.add(index)
    if seen != {0, 1}:
        raise ValueError(f"{path} does not hold both class rows; found {sorted(seen)}")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1, atol=0.01):
        raise ValueError(f"{path}: rows sum to {row_sums}, not 1; not a row-normalised "
                         f"confusion matrix")
    return {'matrix': matrix, 'class_share': share, 'n_sequences': n_sequences}


def use_font_size(size: float) -> None:
    """Set the text size these panels draw with, in points as rendered at 100%.

    Matching a panel assembled from another figure means matching its rendered size: the
    MABe figure's table panel measures 12 pt, from a cap height of 8.75 pt in a render of
    the assembled SVG. Changing this does not rescale the panel, so a larger size leaves
    less room for the bars unless the panel is widened too.

    Args:
        size: text size in points.
    """
    global LABEL_FONTSIZE
    LABEL_FONTSIZE = size


def read_prompt_accuracy(category: str, confusion_dir: str = CONFUSION_DIR) -> np.ndarray:
    """Read one category's mean prompt accuracy matrix out of its PDF.

    The prompt accuracy is how well the probe classified the real frames *before* the
    simulation, averaged over the sequences in each cell. Read alongside the confusion
    matrix it says whether the sequences the simulation got wrong were ones whose prompt
    was already ambiguous.

    Args:
        category: a key of CATEGORY_CLASS_NAMES.
        confusion_dir: directory holding mean_prompt_accuracy_<category>.pdf.

    Returns:
        (2, 3) float: rows are the prompt's class, columns the simulation's label
        (0, 1, mixed), values mean prompt accuracies in [0, 1].

    Raises:
        ValueError: if the PDF does not hold both class rows or holds values outside
            [0, 1], meaning it is not the matrix this expects.
    """
    path = os.path.join(confusion_dir, ACCURACY_FILE.format(category=category))
    text = subprocess.run(['pdftotext', '-layout', path, '-'],
                          capture_output=True, text=True, check=True).stdout
    matrix = np.zeros((2, 3))
    seen = set()
    for line in text.splitlines():
        row = _ACCURACY_ROW.match(line)
        if row is not None:
            index = int(row.group(1))
            matrix[index] = [float(row.group(i)) for i in (2, 3, 4)]
            seen.add(index)
    if seen != {0, 1}:
        raise ValueError(f"{path} does not hold both class rows; found {sorted(seen)}")
    if matrix.min() < 0 or matrix.max() > 1:
        raise ValueError(f"{path}: accuracies outside [0, 1]: {matrix}")
    return matrix


def outcome_values(matrix: np.ndarray, label: int) -> tuple:
    """Split one class's row into (correct, mixed, flipped), whatever the row holds.

    Args:
        matrix: (2, 3) confusion fractions or prompt accuracies; columns are the
            simulation's label 0, 1, mixed.
        label: the prompt's class, 0 or 1.

    Returns:
        The row's value for the simulation keeping that class, being mixed, and giving
        the other class.
    """
    return matrix[label, label], matrix[label, 2], matrix[label, 1 - label]


def prompt_accuracy_bars(categories: list[str], confusion_dir: str = CONFUSION_DIR,
                         figsize: tuple = BASE_FIGSIZE, group_labels: bool = False) -> object:
    """Draw the mean prompt accuracy of each class, split by what the simulation did.

    Rows match the confusion bars, so the two panels line up: one row per prompt class,
    in the same order. Within a row there are three bars -- the sequences the simulation
    kept in that class, the mixed ones, and the ones it gave the other class -- each the
    mean accuracy of the probe on those sequences' real prompt frames.

    Args:
        categories: categories to draw, in top-to-bottom order.
        confusion_dir: directory holding her matrix PDFs.
        figsize: figure size in inches.
        group_labels: also write each category's name to the left of its pair of rows.

    Returns:
        The matplotlib figure.
    """
    figure, axes = plt.subplots(figsize=figsize, layout='constrained')
    colors = (CORRECT_COLOR, MIXED_COLOR, FLIPPED_COLOR)
    bar_height = BAR_HEIGHT / len(colors)

    position = 0.0
    tick_positions, tick_labels = [], []
    for category in categories:
        accuracy = read_prompt_accuracy(category, confusion_dir)
        class_names = CATEGORY_CLASS_NAMES[category]
        first_position = position
        for label in CLASS_DRAW_ORDER:
            values = outcome_values(accuracy, label)
            # The three bars are stacked within one row's height, correct on top.
            offsets = (np.arange(len(colors)) - (len(colors) - 1) / 2) * bar_height
            for value, color, offset in zip(values, colors, offsets):
                axes.barh(position + offset, value, height=bar_height, color=color,
                          edgecolor=FLIPPED_EDGE_COLOR if color == FLIPPED_COLOR else color,
                          linewidth=0.6)
            tick_positions.append(position)
            tick_labels.append(class_names[label])
            position += 1
        if group_labels:
            axes.text(CATEGORY_LABEL_X, (first_position + position - 1) / 2, category,
                      transform=axes.get_yaxis_transform(), ha='center', va='center',
                      fontsize=LABEL_FONTSIZE)
        position += CATEGORY_GAP

    axes.set_yticks(tick_positions, tick_labels, fontsize=LABEL_FONTSIZE)
    axes.invert_yaxis()
    axes.set_xlim(0, 1)
    axes.set_xlabel('mean prompt accuracy', fontsize=LABEL_FONTSIZE)
    axes.tick_params(axis='x', labelsize=LABEL_FONTSIZE)
    axes.spines[['top', 'right']].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color,
                             edgecolor=FLIPPED_EDGE_COLOR if color == FLIPPED_COLOR else color)
               for color in colors]
    axes.legend(handles, OUTCOME_NAMES, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                ncol=len(colors), frameon=False, fontsize=LABEL_FONTSIZE,
                handlelength=1.2, columnspacing=1.2)
    return figure


def confusion_bars(categories: list[str], confusion_dir: str = CONFUSION_DIR,
                   figsize: tuple = BASE_FIGSIZE, group_labels: bool = False) -> object:
    """Draw one stacked bar per prompt class: fraction kept, then fraction mixed.

    Args:
        categories: categories to draw, in top-to-bottom order; keys of
            CATEGORY_CLASS_NAMES.
        confusion_dir: directory holding her confusion matrix PDFs.
        figsize: figure size in inches.
        group_labels: write each category's name and sequence count to the left of its
            pair of bars. Off by default, as in the assembled MABe figure, where the
            class names alone carry the grouping.

    Returns:
        The matplotlib figure.
    """
    figure, axes = plt.subplots(figsize=figsize, layout='constrained')

    position = 0.0
    tick_positions, tick_labels = [], []
    for category in categories:
        confusion = read_confusion(category, confusion_dir)
        class_names = CATEGORY_CLASS_NAMES[category]
        first_position = position
        for label in CLASS_DRAW_ORDER:
            correct = confusion['matrix'][label, label]
            mixed = confusion['matrix'][label, 2]
            axes.barh(position, correct, height=BAR_HEIGHT, color=CORRECT_COLOR)
            axes.barh(position, mixed, height=BAR_HEIGHT, left=correct, color=MIXED_COLOR)
            tick_positions.append(position)
            tick_labels.append(class_names[label])
            position += 1
        if group_labels:
            # The category's name sits to the left of its pair of bars.
            axes.text(CATEGORY_LABEL_X, (first_position + position - 1) / 2,
                      f"{category}\n({confusion['n_sequences']:,} seqs)",
                      transform=axes.get_yaxis_transform(), ha='center', va='center',
                      fontsize=LABEL_FONTSIZE)
        position += CATEGORY_GAP

    axes.set_yticks(tick_positions, tick_labels, fontsize=LABEL_FONTSIZE)
    axes.invert_yaxis()
    axes.set_xlim(0, 1)
    axes.set_xlabel('fraction of simulated sequences', fontsize=LABEL_FONTSIZE)
    axes.tick_params(axis='x', labelsize=LABEL_FONTSIZE)
    axes.spines[['top', 'right']].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CORRECT_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=MIXED_COLOR)]
    axes.legend(handles, ["simulation keeps the prompt's class", 'mixed'],
                loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                fontsize=LABEL_FONTSIZE, handlelength=1.2, columnspacing=1.2)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--categories', nargs='+', default=list(CATEGORY_CLASS_NAMES),
                        choices=list(CATEGORY_CLASS_NAMES), metavar='CATEGORY',
                        help="categories to draw, top to bottom")
    parser.add_argument('--confusion-dir', default=CONFUSION_DIR,
                        help="directory holding the confusion matrix PDFs")
    parser.add_argument('--out-dir', default='figures',
                        help="directory for the vector files (default ./figures)")
    parser.add_argument('--plots', nargs='+', default=['confusion', 'accuracy'],
                        choices=['confusion', 'accuracy'],
                        help="which panels to draw (default both)")
    parser.add_argument('--name', default=None,
                        help="file name without extension; default prompt_<plot>_bars")
    parser.add_argument('--font-size', type=float, default=LABEL_FONTSIZE,
                        help=f"text size in points (default {LABEL_FONTSIZE}; the MABe "
                             f"figure's table panel is 12)")
    parser.add_argument('--figsize', type=float, nargs=2, default=list(BASE_FIGSIZE),
                        metavar=('WIDTH', 'HEIGHT'),
                        help=f"panel size in inches (default {BASE_FIGSIZE[0]} "
                             f"{BASE_FIGSIZE[1]})")
    parser.add_argument('--group-labels', action='store_true',
                        help="also name each category left of its pair of bars")
    args = parser.parse_args()

    configure_vector_text()
    use_font_size(args.font_size)
    for category in args.categories:
        confusion = read_confusion(category, args.confusion_dir)
        accuracy = read_prompt_accuracy(category, args.confusion_dir)
        names = CATEGORY_CLASS_NAMES[category]
        print(f"{category} ({confusion['n_sequences']} sequences)")
        for label in CLASS_DRAW_ORDER:
            kept, mixed, flipped = outcome_values(confusion['matrix'], label)
            print(f"  {names[label]:<16} share {confusion['class_share'][label]:.2f}  "
                  f"kept {kept:.3f}  mixed {mixed:.3f}  flipped {flipped:.3f}  "
                  f"prompt accuracy " + " ".join(
                      f"{name} {value:.3f}" for name, value
                      in zip(OUTCOME_NAMES, outcome_values(accuracy, label))))

    builders = {'confusion': (confusion_bars, 'prompt_confusion_bars'),
                'accuracy': (prompt_accuracy_bars, 'prompt_accuracy_bars')}
    for plot in args.plots:
        build, default_name = builders[plot]
        name = args.name if args.name is not None and len(args.plots) == 1 else default_name
        save_vector(build(args.categories, args.confusion_dir, figsize=tuple(args.figsize),
                          group_labels=args.group_labels), args.out_dir, name)


if __name__ == '__main__':
    main()
