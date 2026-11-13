"""Animating mouse keypoint tracks, e.g. real and simulated ones side by side.  The counterpart of
flyllm/simulation.py's animate_pose."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from apf.utils import save_animation
from mousellm.config import ARENA_SIZE_CM, FPS, keypointnames, skeleton_edges

# One color per mouse slot
MOUSE_COLORS = ['tab:red', 'tab:green', 'tab:blue']


def animate_pose(Xkps: dict, burn_in: int = 0, out_file_path_maybe: str | None = None, fps: float = FPS,
                 view_half_width_cm_maybe: float | None = None, figsizebase: float = 5., ms: float = 2.,
                 lw: float = 1., trail_len: int = 30) -> animation.FuncAnimation:
    """ Animates mice in the arena, one panel per set of tracks, e.g. {'Real': gt_track, 'Simulated': pred_track}.

    Each mouse is drawn as its skeleton in its own color, with a larger dot at the nose and a trail of its
    center keypoint's last trail_len frames.  The square arena floor is drawn in grey.  The title gives the
    frame number and whether it's in the burn-in.  y points down, as in
    the videos.  NaN keypoints aren't drawn.

    Args:
        Xkps: Panel title -> (x, y) keypoint positions in cm, (n_agents, n_frames, 2, n_keypoints) float array,
            as apf.simulation.simulate returns them.  All must have the same number of frames.
        burn_in: Number of leading frames that are real data in every panel, labeled as burn-in.
        out_file_path_maybe: If not None, the animation is saved to this file, e.g. an .mp4.
        fps: Frames per second of the saved animation.
        view_half_width_cm_maybe: Half the width of the view about the arena center, in cm.  None for the
            smallest view that shows the arena and every keypoint, shared by all panels.
        figsizebase: Size of each panel, in inches.
        ms: Marker size of the keypoints; the nose's is 3 times this.
        lw: Line width of the skeleton.
        trail_len: Number of past frames of each mouse's center keypoint to draw as a trail.

    Returns:
        The animation.
    """
    names = list(Xkps.keys())
    n_frames = Xkps[names[0]].shape[1]
    assert all(X.shape[1] == n_frames for X in Xkps.values()), 'All panels must have the same number of frames'
    if view_half_width_cm_maybe is None:
        view_half_width_cm = max(ARENA_SIZE_CM / 2, max(np.nanmax(np.abs(X)) for X in Xkps.values())) * 1.05
    else:
        view_half_width_cm = view_half_width_cm_maybe
    nose_idx = keypointnames.index('nose')
    center_idx = keypointnames.index('center')

    fig, axs = plt.subplots(1, len(names), figsize=(figsizebase * len(names), figsizebase * 1.08), squeeze=False)
    axs = axs[0]
    artists = []
    for ax, name in zip(axs, names):
        half = ARENA_SIZE_CM / 2
        ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color=[.6, .6, .6], lw=1)
        ax.set_xlim(-view_half_width_cm, view_half_width_cm)
        ax.set_ylim(view_half_width_cm, -view_half_width_cm)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        n_agents = Xkps[name].shape[0]
        mice = []
        for m in range(n_agents):
            color = MOUSE_COLORS[m % len(MOUSE_COLORS)]
            edges = [ax.plot([], [], '-', color=color, lw=lw)[0] for _ in skeleton_edges]
            kps, = ax.plot([], [], '.', color=color, ms=ms * 2)
            nose, = ax.plot([], [], 'o', color=color, ms=ms * 3)
            trail, = ax.plot([], [], '-', color=color, lw=lw / 2, alpha=.5)
            mice.append((edges, kps, nose, trail))
        ax.set_title(name)
        artists.append((name, mice))
    suptitle = fig.suptitle('')

    def update(t):
        if burn_in > 0:
            suptitle.set_text(f'Frame {t}, ' + ('burn-in (real data)' if t < burn_in else
                                                f'{t - burn_in + 1} frames after the burn-in'))
        else:
            suptitle.set_text(f'Frame {t}')
        changed = [suptitle]
        for name, mice in artists:
            X = Xkps[name]
            for m, (edges, kps, nose, trail) in enumerate(mice):
                x, y = X[m, t, 0], X[m, t, 1]
                for edge, (a, b) in zip(edges, skeleton_edges):
                    edge.set_data([x[a], x[b]], [y[a], y[b]])
                kps.set_data(x, y)
                nose.set_data([x[nose_idx]], [y[nose_idx]])
                t0 = max(0, t - trail_len)
                trail.set_data(X[m, t0:t + 1, 0, center_idx], X[m, t0:t + 1, 1, center_idx])
                changed += edges + [kps, nose, trail]
        return changed

    fig.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    if out_file_path_maybe is not None:
        save_animation(ani, out_file_path_maybe, fps=fps)
    return ani
