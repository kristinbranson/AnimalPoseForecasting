"""Animating mouse keypoint tracks, e.g. real and simulated ones side by side.  The counterpart of
flyllm/simulation.py's animate_pose."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon

from apf.utils import save_animation
from mousellm.config import ARENA_CENTER_PX, ARENA_SIZE_CM, CM_PER_PX, FPS, keypointnames
from mousellm.convert_mabe22 import FRAME_SIZE_PX

# One color per mouse slot
MOUSE_COLORS = ['tab:red', 'tab:green', 'tab:orange']

# How a mouse is drawn, after Fig. 1 of the MABe22 paper: a filled head (nose, ears and neck) and a filled
# body (neck, paws and tail base), with the spine and the tail as lines.  These are for drawing only, and are
# here rather than in mousellm.config, whose contents are part of the dataset cache key.
_kp = keypointnames.index
HEAD_POLYGON = [_kp('nose'), _kp('left_ear'), _kp('neck'), _kp('right_ear')]
BODY_POLYGON = [_kp('neck'), _kp('left_forepaw'), _kp('left_hindpaw'), _kp('tail_base'), _kp('right_hindpaw'),
                _kp('right_forepaw')]
LINES = [[_kp('left_ear'), _kp('right_ear')],
         [_kp('neck'), _kp('center'), _kp('tail_base')],
         [_kp('tail_base'), _kp('tail_middle'), _kp('tail_tip')]]

# The bounds of the 512-px video frame the keypoints were in, in cm: (x0, x1, y0, y1)
FRAME_BOUNDS_CM = np.array([0 - ARENA_CENTER_PX[0], FRAME_SIZE_PX - ARENA_CENTER_PX[0],
                            0 - ARENA_CENTER_PX[1], FRAME_SIZE_PX - ARENA_CENTER_PX[1]]) * CM_PER_PX


def _box(ax, x0, x1, y0, y1, **kwargs):
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], **kwargs)


def animate_pose(Xkps: dict, burn_in: int = 0, out_file_path_maybe: str | None = None, fps: float = FPS,
                 view_scale: float = 2., figsizebase: float = 5., ms: float = 2., lw: float = 1.,
                 trail_len: int = 30) -> animation.FuncAnimation:
    """ Animates mice in the arena, one panel per set of tracks, e.g. {'Real': gt_track, 'Simulated': pred_track}.

    Each mouse is drawn in its own color, as a filled head and body with the spine and tail as lines, its
    keypoints as black dots, and a trail of its center keypoint's last trail_len frames.  The square arena
    floor is drawn in grey and the video frame's bounds in black.  The view is the video frame scaled by
    view_scale about its center, the same in every panel, so mice that wander far enough go out of view.  The
    title gives the frame number, counted from the first frame after the burn-in, so the burn-in's frames are
    numbered -burn_in to -1.  y points down, as in the videos.  A mouse's
    NaN keypoints aren't drawn, nor are the shapes they're part of.

    Args:
        Xkps: Panel title -> (x, y) keypoint positions in cm, (n_agents, n_frames, 2, n_keypoints) float array,
            as apf.simulation.simulate returns them.  All must have the same number of frames.
        burn_in: Number of leading frames that are real data in every panel, numbered negatively in the title.
        out_file_path_maybe: If not None, the animation is saved to this file, e.g. an .mp4.
        fps: Frames per second of the saved animation.
        view_scale: Size of the view, relative to the video frame.
        figsizebase: Size of each panel, in inches.
        ms: Marker size of the keypoints.
        lw: Line width of the mice's outlines and lines.
        trail_len: Number of past frames of each mouse's center keypoint to draw as a trail.

    Returns:
        The animation.
    """
    names = list(Xkps.keys())
    n_frames = Xkps[names[0]].shape[1]
    assert all(X.shape[1] == n_frames for X in Xkps.values()), 'All panels must have the same number of frames'
    center_idx = keypointnames.index('center')
    x0, x1, y0, y1 = FRAME_BOUNDS_CM
    view_center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    view_half_size = np.array([(x1 - x0) / 2, (y1 - y0) / 2]) * view_scale

    fig, axs = plt.subplots(1, len(names), figsize=(figsizebase * len(names), figsizebase * 1.1), squeeze=False)
    axs = axs[0]
    artists = []
    for ax, name in zip(axs, names):
        half = ARENA_SIZE_CM / 2
        _box(ax, -half, half, -half, half, color=[.6, .6, .6], lw=1)
        _box(ax, x0, x1, y0, y1, color='k', lw=1)
        ax.set_xlim(view_center[0] - view_half_size[0], view_center[0] + view_half_size[0])
        ax.set_ylim(view_center[1] + view_half_size[1], view_center[1] - view_half_size[1])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        mice = []
        for m in range(Xkps[name].shape[0]):
            color = MOUSE_COLORS[m % len(MOUSE_COLORS)]
            polygons = [ax.add_patch(Polygon(np.zeros((len(idx), 2)), closed=True, facecolor=color, alpha=.4,
                                             edgecolor=color, lw=lw, visible=False))
                        for idx in [HEAD_POLYGON, BODY_POLYGON]]
            lines = [ax.plot([], [], '-', color=color, lw=lw)[0] for _ in LINES]
            kps, = ax.plot([], [], '.', color='k', ms=ms * 2)
            trail, = ax.plot([], [], '-', color=color, lw=lw / 2, alpha=.5)
            mice.append((polygons, lines, kps, trail))
        ax.set_title(name)
        artists.append((name, mice))
    suptitle = fig.suptitle('')

    def update(t):
        suptitle.set_text(f'Frame {t - burn_in}')
        changed = [suptitle]
        for name, mice in artists:
            X = Xkps[name]
            for m, (polygons, lines, kps, trail) in enumerate(mice):
                xy = X[m, t].T                                         # (n_keypoints, 2)
                for polygon, idx in zip(polygons, [HEAD_POLYGON, BODY_POLYGON]):
                    ok = not np.isnan(xy[idx]).any()
                    polygon.set_visible(ok)
                    if ok:
                        polygon.set_xy(xy[idx])
                for line, idx in zip(lines, LINES):
                    line.set_data(xy[idx, 0], xy[idx, 1])
                kps.set_data(xy[:, 0], xy[:, 1])
                t0 = max(0, t - trail_len)
                trail.set_data(X[m, t0:t + 1, 0, center_idx], X[m, t0:t + 1, 1, center_idx])
                changed += polygons + lines + [kps, trail]
        return changed

    fig.tight_layout(rect=(0, 0, 1, .97))
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    if out_file_path_maybe is not None:
        save_animation(ani, out_file_path_maybe, fps=fps)
    return ani
