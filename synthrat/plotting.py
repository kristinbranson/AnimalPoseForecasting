"""Plotting helpers for RatInABox synthetic-rat trajectories and sensory neurons.

`plot_episode` draws one saved trajectory on top of the Environment, with the current
agent position and head-direction marker. `visualize_sensory` takes a trajectory plus
its computed sensory firing rates and a timestep, and produces a composite figure:
trajectory and agent state (position, heading, velocity) on the left, one subplot per
sensory population on the right showing firing rates at that time against each cell's
preferred tuning.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import ratinabox
import ratinabox.utils

from synthrat.sensory import CELL_VECTOR_CLASSES, head_direction_from_orientation

def plot_episode(track_curr,Env,axcurr=None,**kwargs):
    """Plot one episode's trajectory on top of the environment.

    Draws the path as semi-transparent dots, the final agent position as a
    solid dot, and a triangle marker rotated to show the final head direction.

    Parameters
    ----------
    track_curr : dict
        Must contain 'pos' (T, 2) and 'head_direction' (T, 2).
    Env : ratinabox Environment
        Drawn in the background via `Env.plot_environment`.
    axcurr : matplotlib Axes, optional
        If None, uses `plt.gca()`.
    **kwargs :
        alpha       : float (default 0.7) — transparency of trajectory dots
        point_size  : float (default 15)  — size of trajectory dots
        agent_color : str   (default "r") — color of final position + heading

    Returns
    -------
    (htraj, hagent, hd) : PathCollection handles for the trajectory, the
        final-position dot, and the head-direction triangle, in that order.
    """
    alpha = kwargs.get("alpha", 0.7) #transparency of trajectory
    point_size = kwargs.get("point_size", 15) #size of trajectory points
    agent_color = kwargs.get("agent_color", "r") #color of the agent if show_agent is True
    
    if axcurr is None:
        axcurr = plt.gca()
    fig = axcurr.figure
    if isinstance(track_curr, dict):
        trajectory = track_curr['pos']
        head_direction = track_curr['head_direction']
    elif hasattr(track_curr,'shape'):
        trajectory = track_curr[..., :2]
        # last dim is the orientation (radians, fly convention); convert to a unit heading
        # vector so downstream code (get_bearing) sees a consistent (T, 2) shape.
        head_direction = head_direction_from_orientation(track_curr[..., 2])
    else:
        raise ValueError("track_curr must be a dict with 'pos' and 'head_direction' keys, or an array with shape (..., 3) where the last dimension is (x, y, orientation).")
    _, _ = Env.plot_environment(fig=fig, ax=axcurr, autosave=False)
    htraj = axcurr.scatter(
        trajectory[:-1, 0],
        trajectory[:-1, 1],
        s=point_size,
        linewidth=0,
        alpha=alpha,
    )
    hagent = axcurr.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        s=40,
        c=agent_color,
        linewidth=0,
        marker="o",
    )
    rotated_agent_marker = matplotlib.markers.MarkerStyle(marker=[(-1,0),(1,0),(0,4)]) # a triangle
    rotated_agent_marker._transform = rotated_agent_marker.get_transform().rotate_deg(-ratinabox.utils.get_bearing(head_direction[-1])*180/np.pi)
    hd = axcurr.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        s=200,
        alpha=1,
        c=agent_color,
        linewidth=0,
        marker=rotated_agent_marker,
    )
    return htraj, hagent, hd

def plot_agent_heading(axcurr, pos, head_dir, triheight=5, triwidth=4,
                       s=600, c='r', edgecolor='r', linewidth=0, zorder=5):
    """Draw a rotated triangle marker indicating the agent's heading.

    The triangle is defined with its base along y=0 and tip at (0, triheight),
    then rotated by `-get_bearing(head_dir)` degrees (compass convention)
    so the tip points in the heading direction.

    Parameters
    ----------
    axcurr : matplotlib Axes (Cartesian)
    pos : (2,) world-frame position where the triangle is placed.
    head_dir : (2,) unit vector in world frame.
    triheight, triwidth : float
        Dimensions of the unit-marker triangle before scatter scales by `s`.
        Larger `triheight` / smaller `triwidth` → more arrow-like.
    s : float, scatter marker area in points².
    c, edgecolor, linewidth : standard matplotlib scatter styling.
    zorder : float, draw order.
    """
    hd_marker = matplotlib.markers.MarkerStyle(marker=[(-triwidth/2, 0), (triwidth/2, 0), (0, triheight)])
    hd_marker._transform = hd_marker.get_transform().rotate_deg(
        -ratinabox.utils.get_bearing(head_dir) * 180 / np.pi
    )
    axcurr.scatter(pos[0], pos[1],
                    s=s, c=c,
                    edgecolor=edgecolor, linewidth=linewidth,
                    marker=hd_marker, zorder=zorder)


def plot_agent_state(axcurr, pos, head_dir, vel, t):
    """Draw the agent's position, heading, and velocity at timestep `t`.

    Rendered elements (in draw order):
      - heading triangle (orange C1) via `plot_agent_heading`
      - position dot (black)
      - velocity arrow (green C2) via `quiver` — only if `vel` is provided

    Parameters
    ----------
    axcurr : matplotlib Axes (Cartesian)
    pos : (T, 2) array of positions over the episode.
    head_dir : (T, 2) array of unit heading vectors.
    vel : (T, 2) array of velocities, or None.
    t : int, index into the arrays above.
    """
    plot_agent_heading(axcurr, pos[t], head_dir[t], triheight=5, triwidth=4,
                       s=2000, c='C1', edgecolor='C1', linewidth=1.5, zorder=4)
    axcurr.scatter(pos[t, 0], pos[t, 1],
                    s=80, c="k", linewidth=0, zorder=5)

    if vel is not None:
        axcurr.quiver(pos[t, 0], pos[t, 1], vel[t, 0], vel[t, 1],
                        angles="xy", scale_units="xy", scale=1,
                        color="C2", width=0.005, zorder=6)
        
def plot_head_direction(axcurr, neuron, fr, head_dir=None, head_bearing=None,
                        min_height=0.1):
    """Plot an HDC/VelocityCells population on a polar axis as von-Mises bumps.

    Each cell is drawn as a smooth curve peaking at its preferred angle, with
    peak height = max(firing rate `fr[j]`, `min_height`) and angular width set
    by `neuron.angular_tunings[j]` (via κ = 1/σ²). The `min_height` floor
    keeps every cell visible even when its firing rate is zero; cells with
    non-zero firing rate appear larger in proportion. An orange (C1) radial
    line marks the current heading direction so you can see how well-aligned
    the population response is with the agent's actual heading.

    Convention: the polar axis is set to the **compass convention** (0° at
    North/top, angles increasing clockwise), matching the Cartesian trajectory
    panel's +y=up orientation. Because `preferred_angles` are math-convention
    (CCW from East) while `get_bearing` is already compass (CW from North),
    the former are converted via `π/2 - angle` and the latter passes through
    unchanged.

    Parameters
    ----------
    axcurr : matplotlib polar Axes (must have projection='polar').
    neuron : live HeadDirectionCells or VelocityCells instance.
    fr : (n,) firing rates at the timestep of interest.
    head_dir : (2,) unit vector OR
    head_bearing : float (radians, compass convention).
        Exactly one of these must be provided. If `head_bearing` is given it
        is used directly, otherwise `get_bearing(head_dir)` is called.
    min_height : float, default 0.1
        Minimum peak height for each cell's bump, so silent cells stay
        visible. Active cells with fr[j] > min_height plot at fr[j].
    """
    if head_bearing is None:
        head_bearing = ratinabox.utils.get_bearing(head_dir)  # radians

    preferred = np.asarray(neuron.preferred_angles)   # (n,) radians, math (CCW from E)
    tunings   = np.asarray(neuron.angular_tunings)    # (n,) radians
    # Compass convention: 0° at N (top), increasing clockwise — so the
    # polar panel visually aligns with the Cartesian trajectory plot
    # (up = +y = N).
    #   - preferred_angles are math-convention (CCW from E); convert
    #     to compass via  plot_angle = π/2 - angle.
    #   - head_bearing (from ratinabox.utils.get_bearing) is ALREADY
    #     compass-convention (CW from N); pass it through unchanged.
    axcurr.set_theta_zero_location("N")
    axcurr.set_theta_direction(-1)
    preferred_plot    = np.pi / 2 - preferred
    head_bearing_plot = head_bearing
    # Plot each cell as a von-Mises bump: peak at preferred_angle,
    # peak height = max(firing rate, min_height), angular width set by angular_tunings.
    theta_grid = np.linspace(-np.pi, np.pi, 360)
    for j in range(neuron.n):
        kappa = 1.0 / max(tunings[j]**2, 1e-6)
        height = max(min_height, fr[j])
        bump = height * np.exp(kappa * (np.cos(theta_grid - preferred_plot[j]) - 1))
        axcurr.plot(theta_grid, bump,
                        color='C0', linewidth=2)
    r_max = max(fr.max(), min_height)
    axcurr.plot([head_bearing_plot, head_bearing_plot],[0, r_max * 1.05],color="C1", linewidth=2)

def get_ax_lims(pos, Env, Sensory):
    """Compute x/y axis limits that show the environment plus all sensory
    ellipses for the trajectory range.

    Takes the max (tuning_distance + sigma_distance) across any vector-cell
    populations in `Sensory` as `reach` — the farthest any ellipse can
    extend from the agent. Then returns limits that union the environment
    extent with the trajectory's bounding box padded by `reach`.

    Parameters
    ----------
    pos : array of positions. Any shape with the last axis of length 2
        (e.g. (T, 2) for one episode, or (..., 2) more generally); the
        bounding box is computed over all dimensions except the last.
    Env : ratinabox Environment — used for `Env.extent = [xmin, xmax, ymin, ymax]`.
    Sensory : dict {name: Neurons} — scanned for vector-cell populations
        via their class name being in `CELL_VECTOR_CLASSES`.

    Returns
    -------
    (xmin, xmax), (ymin, ymax) : tuples of float suitable for
        `ax.set_xlim(...)` / `ax.set_ylim(...)`.
    """
    extent = np.asarray(Env.extent)
    reach = 0
    for neuron in Sensory.values():
        cls_name = type(neuron).__name__
        if cls_name in CELL_VECTOR_CLASSES:
            reachcurr = float(np.max(np.asarray(neuron.tuning_distances) + np.asarray(neuron.sigma_distances)))
            reach = max(reach, reachcurr)
            
    xmin = min(extent[0], np.min(pos[...,0]) - reach)
    xmax = max(extent[1], np.max(pos[...,0]) + reach)
    ymin = min(extent[2], np.min(pos[...,1]) - reach)
    ymax = max(extent[3], np.max(pos[...,1]) + reach)

    return (xmin,xmax), (ymin, ymax)

def visualize_sensory(track_curr, sensory_curr, t, Env, Sensory, fig=None, ax=None,
                      ax_xlim=None, ax_ylim=None):
    """Snapshot visualization at one timestep: trajectory + agent state +
    firing rates of each sensory population.

    Left panel: environment with trajectory up to time t, current agent
    position (red dot), heading (rotated triangle), and velocity (green
    arrow). Vector-cell populations (BVC/OVC/FieldOfView*) are overlaid
    on this panel using RatInABox's `Neurons.display_vector_cells`, which
    draws each cell as an ellipse at its preferred (d, θ) with fill alpha
    scaled by firing rate and size equal to the 1σ receptive-field extent.
    Multiple vector populations overlay on the same axis using each
    population's `.color` attribute.

    Right column: one polar subplot per non-vector population:
      - HeadDirectionCells: polar bar at preferred angles; red line = heading.
      - VelocityCells: same as HDC (bars scale by speed).
      - SpeedCell: horizontal bar, length = firing rate.
      - Any other class: fallback bar plot vs cell index.

    Because RatInABox's `display_vector_cells` reads from
    `neuron.Agent.history` and `neuron.history`, this function populates
    a single-timestep history on each vector-cell neuron before calling it.
    That mutates the live neurons' history dicts — if you also train/update
    these same Sensory objects, the history will be the one-timestep stub
    until the next `update()` appends to it.

    Parameters
    ----------
    track_curr : dict with 'pos' (T, 2), 'head_direction' (T, 2), and
        (optionally) 'vel' (T, 2).
    sensory_curr : dict {name: (T, n_cells)} from `compute_sensory`.
    t : int, timestep to visualize.
    Env : ratinabox Environment (live or rehydrated).
    Sensory : dict {name: Neurons} from `init_sensory`.
    fig : matplotlib Figure, optional. If None, a new figure is created.

    Returns
    -------
    fig : the matplotlib Figure.
    """
    # Accept either a dict {'pos','head_direction'[,'vel']} or an array of
    # shape (T, 3) where the last dim is (x, y, theta).
    if isinstance(track_curr, dict):
        pos      = np.asarray(track_curr["pos"])
        head_dir = np.asarray(track_curr["head_direction"])
        vel      = np.asarray(track_curr["vel"]) if "vel" in track_curr else None
    else:
        track_curr = np.asarray(track_curr)
        pos = track_curr[..., :2]
        head_dir = head_direction_from_orientation(track_curr[..., 2])   # theta is the orientation, fly convention
        vel = None

    # Split populations into "vector" (overlaid on trajectory) vs "other"
    # (separate right-column subplot).
    n_cell_types = len(Sensory)
    nax = 1 + n_cell_types  # trajectory + one subplot per population
    if ax is None:
        if fig is None:
            fig,ax1 = plt.subplots(1, nax, figsize=(5*nax,5))
        else:
            ax1 = fig.subplots(1, nax)
        ax = {}
        ax['traj'] = ax1[0]
        for i, name in enumerate(Sensory.keys()):
            ax[name] = ax1[i+1]
    else:
        if fig is None:
            fig = ax['traj'].figure

    if ax_xlim is None or ax_ylim is None:
        ax_xlim, ax_ylim = get_ax_lims(pos[t], Env, Sensory)

    def plot_env(axcurr):
        Env.plot_environment(fig=fig, ax=axcurr, autosave=False)
        axcurr.set_xlim(ax_xlim)
        axcurr.set_ylim(ax_ylim)
        axcurr.set_aspect('equal')
        
    # --- trajectory + agent state at t ---
    plot_env(ax['traj'])
    ax['traj'].scatter(pos[:t + 1, 0], pos[:t + 1, 1],
                    s=12, alpha=0.5, linewidth=0, c="C0", zorder=2)
    plot_agent_state(ax['traj'], pos, head_dir, vel, t)
    ax['traj'].set_title(f"t = {t}", fontsize=16)

    # not sure what this is for
    # Fake a single-timestep history so display_vector_cells has what
    # it needs. `t_id` inside will be 0.
    ag = next(iter(Sensory.values())).Agent
    ag.history["t"]              = [float(t)]
    ag.history["pos"]            = [pos[t].tolist()]
    ag.history["head_direction"] = [head_dir[t].tolist()]

    for name, neuron in Sensory.items():
        fr = np.asarray(sensory_curr[name])[t]    # (n_cells,)
        cls_name = type(neuron).__name__
        if cls_name in CELL_VECTOR_CLASSES:
            neuron.history["t"]          = [float(t)]
            neuron.history["firingrate"] = [fr.tolist()]
            plot_env(ax[name])
            plot_agent_state(ax[name], pos, head_dir, vel, t)
            neuron.display_vector_cells(fig=fig, ax=ax[name], t=float(t))
        elif cls_name in ("HeadDirectionCells", "VelocityCells"):
            if ax[name].name != "polar":
                axpos = ax[name].get_position()
                ax[name].remove()
                ax[name] = fig.add_axes(axpos, projection='polar')
            plot_head_direction(ax[name],neuron,fr,head_dir=head_dir[t])
        elif cls_name == "SpeedCell":
            ax[name].plot(np.asarray(sensory_curr[name]),'-',color='C0',lw=2)
            ax[name].plot(t, fr, 'o',color='C1',ms=16)
        
        ax[name].set_title(name, fontsize=16)

    fig.tight_layout()
