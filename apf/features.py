import numpy as np

from apf.utils import rotate_2d_points, modrange, boxsum


def compute_global_velocity(
        Xorigin: np.ndarray,
        Xtheta: np.ndarray,
        tspred_global: tuple = (1, )
) -> tuple[np.ndarray, np.ndarray]:
    """ Compute the movement (delta forward, delta side, delta ori) from t to t+dt for all dt in tspred_global .

    Args:
        Xorigin: Position of agent in global reference frame (2, n_frames, n_agents) float
        Xtheta: Orientation of agent in global reference frame (n_frames, n_agents) float
        tspred_global: A list of dt to predict into the future

    Returns:
        dXoriginrel: (dforward, dside) in agent's frame of reference at time t, (n_dt, 2, n_frames, n_agents) float
        dtheta: Change in orientation (n_dt, n_frames, n_agents) float
    """
    T = Xorigin.shape[1]
    nflies = Xorigin.shape[2]
    ntspred_global = len(tspred_global)

    # dXoriginrel[i, :, t, agent] is the global position for agent at time t+tspred_global[i] in the coordinate
    #   system of the agent at time t.
    dXoriginrel = np.zeros((ntspred_global, 2, T, nflies), dtype=Xorigin.dtype)
    dXoriginrel[:] = np.nan
    # dtheta[i, t, agent] is the total change in orientation for agent at time t+tspred_global[i] from time t.
    #   This sums per-frame dthetas, so it could have a value outside [-pi, pi].
    dtheta = np.zeros((ntspred_global, T, nflies), dtype=Xtheta.dtype)
    dtheta[:] = np.nan
    # dtheta1[t, agent] is the change in orientation for agent from frame t to t+1
    dtheta1 = modrange(Xtheta[1:, :] - Xtheta[:-1, :], -np.pi, np.pi)

    for i, toff in enumerate(tspred_global):
        # center and rotate absolute position around position toff frames previous
        dXoriginrel[i, :, :-toff, :] = rotate_2d_points(
            (Xorigin[:, toff:, :] - Xorigin[:, :-toff, :]).transpose((1, 0, 2)), Xtheta[:-toff, :]).transpose((1, 0, 2))
        # compute total change in global orientation in toff frame intervals
        dtheta[i, :-toff, :] = boxsum(dtheta1[None, ...], toff)

    return dXoriginrel, dtheta


def compute_relpose_velocity(
        relpose: np.ndarray,
        tspred_dct: tuple = (),
        is_angle: np.ndarray | None = None
) -> np.ndarray:
    """ Computes the relative pose movement from t to t+dt for all dt in tspred_dct

    Args:
        relpose: Relative pose features, (n_features, n_frames, n_agents) float

    Returns:
        drelpose: (n_features, n_frames, ntspred_dct+1, n_agents) float
    """
    ntspred_dct = len(tspred_dct)
    T = relpose.shape[1]
    nflies = relpose.shape[2]
    nrelative=relpose.shape[0]

    # drelpose1[:, t, agent] is the change in pose for agent from frame t to t+1
    drelpose1 = relpose[:, 1:, :] - relpose[:, :-1, :]
    if is_angle is not None:
        drelpose1[is_angle, :, :] = modrange(drelpose1[is_angle, :, :], -np.pi, np.pi)

    # drelpose[:, dt, t, agent] is the change in pose for agent at time t+dt from time t
    drelpose = np.zeros((nrelative, T, ntspred_dct + 1, nflies), dtype=relpose.dtype)
    drelpose[:] = np.nan
    drelpose[:, :-1, 0, :] = drelpose1

    for i, toff in enumerate(tspred_dct):
        # compute total change in relative pose in toff frame intervals
        drelpose[:, :-toff, i + 1, :] = boxsum(drelpose1, toff)

    return drelpose
