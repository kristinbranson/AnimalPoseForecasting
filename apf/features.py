import numpy as np

from apf.utils import rotate_2d_points, modrange, boxsum


# TODO: Comment these better and remove references to flies
def compute_global_velocity(Xorigin, Xtheta, tspred_global=[1, ]):
    """
    compute_global_velocity(Xorigin,Xtheta,tspred_global=[1,])
    compute the movement from t to t+tau for all tau in tspred_global
    Xorigin is the centroid position of the fly, shape = (2,T,nflies)
    Xtheta is the orientation of the fly, shape = (T,nflies)
    returns dXoriginrel,dtheta
    dXoriginrel[i,:,t,fly] is the global position for fly at time t+tspred_global[i] in the coordinate system of the fly at time t
    shape = (ntspred_global,2,T,nflies)
    dtheta[i,t,fly] is the total change in orientation for fly at time t+tspred_global[i] from time t. this sums per-frame dthetas,
    so it could have a value outside [-pi,pi]. shape = (ntspred_global,T,nflies)
    """

    T = Xorigin.shape[1]
    nflies = Xorigin.shape[2]
    ntspred_global = len(tspred_global)

    # global velocity
    # dXoriginrel[tau,:,t,fly] is the global position for fly at time t+tau in the coordinate system of the fly at time t
    dXoriginrel = np.zeros((ntspred_global, 2, T, nflies), dtype=Xorigin.dtype)
    dXoriginrel[:] = np.nan
    # dtheta[tau,t,fly] is the change in orientation for fly at time t+tau from time t
    dtheta = np.zeros((ntspred_global, T, nflies), dtype=Xtheta.dtype)
    dtheta[:] = np.nan
    # dtheta1[t,fly] is the change in orientation for fly from frame t to t+1
    dtheta1 = modrange(Xtheta[1:, :] - Xtheta[:-1, :], -np.pi, np.pi)

    for i, toff in enumerate(tspred_global):
        # center and rotate absolute position around position toff frames previous
        dXoriginrel[i, :, :-toff, :] = rotate_2d_points(
            (Xorigin[:, toff:, :] - Xorigin[:, :-toff, :]).transpose((1, 0, 2)), Xtheta[:-toff, :]).transpose((1, 0, 2))
        # compute total change in global orientation in toff frame intervals
        dtheta[i, :-toff, :] = boxsum(dtheta1[None, ...], toff)

    return dXoriginrel, dtheta


def compute_relpose_velocity(relpose, tspred_dct=[], is_angle=None):
    """
    compute_relpose_velocity(relpose,tspred_dct=[])
    compute the relative pose movement from t to t+tau for all tau in tspred_dct
    relpose is the relative pose features, shape = (nrelative,T,nflies)
    outputs drelpose, shape = (nrelative,T,ntspred_dct+1,nflies)
    """
    ntspred_dct = len(tspred_dct)
    T = relpose.shape[1]
    nflies = relpose.shape[2]
    nrelative=relpose.shape[0]

    # drelpose1[:,f,fly] is the change in pose for fly from frame t to t+1
    drelpose1 = relpose[:, 1:, :] - relpose[:, :-1, :]
    if is_angle is not None:
        drelpose1[is_angle, :, :] = modrange(drelpose1[is_angle, :, :], -np.pi, np.pi)

    # drelpose[:,tau,t,fly] is the change in pose for fly at time t+tau from time t
    drelpose = np.zeros((nrelative, T, ntspred_dct + 1, nflies), dtype=relpose.dtype)
    drelpose[:] = np.nan
    drelpose[:, :-1, 0, :] = drelpose1

    for i, toff in enumerate(tspred_dct):
        # compute total change in relative pose in toff frame intervals
        drelpose[:, :-toff, i + 1, :] = boxsum(drelpose1, toff)

    return drelpose
