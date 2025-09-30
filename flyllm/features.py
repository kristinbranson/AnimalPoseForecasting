import numpy as np
import warnings
import collections
import torch
import tqdm
import logging
from scipy.stats import circmean, circstd

from apf.utils import modrange, rotate_2d_points, angledist2xy, mod2pi, atleast_4d, compute_npad
from apf.features import compute_global_velocity, compute_relpose_velocity
from flyllm.config import (
    posenames, keypointnames, scalenames,
    nglobal, nrelative, nfeatures, nkptouch, nkptouch_other,
    featangle, featrelative,
    featglobal, kpvision_other, kptouch_other, featthetaglobal, kpeye, kptouch,
    SENSORY_PARAMS, PXPERMM
)

LOG = logging.getLogger(__name__)


""" Pose features
"""


def relfeatidx_to_featidx(relfeatidx):
    return relfeatidx + nglobal


def featidx_to_relfeatidx(featidx):
    return featidx - nglobal


def relfeatidx_to_cossinidx(discreteidx=[]):
    """
    relfeatidx_to_cossinidx(discreteidx=[])
    get the look up table for relative feature index to the cos sin representation index
    discreteidx: list of int, feature indices that are discrete
    returns rel2cossinmap: list, rel2cossinmap[i] is the list of indices for the cos and sin
    representation of the i-th relative feature
    """
    rel2cossinmap = []
    csi = 0
    for relfeati in range(nrelative):
        feati = relfeatidx_to_featidx(relfeati)
        if featangle[featrelative][relfeati] and (feati not in discreteidx):
            rel2cossinmap.append(np.array([csi, csi + 1]))
            csi += 2
        else:
            rel2cossinmap.append(csi)
            csi += 1
    return rel2cossinmap, csi


def relpose_cos_sin_to_angle(relpose_cos_sin, discreteidx=[], epsilon=1e-6):
    sz = relpose_cos_sin.shape[:-1]
    if len(sz) == 0:
        n = 1
    else:
        n = np.prod(sz)
    relpose_cos_sin = relpose_cos_sin.reshape((n, relpose_cos_sin.shape[-1]))
    rel2cossinmap, ncs = relfeatidx_to_cossinidx(discreteidx)

    relpose = np.zeros((n, nrelative), dtype=relpose_cos_sin.dtype)
    for relfeati in range(nrelative):
        csi = rel2cossinmap[relfeati]
        if type(csi) is int:
            relpose[..., relfeati] = relpose_cos_sin[..., csi]
        else:
            # if the norm is less than epsilon, just make the angle 0
            idxgood = np.linalg.norm(relpose_cos_sin[:, csi], axis=-1) >= epsilon
            relpose[idxgood, relfeati] = np.arctan2(relpose_cos_sin[idxgood, csi[1]], relpose_cos_sin[idxgood, csi[0]])

    relpose = relpose.reshape(sz + (nrelative,))
    return relpose


def relpose_angle_to_cos_sin(relpose, discreteidx=[]):
    """
    relpose_angle_to_cos_sin(relposein)
    convert the relative pose angles features from radians to cos and sin
    relposein: shape = (nrelative,...)
    """

    rel2cossinmap, ncs = relfeatidx_to_cossinidx(discreteidx)

    relpose_cos_sin = np.zeros((ncs,) + relpose.shape[1:], dtype=relpose.dtype)
    for relfeati in range(nrelative):
        csi = rel2cossinmap[relfeati]
        if type(csi) is int:
            relpose_cos_sin[csi, ...] = relpose[relfeati, ...]
        else:
            relpose_cos_sin[csi[0], ...] = np.cos(relpose[relfeati, ...])
            relpose_cos_sin[csi[1], ...] = np.sin(relpose[relfeati, ...])
    return relpose_cos_sin


def compute_relpose_tspred(relposein, tspred_dct=[], discreteidx=[]):
    """
    compute_relpose_tspred(relpose,tspred_dct=[])
    concatenate the relative pose at t+tau for all tau in tspred_dct
    relposein: shape = (nrelative,T,nflies)
    tspred_dct: list of int
    returns relpose_tspred: shape = (nrelative,T,ntspred_dct+1,nflies)
    """

    ntspred_dct = len(tspred_dct)
    T = relposein.shape[1]
    nflies = relposein.shape[2]
    relpose = relpose_angle_to_cos_sin(relposein, discreteidx=discreteidx)
    nrelrep = relpose.shape[0]

    # predict next frame pose
    relpose_tspred = np.zeros((nrelrep, T, ntspred_dct + 1, nflies), dtype=relpose.dtype)
    relpose_tspred[:] = np.nan
    relpose_tspred[:, :-1, 0, :] = relpose[:, 1:, :]
    for i, toff in enumerate(tspred_dct):
        relpose_tspred[:, :-toff, i + 1, :] = relpose[:, toff:, :]

    return relpose_tspred


def compute_scale_perfly(Xkp: np.ndarray) -> np.ndarray:
    """Computes mean and std of the following keypoint based measures per fly:
        thorax width, thorax length, abdomen length, wing length, head width, head height

    Args:
        Xkp: n_keypoints x 2 x T [x n_flies]

    Returns:
        scale_perfly: n_scales x n_total_flies
    """
    if np.ndim(Xkp) >= 4:
        n_flies = Xkp.shape[3]
    else:
        n_flies = 1

    scales = {}

    # Thorax width
    rfthorax = Xkp[keypointnames.index('right_front_thorax')]
    lfthorax = Xkp[keypointnames.index('left_front_thorax')]
    scales['thorax_width'] = np.sqrt(np.sum((rfthorax - lfthorax) ** 2., axis=0))

    # Thorax length
    fthorax = (rfthorax + lfthorax) / 2.
    bthorax = Xkp[keypointnames.index('base_thorax')]
    scales['thorax_length'] = np.sqrt(np.sum((fthorax - bthorax) ** 2, axis=0))

    # Abdomen length
    abdomen = Xkp[keypointnames.index('tip_abdomen')]
    scales['abdomen_length'] = np.sqrt(np.sum((bthorax - abdomen) ** 2., axis=0))

    # Wing length
    lwing = Xkp[keypointnames.index('wing_left')]
    rwing = Xkp[keypointnames.index('wing_right')]
    midthorax = (fthorax + bthorax) / 2.
    wing_length_lr = np.concatenate(((lwing - midthorax) ** 2., (rwing - midthorax) ** 2.), axis=1)
    scales['wing_length'] = np.sqrt(np.sum(wing_length_lr, axis=0))

    # Head width
    reye = Xkp[keypointnames.index('right_eye')]
    leye = Xkp[keypointnames.index('left_eye')]
    scales['head_width'] = np.sqrt(np.sum((reye - leye) ** 2., axis=0))

    # Head height
    eye = (leye + reye) / 2.
    ant = Xkp[keypointnames.index('antennae_midpoint')]
    scales['head_height'] = np.sqrt(np.sum((eye - ant) ** 2., axis=0))

    # Move to a numpy array with indices corresponding to the order of scalenames
    scale_perfly = np.zeros((len(scalenames), n_flies))
    for scale_name, scale_value in scales.items():
        scale_perfly[scalenames.index(scale_name)] = np.nanmedian(scale_value, axis=0)

        # TODO: In the previous code the head features were not using axis=0 for median and std, do we want that?
        #   lets keep it perfly for now until we see why this might have been

    return scale_perfly


def body_centric_kp(Xkp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centers keypoints on mean point of "shoulders" and rotates them so that thorax points "up".

    Args:
        Xkp: n_keypoints x 2 [x T [x n_flies]]

    Returns:
        Xn: n_keypoints x 2 x T x n_flies, keypoints transformed to be in the fly's reference frame
        fthorax: Center of the fly's front thorax
        thorax_theta: Orientation of the fly's thorax
    """
    # Reshape Xkp to be of size n_keypoints x 2 x T x n_flies
    Xkp = atleast_4d(Xkp)

    bthorax = Xkp[keypointnames.index('base_thorax')]
    lthorax = Xkp[keypointnames.index('left_front_thorax')]
    rthorax = Xkp[keypointnames.index('right_front_thorax')]
    fthorax = (lthorax + rthorax) / 2.
    vec = fthorax - bthorax
    thorax_theta = mod2pi(np.arctan2(vec[1, ...], vec[0, ...]) - np.pi / 2)
    Xn = rotate_2d_points(Xkp - fthorax[np.newaxis, ...], thorax_theta)

    return Xn, fthorax, thorax_theta


def kp2feat(
        Xkp: np.ndarray,
        scale_perfly: np.ndarray | None = None,
        flyid: np.ndarray | None = None,
        return_scale: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes pose features from keypoints on a fly's body.

    Args:
        Xkp: n_keypoints x 2 [x T [x n_flies]]
        scale_perfly: n_scales x n_total_flies
        flyid: n_flies
        return_scale: If True, returns the scale_perfly and flyid along with the feature array.

    Returns:
        Xfeat: n_features x T x n_flies
        [scale_perfly]: n_scales x n_flies
        [flyid]: n_flies
    """
    # Reshape Xkp to be of size n_keypoints x 2 x T x n_flies
    Xkp = atleast_4d(Xkp)
    _, _, T, n_flies = Xkp.shape

    # Ensure that if scale_perfly is given, so is flyid
    assert (flyid is None) or scale_perfly is not None, f"{flyid} --> {scale_perfly} is False"

    # If scale is not given, compute it
    if scale_perfly is None:
        scale_perfly = compute_scale_perfly(Xkp)
        flyid = np.tile(np.arange(n_flies, dtype=int)[np.newaxis, :], (T, 1))

    # Rotate keypoints to be in fly's frame of reference
    Xn, fthorax, thorax_theta = body_centric_kp(Xkp)

    feat = {}

    # Thorax
    feat['thorax_front_x'] = fthorax[0]
    feat['thorax_front_y'] = fthorax[1]
    feat['orientation'] = thorax_theta

    # Abdomen
    # thorax_length can be a scalar or an array of size T x n_flies
    thorax_length = scale_perfly[scalenames.index('thorax_length'), flyid]
    if np.isscalar(thorax_length) or thorax_length.size == 1:
        pass
    elif thorax_length.size == n_flies:
        thorax_length = thorax_length.reshape((1, n_flies))
    elif thorax_length.size == T * n_flies:
        thorax_length = thorax_length.reshape((T, n_flies))
    else:
        raise ValueError(f'thorax_length size {thorax_length.size} is unexpected')
    pthoraxbase = np.zeros((2, T, n_flies))
    pthoraxbase[1] = -thorax_length
    # TODO: Why do we use the average length here? Can we do without scale_perfly as input to this function?
    vec = Xn[keypointnames.index('tip_abdomen')] - pthoraxbase
    feat['abdomen_angle'] = mod2pi(np.arctan2(vec[1], vec[0]) + np.pi / 2)

    # Head
    # represent with the midpoint of the eyes and the angle from here to the antenna
    mid_ant = Xn[keypointnames.index('antennae_midpoint')]
    mid_eye = (Xn[keypointnames.index('right_eye')] + Xn[keypointnames.index('left_eye')]) / 2
    feat['head_base_x'] = mid_eye[0]
    feat['head_base_y'] = mid_eye[1]
    feat['head_angle'] = np.arctan2(mid_ant[1] - mid_eye[1], mid_ant[0] - mid_eye[0]) - np.pi / 2

    # Front leg
    # parameterize the front leg tips based on angle and distance from origin (shoulder mid-point)
    vec = Xn[keypointnames.index('left_front_leg_tip')]
    feat['left_front_leg_tip_dist'] = np.linalg.norm(vec, axis=0)
    feat['left_front_leg_tip_angle'] = mod2pi(np.pi - np.arctan2(vec[1, :], vec[0, :]))
    vec = Xn[keypointnames.index('right_front_leg_tip')]
    feat['right_front_leg_tip_dist'] = np.linalg.norm(vec, axis=0)
    feat['right_front_leg_tip_angle'] = np.arctan2(vec[1, :], vec[0, :])

    # Middle femur base
    # compute angle around and distance from halfway between the thorax base and thorax front
    pmidthorax = np.zeros((2, T, n_flies))
    pmidthorax[1] = -thorax_length / 2
    vec = Xn[keypointnames.index('left_middle_femur_base')] - pmidthorax
    feat['left_middle_femur_base_dist'] = np.linalg.norm(vec, axis=0)
    left_middle_femur_base_angle = mod2pi(np.pi - np.arctan2(vec[1, :], vec[0, :]))
    feat['left_middle_femur_base_angle'] = left_middle_femur_base_angle
    vec = Xn[keypointnames.index('right_middle_femur_base')] - pmidthorax
    feat['right_middle_femur_base_dist'] = np.linalg.norm(vec, axis=0)
    right_middle_femur_base_angle = np.arctan2(vec[1, :], vec[0, :])
    feat['right_middle_femur_base_angle'] = right_middle_femur_base_angle

    # Middle femur tibia
    # represented as distance from and angle around femur base
    vec = Xn[keypointnames.index('left_middle_femur_tibia_joint')] - Xn[keypointnames.index('left_middle_femur_base')]
    feat['left_middle_femur_tibia_joint_dist'] = np.linalg.norm(vec, axis=0)
    left_angle = mod2pi(np.pi - np.arctan2(vec[1, :], vec[0, :]))
    feat['left_middle_femur_tibia_joint_angle'] = mod2pi(left_angle - left_middle_femur_base_angle)

    vec = Xn[keypointnames.index('right_middle_femur_tibia_joint')] - Xn[keypointnames.index('right_middle_femur_base')]
    feat['right_middle_femur_tibia_joint_dist'] = np.linalg.norm(vec, axis=0)
    right_angle = np.arctan2(vec[1, :], vec[0, :])
    feat['right_middle_femur_tibia_joint_angle'] = mod2pi(right_angle - right_middle_femur_base_angle)
    # TODO: when subtracting angles, is that safe around range boundary?
    #   e.g. if one angle is pi and the other -pi, subtracting one from the other gives us 2pi instead of 0

    # Middle leg tip
    # represented as distance from and angle around femur tibia joint
    vec = Xn[keypointnames.index('left_middle_leg_tip')] - Xn[keypointnames.index('left_middle_femur_tibia_joint')]
    feat['left_middle_leg_tip_dist'] = np.sqrt(np.sum(vec ** 2, axis=0))
    feat['left_middle_leg_tip_angle'] = mod2pi(np.pi - np.arctan2(vec[1, :], vec[0, :]) - left_angle)
    vec = Xn[keypointnames.index('right_middle_leg_tip')] - Xn[keypointnames.index('right_middle_femur_tibia_joint')]
    feat['right_middle_leg_tip_dist'] = np.linalg.norm(vec, axis=0)
    feat['right_middle_leg_tip_angle'] = mod2pi(np.arctan2(vec[1, :], vec[0, :]) - right_angle)

    # Back leg
    # use the thorax base as the origin
    vec = Xn[keypointnames.index('left_back_leg_tip')] - pthoraxbase
    feat['left_back_leg_tip_dist'] = np.linalg.norm(vec, axis=0)
    feat['left_back_leg_tip_angle'] = mod2pi(np.pi - np.arctan2(vec[1, :], vec[0, :]))
    vec = Xn[keypointnames.index('right_back_leg_tip')] - pthoraxbase
    feat['right_back_leg_tip_dist'] = np.linalg.norm(vec, axis=0)
    feat['right_back_leg_tip_angle'] = np.arctan2(vec[1, :], vec[0, :])

    # Wings
    # relative to thorax middle
    vec = Xn[keypointnames.index('wing_left')] - pmidthorax
    feat['left_wing_angle'] = mod2pi(-np.pi + np.arctan2(vec[1, :], vec[0, :]))
    vec = Xn[keypointnames.index('wing_right')] - pmidthorax
    feat['right_wing_angle'] = -np.arctan2(vec[1, :], vec[0, :])
    # TODO: Should all features with _angle go through modrange?
    #   test1: add mod2pi after all angle features, result is different
    #   test2: skip mod2pi on intermediate features, result is different
    #   Should we just do it everywhere to be sure?

    # Move to a numpy array with indices corresponding to the order of posenames
    Xfeat = np.zeros((len(posenames), T, n_flies))
    for feat_name, feat_value in feat.items():
        Xfeat[posenames.index(feat_name)] = feat_value

    if return_scale:
        return Xfeat, scale_perfly, flyid
    else:
        return Xfeat


def compute_pose_features(X, scale):
    posefeat = kp2feat(X, scale)
    relpose = posefeat[featrelative, ...]
    globalpos = posefeat[featglobal, ...]

    return relpose, globalpos


def combine_relative_global(Xrelative, Xglobal, axis=-1):
    X = np.concatenate((Xglobal, Xrelative), axis=axis)
    return X


def combine_relative_global_pose(relpose, globalpos):
    sz = (nfeatures,) + relpose.shape[1:]
    posefeat = np.zeros(sz, dtype=relpose.dtype)
    posefeat[featrelative, ...] = relpose
    posefeat[featglobal, ...] = globalpos
    return posefeat


""" Compute movement
"""


def ravel_label_index(ftidx, dct_m=None, tspred_global=[1, ], nrelrep=None, d_output=None, ntspred_relative=1):
    ftidx = np.array(ftidx)
    sz = ftidx.shape
    assert sz[-1] == 2
    ftidx = ftidx.reshape((-1, 2))

    idx = np.zeros(ftidx.shape[:-1], dtype=int)

    if dct_m is not None:
        ntspred_relative = dct_m.shape[0] + 1
    offrelative = len(tspred_global) * nglobal
    if nrelrep is None:
        if d_output is None:
            nrelrep = nrelative
        else:
            nrelrep = d_output - offrelative

    for i, ft in enumerate(ftidx):
        fidx = ft[0]
        t = ft[1]
        isglobal = fidx < nglobal
        if isglobal:
            # t = 1 corresponds to next frame
            tidx = np.nonzero(tspred_global == t)[0][0]
            assert tidx is not None
            idx[i] = np.ravel_multi_index((tidx, fidx), (len(tspred_global), nglobal))
        else:
            # t = 1 corresponds to next frame
            idx[i] = np.ravel_multi_index((t - 1, fidx - nglobal), (ntspred_relative, nrelrep)) + offrelative

    return idx.reshape(sz[:-1])


def unravel_label_index(
        idx,
        dct_m=None,
        tspred_global=[1, ],
        nrelrep=None,
        d_output=None,
        ntspred_relative=1
):
    idx = np.array(idx)
    sz = idx.shape
    idx = idx.flatten()
    if dct_m is not None:
        ntspred_relative = dct_m.shape[0] + 1
    offrelative = len(tspred_global) * nglobal
    if nrelrep is None:
        if d_output is None:
            nrelrep = nrelative
        else:
            nrelrep = d_output - offrelative
    ftidx = np.zeros((len(idx), 2), dtype=int)
    for ii, i in enumerate(idx):
        if i < offrelative:
            tidx, fidx = np.unravel_index(i, (len(tspred_global), nglobal))
            ftidx[ii] = (fidx, tspred_global[tidx])
        else:
            t, fidx = np.unravel_index(i - offrelative, (ntspred_relative, nrelrep))
            # t = 1 corresponds to next frame
            ftidx[ii] = (fidx + nglobal, t + 1)
    return ftidx.reshape(sz + (2,))


def compute_movement(
        X=None,
        scale=None,
        relpose=None,
        globalpos=None,
        simplify=None,
        dct_m=None,
        tspred_global=[1, ],
        compute_pose_vel=True,
        discreteidx=[],
        returnidx=False,
        debug=False,
):
    """
    movement = compute_movement(X=X,scale=scale,...)
    movement = compute_movement(relpose=relpose,globalpos=globalpos,...)

    Args:
        X (ndarray, nkpts x 2 x T x nflies, optional): Keypoints. Can be None only if relpose and globalpos are input. Defaults to None. T>=2
        scale (ndarray, nscale x nflies): Scaling parameters related to an individual fly. Can be None only if relpose and globalpos are input. Defaults to None.
        relpose (ndarray, nrelative x T x nflies or nrelative x T, optional): Relative pose features. T>=2
        If input, X and scale are ignored. Defaults to None.
        globalpos (ndarray, nglobal x T x nflies or nglobal x T, optional): Global position. If input, X and scale are ignored. Defaults to None. T>=2
        simplify (string or None, optional): Whether/how to simplify the output. Defaults to None for no simplification.
    Optional args:
        dct_m (ndarray, nrelative x ntspred_dct+1 x nflies): DCT matrix for pose features. Defaults to None.
        tspred_global (list of int, optional): Time steps to predict for global features. Defaults to [1,].

    Returns:
        movement (ndarray, d_output x T-1 x nflies): Per-frame movement. movement[:,t,i] is the movement from frame
        t for fly i.
        init (ndarray, npose x ninit=2 x nflies): Initial pose for each fly
    """

    if relpose is None or globalpos is None:
        relpose, globalpos = compute_pose_features(X, scale)

    nd = relpose.ndim
    assert (nd == 2 or nd == 3)
    if nd < 3:
        relpose = relpose[..., None]
        globalpos = globalpos[..., None]
    T = relpose.shape[1]
    nflies = relpose.shape[2]

    # centroid and orientation position
    Xorigin = globalpos[:2, ...]
    Xtheta = globalpos[2, ...]

    # which future frames are we predicting, how many features are there total
    ntspred_global = len(tspred_global)
    if (dct_m is not None) and simplify != 'global':
        ntspred_dct = dct_m.shape[0]
        tspred_dct = np.arange(1, ntspred_dct + 1)
        tspred_relative = tspred_dct
    else:
        ntspred_dct = 0
        tspred_dct = []
        tspred_relative = [1, ]

    # compute the max of tspred_global and tspred_dct
    tspred_all = np.unique(np.concatenate((tspred_global, tspred_relative)))
    lastT = T - np.max(tspred_all)

    # global velocity
    # dXoriginrel[tau,:,t,fly] is the global position for fly at time t+tau in the coordinate system of the fly at time t
    dXoriginrel, dtheta = compute_global_velocity(Xorigin, Xtheta, tspred_global)

    ninit = 2
    # relpose_rep is (nrelrep,T,ntspred_dct+1,nflies)
    if simplify == 'global':
        relpose_rep = np.zeros((0, T, ntspred_global + 1, nflies), dtype=relpose.dtype)
        relpose_init = np.zeros(0, ninit, nflies, dtype=relpose.dtype)
    elif compute_pose_vel:
        relpose_rep = compute_relpose_velocity(relpose, tspred_dct, is_angle=featangle[featrelative])
        relpose_init = relpose[:,:ninit]
    else:
        relpose_rep = compute_relpose_tspred(relpose, tspred_dct, discreteidx=discreteidx)
        relpose_init = relpose[:,:ninit]

    nrelrep = relpose_rep.shape[0]

    init = np.r_[globalpos[:, :ninit],relpose_init]

    if debug:
        # try to reconstruct xorigin, xtheta from dxoriginrel and dtheta
        xtheta0 = Xtheta[0]
        xorigin0 = Xorigin[:, 0]
        thetavel = dtheta[0, :]
        xtheta = np.cumsum(np.concatenate((xtheta0[None], thetavel), axis=0), axis=0)
        xoriginvelrel = dXoriginrel[0]
        xoriginvel = rotate_2d_points(xoriginvelrel.reshape((2, -1)).T, -xtheta[:-1].flatten()).T.reshape(
            xoriginvelrel.shape)
        xorigin = np.cumsum(np.concatenate((xorigin0[:, None], xoriginvel), axis=1), axis=1)
        LOG.debug('xtheta0 = %s' % str(xtheta0[0]))
        LOG.debug('xorigin0 = %s' % str(xorigin0[:, 0]))
        LOG.debug('xtheta[:5] = %s' % str(xtheta[:5, 0]))
        LOG.debug('original Xtheta[:5] = %s' % str(Xtheta[:5, 0]))
        LOG.debug('xoriginvelrel[:5] = \n%s' % str(xoriginvelrel[:, :5, 0]))
        LOG.debug('xoriginvel[:5] = \n%s' % str(xoriginvel[:, :5, 0]))
        LOG.debug('xorigin[:5] = \n%s' % str(xorigin[:, :5, 0]))
        LOG.debug('original Xorigin[:5] = \n%s' % str(Xorigin[:, :5, 0]))
        LOG.debug('max error origin: %e' % np.max(np.abs(xorigin[:, :-1] - Xorigin)))
        LOG.debug('max error theta: %e' % np.max(np.abs(modrange(xtheta[:-1] - Xtheta, -np.pi, np.pi))))

    # only full data up to frame lastT
    # dXoriginrel is (ntspred_global,2,lastT,nflies)
    dXoriginrel = dXoriginrel[:, :, :lastT, :]
    # dtheta is (ntspred_global,lastT,nflies)
    dtheta = dtheta[:, :lastT, :]
    # relpose_rep is (nrelrep,lastT,ntspred_dct+1,nflies)
    relpose_rep = relpose_rep[:, :lastT]

    if (simplify != 'global') and (dct_m is not None):
        # the pose forecasting papers compute error on the actual pose, not the dct. they just force the network to go through the dct
        # representation first.
        relpose_rep[:, :, 1:, :] = dct_m @ relpose_rep[:, :, 1:, :]
    # relpose_rep is now (ntspred_dct+1,nrelrep,lastT,nflies)
    relpose_rep = np.moveaxis(relpose_rep, 2, 0)

    if debug and dct_m is not None:
        idct_m = np.linalg.inv(dct_m)
        relpose_rep_dct = relpose_rep[1:].reshape((ntspred_dct, -1))
        relpose_rep_idct = idct_m @ relpose_rep_dct
        relpose_rep_idct = relpose_rep_idct.reshape((ntspred_dct,) + relpose_rep.shape[1:])
        err_dct_0 = np.max(np.abs(relpose_rep_idct[0] - relpose_rep[0]))
        LOG.debug('max error dct_0: %e' % err_dct_0)
        err_dct_tau = np.max(
            np.abs(relpose_rep[0, :, ntspred_dct - 1:, :] - relpose_rep_idct[-1, :, :-ntspred_dct + 1, :]))
        LOG.debug('max error dct_tau: %e' % err_dct_tau)

    # concatenate the global (dforward, dsideways, dorientation)
    movement_global = np.concatenate((dXoriginrel[:, [1, 0]], dtheta[:, None, :, :]), axis=1)
    # movement_global is now (ntspred_global*nglobal,lastT,nflies)
    movement_global = movement_global.reshape((ntspred_global * nglobal, lastT, nflies))
    # relpose_rep is now ((ntspred_dct+1)*nrelrep,lastT,nflies)
    relpose_rep = relpose_rep.reshape(((ntspred_dct + 1) * nrelrep, lastT, nflies))

    if nd == 2:  # no flies dimension
        movement_global = movement_global[..., 0]
        relpose_rep = relpose_rep[..., 0]
        init = init[..., 0]

    # concatenate everything together
    movement = np.concatenate((movement_global, relpose_rep), axis=0)

    if returnidx:
        idxinfo = {}
        idxinfo['global'] = [0, ntspred_global * nglobal]
        idxinfo['global_feat_tau'] = unravel_label_index(np.arange(ntspred_global * nglobal),
                                                         dct_m=dct_m, tspred_global=tspred_global,
                                                         nrelrep=nrelrep)
        idxinfo['relative'] = [ntspred_global * nglobal, ntspred_global * nglobal + nrelrep * (ntspred_dct + 1)]
        return movement, init, idxinfo

    return movement, init

def integrate_global_movement(globalvel,init_pose):

    globalpos0 = init_pose[featglobal]

    xorigin0 = globalpos0[:2]
    xtheta0 = globalpos0[2]

    thetavel = globalvel[2]
    xtheta = np.cumsum(np.concatenate((xtheta0, thetavel), axis=0), axis=0)

    xoriginvelrel = globalvel[[1,0]]  # forward=y then sideways=x
    xoriginvel = rotate_2d_points(xoriginvelrel.T, -xtheta[:-1])
    xorigin = np.cumsum(np.concatenate((xorigin0, xoriginvel.T), axis=1), axis=1)

    globalpos = np.concatenate((xorigin, xtheta[None,:]), axis=0)
    return globalpos


"""Compute sensory features
"""


def compute_sensory(xeye_main, yeye_main, theta_main,
                    xtouch_main, ytouch_main,
                    xvision_other, yvision_other,
                    xtouch_other, ytouch_other):
    """ Compute sensory input.

    inputs:
    xeye_main: x-coordinate of main fly's position for vision. shape = (T).
    yeye_main: y-coordinate of main fly's position for vision. shape = (T).
    theta_main: orientation of main fly. shape = (T).
    xtouch_main: x-coordinates of main fly's keypoints for computing touch inputs (both wall and other fly). shape = (npts_touch,T).
    ytouch_main: y-coordinates of main fly's keypoints for computing touch inputs (both wall and other fly). shape = (npts_touch,T).
    xvision_other: x-coordinate of keypoints on the other flies for computing vision input. shape = (npts_vision,T,nflies)
    yvision_other: y-coordinate of keypoints on the other flies for computing vision input. shape = (npts_vision,T,nflies)
    xtouch_other: x-coordinate of keypoints on the other flies for computing touch input. shape = (npts_touch_other,T,nflies)
    ytouch_other: y-coordinate of keypoints on the other flies for computing touch input. shape = (npts_touch_other,T,nflies)

    outputs:
    otherflies_vision: appearance of other flies to input fly. this is computed as a
    1. - np.minimum(1.,SENSORY_PARAMS['otherflies_vision_mult'] * dist**SENSORY_PARAMS['otherflies_vision_exp'])
    where dist is the minimum distance to some point on some other fly x,y_vision_other in each of n_oma directions.
    shape = (SENSORY_PARAMS['n_oma'],T).
    wall_touch: height of arena chamber at each keypoint in x,y_touch_main. this is computed as
    np.minimum(SENSORY_PARAMS['arena_height'],np.maximum(0.,SENSORY_PARAMS['arena_height'] -
    (distleg-SENSORY_PARAMS['inner_arena_radius'])*SENSORY_PARAMS['arena_height']/(SENSORY_PARAMS['outer_arena_radius']-
    SENSORY_PARAMS['inner_arena_radius'])))
    shape = (npts_touch,T),
    otherflies_touch: information about touch from other flies to input fly. this is computed as
    1. - np.minimum(1.,SENSORY_PARAMS['otherflies_touch_mult'] * dist**SENSORY_PARAMS['otherflies_touch_exp'])
    where dist is the minimum distance over all other flies from each keypoint in x,y_touch_main to each keypoint in x,y_touch_other
    there are two main difference between this and otherflies_vision. first is this uses multiple keypoints on the main and other flies
    and has an output for each of them. conversely, otherflies_vision has an output for each direction. the second difference is
    based on the parameters in SENSORY_PARAMS. The parameters for touch should be set so that the maximum distance over which there is a
    signal is about how far any keypoint can be from any of the keypoints in x,y_touch_other, which the maximum distance for
    vision is over the entire arena.
    shape = (npts_touch*npts_touch_other,T).
    """
    # increase dimensions if only one frame input
    if xvision_other.ndim < 3:
        T = 1
    else:
        T = xvision_other.shape[2]

    npts_touch = xtouch_main.shape[0]
    npts_vision = xvision_other.shape[0]
    npts_touch_other = xtouch_other.shape[0]
    nflies = xvision_other.shape[1]

    xvision_other = np.reshape(xvision_other, (npts_vision, nflies, T))
    yvision_other = np.reshape(yvision_other, (npts_vision, nflies, T))
    xtouch_other = np.reshape(xtouch_other, (npts_touch_other, nflies, T))
    ytouch_other = np.reshape(ytouch_other, (npts_touch_other, nflies, T))
    xeye_main = np.reshape(xeye_main, (1, 1, T))
    yeye_main = np.reshape(yeye_main, (1, 1, T))
    theta_main = np.reshape(theta_main, (1, 1, T))
    xtouch_main = np.reshape(xtouch_main, (npts_touch, T))
    ytouch_main = np.reshape(ytouch_main, (npts_touch, T))

    # don't deal with missing data :)
    # assert (np.any(np.isnan(xeye_main)) == False)
    # assert (np.any(np.isnan(yeye_main)) == False)
    # assert (np.any(np.isnan(theta_main)) == False)
    if np.any(np.isnan(xeye_main)):
        LOG.warning(f"{np.isnan(xeye_main).sum()} / {T} is nan")

    # vision bin size
    step = 2. * np.pi / SENSORY_PARAMS['n_oma']

    # compute other flies view

    # convert to this fly's coord system
    dx = xvision_other - xeye_main
    dy = yvision_other - yeye_main

    # distance
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # angle in the original coordinate system
    angle0 = np.arctan2(dy, dx)

    # subtract off angle of main fly
    angle = angle0 - theta_main
    angle = modrange(angle, -np.pi, np.pi)

    # which other flies pass beyond the -pi to pi border
    isbackpos = angle > np.pi / 2
    isbackneg = angle < -np.pi / 2
    isfront = np.abs(angle) <= np.pi / 2
    idxmod = np.any(isbackpos, axis=0) & np.any(isbackneg, axis=0) & (np.any(isfront, axis=0) == False)

    # bin - npts x nflies x T
    b_all = np.floor((angle + np.pi) / step)

    # bin range
    # shape: nflies x T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        minb = np.nanmin(b_all, axis=0)
        maxb = np.nanmax(b_all, axis=0)
        mind = np.nanmin(dist, axis=0)

        # n_oma x 1 x 1
    tmpbins = np.arange(SENSORY_PARAMS['n_oma'])[:, None, None]

    # n_oma x nflies x T
    mindrep = np.tile(mind[None, ...], (SENSORY_PARAMS['n_oma'], 1, 1))
    mask = (tmpbins >= minb[None, ...]) & (tmpbins <= maxb[None, ...])

    if np.any(idxmod):
        # this is complicated!!
        # find the max bin for negative angles
        # and the min bin for positive angles
        # store them in min and max for consistency with non-wraparound
        isbackpos1 = isbackpos[:, idxmod]
        isbackneg1 = isbackneg[:, idxmod]
        bmodneg = b_all[:, idxmod]
        bmodneg[isbackpos1] = np.nan
        minbmod = np.nanmax(bmodneg, axis=0)
        bmodpos = b_all[:, idxmod]
        bmodpos[isbackneg1] = np.nan
        maxbmod = np.nanmin(bmodpos, axis=0)
        mask[:, idxmod] = (tmpbins[..., 0] >= maxbmod[None, :]) | (tmpbins[..., 0] <= minbmod[None, :])

    otherflies_vision = np.nanmin(np.where(mask, mindrep, np.inf), axis=1, initial=np.inf)

    otherflies_vision = 1. - np.minimum(1.,
                                        SENSORY_PARAMS['otherflies_vision_mult'] * otherflies_vision ** SENSORY_PARAMS[
                                            'otherflies_vision_exp'])

    # t = 249
    # debug_plot_otherflies_vision(t,xother,yother,xeye_main,yeye_main,theta_main,
    #                                 angle0,angle,dist,b_all,otherflies_vision,params)

    # distance from center of arena
    # center of arena is assumed to be [0,0]
    distarena = np.sqrt(xtouch_main ** 2. + ytouch_main ** 2)

    # height of chamber
    wall_touch = np.zeros(distarena.shape)
    wall_touch[:] = SENSORY_PARAMS['arena_height']
    wall_touch = np.minimum(SENSORY_PARAMS['arena_height'], np.maximum(0., SENSORY_PARAMS['arena_height'] - (
            distarena - SENSORY_PARAMS['inner_arena_radius']) * SENSORY_PARAMS['arena_height'] / (SENSORY_PARAMS[
                                                                                                      'outer_arena_radius'] -
                                                                                                  SENSORY_PARAMS[
                                                                                                      'inner_arena_radius'])))
    wall_touch[distarena >= SENSORY_PARAMS['outer_arena_radius']] = 0.

    # t = 0
    # debug_plot_wall_touch(t,xlegtip_main,ylegtip_main,distleg,wall_touch,params)

    # xtouch_main: npts_touch x T, xtouch_other: npts_touch_other x nflies x T
    if SENSORY_PARAMS['compute_otherflies_touch']:
        dx = xtouch_main.reshape((npts_touch, 1, 1, T)) - xtouch_other.reshape((1, npts_touch_other, nflies, T))
        dy = ytouch_main.reshape((npts_touch, 1, 1, T)) - ytouch_other.reshape((1, npts_touch_other, nflies, T))
        d = np.sqrt(np.nanmin(dx ** 2 + dy ** 2, axis=2)).reshape(npts_touch * npts_touch_other, T)
        otherflies_touch = 1. - np.minimum(1., SENSORY_PARAMS['otherflies_touch_mult'] * d ** SENSORY_PARAMS[
            'otherflies_touch_exp'])
    else:
        otherflies_touch = None

    return (otherflies_vision, wall_touch, otherflies_touch)


def compute_otherflies_touch_mult(data, prct=99):
    # 1/maxd^exp = mult*maxd^exp

    # X is nkeypts x 2 x T x nflies
    nkpts = data['X'].shape[0]
    T = data['X'].shape[2]
    nflies = data['X'].shape[3]
    # isdata is T x nflies
    # X will be nkpts x 2 x N
    X = data['X'].reshape([nkpts, 2, T * nflies])[:, :, data['isdata'].flatten()]
    # maximum distance from some keypoint to any keypoint included in kpother
    d = np.sqrt(
        np.nanmax(np.nanmin(np.sum((X[None, kptouch, :, :] - X[kptouch_other, None, :, :]) ** 2., axis=2), axis=0),
                  axis=0))
    maxd = np.percentile(d, prct)

    otherflies_touch_mult = 1. / ((maxd) ** SENSORY_PARAMS['otherflies_touch_exp'])
    return otherflies_touch_mult


def ensure_otherflies_touch_mult(data):
    if np.isnan(SENSORY_PARAMS['otherflies_touch_mult']):
        LOG.info('computing touch parameters...')
        SENSORY_PARAMS['otherflies_touch_mult'] = compute_otherflies_touch_mult(data)


def compute_sensory_wrapper(Xkp, flynum, theta_main=None, returnall=False, returnidx=False):
    # other flies positions
    idxother = np.ones(Xkp.shape[-1], dtype=bool)
    idxother[flynum] = False
    Xkp_other = Xkp[:, :, :, idxother]

    xeye_main = Xkp[kpeye, 0, :, flynum]
    yeye_main = Xkp[kpeye, 1, :, flynum]
    xtouch_main = Xkp[kptouch, 0, :, flynum]
    ytouch_main = Xkp[kptouch, 1, :, flynum]
    xvision_other = Xkp_other[kpvision_other, 0, ...].transpose((0, 2, 1))
    yvision_other = Xkp_other[kpvision_other, 1, ...].transpose((0, 2, 1))
    xtouch_other = Xkp_other[kptouch_other, 0, ...].transpose((0, 2, 1))
    ytouch_other = Xkp_other[kptouch_other, 1, ...].transpose((0, 2, 1))

    if theta_main is None:
        _, _, theta_main = body_centric_kp(Xkp[..., [flynum, ]])
        theta_main = theta_main[..., 0].astype(np.float64)

    otherflies_vision, wall_touch, otherflies_touch = \
        compute_sensory(xeye_main, yeye_main, theta_main + np.pi / 2,
                        xtouch_main, ytouch_main,
                        xvision_other, yvision_other,
                        xtouch_other, ytouch_other)
    sensory = np.r_[wall_touch, otherflies_vision]
    idxinfo = {}
    idxoff = 0
    idxinfo['wall_touch'] = [0, wall_touch.shape[0]]
    idxoff += wall_touch.shape[0]
    idxinfo['otherflies_vision'] = [idxoff, idxoff + otherflies_vision.shape[0]]
    idxoff += otherflies_vision.shape[0]

    if otherflies_touch is not None:
        sensory = np.r_[sensory, otherflies_touch]
        idxinfo['otherflies_touch'] = [idxoff, idxoff + otherflies_touch.shape[0]]
        idxoff += otherflies_touch.shape[0]

    ret = (sensory,)
    if returnall:
        ret = ret + (wall_touch, otherflies_vision, otherflies_touch)
    if returnidx:
        ret = ret + (idxinfo,)

    return ret


def get_sensory_feature_shapes(simplify=None):
    idx = collections.OrderedDict()
    sz = collections.OrderedDict()
    i0 = 0
    i1 = nrelative
    idx['pose'] = [i0, i1]
    sz['pose'] = (nrelative,)

    if simplify is None:
        i0 = i1
        i1 = i0 + nkptouch
        idx['wall_touch'] = [i0, i1]
        sz['wall_touch'] = (nkptouch,)
        i0 = i1
        i1 = i0 + SENSORY_PARAMS['n_oma']
        idx['otherflies_vision'] = [i0, i1]
        sz['otherflies_vision'] = (SENSORY_PARAMS['n_oma'],)
        i0 = i1
        i1 = i0 + nkptouch * nkptouch_other
        idx['otherflies_touch'] = [i0, i1]
        sz['otherflies_touch'] = (nkptouch * nkptouch_other,)
    return idx, sz


def get_sensory_feature_idx(simplify=None):
    idx, _ = get_sensory_feature_shapes()
    return idx


""" Compute all features
"""


def combine_inputs(relpose=None, sensory=None, input=None, labels=None, dim=0):
    if input is None:
        if sensory is None:
            input = relpose
        else:
            input = np.concatenate((relpose, sensory), axis=dim)
    if labels is not None:
        input = np.concatenate((input, labels), axis=dim)
    return input


def compute_features(X, id=None, flynum=0, scale_perfly=None, smush=True, outtype=None,
                     simplify_out=None, simplify_in=None, dct_m=None, tspred_global=[1, ],
                     npad=None, compute_pose_vel=True, discreteidx=[], returnidx=False,
                     compute_labels=True):
    res = {}

    # convert to relative locations of body parts
    if id is None:
        scale = scale_perfly
    else:
        scale = scale_perfly[:, id]

    relpose, globalpos = compute_pose_features(X[..., flynum], scale)
    relpose = relpose[..., 0]
    globalpos = globalpos[..., 0]
    
    # how many frames should we cut from the end because of future frame predictions
    # use the minimum value by default
    min_npad = compute_npad(tspred_global,dct_m)
    if npad is None:
      npad = min_npad
      
    if npad == 0:
        endidx = None
    else:
        endidx = -npad
    if simplify_in == 'no_sensory':
        sensory = None
        res['input'] = relpose.T
        if returnidx:
            idxinfo = {}
            idxinfo['input'] = {}
            idxinfo['input']['relpose'] = [0, relpose.shape[0]]
    else:
        # some frames may have all nan data, just for pre-allocating. ignore those
        isdata = np.any(np.isfinite(X[...,flynum]), axis=(0, 1))
        if endidx is not None:
            isdata[endidx:] = False
        out = compute_sensory_wrapper(X[:, :, isdata, :], flynum, theta_main=globalpos[featthetaglobal, isdata],
                                      returnall=True, returnidx=returnidx)
        out = list(out)

        for i in range(4):
            outcurr = np.zeros(out[i].shape[:-1]+isdata.shape) + np.nan
            outcurr[...,isdata] = out[i]
            outcurr = outcurr[...,:endidx]
            out[i] = outcurr
        sensory, wall_touch, otherflies_vision, otherflies_touch = out[:4]
        
        if returnidx:
            idxinfo = {}
            idxinfo['input'] = out[4]

        res['input'] = combine_inputs(relpose=relpose[:, :endidx], sensory=sensory).T
        if returnidx:
            idxinfo['input'] = {k: [vv + relpose.shape[0] for vv in v] for k, v in idxinfo['input'].items()}
            idxinfo['input']['relpose'] = [0, relpose.shape[0]]

        if not smush:
            res['wall_touch'] = wall_touch[:, :-1]
            res['otherflies_vision'] = otherflies_vision[:, :-1]
            res['next_wall_touch'] = wall_touch[:, -1]
            res['next_otherflies_vision'] = otherflies_vision[:, -1]
            if otherflies_touch is not None:
                res['otherflies_touch'] = otherflies_touch[:, :-1]
                res['next_otherflies_touch'] = otherflies_touch[:, -1]

    res['scale'] = scale

    # if we can't/shouldn't compute labels, just output the inputs
    if relpose.shape[-1] <= min_npad or not compute_labels:
        # sequence to short to compute movement
        res['labels'] = None
        res['init'] = None
        
        if not smush:
            res['global'] = None
            res['relative'] = None
            res['nextglobal'] = None
            res['nextrelative'] = None
        
    else:
        out = compute_movement(relpose=relpose, globalpos=globalpos, simplify=simplify_out, dct_m=dct_m,
                              tspred_global=tspred_global, compute_pose_vel=compute_pose_vel,
                              discreteidx=discreteidx, returnidx=returnidx)
        movement = out[0]
        init = out[1]
        if returnidx:
            idxinfo['labels'] = out[2]

        if simplify_out is not None:
            if simplify_out == 'global':
                movement = movement[featglobal, ...]
            else:
                raise

        res['labels'] = movement.T
        res['init'] = init
        #res['init'] = np.r_[globalpos[:, :2],relpose[:,:2]]


        if not smush:
            res['global'] = globalpos[:, :-1]
            res['relative'] = relpose[:, :-1]
            res['nextglobal'] = globalpos[:, -1]
            res['nextrelative'] = relpose[:, -1]

    if outtype is not None:
      for key, val in res.items():
        if val is not None:
          res[key] = val.astype(outtype)

    if returnidx:
        return res, idxinfo
    else:
        return res


def split_features(X, simplify=None, axis=-1):
    res = {}
    idx = get_sensory_feature_idx(simplify)
    sz = list(idx.values())[-1][-1]
    assert X.shape[-1] == sz, f'X.shape[-1] = {X.shape[-1]}, should be {sz}'
    for k, v in idx.items():
        if torch.is_tensor(X):
            res[k] = torch.index_select(X, axis, torch.tensor(range(v[0], v[1])))
        else:
            res[k] = X.take(range(v[0], v[1]), axis=axis)

    return res

def feat2kp(Xfeat, scale_perfly, flyid=None):
    # Xfeat is npose x T x nflies
    # scale_perfly is nscale x nflies
    ndim = np.ndim(Xfeat)
    if ndim >= 2:
        T = Xfeat.shape[1]
    else:
        T = 1
    if ndim >= 3:
        nflies = Xfeat.shape[2]
    else:
        nflies = 1
    if np.ndim(scale_perfly) == 1:
        scale_perfly = scale_perfly[:, None]

    if flyid is None:
        assert (scale_perfly.shape[1] == nflies)
        flyid = np.tile(np.arange(nflies, dtype=int)[np.newaxis, :], (T, 1))

    Xfeat = Xfeat.reshape((Xfeat.shape[0], T, nflies))

    porigin = Xfeat[[posenames.index('thorax_front_x'), posenames.index('thorax_front_y')], ...]
    thorax_theta = Xfeat[posenames.index('orientation'), ...]

    # Xkpn will be normalized by the following translation and rotation
    Xkpn = np.zeros((len(keypointnames), 2, T, nflies))
    Xkpn[:] = np.nan

    # thorax
    thorax_width = scale_perfly[scalenames.index('thorax_width'), flyid].reshape((T, nflies))
    thorax_length = scale_perfly[scalenames.index('thorax_length'), flyid].reshape((T, nflies))
    Xkpn[keypointnames.index('left_front_thorax'), 0, ...] = -thorax_width / 2.
    Xkpn[keypointnames.index('left_front_thorax'), 1, ...] = 0.
    Xkpn[keypointnames.index('right_front_thorax'), 0, ...] = thorax_width / 2.
    Xkpn[keypointnames.index('right_front_thorax'), 1, ...] = 0.
    Xkpn[keypointnames.index('base_thorax'), 0, ...] = 0.
    Xkpn[keypointnames.index('base_thorax'), 1, ...] = -thorax_length

    # head
    bhead = Xfeat[[posenames.index('head_base_x'), posenames.index('head_base_y')], ...]
    headangle = Xfeat[posenames.index('head_angle'), ...] + np.pi / 2.
    headwidth = scale_perfly[scalenames.index('head_width'), flyid].reshape((T, nflies))
    headheight = scale_perfly[scalenames.index('head_height'), flyid].reshape((T, nflies))
    cosha = np.cos(headangle - np.pi / 2.)
    sinha = np.sin(headangle - np.pi / 2.)
    leye = bhead.copy()
    leye[0, ...] -= headwidth / 2. * cosha
    leye[1, ...] -= headwidth / 2. * sinha
    reye = bhead.copy()
    reye[0, ...] += headwidth / 2. * cosha
    reye[1, ...] += headwidth / 2. * sinha
    Xkpn[keypointnames.index('left_eye'), ...] = leye
    Xkpn[keypointnames.index('right_eye'), ...] = reye
    Xkpn[keypointnames.index('antennae_midpoint'), ...] = angledist2xy(bhead, headangle, headheight)

    # abdomen
    pthorax = np.zeros((2, T, nflies))
    pthorax[1, ...] = -thorax_length
    abdomenangle = Xfeat[posenames.index('abdomen_angle'), ...] - np.pi / 2.
    abdomendist = scale_perfly[scalenames.index('abdomen_length'), flyid].reshape((T, nflies))
    Xkpn[keypointnames.index('tip_abdomen'), ...] = angledist2xy(pthorax, abdomenangle, abdomendist)

    # front legs
    legangle = np.pi - Xfeat[posenames.index('left_front_leg_tip_angle'), ...]
    legdist = Xfeat[posenames.index('left_front_leg_tip_dist'), ...]
    Xkpn[keypointnames.index('left_front_leg_tip'), ...] = angledist2xy(np.zeros((2, T, nflies)), legangle, legdist)

    legangle = Xfeat[posenames.index('right_front_leg_tip_angle'), ...]
    legdist = Xfeat[posenames.index('right_front_leg_tip_dist'), ...]
    Xkpn[keypointnames.index('right_front_leg_tip'), ...] = angledist2xy(np.zeros((2, T, nflies)), legangle, legdist)

    # middle leg femur base
    pmidthorax = np.zeros((2, T, nflies))
    pmidthorax[1, ...] = -thorax_length / 2.

    lfemurbaseangle = np.pi - Xfeat[posenames.index('left_middle_femur_base_angle'), ...]
    legdist = Xfeat[posenames.index('left_middle_femur_base_dist'), ...]
    lfemurbase = angledist2xy(pmidthorax, lfemurbaseangle, legdist)
    Xkpn[keypointnames.index('left_middle_femur_base'), ...] = lfemurbase

    rfemurbaseangle = Xfeat[posenames.index('right_middle_femur_base_angle'), ...]
    legdist = Xfeat[posenames.index('right_middle_femur_base_dist'), ...]
    rfemurbase = angledist2xy(pmidthorax, rfemurbaseangle, legdist)
    Xkpn[keypointnames.index('right_middle_femur_base'), ...] = rfemurbase

    # middle leg femur tibia joint

    # lftangleoffset = np.pi-lftangle-(np.pi-lbaseangle)
    #                = lbaseangle - lftangle
    # lftangle = lbaseangle - lftangleoffset
    lftangleoffset = Xfeat[posenames.index('left_middle_femur_tibia_joint_angle'), ...]
    lftangle = lfemurbaseangle - lftangleoffset
    legdist = Xfeat[posenames.index('left_middle_femur_tibia_joint_dist'), ...]
    lftjoint = angledist2xy(lfemurbase, lftangle, legdist)
    Xkpn[keypointnames.index('left_middle_femur_tibia_joint'), ...] = lftjoint

    rftangleoffset = Xfeat[posenames.index('right_middle_femur_tibia_joint_angle'), ...]
    rftangle = rfemurbaseangle + rftangleoffset
    legdist = Xfeat[posenames.index('right_middle_femur_tibia_joint_dist'), ...]
    rftjoint = angledist2xy(rfemurbase, rftangle, legdist)
    Xkpn[keypointnames.index('right_middle_femur_tibia_joint'), ...] = rftjoint

    # middle leg tip
    ltipoffset = Xfeat[posenames.index('left_middle_leg_tip_angle'), ...]
    ltipangle = lftangle - ltipoffset
    legdist = Xfeat[posenames.index('left_middle_leg_tip_dist'), ...]
    ltip = angledist2xy(lftjoint, ltipangle, legdist)
    Xkpn[keypointnames.index('left_middle_leg_tip'), ...] = ltip

    rtipoffset = Xfeat[posenames.index('right_middle_leg_tip_angle'), ...]
    rtipangle = rftangle + rtipoffset
    legdist = Xfeat[posenames.index('right_middle_leg_tip_dist'), ...]
    rtip = angledist2xy(rftjoint, rtipangle, legdist)
    Xkpn[keypointnames.index('right_middle_leg_tip'), ...] = rtip

    # back leg
    legangle = np.pi - Xfeat[posenames.index('left_back_leg_tip_angle'), ...]
    legdist = Xfeat[posenames.index('left_back_leg_tip_dist'), ...]
    Xkpn[keypointnames.index('left_back_leg_tip'), ...] = angledist2xy(pthorax, legangle, legdist)

    legangle = Xfeat[posenames.index('right_back_leg_tip_angle'), ...]
    legdist = Xfeat[posenames.index('right_back_leg_tip_dist'), ...]
    Xkpn[keypointnames.index('right_back_leg_tip'), ...] = angledist2xy(pthorax, legangle, legdist)

    wingangle = np.pi + Xfeat[posenames.index('left_wing_angle'), ...]
    wingdist = scale_perfly[scalenames.index('wing_length'), flyid].reshape((T, nflies))
    Xkpn[keypointnames.index('wing_left'), ...] = angledist2xy(pmidthorax, wingangle, wingdist)

    wingangle = -Xfeat[posenames.index('right_wing_angle'), ...]
    Xkpn[keypointnames.index('wing_right'), ...] = angledist2xy(pmidthorax, wingangle, wingdist)

    Xkp = rotate_2d_points(Xkpn, -thorax_theta) + porigin[np.newaxis, ...]

    return Xkp

def sanity_check_tspred(data, compute_feature_params, npad, scale_perfly, contextl=512, t0=510, flynum=0):
    # sanity check on computing features when predicting many frames into the future
    # compute inputs and outputs for frames t0:t0+contextl+npad+1 with tspred_global set by config
    # and inputs ant outputs for frames t0:t0+contextl+1 with just next frame prediction.
    # the inputs should match each other
    # the outputs for each of the compute_feature_params['tspred_global'] should match the next frame
    # predictions for the corresponding frame

    epsilon = 1e-6
    id = data['ids'][t0, flynum]

    # compute inputs and outputs with tspred_global = compute_feature_params['tspred_global']
    contextlpad = contextl + npad
    t1 = t0 + contextlpad - 1
    x = data['X'][..., t0:t1 + 1, :]
    xcurr1, idxinfo1 = compute_features(x, id, flynum, scale_perfly, outtype=np.float32, returnidx=True, npad=npad,
                                        **compute_feature_params)

    # compute inputs and outputs with tspred_global = [1,]
    contextlpad = contextl + 1
    t1 = t0 + contextlpad - 1
    x = data['X'][..., t0:t1 + 1, :]
    xcurr0, idxinfo0 = compute_features(x, id, flynum, scale_perfly, outtype=np.float32, tspred_global=[1, ],
                                        returnidx=True,
                                        **{k: v for k, v in compute_feature_params.items() if k != 'tspred_global'})

    assert np.all(np.abs(xcurr0['input'] - xcurr1['input']) < epsilon)
    for f in featglobal:
        # find row of np.array idxinfo1['labels']['global_feat_tau'] that equals (f,1)
        i1 = np.nonzero(
            (idxinfo1['labels']['global_feat_tau'][:, 0] == f) & (idxinfo1['labels']['global_feat_tau'][:, 1] == 1))[0][
            0]
        i0 = np.nonzero(
            (idxinfo0['labels']['global_feat_tau'][:, 0] == f) & (idxinfo0['labels']['global_feat_tau'][:, 1] == 1))[0][
            0]
        assert np.all(np.abs(xcurr1['labels'][:, i1] - xcurr0['labels'][:, i0]) < epsilon)

    return

def compute_noise_params(data, scale_perfly, sig_tracking=.25 / PXPERMM, delta_kpts=None,
                         simplify_out=None, compute_pose_vel=True,return_extra=False,
                         compute_std=True,compute_prctile=50,sample_correlated=True):
    # contextlpad = 2

    # # all frames for the main fly must have real data
    # allisdata = interval_all(data['isdata'],contextlpad)
    # isnotsplit = interval_all(data['isstart']==False,contextlpad-1)[1:,...]
    # canstart = np.logical_and(allisdata,isnotsplit)

    # X is nkeypts x 2 x T x nflies
    maxnflies = data['X'].shape[3]
    nkpts = data['X'].shape[0]

    sample_distribution = delta_kpts is not None
    if sample_distribution:
        ndist = delta_kpts.shape[-1]

    else:
        iscorrelatednoise = np.isscalar(sig_tracking) == False
        if iscorrelatednoise:
            assert sig_tracking.shape[0] == nkpts*2
    
    if compute_std:
        alld = 0.
    else:
        alld = None
    n = 0
    # loop through ids
    LOG.info('Computing noise parameters...')
    if return_extra:
        all_movement = {}
        all_deltas = {}
    for flynum in tqdm.trange(maxnflies):
        idx0 = data['isdata'][:, flynum] & (data['isstart'][:, flynum] == False)
        # bout starts and ends
        t0s = np.nonzero(np.r_[idx0[0], (idx0[:-1] == False) & (idx0[1:] == True)])[0]
        t1s = np.nonzero(np.r_[(idx0[:-1] == True) & (idx0[1:] == False), idx0[-1]])[0]

        for i in range(len(t0s)):
            t0 = t0s[i]
            t1 = t1s[i]
            id = data['ids'][t0, flynum]
            scale = scale_perfly[:, id]
            xkp = data['X'][:, :, t0:t1 + 1, flynum]
            # np.r_[globalpos[:,:2],relpose[:,:2]] == init
            relpose, globalpos = compute_pose_features(xkp, scale)
            movement,init = compute_movement(relpose=relpose, globalpos=globalpos, simplify=simplify_out,
                                             compute_pose_vel=compute_pose_vel)
            if sample_distribution:
                if sample_correlated:
                    sampleidx = np.random.choice(ndist, size=xkp.shape[-1])
                    nu = delta_kpts[..., sampleidx]
                else:
                    # sampleidx will be nkpts x 2 x ncurr
                    sampleidx = np.random.choice(ndist, size=xkp.shape)
                    kidx,didx,_ = np.meshgrid(range(nkpts),range(2),range(xkp.shape[-1]),indexing='ij')
                    nu = delta_kpts[kidx,didx,sampleidx]
            elif iscorrelatednoise:
                nu = np.random.multivariate_normal(mean=np.zeros(nkpts*2),cov=sig_tracking,size=xkp.shape[-1])
                nu = nu.reshape((nkpts,2,xkp.shape[-1]))
            else:
                nu = np.random.normal(scale=sig_tracking, size=xkp.shape)
            relpose_pert, globalpos_pert = compute_pose_features(xkp + nu, scale)
            movement_pert,init = compute_movement(relpose=relpose_pert, globalpos=globalpos_pert, simplify=simplify_out,
                                                  compute_pose_vel=compute_pose_vel)            
            delta = movement_pert - movement
            if compute_std:
                ncurr = np.sum((np.isnan(movement) == False), axis=1)
                alld += np.nansum(delta ** 2., axis=1)
            else:
                ncurr = delta.shape[1]
                if alld is None:
                    # this won't work if there are nans in the middle of movement
                    alld = np.zeros((delta.shape[0],np.count_nonzero(data['isdata'] & (data['isstart'] == False))))
                    alld[:] = np.nan
                alld[:,n:n+ncurr] = np.abs(delta[...,0])
            n += ncurr
                
            
            if return_extra:
                all_movement[(flynum,t0)] = movement
                # note that this is redundant with alld if compute_std == False
                all_deltas[(flynum,t0)] = delta

    if compute_std:
        epsilon = np.sqrt(alld / n)
    else:
        epsilon = np.nanpercentile(alld[:,:n],compute_prctile,axis=1)
        
    epsilon = epsilon.flatten()

    if return_extra:
        return epsilon, all_movement, all_deltas
    else:
        return epsilon

def compute_pose_distribution_stats(data,scales_perfly,prctiles=[0,.001,.01,.1,.5,1,2.5,5]):
    
    X = data['X'].reshape((data['X'].shape[0],data['X'].shape[1],-1))
    ids = data['ids'].flatten()
    isdata = data['isdata'].flatten() & (ids >= 0)
    X = X[:,:,isdata]
    ids = ids[isdata]
    unique_ids = np.unique(ids)
    N = X.shape[-1]
    print(f'N = {N}')
    relposes = None
    off = 0
    for id in tqdm.tqdm(unique_ids):
        idxcurr = ids == id
        xcurr = X[:,:,idxcurr]
        relpose,_ = compute_pose_features(xcurr,scales_perfly[:,id])
        if relposes is None:
            d = relpose.shape[0]
            relposes = np.zeros((d,N),dtype=relpose.dtype)
            relposes[:] = np.nan
        ncurr = relpose.shape[1]
        relposes[:,off:off+ncurr] = relpose[:,:,0]
        off += ncurr

    relfeatangle = featangle[featrelative]

    prctiles = np.atleast_1d(prctiles)
    low_prctiles = prctiles
    high_prctiles = 100-prctiles[::-1]
    
    poseangles = relposes[relfeatangle,:]
    meanangles = circmean(poseangles,axis=1,nan_policy='omit')
    stdangles = circstd(poseangles,axis=1,nan_policy='omit')
    meanangles = modrange(meanangles,-np.pi,np.pi)
    dmeanangles = modrange(poseangles-meanangles[:,None],-np.pi,np.pi)
    prctiles_angle = np.nanpercentile(dmeanangles,np.r_[low_prctiles,high_prctiles],axis=1)
    prctiles_angle = modrange(prctiles_angle+meanangles[None,:],-np.pi,np.pi)

    meanrest = np.nanmean(relposes[relfeatangle==False,:],axis=1)
    stdrest = np.nanstd(relposes[relfeatangle==False,:],axis=1)
    prctiles_rest = np.nanpercentile(relposes[relfeatangle==False,:],np.r_[low_prctiles,high_prctiles],axis=1)

    meanrelpose = np.zeros((nrelative,))
    meanrelpose[relfeatangle] = meanangles
    meanrelpose[relfeatangle==False] = meanrest
    stdrelpose = np.zeros((nrelative,))
    stdrelpose[relfeatangle] = stdangles
    stdrelpose[relfeatangle==False] = stdrest
    prctiles_relpose = np.zeros((len(prctiles)*2,nrelative))
    prctiles_relpose[:,relfeatangle] = prctiles_angle
    prctiles_relpose[:,relfeatangle==False] = prctiles_rest
    low_prctiles_relpose = prctiles_relpose[:len(prctiles)]
    high_prctiles_relpose = prctiles_relpose[len(prctiles):]

    return {'meanrelpose': meanrelpose,
            'stdrelpose':stdrelpose,
            'low_prctiles_relpose':low_prctiles_relpose,
            'high_prctiles_relpose':high_prctiles_relpose,
            'prctiles': prctiles,
            }
def regularize_pose(pose,posestats,dampenconstant=0,prctilelim=None):
    """
    Average the input pose with posestats['meanrelpose']. Threshold at prctilelim. 
    Inputs:
    pose: ... x nfeatures
    posestats: dict with the following keys:
        meanrelpose: Mean pose, shape: nrelative
        stdrelpose: Std pose, shape: nrelative
        low_prctiles_relpose: Percentiles of the pose, shape: nprctiles x nrelative
        high_prctiles_relpose: Percentiles of the pose, shape: nprctiles x nrelative
        prctiles: percentiles computed, shape: nprctiles
    dampenconstant: 0 to 1, 0 means no dampening, 1 means full dampening
    prctilelim: None or a percentile to limit the pose to. this must be in posestats['prctiles']
    """
    relpose = pose[...,featrelative]
    newrelpose = relpose.copy()
    relfeatangle = featangle[featrelative]

    # relpose*(1-dampenconstant) + meanrelpose*dampenconstant
    if dampenconstant > 0:
        # be careful with angles -- put the mean in the same range as relpose
        dangle = modrange(posestats['meanrelpose'][relfeatangle] - relpose[...,relfeatangle],-np.pi,np.pi)
        meanangle = relpose[...,relfeatangle] + dangle
        newrelpose[...,relfeatangle] = relpose[...,relfeatangle]*(1-dampenconstant) + meanangle*dampenconstant
        newrelpose[...,relfeatangle==False] = relpose[...,relfeatangle==False]*(1-dampenconstant) + \
            posestats['meanrelpose'][relfeatangle==False]*dampenconstant
    
    if prctilelim is not None:
        prcti = np.nonzero(posestats['prctiles'] == prctilelim)[0]
        assert len(prcti) == 1
        prcti = prcti[0]
        lowb = posestats['low_prctiles_relpose'][prcti,:]
        upb = posestats['high_prctiles_relpose'][prcti,:]

        # be careful with angles -- put bounds in the same range as newrelpose
        # do this by putting the average of the bounds as close as possible to the newrelpose
        bangle = (lowb[relfeatangle]+upb[relfeatangle])/2
        widthangle = upb[relfeatangle]-lowb[relfeatangle]
        assert np.all(widthangle <= 2*np.pi)
        bangle = newrelpose[...,relfeatangle] + modrange(bangle - newrelpose[...,relfeatangle],-np.pi,np.pi)
        lowbangle = bangle - widthangle/2
        upbangle = bangle + widthangle/2
        newrelpose[...,relfeatangle] = np.minimum(np.maximum(newrelpose[...,relfeatangle],lowbangle),upbangle)
        
        newrelpose[...,relfeatangle==False] = np.minimum(np.maximum(newrelpose[...,relfeatangle==False],
                                                                    lowb[relfeatangle==False]),upb[relfeatangle==False])
    newpose = pose.copy()
    newpose[...,featrelative] = newrelpose
    return newpose