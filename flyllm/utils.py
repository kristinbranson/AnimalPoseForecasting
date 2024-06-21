import numpy as np
from numpy.typing import ArrayLike
import torch


def modrange(x, l, u):
    return np.mod(x - l, u - l) + l


def mod2pi(radian: ArrayLike) -> ArrayLike:
    """Wraps modrange with l=-np.pi and u=np.pi"""
    return modrange(radian, -np.pi, np.pi)


def atleast_4d(array: np.ndarray) -> np.ndarray:
    """Expands the input array to have 4 dimensions if it doesn't have them already (adding to the back).

    Args:
        array: a numpy array of any size.

    Returns:
        An array with ndim >= 4, with new dimensions (if any) added to the back of the input.
    """
    ndim = np.ndim(array)
    if ndim >= 4:
        return array
    if ndim == 3:
        return array[:, :, :, None]
    if ndim == 2:
        return array[:, :, None, None]
    if ndim == 1:
        return array[:, None, None, None]


def rotate_2d_points(X, theta):
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    xr = X[:, 0, ...] * costheta + X[:, 1, ...] * sintheta
    yr = -X[:, 0, ...] * sintheta + X[:, 1, ...] * costheta
    Xr = np.concatenate((xr[:, np.newaxis, ...], yr[:, np.newaxis, ...]), axis=1)
    return Xr


def boxsum(x, n):
    # TODO: Would be nice if this could be achieved without torch, so that the data does not depend on it
    if n == 1:
        return x
    xtorch = torch.tensor(x[:, None, ...])
    y = torch.nn.functional.conv2d(xtorch, torch.ones((1, 1, n, 1), dtype=xtorch.dtype), padding='valid')
    return y[:, 0, ...].numpy()


def get_dct_matrix(N):
    """ Get the Discrete Cosine Transform coefficient matrix
  Copied from https://github.com/dulucas/siMLPe/blob/main/exps/baseline_h36m/train.py
  Back to MLP: A Simple Baseline for Human Motion Prediction
  Guo, Wen and Du, Yuming and Shen, Xi and Lepetit, Vincent and Xavier, Alameda-Pineda and Francesc, Moreno-Noguer
  arXiv preprint arXiv:2207.01567
  2022
  Args:
      N (int): number of time points

  Returns:
      dct_m: array of shape N x N with the encoding coefficients
      idct_m: array of shape N x N with the inverse coefficients
  """
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def compute_npad(tspred_global, dct_m):
    npad = np.max(tspred_global)
    if dct_m is not None:
        npad = np.maximum(dct_m.shape[0], npad)
    return npad


def get_interval_ends(tf):
    tf = np.r_[False, tf, False]
    idxstart = np.nonzero((tf[:-1] == False) & (tf[1:] == True))[0]
    idxend = np.nonzero((tf[:-1] == True) & (tf[1:] == False))[0]
    return idxstart, idxend


def npindex(big, small):
    """ Returns indices of small in sorted big.
        TODO: Seems like the caller would need to no the sort order of big as well, or that
        the returned indices should refer to the unsorted big).
        big and small should be 1D arrays
    """
    order = np.argsort(big)
    bigsorted = big[order]
    idx = np.searchsorted(bigsorted, small, side='left')
    idx[bigsorted[idx] != small] = -1
    return idx


def angledist2xy(origin, angle, dist):
    u = np.vstack((np.cos(angle[np.newaxis, ...]), np.sin(angle[np.newaxis, ...])))
    d = u * dist[np.newaxis, ...]
    xy = origin + d
    return xy


def len_wrapper(x, defaultlen=None):
    if x is None:
        return defaultlen
    if hasattr(x, '__len__'):
        return len(x)
    return 1


def dict_convert_torch_to_numpy(d):
    for k, v in d.items():
        if type(v) is torch.Tensor:
            d[k] = v.numpy()
        elif type(v) is dict:
            d[k] = dict_convert_torch_to_numpy(v)
    return d
