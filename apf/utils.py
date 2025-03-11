import numpy as np
import torch
import inspect
from typing import Callable


def modrange(x, l, u):
    return np.mod(x - l, u - l) + l


def mod2pi(radian: np.ndarray | float) -> np.ndarray | float:
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
    elif hasattr(x, '__len__'):
        return len(x)
    elif type(x) is slice:
        # figure out the length of the slice
        assert defaultlen is not None, 'if defaultlen is None, the length of the slice cannot be determined'
        return len(range(*x.indices(defaultlen)))
    else:
        return 1


def dict_convert_torch_to_numpy(d):
    for k, v in d.items():
        if type(v) is torch.Tensor:
            d[k] = v.numpy()
        elif type(v) is dict:
            d[k] = dict_convert_torch_to_numpy(v)
    return d


def unzscore(x, mu, sig):
    return x * sig + mu


def zscore(x, mu, sig):
    return (x - mu) / sig


def function_args_from_config(config: dict, function: Callable) -> dict:
    """ Returns a subset of entries from config that correspond to parameters to function.
    """
    function_args = inspect.getfullargspec(function).args[1:]
    return {arg: config[arg] for arg in function_args if arg in config}


def connected_components(vector: np.ndarray) -> np.ndarray:
    """ Finds connected components of vector.

    Parameters
        vector: boolean vector

    Returns
        components: n x 2 array containing (start, end) index for n connected components in vector.
    """
    # pad with zeros
    padded = np.zeros(len(vector) + 2, bool)
    padded[1:-1] = vector > 0

    indices = np.where(np.diff(padded))[0]
    starts = indices[0:-1:2]
    ends = indices[1::2]

    return np.hstack([starts[:, None], ends[:, None]])


def set_invalid_ends(data: np.ndarray, isstart: np.ndarray, dt: int) -> None:
    """ Sets last dt frames at the end of a continuous sequence to be NaN.
    Args:
        data: Data that was computed using dt, e.g. future motion prediction. (n_features, n_frames, n_agents) float
        isstart: Indicates whether a frame is the start of a sequence for an agent, (n_frames, n_agents) bool
        dt: number of frames to set as invalid.
    """
    n_agents = data.shape[-1]
    for i in range(n_agents):
        starts = np.where(isstart[:, i] == 1)[0]
        invalids = np.unique(np.concatenate([starts - i - 1 for i in range(dt)]))
        data[..., invalids, i] = np.nan

