import numpy as np
import torch
import matplotlib.pyplot as plt

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

def allocate_batch_concat(batch,n):
    """
    allocate_batch_concat(batch,n)
    Allocate for concatenating a bunch of batches in the way that batching in torch concatenates. 
    Inputs:
    batch: a single batch
    n: number of examples to allocate for
    Output:
    batch_concat: allocation for a batch of size n*batch.shape[0]
    """
    if isinstance(batch, torch.Tensor):
        batch_concat = torch.zeros((n*batch.shape[0], *batch.shape[1:]), dtype=batch.dtype, device=batch.device)
        if torch.is_floating_point(batch):
            batch_concat[:] = torch.nan
    elif isinstance(batch, np.ndarray):
        batch_concat = np.zeros((n*batch.shape[0], *batch.shape[1:]), dtype=batch.dtype)
        if np.issubdtype(batch.dtype, np.floating):
            batch_concat[:] = np.nan
    elif isinstance(batch, dict):
        batch_concat = {}
        for k in batch.keys():
            batch_concat[k] = allocate_batch_concat(batch[k],n)
    else:
        batch_concat = [None,]*n
    return batch_concat

def set_batch_concat(batch,batch_concat,off):
    """
    set_batch_concat(batch,batch_concat,off)
    Sets a batch starting at off in batch_concat. 
    Inputs:
    batch: a single batch
    batch_concat: the pre-allocated concatenated batch
    off: the offset to start at
    Outputs:
    batch_concat: the updated batch_concat
    off1: the offset after the batch
    """
    if isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
        off1 = off+batch.shape[0]
        batch_concat[off:off1] = batch
    elif isinstance(batch, dict):
        off1prev = None
        for k in batch.keys():
            batch[k],off1 = set_batch_concat(batch[k],batch_concat[k],off)
            if off1prev is not None:
                assert off1 == off1prev, 'set_batch_concat: inconsistent batch sizes'
            off1prev = off1
    else:
        batch_concat[off] = batch
        off1 = off+1
    return batch_concat,off1

def clip_batch_concat(batch_concat,totallen):
    """
    clip_batch_concat(batch_concat,totallen)
    Clip a batch_concat to a total length of totallen.
    Inputs:
    batch_concat: a batch_concat
    totallen: the total length to clip to
    Outputs:
    batch_concat: the clipped batch_concat
    """
    if isinstance(batch_concat, torch.Tensor) or isinstance(batch_concat, np.ndarray):
        return batch_concat[:totallen]
    elif isinstance(batch_concat, dict):
        for k in batch_concat.keys():
            batch_concat[k] = clip_batch_concat(batch_concat[k],totallen)
        return batch_concat
    else:
        return batch_concat[:totallen]
    
def compare_dicts(old_ex,new_ex,maxerr=None):
  for k,v in old_ex.items():
    if not k in new_ex:
      print(f'Missing key {k}')
      continue

    v = v.cpu().numpy() if type(v) is torch.Tensor else v
    newv = new_ex[k].cpu().numpy() if type(new_ex[k]) is torch.Tensor else new_ex[k]
    
    err = 0.
    if type(v) is not type(newv):
      print(f'Type mismatch for key {k}: {type(v)} vs {type(newv)}')
    elif type(v) is np.ndarray:
      if v.shape != newv.shape:
        print(f'Shape mismatch for key {k}: {v.shape} vs {newv.shape}')
        continue
      if v.size == 0:
        print(f'empty arrays for key {k}')
      else:
        err = np.nanmax(np.abs(v-newv))
        print(f'max diff {k}: {err:e}')
    elif type(v) is dict:
      print(f'Comparing dict {k}')
      compare_dicts(v,newv)
    else:
      try:
        err = np.nanmax(np.abs(v-newv))
        print(f'max diff {k}: {err:e}')
      except:
        print(f'not comparing {k}')
    if maxerr is not None:
      assert err < maxerr, f'Error too large for key {k}: {err} >= {maxerr}'
      
  missing_keys = [k for k in new_ex.keys() if not k in old_ex]
  if len(missing_keys) > 0:
    print(f'Missing keys: {missing_keys}')

  return

def draw_ellipse(x=0,y=0,a=1,b=1,theta=0,ax=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    phi = np.linspace(0,2*np.pi,100)
    xplot = x + a*np.cos(phi)*np.cos(theta) - b*np.sin(phi)*np.sin(theta)
    yplot = y + a*np.cos(phi)*np.sin(theta) + b*np.sin(phi)*np.cos(theta)
    h = ax.plot(xplot,yplot,**kwargs)
    return h

def cov2ell(S,nsig=2):
    """
    cov2ell(S)
    Convert covariance matrix to ellipse parameters
    S: ... x 2 x 2 covariance matrix
    """
    # val is the eigenvalues of S, ... x 2
    # U is the eigenvectors, ... x 2 x 2
    val,vec = np.linalg.eig(S)
    a = np.sqrt(val[...,0])*nsig
    b = np.sqrt(val[...,1])*nsig
    theta = np.arctan2(vec[...,1,0],vec[...,0,0])
    theta = np.mod(theta+np.pi/2,np.pi)-np.pi/2
    return a,b,theta

def pre_tile_array(array, drest, newpre_sz):
    """
    pre_tile_array(array, drest, newpre_sz)
    Tile the input array so that it is of size
    newpre_sz + array.shape[-drest:]
    """
    
    if drest == 0:
        oldpre_sz = ()
    else:
        oldpre_sz = array.shape[:-drest]
    d_pre_new = len(newpre_sz)
    d_pre_old = len(oldpre_sz)
    d_add = d_pre_new - d_pre_old
    for i in range(d_pre_old):
        assert newpre_sz[-i-1] == oldpre_sz[-i-1], 'pre_sz mismatch'
    array = np.tile(array, newpre_sz[:d_add] + (1,) * (d_pre_old+drest))
    return array

def pad_axis_array(array, npad, ax, **kwargs):
    """
    pad_axis_array(array, npad, ax)
    Pad the input array along the specified axis
    """
    if npad == 0:
        return array
    if ax < 0:
        ax = array.ndim + ax
    return np.pad(array, ((0,)*ax + (0, npad),), **kwargs)
