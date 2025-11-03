import numpy as np
import torch
import inspect
from typing import Callable
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import torch
import h5py
import scipy.io


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

    Args:
        vector: boolean vector

    Returns:
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

def allocate_batch_concat(batch,n,device=None):
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
        if device is None:
            device = batch.device
        batch_concat = torch.zeros((n*batch.shape[0], *batch.shape[1:]), dtype=batch.dtype, device=device)
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
        res = {}
        for k in batch_concat.keys():
            res[k] = clip_batch_concat(batch_concat[k],totallen)
        return res
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

def save_animation(ani, filename, writer=None, codec='h264', bitrate=None, fps=30, dpi=None, optimize=3,
                   preset='faster',crf=20,video_profile='high',pix_fmt='yuv420p',scale=None,
                   extra_args=[]):
    """
    Creates an optimized animation using matplotlib.
    
    Args:
        fig: matplotlib figure object
        filename: output filename
        writer: 'pillow' or 'ffmpeg', if None, determined from filename extension
        codec: codec for ffmpeg writer, e.g. 'h264', 'hevc', 'mpeg4'. default: None, uses FFMpegWriter default
        bitrate: bitrate for ffmpeg writer, e.g. 1800. default: None, uses FFMpegWriter default
        fps: frames per second, default: 30
        dpi: resolution in dots per inch, only used if writer is 'pillow'. default: None, uses PillowWriter default
        optimize: optimization level for gif output, obsolete, PillowWriter no longer supports optimize, 
        leaving so that old code still works
        preset: preset for ffmpeg output. default: 'faster', only used if writer is 'ffmpeg'
        crf: constant rate factor for ffmpeg output, default: 20, only used if writer is 'ffmpeg'
        video_profile: video profile for ffmpeg output, default: 'high', only used if writer is 'ffmpeg'
        pix_fmt: pixel format for ffmpeg output, default: 'yuv420p', only used if writer is 'ffmpeg'
        scale: scale filter for ffmpeg output, only used if writer is 'ffmpeg', default: None, no scaling
        extra_args: list of extra arguments to pass to the FFMegWriter as extra_args, default: []
    """
    
    if writer is None:
        # get extension from filename
        ext = filename.split('.')[-1]
        if ext == 'gif':
            writer = 'pillow'
        else:
            writer = 'ffmpeg'

    kwargs = {'fps':fps}
    if writer == 'pillow':
        # if optimize is not None:
        #     extra_args += ['-optimize',str(optimize)]
        # if len(extra_args) > 0:
        #     kwargs['extra_args'] = extra_args
        if dpi is not None:
            kwargs['dpi'] = dpi
        writer = matplotlib.animation.PillowWriter(**kwargs)
        
    elif writer == 'ffmpeg':
        if codec is not None:
            kwargs['codec'] = codec
        if bitrate is not None:
            kwargs['bitrate'] = bitrate
        if preset is not None:
            extra_args += ['-preset',preset]
        if crf is not None:
            extra_args += ['-crf',str(crf)]
        if video_profile is not None:
            extra_args += ['-profile:v',video_profile]
        if pix_fmt is not None:
            extra_args += ['-pix_fmt',pix_fmt]
        if scale is not None:
            extra_args += ['-vf','scale='+scale]
        if len(extra_args) > 0:
            kwargs['extra_args'] = extra_args
        writer = matplotlib.animation.FFMpegWriter(**kwargs)
    else:
        raise ValueError('Unknown writer type '+writer)
    
    # number of frames in ani
    try:
        total_frames = ani.save_count
    except AttributeError:
        total_frames = ani._save_count
    
    def progress_callback(current_frame, total_frames):
        progress_bar.update(1)
        progress_bar.refresh()
    
    # Save animation with progress bar
    progress_bar = tqdm.tqdm(total=total_frames, desc='Saving animation', leave=True, position=0)
    ani.save(
        filename,
        writer=writer,
        progress_callback=progress_callback)
    progress_bar.close()
    
def get_cuda_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def matstruct_to_dict(mat):
    """
    d = matstruct_to_dict(mat)
    Convert Matlab struct read with scipy.io.loadmat to a dict
    """
    d = {}
    for k in mat.dtype.names:
        if mat[k].size == 1:
            d[k] = mat[k].flatten()[0]
        else:
            d[k] = mat[k]
    return d

def matdict_to_dict(matdict):
    """
    d = matdict_to_dict(matdict)
    Convert the output from scipy.io.loadmat to a dict
    """
    d = {}
    for k,v in matdict.items():
        if isinstance(v,np.ndarray):
            if v.dtype.names is None:
                d[k] = v
            else:
                if v.size == 1:
                    d[k] = matstruct_to_dict(v)
                else:
                    d[k] = []
                    v = v.flatten()
                    for i in range(len(v)):
                        d[k].append(matstruct_to_dict(v[i]))
        else:
            d[k] = v
    return d


def hdf5_to_py(A, h5file):
    if isinstance(A, h5py._hl.dataset.Dataset):
        if 'Python.Type' in A.attrs and A.attrs['Python.Type'] == b'str':
            out = u''.join([chr(int(t)) for t in A])
        elif 'Python.Type' in A.attrs and A.attrs['Python.Type'] in [b'list', b'tuple']:
            if 'Python.Empty' in A.attrs and A.attrs['Python.Empty'] == 1:
                out = []
            else:
                out = [hdf5_to_py(t, h5file) for t in A[()].flatten().tolist()]
        elif 'Python.Type' in A.attrs and A.attrs['Python.Type'] == b'bool':
            out = bool(A[()])
        elif 'Python.Type' in A.attrs and A.attrs['Python.Type'] == b'int':
            out = int(A[()])
        elif 'Python.Type' in A.attrs and A.attrs['Python.Type'] == b'float':
            out = float(A[()])
        elif 'Python.numpy.Container' in A.attrs and A.attrs['Python.numpy.Container'] == b'scalar' and A.attrs['Python.Type'] == b'numpy.float64':
            out = np.array(A[()].flatten())
        elif 'Python.Type' in A.attrs and A.attrs['Python.Type'] == b'numpy.ndarray' and 'Python.Empty' in A.attrs and A.attrs['Python.Empty'] == 1:
            out = np.zeros(0)
        else:
            out = A[()].T
    elif isinstance(A,h5py._hl.group.Group):
        out = {}
        for key, val in A.items():
            out[key] = hdf5_to_py(val, h5file)
    elif isinstance(A,h5py.h5r.Reference):
        out = hdf5_to_py(h5file[A],h5file)
    elif isinstance(A,np.ndarray) and A.dtype=='O':
        out = np.array([hdf5_to_py(x,h5file) for x in A])
    else:
        out = A
    return out


def read_matfile(mat_file):
    """
    data = read_matfile(mat_file)
    Read mat_file either using scipy.io.loadmat for old-style mat files, h5py if hdf5 files
    """
    try:
        data = scipy.io.loadmat(mat_file)
        data = matdict_to_dict(data)
    except NotImplementedError:
        data = h5py.File(mat_file, 'r')
        data = hdf5_to_py(data,data)
    return data

def compute_ylim(h,margin=0.1):
    """
    Compute y limits based on ydata of input line handles.
    h: list of line handles
    margin: fraction of ylim to add as margin
    Returns:
    ylim: [ymin,ymax]
    """
    ylim = [np.inf,-np.inf]
    for line in h:
        ylim[0] = np.minimum(ylim[0],np.nanmin(line.get_ydata()))
        ylim[1] = np.maximum(ylim[1],np.nanmax(line.get_ydata()))
    dy = ylim[1]-ylim[0]
    ylim[0] -= margin*dy
    ylim[1] += margin*dy
    return ylim

def is_notebook():
    try:
        from IPython import get_ipython
        if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:
            return True
    except:
        pass
    return False

def set_mpl_backend(backend='tkAgg',force=False):
    # Only set non-interactive backend if not in Jupyter
    import matplotlib
    if force or not is_notebook():
        matplotlib.use(backend)
        return
        
def recursive_dict_eval(d, fun: Callable, *args, **kwargs) -> dict | np.ndarray | torch.Tensor:
    """
    dict_eval(d,fun,*args,**kwargs)
    Subindex all arrays in a dict. Recurses into sub-dicts.
    """
    if isinstance(d, dict):
        return {k: recursive_dict_eval(v, fun, *args, **kwargs) for k, v in d.items()}
    else:
        return fun(d, *args, **kwargs)