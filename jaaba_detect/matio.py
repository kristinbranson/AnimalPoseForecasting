import h5py
import numpy as np
import os
import scipy.io
import hdf5storage

def hdf5_to_dict(obj, h5file=None, uint16isstr=True):
    """
    Convert hdf5 object (possibly from matlab) to python object.
    Input:
        A: hdf5 object
        h5file: h5py file object
        Output:
        out: python object
    """
            
    if h5file is None:
        h5file = obj
        
    if isinstance(obj, h5py._hl.dataset.Dataset):

        if 'Python.Type' in obj.attrs:
            type_map = {b'bool': bool, b'int': int, b'float': float, b'str': str}
            if obj.attrs['Python.Type'] in type_map.keys():
                return type_map[obj.attrs['Python.Type']](obj[()])
            if obj.attrs['Python.Type'] == b'builtins.NoneType':
                return None
            if obj.attrs['Python.Type'] == b'numpy.ndarray':
                if 'Python.Empty' in obj.attrs and obj.attrs['Python.Empty'] == 1:
                    return np.zeros(0)
                else:
                    return obj[()].T # god knows why we are transposing...
            if obj.attrs['Python.Type'] in [b'list', b'tuple']:
                # Handle complex Python list/tuple types
                if 'Python.Empty' in obj.attrs and obj.attrs['Python.Empty'] == 1:
                    out = []
                else:
                    out = []
                    for idx in np.ndindex(obj.shape):
                        # append to out
                        out.append(hdf5_to_dict(h5file[obj[idx]], h5file, uint16isstr))
                if obj.attrs['Python.Type'] == b'tuple':
                    out = tuple(out)
                return out
            if 'Python.numpy.Container' in obj.attrs and obj.attrs['Python.numpy.Container'] == b'scalar':
                # Handle numpy scalar containers
                out = np.array(obj[()].flatten())
                return out
            return obj[()]

        # Handle simple MATLAB numeric types
        if obj.dtype.kind in ['f', 'i', 'u', 'b']:  # float, int, uint, bool
            if obj.dtype == 'uint16' and obj.ndim == 2 and obj.shape[1] == 1 and uint16isstr:
                # convert to string
                s = ''.join(chr(c) for c in obj[:].flatten())
                return s
            else:
                return obj[()]
    
        # Handle unicode strings  
        if obj.dtype == '<u2' and obj.size == max(obj.shape):
            return "".join([chr(t) for t in obj[()].flat])
            
        # Handle matlab object array
        if obj.dtype == 'O':
            # flatten to list
            #print(f'Converting object array of shape {obj.shape}')
            out = [ hdf5_to_dict(h5file[obj[idx]], h5file, uint16isstr) for idx in np.ndindex(obj.shape) ]
            return out
        
        raise ValueError(f'Unsupported dataset dtype {obj.dtype}')

    if isinstance(obj,h5py._hl.group.Group):
        out = {}
        for key in obj.keys():
            if key == '#refs#':
                continue
            #print(f'calling hdf5_to_py on A[{key}]')
            out[key] = hdf5_to_dict(obj[key], h5file, uint16isstr)
        return out

    if isinstance(obj,h5py.h5r.Reference):
        out = hdf5_to_dict(h5file[obj],h5file,uint16isstr)
        return out

    if isinstance(obj,np.ndarray) and obj.dtype=='O':
        out = np.array([hdf5_to_dict(x,h5file,uint16isstr) for x in obj])
        return out

    return obj

def loadmat(matfile,**kwargs):
    """
    Load mat file using scipy.io.loadmat or h5py.File.
    Input: 
        matfile: path to mat file
    Output:
        f: loaded mat file object
        datatype: 'scipy' or 'h5py'
    """
    assert os.path.exists(matfile)
    try:
        data = scipy.io.loadmat(matfile, struct_as_record=False)
        datatype = 'scipy'
    except NotImplementedError:
        f = h5py.File(matfile, 'r')
        data = hdf5_to_dict(f,f,**kwargs)
        f.close()
        datatype = 'h5py'    
    except:
        ValueError(f'could not read mat file {matfile}')
        
    return data, datatype

def savemat(matfile, data, matfiletype='scipy'):
    """
    Save data to mat file using scipy.io.savemat.
    Input:
        matfile: path to mat file
        data: data to save
        (optional) matfiletype: whether to save with scipy.io ('scipy') or hdf5storage.savemat
        ('hdf5storage'), default 'scipy'
    Output:
        None
    """
    if matfiletype == 'scipy':
        scipy.io.savemat(matfile, data, appendmat=False)
    elif matfiletype == 'hdf5storage':
        # hdf5storage hasn't been updated in a long time, we should probably 
        # switch to h5py. it relies on setuptools which is deprecated.
        hdf5storage.savemat(matfile,data,appendmat=False,truncate_existing=True)
    else:
        raise ValueError(f'Unknown matfiletype {matfiletype}')