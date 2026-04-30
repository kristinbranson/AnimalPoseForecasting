from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import NamedTuple
import numpy as np
import torch
import logging
import importlib
import inspect
import copy
from typing import Any

from apf.data import fit_discretize_data, discretize_labels, weighted_sample
from apf.utils import connected_components, modrange, rotate_2d_points, set_invalid_ends, tic, toc, reshape_prefix, tile_prefix, get_optional_params

from apf.features import compute_global_velocity, compute_relpose_velocity

LOG = logging.getLogger(__name__)


@dataclass
class Operation(ABC):
    """ Abstract class for an operation to be applied to data.

    Methods to be implemented by inheriting classes are apply and inverse.
    """

    # localattrs are attributes that should not be included in to_dict and from_dict
    localattrs = []
    name = None

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray | tuple[np.ndarray, Any]:
        pass

    @abstractmethod
    def invert(self, data: np.ndarray) -> np.ndarray:
        pass
    
    def __post_init__(self):
        if self.name is None:
            self.name = self.__class__.__name__.lower()
            
    def __call__(self, data, **kwargs):
        """

        Args:
            data: Either np.ndarray or Data (defined below)
            kwargs: Arguments to be sent to the apply operation.

        Returns
            If input is np.ndarray, returns a np.ndarray processed by the operation,
            if input is Data, returns a new Data with processed array and this operation appended to operaitons.
        """
        isdata = isData(data)
        if isdata:
            # instead of checking for Data object, check for attributes
            dataarray = data.array
            invertdata = data.invertdata.copy()
            feature_names = data.feature_names
            name = data.name + '_' + self.name
            operations = data.operations
        else:
            dataarray = data
            invertdata = []
            feature_names = None
            name = self.name
            operations = []
        res = self.apply(dataarray, **kwargs)
        if isinstance(res, tuple):
            array, invert_data_curr = res
        else:
            array = res
            invert_data_curr = None
        # update invertdata
        invertdata.append(invert_data_curr)
        feature_names = self.update_feature_names(feature_names)
        if feature_names is None or len(feature_names) != array.shape[-1]:
            feature_names = [f'{name}_{i}' for i in range(array.shape[-1])]
                    
        return Data(
            name=name,
            array=array,
            operations=operations + [self],
            invertdata=invertdata,
            feature_names=feature_names
        )

    def to_dict(self) -> dict:
        """ Returns operation class and its parameters as a dictionary.
        Returns:
        Dictionary with:
            'class': class name
            'module': module name
            'attributes': dictionary of attributes of the class
        For to_dict and from_dict to work, all parameters for defining the operation must be stored as attributes of the class.
        """
        attributes = {k: v for k,v in self.__dict__.items() if k not in self.localattrs}
        return  {
                'class': self.__class__.__name__,
                'module': self.__class__.__module__,
                'attributes': attributes
                }
        
    def __str__(self):
        return f"Operation {self.name} of class {self.__class__.__name__}"
    
    def update_feature_names(self, input_feature_names: list[str]):
        """ Updates the feature names after applying the operation.
        Args:
            input_feature_names: List of feature names before applying the operation.
        Returns:
            output_feature_names: List of feature names after applying the operation.
        """
            
        return input_feature_names
    
    def invert_feature_names(self, input_feature_names: list[str]):
        """ Updates the feature names after inverting the operation.
        Args:
            input_feature_names: List of feature names before inverting the operation.
        Returns:
            output_feature_names: List of feature names after inverting the operation.
        """
            
        return input_feature_names

    @classmethod
    def from_dict(cls, oper_params: dict):
        """
        Creates an instance of the class from a dictionary created by to_dict().
        Args:
            data: Dictionary with:
                'class': class name
                'module': module name
                'attributes': dictionary of attributes of the class, used as kwargs to create the instance
        """
        # Get the actual class
        class_name = oper_params['class']
        module = oper_params['module']

        # Import the class dynamically
        mod = importlib.import_module(module)
        actual_class = getattr(mod, class_name)

        # Create instance without calling __init__
        # Filter which attributes can be provided to the constructor:
        args = {}
        if 'attributes' in oper_params:
            sig = inspect.signature(actual_class.__init__)
            constructor_params = sig.parameters
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in constructor_params.values())
            for k in oper_params['attributes']:
                if has_kwargs or (k in constructor_params):
                    args[k] = oper_params['attributes'][k]
        obj = actual_class(**args)

        return obj
    
    def invertdata_subindex(self, invertdata: Any, idx: list | tuple):
        """
        Subindexes the invertdata for the operation.
        Args:
            invertdata: Invertdata for the operation. 
            invertdata can be:
                - a dict: called recursively on each entry
                - a list: called recursively on each entry
                - an ndarray or torch.Tensor for this operation: extract indices idx
                - If invertdata is not one of these, then you must override this method in the operation class.
            idx: List of indices to subindex the invertdata.
        Returns:
            Subindexed invertdata.
        """
        
        if invertdata is None:
            return None
        if isinstance(invertdata, np.ndarray) or isinstance(invertdata, torch.Tensor):
            return invertdata[*idx]
        opfcn = lambda op, invertdatacurr: op.invertdata_subindex(invertdatacurr, idx)
        
        return self.invertdata_applyfcn(invertdata, opfcn)
    
    def invertdata_applyfcn(self, invertdata: Any, opfcn):
        """
        Apply function opfcn to invertdata for the operation. Assumes that 
        invertdata is assumed to be either:
            - None
            - a np.ndarray or torch.Tensor (base case)
            - a list of invertdata corresponding to sub operations for fusion-like operations
            - a dict of invertdata for different keys, corresponding to different extra arguments needed for inverting the operation
        Other types of invertdata must be handled by overriding this method in the operation class.
        If invertdata is None, None is returned.
        If invertdata is a np.ndarray or torch.Tensor, opfcn is called on it directly. opfcn must handle this base case. 
        If invertdata is a list corresponding to a fusion-like operation, the sub-operation's opfcn is called on each list instance. 
        If invertdata is a dict, the opfcn is called on each dict value.
        Args:
            invertdata: Invertdata for the operation.
            opfcn: Function to apply to the invertdata. Must take two arguments: the operation and the invertdata.
        """
        
        if invertdata is None:
            return None
        elif isinstance(invertdata, np.ndarray) or isinstance(invertdata, torch.Tensor):
            return opfcn(invertdata)
        elif isinstance(invertdata, list):
            # fusion-style operation
            operations = None
            if hasattr(self,'operations'):
                operations = self.operations
            elif hasattr(self,'fusion') and hasattr(self.fusion,'operations'):
                operations = self.fusion.operations

            if operations is not None:
                invertdata_sub = []
                for invertdatacurr, op in zip(invertdata,operations):
                    invertdata_sub.append(opfcn(op,invertdatacurr))
            
            else:
                invertdata_sub = []
                for invertdatacurr in invertdata:
                    invertdata_sub.append(opfcn(self, invertdatacurr))
            return invertdata_sub
        elif isinstance(invertdata, dict):
            invertdata_sub = {}
            for key in invertdata:
                invertdata_sub[key] = opfcn(self,invertdata[key])
            return invertdata_sub
        else:
            LOG.warning(f"No default method for applying function to invertdata of type {type(invertdata)}, returning original invertdata")
        
        return invertdata
    
    def invertdata_reshape(self,invertdata,prefixshape):
        """ Reshapes invertdata's prefix (dimensions before feature dim) to the given shape.
        Args:
            prefixshape: New shape for the prefix dimensions.
        Returns:
            invertdata is reshaped to have prefixshape in the prefix dimensions
        """
        if isinstance(invertdata, np.ndarray) or isinstance(invertdata, torch.Tensor):
            return reshape_prefix(invertdata,prefixshape)
        opfcn = lambda op,invertdata: op.invertdata_reshape(invertdata,prefixshape)
        return self.invertdata_applyfcn(invertdata, opfcn)
    
    def invertdata_tile(self,invertdata,nreps):
        """ Tiles invertdata's prefix (dimensions before feature dim) by the given repetitions.
        Args:
            nreps: List of repetitions to tile the prefix dimensions.
        Returns:
            Tiled invertdata.
        """
        if isinstance(invertdata, np.ndarray) or isinstance(invertdata, torch.Tensor):
            return tile_prefix(invertdata,nreps)
        opfcn = lambda op,invertdata: op.invertdata_tile(invertdata,nreps)
        return self.invertdata_applyfcn(invertdata, opfcn)

class Data(NamedTuple):
    name: str
    array: np.ndarray
    # Operations that have been applied to the data (can be later applied in inverse to obtain original data).
    operations: list[Operation] = []
    invertdata: list = [] # any additional data needed for inverting the operations, e.g. flyid for Pose operation
    # invertdata is a list with one entry per operation in operations.
    # Each entry in the list must be a dict with keys for each extra argument to invert or apply this operation
    # Each entry[key] must can be a dict, list, ndarray, Tensor, or None. 
    # We need to be able to do ndarray-like operations on entry[key] to match ndarray-like operations done on array,
    # e.g. subindexing, reshaping, tiling. By default, if entry[key] is a dict or list, the ndarray-like operation 
    # is called on each sub-entry recursively. The base case must either be None or an ndarray or Tensor with
    # a prefix shape (e.g. (n_agents, n_frames)) matching the data array in prefix shape.
    # If not, you must override the invertdata_{subindex,reshape,tile} methods in the operation class.
    
    feature_names: list[str] | None = None
    
    def __str__(self):
        s = f"Data {self.name} with shape {self.array.shape} and operations:"
        if len(self.operations) == 0:
            s += "[]\n"
        else:
            s += "\n"
        for op in self.operations:
            s += f"  {str(op)}\n"
        return s[:-1]
    
    def print_feature_names(self):
        if self.feature_names is not None and len(self.feature_names)>0:
            s = ""
            for i,fn in enumerate(self.feature_names):
                s += f"{i}. {fn}\n"
        else:
            s = "  No feature names\n"
        print(s[:-1])
        return
    
    @property
    def shape(self):
        return self.array.shape
    
    @property
    def ndim(self):
        return self.array.ndim
    
    @staticmethod
    def copy_with(data: 'Data', **kwargs) -> 'Data':
        return data._replace(**kwargs)
    
    def invertdata_applyfcn(self,opfcn) -> dict:
        invertdata1 = None
        invertdata1 = []
        for op,invertdatacurr in zip(self.operations,self.invertdata):
            invertdata1.append(opfcn(op,invertdatacurr))
        return invertdata1
    
    def __getitem__(self, idx) -> 'Data':
        """ Returns a subindexed version of the Data.
        Args:
            idx: List/tuple of indices to subindex the data array.
            Returns:
            Subindexed Data object, where
                - array is subindexed by idx
                - invertdata is subindexed by idx using each operation's invertdata_subindex method
        """
        if not isinstance(idx,tuple):
            idx = (idx,)
        array_sub = self.array
        if len(idx) > 0:
            array_sub = array_sub[idx]

        fcn = lambda op,invertdata: op.invertdata_subindex(invertdata,idx)
        invertdata_sub = self.invertdata_applyfcn(fcn)

        return Data.copy_with(self, array=array_sub, invertdata=invertdata_sub)
    
    def reshape(self,prefix_shape) -> 'Data':
        """ Reshapes the prefix (dimensions before feature dim) to the given shape.
        Args:
            prefix_shape: New shape for the prefix dimensions.
        Returns:
            Reshaped Data object, where
                - array is reshaped to have prefix_shape in the prefix dimensions
                - invertdata is reshaped to have prefix_shape in the prefix dimensions using each operation's invertdata_reshape method
            """
        array = reshape_prefix(self.array,prefix_shape)
        fcn = lambda op,invertdata: op.invertdata_reshape(invertdata,prefix_shape)
        invertdata = self.invertdata_applyfcn(fcn)
        return Data.copy_with(self, array=array, invertdata=invertdata)        
        
    def tile(self,prefix_tile) -> 'Data':
        """ Tiles the prefix (dimensions before feature dim) by the given repetitions.
        Args:
            prefix_tile: List of repetitions to tile the prefix dimensions.
        Returns:
            Tiled Data object, where
                - array is tiled by prefix_tile in the prefix dimensions
                - invertdata is tiled by prefix_tile in the prefix dimensions using each operation's invertdata_tile method
        """
        array = tile_prefix(self.array,prefix_tile)
        fcn = lambda op,invertdata: op.invertdata_tile(invertdata,prefix_tile)
        invertdata = self.invertdata_applyfcn(fcn)
        return Data.copy_with(self, array=array, invertdata=invertdata)
    
    def set_last_frame(self, x: np.ndarray | torch.Tensor, t: int = -1, timedim=1) -> 'Data':
        """ Sets the last frame of the data array to the given value
        Note that invertdata is not updated, so care must be taken to ensure consistency
        Args:
            x: New data to set at the given time indices. Must have shape (n_agents, n_frames_set, n_features)
            ts: Time indices to set. Can be an int, list of ints, or np.ndarray of ints.
            timedim: Which dimension of the data array corresponds to time (default 1)
        Returns:
            Data object with data array at time indices ts set to x.
        """
        array_new = self.array.copy()
        nd = array_new.ndim
        if timedim < 0:
            timedim = nd + timedim
        idx = [slice(None),]*nd
        idx[timedim] = t
        array_new[tuple(idx)] = x
        return Data.copy_with(self, array=array_new)
    
    def get_invertdata_dict(self) -> dict:
        """ Groups invertdata by operation name.

        Returns:
            invertdata_dict: Dict mapping operation name to a list of invertdata
                dicts, one per occurrence of that operation in self.operations.
        """
        invertdata_dict = {}
        for oper, invdata in zip(self.operations, self.invertdata):
            invertdata_dict.setdefault(oper.name, []).append(invdata)
        return invertdata_dict

    
def isData(obj: Any) -> bool:
    """ Checks if the object is a Data object based on its attributes.
    Robust to e.g. changing Data class in a notebook and reloading.
    Args:
        obj: Object to check.
    Returns:
        True if the object is a Data object, False otherwise.
    """
    return (
        hasattr(obj, 'name') and
        hasattr(obj, 'array') and
        hasattr(obj, 'operations') and
        hasattr(obj, 'invertdata') and
        hasattr(obj, 'feature_names')
    )
    
@dataclass
class Identity(Operation):
    """ This operation passes data through without modifications.

    Can be useful in combination with Fusion when a subset of dimensions needs to be operated on and others not.
    """
    def apply(self, data: np.ndarray):
        return data

    def invert(self, data: np.ndarray):
        return data
    
    def update_feature_names(self, input_feature_names):
        return input_feature_names
    
    def invert_feature_names(self, input_feature_names):
        return input_feature_names
    
@dataclass
class Zscore(Operation):
    """ Zscores and unzscores data.

    Attributes:
        mean: (n_feat, ) mean value of feature values. If None will be computed the first time the operation is applied.
        std: (n_feat, ) standard deviation of feature values.
    """
    mean: np.ndarray | None = None
    std: np.ndarray | None = None    
    
    def compute(self, data: np.ndarray):
        """ Computes the mean and standard variation of features in data.

        Args:
            data: (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array
        """
        
        n_feat = data.shape[-1]
        data_flat = data.reshape((-1, n_feat))
        self.mean = np.nanmean(data_flat, axis=0)
        self.std = np.nanstd(data_flat, axis=0)

    def apply(self, data: np.ndarray) -> np.ndarray:
        """ Applies zscoring to the data.

        Args:
            data: (n_agents,  n_frames, n_features) float array

        Returns:
            zscored: data with zero mean and std of 1, (n_agents,  n_frames, n_features) float array
        """
        if self.mean is None:
            self.compute(data)
        return (data - self.mean[None, None, :]) / self.std[None, None, :]

    def invert(self, data: np.ndarray) -> np.ndarray:
        """ Applies the inverse zscoring to data.

        Args:
            data: (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array

        Returns:
            unzscored: (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array
        """
        ismultiagent = data.ndim == 3
        if not ismultiagent:
            data = data[None, ...]
        inverted = data * self.std[None, None, :] + self.mean[None, None, :]
        if not ismultiagent:
            inverted = inverted[0]
        return inverted
    
    def __str__(self):
        return f"Operation {self.name} of class Zscore with mean shape {self.mean.shape if self.mean is not None else None} and std shape {self.std.shape if self.std is not None else None}"
    
    def update_feature_names(self, input_feature_names):
        return [f'{name}_zscored' for name in input_feature_names]
    
    def invert_feature_names(self, input_feature_names):
        return [name.replace('_zscored','') for name in input_feature_names]

@dataclass
class OddRoot(Operation):
    """ Takes the given root of the data to smear it out.

    Attributes:
        root: Which root to use, 3 usually works well. Must be odd so that signed data is correctly inverted.
    """
    root: int = 3
    
    def __post_init__(self):
        super().__post_init__()
        assert int(self.root) == self.root, "root must be an integer"
        assert np.mod(self.root, 2) == 1, "only odd roots are invertible with signed data"

    def apply(self, data: np.ndarray) -> np.ndarray:
        """ Applies root to the data.

        Args:
            data: (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array

        Returns:
            rooted: root-th root of data, (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array
        """
        return np.sign(data) * np.abs(data)**(1 / self.root)

    def invert(self, data: np.ndarray) -> np.ndarray:
        """ Applies the inverse root to data.

        Args:
            data: (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array

        Returns:
            unrooted: (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array
        """
        return data**self.root
    
    def __str__(self):
        return f"Operation {self.name} of class OddRoot with root {self.root}"
    
    def update_feature_names(self, input_feature_names):
        return [f'{name}_root{self.root}' for name in input_feature_names]
    
    def invert_feature_names(self, input_feature_names):
        return [name.replace(f'_root{self.root}','') for name in input_feature_names]

@dataclass
class Subset(Operation):
    """ Returns the input array with only a subset of feature dimensions.

    NOTE: this operation is not invertible.

    Attributes:
        include_ids: Which feature ids to select.
    """
    include_ids: np.ndarray

    def apply(self, data: np.ndarray) -> np.ndarray:
        """ Returns data with a subset of features.

        Args:
            data: (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array

        Returns:
            sub_data: data with only the selected feature dimensions,
            (n_agents,  n_frames, n_sub_features) float array or
            (n_frames, n_sub_features) float array
        """
        return data[..., self.include_ids]

    def invert(self, data: np.ndarray) -> None:
        LOG.error(f"Operation {self.name} is not invertible")
        return None
        
    def __str__(self):
        return f"Operation {self.name} of class Subset with include_ids {self.include_ids}"
    
    def update_feature_names(self, input_feature_names):
        if hasattr(self.include_ids, 'dtype') and self.include_ids.dtype == bool:
            include_ids = np.nonzero(self.include_ids)[0]
        else:
            include_ids = self.include_ids
        return [input_feature_names[i] for i in include_ids]
    
    def invert_feature_names(self, input_feature_names):
        LOG.error(f"Operation {self.name} is not invertible")
        return None


# TODO: Move this to apf/data, and find where I copied it from and use the one from apf/data there as well
def labels_discrete_to_continuous(labels_discrete, bin_centers, epsilon=1e-3):
    """ Converts discrete labels to continuous labels by taking the weighted sum of the bin centers.

    Copied from flyllm/pose.py

    Args:
        labels_discrete: ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the discrete labels.
        epsilon: small number to check that the discrete labels sum to 1. Default is 1e-3.

    Returns:
        continuous: ndarray of size (pre_sz x ) ntimepoints x d_discrete with the continuous version of the labels.
    """
    sz = labels_discrete.shape
    nbins = sz[-1]
    nfeat = sz[-2]
    szrest = sz[:-2]
    n = int(np.prod(np.array(szrest)))
    labels_discrete = labels_discrete.reshape((n, nfeat, nbins))

    s = np.sum(labels_discrete, axis=-1)

    # if the sum is not 1, including if nan, set to 1, but nan out results after
    isbad = np.abs(1 - s) >= epsilon
    s[isbad] = 1.

    # nfeat x nbins
    continuous = np.sum(bin_centers[None, ...] * labels_discrete, axis=-1) / s
    continuous[isbad] = np.nan
    continuous = np.reshape(continuous, szrest + (nfeat,))
    return continuous


def sample_discrete_labels(labels_discrete, bin_samples, nsamples=1):
    """ Samples the bins from probability distribution, and picks a random bin sample from the selected bin.

    Copied from flyllm/pose.py

    sample_discrete_labels(labels_discrete, nsamples=1)
    Samples continuous labels from the discrete labels.

    Args:
        labels_discrete: ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the discrete labels.
        nsamples: number of samples to take. Default is 1.

    Returns:
        continuous: ndarray of size nsamples x (pre_sz x ) ntimepoints with the continuous version of the labels.
    """
    sz = labels_discrete.shape
    nbins = sz[-1]
    nfeat = sz[-2]
    szrest = sz[:-2]
    n = int(np.prod(np.array(szrest)))
    labels_discrete = labels_discrete.reshape((n, nfeat, nbins))

    # bin_samples = self._discretize_params['bin_samples']
    nsamples_per_bin = bin_samples.shape[0]
    continuous = np.zeros((nsamples,) + szrest + (nfeat,), dtype=labels_discrete.dtype)
    for f in range(nfeat):
        # to do make weighted_sample work with numpy directly
        binnum = weighted_sample(torch.tensor(labels_discrete[:, f, :]), nsamples=nsamples).numpy()
        idxgood = binnum >= 0
        ngood = np.count_nonzero(idxgood)
        sample = np.random.randint(low=0, high=nsamples_per_bin, size=(ngood,))
        curr = np.zeros(nsamples*n)
        curr[:] = np.nan
        curr[idxgood.flatten()] = bin_samples[sample, f, binnum[idxgood]]
        curr = curr.reshape((nsamples,) + szrest)
        continuous[..., f] = curr

    return continuous

@dataclass
class Discretize(Operation):
    """ Discretizes and undiscretizes data.

    Attributes:
        bin_edges: Boundaries for bins, (n_feat, n_bins + 1) float array
        bin_centers: Centers of bins, (n_feat, n_bins) float array -- computed from bin_edges
        bin_samples: Samples from each bin, (n_samples_per_bin, n_feat, n_bins) float array -- computed from data
        fit_discretize_data_args: Named arguments to be sent to fit_discretize_data
    """

    bin_edges: np.ndarray | None = None
    bin_centers: np.ndarray | None = None
    bin_samples: np.ndarray | None = None
    fit_discretize_data_args: dict = field(default_factory=dict)

    def __init__(self, bin_edges: np.ndarray | None = None, bin_centers: np.ndarray | None = None, 
                 bin_samples: np.ndarray | None = None, fit_discretize_data_args: dict | None = None,
                 **kwargs):

        # extra kwargs are put in fit_discretize_data_args
        self.bin_edges = bin_edges
        self.bin_centers = bin_centers
        self.bin_samples = bin_samples

        # If bin_centers is None, then when we call apply bin_edges and bin_centers are recomputed, so this
        # cannot be removed. (Alternatively we could enforce bin_centers be provided when bin_edges are)
        if self.bin_edges is not None and self.bin_centers is None:
            self.bin_centers = (self.bin_edges[:, 1:] + self.bin_edges[:, :-1]) / 2

        if fit_discretize_data_args is None:
            self.fit_discretize_data_args = kwargs
        else:
            self.fit_discretize_data_args = fit_discretize_data_args | kwargs
            
        super().__post_init__()

    def compute(self, data: np.ndarray, valid: np.ndarray | None = None):
        """ Computes the bin edges for the data.

        Args:
            data: (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array
        """
        n_feat = data.shape[-1]
        data_flat = data.reshape((-1, n_feat))
        if valid is None:
            valid = ~np.isnan(data_flat.sum(-1))
        else:
            valid = valid.flatten()
        data_valid = data_flat[valid, :]
        bin_edges, samples, bin_means, bin_medians = fit_discretize_data(data_valid, bin_edges=self.bin_edges,
                                                                         **self.fit_discretize_data_args)
        self.bin_edges = bin_edges
        # self.bin_centers = (bin_edges[:, 1:] + bin_edges[:, :-1]) / 2
        self.bin_centers = bin_medians
        self.bin_samples = samples

    def apply(self, data: np.ndarray, compute_args: dict = {}) -> np.ndarray:
        """ Bins the data.

        Args:
            data: Continuous data, (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array

        Returns:
            binned: Binned data, (n_agents,  n_frames, n_features * n_bins) float array
            or (n_frames, n_features * n_bins) float array
        """
        
        if self.bin_edges is None or self.bin_centers is None or self.bin_samples is None:
            self.compute(data,**compute_args)
        sz_rest = data.shape[:-1]
        n_feat = data.shape[-1]
        data_flat = data.reshape((-1, n_feat))

        # n x n_feat x nbins float
        data_flat_discrete = discretize_labels(data_flat, self.bin_edges, soften_to_ends=True)

        invertdata = {'to_discretize': data}

        return data_flat_discrete.reshape(sz_rest + (-1,)), invertdata

    def invert(self, data: np.ndarray, do_sampling: bool = True, to_discretize: np.ndarray | None = None, use_to_discretize: bool = False) -> np.ndarray:
        """ Unbins the data.

        Args:
            data: Binned data, (n_agents,  n_frames, n_features * n_bins) float array
            or (n_frames, n_features * n_bins) float array
            do_sampling: If True, samples from the probability distribution given by the bins.
                Otherwise takes the weighted average.
                TODO: Have an option to take argmax
                TODO: Check out pose.py and compare the random sampling scheme.
                TODO: Add an option to blur the prediction?

        Returns:
            continuous: Continuous data, (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array
        """
        ismultiagent = data.ndim == 3
        if not ismultiagent:
            data = data[None, ...]
        n_agents, n_frames, n_bin_feat = data.shape
        n_bins = self.bin_centers.shape[-1]
        n_feat = n_bin_feat // n_bins
        if use_to_discretize:
            assert to_discretize is not None, "to_discretize must be provided if use_to_discretize is True"
            continuous = to_discretize.copy()
        elif do_sampling:
            continuous = sample_discrete_labels(data.reshape((n_agents, n_frames, n_feat, n_bins)), self.bin_samples)[0]
        else:
            continuous = labels_discrete_to_continuous(data.reshape((n_agents, n_frames, n_feat, n_bins)), self.bin_centers)
        if not ismultiagent:
            continuous = continuous[0]

        return continuous
    
    @property
    def nbins(self):
        """
        Returns the number of bins.
        """
        return self.bin_centers.shape[-1]
    
    def unflatten(self,data: Data | np.ndarray | torch.Tensor) -> np.ndarray:
        """
        array_unflattened = discretize_op.unflatten(data)
        Returns the data array unflattened to (n_agents, n_frames, n_features, n_bins).
        """
        if isinstance(data, Data):
            data = data.array
        return data.reshape(data.shape[:-1] + (-1,self.nbins))
    
    def __str__(self):
        return f"Operation {self.name} of class Discretize with {self.nbins if self.bin_centers is not None else None} bins"
    
    def update_feature_names(self, input_feature_names):
        output_feature_names = np.empty((len(input_feature_names), self.nbins), dtype=object)
        for i,name in enumerate(input_feature_names):
            for b in range(self.nbins):
                output_feature_names[i, b] = f'{name}_bin{b}'
        return output_feature_names.flatten().tolist()
    
    def invert_feature_names(self, input_feature_names):
        nfeat = len(input_feature_names) // self.nbins
        output_feature_names = []
        for i in range(nfeat):
            output_feature_names.append(input_feature_names[i*self.nbins].replace(f'_bin0',''))
        return output_feature_names

@dataclass
class Fusion(Operation):
    """ Apply different operations to different parts of the data.

    Atrributes:
        operations: List of operations to be applied
        indices_per_op: list of indices to apply each operation to.
        dims_per_op: list of output dimensions corresponding to each operation, used for inverting.
    """
    
    localattrs = ['dims_per_op']
    operations: list[Operation]
    indices_per_op: list[np.ndarray]
    # This variable will hold the dimensions of the output corresponding to each operation, used for inverting.
    dims_per_op: list[int] | None = None

    def __post_init__(self):
        super().__post_init__()
        assert len(self.operations) == len(self.indices_per_op), "List of indices should have same length as list of operations."

        all_indices = np.concatenate(self.indices_per_op)
        assert len(all_indices) == max(all_indices) + 1 and len(all_indices) == len(np.unique(all_indices)), \
            "Indices must cover all feature dimensions and each dimension can only be provided to one operation."

    def apply(self, data: np.ndarray, kwargs_per_op: list[dict] | dict | None = None) -> np.ndarray:
        """ Applies operation to each subset of data, specified by indices per operation, and concatenates the result.

        Args:
            data: (n_agents,  n_frames, n_features) float array or (n_frames, n_features) float array
            kwargs_per_op: Optional arguments provided to the operations.
                If they are a list, the list should have the same length as the list of operations.
                If they are not a list, the same arguments will be provided to all operations.
                If None, no arguments will be provided to the operations.

        Returns:
            fused: (n_agents,  n_frames, n_fused_features) float array or 
            (n_frames, n_fused_features) float array
        """
        ismultiagent = data.ndim == 3
        if not ismultiagent:
            data = data[None, ...]

        if kwargs_per_op is None:
            kwargs_per_op = [{} for _ in self.operations]
        elif not isinstance(kwargs_per_op, list):
            kwargs_per_op = [kwargs_per_op for _ in self.operations]
        processed = []
        invertdata = []
        for op, indices, kwargs in zip(self.operations, self.indices_per_op, kwargs_per_op):
            kwargs = {k: v for k, v in kwargs.items() if k in get_optional_params(op.apply)}
            res = op.apply(data[..., indices], **kwargs)
            if isinstance(res, tuple):
                processed.append(res[0])
                invertdata.append(res[1])
            else:
                processed.append(res)
                invertdata.append({})
        self.dims_per_op = [proc.shape[-1] for proc in processed]
        fused = np.concatenate(processed, axis=-1)
        
        if not ismultiagent:
            fused = fused[0]

        return fused, {'kwargs_per_op': invertdata}

    def invert(self, data: np.ndarray, kwargs_per_op=None, **extraargs_per_op) -> np.ndarray:
        """ Inverts subsets of the processed data using the operation inverses.

        Args:
            data: (n_agents,  n_frames, n_fused_features) float array or
            (n_frames, n_fused_features) float array

        Returns:
            unfused: (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array
        """
        ismultiagent = data.ndim == 3
        if not ismultiagent:
            data = data[None, ...]
            kwargs_per_op = [{k: v[None, ...] for k, v in kwargs.items()} for kwargs in kwargs_per_op] if kwargs_per_op is not None else None
        
        if kwargs_per_op is None:
            kwargs_per_op = [{} for _ in self.operations]
        elif not isinstance(kwargs_per_op, list):
            kwargs_per_op = [kwargs_per_op for _ in self.operations]
        n_agents, n_frames = data.shape[:2]
        n_feat = sum([len(indices) for indices in self.indices_per_op])

        inverted = np.zeros((n_agents, n_frames, n_feat))
        count = 0
        for i, indices in enumerate(self.indices_per_op):
            n_dims = self.dims_per_op[i]
            kwargs_curr = extraargs_per_op.get(self.operations[i].name, {})
            if len(kwargs_per_op) > i:
                kwargs_curr = kwargs_per_op[i] | kwargs_curr
            kwargs_curr = {k: v for k, v in kwargs_curr.items() if k in get_optional_params(self.operations[i].invert)}
            inverted[..., indices] = self.operations[i].invert(data[..., count:count + n_dims], **kwargs_curr)
            count += n_dims
            
        if not ismultiagent:
            inverted = inverted[0]

        return inverted
    
    def unfuse(self, data: Data | np.ndarray | torch.Tensor) -> np.ndarray:
        """
        array_parts = fusion_op.unfuse(array)
        Returns a dict with a key for each operation. The value for each key is the part of the data
        corresponding to that operation.
        Args:
            data: Either np.ndarray or torch.Tensor or Data
        Returns:
            res: dict with keys being operation names and values being the corresponding parts of the data.
        """
        if isinstance(data, Data):
            data = data.array
        res = {}
        count = 0
        for i, indices in enumerate(self.indices_per_op):
            opname = self.operations[i].name
            n_dims = self.dims_per_op[i]
            res[opname] = data[..., count:count + n_dims]
            count += n_dims
        
        return res
        
    def unfuse_feature_names(self, data: Data) -> dict[str, list[str]]:
        """
        feature_names_per_op = fusion_op.unfuse_feature_names(feature_names)
        Returns a dict with a key for each operation. The value for each key is the list of feature names
        corresponding to that operation.
        Args:
            feature_names: list of feature names corresponding to the fused data. 
        Returns:
            res: dict with keys being operation names and values being the corresponding feature names.
        """
        if data.feature_names is None:
            return None
        res = {}
        count = 0
        for i, indices in enumerate(self.indices_per_op):
            opname = self.operations[i].name
            n_dims = self.dims_per_op[i]
            res[opname] = data.feature_names[count:count + n_dims]
            count += n_dims
        return res
        
    def get_indices_for_operation(self, operation_name: str) -> np.ndarray | None:
        """
        indices = fusion_op.get_indices_for_operation(operation_name)
        Returns the indices corresponding to the given operation name.
        Args:
            operation_name: Name of the operation.
        Returns:
            indices: np.ndarray of indices corresponding to the operation, or None if not found.
        """
        for i, op in enumerate(self.operations):
            if op.name == operation_name:
                return self.indices_per_op[i]
        return None
    
    def __str__(self):
        s = f"Operation {self.name} of class Fusion with operations:\n"
        for i, op in enumerate(self.operations):
            s += f"  {i}: {str(op)}, indices: {self.indices_per_op[i]}\n"
        return s[:-1]
    
    def update_feature_names(self, input_feature_names):
        output_feature_names = []
        for i, op in enumerate(self.operations):
            indices = self.indices_per_op[i]
            input_names_op = [input_feature_names[j] for j in indices]
            output_names_op = op.update_feature_names(input_names_op)
            output_feature_names.extend(output_names_op)
        return output_feature_names
    
    def invert_feature_names(self, input_feature_names):
        nfeat = max([max(indices) for indices in self.indices_per_op]) + 1
        output_feature_names = [f'feat_{i}' for i in range(nfeat)]
        count = 0
        for i, op in enumerate(self.operations):
            indices = self.indices_per_op[i]
            n_dims = self.dims_per_op[i]
            input_names_op = input_feature_names[count:count + n_dims]
            output_names_op = op.invert_feature_names(input_names_op)
            for j, name in zip(indices, output_names_op):
                output_feature_names[j] = name
            count += n_dims
        return output_feature_names

@dataclass
class Roll(Operation):
    """ Rolls data by delta time.

    This is useful e.g. for providing velocities as input. If we have a pose at time t and pose velocity corresponding
    to time t + dt, then we roll the data forward by dt makes it so that the value at index t represents velocity of
    pose from t - dt to t.

    Attributes::
        dt: How much to roll the data by (e.g. value used to compute the velocity data).
    """

    dt: int = 1
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """ Rolls data forward by dt.

        Args:
            data: (n_agents,  n_frames, n_features) float array or (n_frames, n_features) float array

        Returns:
            rolled_data: (n_agents,  n_frames, n_features) float array or (n_frames, n_features) float array
        """
        ismultiagent = data.ndim == 3
        if not ismultiagent:
            data = data[None, ...]
        rolled = np.roll(data, shift=self.dt, axis=1)
        if not ismultiagent:
            rolled = rolled[0]
        return rolled

    def invert(self, data: np.ndarray) -> np.ndarray:
        """ Rolls data backwards by dt.

        Args:
            data: (n_agents,  n_frames, n_features) float array

        Returns:
            unrolled_data: (n_agents,  n_frames, n_features) float array
        """
        ismultiagent = data.ndim == 3
        if not ismultiagent:
            data = data[None, ...]
        unrolled_data = np.roll(data, shift=-self.dt, axis=1)
        if not ismultiagent:
            unrolled_data = unrolled_data[0]
        return unrolled_data

    def __str__(self):
        return f"Operation {self.name} of class Roll with dt {self.dt}"
    
    def update_feature_names(self, input_feature_names):
        return [f'{name}_rolled{self.dt}' for name in input_feature_names]
    
    def invert_feature_names(self, input_feature_names):
        return [name.replace(f'_rolled{self.dt}','') for name in input_feature_names]

def multistart_cumsum(x: np.ndarray, x0s: list[np.ndarray], t0s: list[np.ndarray], dt=None) -> np.ndarray:
    """ Computes cumulative sum of x with multiple starting points.

    Args:
        x: (n_agents, n_frames, ...) float array
        x0s: list of arrays for each agent of shape (n_t0s, dt, ...) if dt is not None, else (n_t0s, ...)
        t0s: list of arrays for each agent of shape (n_t0s,)
        dt: time difference between x0s and first value in x to be summed. if None, assumed to be 1

    Returns:
        cumsum_x: (n_agents, n_frames, ...) float array
    """
        
    n_agents = x.shape[0]
    n_frames = x.shape[1]
    szrest = x.shape[2:]
    
    if dt is None:
        cumsum_x = np.full((n_agents, n_frames+1) + szrest, np.nan)
        for agent_id in range(n_agents):
            t0scurr = t0s[agent_id]
            t1scurr = np.r_[t0scurr[1:], n_frames] # there could be some nans in here but cumsum will propagate them
            x0scurr = x0s[agent_id]
            for t0,t1,x0 in zip(t0scurr, t1scurr, x0scurr):
                cumsum_x[agent_id, t0:t1+1] = np.cumsum(np.concatenate([x0[None],x[agent_id, t0:t1]], axis=0),axis=0)
        return cumsum_x[:, :-1]

    # example if dt = 2:
    # [y0,y1,y2,y3,y4,...]
    # x = [y2-y0, y3-y1, y4-y2, ...]
    # cumsum_x = [y0 + x[0], y1 + x[1], y0 + x[0] + x[2], y1 + x[1] + x[3], ...]
    # and x0s[:,i] corresponds to y_i for i in [0, dt-1]

    cumsum_x = np.full((n_agents, n_frames) + szrest, np.nan)    
    t0s_sub = [t0s_agent//dt for t0s_agent in t0s]
    for i in range(dt):
        cumsum_x[:,i::dt] = multistart_cumsum(x[:,i::dt], [x0s_agent[:,i] for x0s_agent in x0s], t0s_sub)
    return cumsum_x

@dataclass
class LocalVelocity(Operation):
    """ Computes the relative pose movement from t to t + 1.

    Attributes:
        is_angle: Bool vector indicating whether pose index is an angle measurement, used to wrap angles
            into [-pi, pi] range when inverting the operation.
    """
    is_angle: np.ndarray | None = None
    
    def apply(self, pose: np.ndarray, isstart: np.ndarray | None = None) -> np.ndarray:
        """ Compute velocity from pose.

        Args:
            pose: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
            isstart: Indicates whether a new fly track starts at a given frame for an agent.
                (n_agents,n_frames) bool array or (n_frames,) bool array

        Returns:
            velocity: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
        """
        ismultiagent = pose.ndim == 3
        assert isstart is None or (isstart.ndim == pose.ndim - 1), "isstart.ndim must be pose.ndim - 1"
        if not ismultiagent:
            pose = pose[None, ...]
            if isstart is not None:
                isstart = isstart[None, ...]
                
        pose_velocity = np.moveaxis(compute_relpose_velocity(pose.T, is_angle=self.is_angle), 2, 0)
        if isstart is not None:
            # Set pose deltas that cross individuals to NaN.
            set_invalid_ends(pose_velocity, isstart.T, dt=1)
        pose_velocity = pose_velocity[0]
        pose_velocity = pose_velocity.T
        
        if not ismultiagent:
            pose_velocity = pose_velocity[0, ...]
                        
        return pose_velocity, {'pose': pose, 'isstart': isstart}

    def invert(self, velocity: np.ndarray, x0 : np.ndarray | None = None, 
               pose: np.ndarray | None = None, isstart: np.ndarray | None = None) -> np.ndarray:
        """ Compute pose from pose velocity and an initial pose.

        Args:
            velocity: Delta pose (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
            x0: Initial pose (n_agents, n_pose_features) float array or (n_pose_features) float array, or None. Defaults to None. 
            x0s: Initial poses at t0s frames, list of arrays for each agent of shape (n_t0s, n_pose_features), or None. Defaults to None.
                x0 takes priority to x0s if both are provided. t0s must also be provided if x0s is provided.
            t0s: List of arrays indicating the frames where new tracks start for each agent, or None. Defaults to None. 
                x0s must also be provided if t0s is provided.

        Returns:
            pose: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
        """
        ismultiagent = velocity.ndim == 3
        if not ismultiagent:
            velocity = velocity[None, ...]
            if x0 is not None:
                x0 = x0[None, ...]
        
        # Note: here we are assuming dt=1
        # single x0 for all frames
        if (x0 is not None) or (pose is None):
            if x0 is None:
                raise ValueError("x0 or pose must be provided to invert LocalVelocity")
                # n_agents, _, n_features = velocity.shape
                # x0 = np.zeros((n_agents, n_features))
            velocity = np.concatenate([x0[:, None, :], velocity], axis=1)[:, :-1, :]
            pose = np.cumsum(velocity, axis=1)
        else:
            # x0s at multiple frames
            t0s = get_velocity_t0s(pose, isstart)
            x0s = [pose[agent_id, t0scurr, :] for agent_id, t0scurr in enumerate(t0s)]        
            pose = multistart_cumsum(velocity, x0s, t0s)
            
        if self.is_angle is not None:
            pose[..., self.is_angle] = modrange(pose[..., self.is_angle], -np.pi, np.pi)
            
        if not ismultiagent:
            pose = pose[0, ...]
            
        return pose
    
    def __str__(self):
        return f"Operation {self.name} of class LocalVelocity with is_angle {self.is_angle}"
    
    def update_feature_names(self, input_feature_names):
        return [f'{name}_velocity' for name in input_feature_names]
    
    def invert_feature_names(self, input_feature_names):
        return [name.replace('_velocity','') for name in input_feature_names]
    
def get_velocity_t0s(position: np.ndarray, isstart: np.ndarray | None) -> list[np.ndarray]:
    isdata = ~np.all(np.isnan(position),axis=-1)
    n_agents = isdata.shape[0]
    isstart1 = np.c_[np.ones(n_agents,dtype=bool), isdata[:,1:] & ~isdata[:,:-1]]
    if isstart is not None:
        isstart1 = isstart | isstart1
    agent_ids,ts = np.nonzero(isstart1 & isdata)
    
    t0s = [ts[agent_ids==agent_id] for agent_id in range(n_agents)]
    return t0s


@dataclass
class GlobalVelocity(Operation):
    """ Computes the global movement of an agent,  dfwd, dside, dtehta, from its (x, y, theta) position.

    Attributes:
        tspred: A list of dt for which global movement (from t to t+dt) will be computed for.
    """
    tspred: list[int]

    def apply(self, position: np.ndarray, isstart: np.ndarray | None = None) -> np.ndarray:
        """ Compute global movement from position.

        Args:
            position: (n_agents,  n_frames, 3) float array or (n_frames, 3) float array
            isstart: Indicates whether a new fly track starts at a given frame for an agent.
                (n_agents, n_frames) bool array or (n_frames,) bool array

        Returns:
            velocity: Global pose velocity, flattened over tspred.
                (n_agents,  n_frames, 3 * len(self.tspred)) float array or (n_frames, 3 * len(self.tspred)) float array
        """
        ismultiagent = position.ndim == 3
        assert isstart is None or (isstart.ndim == position.ndim - 1), "isstart.ndim must be position.ndim - 1"
        if not ismultiagent:
            position = position[None, ...]
            if isstart is not None:
                isstart = isstart[None, ...]
        
        Xorigin = position[..., :2].T
        Xtheta = position[..., 2].T
        _, n_frames, n_flies = Xorigin.shape
        # dXoriginrel: (len(tspred), 2, n_frames, n_agents) 
        # dtheta: (len(tspred), n_flies, n_frames)
        dXoriginrel, dtheta = compute_global_velocity(Xorigin, Xtheta, self.tspred)
        movement_global = np.concatenate((dXoriginrel[:, [1, 0]], dtheta[:, None, :, :]), axis=1)
        if isstart is not None:
            for movement, dt in zip(movement_global, self.tspred):
                set_invalid_ends(movement, isstart.T, dt=dt)
        # ([forward[dt1, sideways[dt1], dtheta[dt1], forward[dt2], sideways[dt2], dtheta[dt2], ...], n_frames, n_agents)
        movement_global = movement_global.reshape((-1, n_frames, n_flies)) 
        # n_agents, n_frames, n_features        
        movement_global = movement_global.T
        
        if not ismultiagent:
            movement_global = movement_global[0, ...]

        return movement_global, {'pose': position, 'isstart': isstart}

    def invert(self, velocity: np.ndarray, x0: np.ndarray | None = None,
               pose: np.ndarray | None = None, isstart: np.ndarray | None = None, dt: int | None = None) -> np.ndarray:
        """ Compute position from global movement and an initial position.

        NOTE: This assumes velocity is only given for dt=1

        Args:
            velocity: Global movmement (n_agents,  n_frames, 3) float array or (n_frames, 3) float array
            x0: Initial pose (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array.
                Defaults to None.
            x0s: Initial poses at t0s frames, list of arrays for each agent of shape (n_t0s, n_pose_features), or None. Defaults to None.
                x0 takes priority to x0s if both are provided. t0s must also be provided if x0s is provided.
            t0s: List of arrays indicating the frames where new tracks start for each agent, or None. Defaults to None. 
                x0s must also be provided if t0s is provided.
            dt: which tspred to use for inversion. If None, uses the smallest dt in self.tspred.

        Returns:
            pose: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
        """
                
        # invert with smallest dt
        if dt is None:
            dt = np.min(self.tspred)
        else:
            assert dt in self.tspred, "dt must be in tspred"
        
        ismultiagent = velocity.ndim == 3
        if not ismultiagent:
            velocity = velocity[None, ...]
            if x0 is not None:
                x0 = x0[None, ...]
        
        if (x0 is not None) or (pose is None):
            if x0 is None:
                raise ValueError("x0 or pose must be provided to invert GlobalVelocity")
                # n_agents, _, n_dim = velocity.shape
                # x0 = np.zeros((n_agents, n_dim))

            d_theta = np.concatenate([x0[:, None, 2], velocity[:, :, 2]], axis=1)
            theta = np.cumsum(d_theta, axis=1)[:, :-1]
        else:
            
            t0s = get_velocity_t0s(pose, isstart)
            # x0s[agent_id] is (len(t0s[agent_id]),dt,3)
            x0s = [pose[agent_id, t0scurr[:,None]+np.arange(dt)[None,:]] for agent_id,t0scurr in enumerate(t0s)]
            
            theta = multistart_cumsum(velocity[..., 2], [x0s_agent[..., 2] for x0s_agent in x0s], t0s, dt=dt)
        theta = modrange(theta, -np.pi, np.pi)

        d_pos_rel = velocity[..., [1, 0]]
        d_pos = rotate_2d_points(d_pos_rel.transpose((1, 2, 0)), -theta.T).transpose((2, 0, 1))
        if x0 is not None:
            d_pos = np.concatenate([x0[:, None, :2], d_pos], axis=1)
            pos = np.cumsum(d_pos, axis=1)[:, :-1, :]
        else:
            pos = multistart_cumsum(d_pos, [x0s_agent[..., :2] for x0s_agent in x0s], t0s, dt=dt)
        
        inverted = np.concatenate([pos, theta[:, :, None]], axis=-1)
        
        if not ismultiagent:
            inverted = inverted[0, ...]
        
        return inverted
    
    def update_feature_names(self, input_feature_names):
        return [name for dt in self.tspred for name  in [f'forward_velocity_{dt}', f'sideways_velocity_{dt}', f'angular_velocity_{dt}']]
    
    def invert_feature_names(self, input_feature_names):
        output_feature_names = ['x_position', 'y_position', 'orientation']
        return output_feature_names

    def __str__(self):
        return f"Operation {self.name} of class GlobalVelocity with tspred {self.tspred}"

# no dataclass decorator, directly defined __init__
class Velocity(Operation):
    """ Combines global and local velocity.

    Attributes:
        featrelative: Bool vector indicating whether pose index is a relative feature.
        featangle: Bool vector indicating whether pose index is an angle feature.
    """

    localattrs = ['global_inds', 'local_inds', 'fusion']
    
    def __init__(self, featrelative: np.ndarray, featangle: np.ndarray = None):
        # Keep track of global and local indices for the pose features
        self.global_inds = np.where(~featrelative)[0]
        self.local_inds = np.where(featrelative)[0]
        is_angle = featangle[featrelative] if featangle is not None else None
        # Use Fusion to apply global vs local operations to the relevant feature dimensions
        self.fusion = Fusion(
            operations=[GlobalVelocity(tspred=[1]), LocalVelocity(is_angle=is_angle)],
            indices_per_op=[self.global_inds, self.local_inds]
        )
        super().__post_init__()

    def apply(self, pose: np.ndarray, isstart: np.ndarray | None = None, kwargs_per_op: list | None = None):
        """ Compute global and local velocity from pose.

        Args:
            pose: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
            isstart: Indicates whether a new fly track starts at a given frame for an agent.
                (n_agents, n_frames) bool array or (n_frames,) bool array

        Returns:
            velocity: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
        """
        if kwargs_per_op is None:
            kwargs_per_op = [{'isstart': isstart}, {'isstart': isstart}]
        res = self.fusion.apply(pose, kwargs_per_op=kwargs_per_op)
        return res

    def invert(self, velocity: np.ndarray, kwargs_per_op: list | None = None):
        """ Compute pose from pose velocity and an initial pose.

        Args:
            velocity: Delta pose (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
            x0: Initial pose (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array

        Returns:
            pose: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
        """
        return self.fusion.invert(velocity, kwargs_per_op=kwargs_per_op)
    
    def __str__(self):
        return f"Operation {self.name} of class Velocity => {str(self.fusion)}"
    
    def update_feature_names(self, input_feature_names):
        return self.fusion.update_feature_names(input_feature_names)
    
    def invert_feature_names(self, input_feature_names):
        return self.fusion.invert_feature_names(input_feature_names)


@dataclass(frozen=True)
class Session:
    """This can be used to index the loaded raw data for chunking.

    Note: We could let this contain data to map back to the original video
          but that is not necessary for training. If we have the loaded
          raw data and this we can reconstruct that if needed.
    """
    start_frame: int
    duration: int
    agent_id: int


### Dataset helper functions

def compute_sessions(datas: list[Data], isstart: np.ndarray) -> list[Session]:
    """ Extracts intervals of data belonging to a unique agent with valid data.

    Args:
        datas: A list of Data, can be a combination of binned and continuous.
        isstart: (n_agents, n_frames)

    Returns:
        sessions: A list of sessions indicating start_frame, duration, and agent_id of valid
            data intervals from a unique agent.
    """
    max_n_agents, n_frames = datas[0].array.shape[:2]

    # Sum over all feature dimensions of all datas to determine nans
    sum_data = datas[0].array.sum(-1)
    for i in range(1, len(datas)):
        sum_data = sum_data + datas[i].array.sum(-1)
    nans = np.isnan(sum_data)

    sessions = []
    for agent_id in range(max_n_agents):
        start_frames = np.where(isstart[agent_id])[0]
        durations = np.diff(list(start_frames) + [n_frames])
        for start_frame, duration in zip(start_frames, durations):
            frames = np.arange(start_frame, start_frame + duration)
            # Find all intervals of valid frames
            conncomps = connected_components(~nans[agent_id, frames])
            for comp in conncomps:
                sessions.append(
                    Session(
                        start_frame=frames[comp[0]],
                        duration=comp[1] - comp[0],
                        agent_id=agent_id,
                    )
                )

    return sessions


def compute_chunk_indices(sessions: list[Session], chunk_length: int, start_offset: int = 0,
                          useoutputmask: np.ndarray | None = None, stride: int | None = None) -> np.ndarray:
    """ Extracts chunk indices from session data, with chunks non-overlapping.

    Args:
        sessions: A list of sessions indicating start_frame, duration, and agent_id of valid
            data intervals from a unique agent.
        chunk_length: Desired length of chunk.
        start_offset: Index of first frame to be used.
        useoutputmask: (n_frames, n_agents) bool array indicating whether output data is valid.
            If provided, only chunks that have at least one valid output frame will be kept.
        stride: Interval between the start frames of consecutive chunks.
            If None, defaults to chunk_length (non-overlapping chunks).

    Returns:
        chunk_indices: (n_chunks, 2) int array, each row contains (start_frame, agent_id)
    """

    if stride is None:
        stride = chunk_length

    chunk_indices = []
    for session in sessions:
        t0 = session.start_frame + start_offset
        t1 = session.start_frame + session.duration - chunk_length + 1

        # If useoutputmask is provided, only keep chunks that have at least one valid output frame
        if useoutputmask is None or np.all(useoutputmask[t0:t1+chunk_length-1, session.agent_id]):
            start_frames = np.arange(t0, t1, stride)
        else:
            start_frames = []
            for t in range(t0, t1, stride):
                if np.any(useoutputmask[t:t + chunk_length, session.agent_id]):
                    start_frames.append(t)
            if len(start_frames) == 0:
                continue
            start_frames = np.array(start_frames)

        session_chunks = np.zeros((len(start_frames), 2), dtype=int)
        session_chunks[:, 0] = start_frames
        session_chunks[:, 1] = session.agent_id
        chunk_indices.append(session_chunks)
    return np.concatenate(chunk_indices, axis=0)

def get_array_chunk(datas: np.ndarray | torch.Tensor | dict,
                    start_frame: int,
                    agent_id: int,
                    duration: int) -> np.ndarray | dict[str, np.ndarray | torch.Tensor]:
    """ Extracts data from ndarray/Tensor for given chunk indices.

    Args:
        datas: dict to extract data from. if dict of dicts, calls recursively on each item.
        start_frame: Start frame of the chunk
        agent_id: Agent id of the chunk
        duration: Number of frames in the chunk

    Returns:
        data_chunk: dict with deepest entries as (chunk_length, n_feat) float array
    """    
    if isinstance(datas,dict):
        # If datas is a dict, call recursively on each item
        return {key: get_array_chunk(value, start_frame, agent_id, duration) for key, value in datas.items()}
    return datas[agent_id, start_frame:(start_frame + duration)]

def get_data_chunk(
        datas: dict[str, Data],
        start_frame: int | None = None,
        agent_id: int | None = None,
        duration: int | None = None
) -> np.ndarray | torch.Tensor:
    """ Extracts and concatenates data for given chunk indices.

    Args:
        datas: dict of data to extract from
        start_frame: Start frame of the chunk
        agent_id: Agent id of the chunk
        duration: Number of frames in the chunk

    Returns:
        data_chunk: (chunk_length, n_feat_flat), float array
    """
    subindex = [slice(None), slice(None)]
    if agent_id is not None:
        subindex[0] = agent_id
    if start_frame is not None and duration is not None:
        subindex[1] = slice(start_frame, start_frame + duration)
    slices = [data.array[*subindex] for data  in datas.values()]

    if isinstance(slices[0], np.ndarray):
        return np.concatenate(slices, axis=-1)
    else:
        return torch.cat(slices, dim=-1)


def get_bin_indices(datas: dict[str, Data]) -> tuple[list[np.ndarray], list[int]]:
    """ Finds data that have been discretized in the last operation applied to them.

    TODO: Perhaps it would be cleaner to store this information in Data (that way it doesn't
      have to be the last operation that did the binning, and could be done in other ways than
      using Discretize or Fusion).

    Args:
        datas: A dictionary of data where keys represent data names.

    Returns:
        bindices: Indices of feature dimensions that are bins, per data in datas
        n_bins: number of bins used for binning data in datas (note that this assumes only one Discretize operation
            if binned within Fusion).
    """
    dims_per_key = [data.array.shape[-1] for data in datas.values()]
    start_per_key = np.cumsum([0] + dims_per_key)
    bindices = []
    n_bins = []
    for i, data in enumerate(datas.values()):
        last_op = data.operations[-1]
        i0 = start_per_key[i]
        i1 = i0 + dims_per_key[i]
        indices = np.arange(i0, i1)
        if isinstance(last_op, Discretize):
            n_bins.append(last_op.bin_centers.shape[-1])
            bindices.append(indices)
        elif isinstance(last_op, Fusion):
            start_per_op = np.cumsum([0] + last_op.dims_per_op)[:-1]
            for j, op in enumerate(last_op.operations):
                if isinstance(op, Discretize):
                    j0 = start_per_op[j]
                    j1 = j0 + last_op.dims_per_op[j]
                    n_bins.append(op.bin_centers.shape[-1])
                    bindices.append(indices[j0:j1])
    return bindices, n_bins


def split_discr_cont(data: np.ndarray, bin_indices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """ Splits data into its discrete and continuous parts.

    Args:
        data: (n_agents,  n_frames, n_features) float array
        bin_indices: Indices of feature dimensions that are bineed (obtained from get_bin_indices).

    Returns:
        discrete_data: (n_agents,  n_frames, n_discrete_features) float array
        continuous_data: (n_agents,  n_frames, n_continuous_features) float array
            Note: n_discrete_features + n_continuous_features = n_features
    """
    is_binned = np.zeros(data.shape[-1], bool)
    for inds in bin_indices:
        is_binned[inds] = True
    return data[..., is_binned], data[..., ~is_binned]


def get_operation(
        operations: list[Operation], name: str, return_idx: bool = False, recursive: bool = False
) -> Operation | None | tuple[ Operation | None, int | None | tuple ]:
    """ Find operation from a list of operation, given its name.

    Args:
        operations: List of operations to search from.
        name: Name of operation to find.
        return_idx: Whether to also return the index of the operation. Default is False.
        recursive: Whether to search recursively within Fusion operations. Default is False.

    Returns:
        operation: Operation corresponding to the requested name, None if not found in the list.
        idx: If return_idx is True, also returns the index of the operation within the list.
    """
    for i, oper in enumerate(operations):
        if oper.name == name.lower():
            if return_idx:
                return oper, i
            return oper
    if recursive:
        for i, oper in enumerate(operations):
            if hasattr(oper, 'operations'):
                suboper, subidx = get_operation(oper.operations, name, return_idx=True, recursive=True)
                if suboper is not None:
                    if return_idx:
                        if not isinstance(subidx, tuple):
                            subidx = (subidx, )
                        return suboper, (i, ) + subidx
                    return suboper
    if return_idx:
        return None, None
    return None

def get_operation_index(
        operations: list[Operation], name: str | None = None, recursive: bool = False, data: Data | None = None) -> int | None:
    """ Find index of operation from a list of operation, given its name.

    Args:
        operations: List of operations to search from.
        name: Name of operation to find.
        recursive: Whether to search recursively within Fusion operations. Default is False.
    Returns:
        idx: Index of the operation within the list, None if not found.
    """
    if name is not None:
        _, idx = get_operation(operations, name, return_idx=True, recursive=recursive)
    elif data is not None:
        if not isinstance(data, Data):
            idx = 0
        else:
            # check that data.operations is a prefix of operations
            if len(data.operations) > len(operations):
                raise ValueError("data.operations is not a prefix of operations")
            for i in range(len(data.operations)):
                if isinstance(operations[i], Operation):
                    match = data.operations[i] == operations[i]
                elif isinstance(operations[i], dict):
                    match = (data.operations[i].name == operations[i]['attributes']['name']) and \
                        (data.operations[i].__module__ == operations[i]['module'])
                else:
                    raise ValueError("operations must be a list of Operation or dict")
                if not match:
                    raise ValueError("data.operations is not a prefix of operations")
            idx = len(data.operations) - 1        
    else:
        raise ValueError("Either name or data must be provided")
    return idx

def get_post_operations(operations: list[Operation], name: str | None = None, data: Data | None = None) -> list[Operation] | None:
    """ Get a list of operations that come after the operation with either:
    - the given name, if name is not None
    - operations already applied to data, if name is None and data is not None

    Args:
        operations: List of operations to search from.
        name: Name of operation to find if non-None. Can be None if data is provided. Default is None.
        data: Data object to find operations after if name is None. data.operations must be the start of operations, 
        and this returns the operations after. Default is None.
    
    Returns:
        A list of operations following the input prefix.
    """
    if name is not None:
        _, idx = get_operation(operations, name, return_idx = True)
    elif data is not None:
        if not isinstance(data, Data):
            idx = 0
        else:
            # check that data.operations is a prefix of operations
            if len(data.operations) > len(operations):
                raise ValueError("data.operations is not a prefix of operations")
            for i in range(len(data.operations)):
                if isinstance(operations[i], Operation):
                    match = data.operations[i] == operations[i]
                elif isinstance(operations[i], dict):
                    match = (data.operations[i].name == operations[i]['attributes']['name']) and \
                        (data.operations[i].__module__ == operations[i]['module'])
                else:
                    raise ValueError("operations must be a list of Operation or dict")
                if not match:
                    raise ValueError("data.operations is not a prefix of operations")
            idx = len(data.operations) - 1
    else:
        raise ValueError("Either name or data must be provided")
    if idx is None:
        return None
    return operations[idx + 1:]

def get_pre_operations(operations: list[Operation], name: str, inclusive: bool = False) -> list[Operation] | None:
    """ Get a list of operations that come before the operation with the given name.

    Args:
        operations: List of operations to search from.
        name: Name of operation to find.
        inclusive: Whether to include the operation with the given name in the returned list. Default is False.

    Returns:
        A list of operations preceding the input prefix.
    """
    _, idx = get_operation(operations, name, return_idx = True)
    if idx is None:
        return None
    if inclusive:
        idx += 1
    return operations[:idx]

def guess_timedim(data: np.ndarray | torch.Tensor | Data) -> int:
    """ Guess which dimension of data is the time dimension.

    Args:
        data: Data to guess time dimension from. Can be ndarray, Tensor, or Data object.    
    Returns:
        timedim: Index of time dimension.
    """
    if data.ndim >= 3:
        # assume (n_agents, n_frames, n_features)
        timedim = 1
    else:
        # assume (n_frames, n_features)
        timedim = 0
    return timedim

def apply_operations(data: np.ndarray | torch.Tensor | Data, 
                     operations: list[Operation],
                     invertdata: list | None = None,
                     extraargs: dict = {},
                     use_prev_invertdata: bool | None = None,
                     start_frame: int | None = None,
                     timedim: int | None = None) -> Data:
    """ Apply a list of operations to data.
    """
    
    if use_prev_invertdata is None:
        use_prev_invertdata = (invertdata is None) and (len(extraargs) == 0)
    
    if invertdata is None:
        invertdata = [None,] * len(operations)
        
    for oper,invertdataargs in zip(operations, invertdata):
        if isinstance(oper, dict):
            oper = Operation.from_dict(oper)
            
        params = get_optional_params(oper.apply)
            
        kwargs = extraargs.get(oper.name, {})
        if invertdataargs is not None:
            if start_frame is not None:
                timedimcurr = timedim if timedim is not None else guess_timedim(data)
                duration = data.shape[timedimcurr]
                idx = (slice(None),)*timedimcurr + (slice(start_frame, start_frame + duration),)
                kwargs = {k: oper.invertdata_subindex(v,idx) for k, v in invertdataargs.items() if k in params} | kwargs                
            else: 
                kwargs = {k: v for k, v in invertdataargs.items() if k in params} | kwargs
            
        # add previous invertdata to kwargs
        if use_prev_invertdata and isData(data) and (len(data.invertdata) > 0) and (data.invertdata[-1] is not None):
            prevargs = data.invertdata[-1]
            kwargs = {k: v for k, v in prevargs.items() if k in params} | kwargs

        data = oper(data, **kwargs)

    return data


def apply_inverse_operations(data: np.ndarray | torch.Tensor | Data, 
                             operations: list | None = None, invertdata: dict | list | None = None,
                             extraargs: dict = {}, return_feature_names: bool = False):
    """ Apply the inverse of operations to data, in reverse order.
    
    Args: 
    data: Data to invert. Can be ndarray, Tensor, or Data object. If a Data object, operations and invertdata
        are extracted from it if not provided. 
    operations: List of operations to invert. If None, extracted from data. 
    invertdata: If dict, dictionary mapping operation names to arguments to provide to the invert method.
        If list, list of arguments to provide to the invert method, one per operation.
        If None, extracted from data.
    extraargs: Dict with extra arguments to provide to the invert methods of operations. Entries are operation name to
        dict of keyword arguments.
    
    Returns:
        array: Data inverted, ndarray or Tensor
        feature_names (optional): List of feature names after inversion. Only returned if return_feature_names is True.
    """
    
    # allow data to be a Data object, in which case we extract operations and invertdata from it
    if operations is None and hasattr(data, 'operations'):
        operations = data.operations
    if invertdata is None:
        if hasattr(data, 'invertdata'):
            invertdata = data.invertdata
        else:
            invertdata = [None] * len(operations)
    feature_names = None
    if hasattr(data,'feature_names'):
        feature_names = data.feature_names
    if not isinstance(data, (np.ndarray, torch.Tensor)) and hasattr(data, 'array'):
        data = data.array
    if feature_names is None:
        feature_names = [f'feat_{i}' for i in range(data.shape[-1])]
    
    # convert dict invertdata into list, and fill in from data.invertdata if available
    if isinstance(invertdata,dict):
        invertdata1 = []
        for i,oper in enumerate(operations):
            if oper.name in invertdata:
                invertdata1.append(invertdata[oper.name])
            elif hasattr(data,'invertdata'):
                invertdata1.append(data.invertdata[i])
            else:
                invertdata1.append(None)
        invertdata = invertdata1
                
    for oper,invertdataargs in reversed(list(zip(operations, invertdata))):
        kwargs = extraargs.get(oper.name, {})
        if invertdataargs is not None:
            kwargs = invertdataargs | kwargs
        # remove args from kwargs that are not in the invert method signature
        kwargs = {k: v for k, v in kwargs.items() if k in get_optional_params(oper.invert)}
        # elif isinstance(invertdataargs, (list, tuple)):
        #     args += list(invertdataargs)
        data = oper.invert(data, **kwargs)
        if return_feature_names:
            feature_names = oper.invert_feature_names(feature_names)

    if return_feature_names:
        return data, feature_names
    else:      
        return data

def invert_to_named(data: Data, name: str | list | tuple, return_data: bool = False, return_feature_names: bool = False, **kwargs) -> np.ndarray | torch.Tensor:
    """ Inverts data to the state after operation name has been applied. 

    Args:
        data: Data to invert.
        name: Name of operation to invert to.
        return_data: Whether to return a Data object or just the array. Default is False.
    Any extra args passed as extraargs to apply_inverse_operations.

    Returns:
        array: Data inverted to the state after operation name has been applied, ndarray or Tensor
    """
    
    if isinstance(name, (list, tuple)):
        rest = name[1:]
        name = name[0]
    else:
        rest = None
    
    if name == 'original':
        post_opers = data.operations
        post_invertdata = data.invertdata
    else:
        idx = get_operation_index(data.operations, name)
        post_opers = data.operations[idx+1:]
        if data.invertdata is None:
            post_invertdata = [None,]*len(post_opers)
        else:
            post_invertdata = data.invertdata[idx+1:]
        
    if post_opers is None:
        raise ValueError(f"Operation '{name}' not found in data operations")
    array, feature_names = apply_inverse_operations(data, operations=post_opers, invertdata=post_invertdata, extraargs=kwargs, return_feature_names=True)

    # unfuse etc
    while True:
    
        if rest is None or len(rest) == 0:
            break

        name1 = rest[0]
        rest = rest[1:]

        op = get_operation(data.operations, name1)
        if op is None:
            raise ValueError(f"Operation '{name1}' not found in data operations")
        if isinstance(op,Fusion):
            array = op.unfuse(array)
            feature_names = op.unfuse_feature_names(feature_names)
        else:
            raise ValueError(f"Unknown how to process operation '{name1}' of type {type(op)} after inversion")
    
    if return_data:
        if name == 'original':
            idx = 0
            operations = []
        else:
            idx = get_operation_index(data.operations,name)
            operations = data.operations[:idx+1]
        if data.invertdata is None:
            post_invertdata = [None,]*len(operations)
        else:
            post_invertdata = data.invertdata[:idx]
        return Data(name=name, array=array, operations=operations, invertdata=post_invertdata, feature_names=feature_names)

    if return_feature_names:
        return array, feature_names

    return array

def apply_opers_from_data(datas_ref: dict[str, Data], datas: dict, extraargs: dict = {}, 
                          use_data_invertdata: bool | None = None, use_prev_invertdata: bool | None = None,
                          separate_extraargs: bool = False, start_frame: int | None = None, timedim: int | None = None) -> dict[str, Data]:
    """ Applies post processing operations from reference datas to datas, for each key.

    Post-key-operations are all operations after the key of each data, for example for 'velocity' it doesn't
    apply the operation to compute pose from keypoints, but all operations following that (e.g. zscoring).

    This is useful for building a validation set from a training set, or for applying operations with the right
    parameters to data during simulation.

    TODO: This assumes that they key correspond to the name of the operation used to compute data (e.g. Velocity)
        as it uses the key to extract post-processing operations. It would be nicer to label the operations themselves
        so that we can drop this assumption.

    Params:
        datas_ref: Dictionary of data (e.g. train_dataset.inputs) from which to copy post processing operations.
        datas: Dictionary of raw data to which post processing operations should be applied to.

    Returns:
        Post processed datas.
    """
    # assert len(datas_ref.keys()) == len(datas.keys()), "The two data collections must have the same keys"
    if use_data_invertdata is None:
        use_data_invertdata = len(extraargs) == 0
    processed_data = {k: v for k, v in datas.items()}
    for key in datas_ref.keys():
        if key not in datas:
            continue
        if isinstance(datas[key], (np.ndarray, torch.Tensor)):
            opers = datas_ref[key].operations
            invertdata = datas_ref[key].invertdata
        else:
            idx = get_operation_index(datas_ref[key].operations, datas[key].operations[-1].name)
            opers = datas_ref[key].operations[idx+1:]
            invertdata = datas_ref[key].invertdata[idx+1:]
        if separate_extraargs:
            extraargscurr = extraargs.get(key, {})
        else:
            extraargscurr = extraargs
        processed_data[key] = apply_operations(datas[key], opers, 
                                               invertdata=invertdata if use_data_invertdata else None, 
                                               extraargs=extraargscurr,
                                               use_prev_invertdata=use_prev_invertdata,
                                               start_frame=start_frame,timedim=timedim)
    return processed_data


def apply_opers_from_data_params(data_params: list[dict], datas: dict[str, Data], check_match: bool = False, 
                                 invertdata=None, extraargs: dict = {}, 
                                 use_prev_invertdata: bool | None = None,
                                 separate_extraargs: bool = False) -> dict[str, Data]:
    """ Applies post processing operations from data_params (e.g. output of mydataset.get_params()['inputs'])
    to datas, for each key.

    Post-key-operations are all operations after the key of each data, for example for 'velocity' it doesn't
    apply the operation to compute pose from keypoints, but all operations following that (e.g. zscoring).

    This is useful for building a validation set from a training set, or for applying operations with the right
    parameters to data during simulation.

    Params:
        datas_ref: Dictionary of data (e.g. train_dataset.inputs) from which to copy post processing operations.
        datas: Dictionary of raw data to which post processing operations should be applied to.

    Returns:
        Post processed datas.
    """
    # assert len(datas_ref.keys()) == len(datas.keys()), "The two data collections must have the same keys"
    processed_data = {k: v for k, v in datas.items()}
    for key in data_params.keys():
        if key not in datas:
            if check_match:
                raise ValueError(f"Expect both data to have all of the same keys, but did not find '{key}' in datas")
            else:
                LOG.warning(f"Did not find '{key}' in datas, skipping")
                continue 
        if invertdata is not None:
            invertdatacurr = invertdata[key]
        else:
            invertdatacurr = None
        # input datacurr may already have some operations applied, so we need to find the last operation that was applied
        if isinstance(datas[key], (np.ndarray, torch.Tensor)):
            opers = data_params[key]
        else:
            idx = get_operation_index(data_params[key], data=datas[key])
            opers = data_params[key][idx+1:]
            if invertdatacurr is not None:
                invertdatacurr = invertdatacurr[idx+1:]
        if separate_extraargs:
            extraargscurr = extraargs.get(key, {})
        else:
            extraargscurr = extraargs
        processed_data[key] = apply_operations(datas[key], opers, invertdata=invertdatacurr, extraargs=extraargscurr,
                                               use_prev_invertdata=use_prev_invertdata)
    return processed_data



def collate_nested_dicts(batch):
    """Recursively collates nested dicts/arrays into batched tensors.

    For lists (e.g. per-operation invertdata), recurses element-wise so a
    `[[op0, op1, ...]_b for b in batch]` structure becomes
    `[batched_op0, batched_op1, ...]` — keeping the per-op axis intact and
    stacking the leaf arrays inside each op's dict. This lets
    `apply_inverse_operations` consume the batched Data directly.
    """
    if batch[0] is None and all(b is None for b in batch):
        return None
    if isinstance(batch[0], dict):
        return {k: collate_nested_dicts([d[k] for d in batch]) for k in batch[0]}
    elif isinstance(batch[0], list):
        n = len(batch[0])
        if not all(isinstance(b, list) and len(b) == n for b in batch):
            return batch
        return [collate_nested_dicts([b[i] for b in batch]) for i in range(n)]
    elif isinstance(batch[0], np.ndarray):
        return torch.stack([torch.from_numpy(x) for x in batch])
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch)
    elif isinstance(batch[0], (int, float, bool, np.integer, np.floating, np.bool_)):
        return torch.tensor(batch)
    else:
        return batch

class Dataset(torch.utils.data.Dataset):
    """ Contains ground truth data and can be supplied to torch's DataLoader to produce chunks of data.

    Args:
        inputs: A dictionary of data inputs. Each data in inputs has the following:
            array: (n_agents, n_frames, n_features) float array.
            operations: Operations that have been applied to arrive at this data.
        labels: A dictionary of data labels. Same format as inputs.
        isstart: Indicates whether a frame is the start of a sequence for an agent, (n_agents, n_frames) bool array
        context_length: Number of frames in a data chunk provided by __getitem__
        useoutputmask: (n_frames, n_agents) bool array indicating which frames/agents to use for output loss computation.
            If None, all frames/agents with valid data are used. Default: None
        stride: Interval between the start frames of consecutive chunks.
            If None, defaults to context_length (non-overlapping chunks). Default: None
        start_offset: Index of first frame to be used. Default: 0

    NOTE: currently assumes that discrete data in labels all have the same number of bins.
    """
    def __init__(
            self,
            inputs: dict[str, Data],
            labels: dict[str, Data],
            isstart: np.ndarray,
            context_length: int,
            useoutputmask: np.ndarray | None = None,
            stride: int | None = None,
            start_offset: int = 0,
    ):
        self.inputs = inputs
        self.labels = labels
        self.isstart = isstart
        self.context_length = context_length
        self.useoutputmask = useoutputmask
        if stride is None:
            stride = context_length
        self.stride = stride
        self.start_offset = start_offset

        # Compute sessions with continuous valid data per agent
        self.sessions = compute_sessions(
            datas=list(self.inputs.values()) + list(self.labels.values()),
            isstart=self.isstart
        )

        # Compute chunking indices
        self.chunk_indices = compute_chunk_indices(self.sessions, self.context_length, start_offset=self.start_offset, 
                                                   useoutputmask=self.useoutputmask, stride=self.stride)

        # Annotate which dimensions of chunked labels are bins
        self.label_bin_indices, self.label_n_bins = get_bin_indices(self.labels)

        # Input output dimensions
        self.d_input = sum([data.array.shape[-1] for data in inputs.values()])
        d_output_full = sum([data.array.shape[-1] for data in labels.values()])
        d_discrete = sum([len(inds) for inds in self.label_bin_indices])
        self.d_output_continuous = d_output_full - d_discrete
        if len(self.label_n_bins) > 0:
            assert len(np.unique(self.label_n_bins)) == 1
            self.discretize_nbins = self.label_n_bins[0]
            self.d_output_discrete = d_discrete // self.discretize_nbins
        else:
            self.discretize_nbins = 0
            self.d_output_discrete = 0

        # Variables required to initialize model via apf.models.initialize_model
        self.d_output = self.d_output_continuous + self.d_output_discrete
        self.discretize = self.d_output_discrete > 0
        self.flatten = False
        self.input_idx = self.input_szs = None
        self.set_input_shapes()
        
    def get_n_agents(self) -> int:
        """ Returns the number of agents in the dataset.
        """
        return max([s.agent_id for s in self.sessions])+1
    
    def get_n_frames(self) -> int:
        """ Returns the number of frames in the dataset.
        """
        return max([s.start_frame + s.duration for s in self.sessions])+1

    def set_input_shapes(self):
        """ Set feature indices of different types of inputs (note that sensory is split into further indices here).

        TODO: Handling of sensory here is quite specific, think of a better way to achieve this.
        """
        # Collect indices for the inputs
        inds_per_input = {}
        curr_idx = 0
        for key, data in self.inputs.items():
            dim = data.array.shape[-1]
            if key == 'sensory' and hasattr(data.operations[0], 'idxinfo'):
                for sensory_key, lim in data.operations[0].idxinfo.items():
                    inds_per_input[sensory_key] = [curr_idx + lim[0], curr_idx + lim[1]]
            else:
                inds_per_input[key] = [curr_idx, curr_idx + dim]
            curr_idx += dim
        self.input_idx = OrderedDict([(key, value) for key, value in inds_per_input.items()])
        self.input_szs = OrderedDict([(key, (value[1] - value[0],)) for key, value in inds_per_input.items()])

    def get_input_shapes(self) -> tuple[OrderedDict[str, tuple[int, int]], OrderedDict[str, int]]:
        """ Returns the indices and size of inputs.

        input_idx: Start and end index of the different inputs
        input_szs: Duration of different inputs (end - start)
        """
        return self.input_idx, self.input_szs

    def __len__(self) -> int:
        """ Returns the number of data chunks this dataset can produce.
        """
        return self.chunk_indices.shape[0]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | torch.Tensor]:
        """ Returns a data chunk from the dataset.

        Args:
            idx: Index of the chunk, referring to the chunk_indices table.

        Returns:
            chunk: A dictionary containing chunk data. Keys are
                'input': Concatenated input chunk, (context_length, d_input) float
                'labels': Concatenated continuous data in labels, (context_length, d_output_continuous) float
                'labels_discrete': Concatenated flattened discrete data in labels,
                    (context_length, d_output_discrete * n_bins) float
        """
        start_frame, agent_id = self.chunk_indices[idx]
        chunk = self.get_chunk(start_frame, self.context_length, agent_id)
        chunk['metadata']['idx'] = idx
        return chunk
    
    def get_subindex(self, idx: int | None = None, start_frame: int | None = None,
                     agent_id: int | None = None,
                     duration: int | None = None,
                     return_info: bool = False) -> list[slice]:
        """ Returns the index into data array prefixes corresponding to chunk idx.

        Args:
            idx: Index of the chunk, referring to the chunk_indices table.
            start_frame: Start frame of the chunk. If provided, overrides idx.
            agent_id: Agent id of the chunk. If provided, overrides idx.
            duration: Length of the chunk. If None, defaults to context_length.
        Returns:
            subindex: A tuple of slices to index into data arrays.
        """
        if start_frame is None or agent_id is None:
            start_frame, agent_id = self.chunk_indices[idx]
        if duration is None:
            duration = self.context_length
        subindex = [slice(None), slice(None)]
        if agent_id is not None:
            subindex[0] = agent_id
        if start_frame is not None and duration is not None:
            subindex[1] = slice(start_frame, start_frame + duration)
        subindex = tuple(subindex)
        if return_info:
            return subindex, {'start_frame': start_frame, 'agent_id': agent_id}
        else:
            return subindex
    
    def get_example_datadict(self,idx: int | None = None, start_frame: int | None = None,
                             agent_id: int | None = None,
                             duration: int | None = None) -> dict:
        """ Returns a data chunk from the dataset as a datadict with 'inputs' and 'labels' keys.

        Args:
            idx: Index of the chunk, referring to the chunk_indices table.
            start_frame: Start frame of the chunk. If provided, overrides idx.
            agent_id: Agent id of the chunk. If provided, overrides idx.
            duration: Length of the chunk. If None, defaults to context_length.

        Returns:
            chunk: A dictionary containing chunk data. Keys are
                'inputs': Concatenated input chunk, (context_length, d_input) float
                'labels': Concatenated label chunk, (context_length, d_output) float
                'metadata': Metadata about the chunk, extracted from each key in self.metadata
        """
        subindex, info = self.get_subindex(idx=idx, start_frame=start_frame, agent_id=agent_id, duration=duration, return_info=True)

        return {
            'inputs': {k: v[subindex] for k, v in self.inputs.items()},
            'labels': {k: v[subindex] for k, v in self.labels.items()},
            'metadata': info
        }

    def get_chunk(self, start_frame: int, duration: int, agent_id: int) -> dict[str, np.ndarray | torch.Tensor]:
        """ Returns a data chunk from the dataset.

            Args:
                start_frame: Start frame of the chunk
                duration: Length of the chunk
                agent_id: Agent id of the chunk

            Returns:
                chunk: A dictionary containing chunk data. Keys are
                    'input': Concatenated input chunk, (duration, d_input) float
                    'labels': Concatenated label chunk, (duration, d_output) float
                    'metadata': Metadata about the chunk, extracted from each key in self.metadata

        """
        chunk = get_chunk({'labels': self.labels, 'inputs': self.inputs}, start_frame, duration, agent_id, 
                          label_bin_indices = self.label_bin_indices, 
                          useoutputmask = self.useoutputmask)
        
        return chunk

    def set_context_length(self, context_length: int):
        """ Sets the context length of chunks and recomputes chunk indices.

        Args:
            context_length: Number of frames in a chunk.
        """
        self.context_length = context_length
        self.recompute_chunk_indices(start_offset=self.start_offset)    
    
    def set_stride(self, stride: int | None = None):
        """ Sets the stride between chunks and recomputes chunk indices.

        Args:
            stride: Interval between the start frames of consecutive chunks.
                If None, defaults to context_length (non-overlapping chunks).
        """
        if stride is None:
            stride = self.context_length
        self.stride = stride
        self.recompute_chunk_indices(start_offset=self.start_offset)

    def recompute_chunk_indices(self, start_offset: int | None = None):
        """ Computes chunk indices for a given start_offset.

        Args:
            start_offset: First frame of the first chunk. If None picks a random frame in [0, contex_length).
        """
        if start_offset is None:
            start_offset = np.random.randint(self.context_length)
        self.start_offset = start_offset
        self.chunk_indices = compute_chunk_indices(
            self.sessions, self.context_length, start_offset=start_offset, useoutputmask=self.useoutputmask,
            stride=self.stride
        )
        
    def split_input_by_names(self, input: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        """ Splits input by data names (from self.inputs) 

        Args:
            input: Concatenated input data, (context_length, d_input) float array

        Returns:
            input_dict: Input data split into dataset.inputs.keys().
        """
        dims_per_key = [data.array.shape[-1] for data in self.inputs.values()]
        start_per_key = np.cumsum([0] + dims_per_key)[:-1]
        inds_per_key = [np.arange(start, start + dims) for start, dims in zip(start_per_key, dims_per_key)]
        input_dict = {}
        for key, inds in zip(self.inputs.keys(), inds_per_key):
            x = input[..., inds]
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            input_dict[key] = x
        return input_dict
        
    def split_example_by_names(self, example: dict[str, np.ndarray | torch.Tensor]) -> dict[str, np.ndarray | torch.Tensor]:
        """Similar to split_output_by_names, but works on a single example returned by __getitem__.
        Looks for the keys 'inputs','labels',and 'labels_discrete' to split. 
        Args:
            example: A single example returned by __getitem__.
        Returns:
            example_dict: Example split into dataset.inputs.keys() and dataset.labels.keys().
        """
        
        return {'input': self.split_input_by_names(example['input']),
                'labels': self.split_output_by_names(example,continuous_key='labels',discrete_key='labels_discrete')}

    def split_output_by_names(self, output_discr_cont: dict[str, np.ndarray], continuous_key: str | None = None, 
                              discrete_key: str | None = None) -> dict[str, np.ndarray]:
        """ Splits output by data names (from self.labels) rather than by discrete vs continuous.

        Args:
            output_discr_cont: Output data split into continuous_key and discrete_key keys.
            continuous_key: Key in output_discr_cont corresponding to continuous data. Default 'continuous'.
            discrete_key: Key in output_discr_cont corresponding to discrete data. Default 'discrete'.
            Default values correspond to keys in predictions.

        Returns:
            output_names: Output data split into dataset.labels.keys().
        """
        
        if continuous_key is None:
            if 'labels' in output_discr_cont:
                continuous_key = 'labels'
            elif 'continuous' in output_discr_cont:
                continuous_key = 'continuous'
            else:
                continuous_key = None
        if discrete_key is None:
            if 'labels_discrete' in output_discr_cont:
                discrete_key = 'labels_discrete'
            elif 'discrete' in output_discr_cont:
                discrete_key = 'discrete'
            else:
                discrete_key = None
        
        # assemble output to look like original concatenated data (before splitting discrete and continuous)
        n_dim = self.d_output_discrete * self.discretize_nbins + self.d_output_continuous
        is_binned = np.zeros(n_dim, bool)
        for inds in self.label_bin_indices:
            is_binned[inds] = True
        if continuous_key is not None:
            sz = list(output_discr_cont[continuous_key].shape[:-1])
        else:
            sz = list(output_discr_cont[discrete_key].shape[:-1])
        concated = np.ones(sz + [n_dim]) * np.nan
        if discrete_key in output_discr_cont:
            x = output_discr_cont[discrete_key]
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            concated[..., is_binned] = x.reshape(sz + [-1])
        if continuous_key in output_discr_cont:
            x = output_discr_cont[continuous_key]
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            concated[..., ~is_binned] = x

        # split concatenated data
        dims_per_key = [data.array.shape[-1] for data in self.labels.values()]
        start_per_key = np.cumsum([0] + dims_per_key)[:-1]
        inds_per_key = [np.arange(start, start + dims) for start, dims in zip(start_per_key, dims_per_key)]
        return {key: concated[..., inds] for key, inds in zip(self.labels.keys(), inds_per_key)}
    
    def get_items(self,idx: int | list[int] | np.ndarray):
        """
        get_items(idx)
        Returns the data for the indices by call ing __getitem__ on each index
        """
        idx = np.atleast_1d(idx)
        data = [self.__getitem__(i) for i in idx]
        # concatenate each field with a new first dimension
        data = {key: np.stack([d[key] for d in data], axis=0) for key in data[0].keys()}

        return data
    
    def get_params(self) -> dict:
        """
        get_params()
        Returns a dictionary of parameters of the dataset operations. Calls get_params on each operation 
        of the inputs and labels.
        """
        params = {'inputs': {}, 'labels': {}}
        for key, data in self.inputs.items():
            params['inputs'][key] = [oper.to_dict() for oper in data.operations]
        for key, data in self.labels.items():
            params['labels'][key] = [oper.to_dict() for oper in data.operations]
        return params

    def item_to_data(self,item: dict[str, np.ndarray | torch.Tensor],subindex: dict | None = None) -> dict:
        """
        item_to_data(item)
        Converts an item (as returned by __getitem__) or the prediction of a model to a dictionary of 
        Data objects, with the operations from the dataset. 
        Args:
            item (dict[str, np.ndarray  |  torch.Tensor]): Dictionary produced by __getitem__ or a model prediction.
            Should have (some of) the keys:
                'input': Concatenated input data, (context_length, d_input) float array
                'labels' or 'continuous': Concatenated continuous data in labels, (context_length, d_output_continuous) float
                'labels_discrete' or 'discrete': Concatenated flattened discrete data in labels, 
                (context_length, d_output_discrete * n_bins) float

        Returns:
            dict[str, Data]: Dictionary containing Data objects for each key in item, with the operations from the dataset.
            Should have (some of) the keys:
                'inputs': A dictionary of Data objects for each input in the dataset.inputs.keys(). Missing if 'input' not in item.
                'labels': A dictionary of Data objects for each label in the dataset.labels.keys(). Missing if 
                'labels', 'continuous', etc. not in item.
        """
        
        continuous_key = None
        discrete_key = None
        if 'labels' in item: 
            continuous_key = 'labels'
        elif 'continuous' in item:
            continuous_key = 'continuous'
        if 'labels_discrete' in item:
            discrete_key = 'labels_discrete'
        elif 'discrete' in item:
            discrete_key = 'discrete'            
        
        # if subindex is provided, convert item to full size with NaNs in missing entries
        datadict = {}
        issubindex = subindex is not None
        if issubindex:
            item0 = item
            item = {}
            if 'input' in item0:
                item['input'] = self._subindex_to_full(item0['input'],subindex,nfeatdim=1)
            if continuous_key is not None:
                item[continuous_key] = self._subindex_to_full(item0[continuous_key],subindex,nfeatdim=1)
            if discrete_key is not None:
                item[discrete_key] = self._subindex_to_full(item0[discrete_key],subindex,nfeatdim=2)
            if 'metadata' in item0:
                item['metadata'] = item0['metadata']
                logging.warning("Subindexing metadata has not been debugged!!! Not sure what should happen!")            
            
        if 'input' in item:
            datadict['inputs'] = {}
            input_dict = self.split_input_by_names(item['input'])
            for k,v in input_dict.items():
                if 'metadata' in item and 'inputs' in item['metadata'] and k in item['metadata']['inputs']:
                    invertdatacurr = item['metadata']['inputs'][k]
                else:
                    invertdatacurr = [None,]*len(self.inputs[k].operations)
                datadict['inputs'][k] = Data.copy_with(self.inputs[k],array=v,invertdata=invertdatacurr)
        continuous_key = None
        discrete_key = None
        if 'labels' in item: 
            continuous_key = 'labels'
        elif 'continuous' in item:
            continuous_key = 'continuous'
        if 'labels_discrete' in item:
            discrete_key = 'labels_discrete'
        elif 'discrete' in item:
            discrete_key = 'discrete'            
        if continuous_key is not None or discrete_key is not None:
            datadict['labels'] = {}
            output_dict = self.split_output_by_names(item,continuous_key=continuous_key,discrete_key=discrete_key)
            for k,v in output_dict.items():
                if 'metadata' in item and 'labels' in item['metadata'] and k in item['metadata']['labels']:
                    invertdatacurr = item['metadata']['labels'][k]
                else:
                    invertdatacurr = [None,]*len(self.labels[k].operations)
                datadict['labels'][k] = Data.copy_with(self.labels[k],array=v,invertdata=invertdatacurr)
        datadict['metadata'] = {k: v for k,v in item.get('metadata',{}).items() if k not in ['inputs','labels']}        
        if 'useoutputmask' in item:
            datadict['useoutputmask'] = item['useoutputmask']
        
        return datadict
    
    def data_to_item(self,datadict: dict[str, Data], start_frame=None, agent_id=None, duration=None) -> dict[str, np.ndarray | torch.Tensor]:
        """
        data_to_item(datadict)
        Converts a dictionary of Data objects to an item (as returned by __getitem__) or the prediction of a model.
        Args:
            datadict (dict[str, Data]): Dictionary containing Data objects for each key. Should have (some of) the keys:
                'inputs': A dictionary of Data objects for each input in the dataset.inputs.keys().
                'labels': A dictionary of Data objects for each label in the dataset.labels.keys().
        Returns:
            item (dict[str, np.ndarray  |  torch.Tensor]): Dictionary produced by __getitem__ or a model prediction.
            Should have (some of) the keys:
                'input': Concatenated input data, (context_length, d_input) float array
                'labels' or 'continuous': Concatenated continuous data in labels, (context_length, d_output_continuous) float
                'labels_discrete' or 'discrete': Concatenated flattened discrete data in labels, 
                (context_length, d_output_discrete * n_bins) float
        """
        
        item = get_chunk(datadict,start_frame, duration, agent_id, 
                        label_bin_indices = self.label_bin_indices, 
                        useoutputmask = datadict.get('useoutputmask',None))
        
        return item
    
    def get_operations(self) -> tuple[dict, dict]:
        """
        get_operations()
        Returns a dictionary of operations of the dataset. 
        """
        operations = {'inputs': {}, 'labels': {}}
        invertdata = {'inputs': {}, 'labels': {}}
        for key, data in self.inputs.items():
            operations['inputs'][key] = data.operations
            invertdata['inputs'][key] = data.invertdata
        for key, data in self.labels.items():
            operations['labels'][key] = data.operations
            invertdata['labels'][key] = data.invertdata
        return operations, invertdata
    
    def rawdata_to_datadict(self, inputs: dict | None = None, labels: dict | None = None, idx: list | tuple | None = None,
                            extraargs: dict | None = {}, extraargs_inputs: dict | None = None, 
                            extraargs_labels: dict | None = None, separate_extraargs: bool = False,
                            use_data_invertdata: bool | None = None,
                            use_prev_invertdata: bool | None = None,
                            start_frame: int | None = None,timedim: int | None = None) -> dict:
        """
        rawdata_to_datadict(inputs, labels)
        Converts raw data to a datadict with Data objects for inputs and labels by applying 
        all operations to raw data.
        Args:
            inputs (dict): Dictionary of raw input data arrays. Keys should correspond to self.inputs keys.
                inputs[key] should be an ndarray or Tensor containing the data before the 
                operations in self.inputs[key].operations are applied.
            labels (dict): Dictionary of raw label data arrays. Keys should correspond to self.labels keys.
                labels[key] should be an ndarray or Tensor containing the data before the 
                operations in self.labels[key].operations are applied.
        Returns:
            datadict (dict): Dictionary containing Data objects for 'inputs' and 'labels'.    
        """
        datadict = {}
        if inputs is not None:
            if idx is not None:
                inputs_ref = {k: v[*idx] for k, v in self.inputs.items()}
            else:
                inputs_ref = self.inputs
            
            datadict['inputs'] = apply_opers_from_data(inputs_ref,inputs,
                                                        extraargs=extraargs if extraargs_inputs is None else extraargs_inputs,
                                                        use_data_invertdata=use_data_invertdata,
                                                        use_prev_invertdata=use_prev_invertdata,
                                                        separate_extraargs=separate_extraargs,
                                                        start_frame=start_frame, timedim=timedim)
        if labels is not None:
            if idx is not None:
                labels_ref = {k: v[*idx] for k, v in self.labels.items()}
            else:
                labels_ref = self.labels
            datadict['labels'] = apply_opers_from_data(labels_ref,labels,
                                                        extraargs=extraargs if extraargs_labels is None else extraargs_labels,
                                                        use_data_invertdata=use_data_invertdata,
                                                        use_prev_invertdata=use_prev_invertdata,
                                                        separate_extraargs=separate_extraargs,
                                                        start_frame=start_frame, timedim=timedim)
        return datadict
    
    def rawdata_to_item(self, inputs: dict | None = None, labels: dict | None = None, start_frame: int | None = None, 
                        agent_id: int | None = None, duration: int | None = None, **kwargs) -> dict:
        datadict = self.rawdata_to_datadict(inputs, labels, start_frame=start_frame, **kwargs)
        item = self.data_to_item(datadict, start_frame=start_frame, agent_id=agent_id, duration=duration)
        return item
    
    def _subindex_to_full(self,x,subindex,nfeatdim=1):
        """
        _subindex_to_full(x,subindex)
        Converts a subindexed array x to a full array with NaNs in missing entries.
        Args:
            x (np.ndarray  |  torch.Tensor): Input array of size (..., d), where d are the feature dimension(s). The total number of 
            entries in all but the last nfeatdim dimension(s) should match the number of entries in subindex. For example, if 
            x is of shape (n_batches, l, d) then subindex should contain n_batches*l entries.
            subindex (dict): Subindex dictionary with keys 'agent_id' and 'frame' describing what x corresponds to in the full array.
            'agent_id' is of shape (n_batches, l) and 'frame' is of shape (n_batches, l).
        Returns:
            y (np.ndarray): Full array of shape (n_agents, n_frames, d) with NaNs in missing entries. y[idx] will be set to x, and
            idx is computed from subindex.
        """

        n_agents = self.get_n_agents()
        n_frames = self.get_n_frames()
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        d = x.shape[-nfeatdim:] # last dimension(s) are feature dimensions
        x = x.reshape((-1,)+d) # flatten all but last nfeatdim dimensions
        sz = (n_agents*n_frames,)+d # will be (n_agents, n_frames, d) when reshaped
        y = np.full(sz,np.nan,dtype=x.dtype)
        # repeat agent_id to match frame dimension if dimensions don't match
        if subindex['agent_id'].ndim < subindex['frame'].ndim:
            agent_id = np.broadcast_to(subindex['agent_id'][:,None],subindex['frame'].shape).flatten()
        frame = subindex['frame'].flatten()
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        idx = np.ravel_multi_index((agent_id,frame),(n_agents,n_frames))
        assert len(idx) == len(np.unique(idx)), "subindex contains duplicate entries"
        y[idx] = x
        return y.reshape((n_agents,n_frames)+d)
    
    def get_input_names(self):
        """
        get_input_names()
        Returns a list of input feature names, prefixed by the data key in self.inputs
        {key}__{feature_name}.
        """
        
        names = []
        for k,v in self.inputs.items():
            names.extend([f"{k}__{name}" for name in v.feature_names])
        return names
    
    def get_label_names(self):
        """
        get_label_names()
        Returns a list of label feature names, prefixed by the data key in self.labels
        {key}__{feature_name}.
        """
        
        names = []
        for k,v in self.labels.items():
            names.extend([f"{k}__{name}" for name in v.feature_names])
        return names

def array_to_data(array: np.ndarray, datalike: Data) -> Data:
    """
    array_to_data(array, datalike)
    Converts an array to a Data object using the operations and name from datalike.
    Args:
        array (np.ndarray): Array to convert to Data object.
        datalike (Data): Data object to copy operations and name from.
    """
    return Data.copy_with(datalike,array=array)

def copy_data_subindex(data: Data, agentidx: slice | np.ndarray | list | None = None, frameidx: slice | np.ndarray | list | None = None) -> Data:
    """
    copy_data_subindex(data, agentidx, frameidx)
    Copies a subindex of a Data object to a new Data object.
    Args:
        data (Data): Data object to copy from.
        agentidx (slice | np.ndarray | list | None): Indices for the agent dimension.
        frameidx (slice | np.ndarray | list | None): Indices for the frame dimension.
    Returns:
        Data: New Data object with the subindexed array.
        invertdata is set to None as we haven't implemented subindexing for invertdata
    """
    array_sub = data.array
    if agentidx is not None:
        array_sub = array_sub[agentidx]
    if frameidx is not None:
        array_sub = array_sub[:,frameidx]
    return Data.copy_with(data,array=array_sub,invertdata=None)

def get_datadict_invertdata(datadict: dict[str, Data]) -> dict:
    invertdata = {}
    for key in ['inputs','labels']:
        invertdata[key] = {}
        for k,data in datadict.get(key,{}).items():
            invertdata[key][k] = data.invertdata
    return invertdata

def get_chunk(datadict: dict, 
            start_frame: int | None = None, 
            duration: int | None = None, agent_id: int | None = None,
            useoutputmask: np.ndarray | None = None,
            label_bin_indices: np.ndarray | None = None,
            return_invertdata: bool = True) -> dict[str, np.ndarray | torch.Tensor]:
    
    """ Returns a data chunk from datadict.

    Args:
        start_frame: Start frame of the chunk
        duration: Length of the chunk
        agent_id: Agent id of the chunk
        datadict: If provided, extracts chunk from this Data object instead of self. Default: None

    Returns:
        chunk: A dictionary containing chunk data. Keys are
            'input': Concatenated input chunk, (duration, d_input) float
            'labels': Concatenated label chunk, (duration, d_output) float
            'metadata': Metadata about the chunk, extracted from each key in self.metadata

    """
    chunk = {}
    if 'inputs' in datadict:
        input = get_data_chunk(datadict['inputs'], start_frame, agent_id, duration).astype(np.float32)
        chunk['input'] = input

    if 'labels' in datadict:
        labels = get_data_chunk(datadict['labels'], start_frame, agent_id, duration)
        if label_bin_indices is None:
            label_bin_indices, label_n_bins = get_bin_indices(datadict['labels'])
        labels_discrete, labels_continuous = split_discr_cont(labels, label_bin_indices)
        if labels_continuous.shape[-1] > 0:
            chunk['labels'] = labels_continuous.astype(np.float32)
        if labels_discrete.shape[-1] > 0:
            chunk['labels_discrete'] = labels_discrete.astype(np.float32)
    
    if useoutputmask is None and 'useoutputmask' in datadict:
        useoutputmask = datadict['useoutputmask']

    if start_frame is None:
        start_frame = 0
    if duration is None and (len(chunk) > 0):
        key = list(chunk.keys())[0]
        duration = chunk[key].shape[-1]
    
    if useoutputmask is None:
        chunk['useoutputmask'] = np.ones((duration,), dtype=bool)
    else:
        chunk['useoutputmask'] = useoutputmask[start_frame:start_frame+duration,
                                               agent_id if agent_id is not None else slice(None)]
            
    chunk['metadata'] = {
        'start_frame': start_frame,
        'duration': duration,
        'agent_id': agent_id
        }
    
    # invertdata
    if return_invertdata:
        do_subindex = (start_frame is not None and duration is not None) or (agent_id is not None)
        if do_subindex:
            subindex = [slice(None), slice(None)]
            if agent_id is not None:
                subindex[0] = agent_id
            if start_frame is not None and duration is not None:
                subindex[1] = slice(start_frame, start_frame + duration)

        for key in ['inputs','labels']:
            if key not in datadict:
                continue
            chunk['metadata'][key] = {}
            for k,data in datadict[key].items():
                chunk['metadata'][key][k] = data[*subindex].invertdata if do_subindex else data.invertdata

    return chunk

class DataLoader(torch.utils.data.DataLoader):
    """ A thin wrapper around torch's DataLoader to use collate_nested_dicts as the collate_fn.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=collate_nested_dicts)
