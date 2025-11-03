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
from apf.utils import connected_components, modrange, rotate_2d_points, set_invalid_ends

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
    def apply(self, data: np.ndarray) -> np.ndarray:
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
        if isinstance(data, Data):
            return Data(
                name=f"{data.name}_{self.name}",
                array=self.apply(data.array, **kwargs),
                operations=data.operations + [self]
            )
        elif isinstance(data, np.ndarray):
            return self.apply(data, **kwargs)
        else:
            raise ValueError(f"Data must be either np.ndarray or Data, not {type(data)}")

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

class Data(NamedTuple):
    name: str
    array: np.ndarray
    # Operations that have been applied to the data (can be later applied in inverse to obtain original data).
    operations: list[Operation] = []
    invertdata: Any = None # any additional data needed for inverting the operations, e.g. flyid for Pose operation

@dataclass
class Identity(Operation):
    """ This operation passes data through without modifications.

    Can be useful in combination with Fusion when a subset of dimensions needs to be operated on and others not.
    """
    def apply(self, data: np.ndarray):
        return data

    def invert(self, data: np.ndarray):
        return data

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
        LOG.error(f"Operation {self} is not invertible")


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
        
        if self.bin_edges is not None and self.bin_centers is None:
            self.bin_centers = (self.bin_edges[:, 1:] + self.bin_edges[:, :-1]) / 2
        if fit_discretize_data_args is None:
            self.fit_discretize_data_args = kwargs
        else:
            self.fit_discretize_data_args = fit_discretize_data_args | kwargs
            
        super().__post_init__()

    def compute(self, data: np.ndarray):
        """ Computes the bin edges for the data.

        Args:
            data: (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array
        """
        n_feat = data.shape[-1]
        data_flat = data.reshape((-1, n_feat))
        valid = ~np.isnan(data_flat.sum(-1))
        data_valid = data_flat[valid, :]
        bin_edges, samples, bin_means, bin_medians = fit_discretize_data(data_valid, **self.fit_discretize_data_args)
        self.bin_edges = bin_edges
        # self.bin_centers = (bin_edges[:, 1:] + bin_edges[:, :-1]) / 2
        self.bin_centers = bin_medians
        self.bin_samples = samples

    def apply(self, data: np.ndarray) -> np.ndarray:
        """ Bins the data.

        Args:
            data: Continuous data, (n_agents,  n_frames, n_features) float array
            or (n_frames, n_features) float array

        Returns:
            binned: Binned data, (n_agents,  n_frames, n_features * n_bins) float array
            or (n_frames, n_features * n_bins) float array
        """
        if self.bin_edges is None:
            self.compute(data)
        sz_rest = data.shape[:-1]
        n_feat = data.shape[-1]
        data_flat = data.reshape((-1, n_feat))

        data_flat_discrete = discretize_labels(data_flat, self.bin_edges, soften_to_ends=True)
        return data_flat_discrete.reshape(sz_rest + (-1,))

    def invert(self, data: np.ndarray, do_sampling: bool = True) -> np.ndarray:
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
        if do_sampling:
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
        elif ~isinstance(kwargs_per_op, list):
            kwargs_per_op = [kwargs_per_op for _ in self.operations]
        processed = [op.apply(data[..., indices], **kwargs) for op, indices, kwargs in zip(self.operations, self.indices_per_op, kwargs_per_op)]
        self.dims_per_op = [proc.shape[-1] for proc in processed]
        fused = np.concatenate(processed, axis=-1)
        
        if not ismultiagent:
            fused = fused[0]

        return fused

    def invert(self, data: np.ndarray, kwargs_per_op=None) -> np.ndarray:
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
            inverted[..., indices] = self.operations[i].invert(data[..., count:count + n_dims], **kwargs_per_op[i])
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
                (n_frames, n_agents) bool array or (n_frames,) bool array

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
            set_invalid_ends(pose_velocity, isstart, dt=1)
        pose_velocity = pose_velocity[0]
        pose_velocity = pose_velocity.T
        
        if not ismultiagent:
            pose_velocity = pose_velocity[0, ...]
            
        return pose_velocity

    def invert(self, velocity: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        """ Compute pose from pose velocity and an initial pose.

        Args:
            velocity: Delta pose (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
            x0: Initial pose (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array

        Returns:
            pose: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
        """
        ismultiagent = velocity.ndim == 3
        if not ismultiagent:
            velocity = velocity[None, ...]
            if x0 is not None:
                x0 = x0[None, ...]
        
        # Note: here we are assuming dt=1
        if x0 is None:
            n_agents, _, n_features = velocity.shape
            x0 = np.zeros((n_agents, n_features))
        velocity = np.concatenate([x0[:, None, :], velocity], axis=1)[:, :-1, :]
        pose = np.cumsum(velocity, axis=1)
        if self.is_angle is not None:
            pose[..., self.is_angle] = modrange(pose[..., self.is_angle], -np.pi, np.pi)
            
        if not ismultiagent:
            pose = pose[0, ...]
            
        return pose


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
                (n_frames, n_agents) bool array or (n_frames,) bool array

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
        dXoriginrel, dtheta = compute_global_velocity(Xorigin, Xtheta, self.tspred)
        movement_global = np.concatenate((dXoriginrel[:, [1, 0]], dtheta[:, None, :, :]), axis=1)
        if isstart is not None:
            for movement, dt in zip(movement_global, self.tspred):
                set_invalid_ends(movement, isstart, dt=dt)
        movement_global = movement_global.reshape((-1, n_frames, n_flies))
        
        movement_global = movement_global.T
        
        if not ismultiagent:
            movement_global = movement_global[0, ...]

        return movement_global

    def invert(self, velocity: np.ndarray, x0: np.ndarray | None = None):
        """ Compute position from global movement and an initial position.

        NOTE: This assumes velocity is only given for dt=1

        Args:
            velocity: Global movmement (n_agents,  n_frames, 3) float array or (n_frames, 3) float array
            x0: Initial pose (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array

        Returns:
            pose: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
        """
        ismultiagent = velocity.ndim == 3
        if not ismultiagent:
            velocity = velocity[None, ...]
            if x0 is not None:
                x0 = x0[None, ...]
        
        if x0 is None:
            n_agents, _, n_dim = velocity.shape
            x0 = np.zeros((n_agents, n_dim))

        d_theta = np.concatenate([x0[:, None, 2], velocity[:, :, 2]], axis=1)
        theta = modrange(np.cumsum(d_theta, axis=1), -np.pi, np.pi)[:, :-1]

        d_pos_rel = velocity[..., [1, 0]]
        d_pos = rotate_2d_points(d_pos_rel.transpose((1, 2, 0)), -theta.T).transpose((2, 0, 1))
        d_pos = np.concatenate([x0[:, None, :2], d_pos], axis=1)
        pos = np.cumsum(d_pos, axis=1)[:, :-1, :]
        
        inverted = np.concatenate([pos, theta[:, :, None]], axis=-1)
        
        if not ismultiagent:
            inverted = inverted[0, ...]
        
        return inverted

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

    def apply(self, pose: np.ndarray, isstart: np.ndarray | None = None):
        """ Compute global and local velocity from pose.

        Args:
            pose: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
            isstart: Indicates whether a new fly track starts at a given frame for an agent.
                (n_frames, n_agents) bool array or (n_frames,) bool array

        Returns:
            velocity: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
        """
        return self.fusion.apply(pose, kwargs_per_op={'isstart': isstart})

    def invert(self, velocity: np.ndarray, x0: np.ndarray | None = None):
        """ Compute pose from pose velocity and an initial pose.

        Args:
            velocity: Delta pose (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
            x0: Initial pose (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array

        Returns:
            pose: (n_agents,  n_frames, n_pose_features) float array or (n_frames, n_pose_features) float array
        """
        if x0 is not None:
            
            # if has a value for every frame, just take the first frame
            if x0.ndim == velocity.ndim:
                x0 = x0[0]
            
            kwargs_per_op = [{'x0': x0[..., self.global_inds]}, {'x0': x0[..., self.local_inds]}]
        else:
            kwargs_per_op = None
        return self.fusion.invert(velocity, kwargs_per_op)


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
        isstart: (n_frames, n_agents)

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
        start_frames = np.where(isstart[:, agent_id])[0]
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


def compute_chunk_indices(sessions: list[Session], chunk_length: int, start_offset: int = 0) -> np.ndarray:
    """ Extracts chunk indices from session data, with chunks non-overlapping.

    Args:
        sessions: A list of sessions indicating start_frame, duration, and agent_id of valid
            data intervals from a unique agent.
        chunk_length: Desired length of chunk.
        start_offset: Index of first frame to be used.

    Returns:
        chunk_indices: (n_chunks, 2) int array, each row contains (start_frame, agent_id)
    """
    chunk_indices = []
    for session in sessions:
        t0 = session.start_frame + start_offset
        t1 = session.start_frame + session.duration - chunk_length + 1
        start_frames = np.arange(t0, t1, chunk_length)
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
        start_frame: int,
        agent_id: int,
        duration: int
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
    slices = [data.array[agent_id, start_frame:(start_frame + duration)] for data  in datas.values()]
    if isinstance(slices[0], np.ndarray):
        return np.concatenate(slices, axis=1)
    else:
        return torch.cat(slices, dim=1)


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
            if isinstance(oper, Fusion):
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

def get_pre_operations(operations: list[Operation], name: str) -> list[Operation] | None:
    """ Get a list of operations that come before the operation with the given name.

    Args:
        operations: List of operations to search from.
        name: Name of operation to find.

    Returns:
        A list of operations preceding the input prefix.
    """
    _, idx = get_operation(operations, name, return_idx = True)
    if idx is None:
        return None
    return operations[:idx]


def apply_operations(data: Data, operations: list[Operation]) -> Data:
    """ Apply a list of operations to data.
    """
    for oper in operations:
        if isinstance(oper, dict):
            oper = Operation.from_dict(oper)
        data = oper(data)
    return data


def apply_inverse_operations(data: np.ndarray | torch.Tensor | Data, 
                             operations: list | None = None, invertdata: dict | None = None,
                             extraargs: dict = {}):
    """ Apply the inverse of operations to data, in reverse order.
    """
    
    # allow data to be a Data object, in which case we extract operations and invertdata from it
    if isinstance(data, Data):
        if operations is None:
            operations = data.operations
        if invertdata is None:
            invertdata = data.invertdata
        data = data.array
    
    for oper in reversed(operations):
        invertdataargs = invertdata.get(oper.name, None) if invertdata else None
        kwargs = extraargs.get(oper.name, {})
        args = []
        if isinstance(invertdataargs, dict):
            kwargs = kwargs | invertdataargs
        elif isinstance(invertdataargs, (list, tuple)):
            args += list(invertdataargs)
        elif invertdataargs is not None:
            args.append(invertdataargs)
        data = oper.invert(data, *args, **kwargs)
        
    return data

def apply_opers_from_data(datas_ref: dict[str, Data], datas: dict[str, Data]) -> dict[str, Data]:
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
    processed_data = {}
    for key in datas_ref.keys():
        assert key in datas, "Expect both data to have all of the same keys"
        opers = get_post_operations(datas_ref[key].operations, key)
        if opers is None:
            LOG.warning(f"Did not find an operation '{key}', applying all operations")
            opers = datas_ref[key].operations
        processed_data[key] = apply_operations(datas[key], opers)
    return processed_data


def apply_opers_from_data_params(data_params: list[dict], datas: dict[str, Data], check_match: bool = False) -> dict[str, Data]:
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
    processed_data = {}
    for key in data_params.keys():
        if key not in datas:
            if check_match:
                raise ValueError(f"Expect both data to have all of the same keys, but did not find '{key}' in datas")
            else:
                LOG.warning(f"Did not find '{key}' in datas, skipping")
                continue 
        # input datacurr may already have some operations applied, so we need to find the last operation that was applied
        opers = get_post_operations(data_params[key], data=datas[key])
        processed_data[key] = apply_operations(datas[key], opers)
    return processed_data



def collate_nested_dicts(batch):
    """Recursively collates nested dicts/arrays into batched tensors"""
    if isinstance(batch[0], dict):
        return {k: collate_nested_dicts([d[k] for d in batch]) for k in batch[0]}
    elif isinstance(batch[0], np.ndarray):
        return torch.stack([torch.from_numpy(x) for x in batch])
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch)
    else:
        return batch


class Dataset(torch.utils.data.Dataset):
    """ Contains ground truth data and can be supplied to torch's DataLoader to produce chunks of data.

    Args:
        inputs: A dictionary of data inputs. Each data in inputs has the following:
            array: (n_agents, n_frames, n_features) float array.
            operations: Operations that have been applied to arrive at this data.
        labels: A dictionary of data labels. Same format as inputs.
        isstart: Indicates whether a frame is the start of a sequence for an agent, (n_frames, n_agents) bool array
        context_length: Number of frames in a data chunk provided by __getitem__
        metadata: Metadata about the dataset that is needed for applying/inverting operations beyond what can 
            be captured by the operations. These are extra arguments to the operations' apply and invert function, and there 
            should be a value for each agent and frame. This is a dict with keys 'inputs' and 'labels', each containing a dict
            mapping metadata keys to (n_frames, n_agents) arrays. If any key is missing, metadata for that key is assumed to
            be empty. Default: None

    NOTE: currently assumes that discrete data in labels all have the same number of bins.
    """
    def __init__(
            self,
            inputs: dict[str, Data],
            labels: dict[str, Data],
            isstart: np.ndarray,
            context_length: int,
            metadata: dict | None = None
    ):
        self.inputs = inputs
        self.labels = labels
        self.isstart = isstart
        self.context_length = context_length
        self.metadata = metadata

        # Compute sessions with continuous valid data per agent
        self.sessions = compute_sessions(
            datas=list(self.inputs.values()) + list(self.labels.values()),
            isstart=self.isstart
        )

        # Compute chunking indices
        self.chunk_indices = compute_chunk_indices(self.sessions, self.context_length, start_offset=0)

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
        return self.get_chunk(start_frame, self.context_length, agent_id)

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
        labels = get_data_chunk(self.labels, start_frame, agent_id, duration)
        labels_discrete, labels_continuous = split_discr_cont(labels, self.label_bin_indices)
        chunk = {
            'input': get_data_chunk(self.inputs, start_frame, agent_id, duration).astype(np.float32),
        }
        if labels_continuous.shape[-1] > 0:
            chunk['labels'] = labels_continuous.astype(np.float32)
        if labels_discrete.shape[-1] > 0:
            chunk['labels_discrete'] = labels_discrete.astype(np.float32)
        chunk['metadata'] = {
            'start_frame': start_frame,
            'duration': duration,
            'agent_id': agent_id
            }
        if self.metadata is not None:
            chunk['metadata'].update(get_array_chunk(self.metadata,start_frame,agent_id,duration))
                
        return chunk

    def recompute_chunk_indices(self, start_offset: int | None = None):
        """ Computes chunk indices for a given start_offset.

        Args:
            start_offset: First frame of the first chunk. If None picks a random frame in [0, contex_length).
        """
        if start_offset is None:
            start_offset = np.random.randint(self.context_length)
        self.chunk_indices = compute_chunk_indices(self.sessions, self.context_length, start_offset=start_offset)
        
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
            else:
                continuous_key = 'continuous'
        if discrete_key is None:
            if 'labels_discrete' in output_discr_cont:
                discrete_key = 'labels_discrete'
            else:
                discrete_key = 'discrete'
        
        # assemble output to look like original concatenated data (before splitting discrete and continuous)
        n_dim = self.d_output_discrete * self.discretize_nbins + self.d_output_continuous
        is_binned = np.zeros(n_dim, bool)
        for inds in self.label_bin_indices:
            is_binned[inds] = True
        sz = list(output_discr_cont[continuous_key].shape[:-1])
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

    def item_to_data(self,item: dict[str, np.ndarray | torch.Tensor]) -> dict:
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
        datadict = {}
        if 'input' in item:
            datadict['inputs'] = {}
            input_dict = self.split_input_by_names(item['input'])
            for k,v in input_dict.items():
                if 'metadata' in item and 'inputs' in item['metadata'] and k in item['metadata']['inputs']:
                    metadatacurr = item['metadata']['inputs'][k]
                else:
                    metadatacurr = None
                datadict['inputs'][k] = Data(name=self.inputs[k].name, array=v, operations=self.inputs[k].operations, invertdata=metadatacurr)
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
                    metadatacurr = item['metadata']['labels'][k]
                else:
                    metadatacurr = None
                datadict['labels'][k] = Data(name=self.labels[k].name, array=v, operations=self.labels[k].operations, invertdata=metadatacurr)
        datadict['metadata'] = {k: v for k,v in item.get('metadata',{}).items() if k not in ['inputs','labels']}        
        
        return datadict


class DataLoader(torch.utils.data.DataLoader):
    """ A thin wrapper around torch's DataLoader to use collate_nested_dicts as the collate_fn.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=collate_nested_dicts)