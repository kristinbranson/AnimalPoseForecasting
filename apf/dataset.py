from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import OrderedDict
from typing import NamedTuple
import numpy as np
import torch

from apf.data import fit_discretize_data, discretize_labels, weighted_sample
from apf.utils import connected_components


class Operation(ABC):
    """ Abstract class for an operation to be applied to data.

    Methods to be implemented by inheriting classes are apply and inverse.
    """
    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def invert(self, data: np.ndarray) -> np.ndarray:
        pass

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
                name=f"{data.name}_{self.__class__.__name__.lower()}",
                array=self.apply(data.array, **kwargs),
                operations=data.operations + [self]
            )
        elif isinstance(data, np.ndarray):
            return self.apply(data, **kwargs)
        else:
            raise ValueError


class Data(NamedTuple):
    name: str
    array: np.ndarray
    # Operations that have been applied to the data (can be later applied in inverse to obtain original data).
    operations: list[Operation] = []


class Identity(Operation):
    """ This operation passes data through without modifications.

    Can be useful in combination with Fusion when a subset of dimensions needs to be operated on and others not.
    """
    def apply(self, data: np.ndarray):
        return data

    def invert(self, data: np.ndarray):
        return data


class Zscore(Operation):
    """ Zscores and unzscores data.

    Args:
        mean: (n_feat, ) mean value of feature values. If None will be computed the first time the operation is applied.
        std: (n_feat, ) standard deviation of feature values.
    """
    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def compute(self, data: np.ndarray):
        """ Computes the mean and standard variation of features in data.

        Args:
            data: (n_agents,  n_frames, n_features) float array
        """
        n_frames, n_agents, n_feat = data.shape
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

        Returns:
            unzscored: (n_agents,  n_frames, n_features) float array
        """
        return data * self.std[None, None, :] + self.mean[None, None, :]


class OddRoot(Operation):
    """ Takes the given root of the data to smear it out.

    Args:
        root: Which root to use, 3 usually works well. Must be odd so that signed data is correctly inverted.
    """
    def __init__(self, root: int = 3):
        assert int(root) == root, "root must be an integer"
        assert np.mod(root, 2) == 1, "only odd roots are invertible with signed data"
        self.root = root

    def apply(self, data: np.ndarray) -> np.ndarray:
        """ Applies root to the data.

        Args:
            data: (n_agents,  n_frames, n_features) float array

        Returns:
            rooted: root-th root of data, (n_agents,  n_frames, n_features) float array
        """
        return np.sign(data) * np.abs(data)**(1 / self.root)

    def invert(self, data: np.ndarray) -> np.ndarray:
        """ Applies the inverse root to data.

        Args:
            data: (n_agents,  n_frames, n_features) float array

        Returns:
            unrooted: (n_agents,  n_frames, n_features) float array
        """
        return data**self.root


class Subset(Operation):
    """ Returns the input array with only a subset of feature dimensions.

    NOTE: this operation is not invertible.

    Args:
        include_ids: Which feature ids to select.
    """
    def __init__(self, include_ids: np.ndarray) -> None:
        self.include_ids = include_ids

    def apply(self, data: np.ndarray) -> np.ndarray:
        """ Returns data with a subset of features.

        Args:
            data: (n_agents,  n_frames, n_features) float array

        Returns:
            sub_data: data with only the selected feature dimensions (n_agents,  n_frames, n_sub_features) float array
        """
        return data[..., self.include_ids]

    def invert(self, data: np.ndarray) -> None:
        print(f"Operation {self} is not invertible")


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


class Discretize(Operation):
    """ Discretizes and undiscretizes data.

    Args:
        bin_edges: Boundaries for bins, (n_feat, n_bins + 1) float array
        kwargs: Named arguments to be sent to fit_discretize_data
    """
    def __init__(self, bin_edges: np.ndarray | None = None, **kwargs):
        self.bin_edges = bin_edges
        if bin_edges is None:
            self.bin_centers = None
        else:
            self.bin_centers = (bin_edges[:, 1:] + bin_edges[:, :-1]) / 2
        self.bin_samples = None
        self.kwargs = kwargs

    def compute(self, data: np.ndarray):
        """ Computes the bin edges for the data.

        Args:
            data: (n_agents,  n_frames, n_features) float array
        """
        n_feat = data.shape[-1]
        data_flat = data.reshape((-1, n_feat))
        valid = ~np.isnan(data_flat.sum(-1))
        data_valid = data_flat[valid, :]
        bin_edges, samples, bin_means, bin_medians = fit_discretize_data(data_valid, **self.kwargs)
        self.bin_edges = bin_edges
        # self.bin_centers = (bin_edges[:, 1:] + bin_edges[:, :-1]) / 2
        self.bin_centers = bin_medians
        self.bin_samples = samples

    def apply(self, data: np.ndarray) -> np.ndarray:
        """ Bins the data.

        Args:
            data: Continuous data, (n_agents,  n_frames, n_features) float array

        Returns:
            binned: Binned data, (n_agents,  n_frames, n_features * n_bins) float array
        """
        if self.bin_edges is None:
            self.compute(data)
        n_agents, n_frames, n_feat = data.shape
        data_flat = data.reshape((-1, n_feat))

        data_flat_discrete = discretize_labels(data_flat, self.bin_edges, soften_to_ends=True)
        return data_flat_discrete.reshape((n_agents, n_frames, -1))

    def invert(self, data: np.ndarray, do_sampling: bool = True) -> np.ndarray:
        """ Unbins the data.

        Args:
            data: Binned data, (n_agents,  n_frames, n_features * n_bins) float array
            do_sampling: If True, samples from the probability distribution given by the bins.
                Otherwise takes the weighted average.
                TODO: Have an option to take argmax
                TODO: Check out pose.py and compare the random sampling scheme.
                TODO: Add an option to blur the prediction?

        Returns:
            continuous: Continuous data, (n_agents,  n_frames, n_features) float array
        """
        n_agents, n_frames, n_bin_feat = data.shape
        n_bins = self.bin_centers.shape[-1]
        n_feat = n_bin_feat // n_bins
        if do_sampling:
            return sample_discrete_labels(data.reshape((n_agents, n_frames, n_feat, n_bins)), self.bin_samples)
        else:
            return labels_discrete_to_continuous(data.reshape((n_agents, n_frames, n_feat, n_bins)), self.bin_centers)


class Fusion(Operation):
    """ Apply different operations to different parts of the data.

    Args:
        operations: List of operations to be applied
        indices_per_op: list of indices to apply each operation to.
    """
    def __init__(
            self,
            operations: list[Operation],
            indices_per_op: list[np.ndarray],
    ):
        assert len(operations) == len(indices_per_op), "List of indices should have same length as list of operations."

        all_indices = np.concatenate(indices_per_op)
        assert len(all_indices) == max(all_indices) and len(all_indices) == len(np.unique(all_indices)), \
            "Indices must cover all feature dimensions and each dimension can only be provided to one operation."

        self.operations = operations
        self.indices_per_op = indices_per_op
        # This variable will hold the dimensions of the output corresponding to each operation, used for inverting.
        self.dims_per_op = None

    def apply(self, data: np.ndarray, kwargs_per_op: list[dict] | dict | None = None) -> np.ndarray:
        """ Applies operation to each subset of data, specified by indices per operation, and concatenates the result.

        Args:
            data: (n_agents,  n_frames, n_features) float array
            kwargs_per_op: Optional arguments provided to the operations.
                If they are a list, the list should have the same length as the list of operations.
                If they are not a list, the same arguments will be provided to all operations.
                If None, no arguments will be provided to the operations.

        Returns:
            fused: (n_agents,  n_frames, n_fused_features) float array
        """
        if kwargs_per_op is None:
            kwargs_per_op = [{} for _ in self.operations]
        elif ~isinstance(kwargs_per_op, list):
            kwargs_per_op = [kwargs_per_op for _ in self.operations]
        processed = [op.apply(data[..., indices], **kwargs) for op, indices, kwargs in zip(self.operations, self.indices_per_op, kwargs_per_op)]
        self.dims_per_op = [proc.shape[-1] for proc in processed]

        return np.concatenate(processed, axis=-1)

    def invert(self, data: np.ndarray, kwargs_per_op=None) -> np.ndarray:
        """ Inverts subsets of the processed data using the operation inverses.

        Args:
            data: (n_agents,  n_frames, n_fused_features) float array

        Returns:
            unfused: (n_agents,  n_frames, n_features) float array
        """
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

        return inverted


class Roll(Operation):
    """ Rolls data by delta time.

    This is useful e.g. for providing velocities as input. If we have a pose at time t and pose velocity corresponding
    to time t + dt, then we roll the data forward by dt makes it so that the value at index t represents velocity of
    pose from t - dt to t.

    Args:
        dt: How much to roll the data by (e.g. value used to compute the velocity data).
    """
    def __init__(self, dt: int = 1):
        self.dt = dt

    def apply(self, data: np.ndarray) -> np.ndarray:
        """ Rolls data forward by dt.

        Args:
            data: (n_agents,  n_frames, n_features) float array

        Returns:
            rolled_data: (n_agents,  n_frames, n_features) float array
        """
        return np.roll(data, shift=self.dt, axis=1)

    def invert(self, data: np.ndarray) -> np.ndarray:
        """ Rolls data backwards by dt.

        Args:
            data: (n_agents,  n_frames, n_features) float array

        Returns:
            unrolled_data: (n_agents,  n_frames, n_features) float array
        """
        return np.roll(data, shift=-self.dt, axis=1)


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


def get_data_chunk(
        datas: dict[str, Data],
        start_frame: int,
        agent_id: int,
        duration: int
) -> np.ndarray | torch.Tensor:
    """ Extracts and concatenates data for given chunk indices.

    Args:
        datas: A list of data to extract from
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
    is_binned = np.zeros(data.shape[-1], np.bool)
    for inds in bin_indices:
        is_binned[inds] = True
    return data[..., is_binned], data[..., ~is_binned]


def get_operation(
        operations: list[Operation], name: str, return_idx: bool = False
) -> Operation | None | tuple[Operation | None, int | None]:
    """ Find operation from a list of operation, given its name.

    Args:
        operations: List of operations to search from.
        name: Name of operation to find.
        return_idx: Whether to also return the index of the operation.

    Returns:
        operation: Operation corresponding to the requested name, None if not found in the list.
        idx: If return_idx is True, also returns the index of the operation within the list.
    """
    for i, oper in enumerate(operations):
        if oper.__class__.__name__.lower() == name.lower():
            if return_idx:
                return oper, i
            return oper
    if return_idx:
        return None, None
    return None


def get_post_operations(operations: list[Operation], name: str):
    """ Get a list of operations that come after the operation with the specified name.

    Args:
        operations: List of operations to search from.
        name: Name of operation to find.

    Returns:
         A list of operations following the specified name.
    """
    _, idx = get_operation(operations, name, return_idx = True)
    if idx is None:
        return None
    return operations[idx + 1:]


def apply_operations(data: Data, operations: list[Operation]) -> Data:
    """ Apply a list of operations to data.
    """
    for oper in operations:
        data = oper(data)
    return data


def apply_inverse_operations(data, operations):
    """ Apply the inverse of operations to data, in reverse order.
    """
    for oper in reversed(operations):
        data = oper.invert(data)
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
    assert len(datas_ref.keys()) == len(datas.keys()), "The two data collections must have the same keys"
    processed_data = {}
    for key in datas_ref.keys():
        assert key in datas, "Expect both data to have all of the same keys"
        opers = get_post_operations(datas_ref[key].operations, key)
        processed_data[key] = apply_operations(datas[key], opers)
    return processed_data


class Dataset(torch.utils.data.Dataset):
    """ Contains ground truth data and can be supplied to torch's DataLoader to produce chunks of data.

    Args:
        inputs: A dictionary of data inputs. Each data in inputs has the following:
            array: (n_agents, n_frames, n_features) float array.
            operations: Operations that have been applied to arrive at this data.
        labels: A dictionary of data labels. Same format as inputs.
        isstart: Indicates whether a frame is the start of a sequence for an agent, (n_frames, n_agents) bool array
        context_length: Number of frames in a data chunk provided by __getitem__

    NOTE: currently assumes that discrete data in labels all have the same number of bins.
    """
    def __init__(
            self,
            inputs: dict[str, Data],
            labels: dict[str, Data],
            isstart: np.ndarray,
            context_length: int,
    ):
        self.inputs = inputs
        self.labels = labels
        self.isstart = isstart
        self.context_length = context_length

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
            self.d_output_discrete = d_discrete // self.n_bins
        else:
            self.discretize_nbins = 0
            self.d_output_discrete = 0

        # Variables required to initialize model via apf.models.initialize_model
        self.d_output = self.d_output_continuous + self.d_output_discrete
        self.discretize = self.d_output_discrete > 0
        self.flatten = False
        self.input_idx = self.input_szs = None
        self.set_input_indices()

    def set_input_shapes(self):
        """ Set feature indices of different types of inputs (note that sensory is split into further indices here).

        TODO: Handling of sensory here is quite specific, think of a better way to achieve this.
        """
        # Collect indices for the inputs
        inds_per_input = {}
        curr_idx = 0
        for key, data in self.inputs.items():
            dim = data.array.shape[-1]
            if key == 'sensory':
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
        return chunk

    def recompute_chunk_indices(self, start_offset: int | None = None):
        """ Computes chunk indices for a given start_offset.

        Args:
            start_offset: First frame of the first chunk. If None picks a random frame in [0, contex_length).
        """
        if start_offset is None:
            start_offset = np.random.randint(self.context_length)
        self.chunk_indices = compute_chunk_indices(self.sessions, self.context_length, start_offset=start_offset)

    def split_output_by_names(self, output_discr_cont: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ Splits output by data names (from self.labels) rather than by discrete vs continuous.

        Args:
            output_discr_cont: Output data split into 'continuous' and 'discrete' keys.

        Returns:
            output_names: Output data split into dataset.labels.keys().
        """
        # assemble output to look like original concatenated data (before splitting discrete and continuous)
        n_dim = self.d_output_discrete * self.n_bins + self.d_output_continuous
        is_binned = np.zeros(n_dim, np.bool)
        for inds in self.label_bin_indices:
            is_binned[inds] = True
        sz = list(output_discr_cont['continuous'].shape[:-1])
        concated = np.ones(sz + [n_dim]) * np.nan
        if 'discrete' in output_discr_cont:
            concated[..., is_binned] = output_discr_cont['discrete'].detach().cpu().numpy().reshape(sz + [-1])
        if 'continuous' in output_discr_cont:
            concated[..., ~is_binned] = output_discr_cont['continuous'].detach().cpu().numpy()

        # split concatenated data
        dims_per_key = [data.array.shape[-1] for data in self.labels.values()]
        start_per_key = np.cumsum([0] + dims_per_key)[:-1]
        inds_per_key = [np.arange(start, start + dims) for start, dims in zip(start_per_key, dims_per_key)]
        return {key: concated[..., inds] for key, inds in zip(self.labels.keys(), inds_per_key)}
