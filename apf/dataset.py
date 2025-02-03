from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import torch

from apf.data import fit_discretize_data, discretize_labels, weighted_sample #, labels_discrete_to_continuous
from apf.utils import connected_components



class Operation(ABC):
    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse(self, data: np.ndarray) -> np.ndarray:
        pass

    def get_name(self, curr_name: str) -> str:
        pass

    def __call__(self, data, **kwargs):
        if isinstance(data, Data):
            return Data(
                name=f"{data.name}_{self.__class__.__name__.lower()}",
                array=self.apply(data.array, **kwargs),
                operations=data.operations + [self])
        elif isinstance(data, np.ndarray):
            return self.apply(data, **kwargs)
        else:
            raise ValueError


class Data(NamedTuple):
    name: str
    array: np.ndarray
    operations: list[Operation] = []


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

    def inverse(self, data: np.ndarray) -> np.ndarray:
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

    def inverse(self, data: np.ndarray) -> np.ndarray:
        """ Applies the inverse root to data.

        Args:
            data: (n_agents,  n_frames, n_features) float array

        Returns:
            unrooted: (n_agents,  n_frames, n_features) float array
        """
        return data**self.root


class Subset(Operation):
    def __init__(self, include_ids):
        self.include_ids = include_ids

    def apply(self, data: np.ndarray) -> np.ndarray:
        return data[..., self.include_ids]

    def inverse(self, data: np.ndarray) -> np.ndarray:
        print(f"Operation {self} is not invertible")
        return None


def labels_discrete_to_continuous(labels_discrete, bin_centers, epsilon=1e-3):
    """ Converts discrete labels to continuous labels by taking the weighted sum of the bin centers.

    Parameters:
        labels_discrete: ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the discrete labels.
        epsilon: small number to check that the discrete labels sum to 1. Default is 1e-3.
    Output:
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
    # bin_centers = self._discretize_params['bin_medians']
    continuous = np.sum(bin_centers[None, ...] * labels_discrete, axis=-1) / s
    continuous[isbad] = np.nan
    continuous = np.reshape(continuous, szrest + (nfeat,))
    return continuous


def sample_discrete_labels(labels_discrete, bin_samples, nsamples=1):
    """
    sample_discrete_labels(labels_discrete, nsamples=1)
    Samples continuous labels from the discrete labels.
    Parameters:
    labels_discrete: ndarray of size (pre_sz x ) ntimepoints x d_discrete x nbins with the discrete labels.
    nsamples: number of samples to take. Default is 1.
    Output:
    continuous: ndarray of size nsamples x (pre_sz x ) ntimepoints with the continuous version of the labels."""
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

        # old code that assumed that all weights were good
        # binnum = weighted_sample(torch.tensor(labels_discrete[:, f, :]), nsamples=nsamples).numpy()
        # idxgood = binnum >= 0
        # sample = np.random.randint(low=0, high=nsamples_per_bin, size=(nsamples, n))
        # curr = np.zeros((nsamples,n))
        # curr = bin_samples[sample, f, binnum].reshape((nsamples,) + szrest)
        # continuous[..., f] = curr

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

    def inverse(self, data: np.ndarray, do_sampling: bool = False) -> np.ndarray:
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
            # continuous = np.zeros((n_agents, n_frames, n_feat))
            # for agent_id in range(n_agents):
            #     for frame_id in range(n_frames):
            #         for feat_id in range(n_feat):
            #             # val = np.random.choice(self.bin_centers[feat_id], p=data[agent_id, frame_id, feat_id, :])
            #             # continuous[agent_id, frame_id, feat_id] = val
            #
            #             # pick a random bin from probability distribution
            #             prob = data[agent_id, frame_id, feat_id * n_bins:(feat_id + 1) * n_bins]
            #
            #             prob = prob.astype(np.float64)
            #             prob = prob / np.sum(prob)
            #
            #             # binnum = np.random.choice(len(prob), p=prob)
            #
            #             binnum = np.argmax(prob)
            #
            #             # if binnum in [0, len(prob)-1]: # sample only at ends, otherwise use bin center
            #             # pick a random sample from that bin
            #             nsamples_per_bin = self.bin_samples.shape[0]
            #             sample_id = np.random.randint(nsamples_per_bin)
            #             continuous[agent_id, frame_id, feat_id] = self.bin_samples[sample_id, feat_id, binnum]
            #             # else:
            #             #     continuous[agent_id, frame_id, feat_id] = self.bin_centers[feat_id][binnum]
            #
            # return continuous
            return sample_discrete_labels(data.reshape((n_agents, n_frames, n_feat, n_bins)), self.bin_samples)
        else:
            return labels_discrete_to_continuous(data.reshape((n_agents, n_frames, n_feat, n_bins)), self.bin_centers)
            # return labels_discrete_to_continuous(data.reshape((n_agents, n_frames, n_feat, n_bins)), self.bin_edges)


# from simple.simple_data import bin_features
# class DiscretizeSimple(Operation):
#     """ Discretizes and undiscretizes data.
#
#     Args:
#         bin_edges: Boundaries for bins, (n_feat, n_bins + 1) float array
#         kwargs: Named arguments to be sent to fit_discretize_data
#     """
#     def __init__(self, bin_edges: np.ndarray | None = None, **kwargs):
#         self.bin_edges = bin_edges
#         if isinstance(bin_edges, int):
#             self.bin_centers = None
#         else:
#             self.compute_bin_centers()
#         self.kwargs = kwargs
#
#     def compute_bin_centers(self):
#         halfbins = np.diff(self.bin_edges, axis=-1)[:, 0] / 2
#         first = self.bin_edges[:, 0] - halfbins
#         rest = self.bin_edges + halfbins[:, None]
#         self.bin_centers = np.hstack([first[:, None], rest])
#
#     def compute(self, data: np.ndarray):
#         """ Computes the bin edges for the data.
#
#         Args:
#             data: (n_agents,  n_frames, n_features) float array
#         """
#         n_agents, n_frames, n_feat = data.shape
#         data_flat = data.reshape((-1, n_feat))
#
#         binned_data, bin_edges = bin_features(data_flat, self.bin_edges, p=1)
#         if isinstance(self.bin_edges, int):
#             self.bin_edges = bin_edges
#             self.compute_bin_centers()
#         return binned_data.reshape((n_agents, n_frames, n_feat, -1))
#
#     def apply(self, data: np.ndarray) -> np.ndarray:
#         """ Bins the data.
#
#         Args:
#             data: Continuous data, (n_agents,  n_frames, n_features) float array
#
#         Returns:
#             binned: Binned data, (n_agents,  n_frames, n_features, n_bins) float array
#         """
#         return self.compute(data)
#
#     def inverse(self, data: np.ndarray, do_sampling: bool = True) -> np.ndarray:
#         """ Unbins the data.
#
#         Args:
#             data: Binned data, (n_agents,  n_frames, n_features, n_bins) float array
#             do_sampling: If True, samples from the probability distribution given by the bins.
#                 Otherwise, takes the argmax.
#
#         Returns:
#             continuous: Continuous data, (n_agents,  n_frames, n_features) float array
#         """
#         n_agents, n_frames, n_feat, n_bins = data.shape
#         continuous = np.zeros((n_agents, n_frames, n_feat))
#         if do_sampling:
#             for agent_id in range(n_agents):
#                 for frame_id in range(n_frames):
#                     for feat_id in range(n_feat):
#                         val = np.random.choice(self.bin_centers[feat_id], p=data[agent_id, frame_id, feat_id, :])
#                         continuous[agent_id, frame_id, feat_id] = val
#         else:
#             for feat_id in range(n_feat):
#                 bin_ids = np.argmax(data[:, :, feat_id, :], axis=-1)
#                 continuous[:, :, feat_id] = self.bin_centers[feat_id][bin_ids]
#         return continuous

class Fusion(Operation):
    def __init__(
            self,
            operations: list[Operation],
            indices_per_op: list[np.ndarray],
    ):
        """Note: no need to keep track of kwargs here as operations are created before this"""
        self.operations = operations
        self.indices_per_op = indices_per_op
        self.dims_per_op = None

    @property
    def __name__(self):
        name = "("
        for op in self.operations:
            name += f"{op.__class__.__name__} "
        name[-1] = ")"
        return name

    def apply(self, data: np.ndarray, kwargs_per_op=None) -> np.ndarray:
        """
        """
        if kwargs_per_op is None:
            kwargs_per_op = [{} for _ in self.operations]
        elif ~isinstance(kwargs_per_op, list):
            kwargs_per_op = [kwargs_per_op for _ in self.operations]
        processed = [op.apply(data[..., indices], **kwargs) for op, indices, kwargs in zip(self.operations, self.indices_per_op, kwargs_per_op)]
        self.dims_per_op = [proc.shape[-1] for proc in processed]

        return np.concatenate(processed, axis=-1)

    def inverse(self, data: np.ndarray, kwargs_per_op=None) -> np.ndarray:
        """
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
            inverted[..., indices] = self.operations[i].inverse(data[..., count:count + n_dims], **kwargs_per_op[i])
            count += n_dims

        return inverted


class FutureAsInput(Operation):
    def __init__(self, dt=1):
        self.dt = dt

    def apply(self, data):
        return np.roll(data, shift=self.dt, axis=1)

    def inverse(self, data):
        return np.roll(data, shift=-self.dt, axis=1)


# # TODO: remove this?
# class Multi(Operation):
#     """ Applies and unapplies a list of operations.
#
#     Args:
#         operations: A list of operations. The first operation in the list is applied first to the data.
#     """
#     def __init__(self, operations: list[Operation]):
#         self.operations = operations
#
#     def apply(self, data: np.ndarray) -> np.ndarray:
#         """ Applies operations to the data sequentially.
#
#         Args:
#             data: (n_agents,  n_frames, n_features) float array
#
#         Returns:
#             processed: Data processed by all operations, (n_agents,  n_frames, n_features[, n_bins]) float array
#         """
#         for operation in self.operations:
#             data = operation.apply(data)
#         return data
#
#     def inverse(self, data: np.ndarray) -> np.ndarray:
#         """
#
#         Args:
#             data: (n_agents,  n_frames, n_features[, n_bins]) float array
#
#         Returns:
#             unprocessed: Data that has been iverse processed by all operations in reverse order,
#                 (n_agents,  n_frames, n_features) float array
#         """
#         for operation in reversed(self.operations):
#             data = operation.inverse(data)
#         return data





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


# TODO: Modify everything that uses Data to use the new version
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


def get_data_chunk(datas: dict[str, Data], start_frame: int, agent_id: int, duration: int) -> np.ndarray | torch.Tensor:
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
    # slices = []
    # for data in datas:
    #     slice = data.array[agent_id, start_frame:(start_frame + duration)]
    #     slices.append(slice.reshape((duration, -1)))
    # TODO: support both np and torch
    if isinstance(slices[0], np.ndarray):
        return np.concatenate(slices, axis=1)
    else:
        return torch.cat(slices, dim=1)


def get_bin_indices(datas: dict[str, Data]):
    """Returns the indices of features that are binned in the concatenated datas and the number corresponding bins.
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


def split_discr_cont(data: np.ndarray, bin_indices):
    is_binned = np.zeros(data.shape[-1], np.bool)
    for inds in bin_indices:
        is_binned[inds] = True
    return data[..., is_binned], data[..., ~is_binned]


def slice_from_feat_name(array: np.ndarray, datas: [str, Data], feat_name: str):
    """
    Params:
        array: (..., n_feat)
    """
    if feat_name not in datas:
        return None
    dims_per_key = [data.array.shape[-1] for data in datas.values()]
    start_per_key = np.cumsum([0] + dims_per_key)
    idx = list(datas.keys()).index(feat_name)
    return array[..., start_per_key[idx]:start_per_key[idx] + dims_per_key[idx]]


def get_operation(operations: list[Operation], name: str, return_idx: bool = False):
    for i, oper in enumerate(operations):
        if oper.__class__.__name__.lower() == name.lower():
            if return_idx:
                return oper, i
            return oper
    if return_idx:
        return None, None
    return None


def get_post_operations(operations: list[Operation], name: str):
    _, idx = get_operation(operations, name, return_idx = True)
    if idx is None:
        return None
    return operations[idx + 1:]


def apply_operations(data, operations):
    for oper in operations:
        data = oper(data)
    return data


def apply_inverse_operations(data, operations):
    for oper in reversed(operations):
        data = oper.inverse(data)
    return data


def apply_opers_from_data(data_from, data_to):
    """
    I think I can just use this instead of assemble inputs, just need to concatenate the output
    """
    assert len(data_from.keys()) == len(data_to.keys()), "The two data collections must have the same keys"
    for key_from, key_to in zip(data_from.keys(), data_to.keys()):
        assert key_from == key_to, "The two data collections must have the same keys in the same order"
        opers = get_post_operations(data_from[key_from].operations, key_from)
        data_to[key_from] = apply_operations(data_to[key_from],opers)
    return data_to


class Dataset(torch.utils.data.Dataset):
    """ Contains ground truth data and can be supplied to torch's DataLoader to produce chunks of data.

    Args:
        inputs: A list of data inputs. Each data in inputs has the following:
            raw: (n_agents, n_frames, n_features) float array
            operation: Operation that shall be applied to that data, e.g. Zscore or Discretize
            processed: (n_agents, n_frames, n_features[, n_bins) float array
        labels: A list of data labels. Same format as inputs.
        isstart: Indicates whether a frame is the start of a sequence for an agent, (n_frames, n_agents) bool array
        context_length: Number of frames in a data chunk provided by __getitem__

    NOTE: currently assumes that
        Discrete data in labels are provided last (would be easy to modify)
        Discrete data in labels all have the same number of bins
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
        self.sessions = compute_sessions(self.all_data(), self.isstart)

        # Compute chunking indices
        self.chunk_indices = compute_chunk_indices(self.sessions, self.context_length, start_offset=0)

        # Annotate which dimensions of chunked labels are bins
        self.label_bin_indices, self.label_n_bins = get_bin_indices(self.labels)

        # Input output dimensions
        self.d_input = sum([data.array.shape[-1] for data in inputs.values()])
        d_output = sum([data.array.shape[-1] for data in labels.values()])
        d_discrete = sum([len(inds) for inds in self.label_bin_indices])
        self.d_output_continuous = d_output - d_discrete
        if len(self.label_n_bins) > 0:
            assert len(np.unique(self.label_n_bins)) == 1
            self.n_bins = self.label_n_bins[0]
            self.d_output_discrete = d_discrete // self.n_bins
        else:
            self.n_bins = 0
            self.d_output_discrete = 0

    def all_data(self):
        return list(self.inputs.values()) + list(self.labels.values())

    def recompute_chunk_indices(self, start_offset: int | None = None):
        """ Computes chunk indices for a given start_offset

        Args:
            start_offset: First frame of the first chunk. If None picks a random frame in [0, contex_length).
        """
        if start_offset is None:
            start_offset = np.random.randint(self.context_length)
        self.chunk_indices = compute_chunk_indices(self.sessions, self.context_length, start_offset=start_offset)

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
        # return {
        #     'input': get_data_chunk(self.inputs, start_frame, agent_id, duration).astype(np.float32),
        #     'labels_discrete': labels_discrete.astype(np.float32),
        #     'labels': labels_continuous.astype(np.float32),
        # }

    def __len__(self) -> int:
        """ Returns the number of data chunks this dataset can produce.
        """
        return self.chunk_indices.shape[0]

    def split_output_to_labels(self, output):
        # assemble output to look like original concatenated data (before splitting discrete and continuous)
        n_dim = self.d_output_discrete * self.n_bins + self.d_output_continuous
        is_binned = np.zeros(n_dim, np.bool)
        for inds in self.label_bin_indices:
            is_binned[inds] = True
        sz = list(output['continuous'].shape[:-1])
        concated = np.ones(sz + [n_dim]) * np.nan
        if 'discrete' in output:
            concated[..., is_binned] = output['discrete'].detach().cpu().numpy().reshape(sz + [-1])
        if 'continuous' in output:
            concated[..., ~is_binned] = output['continuous'].detach().cpu().numpy()

        # split concatenated data
        dims_per_key = [data.array.shape[-1] for data in self.labels.values()]
        start_per_key = np.cumsum([0] + dims_per_key)[:-1]
        inds_per_key = [np.arange(start, start + dims) for start, dims in zip(start_per_key, dims_per_key)]
        return {key: concated[..., inds] for key, inds in zip(self.labels.keys(), inds_per_key)}


# TODO:
# We could more generally have operations specify whether data is discrete, continuous (or something else)
# Then we can collect all the different labels and provide as separate outputs (the label determines
#  what type of loss function is to be used)


# class DataOld:
#     """ Keeps track of data and operations that go along with it.
#
#     Args:
#         raw: n_agents x n_frames x n_features
#         operation: Operation to be applied to the data (contains all necessary  parameters).
#     """
#     def __init__(self, raw: np.ndarray, operation: Operation | None = None):
#         self.raw = raw
#         self.operation = operation
#         if operation is None:
#             self.processed = self.raw
#         else:
#             self.processed = self.operation.apply(self.raw)
#
#
# class DatasetOld(torch.utils.data.Dataset):
#     """ Contains ground truth data and can be supplied to torch's DataLoader to produce chunks of data.
#
#     Args:
#         inputs: A list of data inputs. Each data in inputs has the following:
#             raw: (n_agents, n_frames, n_features) float array
#             operation: Operation that shall be applied to that data, e.g. Zscore or Discretize
#             processed: (n_agents, n_frames, n_features[, n_bins) float array
#         labels: A list of data labels. Same format as inputs.
#         isstart: Indicates whether a frame is the start of a sequence for an agent, (n_frames, n_agents) bool array
#         context_length: Number of frames in a data chunk provided by __getitem__
#
#     NOTE: currently assumes that
#         Discrete data in labels are provided last (would be easy to modify)
#         Discrete data in labels all have the same number of bins
#     """
#     def __init__(
#             self,
#             inputs: dict[str, Data],
#             labels: dict[str, Data],
#             isstart: np.ndarray,
#             context_length: int,
#     ):
#         self.inputs = inputs
#         self.labels_continuous = [data for data in labels if len(data.array.shape) == 3]
#         self.labels_discrete = [data for data in labels if len(data.processed.shape) == 4]
#         self.isstart = isstart
#         self.context_length = context_length
#
#         # Compute data dimensions
#         self.d_input = sum([data.processed.shape[2] for data in self.inputs])
#         self.d_output_continuous = sum([data.processed.shape[2] for data in self.labels_continuous])
#         self.d_output_discrete = sum([data.processed.shape[2] for data in self.labels_discrete])
#         bins = [data.processed.shape[3] for data in self.labels_discrete]
#         assert len(np.unique(bins)) == 1, "All discrete data expected to have same number of bins"
#         self.n_bins = bins[0]
#
#         # Compute sessions with continuous valid data per agent
#         self.sessions = compute_sessions(self.all_data(), self.isstart)
#
#         # Compute chunking indices
#         self.chunk_indices = compute_chunk_indices(self.sessions, self.context_length, start_offset=0)
#
#     def all_data(self):
#         return self.inputs + self.labels_continuous + self.labels_discrete
#
#     def recompute_chunk_indices(self, start_offset: int | None = None):
#         """ Computes chunk indices for a given start_offset
#
#         Args:
#             start_offset: First frame of the first chunk. If None picks a random frame in [0, contex_length).
#         """
#         if start_offset is None:
#             start_offset = np.random.randint(self.context_length)
#         self.chunk_indices = compute_chunk_indices(self.sessions, self.context_length, start_offset=start_offset)
#
#     def __getitem__(self, idx: int) -> dict[str, np.ndarray | torch.Tensor]:
#         """ Returns a data chunk from the dataset.
#
#         Args:
#             idx: Index of the chunk, referring to the chunk_indices table.
#
#         Returns:
#             chunk: A dictionary containing chunk data. Keys are
#                 'input': Concatenated input chunk, (context_length, d_input) float
#                 'labels': Concatenated continuous data in labels, (context_length, d_output_continuous) float
#                 'labels_discrete': Concatenated flattened discrete data in labels,
#                     (context_length, d_output_discrete * n_bins) float
#         """
#         start_frame, agent_id = self.chunk_indices[idx]
#         return self.get_chunk(start_frame, self.context_length, agent_id)
#
#     def get_chunk(self, start_frame: int, duration: int, agent_id: int) -> dict[str, np.ndarray | torch.Tensor]:
#         """ Returns a data chunk from the dataset.
#
#             Args:
#                 start_frame: Start frame of the chunk
#                 duration: Length of the chunk
#                 agent_id: Agent id of the chunk
#
#             Returns:
#                 chunk: A dictionary containing chunk data. Keys are
#                     'input': Concatenated input chunk, (duration, d_input) float
#                     'labels': Concatenated continuous data in labels, (context_length, d_output_continuous) float
#                     'labels_discrete': Concatenated flattened discrete data in labels,
#                         (context_length, d_output_discrete * n_bins) float
#             """
#         chunk = {
#             'input': get_data_chunk(self.inputs, start_frame, agent_id, duration),
#         }
#         if len(self.labels_continuous) > 0:
#             chunk['labels'] = get_data_chunk(self.labels_continuous, start_frame, agent_id, duration)
#         if len(self.labels_discrete) > 0:
#             chunk['labels_discrete'] = get_data_chunk(self.labels_discrete, start_frame, agent_id, duration)
#
#         return chunk
#
#     def __len__(self) -> int:
#         """ Returns the number of data chunks this dataset can produce.
#         """
#         return self.chunk_indices.shape[0]
