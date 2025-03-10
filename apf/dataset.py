from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import torch

from apf.data import fit_discretize_data, discretize_labels, labels_discrete_to_continuous


class Operation(ABC):
    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse(self, data: np.ndarray) -> np.ndarray:
        pass


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


class Discretize(Operation):
    """ Discretizes and undiscretizes data.

    Args:
        bin_edges: Boundaries for bins, (n_feat, n_bins + 1) float array
        kwargs: Named arguments to be sent to fit_discretize_data
    """
    def __init__(self, bin_edges: np.ndarray | None = None, kwargs: dict | None = None):
        self.bin_edges = bin_edges
        if bin_edges is None:
            self.bin_centers = None
        else:
            self.bin_centers = (bin_edges[:, 1:] + bin_edges[:, :-1]) / 2
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
        kwargs = self.kwargs
        if kwargs is None:
            kwargs = {}
        bin_edges, samples, bin_means, bin_medians = fit_discretize_data(data_valid, **kwargs)
        self.bin_edges = bin_edges
        self.bin_centers = (bin_edges[:, 1:] + bin_edges[:, :-1]) / 2

    def apply(self, data: np.ndarray) -> np.ndarray:
        """ Bins the data.

        Args:
            data: Continuous data, (n_agents,  n_frames, n_features) float array

        Returns:
            binned: Binned data, (n_agents,  n_frames, n_features, n_bins) float array
        """
        if self.bin_edges is None:
            self.compute(data)
        n_agents, n_frames, n_feat = data.shape
        data_flat = data.reshape((-1, n_feat))

        data_flat_discrete = discretize_labels(data_flat, self.bin_edges, soften_to_ends=True)
        n_bins = data_flat_discrete.shape[-1]
        return data_flat_discrete.reshape((n_agents, n_frames, n_feat, n_bins))

    def inverse(self, data: np.ndarray, do_sampling: bool = True) -> np.ndarray:
        """ Unbins the data.

        Args:
            data: Binned data, (n_agents,  n_frames, n_features, n_bins) float array
            do_sampling: If True, samples from the probability distribution given by the bins.
                Otherwise takes the weighted average.
                TODO: Have an option to take argmax
                TODO: Check out pose.py and compare the random sampling scheme.
                TODO: Add an option to blur the prediction?

        Returns:
            continuous: Continuous data, (n_agents,  n_frames, n_features) float array
        """
        if do_sampling:
            n_agents, n_frames, n_feat, n_bins = data.shape
            continuous = np.zeros((n_agents, n_frames, n_feat))
            for agent_id in range(n_agents):
                for frame_id in range(n_frames):
                    for feat_id in range(n_feat):
                        val = np.random.choice(self.bin_centers[feat_id], p=data[agent_id, frame_id, feat_id, :])
                        continuous[agent_id, frame_id, feat_id] = val
            return continuous
        else:
            return labels_discrete_to_continuous(data, self.bin_edges)


class Multi(Operation):
    """ Applies and unapplies a list of operations.

    Args:
        operations: A list of operations. The first operation in the list is applied first to the data.
    """
    def __init__(self, operations: list[Operation]):
        self.operations = operations

    def apply(self, data: np.ndarray) -> np.ndarray:
        """ Applies operations to the data sequentially.

        Args:
            data: (n_agents,  n_frames, n_features) float array

        Returns:
            processed: Data processed by all operations, (n_agents,  n_frames, n_features[, n_bins]) float array
        """
        for operation in self.operations:
            data = operation.apply(data)
        return data

    def inverse(self, data: np.ndarray) -> np.ndarray:
        """

        Args:
            data: (n_agents,  n_frames, n_features[, n_bins]) float array

        Returns:
            unprocessed: Data that has been iverse processed by all operations in reverse order,
                (n_agents,  n_frames, n_features) float array
        """
        for operation in reversed(self.operations):
            data = operation.inverse(data)
        return data


class Data:
    """ Keeps track of data and operations that go along with it.

    Args:
        raw: n_agents x n_frames x n_features
        operation: Operation to be applied to the data (contains all necessary  parameters).
    """
    def __init__(self, raw: np.ndarray, operation: Operation | None = None):
        self.raw = raw
        self.operation = operation
        if operation is None:
            self.processed = self.raw
        else:
            self.processed = self.operation.apply(self.raw)


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


def compute_sessions(datas: list[Data], isstart: np.ndarray) -> list[Session]:
    """ Extracts intervals of data belonging to a unique agent with valid data.

    Args:
        datas: A list of Data, can be a combination of binned and continuous.
        isstart: (n_frames, n_agents)

    Returns:
        sessions: A list of sessions indicating start_frame, duration, and agent_id of valid
            data intervals from a unique agent.
    """
    max_n_agents, n_frames = datas[0].processed.shape[:2]

    # Sum the first dimension of all datas to determine nans
    sum_first = datas[0].raw[..., 0]
    for i in range(1, len(datas)):
        sum_first += datas[i].raw[..., 0]
    nans = np.isnan(sum_first)

    sessions = []
    for agent_id in range(max_n_agents):
        start_frames = np.where(isstart[:, agent_id])[0]
        durations = np.diff(list(start_frames) + [n_frames])
        for start_frame, duration in zip(start_frames, durations):
            frames = np.arange(start_frame, start_frame + duration)
            # find first and last valid frame
            valids = np.where(nans[agent_id, frames] == 0)[0]
            first_frame = int(frames[valids[0]])
            last_frame = int(frames[valids[-1]])
            sessions.append(
                Session(
                    start_frame=first_frame,
                    duration=last_frame - first_frame + 1,
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


def get_data_chunk(datas: list[Data], start_frame: int, agent_id: int, duration: int) -> np.ndarray | torch.Tensor:
    """ Extracts and concatenates data for given chunk indices.

    Args:
        datas: A list of data to extract from
        start_frame: Start frame of the chunk
        agent_id: Agent id of the chunk
        duration: Number of frames in the chunk

    Returns:
        data_chunk: (chunk_length, n_feat_flat), float array
    """
    slices = []
    for data in datas:
        slice = data.processed[agent_id, start_frame:(start_frame + duration)]
        slices.append(slice.reshape((duration, -1)))
    # TODO: support both np and torch
    # return np.concatenate(slices, axis=1)
    return torch.cat(slices, dim=1)


def get_data_dims(datas: list[Data]) -> tuple[int, int, int, int | None]:
    """ Returns the accumulated dimensions of data contained in datas list.

    Args:
        datas: A combination of continuous and binned data.

    Returns:
        d_continuous: Total number of continuous features across all data in datas
        d_discrete: Total number of discrete features across all data in datas
        d_flat: Dimension of flattened concatenated data in datas
        n_bins: Number of bins for binned data. None of none of the data in datas is discrete.
    """
    d_flat = sum([np.prod(data.processed.shape[2:]) for data in datas])
    d_continuous = 0
    d_discrete = 0
    n_bins = None
    for i, data in enumerate(datas):
        if len(data.processed.shape) == 3:
            assert d_discrete == 0, "All continuous outputs should be provided before discrete ones."
            d_continuous += data.processed.shape[-1]
        else:
            if n_bins is not None:
                assert n_bins == data.processed.shape[-1], "All discrete data should have the same number of bins."
            n_bins = data.processed.shape[-1]
            d_discrete += data.processed.shape[-2]
    return d_continuous, d_discrete, d_flat, n_bins


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
            inputs: list[Data],
            labels: list[Data],
            isstart: np.ndarray,
            context_length: int,
    ):
        self.inputs = inputs
        self.labels = labels
        self.isstart = isstart
        self.context_length = context_length

        # Compute data dimensions used for training a model
        _, _, self.d_input, _ = get_data_dims(inputs)
        self.d_output_continuous, self.d_output_discrete, _, self.n_bins = get_data_dims(labels)

        # Compute sessions with continuous valid data per agent
        self.sessions = compute_sessions(self.inputs + self.labels, self.isstart)

        # Compute chunking indices
        self.chunk_indices = compute_chunk_indices(self.sessions, self.context_length, start_offset=0)

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

    def get_chunk(self, start_frame: int, duration: int, agent_id: int) -> np.ndarray | torch.Tensor:
        """ Returns a data chunk from the dataset.

            Args:
                start_frame: Start frame of the chunk
                duration: Length of the chunk
                agent_id: Agent id of the chunk

            Returns:
                chunk: A dictionary containing chunk data. Keys are
                    'input': Concatenated input chunk, (duration, d_input) float
                    'labels': Concatenated continuous data in labels, (context_length, d_output_continuous) float
                    'labels_discrete': Concatenated flattened discrete data in labels,
                        (context_length, d_output_discrete * n_bins) float
            """
        chunk = {
            'input': get_data_chunk(self.inputs, start_frame, agent_id, duration),
        }
        # TODO: support a mix of discrete and continuous
        if self.n_bins is None:
            chunk['labels'] = get_data_chunk(self.labels, start_frame, agent_id, duration)
        else:
            chunk['labels_discrete'] = get_data_chunk(self.labels, start_frame, agent_id, duration)

        return chunk

    def __len__(self) -> int:
        """ Returns the number of data chunks this dataset can produce.
        """
        return self.chunk_indices.shape[0]

# class AgentLLMDataset:
#     def __init__(self):
#         return
#
