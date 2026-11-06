"""Fast hidden state pass over real data, and the compact on-disk store of its output."""
import os

import numpy as np
import torch
from numpy.lib.format import open_memmap
from numpy.lib.stride_tricks import sliding_window_view
from tqdm.auto import tqdm

from apf.eval_pipeline.params import atomic_path


class WindowBatches(torch.utils.data.Dataset):
    """Batches of input windows, each ending at one of the given (agent, frame) rows.

    Slices the input arrays directly: Dataset.get_chunk also extracts labels and metadata, one frame at a time.
    """

    def __init__(self, dataset, agents, frames, batch_size):
        self.arrays = [np.asarray(data.array) for data in dataset.inputs.values()]
        self.contextl = dataset.context_length
        self.agents = agents
        self.frames = frames
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.frames) / self.batch_size))

    def __getitem__(self, i):
        agents = self.agents[i * self.batch_size:(i + 1) * self.batch_size]
        frames = self.frames[i * self.batch_size:(i + 1) * self.batch_size]
        batch = np.zeros((len(frames), self.contextl, sum(array.shape[-1] for array in self.arrays)), np.float32)
        for agent in np.unique(agents):
            rows = agents == agent
            first = frames[rows] - self.contextl + 1
            batch[rows] = np.concatenate(
                [np.moveaxis(sliding_window_view(array[agent], self.contextl, axis=0)[first], -1, 1)
                 for array in self.arrays], axis=-1)
        return batch


def iter_hidden(model, dataset, agents, frames, layer, batch_size=128, causal=True, num_workers=0,
                mixed_precision=False):
    """Yields (first row, hidden states as a float32 array of shape (batch, d_model)).

    Only the requested layer's last position is computed and returned: later layers and the decoder are not run.
    Requires a model that was set to eval mode. mixed_precision only has an effect on a GPU.
    """
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(
        WindowBatches(dataset, agents, frames, batch_size), batch_size=None, num_workers=num_workers)
    mask = None
    if causal:
        mask = torch.nn.Transformer.generate_square_subsequent_mask(dataset.context_length, device=device)
    row = 0
    use_autocast = mixed_precision and device.type == 'cuda'
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=use_autocast):
        for batch in tqdm(loader, desc='Hidden states', mininterval=30):
            hidden = model.forward(batch.to(device), mask=mask, is_causal=causal, return_hidden=True,
                                   hidden_layers=[layer], hidden_last_only=True, hidden_only=True)['hidden'][0]
            yield row, hidden.astype(np.float32)
            row += len(batch)


def unit_ranges(n_rows, frames_per_unit):
    """Rows are split into units, the granularity at which a killed job resumes."""
    return [(start, min(start + frames_per_unit, n_rows)) for start in range(0, n_rows, frames_per_unit)]


def unit_path(directory, unit_id):
    return os.path.join(directory, f"unit_{unit_id:05d}.npy")


def write_feature_unit(path, model, dataset, agents, frames, layer, dtype, **kwargs):
    """Computes the hidden states for the rows of one unit and writes them atomically."""
    tmp = atomic_path(path)
    out = open_memmap(tmp, mode='w+', dtype=dtype, shape=(len(frames), model.d_model))
    for row, hidden in iter_hidden(model, dataset, agents, frames, layer, **kwargs):
        out[row:row + len(hidden)] = hidden
    out.flush()
    del out
    os.replace(tmp, path)


class FeatureStore:
    """Reads rows from the units written by write_feature_unit."""

    def __init__(self, directory, n_rows, frames_per_unit):
        self.directory = directory
        self.n_rows = n_rows
        self.frames_per_unit = frames_per_unit
        self.units = {}

    def _unit(self, unit_id):
        if unit_id not in self.units:
            self.units[unit_id] = np.load(unit_path(self.directory, unit_id), mmap_mode='r')
        return self.units[unit_id]

    @property
    def d_feature(self):
        return self._unit(0).shape[1]

    def read(self, rows):
        """Returns features as float32 in the order of rows, reading from disk in sorted order."""
        rows = np.asarray(rows)
        out = np.zeros((len(rows), self.d_feature), np.float32)
        order = np.argsort(rows)
        unit_ids = rows[order] // self.frames_per_unit
        for unit_id in np.unique(unit_ids):
            selection = order[unit_ids == unit_id]
            out[selection] = self._unit(int(unit_id))[rows[selection] % self.frames_per_unit]
        return out
