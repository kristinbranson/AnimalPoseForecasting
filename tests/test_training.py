"""Checks of apf/training.py's count of the training steps, which the learning rate schedule is laid out over.

Run from the repository root with
    python -m pytest tests/test_training.py
or without pytest as
    python tests/test_training.py
No data, trained model or GPU is needed.
"""
import types

import testing_utils  # puts this checkout first on sys.path; must precede the imports below

import numpy as np
import torch

import apf.distributed as distributed
from apf.dataset import Session, compute_chunk_indices
from apf.training import chunk_counts_by_start_offset, count_training_steps, steps_per_epoch

CHUNK_LENGTH = 16


class ChunkDataset(torch.utils.data.Dataset):
    """A stand-in for apf.dataset.Dataset with only what the step count reads: its sessions, chunk length,
    output mask, and chunk tiling."""

    def __init__(self, sessions: list[Session], useoutputmask: np.ndarray | None):
        self.sessions = sessions
        self.context_length = CHUNK_LENGTH
        self.useoutputmask = useoutputmask
        self.recompute_chunk_indices(start_offset=0)

    def recompute_chunk_indices(self, start_offset: int) -> None:
        self.chunk_indices = compute_chunk_indices(self.sessions, self.context_length, start_offset=start_offset,
                                                   useoutputmask=self.useoutputmask)

    def __len__(self) -> int:
        return self.chunk_indices.shape[0]

    def __getitem__(self, i: int) -> int:
        return i


def chunk_dataset(with_mask: bool) -> ChunkDataset:
    """Sessions of many lengths, some shorter than a chunk, in 3 agents' tracks, and, if with_mask, an output
    mask with gaps longer than a chunk, so that some chunks are dropped."""
    rng = np.random.default_rng(0)
    n_frames, n_agents = 2000, 3
    sessions = []
    for agent_id in range(n_agents):
        t = 0
        while True:
            duration = int(rng.integers(5, 6 * CHUNK_LENGTH))
            if t + duration > n_frames:
                break
            sessions.append(Session(start_frame=t, duration=duration, agent_id=agent_id))
            t += duration + int(rng.integers(1, 10))
    useoutputmask_maybe = None
    if with_mask:
        useoutputmask_maybe = np.ones((n_frames, n_agents), dtype=bool)
        for _ in range(40):
            t = int(rng.integers(n_frames))
            useoutputmask_maybe[t:t + int(rng.integers(1, 3 * CHUNK_LENGTH)), int(rng.integers(n_agents))] = False
    return ChunkDataset(sessions, useoutputmask_maybe)


def test_chunk_counts_match_retiling():
    """chunk_counts_by_start_offset gives the number of chunks retiling from each start offset gives, with
    and without an output mask that drops chunks."""
    for with_mask in [False, True]:
        dataset = chunk_dataset(with_mask)
        expected = []
        for start_offset in range(CHUNK_LENGTH):
            dataset.recompute_chunk_indices(start_offset)
            expected.append(len(dataset))
        assert len(set(expected)) > 1, 'the test needs chunk counts that vary with the start offset'
        np.testing.assert_array_equal(chunk_counts_by_start_offset(dataset), expected)


def test_steps_per_epoch_matches_dataloader_length():
    """steps_per_epoch gives len(dataloader) for the dataloader's dataset length, with a ShardedSampler and
    with torch's default sampler, with and without drop_last."""
    dataset = chunk_dataset(with_mask=True)
    for start_offset in [0, 7]:
        dataset.recompute_chunk_indices(start_offset)
        for drop_last in [False, True]:
            for sampler in [distributed.ShardedSampler(dataset, shuffle=True), None]:
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, sampler=sampler, drop_last=drop_last)
                assert steps_per_epoch(dataloader, len(dataset)) == len(dataloader)


def test_training_step_count():
    """count_training_steps gives the first epoch's steps plus the mean over the start offsets for each
    later epoch, which is what the learning rate schedule needs to end at 0 on train()'s last step."""
    dataset = chunk_dataset(with_mask=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, sampler=distributed.ShardedSampler(dataset, True))
    first_epoch_steps = len(dataloader)
    later_epoch_steps = []
    for start_offset in range(CHUNK_LENGTH):
        dataset.recompute_chunk_indices(start_offset)
        later_epoch_steps.append(len(dataloader))
    dataset.recompute_chunk_indices(0)
    assert count_training_steps(dataloader, 1) == first_epoch_steps
    assert count_training_steps(dataloader, 101) == first_epoch_steps + round(100 * np.mean(later_epoch_steps))


if __name__ == '__main__':
    testing_utils.run_as_script(globals())
