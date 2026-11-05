"""
Helpers for training on several GPUs with DistributedDataParallel (DDP), one process per GPU.

The processes are forked from a parent that has already loaded the datasets, so they share the datasets' arrays
copy-on-write, rather than each holding its own copy.  CUDA can't be used in a process forked from one that has
initialized it, so the parent must not touch CUDA before forking: set PYTORCH_NVML_BASED_CUDA_CHECK=1 before
importing torch, so that torch.cuda.is_available() doesn't initialize it, and create no CUDA tensors.

The helpers other than run_on_gpus also work when torch.distributed isn't initialized, as one process with
rank 0 in a world of size 1, so that code using them also runs as before outside of run_on_gpus.
"""

import logging
import math
import os
import socket
from typing import Callable, Iterator

import torch
import torch.distributed as dist

LOG = logging.getLogger(__name__)


def rank() -> int:
    """ This process's rank, or 0 if torch.distributed isn't initialized. """
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    """ The number of processes, or 1 if torch.distributed isn't initialized. """
    return dist.get_world_size() if dist.is_initialized() else 1


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """ Returns the sum of x over all processes, as a new tensor on x's device.  x must be on the process's GPU
    if torch.distributed is initialized, since NCCL works only on GPU tensors. """
    total = x.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total


def all_true(x: torch.Tensor) -> bool:
    """ Whether the boolean scalar tensor x is true in every process.  Syncs with the GPU. """
    x_int = x.detach().to(torch.int32)
    if dist.is_initialized():
        dist.all_reduce(x_int, op=dist.ReduceOp.MIN)
    return bool(x_int.item())


def broadcast_int(value: int, device: torch.device) -> int:
    """ Returns rank 0's value, in every process.  device is the process's GPU. """
    if not dist.is_initialized():
        return value
    x = torch.tensor(value, dtype=torch.int64, device=device)
    dist.broadcast(x, src=0)
    return int(x.item())


class ShardedSampler(torch.utils.data.Sampler):
    """
    Samples this process's share of a dataset's indices, so that the processes together cover the dataset once
    per epoch.  Like torch's DistributedSampler, but the dataset's length is read at the start of each epoch,
    since the training dataset's length changes when it retiles its chunks after each epoch.  (DistributedSampler
    reads it once, when it's created.)

    If shuffle is True, the indices are permuted with a generator seeded from (seed, epoch), the same in every
    process, and the permutation is padded by wrapping around, so that every process gets the same number of
    indices, and so takes the same number of steps.  Call set_epoch at the start of each epoch.  If shuffle is
    False, process r gets indices r, r + world size, ..., with no padding, so the dataset is covered exactly
    once, but the processes can get different numbers of indices.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, shuffle: bool, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.rank = rank()
        self.world_size = world_size()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.shuffle:
            return math.ceil(n / self.world_size)
        return len(range(self.rank, n, self.world_size))

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        if not self.shuffle:
            return iter(range(self.rank, n, self.world_size))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(n, generator=generator).tolist()
        padded_size = math.ceil(n / self.world_size) * self.world_size
        indices += indices[:padded_size - n]
        return iter(indices[self.rank:padded_size:self.world_size])


def free_local_port() -> int:
    """ A TCP port on 127.0.0.1 that's free now, for the processes to rendezvous on. """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _run_on_gpu(process_rank: int, gpu_count: int, port: int, fn: Callable, args: tuple) -> None:
    """ Runs in each forked process: sets up its GPU and torch.distributed, then calls fn(device, *args). """
    device = torch.device('cuda', process_rank)
    torch.cuda.set_device(device)
    # split the CPU threads among the processes
    torch.set_num_threads(max(1, torch.get_num_threads() // gpu_count))
    # prefix log lines with the rank, since all the processes log to the same files
    logging.basicConfig(level=logging.getLogger().level, force=True,
                        format=f'[rank {process_rank}] %(asctime)s %(levelname)s:%(name)s:%(message)s')
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{port}', rank=process_rank,
                            world_size=gpu_count, device_id=device)
    try:
        fn(device, *args)
    finally:
        dist.destroy_process_group()


def run_on_gpus(fn: Callable, gpu_count: int, args: tuple = ()) -> None:
    """
    Forks one process per GPU, for GPUs 0 to gpu_count - 1, and in each one initializes torch.distributed (NCCL)
    and calls fn(device, *args), where device is that process's GPU.  Returns once all the processes are done,
    and raises if any of them raised.  The processes share the parent's memory copy-on-write.

    Raises if CUDA was initialized in this process, since CUDA doesn't work in processes forked after that.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was initialized before forking the training processes, so they could not use it')
    if gpu_count < 1:
        raise ValueError(f'gpu_count is {gpu_count}, but must be at least 1')
    port = free_local_port()
    LOG.info(f'Forking {gpu_count} training process(es), one per GPU, rendezvousing on port {port}')
    torch.multiprocessing.start_processes(_run_on_gpu, args=(gpu_count, port, fn, args), nprocs=gpu_count,
                                          start_method='fork')
