"""The tasks of an evaluation run, their dependencies and completion state."""
import os
import socket
import time
from dataclasses import dataclass

from apf.eval_pipeline.params import EvalParams, RunPaths, write_json


@dataclass(frozen=True)
class Task:
    name: str
    n_jobs: int
    gpu: bool
    deps: tuple[str, ...] = ()


def build_graph(params: EvalParams) -> list[Task]:
    """Returns the tasks in an order where dependencies come first."""
    return [
        Task('prepare', 1, False),
        Task('simulate', params.n_simulate_jobs, True, ('prepare',)),
        Task('train_features', params.n_feature_jobs, True, ('prepare',)),
        Task('stitch', 1, False, ('simulate',)),
        Task('probes', params.n_probe_jobs, True, ('train_features',)),
        Task('logits', params.n_logit_jobs, True, ('simulate', 'probes')),
        Task('merge_logits', 1, False, ('logits', 'stitch')),
        Task('probe_metrics', 1, False, ('merge_logits',)),
        Task('action_stats', 1, False, ('merge_logits',)),
        Task('action_videos', params.n_video_jobs, False, ('merge_logits',)),
        Task('distributions', 1, False, ('stitch',)),
        Task('discrimination', 1, True, ('stitch',)),
        Task('sim_videos', 1, False, ('stitch',)),
        Task('report', 1, False, ('probe_metrics', 'action_stats', 'action_videos', 'distributions',
                                  'discrimination', 'sim_videos')),
    ]


def marker_path(paths: RunPaths, task_name: str, shard: int) -> str:
    return os.path.join(paths.done_dir, f"{task_name}.{shard}")


def mark_done(paths: RunPaths, task_name: str, shard: int):
    write_json(marker_path(paths, task_name, shard), {'host': socket.gethostname(), 'time': time.time()})


def is_done(paths: RunPaths, task_name: str, shard: int) -> bool:
    return os.path.exists(marker_path(paths, task_name, shard))


def missing_shards(paths: RunPaths, task: Task) -> list[int]:
    """Array elements (1-indexed) of the task that have not finished."""
    return [shard for shard in range(1, task.n_jobs + 1) if not is_done(paths, task.name, shard)]


def missing_dependencies(paths: RunPaths, task: Task, graph: list[Task]) -> list[str]:
    by_name = {t.name: t for t in graph}
    return [dep for dep in task.deps if missing_shards(paths, by_name[dep])]
