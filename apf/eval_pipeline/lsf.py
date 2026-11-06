"""Submitting and inspecting the task graph on LSF.

The graph is submitted in one go with `ended()` dependencies, so nothing has to stay alive after submission. A task
checks its own inputs when it starts, so a failure upstream shows up as a clear message rather than bad results.
"""
import os
import shlex
import shutil
import subprocess
import sys

from apf.eval_pipeline.graph import Task, missing_shards
from apf.eval_pipeline.params import Resources, RunPaths

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTIVE_STATES = {'RUN', 'PROV', 'WAIT', 'USUSP', 'SSUSP'}
PENDING_STATES = {'PEND', 'PSUSP'}


def run_lsf(cmd: str) -> str:
    """Runs an LSF command directly when available, otherwise through the login node."""
    if shutil.which(cmd.split()[0]):
        res = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
    else:
        res = subprocess.run(['ssh', 'login1', 'bash -lc ' + shlex.quote(cmd)], capture_output=True, text=True)
    # bjobs and bkill exit non-zero when no job matches, which is a normal outcome here
    if res.returncode != 0 and not cmd.startswith(('bjobs', 'bkill')):
        raise RuntimeError(f"Command failed: {cmd}\n{res.stdout}\n{res.stderr}")
    return res.stdout


def job_name(paths: RunPaths, task_name: str) -> str:
    return f"{paths.job_prefix()}_{task_name}"


def array_spec(shards: list[int]) -> str:
    """Compresses shard ids into LSF array syntax, e.g. [1,2,3,7] -> '[1-3,7]'."""
    parts = []
    start = prev = shards[0]
    for shard in shards[1:] + [None]:
        if shard is not None and shard == prev + 1:
            prev = shard
            continue
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = shard
    return '[' + ','.join(parts) + ']'


def parse_job_elements(output: str, prefix: str) -> dict[str, dict[int, tuple[str, str]]]:
    """Parses `bjobs -o 'jobid stat job_name'` into {task: {shard: (state, id to pass to bkill)}}."""
    jobs = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3 or not fields[2].startswith(prefix + '_'):
            continue
        jobid, state, name = fields
        name = name[len(prefix) + 1:]
        shard = 1
        if name.endswith(']'):
            name, _, index = name[:-1].rpartition('[')
            shard = int(index)
            jobid = f"{jobid}[{index}]"
        jobs.setdefault(name, {})[shard] = (state, jobid)
    return jobs


def query_jobs(paths: RunPaths) -> dict[str, dict[int, tuple[str, str]]]:
    prefix = paths.job_prefix()
    output = run_lsf(f"bjobs -noheader -o 'jobid stat job_name' -J '{prefix}_*' 2>/dev/null")
    return parse_job_elements(output, prefix)


def worker_command(paths: RunPaths, task_name: str, array: bool) -> str:
    shard = '$LSB_JOBINDEX' if array else '1'
    python = shlex.quote(sys.executable)
    return (f"unset XDG_RUNTIME_DIR && export MPLBACKEND=Agg && cd {shlex.quote(REPO_ROOT)} && "
            f"{python} -u -m apf.eval_pipeline.worker {task_name} {shlex.quote(paths.run_dir)} --shard {shard}")


def bsub_command(paths: RunPaths, task: Task, shards: list[int], resources: Resources, depends_on: list[str]) -> str:
    override = resources.overrides.get(task.name, {})
    queue = override.get('queue', resources.queue if task.gpu else resources.cpu_queue)
    slots = override.get('slots', resources.gpu_slots if task.gpu else resources.cpu_slots)
    walltime = override.get('walltime', resources.walltime)

    name = job_name(paths, task.name)
    is_array = task.n_jobs > 1
    full_name = name + array_spec(shards) if is_array else name
    log_name = f"{task.name}.%I.log" if is_array else f"{task.name}.1.log"

    parts = ['bsub', '-n', str(slots)]
    if task.gpu:
        parts += ['-gpu', shlex.quote('num=1')]
    parts += ['-q', shlex.quote(queue), '-W', shlex.quote(walltime), '-J', shlex.quote(full_name),
              '-o', shlex.quote(os.path.join(paths.log_dir, log_name))]
    if depends_on:
        condition = ' && '.join(f"ended({job_name(paths, dep)})" for dep in depends_on)
        parts += ['-w', shlex.quote(condition)]
    parts.append(shlex.quote(worker_command(paths, task.name, is_array)))
    return ' '.join(parts)


def plan_submission(paths: RunPaths, graph: list[Task], jobs: dict) -> list[tuple[Task, list[int], list[str]]]:
    """Decides what to submit: (task, shards, names of tasks to wait for).

    Running elements are left alone. A dependency is only waited for if it is going to have a job, since LSF rejects
    a dependency on a job name it has never seen.
    """
    plan = []
    active_tasks = set()
    for task in graph:
        running = {shard for shard, (state, _) in jobs.get(task.name, {}).items() if state in ACTIVE_STATES}
        shards = [shard for shard in missing_shards(paths, task) if shard not in running]
        if running:
            active_tasks.add(task.name)
        if shards:
            plan.append((task, shards, [dep for dep in task.deps if dep in active_tasks]))
            active_tasks.add(task.name)
    return plan


def kill_pending(paths: RunPaths, jobs: dict) -> int:
    """Kills pending elements. They wait on an earlier submission's jobs and are resubmitted against the current ones."""
    n_killed = 0
    for elements in jobs.values():
        for state, jobid in elements.values():
            if state in PENDING_STATES:
                run_lsf(f"bkill '{jobid}'")
                n_killed += 1
    return n_killed


def submit(paths: RunPaths, graph: list[Task], resources: Resources, dry_run: bool = False) -> list[tuple[str, list[int]]]:
    """Submits whatever is missing, returning (task name, shards) of what was submitted."""
    jobs = query_jobs(paths)
    if dry_run:
        # Pending jobs would be killed, so plan as if they are already gone
        jobs_after_kill = {name: {s: v for s, v in els.items() if v[0] in ACTIVE_STATES} for name, els in jobs.items()}
    else:
        kill_pending(paths, jobs)
        jobs_after_kill = jobs

    submitted = []
    for task, shards, deps in plan_submission(paths, graph, jobs_after_kill):
        cmd = bsub_command(paths, task, shards, resources, deps)
        if dry_run:
            print(cmd)
        else:
            print(run_lsf(cmd).strip())
        submitted.append((task.name, shards))
    return submitted


def tail(path: str, n_lines: int = 8) -> str:
    try:
        with open(path, errors='replace') as f:
            return ''.join(f.readlines()[-n_lines:]).replace('\r', '\n').rstrip()
    except OSError:
        return ''


def status_lines(paths: RunPaths, graph: list[Task], jobs: dict) -> list[str]:
    """One line per task, followed by the end of the log of anything that ran and did not finish."""
    lines = []
    failures = []
    for task in graph:
        missing = missing_shards(paths, task)
        elements = jobs.get(task.name, {})
        n_running = sum(state in ACTIVE_STATES for state, _ in elements.values())
        n_pending = sum(state in PENDING_STATES for state, _ in elements.values())
        lines.append(f"{task.name:16s} done {task.n_jobs - len(missing):3d}/{task.n_jobs:<3d} "
                     f"running {n_running:3d}  pending {n_pending:3d}")
        for shard in missing:
            if shard in elements and elements[shard][0] not in ('EXIT',):
                continue
            log = os.path.join(paths.log_dir, f"{task.name}.{shard}.log")
            if os.path.exists(log):
                failures.append((task.name, shard, log))
    for name, shard, log in failures:
        lines.append(f"\n--- not finished: {name}[{shard}] {log}\n{tail(log)}")
    return lines
