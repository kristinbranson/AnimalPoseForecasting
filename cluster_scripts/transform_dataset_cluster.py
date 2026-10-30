"""Cluster-parallel version of transform_dataset.

The Janelia LSF array-job runs N parallel workers. Work is divided into chunks
of ``frames_per_chunk`` consecutive valid frames per agent, giving
``n_agents × ceil(valid_frames / frames_per_chunk)`` chunks in total.
Each worker processes the chunks where ``chunk_idx % n_workers == worker_id - 1``,
mirroring the simulate_tracks_cluster pattern.

Each worker writes its results directly into the shared ``transformed`` memmap
at ``transformed_path`` — no two workers ever touch the same (agent, frame)
slot, so there is no locking and no stitching step required.

The memmap is **preallocated** by the notebook-side function
``transform_dataset_cluster`` before workers start, exactly as is done in the
original ``transform_dataset``.

Notebook usage:

    from transform_dataset_cluster import transform_dataset_cluster
    transformed, info = transform_dataset_cluster(
        dataset, transformer, transformed_path,
        configfile=configfile, modelfile=modelfile,
        n_jobs=16, queue='gpu_a100', walltime='12:00',
        batch_size=64, frames_per_chunk=50000, save_layers=None,
        wait=True,
    )

CLI worker usage (invoked by bsub on each array task):
    python transform_dataset_cluster.py worker --configfile ... --modelfile ... \\
        --transformed_path ... --worker_id $LSB_JOBINDEX --n_workers 16 \\
        --frames_per_chunk 50000
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from typing import List, Optional
from tqdm.auto import tqdm

import numpy as np

# Make the repo importable when this file is run as a CLI on the cluster.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_shape(dataset, transformer):
    """Return (n_layers, n_agents, n_frames, d_model, d_input, contextl)."""
    d_model = transformer.d_model
    n_layers = len(transformer.transformer_encoder.layers)
    d_input = dataset.d_input
    n_agents, n_frames = dataset.inputs['velocity'].array.shape[:2]
    contextl = dataset.context_length
    return n_layers, n_agents, n_frames, d_model, d_input, contextl


def _resolve_save_layers(save_layers, n_layers):
    if save_layers is None:
        return np.arange(n_layers)
    elif save_layers == 'some':
        return [0, int(round(n_layers / 2)), n_layers - 1]
    return save_layers


def _collect_frames_per_agent(dataset, contextl):
    """Return a dict mapping agent_id -> sorted array of valid frame ids."""
    agent_frames: dict = {}
    for session in dataset.sessions:
        aid = session.agent_id
        sframes = np.arange(session.start_frame + contextl - 1,
                            session.start_frame + session.duration)
        if aid not in agent_frames:
            agent_frames[aid] = []
        agent_frames[aid].append(sframes)
    return {aid: np.concatenate(flist) for aid, flist in agent_frames.items()}


def _enumerate_chunks(dataset, contextl, frames_per_chunk):
    """Return list of (agent_id, frame_array) tuples.

    Each chunk covers at most ``frames_per_chunk`` consecutive valid frames
    for one agent.  Chunks are ordered agent-major so that a small n_workers
    value still spreads load evenly across agents.
    """
    agent_frames = _collect_frames_per_agent(dataset, contextl)
    chunks = []
    for agent_id in sorted(agent_frames):
        frames = agent_frames[agent_id]
        for start in range(0, len(frames), frames_per_chunk):
            chunks.append((agent_id, frames[start:start + frames_per_chunk]))
    return chunks


# ---------------------------------------------------------------------------
# Worker (runs on a cluster GPU node)
# ---------------------------------------------------------------------------

def transform_dataset_worker(
    dataset, transformer, transformed_path,
    *, batch_size=64, frames_per_chunk=50000, overwrite=False, save_layers=None,
    worker_id=1, n_workers=1,
):
    """Compute and write the chunks assigned to (worker_id, n_workers).

    worker_id is 1-indexed to match LSB_JOBINDEX.  The full set of
    (agent_id, frame_range) chunks is enumerated deterministically and each
    worker processes those where ``chunk_idx % n_workers == worker_id - 1``.
    No two workers ever write to the same (agent, frame) slot.

    The memmap at ``transformed_path`` must already exist (preallocated by
    ``transform_dataset_cluster``).
    """
    import torch

    device = next(transformer.parameters()).device
    transformer.eval()

    n_layers, n_agents, n_frames, d_model, d_input, contextl = _get_shape(dataset, transformer)
    resolved_save_layers = _resolve_save_layers(save_layers, n_layers)

    if not os.path.exists(transformed_path):
        raise FileNotFoundError(
            f"transformed_path {transformed_path!r} must be preallocated by "
            f"transform_dataset_cluster before workers start.")
    transformed = np.memmap(transformed_path, dtype=np.float32, mode='r+',
                            shape=(n_layers, n_agents, n_frames, d_model))

    all_chunks = _enumerate_chunks(dataset, contextl, frames_per_chunk)
    my_chunks = [(agent_id, chunk_frames)
                 for idx, (agent_id, chunk_frames) in enumerate(all_chunks)
                 if idx % n_workers == (worker_id - 1)]
    total_frames = sum(len(f) for _, f in my_chunks)
    print(f"[worker {worker_id}/{n_workers}] "
          f"{len(my_chunks)}/{len(all_chunks)} chunks, {total_frames} frame-slots",
          flush=True)

    batch_inputs = np.zeros((batch_size, contextl, d_input), np.float32)
    batch_indices = np.zeros((2, batch_size), int)
    batch_idx = 0

    def _flush(n):
        if n == 0:
            return
        with torch.no_grad():
            layer_outs = transformer.forward(
                torch.from_numpy(batch_inputs[:n]).to(device),
                return_hidden=True
            )['hidden']
        bagents = batch_indices[0, :n]
        bframes = batch_indices[1, :n]
        for layer_id, out in enumerate(layer_outs):
            if layer_id in resolved_save_layers:
                transformed[layer_id, bagents, bframes, :] = out[:, -1]  #.detach().cpu().numpy()
        transformed.flush()

    for chunk_num, (agent_id, chunk_frames) in enumerate(my_chunks):
        print(f"  [worker {worker_id}/{n_workers}] chunk {chunk_num + 1}/{len(my_chunks)} "
              f"agent={agent_id} frames={chunk_frames[0]}..{chunk_frames[-1]}",
              flush=True)
        for frame_id in tqdm(chunk_frames):
            if not overwrite and np.abs(transformed[:, agent_id, frame_id, :]).sum() > 0:
                # Already written — skip (matches original logic).
                continue
            batch_inputs[batch_idx] = dataset.get_chunk(
                start_frame=frame_id - contextl + 1, duration=contextl, agent_id=agent_id
            )['input']
            batch_indices[0, batch_idx] = agent_id
            batch_indices[1, batch_idx] = frame_id
            batch_idx += 1
            if batch_idx == batch_size:
                _flush(batch_idx)
                batch_idx = 0

    # Flush any remaining partial batch.
    _flush(batch_idx)
    print(f"[worker {worker_id}/{n_workers}] done", flush=True)
    return transformed


# ---------------------------------------------------------------------------
# Cluster submission (called from the notebook)
# ---------------------------------------------------------------------------

def _ssh_login1(cmd: str) -> str:
    """Run a command on login1 and return stdout."""
    full = ["ssh", "login1", "bash -lc " + shlex.quote(cmd)]
    res = subprocess.run(full, capture_output=True, text=True, check=True)
    return res.stdout


_TERMINAL_STATES = {"DONE", "EXIT", "UNKWN", "ZOMBI"}
_ACTIVE_STATES = {"PEND", "RUN", "PROV", "WAIT", "USUSP", "SSUSP", "PSUSP"}


def _parse_bjobs_rows(out: str):
    rows = []
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if not parts[0].isdigit():
            continue
        if len(parts) < 2:
            continue
        rows.append((parts[0], parts[1]))
    return rows


def _wait_for_jobid(jobid: str, poll_sec: int = 30) -> str:
    """Poll bjobs until jobid leaves any active state. Returns final summary text."""
    while True:
        out = _ssh_login1(
            f"bjobs -a -o 'jobid stat exit_code' -noheader {jobid} 2>/dev/null || true")
        rows = _parse_bjobs_rows(out)
        if not rows:
            return out
        states = {stat for _, stat in rows}
        if not (states & _ACTIVE_STATES):
            return out
        time.sleep(poll_sec)


def check_job_status(jobid=None, jobname=None, transformed_path=None,
                     tail_lines=40, verbose=True):
    """Check status of a running or finished transform_dataset_cluster job.

    Pass any combination of:
      - jobid: LSF job id (e.g. info['jobid'])
      - jobname: job name pattern (e.g. 'transform_smoke')
      - transformed_path: path to the memmap; used to locate worker logs

    When tasks have exited with EXIT status, prints the tail of each failed
    task's log file so you can see the actual error (e.g. CUDA OOM).

    Returns a dict with keys: states, tasks, failed_logs (dict of
    task_jobid -> log tail string).
    """
    info = {"states": {}, "tasks": [], "failed_logs": {}}

    sel = []
    if jobid is not None:
        sel.append(str(jobid))
    if jobname is not None:
        sel.append(f"-J {shlex.quote(jobname)}")
    selector = " ".join(sel) if sel else "-u $USER"

    out = _ssh_login1(
        f"bjobs -a -o 'jobid stat queue exec_host job_name submit_time' "
        f"-noheader {selector} 2>&1 || true")
    rows = []
    for ln in out.strip().splitlines():
        ln = ln.strip()
        if not ln or ln.lower().startswith("no ") or ln.startswith("bashrc"):
            continue
        parts = ln.split(None, 5)
        if len(parts) < 2:
            continue
        rows.append({
            "jobid": parts[0], "stat": parts[1],
            "queue": parts[2] if len(parts) > 2 else "",
            "host": parts[3] if len(parts) > 3 else "",
            "name": parts[4] if len(parts) > 4 else "",
            "submit": parts[5] if len(parts) > 5 else "",
        })
    info["tasks"] = rows
    for r in rows:
        info["states"][r["stat"]] = info["states"].get(r["stat"], 0) + 1

    # Locate log directory.
    log_dir = None
    if transformed_path is not None:
        candidate = os.path.join(os.path.dirname(os.path.abspath(transformed_path)),
                                 "cluster_logs")
        if os.path.isdir(candidate):
            log_dir = candidate

    print(f"log_dir = {log_dir}")

    # For each failed task, find its log by matching jobid in the filename
    # (LSF writes <jobname>_<arrayid>_<taskindex>.log).
    # We match the array job id and task index exactly to avoid false matches
    # when multiple jobs with similar IDs are present (e.g. 12345 vs 123456).
    failed_tasks = [r["jobid"] for r in rows if r["stat"] == "EXIT"]
    if log_dir and failed_tasks:
        import re
        all_logs = [os.path.join(log_dir, f)
                    for f in os.listdir(log_dir) if f.endswith(".log")]
        for task_jid in failed_tasks:
            # task_jid looks like "12345[3]" — extract array id and task index.
            m = re.match(r"(\d+)(?:\[(\d+)\])?", task_jid)
            if not m:
                continue
            base_jid, task_idx = m.group(1), m.group(2)

            # LSF log pattern: <jobname>_<arrayid>_<taskindex>.log
            # Use exact token matching: arrayid and taskindex must be surrounded
            # by underscores or start/end of the stem, never substrings of digits.
            def _log_matches(path):
                stem = os.path.splitext(os.path.basename(path))[0]
                parts = stem.split("_")
                if base_jid not in parts:
                    return False
                if task_idx is not None and task_idx not in parts:
                    return False
                # Both tokens must appear at the correct relative positions:
                # stem ends with ..._<arrayid>_<taskindex>
                try:
                    jid_pos = len(parts) - 1 - parts[::-1].index(base_jid)
                    if task_idx is not None:
                        return (jid_pos == len(parts) - 2 and
                                parts[len(parts) - 1] == task_idx)
                    return True
                except ValueError:
                    return False

            matches = [p for p in all_logs if _log_matches(p)]
            matches.sort(key=lambda p: os.path.getmtime(p))
            if matches:
                log_path = matches[-1]
                try:
                    with open(log_path, "r", errors="replace") as fh:
                        lines = fh.readlines()
                    info["failed_logs"][task_jid] = (log_path, "".join(lines[-tail_lines:]))
                except OSError:
                    pass

    # If no failed tasks but there are logs, show the tail of the most recent one
    # that belongs to this job (matched by array job id token in the filename).
    if not failed_tasks and log_dir:
        import re
        base_jid = re.match(r"(\d+)", str(jobid)).group(1) if jobid else None
        all_logs = sorted(
            (os.path.join(log_dir, f) for f in os.listdir(log_dir)
             if f.endswith(".log")
             and (base_jid is None or
                  base_jid in os.path.splitext(os.path.basename(f))[0].split("_"))),
            key=lambda p: os.path.getmtime(p))
        if all_logs:
            try:
                with open(all_logs[-1], "r", errors="replace") as fh:
                    lines = fh.readlines()
                info["latest_log"] = (all_logs[-1], "".join(lines[-tail_lines:]))
            except OSError:
                pass

    if verbose:
        if not rows:
            print("(no matching jobs in bjobs — already aged out, or wrong selector)")
        else:
            print(f"{'JOBID':<14}{'STAT':<6}{'QUEUE':<10}{'HOST':<14}{'NAME':<24}SUBMIT")
            for r in rows:
                print(f"{r['jobid']:<14}{r['stat']:<6}{r['queue']:<10}"
                      f"{r['host'][:13]:<14}{r['name'][:23]:<24}{r['submit']}")
            print(f"  states: {info['states']}")

        if info["failed_logs"]:
            print(f"\n  {len(info['failed_logs'])} failed task(s) — last {tail_lines} lines each:")
            for task_jid, (log_path, tail) in info["failed_logs"].items():
                print(f"\n  {'=' * 60}")
                print(f"  task {task_jid}  log: {log_path}")
                print(f"  {'=' * 60}")
                print(tail)
        elif info.get("latest_log"):
            log_path, tail = info["latest_log"]
            print(f"\n  latest log: {log_path}")
            print(f"  --- last {tail_lines} lines ---")
            print(tail)

    return info


def check_job_status_old(jobid=None, jobname=None, transformed_path=None,
                     tail_lines=20, verbose=True):
    """Check status of a running transform_dataset_cluster job.

    Pass any combination of:
      - jobid: LSF job id (e.g. info['jobid'])
      - jobname: job name pattern (e.g. 'transform_smoke')
      - transformed_path: path to the memmap; used to locate worker logs

    Returns a dict with keys: states, tasks, log_tail.
    """
    info = {"states": {}, "tasks": [], "log_tail": None}

    sel = []
    if jobid is not None:
        sel.append(str(jobid))
    if jobname is not None:
        sel.append(f"-J {shlex.quote(jobname)}")
    selector = " ".join(sel) if sel else "-u $USER"

    out = _ssh_login1(
        f"bjobs -a -o 'jobid stat queue exec_host job_name submit_time' "
        f"-noheader {selector} 2>&1 || true")
    rows = []
    for ln in out.strip().splitlines():
        ln = ln.strip()
        if not ln or ln.lower().startswith("no ") or ln.startswith("bashrc"):
            continue
        parts = ln.split(None, 5)
        if len(parts) < 2:
            continue
        rows.append({
            "jobid": parts[0], "stat": parts[1],
            "queue": parts[2] if len(parts) > 2 else "",
            "host": parts[3] if len(parts) > 3 else "",
            "name": parts[4] if len(parts) > 4 else "",
            "submit": parts[5] if len(parts) > 5 else "",
        })
    info["tasks"] = rows
    for r in rows:
        info["states"][r["stat"]] = info["states"].get(r["stat"], 0) + 1

    if transformed_path is not None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(transformed_path)),
                               "cluster_logs")
        if os.path.isdir(log_dir):
            logs = sorted(
                (os.path.join(log_dir, f) for f in os.listdir(log_dir)
                 if f.endswith(".log")),
                key=lambda p: os.path.getmtime(p))
            if logs:
                latest = logs[-1]
                try:
                    with open(latest, "r", errors="replace") as fh:
                        lines = fh.readlines()
                    info["log_tail"] = "".join(lines[-tail_lines:])
                    info["log_path"] = latest
                except OSError:
                    pass

    if verbose:
        if not rows:
            print("(no matching jobs in bjobs — already aged out, or wrong selector)")
        else:
            print(f"{'JOBID':<12}{'STAT':<6}{'QUEUE':<10}{'HOST':<14}{'NAME':<24}SUBMIT")
            for r in rows:
                print(f"{r['jobid']:<12}{r['stat']:<6}{r['queue']:<10}"
                      f"{r['host'][:13]:<14}{r['name'][:23]:<24}{r['submit']}")
            print(f"  states: {info['states']}")
        if info.get("log_path"):
            print(f"  log: {info['log_path']}")
            print("  --- last lines ---")
            print(info["log_tail"])
    return info


def transform_dataset_cluster(
    dataset, transformer, transformed_path,
    *,
    configfile: str,
    modelfile: str,
    n_jobs: int = 8,
    batch_size: int = 64,
    frames_per_chunk: int = 50000,
    overwrite: bool = False,
    save_layers=None,
    queue: str = "gpu_a100",
    slots: int = 12,
    walltime: str = "24:00",
    jobname: str = "transform",
    conda_env: str = "transformer312",
    conda_path: str = "$HOME/miniforge3/etc/profile.d/conda.sh",
    repo_root: Optional[str] = None,
    extra_python_args: Optional[List[str]] = None,
    wait: bool = True,
    poll_sec: int = 30,
    submit_only: bool = False,
) -> Optional[np.memmap]:
    """Submit an LSF array job to run transform_dataset in parallel.

    Work is split into chunks of ``frames_per_chunk`` valid frames per agent.
    With 5 agents and frames_per_chunk=50000, a 10M-frame dataset produces
    ~1000 chunks that can be spread across many workers regardless of the
    small agent count.

    Preallocates the shared memmap at ``transformed_path`` (if absent), then
    submits ``n_jobs`` array tasks that each process a disjoint set of chunks
    and write directly into the memmap.

    Returns (transformed, info) once all workers finish, or (None, info) if
    ``wait=False`` / ``submit_only=True``.  In the latter case use
    ``open_transformed`` to get the handle after the job completes.

    All paths (``transformed_path``, ``configfile``, ``modelfile``) must be
    cluster-visible (e.g. under /groups, /nrs, or /misc).
    """
    repo_root = repo_root or _REPO_ROOT
    log_dir = os.path.join(os.path.dirname(os.path.abspath(transformed_path)),
                           "cluster_logs")
    os.makedirs(log_dir, exist_ok=True)

    n_layers, n_agents, n_frames, d_model, d_input, contextl = _get_shape(
        dataset, transformer)
    expected_bytes = int(n_layers * n_agents * n_frames * d_model *
                         np.dtype(np.float32).itemsize)

    # Preallocate (or validate) the shared memmap.
    if not os.path.exists(transformed_path):
        print(f"[transform_dataset_cluster] preallocating memmap "
              f"shape=({n_layers},{n_agents},{n_frames},{d_model}) "
              f"({expected_bytes / 1e9:.2f} GB) at {transformed_path}")
        mm = np.memmap(transformed_path, dtype=np.float32, mode='w+',
                       shape=(n_layers, n_agents, n_frames, d_model))
        mm.flush()
        del mm
    else:
        actual_bytes = os.path.getsize(transformed_path)
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"Existing transformed_path {transformed_path!r} has size "
                f"{actual_bytes} bytes, but current shape "
                f"(n_layers={n_layers}, n_agents={n_agents}, n_frames={n_frames}, "
                f"d_model={d_model}, float32) expects {expected_bytes} bytes. "
                f"Refusing to reuse — delete the file or pick a different path.")
        print(f"[transform_dataset_cluster] reusing existing memmap at "
              f"{transformed_path} ({actual_bytes / 1e9:.2f} GB) — "
              f"workers will skip already-written slices.")

    all_chunks = _enumerate_chunks(dataset, contextl, frames_per_chunk)
    n_chunks = len(all_chunks)
    n_jobs = max(1, min(n_jobs, n_chunks))
    print(f"[transform_dataset_cluster] {n_agents} agents, {n_chunks} chunks "
          f"({frames_per_chunk} frames/chunk) -> {n_jobs} array tasks")

    # Build the per-task python command.
    py_script = os.path.join(repo_root, "cluster_scripts", "transform_dataset_cluster.py")
    py_args = [
        "worker",
        "--configfile", configfile,
        "--modelfile", modelfile,
        "--transformed_path", transformed_path,
        "--n_workers", str(n_jobs),
        "--worker_id", "$LSB_JOBINDEX",
        "--batch_size", str(batch_size),
        "--frames_per_chunk", str(frames_per_chunk),
    ]
    if overwrite:
        py_args += ["--overwrite"]
    if save_layers is not None:
        py_args += ["--save_layers", str(save_layers)]
    if extra_python_args:
        py_args += list(extra_python_args)

    py_cmd_parts = ["python", py_script] + [
        a if a == "$LSB_JOBINDEX" else shlex.quote(a) for a in py_args
    ]
    py_cmd = " ".join(py_cmd_parts)

    inner_cmd = (
        "unset XDG_RUNTIME_DIR && "
        f"cd {shlex.quote(repo_root)} && "
        f"{py_cmd}"
    )

    log_pattern = os.path.join(log_dir, f"{jobname}_%J_%I.log")
    bsub_cmd = (
        f"bsub -n {slots} -gpu 'num=1' -q {shlex.quote(queue)} "
        f"-W {shlex.quote(walltime)} "
        f"-J {shlex.quote(f'{jobname}[1-{n_jobs}]')} "
        f"-o {shlex.quote(log_pattern)} "
        + shlex.quote(inner_cmd)
    )
    print(f"[transform_dataset_cluster] submitting: {bsub_cmd}")
    out = _ssh_login1(bsub_cmd)
    print(out.strip())

    # Parse job id.
    jobid = None
    for tok in out.split():
        if tok.startswith("<") and tok.endswith(">") and tok[1:-1].isdigit():
            jobid = tok[1:-1]
            break
    info = {"jobid": jobid, "n_chunks": n_chunks, "n_agents": n_agents, "n_jobs": n_jobs,
            "log_pattern": log_pattern, "transformed_path": transformed_path}

    if submit_only or not wait:
        return None, info

    if jobid is None:
        raise RuntimeError(f"Could not parse LSF job id from bsub output:\n{out}")
    print(f"[transform_dataset_cluster] waiting for job {jobid} (poll every {poll_sec}s)")
    final = _wait_for_jobid(jobid, poll_sec=poll_sec)
    print(f"[transform_dataset_cluster] job {jobid} finished:\n{final}")

    final_rows = _parse_bjobs_rows(final)
    final_states = [stat for _, stat in final_rows]
    failed_tasks = [jid for jid, stat in final_rows if stat == "EXIT"]
    info["task_states"] = final_states
    info["failed_tasks"] = failed_tasks

    if failed_tasks:
        logs = []
        try:
            logs = sorted(
                (os.path.join(log_dir, f) for f in os.listdir(log_dir)
                 if f.endswith(".log") and jobid in f),
                key=lambda p: os.path.getmtime(p))
        except OSError:
            pass
        log_hint = f" Latest log: {logs[-1]}" if logs else ""
        raise RuntimeError(
            f"LSF job {jobid}: {len(failed_tasks)}/{len(final_rows)} tasks "
            f"ended in EXIT (failed: {failed_tasks}). Check the worker logs in "
            f"{log_dir}.{log_hint}"
        )

    return open_transformed(dataset, transformer, transformed_path), info


def open_transformed(dataset, transformer, transformed_path) -> np.memmap:
    """Open the completed memmap read-only.

    Convenience function for use after a fire-and-forget submission
    (``wait=False``) once the job has finished.
    """
    n_layers, n_agents, n_frames, d_model, *_ = _get_shape(dataset, transformer)
    return np.memmap(transformed_path, dtype=np.float32, mode='r',
                     shape=(n_layers, n_agents, n_frames, d_model))


# ---------------------------------------------------------------------------
# CLI entry point used inside each LSF array task
# ---------------------------------------------------------------------------

def _worker_main(argv):
    import torch
    from apf.io import load_config_from_model_file, load_model
    from apf.models import initialize_model
    from flyllm.prepare import init_datasets
    from flyllm.config import read_config

    p = argparse.ArgumentParser()
    p.add_argument("--configfile", required=True)
    p.add_argument("--modelfile", required=True)
    p.add_argument("--transformed_path", required=True)
    p.add_argument("--worker_id", type=int, required=True)
    p.add_argument("--n_workers", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--frames_per_chunk", type=int, default=50000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_layers", default=None,
                   help="None, 'some', or a comma-separated list of layer indices")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    print(f"[worker] argv={argv}", flush=True)

    # Parse save_layers.
    save_layers = None
    if args.save_layers is not None:
        if args.save_layers == 'some':
            save_layers = 'some'
        else:
            save_layers = [int(x) for x in args.save_layers.split(",")]

    config = read_config(args.configfile)
    load_config_from_model_file(loadmodelfile=args.modelfile, config=config,
                                weights_only=False)
    if args.device is not None:
        config["device"] = args.device

    print("[worker] init_datasets", flush=True)

    needtraindata = args.transformed_path.split('/')[-1].startswith('train')
    needvaldata = args.transformed_path.split('/')[-1].startswith('val')

    # Modification for using test data
    config['invalfilestr'] = config['invalfilestr'].replace('testtrain', 'test2')
    config['invalfile'] = config['invalfile'].replace('testtrain', 'test2')
    needvaldata = True

    assert needtraindata or needvaldata, "Expected transformed_path to start with 'train' or 'val'"
    res = init_datasets(config, needtraindata=needtraindata, needvaldata=needvaldata, res={"config": config})
    if needtraindata:
        dataset = res["train_dataset"]
    else:
        dataset = res["val_dataset"]

    device = torch.device(config["device"])
    print(f"[worker] loading model on {device}", flush=True)
    model, _ = initialize_model(config, dataset, device)
    load_model(args.modelfile, model, device)
    model.eval()

    transform_dataset_worker(
        dataset, model, args.transformed_path,
        batch_size=args.batch_size,
        frames_per_chunk=args.frames_per_chunk,
        overwrite=args.overwrite,
        save_layers=save_layers,
        worker_id=args.worker_id,
        n_workers=args.n_workers,
    )
    print("[worker] done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        _worker_main(sys.argv[2:])
    else:
        print("Usage: python transform_dataset_cluster.py worker [args...]",
              file=sys.stderr)
        sys.exit(2)