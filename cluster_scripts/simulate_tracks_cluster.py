"""Cluster-parallel version of apf.evaluation.simulate_tracks.

The Janelia LSF array-job runs N parallel workers; each worker processes
chunk indices where ``chunk_idx % n_workers == worker_id - 1``. Each chunk
already saves its result as an independent ``.npy`` file in ``savedir``,
which is the natural unit of parallelism. After all workers finish, the
notebook-side function reads the cached chunks and stitches them into the
final ``sim_track`` / ``sim_frame``.

Notebook usage:

    from simulate_tracks_cluster import simulate_tracks_cluster
    sim_track, sim_frame, sim_transformed = simulate_tracks_cluster(
        model, train_dataset, train_data, savedir, sim_len,
        configfile=configfile, modelfile=modelfile,
        n_jobs=16, queue='gpu_a100', walltime='12:00',
        first_sim_frame=512, stride=None,
        sim_transformed_path=None,  # optional memmap path
        wait=True,                  # block until LSF jobs finish
    )

CLI worker usage (invoked by bsub on each array task):
    python simulate_tracks_cluster.py worker --configfile ... --modelfile ... \\
        --savedir ... --sim_len 2048 --first_sim_frame 512 \\
        --worker_id $LSB_JOBINDEX --n_workers 16
"""
from __future__ import annotations

import argparse
import copy
import os
import shlex
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

# Make the repo importable when this file is run as a CLI on the cluster.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _enumerate_sessions(train_dataset, sim_len, first_sim_frame, simulate_single):
    """Reproduce the session enumeration from apf.evaluation.simulate_tracks."""
    sessions = []
    if simulate_single:
        for session in train_dataset.sessions:
            if session.duration < sim_len + first_sim_frame + 1:
                continue
            sessions.append((session.start_frame, session.duration, [session.agent_id]))
    else:
        key = list(train_dataset.inputs.keys())[0]
        n_flies, n_frames, _ = train_dataset.inputs[key].array.shape
        session_frames = np.zeros((n_flies, n_frames), int)
        for i, session in enumerate(train_dataset.sessions):
            session_frames[session.agent_id,
                           session.start_frame:session.start_frame + session.duration] = i
        diffs = np.zeros_like(session_frames)
        diffs[:, 1:] = np.abs(np.diff(session_frames, axis=1))
        start_frames = np.where(diffs.max(0) > 0)[0]
        for i in range(len(start_frames) - 1):
            start_frame = start_frames[i]
            end_frame = start_frames[i + 1]
            duration = end_frame - start_frame
            if duration < sim_len + first_sim_frame + 1:
                continue
            agent_ids = np.where(session_frames[:, start_frame] > 0)[0]
            if len(agent_ids) == 0:
                continue
            sessions.append((start_frame, duration, agent_ids))
    return sessions


def _agent_str(agent_ids):
    s = str(agent_ids[0])
    for a in range(1, len(agent_ids)):
        s = s + f"_{agent_ids[a]}"
    return s


def _enumerate_chunks(train_dataset, sim_len, first_sim_frame, stride, simulate_single,
                      max_chunks=None):
    """Return list of (session_i, start_sim_frame, agent_ids, save_str)."""
    if stride is None:
        stride = sim_len
    contextl = train_dataset.context_length
    if first_sim_frame is None:
        first_sim_frame = contextl

    sessions = _enumerate_sessions(train_dataset, sim_len, first_sim_frame, simulate_single)
    chunks = []
    for i, (session_start, session_duration, agent_ids) in enumerate(sessions):
        start_sim_frames = np.arange(session_start + first_sim_frame,
                                     session_start + session_duration - sim_len, stride)
        astr = _agent_str(agent_ids)
        for start_sim_frame in start_sim_frames:
            save_str = (f"session_{i}_startsimframe_{int(start_sim_frame)}_"
                        f"agentid_{astr}_simlen_{sim_len}.npy")
            chunks.append((i, int(start_sim_frame), agent_ids, save_str))
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    return chunks, contextl, sessions


# ---------------------------------------------------------------------------
# Worker (runs on a cluster GPU node)
# ---------------------------------------------------------------------------

def simulate_tracks_worker(
    model, train_dataset, train_data, savedir, sim_len,
    *, stride=None, simulate_single=False, first_sim_frame=None,
    sim_transformed_path=None, worker_id=1, n_workers=1, max_chunks=None,
):
    """Compute and cache the chunks assigned to (worker_id, n_workers).

    worker_id is 1-indexed to match LSB_JOBINDEX. Stitching is left to the
    notebook-side function once all workers complete.
    """
    from apf.simulation import simulate

    chunks, contextl, _ = _enumerate_chunks(
        train_dataset, sim_len, first_sim_frame, stride, simulate_single,
        max_chunks=max_chunks)

    track = train_data['track']
    pose = train_data['pose']
    flyids = train_data['flyids']

    transformed = None
    if sim_transformed_path is not None:
        d_model = model.d_model
        n_layers = len(model.transformer_encoder.layers)
        n_agents, n_frames = train_dataset.inputs['velocity'].array.shape[:2]
        # Workers share a pre-allocated memmap; open r+ so we don't truncate it.
        if not os.path.exists(sim_transformed_path):
            raise FileNotFoundError(
                f"sim_transformed_path {sim_transformed_path!r} must be preallocated by "
                f"simulate_tracks_cluster before workers start.")
        transformed = np.memmap(sim_transformed_path, dtype=np.float32, mode="r+",
                                shape=(n_layers, n_agents, n_frames, d_model))

    os.makedirs(savedir, exist_ok=True)
    my_chunks = [c for idx, c in enumerate(chunks) if idx % n_workers == (worker_id - 1)]
    print(f"[worker {worker_id}/{n_workers}] processing {len(my_chunks)}/{len(chunks)} chunks",
          flush=True)

    for k, (i, start_sim_frame, agent_ids, save_str) in enumerate(my_chunks):
        save_path = os.path.join(savedir, save_str)
        if os.path.exists(save_path):
            print(f"  [{k+1}/{len(my_chunks)}] cached {save_str}", flush=True)
            continue
        print(f"  [{k+1}/{len(my_chunks)}] computing {save_str}", flush=True)
        _, pred_track = simulate(
            dataset=train_dataset, model=model,
            track=track, pose=pose, identities=flyids,
            track_len=sim_len + contextl, burn_in=contextl, max_contextl=contextl,
            agent_ids=agent_ids, start_frame=start_sim_frame - contextl,
            transformed_cache=transformed,
        )
        pred_track = pred_track[agent_ids]
        np.save(save_path, pred_track)


# ---------------------------------------------------------------------------
# Stitching (runs in the notebook after workers complete)
# ---------------------------------------------------------------------------

def _stitch_from_cache(train_dataset, train_data, savedir, sim_len, *,
                       stride=None, simulate_single=False, first_sim_frame=None,
                       sim_transformed_path=None, allow_missing=False,
                       max_chunks=None):
    chunks, contextl, _ = _enumerate_chunks(
        train_dataset, sim_len, first_sim_frame, stride, simulate_single,
        max_chunks=max_chunks)

    sim_track = copy.deepcopy(train_data['track'])
    sim_frame = np.zeros(sim_track.array.shape[:2])

    transformed = None
    if sim_transformed_path is not None and os.path.exists(sim_transformed_path):
        n_agents, n_frames = train_dataset.inputs['velocity'].array.shape[:2]
        # Caller knows shape; we only return a handle on a best-effort basis.
        transformed = np.memmap(sim_transformed_path, dtype=np.float32, mode="r")

    missing = []
    for i, start_sim_frame, agent_ids, save_str in chunks:
        save_path = os.path.join(savedir, save_str)
        if not os.path.exists(save_path):
            missing.append(save_str)
            continue
        pred_track = np.load(save_path)
        sim_track.array[agent_ids, start_sim_frame:start_sim_frame + sim_len] = pred_track[:, contextl:]
        sim_frame[agent_ids, start_sim_frame:start_sim_frame + sim_len] = (np.arange(sim_len) + 1)[None, :]

    if missing and not allow_missing:
        raise RuntimeError(
            f"{len(missing)}/{len(chunks)} chunks missing in {savedir}. "
            f"First missing: {missing[0]}")
    return sim_track, sim_frame, transformed, len(missing), len(chunks)


# ---------------------------------------------------------------------------
# Cluster submission (called from the notebook)
# ---------------------------------------------------------------------------

def _ssh_login1(cmd: str) -> str:
    """Run a command on login1 and return stdout."""
    full = ["ssh", "login1", "bash -lc " + shlex.quote(cmd)]
    res = subprocess.run(full, capture_output=True, text=True, check=True)
    return res.stdout


def check_job_status(jobid=None, jobname=None, savedir=None, tail_lines=20, verbose=True):
    """Check status of a cluster simulate_tracks_cluster job.

    Pass any combination of:
      - jobid: LSF job id (e.g. info['jobid'])
      - jobname: job name pattern (e.g. 'simtracks_smoke')
      - savedir: savedir used in the original call; tails the latest worker log

    Returns a dict with keys: states (per-task state counts), tasks (raw bjobs
    rows), n_chunks_done (.npy count in savedir, if given), log_tail (str).
    """
    info = {"states": {}, "tasks": [], "n_chunks_done": None, "log_tail": None}

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

    if savedir and os.path.isdir(savedir):
        info["n_chunks_done"] = sum(
            1 for f in os.listdir(savedir)
            if f.startswith("session_") and f.endswith(".npy"))
        log_dir = os.path.join(savedir, "cluster_logs")
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
        if info["n_chunks_done"] is not None:
            print(f"  chunks completed in savedir: {info['n_chunks_done']}")
        if info.get("log_path"):
            print(f"  log: {info['log_path']}")
            print("  --- last lines ---")
            print(info["log_tail"])
    return info


_TERMINAL_STATES = {"DONE", "EXIT", "UNKWN", "ZOMBI"}
_ACTIVE_STATES = {"PEND", "RUN", "PROV", "WAIT", "USUSP", "SSUSP", "PSUSP"}


def _parse_bjobs_rows(out: str):
    """Filter shell-init noise out of bjobs output and return (jobid, stat) rows."""
    rows = []
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        # Real job rows start with a numeric jobid
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
            # bjobs forgets very old jobs; treat as terminal
            return out
        states = {stat for _, stat in rows}
        if not (states & _ACTIVE_STATES):
            return out
        time.sleep(poll_sec)


def simulate_tracks_cluster(
    model, train_dataset, train_data, savedir, sim_len,
    *,
    configfile: str,
    modelfile: str,
    n_jobs: int = 8,
    stride: Optional[int] = None,
    simulate_single: bool = False,
    first_sim_frame: Optional[int] = None,
    sim_transformed_path: Optional[str] = None,
    queue: str = "gpu_a100",
    slots: int = 12,
    walltime: str = "24:00",
    jobname: str = "simtracks",
    conda_env: str = "transformer312",
    conda_path: str = "$HOME/miniforge3/etc/profile.d/conda.sh",
    repo_root: Optional[str] = None,
    extra_python_args: Optional[List[str]] = None,
    wait: bool = True,
    poll_sec: int = 30,
    submit_only: bool = False,
    max_chunks: Optional[int] = None,
) -> Tuple[object, np.ndarray, Optional[np.memmap], dict]:
    """Submit an LSF array job to compute simulate_tracks chunks in parallel.

    Returns (sim_track, sim_frame, sim_transformed, info). If ``wait=False`` or
    ``submit_only=True``, sim_track/sim_frame may be None and info contains the
    LSF job id so the caller can poll later.

    Paths in ``savedir``, ``configfile``, ``modelfile``, ``sim_transformed_path``
    must all be cluster-visible (i.e. under /groups, /nrs, or /misc).
    """
    repo_root = repo_root or _REPO_ROOT
    os.makedirs(savedir, exist_ok=True)
    log_dir = os.path.join(savedir, "cluster_logs")
    os.makedirs(log_dir, exist_ok=True)

    chunks, contextl, sessions = _enumerate_chunks(
        train_dataset, sim_len, first_sim_frame, stride, simulate_single,
        max_chunks=max_chunks)
    if len(chunks) == 0:
        raise RuntimeError("No chunks to simulate (check sim_len / first_sim_frame).")
    n_jobs = max(1, min(n_jobs, len(chunks)))
    print(f"[simulate_tracks_cluster] {len(chunks)} chunks across {len(sessions)} sessions"
          f" -> {n_jobs} array tasks")

    # Preallocate (or validate) the shared transformed memmap so workers can
    # open it r+. If the file already exists, we leave its contents intact and
    # workers will write only into their assigned (agent, frame) slices.
    if sim_transformed_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(sim_transformed_path)) or ".",
                    exist_ok=True)
        d_model = model.d_model
        n_layers = len(model.transformer_encoder.layers)
        n_agents, n_frames = train_dataset.inputs['velocity'].array.shape[:2]
        expected_bytes = n_agents * n_frames * d_model * n_layers * np.dtype(
            np.float32).itemsize
        if not os.path.exists(sim_transformed_path):
            print(f"[simulate_tracks_cluster] preallocating transformed memmap shape="
                  f"({n_agents},{n_frames},{d_model},{n_layers}) "
                  f"({expected_bytes/1e9:.2f} GB) at {sim_transformed_path}")
            mm = np.memmap(sim_transformed_path, dtype=np.float32, mode="w+",
                           shape=(n_layers, n_agents, n_frames, d_model))
            mm.flush()
            del mm
        else:
            actual_bytes = os.path.getsize(sim_transformed_path)
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"Existing sim_transformed_path {sim_transformed_path!r} "
                    f"has size {actual_bytes} bytes, but current "
                    f"(n_agents={n_agents}, n_frames={n_frames}, d_model={d_model}, "
                    f"n_layers={n_layers}, float32) expects {expected_bytes} bytes. "
                    f"Refusing to reuse — delete the file or pick a different path."
                )
            print(f"[simulate_tracks_cluster] reusing existing transformed memmap "
                  f"at {sim_transformed_path} ({actual_bytes/1e9:.2f} GB) — "
                  f"new chunks will write into previously-zero slices.")

    # Build the per-task python command. Use $LSB_JOBINDEX as worker_id.
    py_script = os.path.join(repo_root, "cluster_scripts", "simulate_tracks_cluster.py")
    py_args = [
        "worker",
        "--configfile", configfile,
        "--modelfile", modelfile,
        "--savedir", savedir,
        "--sim_len", str(sim_len),
        "--n_workers", str(n_jobs),
        "--worker_id", "$LSB_JOBINDEX",
    ]
    if stride is not None:
        py_args += ["--stride", str(stride)]
    if first_sim_frame is not None:
        py_args += ["--first_sim_frame", str(first_sim_frame)]
    if simulate_single:
        py_args += ["--simulate_single"]
    if sim_transformed_path is not None:
        py_args += ["--sim_transformed_path", sim_transformed_path]
    if max_chunks is not None:
        py_args += ["--max_chunks", str(max_chunks)]
    if extra_python_args:
        py_args += list(extra_python_args)

    py_cmd_parts = ["python", py_script] + [
        a if a == "$LSB_JOBINDEX" else shlex.quote(a) for a in py_args
    ]
    py_cmd = " ".join(py_cmd_parts)

    inner_cmd = (
        # f"source {shlex.quote(conda_path)} && "
        # f"conda activate {shlex.quote(conda_env)} && "  # had to remove this since this is already done in bashrc
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
    print(f"[simulate_tracks_cluster] submitting: {bsub_cmd}")
    out = _ssh_login1(bsub_cmd)
    print(out.strip())

    # Parse "Job <12345> is submitted to queue ..."
    jobid = None
    for tok in out.split():
        if tok.startswith("<") and tok.endswith(">") and tok[1:-1].isdigit():
            jobid = tok[1:-1]
            break
    info = {"jobid": jobid, "n_chunks": len(chunks), "n_jobs": n_jobs,
            "log_pattern": log_pattern, "savedir": savedir}

    if submit_only or not wait:
        return None, None, None, info

    if jobid is None:
        raise RuntimeError(f"Could not parse LSF job id from bsub output:\n{out}")
    print(f"[simulate_tracks_cluster] waiting for job {jobid} (poll every {poll_sec}s)")
    final = _wait_for_jobid(jobid, poll_sec=poll_sec)
    print(f"[simulate_tracks_cluster] job {jobid} finished:\n{final}")

    final_rows = _parse_bjobs_rows(final)
    final_states = [stat for _, stat in final_rows]
    failed_tasks = [jid for jid, stat in final_rows if stat == "EXIT"]
    info["task_states"] = final_states
    info["failed_tasks"] = failed_tasks

    if failed_tasks:
        log_dir = os.path.dirname(log_pattern)
        log_hint = ""
        try:
            logs = sorted(
                (os.path.join(log_dir, f) for f in os.listdir(log_dir)
                 if f.endswith(".log") and jobid in f),
                key=lambda p: os.path.getmtime(p))
            if logs:
                log_hint = f" Latest log: {logs[-1]}"
        except OSError:
            pass
        raise RuntimeError(
            f"LSF job {jobid}: {len(failed_tasks)}/{len(final_rows)} tasks "
            f"ended in EXIT (failed: {failed_tasks}). Check the worker logs in "
            f"{log_dir}.{log_hint}"
        )

    print("[simulate_tracks_cluster] stitching cached chunks")
    sim_track, sim_frame, transformed, n_missing, n_total = _stitch_from_cache(
        train_dataset, train_data, savedir, sim_len,
        stride=stride, simulate_single=simulate_single,
        first_sim_frame=first_sim_frame,
        sim_transformed_path=sim_transformed_path,
        allow_missing=False, max_chunks=max_chunks,
    )
    info["n_missing"] = n_missing
    info["n_total"] = n_total
    return sim_track, sim_frame, transformed, info


def stitch_cluster_results(
    train_dataset, train_data, savedir, sim_len,
    *, stride=None, simulate_single=False, first_sim_frame=None,
    sim_transformed_path=None, allow_missing=False, max_chunks=None,
):
    """Read cached chunks from savedir and assemble (sim_track, sim_frame).

    Use this from the notebook after a fire-and-forget submission with
    ``wait=False``. Pass the same ``sim_len`` / ``stride`` / ``simulate_single``
    / ``first_sim_frame`` / ``max_chunks`` you passed to
    ``simulate_tracks_cluster``.
    """
    return _stitch_from_cache(
        train_dataset, train_data, savedir, sim_len,
        stride=stride, simulate_single=simulate_single,
        first_sim_frame=first_sim_frame,
        sim_transformed_path=sim_transformed_path,
        allow_missing=allow_missing, max_chunks=max_chunks,
    )

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
    p.add_argument("--savedir", required=True)
    p.add_argument("--sim_len", type=int, required=True)
    p.add_argument("--worker_id", type=int, required=True)
    p.add_argument("--n_workers", type=int, required=True)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--first_sim_frame", type=int, default=None)
    p.add_argument("--simulate_single", action="store_true")
    p.add_argument("--sim_transformed_path", default=None)
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    print(f"[worker] argv={argv}", flush=True)

    config = read_config(args.configfile)
    load_config_from_model_file(loadmodelfile=args.modelfile, config=config,
                                weights_only=False)
    if args.device is not None:
        config["device"] = args.device

    # Modification for using test data
    config['invalfilestr'] = config['invalfilestr'].replace('testtrain', 'test2')
    config['invalfile'] = config['invalfile'].replace('testtrain', 'test2')

    print("[worker] init_datasets", flush=True)
    res = init_datasets(config, needvaldata=True, res={"config": config})
    train_dataset = res["train_dataset"]
    train_data = res["train_data"]
    test_dataset = res['val_dataset']
    test_data = res['val_data']

    device = torch.device(config["device"])
    print(f"[worker] loading model on {device}", flush=True)
    model, _ = initialize_model(config, train_dataset, device)
    load_model(args.modelfile, model, device)
    model.eval()

    simulate_tracks_worker(
        model,
        # train_dataset, train_data,
        test_dataset, test_data,
        args.savedir, args.sim_len,
        stride=args.stride, simulate_single=args.simulate_single,
        first_sim_frame=args.first_sim_frame,
        sim_transformed_path=args.sim_transformed_path,
        worker_id=args.worker_id, n_workers=args.n_workers,
        max_chunks=args.max_chunks,
    )
    print("[worker] done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        _worker_main(sys.argv[2:])
    else:
        print("Usage: python simulate_tracks_cluster.py worker [args...]",
              file=sys.stderr)
        sys.exit(2)
