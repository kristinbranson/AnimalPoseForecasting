"""Submit ``experiments.full_evaluation.evaluate_model`` jobs to the Janelia
cluster, one A100 job per experiment.

Usage from a login node (or anywhere with ssh access to login1):

    python evaluate_model_cluster.py submit            # submits all experiments
    python evaluate_model_cluster.py submit --only ref short
    python evaluate_model_cluster.py status            # bjobs + log tails
    python evaluate_model_cluster.py tail ref          # live-tail one log

Each job runs:

    python evaluate_model_cluster.py worker --nickname <nick>

inside the cluster_scripts/ working directory (required: evaluate_model
asserts a relative ``hidden_cache`` folder).

Logs land at cluster_scripts/out/eval_<nickname>_<jobid>.log so you can
tail them while jobs are running.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

EXPERIMENTS = {
    'alldata': (
        "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_optimalbinning_20251113_alldata.json",
        "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/flyllm_models/flypredvel_optbin_alldata_20260907T081727_epoch155.pth"
    ),
    # 'ref': (
    #     "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_optimalbinning_20251113.json",
    #     "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_20251114T194024_bestepoch200.pth",
    # ),
    'short': (
        "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_optimalbinning_short_20260430.json",
        # "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/flyllm_models/flypredvel_optimal_binning_64_20260502T050431_epoch235.pth",
        "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/flyllm_models/flypredvel_optimal_binning_64_20260502T050431_bestepoch500.pth",
    ),
    'nobin': (
        "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_nobinning_20260430.json",
        # "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/flyllm_models/flypredvel_nobinning_20260501T050753_epoch300.pth",
        "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/flyllm_models/flypredvel_nobinning_20260501T050753_bestepoch500.pth",
    ),
    'bodycentric': (
        "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_optimalbinning_bodycentrickp_20260430.json",
        # "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/flyllm_models/flypredvel_body_centric_kp_20260501T060729_epoch295.pth",
        "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/flyllm_models/flypredvel_body_centric_kp_20260501T060729_bestepoch500.pth",
    ),
    'predpose': (
        "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_optimalbinning_globalvel_20260430.json",
        # "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/flyllm_models/flypredvelandpose_20260501T070152_epoch295.pth",
        "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/flyllm_models/flypredvelandpose_20260605T110442_epoch320.pth",
    ),
    'binall': (
        "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_optimalbinning_20251113_binall.json",
        "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/flyllm_models/flypredvel_optimal_binning_binall_20260605T113925_epoch205.pth",
    ),
    'rawkp': (
        "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_raw_kp.json",
        # "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/flyllm_models/flypredvel_nobinning_20260615T084955_epoch90.pth",
        "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/flyllm_models/flypredvel_nobinning_20260619T070821_bestepoch500.pth",
    ),
}


# ---------------------------------------------------------------------------
# Worker (runs inside the LSF job on a GPU node)
# ---------------------------------------------------------------------------

def _worker_main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--nickname", required=True, choices=sorted(EXPERIMENTS))
    p.add_argument("--frames_chunk_for_eval", type=int, default=64)
    p.add_argument("--overwrite_classifiers", action="store_true")
    p.add_argument("--group", default=None)
    args = p.parse_args(argv)

    # evaluate_model uses a relative 'hidden_cache' path.
    os.chdir(_HERE)

    # Local editable copy of experiments/full_evaluation.evaluate_model.
    sys.path.insert(0, _HERE)
    # from full_evaluation_kb import evaluate_model
    # from experiments.full_evaluation import evaluate_model
    # from experiments.test_evaluation import evaluate_model
    from experiments.alldata_evaluation import evaluate_model

    configfile, modelfile = EXPERIMENTS[args.nickname]
    print(f"[worker] nickname={args.nickname}", flush=True)
    print(f"[worker] configfile={configfile}", flush=True)
    print(f"[worker] modelfile={modelfile}", flush=True)
    print(f"[worker] cwd={os.getcwd()}", flush=True)

    evaluate_model(
        configfile, modelfile,
        frames_chunk_for_eval=args.frames_chunk_for_eval,
        show_figures=False,
        overwrite_classifiers=args.overwrite_classifiers,
        group=args.group,
    )
    print("[worker] done", flush=True)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def _ssh_login1(cmd: str) -> str:
    res = subprocess.run(
        ["ssh", "login1", "bash -lc " + shlex.quote(cmd)],
        capture_output=True, text=True, check=True,
    )
    return res.stdout


def _submit_one(nickname: str, *, queue: str, slots: int, walltime: str,
                extra_args: list[str], dryrun: bool = False) -> str:
    log_dir = os.path.join(_HERE, "out")
    os.makedirs(log_dir, exist_ok=True)
    log_pattern = os.path.join(log_dir, f"eval_{nickname}_%J.log")

    py_script = os.path.join(_HERE, "evaluate_model_cluster.py")
    py_args = ["worker", "--nickname", nickname] + list(extra_args)
    py_cmd = " ".join(["python", "-u", shlex.quote(py_script)] +
                      [shlex.quote(a) for a in py_args])

    inner_cmd = (
        "unset XDG_RUNTIME_DIR && "
        f"cd {shlex.quote(_HERE)} && "
        f"{py_cmd}"
    )

    bsub_cmd = (
        f"bsub -n {slots} -gpu 'num=1:aff=yes' -q {shlex.quote(queue)} "
        f"-W {shlex.quote(walltime)} "
        f"-J {shlex.quote(f'eval_{nickname}')} "
        f"-o {shlex.quote(log_pattern)} "
        + shlex.quote(inner_cmd)
    )
    print(f"[submit] {nickname}: {bsub_cmd}")
    if dryrun:
        print(f"[submit] {nickname}: DRYRUN — not submitted")
        return ""
    out = _ssh_login1(bsub_cmd)
    print(out.strip())

    jobid = None
    for tok in out.split():
        if tok.startswith("<") and tok.endswith(">") and tok[1:-1].isdigit():
            jobid = tok[1:-1]
            break
    if jobid:
        # Resolve the actual log path so the user can tail it.
        log_path = log_pattern.replace("%J", jobid)
        print(f"[submit] {nickname} -> jobid={jobid}  log={log_path}")
    return jobid or ""


def _submit_main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=None,
                   help="Only submit these nicknames (default: all).")
    p.add_argument("--queue", default="gpu_a100")
    p.add_argument("--slots", type=int, default=12)
    p.add_argument("--walltime", default="24:00")
    p.add_argument("--frames_chunk_for_eval", type=int, default=None)
    p.add_argument("--overwrite_classifiers", action="store_true")
    p.add_argument("--group", default=None)
    p.add_argument("--dryrun", action="store_true",
                   help="Print bsub commands without submitting.")
    args = p.parse_args(argv)

    nicks = args.only or list(EXPERIMENTS)
    bad = [n for n in nicks if n not in EXPERIMENTS]
    if bad:
        raise SystemExit(f"unknown nicknames: {bad}. valid: {sorted(EXPERIMENTS)}")

    extra = []
    if args.frames_chunk_for_eval is not None:
        extra += ["--frames_chunk_for_eval", str(args.frames_chunk_for_eval)]
    if args.overwrite_classifiers:
        extra += ["--overwrite_classifiers"]
    if args.group is not None:
        extra += ["--group", str(args.group)]

    submitted = {}
    for n in nicks:
        submitted[n] = _submit_one(
            n, queue=args.queue, slots=args.slots, walltime=args.walltime,
            extra_args=extra, dryrun=args.dryrun,
        )

    if args.dryrun:
        print("\n[dryrun] no jobs submitted.")
        return

    print("\n=== submitted ===")
    for n, jid in submitted.items():
        print(f"  {n:14s}  jobid={jid}  "
              f"log=out/eval_{n}_{jid}.log")
    print("\nMonitor with:")
    print("  python evaluate_model_cluster.py status")
    print("  python evaluate_model_cluster.py tail <nickname>")


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def _status_main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--tail_lines", type=int, default=15)
    p.add_argument("--user", default="eyjolfsdottire",
                   help="bjobs -u USER (default: eyjolfsdottire). Pass 'all' for everyone.")
    args = p.parse_args(argv)

    user_flag = f"-u {shlex.quote(args.user)}"
    out = _ssh_login1(
        f"bjobs -a {user_flag} -o 'jobid stat queue exec_host job_name:30 submit_time run_time' "
        f"-noheader -J 'eval_*' 2>&1 || true"
    )
    print("=== bjobs (eval_*) ===")
    print(out.strip() or "(no matching jobs)")

    log_dir = os.path.join(_HERE, "out")
    if not os.path.isdir(log_dir):
        return
    logs = [os.path.join(log_dir, f) for f in os.listdir(log_dir)
            if f.startswith("eval_") and f.endswith(".log")]
    logs.sort(key=lambda p: os.path.getmtime(p))
    by_nick: dict[str, str] = {}
    for lp in logs:
        # eval_<nick>_<jobid>.log -- nickname can contain underscores? ours don't.
        base = os.path.basename(lp)[len("eval_"):-len(".log")]
        nick = base.rsplit("_", 1)[0]
        by_nick[nick] = lp  # keep latest per nickname

    print(f"\n=== latest logs ({len(by_nick)}) ===")
    for nick, lp in by_nick.items():
        print(f"\n--- {nick}: {lp} ---")
        try:
            with open(lp, "r", errors="replace") as fh:
                lines = fh.readlines()
            print("".join(lines[-args.tail_lines:]).rstrip())
        except OSError as e:
            print(f"(cannot read: {e})")


def _tail_main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("nickname")
    p.add_argument("-n", "--lines", type=int, default=200)
    p.add_argument("-f", "--follow", action="store_true")
    args = p.parse_args(argv)

    log_dir = os.path.join(_HERE, "out")
    candidates = sorted(
        (os.path.join(log_dir, f) for f in os.listdir(log_dir)
         if f.startswith(f"eval_{args.nickname}_") and f.endswith(".log")),
        key=lambda p: os.path.getmtime(p),
    )
    if not candidates:
        raise SystemExit(f"no log found for nickname={args.nickname} in {log_dir}")
    lp = candidates[-1]
    print(f"# {lp}")
    cmd = ["tail", f"-n{args.lines}"]
    if args.follow:
        cmd.append("-f")
    cmd.append(lp)
    subprocess.call(cmd)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sub = sys.argv[1]
    if sub == "worker":
        _worker_main(sys.argv[2:])
    elif sub == "submit":
        _submit_main(sys.argv[2:])
    elif sub == "status":
        _status_main(sys.argv[2:])
    elif sub == "tail":
        _tail_main(sys.argv[2:])
    else:
        print(f"unknown subcommand: {sub}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(2)
