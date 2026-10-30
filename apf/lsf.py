"""
Helpers for running under LSF, the job scheduler on the Janelia cluster.  Imports nothing heavy, so it can be
imported before numpy and torch.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

LOGIN_HOST = 'login2'

THREAD_COUNT_VARIABLE_NAMES = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']


def limit_threads_to_lsf_slots() -> int | None:
    """
    If running in an LSF job, sets OMP_NUM_THREADS, MKL_NUM_THREADS, and OPENBLAS_NUM_THREADS to the job's
    slot count, $LSB_DJOB_NUMPROC, so numpy and torch don't start a thread for every core on the node.
    Returns the slot count, or None if not in an LSF job (and then changes nothing).

    Must be called before numpy or torch is imported, since they read these variables when they're imported.
    """
    slot_count_str_maybe: str | None = os.environ.get('LSB_DJOB_NUMPROC')
    if slot_count_str_maybe is None:
        return None
    for name in THREAD_COUNT_VARIABLE_NAMES:
        os.environ[name] = slot_count_str_maybe
    return int(slot_count_str_maybe)


def installed_script_file_path(name: str) -> Path:
    """
    Returns the absolute path of the console script with this name installed in the same environment as the
    running Python.  Jobs submitted with submit_job don't get this environment's PATH, so they need it.
    """
    return Path(sys.executable).parent / name


def submit_job(bsub_args: list[str]) -> None:
    """ Runs bsub with these arguments (bsub_args[0] is 'bsub') on the login node, via ssh. """
    # ssh joins its arguments into one string for the remote shell, so quote them here
    remote_command = shlex.join(bsub_args)
    print(f'Submitting: {remote_command}')
    subprocess.run(['ssh', '-A', LOGIN_HOST, remote_command], check=True)
