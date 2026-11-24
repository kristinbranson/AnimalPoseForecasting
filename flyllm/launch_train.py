"""
Submit a job to the Janelia cluster that trains a model from a config file with flyllm.train.  Run it as
python -m flyllm.launch_train.

Submits, from the current dir, the equivalent of:
    ssh -A login2 bsub -n <slot count> -gpu "num=<GPU count>" -q <queue> -W 72:00 -P branson -cwd <current dir> \\
        -oo <out dir>/out.txt -eo <out dir>/err.txt \\
        <python> -m flyllm.train <config file path> --outdir <out dir> \\
            [--overrideconfig <overrides>] [--debug-uselessdata]

where <out dir> is given by the --outdir option (default: <current dir>/runs/<config name>_run_NN), <queue> by
the -q option (default: gpu_a100), <slot count> and <GPU count> by the -n and --gpus options (default: 12 and 1),
and <overrides> by the --overrideconfig option.  In the default <out dir>, <config name> is the config file name
without .json, and NN is the smallest run index greater than that of any existing <config name>_run_NN dir or
<config name>_run_NN.{out.txt,err.txt,ipynb} file in <current dir>/runs (created if needed).

Training uses only one GPU, but asking for all of a node's GPUs and slots (4 and 48 on gpu_a100 nodes) gets the
whole node, so that other jobs don't affect its speed.  <python> is the running Python, so the job runs in the
same environment.

Everything the run writes goes in <out dir>: the job's logs, models in <out dir>/models, and figures in
<out dir>/figs.  <out dir> must not already exist, so that runs can't overwrite each other's files.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from apf.lsf import submit_job

DEFAULT_QUEUE = 'gpu_a100'
# 12 slots per GPU is the natural ratio in gpu_a100 and gpu_h100
DEFAULT_SLOT_COUNT = 12
DEFAULT_GPU_COUNT = 1


def next_run_index(run_dir_path: Path, name: str) -> int:
    """ Returns 1 + the largest run index among existing <name>_run_NN dirs and <name>_run_NN.{out.txt,err.txt,ipynb}
    files.  The files are the logs and notebooks of runs from before the logs went in the run's dir. """
    pattern = re.compile(rf'^{re.escape(name)}_run_(\d+)(?:\.(?:out\.txt|err\.txt|ipynb))?$')
    indices = [int(match.group(1)) for path in run_dir_path.iterdir() if (match := pattern.match(path.name))]
    return max(indices, default=0) + 1


def default_out_dir_path(name: str) -> Path:
    """ Returns <current dir>/runs/<name>_run_NN, for the next free run index NN, creating the runs dir if needed. """
    run_dir_path = Path.cwd() / 'runs'
    run_dir_path.mkdir(exist_ok=True)
    return run_dir_path / f'{name}_run_{next_run_index(run_dir_path, name):02d}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='config file name')
    parser.add_argument('--outdir', default=None,
                        help='directory for the job logs and the models and figs dirs, which must not already exist '
                             '(default: runs/<config name>_run_NN)')
    parser.add_argument('-q', '--queue', default=DEFAULT_QUEUE, help=f'LSF queue to submit to (default: {DEFAULT_QUEUE})')
    parser.add_argument('-n', '--slots', type=int, default=DEFAULT_SLOT_COUNT,
                        help=f'number of slots (default: {DEFAULT_SLOT_COUNT})')
    parser.add_argument('--gpus', type=int, default=DEFAULT_GPU_COUNT,
                        help=f'number of GPUs (default: {DEFAULT_GPU_COUNT})')
    parser.add_argument('--overrideconfig', type=json.loads, default=None,
                        help='JSON dict of config entries to override, e.g. \'{"num_train_epochs": 25}\'')
    parser.add_argument('--debug-uselessdata', action='store_true',
                        help='train on a small subset of the data, for quick debugging runs')
    args = parser.parse_args()
    if args.overrideconfig is not None and not isinstance(args.overrideconfig, dict):
        parser.error('--overrideconfig must be a JSON dict')
    config_file_path = Path(args.configfile).resolve()
    if not config_file_path.exists():
        parser.error(f'config file {config_file_path} does not exist')
    if args.outdir is not None:
        out_dir_path = Path(args.outdir).resolve()
    else:
        out_dir_path = default_out_dir_path(config_file_path.stem)
    if out_dir_path.exists():
        parser.error(f'output dir {out_dir_path} already exists')

    # LSF writes the logs into the output dir, so it must exist before the job starts.  Creating it also reserves its
    # run index for later submissions.
    out_dir_path.mkdir(parents=True)
    current_dir_path = Path.cwd()
    bsub_args = [
        'bsub',
        '-n', str(args.slots),
        '-gpu', f'num={args.gpus}',
        '-q', args.queue,
        '-W', '72:00',
        '-P', 'branson',
        '-cwd', str(current_dir_path),
        '-oo', str(out_dir_path / 'out.txt'),
        '-eo', str(out_dir_path / 'err.txt'),
        sys.executable, '-m', 'flyllm.train', str(config_file_path),
        '--outdir', str(out_dir_path),
    ]
    if args.debug_uselessdata:
        bsub_args.append('--debug-uselessdata')
    if args.overrideconfig:
        bsub_args += ['--overrideconfig', json.dumps(args.overrideconfig)]
    print(f'Output dir: {out_dir_path}')
    submit_job(bsub_args)


if __name__ == '__main__':
    main()
