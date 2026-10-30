"""
Submit a job to the Janelia cluster that trains a model from a config file with train-flyllm.  Run it as
launch-train-flyllm.

Submits, from the current dir, the equivalent of:
    ssh -A login2 bsub -n 12 -gpu "num=1" -q <queue> -W 72:00 -P branson -cwd <current dir> \\
        -oo <current dir>/<name>_run_NN.out.txt -eo <current dir>/<name>_run_NN.err.txt \\
        train-flyllm <config file path> [--overrideconfig <overrides>] [--debug-uselessdata]

where <name> is given by the --name option (default: the config file name without .json), NN is the smallest
run index greater than that of any existing <name>_run_NN.{out.txt,err.txt,ipynb} file in the current dir,
<queue> is given by the -q option (default: gpu_a100), and <overrides> is given by the --overrideconfig
option.  The job runs the train-flyllm installed alongside this one, so models go to <current dir>/models and
figures to <current dir>/figs.
"""

import argparse
import json
import re
from pathlib import Path

from apf.lsf import installed_script_file_path, submit_job

DEFAULT_QUEUE = 'gpu_a100'


def next_run_index(run_dir_path: Path, name: str) -> int:
    """ Returns 1 + the largest run index among existing <name>_run_NN.{out.txt,err.txt,ipynb} files. """
    pattern = re.compile(rf'^{re.escape(name)}_run_(\d+)\.(?:out\.txt|err\.txt|ipynb)$')
    indices = [int(match.group(1)) for path in run_dir_path.iterdir() if (match := pattern.match(path.name))]
    return max(indices, default=0) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='config file name')
    parser.add_argument('-q', '--queue', default=DEFAULT_QUEUE, help=f'LSF queue to submit to (default: {DEFAULT_QUEUE})')
    parser.add_argument('--overrideconfig', type=json.loads, default=None,
                        help='JSON dict of config entries to override, e.g. \'{"num_train_epochs": 25}\'')
    parser.add_argument('--debug-uselessdata', action='store_true',
                        help='train on a small subset of the data, for quick debugging runs')
    parser.add_argument('--name', default=None,
                        help='start of the log file names (default: the config file name without .json)')
    args = parser.parse_args()
    if args.overrideconfig is not None and not isinstance(args.overrideconfig, dict):
        parser.error('--overrideconfig must be a JSON dict')
    config_file_path = Path(args.configfile).resolve()
    if not config_file_path.exists():
        parser.error(f'config file {config_file_path} does not exist')
    name = args.name if args.name is not None else config_file_path.stem

    run_dir_path = Path.cwd()
    run_index = next_run_index(run_dir_path, name)
    out_file_path = run_dir_path / f'{name}_run_{run_index:02d}.out.txt'
    err_file_path = run_dir_path / f'{name}_run_{run_index:02d}.err.txt'
    # LSF creates the log files only when the job starts, so create one now to reserve this run index
    # for later submissions (bsub -oo overwrites it)
    out_file_path.touch(exist_ok=False)
    bsub_args = [
        'bsub',
        '-n', '12',
        '-gpu', 'num=1',
        '-q', args.queue,
        '-W', '72:00',
        '-P', 'branson',
        '-cwd', str(run_dir_path),
        '-oo', str(out_file_path),
        '-eo', str(err_file_path),
        str(installed_script_file_path('train-flyllm')), str(config_file_path),
    ]
    if args.debug_uselessdata:
        bsub_args.append('--debug-uselessdata')
    if args.overrideconfig:
        bsub_args += ['--overrideconfig', json.dumps(args.overrideconfig)]
    print(f'Run {run_index:02d}')
    submit_job(bsub_args)


if __name__ == '__main__':
    main()
