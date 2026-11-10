"""
Submit a job to the Janelia cluster that finds optimal bin edges for a config's discretized outputs with
flyllm.batch_optimal_binning, on a CPU-only node.  Run it as python -m flyllm.launch_optimal_binning.

Submits, from the current dir, the equivalent of:
    ssh -A login2 bsub -n <slot count> -q <queue> -W <run limit> -P branson -cwd <current dir> \\
        -oo <out dir>/out.txt -eo <out dir>/err.txt \\
        <python> -m flyllm.batch_optimal_binning <config file path> --outdir <out dir> \\
            [--overrideconfig <overrides>] [--debug-uselessdata]

where <out dir> is given by the --outdir option (default:
<current dir>/runs/<config name>_batch_optimal_binning_run_NN, numbered as flyllm.launch_train numbers its
runs), and must not already exist.  <config name> is the config file name without .json.  Everything the job
writes goes in <out dir>: its logs, figures, binning info and bin edges.  <python> is the running Python, so the
job runs in the same environment.
"""

import argparse
import json
import sys
from pathlib import Path

from apf.lsf import submit_job
from flyllm.launch_train import default_out_dir_path

DEFAULT_QUEUE = 'local'
# 15GB per slot in the CPU queues, and building the female fly datasets peaked at 157GB
DEFAULT_SLOT_COUNT = 16
# building the female fly datasets has taken from 47 min to over 2 h, depending on the node, and then the bins
# are optimized for each feature
DEFAULT_RUN_LIMIT = '8:00'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='config file name')
    parser.add_argument('--outdir', default=None,
                        help='directory for the job logs and all output, which must not already exist '
                             '(default: runs/<config name>_batch_optimal_binning_run_NN)')
    parser.add_argument('-q', '--queue', default=DEFAULT_QUEUE, help=f'LSF queue to submit to (default: {DEFAULT_QUEUE})')
    parser.add_argument('-n', '--slots', type=int, default=DEFAULT_SLOT_COUNT,
                        help=f'number of slots, each one core and 15GB in the CPU queues (default: {DEFAULT_SLOT_COUNT})')
    parser.add_argument('-W', '--runlimit', default=DEFAULT_RUN_LIMIT,
                        help=f'hard run time limit, as [hours:]minutes (default: {DEFAULT_RUN_LIMIT})')
    parser.add_argument('--overrideconfig', type=json.loads, default=None,
                        help='JSON dict of config entries to override, e.g. \'{"categories": ["courtship"]}\'')
    parser.add_argument('--debug-uselessdata', action='store_true',
                        help='use a small subset of the data, for quick debugging runs')
    args = parser.parse_args()
    if args.overrideconfig is not None and not isinstance(args.overrideconfig, dict):
        parser.error('--overrideconfig must be a JSON dict')
    config_file_path = Path(args.configfile).resolve()
    if not config_file_path.exists():
        parser.error(f'config file {config_file_path} does not exist')
    if args.outdir is not None:
        out_dir_path = Path(args.outdir).resolve()
    else:
        out_dir_path = default_out_dir_path(f'{config_file_path.stem}_batch_optimal_binning')
    if out_dir_path.exists():
        parser.error(f'output dir {out_dir_path} already exists')

    # LSF writes the logs into the output dir, so it must exist before the job starts
    out_dir_path.mkdir(parents=True)
    current_dir_path = Path.cwd()
    bsub_args = [
        'bsub',
        '-n', str(args.slots),
        '-q', args.queue,
        '-W', args.runlimit,
        '-P', 'branson',
        '-cwd', str(current_dir_path),
        '-oo', str(out_dir_path / 'out.txt'),
        '-eo', str(out_dir_path / 'err.txt'),
        sys.executable, '-m', 'flyllm.batch_optimal_binning', str(config_file_path),
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
