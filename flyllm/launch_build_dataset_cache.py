"""
Submit a job to the Janelia cluster that builds the dataset cache file for a config with
flyllm.batch_build_dataset_cache, on a CPU-only node.  Run it as python -m flyllm.launch_build_dataset_cache.

Submits, from the current dir, the equivalent of:
    ssh -A login2 bsub -n <slot count> -q <queue> -W <run limit> -P branson -cwd <current dir> \\
        -oo <out dir>/out.txt -eo <out dir>/err.txt \\
        <python> -m flyllm.batch_build_dataset_cache <config file path>

where <out dir> is given by the --outdir option (default: <current dir>/runs/<config name>_build_dataset_cache_run_NN,
numbered as flyllm.launch_train numbers its runs), and must not already exist.  <config name> is the config file
name without .json.  The job's logs go in <out dir>, and the cache file goes in the config's dataset_cache_dir.  The
job does nothing if the cache file already exists.  <python> is the running Python, so the job runs in the same
environment.
"""

import argparse
import sys
from pathlib import Path

from apf.lsf import submit_job
from flyllm.launch_train import default_out_dir_path

DEFAULT_QUEUE = 'local'
# 15GB per slot in the CPU queues, and building the female fly datasets peaked at 157GB
DEFAULT_SLOT_COUNT = 16
# building the female fly datasets has taken from 47 min to over 2 h, and their setup in training jobs up to
# 4.7 h, depending on the node
DEFAULT_RUN_LIMIT = '6:00'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='config file name')
    parser.add_argument('--outdir', default=None,
                        help='directory for the job logs, which must not already exist '
                             '(default: runs/<config name>_build_dataset_cache_run_NN)')
    parser.add_argument('-q', '--queue', default=DEFAULT_QUEUE, help=f'LSF queue to submit to (default: {DEFAULT_QUEUE})')
    parser.add_argument('-n', '--slots', type=int, default=DEFAULT_SLOT_COUNT,
                        help=f'number of slots, each one core and 15GB in the CPU queues (default: {DEFAULT_SLOT_COUNT})')
    parser.add_argument('-W', '--runlimit', default=DEFAULT_RUN_LIMIT,
                        help=f'hard run time limit, as [hours:]minutes (default: {DEFAULT_RUN_LIMIT})')
    args = parser.parse_args()
    config_file_path = Path(args.configfile).resolve()
    if not config_file_path.exists():
        parser.error(f'config file {config_file_path} does not exist')
    if args.outdir is not None:
        out_dir_path = Path(args.outdir).resolve()
    else:
        out_dir_path = default_out_dir_path(f'{config_file_path.stem}_build_dataset_cache')
    if out_dir_path.exists():
        parser.error(f'output dir {out_dir_path} already exists')

    # LSF writes the logs into the output dir, so it must exist before the job starts
    out_dir_path.mkdir(parents=True)
    current_dir_path = Path.cwd()
    print(f'Output dir: {out_dir_path}')
    submit_job([
        'bsub',
        '-n', str(args.slots),
        '-q', args.queue,
        '-W', args.runlimit,
        '-P', 'branson',
        '-cwd', str(current_dir_path),
        '-oo', str(out_dir_path / 'out.txt'),
        '-eo', str(out_dir_path / 'err.txt'),
        sys.executable, '-m', 'flyllm.batch_build_dataset_cache', str(config_file_path),
    ])


if __name__ == '__main__':
    main()
