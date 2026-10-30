"""
Submit a job to the Janelia cluster that builds the dataset cache file for a config with build-dataset-cache,
on a CPU-only node.  Run it as launch-build-dataset-cache.

Submits, from the current dir, the equivalent of:
    ssh -A login2 bsub -n <slot count> -q <queue> -W <run limit> -P branson -cwd <current dir> \\
        -oo <current dir>/<config name>_build_dataset_cache.out.txt \\
        -eo <current dir>/<config name>_build_dataset_cache.err.txt \\
        build-dataset-cache <config file path>

where <config name> is the config file name without .json.  The job does nothing if the cache file already
exists.
"""

import argparse
from pathlib import Path

from apf.lsf import installed_script_file_path, submit_job

DEFAULT_QUEUE = 'local'
# 15GB per slot in the CPU queues, and building the female fly datasets peaked at 157GB
DEFAULT_SLOT_COUNT = 16
# building the female fly datasets took 47 min
DEFAULT_RUN_LIMIT = '2:00'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='config file name')
    parser.add_argument('-q', '--queue', default=DEFAULT_QUEUE, help=f'LSF queue to submit to (default: {DEFAULT_QUEUE})')
    parser.add_argument('-n', '--slots', type=int, default=DEFAULT_SLOT_COUNT,
                        help=f'number of slots, each one core and 15GB in the CPU queues (default: {DEFAULT_SLOT_COUNT})')
    parser.add_argument('-W', '--runlimit', default=DEFAULT_RUN_LIMIT,
                        help=f'hard run time limit, as [hours:]minutes (default: {DEFAULT_RUN_LIMIT})')
    args = parser.parse_args()
    config_file_path = Path(args.configfile).resolve()
    if not config_file_path.exists():
        parser.error(f'config file {config_file_path} does not exist')

    run_dir_path = Path.cwd()
    log_file_path_prefix = run_dir_path / f'{config_file_path.stem}_build_dataset_cache'
    submit_job([
        'bsub',
        '-n', str(args.slots),
        '-q', args.queue,
        '-W', args.runlimit,
        '-P', 'branson',
        '-cwd', str(run_dir_path),
        '-oo', f'{log_file_path_prefix}.out.txt',
        '-eo', f'{log_file_path_prefix}.err.txt',
        str(installed_script_file_path('build-dataset-cache')), str(config_file_path),
    ])


if __name__ == '__main__':
    main()
