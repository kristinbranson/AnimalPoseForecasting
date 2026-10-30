"""Build the dataset cache file for a config, the one a training run with that config would load, and exit.

Does nothing if the cache file already exists.  Doesn't use the GPU, so it can run on CPU-only nodes.  Run it as
build-dataset-cache.
"""

import argparse
import logging
import os
import sys

from apf.lsf import limit_threads_to_lsf_slots

limit_threads_to_lsf_slots()

from flyllm.prepare import (dataset_cache_dir_path_maybe, dataset_cache_file_path, dataset_cache_key, init_config,
                            init_flyllm)

LOG = logging.getLogger('build_dataset_cache')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configfile', help='config file name')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    config_file_path = os.path.abspath(args.configfile)
    config = init_config(configfile=config_file_path, overrideconfig={'device': 'cpu'})['config']

    cache_dir_path_maybe: str | None = dataset_cache_dir_path_maybe(config)
    if cache_dir_path_maybe is None:
        sys.exit(f'dataset_cache_dir is not set in {config_file_path}, so there is no cache to build')
    cache_dir_path: str = cache_dir_path_maybe

    # the same key as init_datasets computes for a training run (mode='train')
    cache_file_path = dataset_cache_file_path(cache_dir_path, dataset_cache_key(config, True, True, False))
    if os.path.exists(cache_file_path):
        LOG.info(f'Dataset cache file {cache_file_path} already exists, nothing to do')
        return

    res = init_flyllm(config=config, mode='train', doinitmodel=False)
    if not res['success']:
        sys.exit('init_flyllm failed')
    if not os.path.exists(cache_file_path):
        sys.exit(f'init_flyllm succeeded, but did not write the expected dataset cache file {cache_file_path}')
    LOG.info(f'Built dataset cache file {cache_file_path}')


if __name__ == '__main__':
    main()
