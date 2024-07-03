import json
import os
import numpy as np

from apf.data import load_raw_npz_data, debug_less_data
from flyllm.io import json_load_helper


def make_test_data() -> str:
    """Creates test data for test_run_flyllm.py

    Returns:
        test_config_path: Path to config to be used for testing
    """
    # Load existing config
    file = "/groups/branson/home/eyjolfsdottire/code/MABe2022/config_fly_llm_multitimeglob_discrete_20230907.json"
    config = json_load_helper(file)

    # Create smaller train and validation data and save it to the test directory
    train_path = os.path.join(config['datadir'], config['intrainfilestr'])
    train_data = load_raw_npz_data(train_path)
    debug_less_data(train_data, n_frames_per_video=2000, max_n_videos=5)
    val_path = os.path.join(config['datadir'], config['invalfilestr'])
    val_data = load_raw_npz_data(val_path)
    debug_less_data(val_data, n_frames_per_video=2000, max_n_videos=5)

    # Save the new data
    test_dir = '/groups/branson/bransonlab/test_data_apf/'
    test_train_path = os.path.join(test_dir, 'small_intrainfile.npz')
    np.savez(test_train_path, **train_data)
    test_val_path = os.path.join(test_dir, 'small_invalfile.npz')
    np.savez(test_val_path, **val_data)

    # Update config to point to this data and do faster training/eval
    config['datadir'] = test_dir
    config['savedir'] = os.path.join(test_dir, 'llmnets')
    config['intrainfilestr'] = os.path.relpath(test_train_path, test_dir)
    config['invalfilestr'] = os.path.relpath(test_val_path, test_dir)
    config['batch_size'] = 8
    config['num_train_epochs'] = 2
    config['niterplot'] = 2
    config['n_layers'] = 3

    if not os.path.exists(config['savedir']):
        os.makedirs(config['savedir'])

    # Save the new config
    test_config_path = os.path.join(test_dir, 'test_config.json')
    with open(test_config_path, 'w') as f:
        json.dump(config, f)

    return test_config_path
