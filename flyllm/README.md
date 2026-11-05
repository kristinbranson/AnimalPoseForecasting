# FlyLLM

Reorganization of flyllm.py and a subset of MABeFlyUtils.py, copied from https://github.com/kristinbranson/MABe2022

### Modules

**run_flyllm.py**: Main entry point. Loads data, trains a model, visualizes process.

**train.py** (`python -m flyllm.train`): Trains a model from a config file, like notebooks/train_fly_llm.py but as a script, saving the debug and loss figures as PNG files.  Trains on all visible GPUs (or `--gpus N` of them), one process each, with DistributedDataParallel.

**build_dataset_cache.py** (`python -m flyllm.build_dataset_cache`): Builds the dataset cache file for a config, so a cluster job can build it on a CPU-only node before training jobs use it.

**launch_train.py** (`python -m flyllm.launch_train`): Submits a train.py job to the Janelia cluster, with its logs, models, and figures in one output dir (by default a new numbered dir in `runs/`).

**launch_build_dataset_cache.py** (`python -m flyllm.launch_build_dataset_cache`): Submits a build_dataset_cache.py job to a CPU queue on the Janelia cluster.

**config.py**: Parameters for feature extraction.

**features.py**: Functions for extracting features from tracking data.

**dataset.py**: FlyMLMDataset class.

**model.py**: Neural network modules and related functions.

**io.py**: Functions for loading and saving models, data and config.

**plotting.py**: Functions for plotting flies, features, and other debugging information.

**simulation.py**: Functions to run model in an open loop and animate the output.

**utils.py**: Math utils, completely agnostic of data, features or models.

**pose.py**: 

**legacy.py**: Functions not currently used by rest of the code.