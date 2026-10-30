# FlyLLM

Reorganization of flyllm.py and a subset of MABeFlyUtils.py, copied from https://github.com/kristinbranson/MABe2022

### Modules

**run_flyllm.py**: Main entry point. Loads data, trains a model, visualizes process.

**train.py** (`train-flyllm`): Trains a model from a config file, like notebooks/train_fly_llm_female_for_reals.py but as a script, saving the debug and loss figures as PNG files.

**launch_train.py** (`launch-train-flyllm`): Submits a train-flyllm job to the Janelia cluster.

**build_dataset_cache.py** (`build-dataset-cache`): Builds the dataset cache file for a config, so a cluster job can build it on a CPU-only node before training jobs use it.

**launch_build_dataset_cache.py** (`launch-build-dataset-cache`): Submits a build-dataset-cache job to a CPU-only node of the Janelia cluster.

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