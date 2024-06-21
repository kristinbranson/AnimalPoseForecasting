# FlyLLM

Reorganization of flyllm.py and a subset of MABeFlyUtils.py, copied from https://github.com/kristinbranson/MABe2022

### Modules

**run_flyllm.py**: Main entry point. Loads data, trains a model, visualizes process.

**config.py**: Parameters for feature extraction.

**features.py**: Functions for extracting features from tracking data.

**data.py**: Functions for loading data, chunking data, splitting data, flipping data, etc.

**dataset.py**: FlyMLMDataset class.

**model.py**: Neural network modules and related functions.

**io.py**: Functions for loading and saving models and config.

**plotting.py**: Functions for plotting flies, features, and other debugging information.

**simulation.py**: Functions to run model in an open loop and animate the output.

**utils.py**: Math utils, completely agnostic of data, features or models.

**pose.py**: 

**legacy.py**: Functions not currently used by rest of the code.