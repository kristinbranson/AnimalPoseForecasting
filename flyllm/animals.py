"""Which animal a config is for, and the code that handles that animal's data.

A config's 'animal' entry, 'fly' (the default) or 'mouse', picks the modules that read the config, build
the datasets and plot the debug figures, so that the binning, dataset-cache and training scripts in
flyllm/ work for either.
"""
import importlib
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

DEFAULT_ANIMAL_NAME = 'fly'


@dataclass(frozen=True)
class Animal:
    """ The code for one animal's data.

    Attributes:
        name: the config's 'animal' value.
        read_config: reads a config file, filling in the animal's defaults.
        make_dataset: experiments.flyllm.make_dataset or its counterpart.
        posenames: pose feature names, which the velocity features share.
        featangle: whether each pose feature is an angle.
        length_unit: the unit of positions, e.g. 'mm'.
        dataset_source_file_names: source files, relative to the repository root, that build the animal's
            datasets, beyond the ones all animals share (flyllm.prepare.DATASET_SOURCE_FILES).
        debug_plot_kwargs: keyword arguments for flyllm.plotting.initialize_debug_plots and
            update_debug_plots, for the animal's keypoints and features.
    """
    name: str
    read_config: Callable
    make_dataset: Callable
    posenames: list[str]
    featangle: np.ndarray
    length_unit: str
    dataset_source_file_names: list[str]
    debug_plot_kwargs: dict = field(default_factory=dict)


def _fly() -> Animal:
    import experiments.flyllm
    import flyllm.config
    return Animal(
        name='fly',
        read_config=flyllm.config.read_config,
        make_dataset=experiments.flyllm.make_dataset,
        posenames=flyllm.config.posenames,
        featangle=flyllm.config.featangle,
        length_unit='mm',
        dataset_source_file_names=['experiments/flyllm.py', 'flyllm/config.py', 'flyllm/features.py'],
    )


def _mouse() -> Animal:
    import experiments.mouse
    import mousellm.config
    return Animal(
        name='mouse',
        read_config=mousellm.config.read_config,
        make_dataset=experiments.mouse.make_dataset,
        posenames=mousellm.config.posenames,
        featangle=mousellm.config.featangle,
        length_unit='cm',
        dataset_source_file_names=['experiments/mouse.py', 'mousellm/config.py', 'mousellm/features.py'],
        debug_plot_kwargs={'kptidx': mousellm.config.keypointidx, 'skelidx': mousellm.config.skeleton_edges,
                           'outnames': mousellm.config.posenames},
    )


ANIMAL_FUNCTIONS = {'fly': _fly, 'mouse': _mouse}


def get_animal(name: str) -> Animal:
    """ The Animal for the 'animal' config value name. """
    if name not in ANIMAL_FUNCTIONS:
        raise ValueError(f"animal is {name!r}, but must be one of {list(ANIMAL_FUNCTIONS)}")
    return ANIMAL_FUNCTIONS[name]()


def config_animal(config: dict) -> Animal:
    """ The Animal a config dict is for. """
    return get_animal(config.get('animal', DEFAULT_ANIMAL_NAME))


def config_file_animal(config_file_name: str) -> Animal:
    """ The Animal the config file named config_file_name is for, from its 'animal' entry, before its
    animal's defaults are filled in. """
    with open(config_file_name) as f:
        return get_animal(json.load(f).get('animal', DEFAULT_ANIMAL_NAME))


def read_config(config_file_name: str) -> dict:
    """ Reads the config file named config_file_name, with its animal's defaults, and sets 'animal'. """
    animal = config_file_animal(config_file_name)
    config = animal.read_config(config_file_name)
    config['animal'] = animal.name
    return config
