import numpy as np
import re
import pathlib
import os

import apf.io

codedir = pathlib.Path(__file__).parent.resolve()
REPODIR = codedir.parent
DEFAULTCONFIGFILE = os.path.join(codedir, 'config_synthrat_default.json')
assert os.path.exists(DEFAULTCONFIGFILE), f"{DEFAULTCONFIGFILE} does not exist."

# names of features
posenames = ['forward','sideways','orientation']
read_config_kwargs = {
    'default_configfile': DEFAULTCONFIGFILE,
    'posenames': posenames,
    'featglobal': posenames,
}

# Directories a config may name relative to the repository, so a checkout works wherever
# it sits: 'datadir' holds the generated trajectories, 'savedir' the trained models.
REPO_RELATIVE_DIRS = ('datadir', 'savedir')


def resolve_repo_relative_dirs(config: dict, repodir: str = REPODIR) -> dict:
    """Turn a config's relative directories into absolute paths under the repository.

    A relative 'datadir' would otherwise be resolved against the working directory, so the
    same config would read different data depending on where it was run from. Resolving
    against the repository instead makes a checkout self-contained: generation writes into
    `synthrat/data` beside the scripts, and training reads the same place. Absolute paths
    are left alone, so a config can still point at data stored outside the checkout.

    Args:
        config: a config dict, modified in place.
        repodir: the directory relative paths are resolved against; the repository root.

    Returns:
        The same config.
    """
    for key in REPO_RELATIVE_DIRS:
        value = config.get(key)
        if value is not None and not os.path.isabs(value):
            config[key] = os.path.join(repodir, value)
    return config


def read_config(jsonfile: str = DEFAULTCONFIGFILE, **kwargs) -> dict:
    """Read a synthrat config, with its directories resolved against the repository.

    Wraps `apf.io.read_config` with synthrat's defaults, then fixes up the directories and
    recomputes the file paths that `apf.io.read_config` builds from `datadir`.

    Args:
        jsonfile: the config to read; its entries override the synthrat defaults.
        kwargs: passed to apf.io.read_config, overriding read_config_kwargs.

    Returns:
        The config dict, with absolute 'datadir', 'savedir', 'intrainfile' and 'invalfile'.
    """
    config = apf.io.read_config(jsonfile, **{**read_config_kwargs, **kwargs})
    resolve_repo_relative_dirs(config)
    config['intrainfile'] = os.path.join(config['datadir'], config['intrainfilestr'])
    config['invalfile'] = os.path.join(config['datadir'], config['invalfilestr'])
    return config
