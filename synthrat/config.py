import numpy as np
import re
import pathlib
import os

codedir = pathlib.Path(__file__).parent.resolve()
DEFAULTCONFIGFILE = os.path.join(codedir, 'config_synthrat_default.json')
assert os.path.exists(DEFAULTCONFIGFILE), f"{DEFAULTCONFIGFILE} does not exist."

# names of features
posenames = ['forward','sideways','orientation']
read_config_kwargs = {
    'default_configfile': DEFAULTCONFIGFILE,
    'posenames': posenames,
    'featglobal': posenames,
}