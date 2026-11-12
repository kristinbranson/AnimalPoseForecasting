"""Constants for modeling the MABe22 mouse triplets: keypoints, pose features, sensory parameters.

Positions are in cm, in a frame centered on the typical arena center, with x to the right and y down in
the video image.  See mousellm/convert_mabe22.py for how the MABe22 pixel coordinates are converted.
"""
import os
import pathlib

import numpy as np

codedir = pathlib.Path(__file__).parent.resolve()
DEFAULTCONFIGFILE = os.path.join(codedir, 'config_mouse_llm_default.json')
assert os.path.exists(DEFAULTCONFIGFILE), f"{DEFAULTCONFIGFILE} does not exist."

FPS = 30

# Keypoints, in the order of the MABe22 (JAX) data.  Left/right pairs must differ only in 'left'/'right',
# for flip augmentation (apf.data.get_flip_idx).
keypointnames = [
    'nose',
    'left_ear',
    'right_ear',
    'neck',
    'left_forepaw',
    'right_forepaw',
    'center',
    'left_hindpaw',
    'right_hindpaw',
    'tail_base',
    'tail_middle',
    'tail_tip',
]
keypointidx = np.arange(len(keypointnames), dtype=int)
skeleton_edges = np.array([
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [3, 6],
    [6, 9],
    [9, 10],
    [10, 11],
    [3, 4],
    [3, 5],
    [9, 7],
    [9, 8],
])

# The body frame: origin at the center keypoint, y axis from the tail base to the neck.
kporigin = keypointnames.index('center')
kpheading_back = keypointnames.index('tail_base')
kpheading_front = keypointnames.index('neck')

# Pose features, from BodyCentricKP: the origin and orientation, then the x and y of every keypoint but
# the origin one (which is always (0, 0)) in the body frame.
kprelative = [i for i in range(len(keypointnames)) if i != kporigin]
posenames = ['center_x', 'center_y', 'orientation'] + \
    [f'{keypointnames[i]}_{c}' for i in kprelative for c in ['x', 'y']]
featglobal = [0, 1, 2]
featthetaglobal = 2
featrelative = np.ones(len(posenames), dtype=bool)
featrelative[featglobal] = False
featangle = np.zeros(len(posenames), dtype=bool)
featangle[featthetaglobal] = True
nfeatures = len(posenames)

# Pixel to cm conversion, for pixels in the 512 x 512 frame of the MABe22 files with videos.  The 52 cm x
# 52 cm arena floor is 379 px across in the median video, centered at (256, 239) px, though both vary from
# video to video (step0/arena_bounds_from_video.py in the project dir).
ARENA_SIZE_CM = 52.
ARENA_SIZE_PX = 379.43
CM_PER_PX = ARENA_SIZE_CM / ARENA_SIZE_PX
ARENA_CENTER_PX = np.array([256., 238.86])

# Typical nose to tail base distance, in cm (44 px, median over user train)
BODY_LENGTH_CM = 6.

SENSORY_PARAMS = {
    # vision of the other mice, from the head, in n_oma direction bins
    'n_oma': 72,
    'vision_kpnames': ['nose', 'neck', 'center', 'tail_base'],
    'othermice_vision_exp': .6,
    # touch: the distance from each touch keypoint to the nearest touch_other keypoint of another mouse
    'touch_kpnames': keypointnames,
    'touch_other_kpnames': ['nose', 'neck', 'center', 'tail_base', 'tail_middle'],
    'othermice_touch_exp': 1.3,
}
# Vision reaches across the arena diagonal, and touch one body length.
SENSORY_PARAMS['othermice_vision_mult'] = 1. / ((np.sqrt(2.) * ARENA_SIZE_CM) ** SENSORY_PARAMS['othermice_vision_exp'])
SENSORY_PARAMS['othermice_touch_mult'] = 1. / (BODY_LENGTH_CM ** SENSORY_PARAMS['othermice_touch_exp'])

# The head point vision is from: the mean of these keypoints.
kphead = [keypointnames.index(x) for x in ['nose', 'left_ear', 'right_ear']]
kpvision_other = [keypointnames.index(x) for x in SENSORY_PARAMS['vision_kpnames']]
kptouch = [keypointnames.index(x) for x in SENSORY_PARAMS['touch_kpnames']]
kptouch_other = [keypointnames.index(x) for x in SENSORY_PARAMS['touch_other_kpnames']]


def read_config(configfile, **kwargs):
    from apf.io import read_config as apf_read_config
    return apf_read_config(
        configfile,
        default_configfile=DEFAULTCONFIGFILE,
        posenames=posenames,
        featglobal=featglobal,
        **kwargs,
    )
