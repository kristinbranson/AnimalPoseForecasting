import numpy as np
import re
import pathlib
import os

codedir = pathlib.Path(__file__).parent.resolve()
DEFAULTCONFIGFILE = os.path.join(codedir, 'config_fly_llm_default.json')
assert os.path.exists(DEFAULTCONFIGFILE), f"{DEFAULTCONFIGFILE} does not exist."


# Names of features
keypointnames = [
    'wing_left',
    'wing_right',
    'antennae_midpoint',
    'right_eye',
    'left_eye',
    'left_front_thorax',
    'right_front_thorax',
    'base_thorax',
    'tip_abdomen',
    'right_middle_femur_base',
    'right_middle_femur_tibia_joint',
    'left_middle_femur_base',
    'left_middle_femur_tibia_joint',
    'right_front_leg_tip',
    'right_middle_leg_tip',
    'right_back_leg_tip',
    'left_back_leg_tip',
    'left_middle_leg_tip',
    'left_front_leg_tip',
    # 'right_outer_wing',
    # 'left_outer_wing'
]

posenames = [
    'thorax_front_x',
    'thorax_front_y',
    'orientation',
    'head_base_x',
    'head_base_y',
    'head_angle',
    'abdomen_angle',
    'left_middle_femur_base_dist',
    'left_middle_femur_base_angle',
    'right_middle_femur_base_dist',
    'right_middle_femur_base_angle',
    'left_middle_femur_tibia_joint_dist',
    'left_middle_femur_tibia_joint_angle',
    'left_front_leg_tip_dist',
    'left_front_leg_tip_angle',
    'right_front_leg_tip_dist',
    'right_front_leg_tip_angle',
    'right_middle_femur_tibia_joint_dist',
    'right_middle_femur_tibia_joint_angle',
    'left_middle_leg_tip_dist',
    'left_middle_leg_tip_angle',
    'right_middle_leg_tip_dist',
    'right_middle_leg_tip_angle',
    'left_back_leg_tip_dist',
    'left_back_leg_tip_angle',
    'right_back_leg_tip_dist',
    'right_back_leg_tip_angle',
    'left_wing_angle',
    'right_wing_angle',
]

scalenames = [
    'thorax_width',
    'thorax_length',
    'abdomen_length',
    'wing_length',
    'head_width',
    'head_height',
]

vision_kpnames_v1 = [
    'antennae_midpoint',
    'tip_abdomen',
    'left_middle_femur_base',
    'right_middle_femur_base',
]

touch_other_kpnames_v1 = [
    'antennae_midpoint',
    'left_front_thorax',
    'right_front_thorax',
    'base_thorax',
    'tip_abdomen',
]

keypointidx = np.arange(19,dtype=int)
skeleton_edges = np.array([
  [ 7,  8],
  [10, 14],
  [11, 12],
  [12, 17],
  [ 7, 11],
  [ 9, 10],
  [ 7,  9],
  [ 5,  7],
  [ 2,  3],
  [ 2,  7],
  [ 5, 18],
  [ 6, 13],
  [ 7, 16],
  [ 7, 15],
  [ 2,  4],
  [ 6,  7],
  [ 7,  0],
  [ 7,  1]
  ])

# Sensory parameters
ARENA_RADIUS_MM = 26.689  # size of the arena the flies are enclosed in
ARENA_RADIUS_PX = 507.611429  # median over all videos
PXPERMM = ARENA_RADIUS_PX/ARENA_RADIUS_MM
SENSORY_PARAMS = {
    'n_oma': 72,
    'inner_arena_radius': 17.5,  # in mm
    'outer_arena_radius': ARENA_RADIUS_MM,
    'arena_height': 3.5,
    'otherflies_vision_exp': .6,
    'touch_kpnames': keypointnames,
    'vision_kpnames': vision_kpnames_v1,
    'touch_other_kpnames': touch_other_kpnames_v1,
    'compute_otherflies_touch': True,
    'otherflies_touch_exp': 1.3,
    'otherflies_touch_mult': 0.3110326159171111,  # set 20230807 based on courtship male data
}
SENSORY_PARAMS['otherflies_vision_mult'] = 1. / ((2. * ARENA_RADIUS_MM) ** SENSORY_PARAMS['otherflies_vision_exp'])

# Pose feature indices
featorigin = [posenames.index('thorax_front_x'), posenames.index('thorax_front_y')]
feattheta = posenames.index('orientation')
featglobal = featorigin + [feattheta, ]
featthetaglobal = 2

# Keypoint feature indices
kpvision_other = [keypointnames.index(x) for x in SENSORY_PARAMS['vision_kpnames']]
kpeye = keypointnames.index('antennae_midpoint')
kptouch = [keypointnames.index(x) for x in SENSORY_PARAMS['touch_kpnames']]
kptouch_other = [keypointnames.index(x) for x in SENSORY_PARAMS['touch_other_kpnames']]

# Indicator vectors
featrelative = np.ones(len(posenames), dtype=bool)
featrelative[featglobal] = False
featangle = np.array([re.search('angle$', s) is not None for s in posenames])
featangle[feattheta] = True

# Counts
nrelative = np.count_nonzero(featrelative)
nglobal = len(featglobal)
nfeatures = len(posenames)
nkptouch = len(kptouch)
nkptouch_other = len(kptouch_other)
narena = 2 ** 10

# Other variables
theta_arena = np.linspace(-np.pi, np.pi, narena + 1)[:-1]

def read_config(configfile, **kwargs):
    from apf.io import read_config as apf_read_config
    from flyllm.features import featglobal, get_sensory_feature_idx
    return apf_read_config(
        configfile,
        default_configfile=DEFAULTCONFIGFILE,
        posenames=posenames,
        featglobal=featglobal,
        get_sensory_feature_idx=get_sensory_feature_idx,
        **kwargs,
    )