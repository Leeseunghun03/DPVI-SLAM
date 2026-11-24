from yacs.config import CfgNode as CN

_C = CN()

# max number of keyframes
_C.BUFFER_SIZE = 4096

# bias patch selection towards high gradient regions?
_C.CENTROID_SEL_STRAT = 'RANDOM'

# VO config (increase for better accuracy)
_C.PATCHES_PER_FRAME = 80
_C.REMOVAL_WINDOW = 20
_C.OPTIMIZATION_WINDOW = 12
_C.PATCH_LIFETIME = 12

# threshold for keyframe removal
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5

# camera motion model
_C.MOTION_MODEL = 'DAMPED_LINEAR'
_C.MOTION_DAMPING = 0.5

_C.MIXED_PRECISION = True

# Loop closure
_C.LOOP_CLOSURE = False
_C.BACKEND_THRESH = 64.0
_C.MAX_EDGE_AGE = 1000
_C.GLOBAL_OPT_FREQ = 15

# Classic loop closure
_C.CLASSIC_LOOP_CLOSURE = False
_C.LOOP_CLOSE_WINDOW_SIZE = 3
_C.LOOP_RETR_THRESH = 0.04

# Inertial
_C.ENABLE_IMU = False
_C.GRAVITY = 9.81
_C.T_BC = [1.0, 0.0, 0.0, 0.0,
           0.0, 1.0, 0.0, 0.0,
           0.0, 0.0, 1.0, 0.0,
           0.0, 0.0, 0.0, 1.0] # camera to body, row-major
_C.ACC_NOISE = 2.0e-3      # m/s^2/sqrt(Hz)
_C.GYR_NOISE = 1.7e-4      # rad/s/sqrt(Hz)
_C.ACC_BIAS_NOISE = 3.0e-3 # m/s^3/sqrt(Hz)
_C.GYR_BIAS_NOISE = 2.0e-5 # rad/s^2/sqrt(Hz)
_C.VI_WARMUP = 12          # keyframes before visual-inertial alignment
_C.VI_INIT_WINDOW = 8

cfg = _C
