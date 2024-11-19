import sys
if sys.prefix == '/Users/jakobc/miniforge3/envs/ros_env':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/Users/jakobc/Documents/GitHub/Drone/Software/ros2_ws/src/quadcopter_v1/install/quadcopter_v1'
