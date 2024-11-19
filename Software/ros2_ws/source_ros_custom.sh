#!/bin/bash
# Custom ROS 2 setup script for Miniforge environment

# Activate the Miniforge environment
source ~/miniforge3/envs/ros_env/bin/activate

# Source the Miniforge `local_setup.sh` explicitly
if [ -f ~/miniforge3/envs/ros_env/local_setup.sh ]; then
    . ~/miniforge3/envs/ros_env/local_setup.sh
else
    echo "local_setup.sh not found in ~/miniforge3/envs/ros_env"
fi

# Manually source the ROS 2 workspace setup file
if [ -f ~/Documents/GitHub/Drone/Software/ros2_ws/install/setup.bash ]; then
    . ~/Documents/GitHub/Drone/Software/ros2_ws/install/setup.bash
else
    echo "setup.bash not found in ~/Documents/GitHub/Drone/Software/ros2_ws/install"
fi

