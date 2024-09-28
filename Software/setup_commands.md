sudo apt update  #Updating package list
sudo apt upgrade -y  #Upgrading the packages to latest versions
sudo apt install -y build-essential cmake git
sudo apt install -y python3 python3-pip
# Clone github repository
git clone https://github.com/j-ceric/Drone.git 
# check for UTF-8
locale 
sudo apt update && sudo apt install locales
sudo apt update && sudo apt install curl gnupg2 lsb-release
# Add ROS2 APT repository to get the ROS2 packages
sudo apt update && sudo apt install curl gnupg2 lsb-release
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
# install some pip packages needed for testing
python3 -m pip install -U   argcomplete   flake8   flake8-blind-except   flake8-builtins   flake8-class-newline   flake8-comprehensions   flake8-deprecated   flake8-docstrings   flake8-import-order   flake8-quotes   pytest-repeat   pytest-rerunfailures   pytest   pytest-cov   pytest-runner   setuptools 
# install Fast-RTPS dependencies
sudo apt install --no-install-recommends -y   libasio-dev   libtinyxml2-dev
mkdir -p ~/ros2_eloquent/src
cd ~/ros2_eloquent
wget https://raw.githubusercontent.com/ros2/ros2/eloquent/ros2.repos 
#Initialize and update rosdep (common dependencies)
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro eloquent -y --skip-keys "console_bridge fastcdr fastrtps libopensplice67 libopensplice69 rti-connext-dds-5.3.1 urdfdom_headers
#install ROS2 Base Package
sudo apt install ros-eloquent-desktop 
# Source the ROS2 setup file
source /opt/ros/eloquent/setup.bash
# Add previous line to ~/.bashrc file, to automatically source the setup on each terminal start
echo "source /opt/ros/eloquent/setup.bash" >> ~/.bashrc
source ~/.bashrc
