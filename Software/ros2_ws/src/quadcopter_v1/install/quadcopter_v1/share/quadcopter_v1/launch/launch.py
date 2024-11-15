from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    gazebo_ros_pkg = get_package_share_directory('gazebo_ros')
    quadcopter_pkg = get_package_share_directory('quadcopter_v1')

    # Path to the SDF model file
    model_path = os.path.join(quadcopter_pkg, 'models', 'quadcopter_with_camera.sdf')
    # Path to the world file
    world_path = os.path.join(quadcopter_pkg, 'models', 'environment.world')

    # Include Gazebo with the specified world file
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkg, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_path}.items()
    )

    # Spawn the quadcopter entity
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'quadcopter_with_camera', '-file', model_path],
        output='screen'
    )

    # Add your sensor reading script as a node
    sensor_reader_node = Node(
        package='quadcopter_v1',   # Your package name
        executable='sensor_reader.py',  # Your script's entry point (e.g., Python/C++ executable name)
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        spawn_entity
    ])

