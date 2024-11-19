from setuptools import find_packages, setup
from glob import glob  # Add this line to import glob
import os

package_name = 'quadcopter_v1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include the models directory
        (os.path.join('share', package_name, 'models'), glob('models/*.sdf')),
        # Include the launch directory
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jakobc',
    maintainer_email='jakobc@todo.todo',
    description='TODO: Package description',
    license='Quadcopter simulation with a camera',
    entry_points={
        'console_scripts': [
            'sensor_reader = quadcopter_v1.sensor_reader:main',
            'control = quadcopter_v1.control:main'
        ],
    },
    
)
