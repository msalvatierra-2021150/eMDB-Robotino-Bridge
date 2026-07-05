from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robotino_emdb_experiments'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        ('share/' + package_name,
            ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
            glob('launch/*launch.py')),

        (os.path.join('share', package_name, 'experiments'),
            glob('experiments/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mike',
    maintainer_email='salvmike0@gmail.com',
    description='Robotino e-MDB experiments',
    license='MIT',
    entry_points={
        'console_scripts': [],
    },
)