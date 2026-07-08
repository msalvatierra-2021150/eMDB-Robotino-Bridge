from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robotino_emdb_actuation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),

        (os.path.join('share', package_name, 'maps'),
            glob(os.path.join('maps', '*.yaml')) +
            glob(os.path.join('maps', '*.pgm')) +
            glob(os.path.join('maps', '*.png'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mike',
    maintainer_email='salvmike0@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'policy_executor = robotino_emdb_actuation.robotino_policy_executor:main',
        ],
    },
)
