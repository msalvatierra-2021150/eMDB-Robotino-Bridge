from setuptools import find_packages, setup

package_name = "robotino_emdb_decision"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mike",
    maintainer_email="salvmike0@gmail.com",
    description="Robotino adapters for official GII e-MDB policy execution.",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "policy_selector = "
            "robotino_emdb_decision.robotino_policy_selector:main",
            "policy_execution_bridge = "
            "robotino_emdb_decision.robotino_policy_execution_bridge:main",
            "demo_autonomy = "
            "robotino_emdb_decision.robotino_demo_autonomy:main",
        ],
    },
)