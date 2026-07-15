from setuptools import find_packages, setup

package_name = "robotino_emdb_perception"

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
    description="Robotino perception adapters for official GII e-MDB.",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "tag_perception = "
            "robotino_emdb_perception.robotino_tag_perception:main",
            "foraging_perception = "
            "robotino_emdb_perception.robotino_foraging_perception:main",
            "context_perception = "
            "robotino_emdb_perception.robotino_context_perception:main",
        ],
    },
)