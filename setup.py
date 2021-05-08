from setuptools import find_packages, setup

from shift_planning import __version__

setup(
    name="shift_planning",
    version=__version__,
    description="Shift Planning Tool",
    url="https://github.com/DmtUsa/shift_planning",
    author="Dmitrii Usanov",
    author_email="usanovdmal@gmail.com",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=[
        "matplotlib==3.1.3",
        "numpy==1.18.1",
        "pandas==1.0.1",
    ],
    tests_require=["pytest"],
    python_requires=">=3.6",
)
