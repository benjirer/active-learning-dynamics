from setuptools import setup, find_packages

required = [
    "numpy >= 1.18",
    "opencv-python >= 4.2",
    "keyboard >= 0.13.5",
    "distro >= 1.8.0",
    "tensorboard >= 2.14.0",
    "pykalman >= 0.9.5",
    "netaddr >= 0.8",
    "netifaces >= 0.10",
    "myqr >= 2.3",
    "scikit-learn >= 1.0.2",
    "scipy >= 1.7.3",
    "gym>=0.26.0",
    "bosdyn-client>=3.3.0",
    "bosdyn-mission>=3.3.0",
    "bosdyn-choreography-client>=3.3.0",
    "shapely>=2.0.1",
    "casadi >= 3.5.5",
    "opax @ git+https://github.com/benjirer/opax.git",
    "jdm_control @ git+http://github.com/idrave/jdm_control.git",
    "urdf2casadi @ git+https://github.com/mahaarbo/urdf2casadi.git",
]

extras = {}
setup(
    name="alrd",
    version="1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    include_package_data=True,
    install_requires=required,
    extras_require=extras,
)
