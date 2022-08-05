from setuptools import setup

setup(
    name='earl_benchmark',
    packages=['earl_benchmark'],
    version='0.0.1',
    install_requires=[
        'gym==0.23.1'
        'metaworld',
        'mujoco-py==2.0.2.9',
        'numpy==1.21.5',
        'matplotlib',
        'scipy',
        'pybullet==3.2.0',
        'termcolor'
    ],
    dependency_links=[
        'git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld',
    ],
    include_package_data=True
)
