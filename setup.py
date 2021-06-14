from setuptools import setup

setup(
    name='hearts_gym',
    python_requires='>=3.6',
    version='0.0.1',
    install_requires=[
        'gym>=0.18.0',
        'numpy>=1.17',
        'ray[default,rllib,tune]>=1.3.0,<2.0',
    ],
)
