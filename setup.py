from setuptools import setup

setup(
    name='hearts_gym',
    packages=['hearts_gym'],
    # Ray wheels built up to and including 3.9.
    python_requires='>=3.7<3.10',
    version='0.0.1',
    install_requires=[
        'gym>=0.18.0',
        'numpy>=1.17',
        'ray[default,rllib,tune]==1.12,<2.0',
    ],
)
