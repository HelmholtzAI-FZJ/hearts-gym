from setuptools import setup

setup(
    name='hearts_gym',
    # RLlib wheels built up to and including 3.8
    python_requires='>=3.6<3.9',
    version='0.0.1',
    install_requires=[
        'gym>=0.18.0',
        'numpy>=1.17',
        # 1.4.0 and 1.4.1 have forced TensorFlow installation.
        'ray[default,rllib,tune]>=1.3.0,!=1.4.0,!=1.4.1,<2.0',
    ],
)
