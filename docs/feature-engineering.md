<link rel="stylesheet" href="style.css">

# Feature Engineering

To add multiple transformations to the observations of your
environment, follow the following steps:

1. Subclass `hearts_gym.envs.ObsTransform`.
2. Implement a `transform` method with the same signature as
   given by the superclass.
3. Add the newly implemented class to the list of transformations
   `obs_transforms` in `configuration.py`, for example like this:

   ```python
   # Note that we are adding instances of the class.
   obs_transforms: List[ObsTransform] = [
       MyObsTransform(),
   ]
   ```
4. Adjust the first definition of the `obs_space` variable in the
   `HeartsEnv.__init__` method according to the observation space you
   receive after running through all transformations. [Look through
   the `gym.spaces`
   module](https://github.com/openai/gym/tree/master/gym/spaces) for
   help with this step.

The transformations in `obs_transforms` will be applied in order of
appearing in the list.
