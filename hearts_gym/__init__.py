# from gym.envs.registration import register
from ray.tune import register_env

from hearts_gym.envs import HeartsEnv

# register(
#     id='Hearts-v0',
#     entry_point='hearts_gym.envs:HeartsEnv',
# )


def register_envs() -> None:
    """Register all `hearts_gym` environments in `ray`."""
    register_env('Hearts-v0', lambda config: HeartsEnv(**config))


register_envs()
