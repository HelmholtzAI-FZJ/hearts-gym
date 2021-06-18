"""
Common utility methods for RLlib.
"""

import multiprocessing
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from gym.spaces import Space
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.framework import (
    try_import_jax,
    try_import_tf,
    try_import_torch,
)
from ray.rllib.utils.typing import (
    EnvConfigDict,
    EnvType,
    ModelConfigDict,
    TrainerConfigDict,
)
from ray.tune.registry import (
    get_trainable_cls,
    _global_registry,
    ENV_CREATOR,
    RLLIB_MODEL,
)
from ray.tune.trainable import Trainable

# FIXME take default config options from RLlib (e.g. COMMON_CONFIG, MODEL_DEFAULTS, ...)

__all__ = [
    'DEFAULT_FRAMEWORK',
    'parse_bool',
    'fix_ray_shutdown',
    'get_registered_env',
    'register_model',
    'get_registered_model',
    'preprocessed_get_default_model',
    'get_default_model',
    'register_masked_actions_models',
    'maybe_set_up_masked_actions_model',
    'get_num_gpus',
    'get_num_cpus',
    'get_spaces',
    'to_preprocessed_obs_space',
    'get_preprocessed_obs_space',
    'create_agent',
    'load_agent',
]

_, tf, _ = try_import_tf()
th, _ = try_import_torch()
jax, _ = try_import_jax()

DEFAULT_FRAMEWORK = next(
    config
    for (fw, config) in [(tf, 'tf'), (th, 'th'), (jax, 'jax')]
    if fw is not None
)
"""Configuration option for the first framework available in the
following order:
- TensorFlow
- PyTorch
- JAX
"""
MASKED_ACTIONS_MODEL_KEY = 'masked_actions'


def parse_bool(string):
    assert string == 'False' or string == 'True', \
        'please only use "False" or "True" as boolean arguments.'
    return string != 'False'


def fix_ray_shutdown() -> None:
    """Modify `ray` in order to fix its shutdown behaviour."""
    old_kill_process = ray.node.Node._kill_process_type

    def new_kill_process(
            self,
            process_type,
            allow_graceful=False,
            check_alive=True,
            wait=None,
    ):
        return old_kill_process(
            self,
            process_type,
            allow_graceful,
            check_alive,
            wait if wait is not None else True,
        )

    ray.node.Node._kill_process_type = new_kill_process


def get_registered_env(name: str) -> Callable[[EnvConfigDict], EnvType]:
    """Return the environment registered in `ray` under the given name.

    Args:
        name (str): Name to query.

    Returns:
        Callable[[EnvConfigDict], EnvType]: An environment creation
            function, called with the environment configuration
            dictionary as its only argument.
    """
    return _global_registry.get(ENV_CREATOR, name)


def register_model(name: str, cls: type) -> None:
    """Register the given model class in `ray` under the given name.

    Args:
        name (str): Name to register the model under.
        cls (type): Class of the model to register.
    """
    ModelCatalog.register_custom_model(name, cls)


def get_registered_model(name: str) -> type:
    """Return the model class registered in `ray` under the given name.

    Args:
        name (str): Name to query.

    Returns:
        type: A model class.
    """
    return _global_registry.get(RLLIB_MODEL, name)


def preprocessed_get_default_model(
        obs_space: Space,
        model_config: ModelConfigDict,
        framework: str,
) -> type:
    """Return the default model class as specified by RLlib for an
    environment with the given preprocessed observation space.

    Args:
        obs_space (Space): An already preprocessed observation space.
        model_config (ModelConfigDict): Configuration for the model.
        framework (str): Deep learning framework used.

    Returns:
        type: The default model class for the given preprocessed
            observation space.

    """
    model_cls = ModelCatalog._get_v2_model_class(
        obs_space, model_config, framework=framework)
    return model_cls


def get_default_model(
        config: TrainerConfigDict,
        framework: str,
) -> type:
    """Return the default model class as specified by RLlib for the
    given configuration.

    Args:
        config (TrainerConfigDict): Training configuration.
        framework (str): Deep learning framework used.

    Returns:
        type: The default model class for the given configuration.
    """
    env_config: EnvConfigDict = config.get('env_config', {})
    obs_space = get_preprocessed_obs_space(config['env'], env_config)

    model_config: ModelConfigDict = config.get('model', {})
    model_cls = preprocessed_get_default_model(
        obs_space, model_config, framework)
    return model_cls


def _is_tf_framework(framework: str) -> bool:
    """Return whether the given framework is based on TensorFlow.

    Args:
        framework (str): Deep learning framework to query for.

    Returns:
        bool: hether the given framework is based on TensorFlow.
    """
    return framework in ['tf', 'tf2', 'tfe']


def _adjust_dqn_config(config: TrainerConfigDict) -> None:
    """Modify the given configuration in-place so action masking with a DQN
    model does not cause an error.

    Args:
        config (TrainerConfigDict): Training configuration.
    """
    prev_hiddens = config.get('hiddens', [256])
    if prev_hiddens:
        print(f'Warning: setting `config["hiddens"] = [] '
              f'(was {prev_hiddens})`')
    prev_dueling = config.get('dueling', True)
    if prev_dueling:
        print(f'Warning: setting `config["dueling"] = False '
              f'(was {prev_dueling})`')

    # Optionally we could bla
    config['hiddens'] = []
    config['dueling'] = False


def _adjust_other_config_for_action_masking(
        algorithm: str,
        config: TrainerConfigDict,
) -> None:
    """Modify the given configuration in-place so action masking does not
    cause an error.

    Args:
        algorithm (str): Name of the reinforcement learning algorithm
            to use.
        config (TrainerConfigDict): Training configuration.
    """
    model_config: ModelConfigDict = config.setdefault('model', {})
    framework = config.get('framework', 'tf')
    masked_actions_model_key = MASKED_ACTIONS_MODEL_KEY

    # Validate early because we change some stuff.
    ModelCatalog._validate_config(config=model_config, framework=framework)

    prev_custom_model = model_config.get('custom_model', None)

    use_lstm = model_config.get('use_lstm', False)
    if use_lstm:
        model_config['use_lstm'] = False
        masked_actions_model_key = masked_actions_model_key + '_lstm'

    use_attention = model_config.get('use_attention', False)
    if use_attention:
        assert framework in ['tf', 'torch'], \
            f'attention not properly supported for framework {framework}'
        model_config['use_attention'] = False
        masked_actions_model_key = masked_actions_model_key + '_attn'

    # FIXME handle other DQN variants
    if algorithm == 'DQN':
        _adjust_dqn_config(config)
        if _is_tf_framework(framework):
            masked_actions_model_key = 'q_' + masked_actions_model_key

    model_config['custom_model'] = masked_actions_model_key
    custom_model_config = model_config.get('custom_model_config', {})
    model_config['custom_model_config'] = {
        **custom_model_config,
        'model_cls': prev_custom_model,
        'framework': framework,
    }


def register_masked_actions_models(framework: str) -> None:
    """Register models supporting action masking for the given framework.

    Args:
        framework (str): Deep learning framework used.
    """
    if _is_tf_framework(framework):
        from hearts_gym.models import (
            DistributionalQTFMaskedActionsWrapper,
            TFMaskedActionsAttentionWrapper,
            TFMaskedActionsRecurrentWrapper,
            TFMaskedActionsWrapper,
        )
        model_cls = TFMaskedActionsWrapper

        register_model(
            MASKED_ACTIONS_MODEL_KEY + '_lstm',
            TFMaskedActionsRecurrentWrapper,
        )
        register_model(
            MASKED_ACTIONS_MODEL_KEY + '_attn',
            TFMaskedActionsAttentionWrapper,
        )
        register_model(
            'q_' + MASKED_ACTIONS_MODEL_KEY,
            DistributionalQTFMaskedActionsWrapper,
        )
    elif framework == 'torch':
        from hearts_gym.models import (
            TorchMaskedActionsAttentionWrapper,
            TorchMaskedActionsRecurrentWrapper,
            TorchMaskedActionsWrapper,
        )
        model_cls = TorchMaskedActionsWrapper

        register_model(
            MASKED_ACTIONS_MODEL_KEY + '_lstm',
            TorchMaskedActionsRecurrentWrapper,
        )
        register_model(
            MASKED_ACTIONS_MODEL_KEY + '_attn',
            TorchMaskedActionsAttentionWrapper,
        )
    else:
        raise NotImplementedError(
            f'masked actions not available for framework {framework}')

    register_model(MASKED_ACTIONS_MODEL_KEY, model_cls)


def maybe_set_up_masked_actions_model(
        algorithm: str,
        config: TrainerConfigDict,
) -> None:
    """Do the necessary set up and modify the given configuration to support
    action masking if it is enabled.

    Args:
        algorithm (str): Name of the reinforcement learning algorithm
            to use.
        config (TrainerConfigDict): Training configuration.
    """
    env_config: EnvConfigDict = config.get('env_config', {})
    if not env_config.get('mask_actions', False):
        return

    register_masked_actions_models(config.get('framework', 'tf'))

    _adjust_other_config_for_action_masking(algorithm, config)


def get_num_gpus(framework: str) -> int:
    """Return the number of GPUs, using the given framework for querying.

    Args:
        framework (str): Deep learning framework used.

    Returns:
        int: Number of GPUs available.
    """
    if _is_tf_framework(framework):
        return len(tf.config.list_physical_devices('GPU'))
    elif framework == 'torch':
        return th.cuda.device_count()
    elif framework == 'jax':
        return jax.device_count('gpu')
    print(
        f'Warning: automatically getting number of GPUs not '
        f'available for framework {framework}; returning 0'
    )
    return 0


def get_num_cpus() -> int:
    """Return the number of CPUs.

    Note that the number of CPUs is not necessarily the number of usable
    CPUs. Use `os.sched_getaffinity(0)` for that.

    Returns:
        int: Number of CPUs available.
    """
    return multiprocessing.cpu_count()


def get_spaces(
        env_name: str,
        env_config: EnvConfigDict,
) -> Tuple[Space, Space]:
    """Return the observation and action space of the given environment.

    Args:
        env_name (str): Name of the environment to query for.
        env_config (EnvConfigDict): Environment configuration.

    Returns:
        Space: Observation space.
        Space: Action space.
    """
    env_creator = get_registered_env(env_name)
    env = env_creator(env_config)
    return (env.observation_space, env.action_space)


def to_preprocessed_obs_space(obs_space: Space) -> Space:
    """Return the given observation space in RLlib-preprocessed form.

    Args:
        obs_space (Space): Observation space to preprocess.

    Returns:
        Space: Preprocessed observation space.
    """
    prep = get_preprocessor(obs_space)(obs_space)
    return prep.observation_space


def get_preprocessed_obs_space(
        env_name: str,
        env_config: EnvConfigDict,
) -> Space:
    """Return the RLlib-preprocessed observation space for the
    given environment.

    Args:
        env_name (str): Name of the environment to query for.
        env_config (EnvConfigDict): Environment configuration.

    Returns:
        Space: Preprocessed observation space.
    """
    env_creator = get_registered_env(env_name)
    env = env_creator(env_config)

    obs_space = env.observation_space
    return to_preprocessed_obs_space(obs_space)


def create_agent(
        agent: Union[str, type, Trainable],
        config: Optional[TrainerConfigDict] = None,
) -> Trainable:
    """Return an agent instance with the given configuration.

    If passed an instance, it is returned as is; its configuration is
    not changed.

    Args:
        agent (Union[str, type, Trainable]): Name of a reinforcement
            learning algorithm, its trainer class, or a trainer instance.
        config (Optional[TrainerConfigDict]): Training configuration.

    """
    if isinstance(agent, str):
        agent = get_trainable_cls(agent)
    if isinstance(agent, type):
        agent = agent(config=config)
    return agent


def load_agent(
        agent: Union[str, type, Trainable],
        cp_path: Union[Path, str],
        config: Optional[TrainerConfigDict] = None,
) -> Trainable:
    """Return an agent instance with the given configuration.

    If passed an instance, its configuration is not changed; the
    checkpoint is still appropriately loaded, though.

    Args:
        agent (Union[str, type, Trainable]): Name of a reinforcement
            learning algorithm, its trainer class, or a trainer instance.
        cp_path (Union[Path, str]): Path of a checkpoint to load.
        config (Optional[TrainerConfigDict]): Training configuration.

    Returns:
        Trainable: An agent with the checkpoint loaded on it.
    """
    agent = create_agent(agent, config)

    agent.restore(str(cp_path))
    return agent
