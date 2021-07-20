"""
Common utility methods for RLlib.
"""

import multiprocessing
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from gym.spaces import Space
import ray
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.models import MODEL_DEFAULTS, ModelCatalog
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
    PolicyID,
    TrainerConfigDict,
)
from ray.tune.registry import (
    get_trainable_cls,
    _global_registry,
    ENV_CREATOR,
    RLLIB_MODEL,
)
from ray.tune.trainable import Trainable

from hearts_gym.utils.typing import Seed

__all__ = [
    'DEFAULT_FRAMEWORK',
    'parse_bool',
    'get_default',
    'default_policies',
    'create_custom_rulebased_policies',
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
    (
        config
        for (fw, config) in [(tf, 'tf'), (th, 'torch'), (jax, 'jax')]
        if fw is not None
    ),
    None,
)
"""Configuration option for the first framework available in the
following order:
- TensorFlow
- PyTorch
- JAX
"""
assert DEFAULT_FRAMEWORK is not None, \
    'please install a deep learning framework (TensorFlow, PyTorch, or JAX)'

MASKED_ACTIONS_MODEL_KEY = 'masked_actions'


def parse_bool(string: str) -> bool:
    """Return whether the given string is "True" or "False".

    If the string is neither, raise an error.

    Args:
        string (str): String to parse for a boolean value.

    Returns:
        bool: `True` when the string was "True", `False` when the string
            was "False".
    """
    assert string == 'False' or string == 'True', \
        'please only use "False" or "True" as boolean arguments.'
    return string != 'False'


def get_default(query_dict: Dict, key: Any, default_dict: Dict) -> Any:
    """Return the entry for the given key in the given dictionary or the
    entry in `default_dict` if `query_dict` does not have an entry under
    `key`.

    `default_dict` is expected to contain `key`.

    Args:
        query_dict (Dict): Dictionary to get the value under the given
            key from.
        key (Any): Key to query.
        default_dict (Dict): Dictionary containing a default value under
            the given key that is returned if `query_dict` does not have
            an entry for it.

    Returns:
        Any: Queried value obtained from `query_dict` or `default_dict`,
            depending on inclusion of `key`.
    """
    return query_dict.get(key, default_dict[key])


def default_policies(
        env_name: str,
        env_config: EnvConfigDict,
        learned_policy_id: PolicyID,
        random_policy_id: PolicyID,
        rulebased_policy_id: PolicyID,
        random_policy_seed: Seed,
) -> Dict[PolicyID, Tuple[Optional[type], Space, Space, Dict[str, Any]]]:
    """Return a correctly configured dictionary of the default policies
    to be used in a configuration.

    Args:
        env_name (str): Name of the environment the policies will
            act in.
        env_config (EnvConfigDict): Configuration of the environment the
            policies will act in.
        learned_policy_id (PolicyID): ID of the learned policy.
        random_policy_id (PolicyID): ID of the random policy.
        rulebased_policy_id (PolicyID): ID of the rule-based policy.
        random_policy_seed (Seed): Random number generator seed for the
            random policy.
    """
    from hearts_gym import HeartsEnv
    mask_actions = env_config.get(
        'mask_actions', HeartsEnv.MASK_ACTIONS_DEFAULT)
    obs_space, act_space = get_spaces(env_name, env_config)

    from hearts_gym.policies import RandomPolicy, RuleBasedPolicy
    return {
        learned_policy_id: (None, obs_space, act_space, {}),
        random_policy_id: (
            RandomPolicy,
            obs_space,
            act_space,
            {'seed': random_policy_seed, 'mask_actions': mask_actions},
        ),
        rulebased_policy_id: (
            RuleBasedPolicy,
            obs_space,
            act_space,
            {'mask_actions': mask_actions},
        ),
    }


def create_custom_rulebased_policies(
        env_name: str,
        env_config: EnvConfigDict,
        custom_rulebased_policies: Dict[PolicyID, type],
) -> Dict[PolicyID, Tuple[type, Space, Space, Dict[str, Any]]]:
    """Return a correctly configured dictionary of the default policies
    to be used in a configuration.

    Args:
        env_name (str): Name of the environment the policies will
            act in.
        env_config (EnvConfigDict): Configuration of the environment the
            policies will act in.
        learned_policy_id (PolicyID): ID of the learned policy.
    """
    from hearts_gym import HeartsEnv
    mask_actions = env_config.get(
        'mask_actions', HeartsEnv.MASK_ACTIONS_DEFAULT)
    obs_space, act_space = get_spaces(env_name, env_config)

    from hearts_gym.policies import RuleBasedPolicy
    return {
        policy_id: (
            RuleBasedPolicy,
            obs_space,
            act_space,
            {'mask_actions': mask_actions, 'policy_impl_cls': policy_impl_cls},
        )
        for (policy_id, policy_impl_cls) in custom_rulebased_policies.items()
    }


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
    env_config: EnvConfigDict = get_default(
        config, 'env_config', COMMON_CONFIG)
    obs_space = get_preprocessed_obs_space(config['env'], env_config)

    model_config: ModelConfigDict = get_default(config, 'model', COMMON_CONFIG)
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
    prev_hiddens = get_default(config, 'hiddens', DQN_DEFAULT_CONFIG)
    if prev_hiddens:
        print(f'Warning: setting `config["hiddens"] = [] '
              f'(was {prev_hiddens})`')
    prev_dueling = get_default(config, 'dueling', DQN_DEFAULT_CONFIG)
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
    framework = get_default(config, 'framework', COMMON_CONFIG)
    masked_actions_model_key = MASKED_ACTIONS_MODEL_KEY

    # Validate early because we change some stuff.
    ModelCatalog._validate_config(config=model_config, framework=framework)

    prev_custom_model = get_default(
        model_config, 'custom_model', MODEL_DEFAULTS)
    if (
            isinstance(prev_custom_model, str)
            and prev_custom_model.startswith(MASKED_ACTIONS_MODEL_KEY)
    ):
        # We already configured this.
        return

    use_lstm = get_default(model_config, 'use_lstm', MODEL_DEFAULTS)
    if use_lstm:
        model_config['use_lstm'] = False
        masked_actions_model_key = masked_actions_model_key + '_lstm'

    use_attention = get_default(model_config, 'use_attention', MODEL_DEFAULTS)
    if use_attention:
        # TensorFlow eager mode does not support attention yet.
        assert framework in ['tf', 'torch'], \
            f'attention not properly supported for framework {framework}'
        model_config['use_attention'] = False
        masked_actions_model_key = masked_actions_model_key + '_attn'

    # TODO handle other DQN variants
    if algorithm == 'DQN':
        _adjust_dqn_config(config)
        if _is_tf_framework(framework):
            masked_actions_model_key = 'q_' + masked_actions_model_key

    model_config['custom_model'] = masked_actions_model_key
    custom_model_config = get_default(
        model_config, 'custom_model_config', MODEL_DEFAULTS)
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
    env_config: EnvConfigDict = get_default(
        config, 'env_config', COMMON_CONFIG)
    from hearts_gym import HeartsEnv
    if not env_config.get('mask_actions', HeartsEnv.MASK_ACTIONS_DEFAULT):
        return

    register_masked_actions_models(
        get_default(config, 'framework', COMMON_CONFIG))

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
