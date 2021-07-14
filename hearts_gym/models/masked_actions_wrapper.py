"""
Wrapper classes to support action masking for arbitrary
non-recurrent models.

Action masking gives prohibited actions a probability close to zero.
"""

from typing import Any, Tuple, Type, Union

from gym.spaces import Space
from ray.rllib.agents.dqn.distributional_q_tf_model import \
    DistributionalQTFModel
from ray.rllib.models import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import Dict, List, TensorType, ModelConfigDict

from hearts_gym import utils
from hearts_gym.envs import HeartsEnv

_, tf, _ = try_import_tf()
th, nn = try_import_torch()


def _split_input_dict(
        input_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any], TensorType]:
    """Return a modified input dictionary and its removed action mask.

    `input_dict` is modified in-place so that the "obs_flat" key
    contains the flattened observations without the action mask.

    Args:
        input_dict (Dict[str, Any]): Input dictionary containing
            observations with support for action masking.

    Returns:
        Dict[str, Any]: Input dictionary with the action mask removed
            from its flattened observations.
        TensorType: The action mask removed from the input dictionary.
    """
    action_mask = input_dict['obs'][HeartsEnv.ACTION_MASK_KEY]
    # FIXME allow ACTION_MASK_KEY to be placed anywhere, not just at
    # start (get start index)
    action_mask_len = action_mask.shape[-1]

    # The action mask is at the front as the DictFlatteningProcessor
    # sorts its dictionary's items.
    sans_action_mask = input_dict['obs_flat'][:, action_mask_len:]
    input_dict['obs_flat'] = sans_action_mask
    return input_dict, action_mask


def _create_with_adjusted_obs(
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        model_cls: Union[Type[ModelV2], str, None],
        framework: str,
) -> ModelV2:
    """Return a model constructed with an observation space adjusted to
    _not_ include an action mask.

    See also `ModelV2.__init__`.

    Args:
        obs_space (Space): Original observation space (including the
            action mask) of the environment.
        action_space (Space): Action space of the environment.
        num_outputs (int): Number of output units of the model.
        model_config (ModelConfigDict): Model configuration dictionary.
        name (str): Name (scope) to give the model.
        model_cls (Union[Type[ModelV2], str, None]): Class of the model
            to construct.
        framework (str): Deep learning framework used.

    Returns:
        ModelV2: Model instance with an adjusted observation space.
    """
    if model_cls is None:
        model_cls = utils.preprocessed_get_default_model(
            obs_space, model_config, framework)
    elif isinstance(model_cls, str):
        model_cls = utils.get_registered_model(model_cls)

    original_obs_space = utils.to_preprocessed_obs_space(
        obs_space.original_space['obs'])

    return model_cls(
        original_obs_space,
        action_space,
        num_outputs,
        model_config,
        name + '_wrapped',
    )


class TFMaskedActionsWrapper(TFModelV2):
    """Wrapper class to support action masking for arbitrary
    non-recurrent TensorFlow models.
    """

    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            *,
            model_cls: Union[Type[ModelV2], str, None] = None,
            framework: str = 'tf',
    ) -> None:
        """Construct an action masking wrapper model around a given
        model class.

        See also `TFModelV2.__init__`.

        Args:
            obs_space (Space): Observation space of the environment.
            action_space (Space): Action space of the environment.
            num_outputs (int): Number of output units of the model.
            model_config (ModelConfigDict): Model configuration dictionary.
            name (str): Name (scope) to give the model.
            model_cls (Union[Type[ModelV2], str, None]): Class of the model
                to construct.
            framework (str): Deep learning framework used.
        """
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )

        self._wrapped = _create_with_adjusted_obs(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            model_cls,
            framework,
        )
        self.view_requirements: Dict[str, ViewRequirement] = {
            **self._wrapped.view_requirements,
            SampleBatch.OBS: self.view_requirements[SampleBatch.OBS],
        }

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Return the model applied to the given inputs. Apply the action mask
        to give prohibited actions a probability close to zero.

        See also `TFModelV2.forward`.

        Args:
            input_dict (Dict[str, TensorType]): Input tensors, including
                keys "obs", "obs_flat", "prev_action", "prev_reward",
                "is_training", "eps_id", "agent_id", "infos", and "t".
            state (List[TensorType]): List of RNN state tensors.
            seq_lens (TensorType): 1-D tensor holding input
                sequence lengths.

        Returns:
            TensorType: Output of the model with action masking applied.
            List[TensorType]: New RNN state.
        """
        _, action_mask = _split_input_dict(input_dict)
        model_out, state = self._wrapped.forward(input_dict, state, seq_lens)

        # We don't use -infinity for numerical stability.
        inf_mask = tf.maximum(tf.math.log(action_mask),
                              model_out.dtype.min)
        return model_out + inf_mask, state

    def value_function(self):
        return self._wrapped.value_function()


class DistributionalQTFMaskedActionsWrapper(
        DistributionalQTFModel,
        TFMaskedActionsWrapper,
):
    """Wrapper class to support action masking for arbitrary
    non-recurrent TensorFlow models with the DQN algorithm.
    """

    pass


class TorchMaskedActionsWrapper(
        TorchModelV2,
        nn.Module,  # type: ignore[name-defined]
):
    """Wrapper class to support action masking for arbitrary
    non-recurrent PyTorch models.
    """

    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            *,
            model_cls: Union[Type[ModelV2], str, None] = None,
            framework: str = 'torch',
    ) -> None:
        """Construct an action masking wrapper model around a given
        model class.

        See also `TorchModelV2.__init__`.

        Args:
            obs_space (Space): Observation space of the environment.
            action_space (Space): Action space of the environment.
            num_outputs (int): Number of output units of the model.
            model_config (ModelConfigDict): Model configuration dictionary.
            name (str): Name (scope) to give the model.
            model_cls (Union[Type[ModelV2], str, None]): Class of the model
                to construct.
            framework (str): Deep learning framework used.
        """
        nn.Module.__init__(self)
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )

        self._wrapped = _create_with_adjusted_obs(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            model_cls,
            framework,
        )
        self.view_requirements: Dict[str, ViewRequirement] = {
            **self._wrapped.view_requirements,
            SampleBatch.OBS: self.view_requirements[SampleBatch.OBS],
        }

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Return the model applied to the given inputs. Apply the action mask
        to give prohibited actions a probability close to zero.

        See also `TorchModelV2.forward`.

        Args:
            input_dict (Dict[str, TensorType]): Input tensors, including
                keys "obs", "obs_flat", "prev_action", "prev_reward",
                "is_training", "eps_id", "agent_id", "infos", and "t".
            state (List[TensorType]): List of RNN state tensors.
            seq_lens (TensorType): 1-D tensor holding input
                sequence lengths.

        Returns:
            TensorType: Output of the model with action masking applied.
            List[TensorType]: New RNN state.
        """
        _, action_mask = _split_input_dict(input_dict)
        model_out, state = self._wrapped.forward(input_dict, state, seq_lens)

        # We don't use -infinity for numerical stability.
        inf_mask = th.maximum(th.log(action_mask),
                              th.tensor(th.finfo(model_out.dtype).min))
        return model_out + inf_mask, state

    def value_function(self):
        return self._wrapped.value_function()
