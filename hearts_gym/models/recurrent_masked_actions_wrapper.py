"""
Wrapper classes to support action masking for arbitrary
non-attention-based recurrent models.

Action masking gives prohibited actions a probability close to zero.
"""

from typing import Tuple, Type, Union

from gym.spaces import Space
from ray.rllib.models import ModelV2
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.tf.recurrent_net import (
    LSTMWrapper as TFLSTMWrapper,
    RecurrentNetwork as TFRecurrentNetwork,
)
from ray.rllib.models.torch.recurrent_net import (
    LSTMWrapper as TorchLSTMWrapper,
    RecurrentNetwork as TorchRecurrentNetwork,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import Dict, List, TensorType, ModelConfigDict

from hearts_gym import utils
from .masked_actions_wrapper import (
    TFMaskedActionsWrapper,
    TorchMaskedActionsWrapper,
)

_, tf, _ = try_import_tf()
th, nn = try_import_torch()


def _create_wrapped(
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        model_cls: Union[Type[ModelV2], str, None],
        wrapper_cls: type,
        framework: str,
) -> ModelV2:
    """Return a model constructed with an observation space adjusted to
    _not_ include an action mask. Also wrap the model in the given
    wrapper class.

    See also `ModelV2.__init__`.

    Args:
        obs_space (Space): Original observation space (including the
            action mask) of the environment.
        action_space (Space): Action space of the environment.
        num_outputs (int): Number of output units of the model.
        model_config (ModelConfigDict): Model configuration dictionary.
        name (str): Name (scope) to give the model.
        model_cls (Union[Type[ModelV2], str, None]): Class of the model
            to wrap.
        wrapper_cls (Union[type, str, None]): Class to wrap the model
            in. This is also used to construct the model.
        framework (str): Deep learning framework used.

    Returns:
        ModelV2: Wrapped model instance with an adjusted
            observation space.
    """
    if model_cls is None:
        model_cls = utils.preprocessed_get_default_model(
            obs_space, model_config, framework)
    elif isinstance(model_cls, str):
        model_cls = utils.get_registered_model(model_cls)

    original_obs_space = utils.to_preprocessed_obs_space(
        obs_space.original_space['obs'])

    # double_wrapped = model_cls(
    #     original_obs_space,
    #     action_space,
    #     int(np.prod(original_obs_space.shape)),
    #     model_config,
    #     name + '_wrapped',
    # )

    wrapper_cls = ModelCatalog._wrap_if_needed(model_cls, wrapper_cls)
    wrapper_cls._wrapped_forward = (  # type: ignore[attr-defined]
        model_cls.forward)
    return wrapper_cls(
        original_obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    )


class TFMaskedActionsRecurrentWrapper(
        TFRecurrentNetwork,
        TFMaskedActionsWrapper,
):
    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            *,
            model_cls: Union[Type[ModelV2], str, None] = None,
            lstm_cls: type = TFLSTMWrapper,
            framework: str = 'tf',
    ) -> None:
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )

        self._wrapped = _create_wrapped(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + '_lstm',
            model_cls,
            lstm_cls,
            framework,
        )
        self.view_requirements = {
            **self._wrapped.view_requirements,
            SampleBatch.OBS: self.view_requirements[SampleBatch.OBS],
        }

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        return TFMaskedActionsWrapper.forward(
            self,
            input_dict,
            state,
            seq_lens,
        )

    def forward_rnn(
            self,
            inputs: TensorType,
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        return self._wrapped.forward_rnn(inputs, state, seq_lens)

    def get_initial_state(self) -> List[TensorType]:
        return self._wrapped.get_initial_state()

    def value_function(self):
        return TFMaskedActionsWrapper.value_function(self)


class TorchMaskedActionsRecurrentWrapper(
        TorchRecurrentNetwork,
        TorchMaskedActionsWrapper,
):
    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            *,
            model_cls: Union[Type[ModelV2], str, None] = None,
            lstm_cls: type = TorchLSTMWrapper,
            framework: str = 'torch',
    ) -> None:
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )

        self._wrapped = _create_wrapped(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + '_lstm',
            model_cls,
            lstm_cls,
            framework,
        )
        self.view_requirements = {
            **self._wrapped.view_requirements,
            SampleBatch.OBS: self.view_requirements[SampleBatch.OBS],
        }

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        return TorchMaskedActionsWrapper.forward(
            self,
            input_dict,
            state,
            seq_lens,
        )

    def forward_rnn(
            self,
            inputs: TensorType,
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        return self._wrapped.forward_rnn(inputs, state, seq_lens)

    def get_initial_state(self) -> List[TensorType]:
        return self._wrapped.get_initial_state()

    def value_function(self):
        return TorchMaskedActionsWrapper.value_function(self)
