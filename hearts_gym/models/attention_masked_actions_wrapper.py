"""
Wrapper classes to support action masking for arbitrary
attention-based recurrent models.

Action masking gives prohibited actions a probability close to zero.
"""

from typing import Type, Union

from gym.spaces import Space
from ray.rllib.models import ModelV2
from ray.rllib.models.tf.attention_net import (
    AttentionWrapper as TFAttentionWrapper,
)
from ray.rllib.models.torch.attention_net import (
    AttentionWrapper as TorchAttentionWrapper,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import List, TensorType, ModelConfigDict

from .masked_actions_wrapper import (
    TFMaskedActionsWrapper,
    TorchMaskedActionsWrapper,
)
from .recurrent_masked_actions_wrapper import _create_wrapped

_, tf, _ = try_import_tf()
th, nn = try_import_torch()


# class TFMaskedActionsAttentionWrapper(TFModelV2):
class TFMaskedActionsAttentionWrapper(TFMaskedActionsWrapper):
    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            *,
            model_cls: Union[Type[ModelV2], str, None] = None,
            attn_cls: type = TFAttentionWrapper,
            framework: str = 'tf',
    ) -> None:
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            framework=framework,
        )

        self._wrapped = _create_wrapped(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + '_attn',
            model_cls,
            attn_cls,
            framework,
        )
        self.view_requirements = {
            **self._wrapped.view_requirements,
            SampleBatch.OBS: self.view_requirements[SampleBatch.OBS],
        }

    def get_initial_state(self) -> List[TensorType]:
        return self._wrapped.get_initial_state()
        # return self._wrapped.gtrxl.get_initial_state()
        # return [np.zeros((self._wrapped.gtrxl.max_seq_len,
        #                   self._wrapped.gtrxl.obs_dim), np.float32)]


class TorchMaskedActionsAttentionWrapper(TorchMaskedActionsWrapper):
    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            *,
            model_cls: Union[Type[ModelV2], str, None] = None,
            attn_cls: type = TorchAttentionWrapper,
            framework: str = 'torch',
    ) -> None:
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            framework=framework,
        )

        self._wrapped = _create_wrapped(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + '_attn',
            model_cls,
            attn_cls,
            framework,
        )
        self.view_requirements = {
            **self._wrapped.view_requirements,
            SampleBatch.OBS: self.view_requirements[SampleBatch.OBS],
        }

    # def forward(
    #         self,
    #         input_dict: Dict[str, TensorType],
    #         state: List[TensorType],
    #         seq_lens: TensorType,
    # ) -> Tuple[TensorType, List[TensorType]]:
    #     return TorchMaskedActionsWrapper.forward(
    #         self,
    #         input_dict,
    #         state,
    #         seq_lens,
    #     )

    def get_initial_state(self) -> List[TensorType]:
        return self._wrapped.get_initial_state()

    # def value_function(self):
    #     return TorchMaskedActionsWrapper.value_function(self)
