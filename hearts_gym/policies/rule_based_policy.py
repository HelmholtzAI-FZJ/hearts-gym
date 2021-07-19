"""
A policy following a rule-based strategy.

Rule-based in this context means fixed behavior according to pre-defined
rules (e.g. "always play the second legal card in hand").
"""

from typing import Dict, List, Optional

import numpy as np
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import (
    ModelWeights,
    TensorType,
    Tuple,
    Union,
)

from hearts_gym.envs import HeartsEnv
from hearts_gym.utils.typing import Action
from .deterministic_policy_impl import DeterministicPolicyImpl
from .observed_game import ObservedGame
from .rule_based_policy_impl import RuleBasedPolicyImpl


class RuleBasedPolicy(Policy):
    """A policy following a rule-based strategy.

    Rule-based in this context means fixed behavior according to pre-defined
    rules (e.g. "always play the second legal card in hand").
    """

    def __init__(self, *args, **kwargs) -> None:
        """Construct a rule-based policy.

        The following policy configuration options are used:
        - "policy_impl_cls": Rule-based policy implementation to use.
          Must subclass `DeterministicPolicyImpl`. Default
          is `RuleBasedPolicyImpl`.
        - "mask_actions": Whether action masking is enabled.
          Default is `True`.

        See also `Policy.__init__`.

        Args:
            *args: Arguments to pass to the superclass constructor.
            **kwargs: Keyword arguments to pass to the
                superclass constructor.
        """
        super().__init__(*args, **kwargs)

        mask_actions = self.config.get(
            'mask_actions', HeartsEnv.MASK_ACTIONS_DEFAULT)
        self._mask_actions = mask_actions
        policy_impl_cls = self.config.get(
            'policy_impl_cls', RuleBasedPolicyImpl)
        assert type(policy_impl_cls) == type, \
            '`policy_impl_cls` must not be an instance, but the class itself'
        assert issubclass(policy_impl_cls, DeterministicPolicyImpl), \
            '`policy_impl_cls` must subclass `DeterministicPolicyImpl`'

        # Set up helper variables
        original_space = self.observation_space.original_space

        if self._mask_actions:
            original_obs_space = original_space[HeartsEnv.OBS_KEY]

            action_mask_space = original_space[HeartsEnv.ACTION_MASK_KEY]
            self._action_mask_len = np.prod(action_mask_space.shape).item()
        else:
            original_obs_space = original_space
            self._action_mask_len = 0
        self._game = ObservedGame(original_obs_space)
        self._policy_impl = policy_impl_cls(self._game)

    def _split_obs_and_mask(
            self,
            obs_batch: TensorType,
    ) -> Tuple[TensorType, TensorType]:
        """Return two batches of parts of the observations; one contains the
        observations without action masks, one contains the action masks.

        Args:
            obs_batch (TensorType): Batch of observations with support
                for action masking.

        Returns:
            TensorType: Batch of observations without action masks.
            TensorType: Batch of action masks.
        """
        action_mask = obs_batch[:, :self._action_mask_len]
        sans_action_mask = obs_batch[:, self._action_mask_len:]
        return sans_action_mask, action_mask

    def _compute_action(
            self,
            obs: TensorType,
    ) -> Action:
        """Compute the action to take for the given observations.

        Args:
            obs (TensorType): Observations to compute the action for.

        Returns:
            Action: Which action to take. Assumed to be deterministic.
        """
        return self._policy_impl.compute_action(obs)

    @override(Policy)
    def compute_actions(
            self,
            obs_batch: Union[List[TensorType], TensorType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorType], TensorType] = None,
            prev_reward_batch: Union[List[TensorType], TensorType] = None,
            info_batch: Optional[Dict[str, list]] = None,
            **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        """Return which actions to take.

        See also `Policy.compute_actions`.

        Args:
            obs_batch (Union[List[TensorType], TensorType]): Single
                observation or batch of observations.
            state_batches (Optional[List[TensorType]]): List of RNN
                state input batches, if any.
            prev_action_batch (Union[List[TensorType], TensorType]):
                Ignored.
            prev_reward_batch (Union[List[TensorType], TensorType]):
                Ignored.
            info_batch (Optional[Dict[str, list]]): Batch of
                environment info dictionaries.
            **kwargs: Ignored.

        Returns:
            TensorType: Batch of output actions of shape
                `(BATCH_SIZE, ACTION_SHAPE)`.
            List[TensorType]: Empty list of RNN state output batches.
            Dict[str, TensorType]: Empty dictionary of extra
                feature batches.
        """
        if isinstance(obs_batch, list):
            obs_batch = np.array(obs_batch)
        if self._mask_actions:
            obs_batch, _ = self._split_obs_and_mask(obs_batch)

        actions = np.empty(len(obs_batch), dtype=self.action_space.dtype)
        for (i, obs) in enumerate(obs_batch):
            is_done = self._game.recreate_state(obs)

            # Even though we should never be asked to compute actions
            # for a terminal observation, we catch this case here.
            if is_done:
                # We have a terminal observation; no use to calculate
                # an action.
                actions[i] = 0
                continue

            action = self._compute_action(obs)
            actions[i] = action

        np.expand_dims(actions, 1)
        return actions, [], {}

    @override(Policy)
    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
            self,
            actions: Union[List[TensorType], TensorType],
            obs_batch: Union[List[TensorType], TensorType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Optional[Union[List[TensorType],
                                              TensorType]] = None,
            prev_reward_batch: Optional[Union[List[TensorType],
                                              TensorType]] = None,
    ) -> TensorType:
        """Return log-probabilities of 0 (so probabilities of 1) for the
        given actions.

        See also `Policy.compute_log_likelihoods`.

        Args:
            actions (Union[List[TensorType], TensorType]): Single
                action or batch of actions.
            obs_batch (Union[List[TensorType], TensorType]): Single
                observation or batch of observations.
            state_batches (Optional[List[TensorType]]): List of RNN
                state input batches, if any.
            prev_action_batch (Optional[Union[List[TensorType],
                                              TensorType]]):
                Ignored.
            prev_reward_batch (Optional[Union[List[TensorType],
                                              TensorType]]):
                Ignored.

        Returns:
            TensorType: log probabilities/likelihoods for the given
                actions depending on the observations.
        """
        return np.zeros(size=(len(obs_batch), 1))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass
