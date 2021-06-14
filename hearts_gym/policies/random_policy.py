"""
A policy executing actions at random.
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


class RandomPolicy(Policy):
    """A policy executing actions at random.

    If action masking is not enabled, actions are sampled uniformly
    based on the hand size. This means illegal actions may be chosen. If
    action masking is enabled, only legal actions are chosen.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Construct a randomly acting policy.

        The following policy configuration options are used:
        - "seed": Random number generator seed.
        - "mask_actions": Whether action masking is enabled.

        See also `Policy.__init__`.

        Args:
            *args: Arguments to pass to the superclass constructor.
            **kwargs: Keyword arguments to pass to the
                superclass constructor.
        """
        super().__init__(*args, **kwargs)
        seed = self.config.get('seed', None)
        self._rng = np.random.default_rng(seed)

        mask_actions = self.config.get('mask_actions', False)
        self._mask_actions = mask_actions

        if mask_actions:
            original_space = self.observation_space.original_space
            action_mask_space = original_space[HeartsEnv.ACTION_MASK_KEY]
            self._len_action_mask = np.prod(action_mask_space.shape)

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
        action_mask = obs_batch[:, :self._len_action_mask]
        sans_action_mask = obs_batch[:, self._len_action_mask:]
        return sans_action_mask, action_mask

    @override(Policy)
    def compute_actions(
            self,
            obs_batch: Union[List[TensorType], TensorType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorType], TensorType] = None,
            prev_reward_batch: Union[List[TensorType], TensorType] = None,
            **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        """Return a randomly sampled action.

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
            _, action_masks = self._split_obs_and_mask(obs_batch)
            actions = []
            # possible_actions = np.arange(action_masks.shape[-1])
            # Currently not possible to sample from masked arrays, so we
            # can't vectorize this.
            for action_mask in action_masks:
                # legal = possible_actions[action_mask == 1]
                # Could use np.argwhere(...).ravel()
                legal = np.argwhere(action_mask == 1).ravel()
                choice = self._rng.choice(legal)
                actions.append(choice)
            actions = np.array(actions)
        else:
            # TODO Could calculate perfectly so we never take illegal actions.
            on_hand_indices = obs_batch == HeartsEnv.STATE_ON_HAND
            hand_lens = np.count_nonzero(on_hand_indices, axis=1)
            assert isinstance(hand_lens, np.ndarray)
            np.clip(hand_lens, 1, None, out=hand_lens)
            actions = self._rng.integers(low=0, high=hand_lens)

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
        """Return log uniform probabilities for the given actions.

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
        if isinstance(actions, list):
            actions = np.array(actions)
        if isinstance(obs_batch, list):
            obs_batch = np.array(obs_batch)

        if self._mask_actions:
            _, action_masks = self._split_obs_and_mask(obs_batch)
            num_actions = np.count_nonzero(action_masks)
            probs = 1 / num_actions
        else:
            on_hand_indices = obs_batch == HeartsEnv.STATE_ON_HAND
            hand_lens = np.count_nonzero(on_hand_indices, axis=1)
            assert isinstance(hand_lens, np.ndarray)
            # Prevent division by 0.
            np.clip(hand_lens, 1, None, out=hand_lens)
            probs = 1 / hand_lens

        probs = np.log(probs)
        np.expand_dims(probs, 1)
        return probs

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass
