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
from hearts_gym.envs.card_deck import Card


class RuleBasedPolicy(Policy):
    """A policy following a rule-based strategy.

    Rule-based in this context means fixed behavior according to pre-defined
    rules (e.g. "always play the second legal card in hand").
    """

    def __init__(self, *args, **kwargs) -> None:
        """Construct a rule-based policy.

        The following policy configuration options are used:
        - "mask_actions": Whether action masking is enabled.
          Default is `True`.

        See also `Policy.__init__`.

        Args:
            *args: Arguments to pass to the superclass constructor.
            **kwargs: Keyword arguments to pass to the
                superclass constructor.
        """
        super().__init__(*args, **kwargs)

        mask_actions = self.config.get('mask_actions', True)
        self._mask_actions = mask_actions

        self._setup_variables()

    def _setup_variables(self) -> None:
        """Set up helper variables."""
        original_space = self.observation_space.original_space

        if self._mask_actions:
            original_obs_space = original_space[HeartsEnv.OBS_KEY]

            action_mask_space = original_space[HeartsEnv.ACTION_MASK_KEY]
            self._action_mask_len = np.prod(action_mask_space.shape)
        else:
            original_obs_space = original_space

        self._num_cards = np.prod(original_obs_space['cards'].shape)
        # FIXME not so simple anymore
        self._cards_per_suit = self._num_cards // Card.NUM_SUITS

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

    def _index_to_card(self, index: int) -> Card:
        """Return the card from a given index for the
        observation vector.

        Args:
            index (int): Index into the observation vector.

        Returns:
            Card: Card obtained from the observation vector index.
        """
        suit = index // self._cards_per_suit
        rank = index - suit * self._cards_per_suit
        return Card(suit, rank)

    def _suit_from_index(self, index: int) -> int:
        """Return the suit of the card at the given index in the
        observation vector.

        Args:
            index (int): Index into the observation vector.

        Returns:
            int: Suit of the card obtained from the observation
                vector index.
        """
        return index // self._cards_per_suit

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
            obs_batch, action_masks = self._split_obs_and_mask(obs_batch)
            # You could just disregard the `action_masks` and not treat
            # this as a special case. Although you do need to keep the
            # splitting for consistency in the observations!
            raise NotImplementedError('please implement the rule-based agent')
        else:
            raise NotImplementedError('please implement the rule-based agent')

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
