"""
A mock game that is created from environmental observations.

Provides "normalized" information of a single observing player.
"""

import functools
import itertools
from typing import List, Optional, Union

from gym.spaces import Space
import numpy as np
from ray.rllib.utils.typing import TensorType

from hearts_gym.envs import HeartsEnv
from hearts_gym.envs.card_deck import Card
from hearts_gym.envs.hearts_game import HeartsGame


class ObservedGame:
    """A mock game that is created from environmental observations.

    Due to being created from observations, an observed game only has
    access to information for a single observing player. Due to the same
    reason, the observed game does not know about the specific index of
    each player. Instead, it works with index offsets. This is important
    to keep in mind when accessing the specifically labeled variables
    like `offset_collected` or `offset_penalties`. Indexing these with
    0, for example, yields the value for the observing player (as an
    offset of 0 gives us the position of the observing player itself).
    """

    def __init__(self, original_obs_space: Space) -> None:
        """Construct a new observed game getting observations with the
        given original space (before preprocessing).

        Args:
            original_obs_space (Space): Observation space before
                preprocessing of the observations to process.
        """
        self.deck_size = np.prod(original_obs_space['cards'].shape).item()
        self.num_players = (
            original_obs_space['cards'].high.item(0)
            + 1
            - original_obs_space['cards'].low.item(0)
            - HeartsEnv.NUM_GENERAL_OBSERVATION_STATES
        ) // 2

        # Cards removed to reach the desired deck size.
        removed_cards = HeartsGame._removed_for_deck_size(self.deck_size)
        # Cards removed so all players have the same amount.
        player_removed_cards = HeartsGame._removed_for_num_players(
            self.deck_size, self.num_players)
        removed_cards.extend(player_removed_cards)

        self._cards_per_suit = [
            Card.NUM_RANKS - sum(
                1
                for _ in filter(lambda card: card.suit == suit, removed_cards)
            )
            for suit in range(Card.NUM_SUITS)
        ]
        self._accumulated_cards_per_suit = \
            list(itertools.accumulate(self._cards_per_suit))

    def _index_to_card(self, index: int) -> Card:
        """Return the card from a given index for the
        observation vector.

        Args:
            index (int): Index into the observation vector.

        Returns:
            Card: Card obtained from the observation vector index.
        """
        suit, num_accumulated = next(
            (suit, num_cards)
            for (suit, num_cards) in enumerate(
                    self._accumulated_cards_per_suit,
            )
            if index < num_cards
        )
        rank = index - (num_accumulated - self._accumulated_cards_per_suit[0])
        return Card(suit, rank)

    def _cards_with_state(self, obs: TensorType, state: int) -> List[Card]:
        """Return the cards with a given state in the observation vector.

        Args:
            obs (TensorType): Observation vector to get cards with a
                given state from.
            state (int): Obtain all cards with this state.

        Returns:
            List[Card]: Cards observed with the given state.
        """
        indices = np.argwhere(obs == state).ravel()
        return list(map(self._index_to_card, indices))

    def _cards_on_hand(self, obs: TensorType) -> List[Card]:
        """Return the cards on hand given by the observation vector.

        Args:
            obs (TensorType): Observation vector to convert to a hand
                of cards.

        Returns:
            List[Card]: Cards on hand of the player observing.
        """
        return self._cards_with_state(obs, HeartsEnv.STATE_ON_HAND)

    def _cards_unknown(self, obs: TensorType) -> List[Card]:
        """Return the cards that haven't been seen yet.

        Args:
            obs (TensorType): Observation vector to convert to
                unseen cards.

        Returns:
            List[Card]: Cards whose location is unknown.
        """
        return self._cards_with_state(obs, HeartsEnv.STATE_UNKNOWN)

    def _compute_leading_player_index_offset(
            self,
            obs: TensorType,
    ) -> None:
        """Compute and store the index offset of the leading player
        given by the observation vector.

        Additionally, store the indices in the observation vector of all
        cards on the table.

        Args:
            obs (TensorType): Observation vector to compute the leading
                player of.
        """
        all_on_table = functools.reduce(
            np.logical_or,
            (
                obs == HeartsEnv.on_table_state(i)
                for i in range(self.num_players)
            ),
        )
        all_on_table = np.argwhere(all_on_table).ravel()
        self._indices_on_table = all_on_table

        if len(all_on_table) == self.num_players:
            # Can't find leading player; is last observation.
            self.leading_player_index_offset = None
            return
        if len(all_on_table) == 0:
            # We are leading the trick.
            self.leading_player_index_offset = 0
            return

        states_on_table = obs[all_on_table]
        states_on_table.sort()
        for index_offset in range(self.num_players):
            if HeartsEnv.on_table_state(index_offset) == states_on_table[0]:
                break

        if index_offset != 0:
            self.leading_player_index_offset = index_offset
            return

        for (index_offset, state) in zip(
                range(self.num_players - 1, -1, -1),
                states_on_table[::-1],
        ):
            if HeartsEnv.on_table_state(index_offset) != state:
                break
        return (index_offset + 1) % self.num_players

    def _cards_on_table(
            self,
            obs: TensorType,
    ) -> List[Card]:
        """Return the cards on hand given by the observation vector.

        Args:
            obs (TensorType): Observation vector to convert to a hand
                of cards.

        Returns:
            List[Card]: Cards on the table in the order of placement.
        """
        states_on_table = obs[self._indices_on_table]
        if len(states_on_table) == 0:
            return []
        sort_indices = np.argsort(states_on_table)
        states_on_table = states_on_table[sort_indices]
        sorted_indices = self._indices_on_table[sort_indices]

        on_table = []
        leading_state = \
            HeartsEnv.on_table_state(self.leading_player_index_offset)
        start_i = next(
            i
            for (i, state) in enumerate(states_on_table)
            if state == leading_state
        )
        for i in range(start_i, start_i + len(states_on_table)):
            i = i % self.num_players

            index = sorted_indices[i]
            card = self._index_to_card(index)
            on_table.append(card)
        return on_table

    def _cards_collected(self, obs: TensorType) -> List[List[Card]]:
        """Return the cards collected by each player given by the
        observation vector. The result will be ordered by index offsets
        from the observing player.

        Args:
            obs (TensorType): Observation vector to convert to
                collected cards.

        Returns:
            List[List[Card]]: Cards collected, ordered by index offsets
                from the observing player.
        """
        return [
            self._cards_with_state(
                obs,
                HeartsEnv.collected_state(i, self.num_players),
            )
            for i in range(self.num_players)
        ]

    @staticmethod
    def get_penalty(card: Card) -> int:
        """Return the penalty score of the given card.

        Args:
            card (Card): Card to return the penalty score for.

        Returns:
            int: Penalty score of the card.
        """
        return HeartsGame.get_penalty(card)

    @staticmethod
    def has_penalty(card: Card) -> bool:
        """Return whether the given card has a penalty score greater
        than zero.

        Args:
            card (Card): Card to return whether it has a penalty
                score for.

        Returns:
            bool: Whether the card has a penalty score greater
                than zero.
        """
        return HeartsGame.has_penalty(card)

    def recreate_state(self, obs: TensorType) -> bool:
        """Build an internal game state matching the one given by the
        supplied observation vector. Return whether the observation was
        terminal, meaning the episode is over and no action needs to
        be computed.

        The state only has access to the information in the observation
        and thus provides only information the observing player has. The
        states are also normalized, meaning that some values are
        accessed through index offsets instead of player indices.

        Args:
            obs (TensorType): Observation to recreate a game state from.
                Expected to have a possible action mask stripped.

        Returns:
            bool: Whether the observation was a terminal one.
        """
        self._compute_leading_player_index_offset(obs)
        self.hand = self._cards_on_hand(obs)
        self.unknown_cards = self._cards_unknown(obs)
        self.table_cards = self._cards_on_table(obs)
        self.offset_collected = self._cards_collected(obs)
        """Cards collected by each player. Ordered by index offset from
        the observing player.
        """

        self.is_first_trick = (
            len(self.hand)
            == self.deck_size // self.num_players
        )
        self.leading_hearts_allowed = obs[self.deck_size]

        self.offset_penalties = [
            sum(map(self.get_penalty, cards))
            for cards in self.offset_collected
        ]
        """Total penalty scores of each player. Ordered by index offset
        from the observing player.
        """

        return self.leading_player_index_offset is None
