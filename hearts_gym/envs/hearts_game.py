"""
The game of Hearts (specifically, Black Lady).

Rules are mostly implemented as specified by the modern rules from
Morehead (2001).

See `HeartsGame` for a detailed description of the rules and differences
from the original.

References:
- https://en.wikipedia.org/wiki/Black_Lady
- https://en.wikipedia.org/wiki/Microsoft_Hearts
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .card_deck import Card, Deck, Seed


class HeartsGame:
    """A game of Hearts.

    Needs to be `reset` before being able to play.

    Rules are mostly implemented as specified by the modern rules from
    Morehead (2001).

    - The game starts with the player with the two of clubs playing it.
    - In the first trick, players are not allowed to play scored cards
      (any hearts or the queen of spades).
    - Players may only lead with a hearts card once a heart has been
      played due to not being able to follow suit or when they only have
      hearts on their hand.
    - Two or more players with equal score obtain the higher ranking.

    Each "hand" (terminology in the Wikipedia article for a single game
    from start to finish) is viewed independently; there is no state
    between games such as observing whether a player reached 100 points,
    then taking the player with the least amount of points as the
    winner.

    In addition and due to that, currently, there is no implementation
    for discarding cards at the start of each game.

    Instead of removing a certain set of cards from the deck when the
    deck size is not divisible by the number of players, we instead
    remove cards with higher ranks.

    This leads us to pick the starting player simply by finding the
    player with the lowest clubs card.
    """

    NUM_GENERAL_STATES = 1
    """Amount of states that do not refer to a specific player."""

    STATE_UNKNOWN = 0
    """This card has not been seen."""

    MAX_SCORE = 26
    """Maximum score possibly reachable."""
    RANK_QUEEN = Card.RANKS.index('Q')

    def __init__(
            self,
            *,
            num_players: int = 4,
            deck_size: int = 52,
            seed: Seed = None,
    ) -> None:
        """Construct a Hearts game for a fixed amount of players and cards.

        Args:
            num_players (int): Amount of players.
            deck_size (int): Amount of cards in the deck.
            seed (Seed): Random number generator seed.
        """
        assert num_players > 1, 'need at least two players'
        assert deck_size % Card.NUM_SUITS == 0, \
            'deck size must be divisible by ' + str(Card.NUM_SUITS)

        # We have maximally this many cards on hand.
        self.max_num_cards_on_hand = deck_size // num_players
        self.num_players = num_players

        # Each card can either be
        #    0: unknown
        #    1, ..., num_players:
        #       on the table (i.e. part of the current trick),
        #       played by player at "clockwise" index + 2
        #    1 + num_players, ..., 2 * num_players:
        #       in hand of player at "clockwise" index + 1
        #       + num_players
        #    1 + 2 * num_players, ..., 3 * num_players:
        #       collected by player at "clockwise" index + 1
        #       + 2 * num_players
        self.num_states = \
            self.NUM_GENERAL_STATES + 3 * self.num_players

        self.deck = Deck(deck_size, build_ordered=True, seed=seed)
        # We don't allow decks that don't have the same amount of cards
        # per suit, so this works for all suits.
        self._cards_per_suit = deck_size // Card.NUM_SUITS

        self.state = np.empty(deck_size, np.int8)
        """The state for each card."""
        self.hands: List[List[Card]] = []
        self.table_cards: List[Card] = []

        self._is_reset = False
        # self.reset()

        # Type hints
        self.scores: List[int]
        self.collected: List[List[Card]]

        self.max_score: int
        self.is_first_trick: bool
        self.leading_hearts_allowed: bool

        self.leading_player_index: Optional[int]
        """Index of the player that lead the current trick."""
        self.active_player_index: int
        """Index of the currently active player."""
        self.leading_suit: int
        self.prev_trick_score: Optional[int]

    def card_to_index(self, card: Card) -> int:
        """Return the index in the card state vector for the given card.

        Args:
            card (Card): Card to return the index for.

        Returns:
            int: Index into the card state vector.
        """
        return card.rank + card.suit * self._cards_per_suit

    def index_to_card(self, index: int) -> Card:
        """Return the card from a given index for the card state vector.

        Args:
            index (int): Index into the card state vector.

        Returns:
            Card: Card obtained from the card state vector index.
        """
        suit = index // self._cards_per_suit
        rank = index - suit * self._cards_per_suit
        return Card(suit, rank)

    def on_table_state(self, player_index: int) -> int:
        """Return the state for a card put on the table by the player with the
        given index.

        Args:
            player_index (int): Index of the player that put the card on
                the table.

        Returns:
            int: State for a card put on the table by a given player.
        """
        return self.NUM_GENERAL_STATES + player_index

    def in_hand_state(self, player_index: int) -> int:
        """Return the state for a card in hand of the player with the
        given index.

        Args:
            player_index (int): Index of the player that has the card
                in hand.

        Returns:
            int: State for a card held in hand by a given player.
        """
        return self.NUM_GENERAL_STATES + self.num_players + player_index

    def collected_state(self, player_index: int) -> int:
        """Return the state for a card collected (picked up by winning a trick)
        by the player with the given index.

        Args:
            player_index (int): Index of the player that has collected
                the card.

        Returns:
            int: State for a card collected by a given player.
        """
        return self.NUM_GENERAL_STATES + 2 * self.num_players + player_index

    def _update_state(
            self,
            cards: Union[List[Card], Card, np.ndarray],
            new_state: int,
    ) -> None:
        """Set the state for the given cards to the given state.

        Args:
            cards (Union[List[Card], Card, np.ndarray]): Card(s) to
                update the state for.
            new_state (int): New state to assign to all cards.
        """
        state_indices: Union[List[int], int]

        if isinstance(cards, list):
            state_indices = list(map(self.card_to_index, cards))
        elif isinstance(cards, np.ndarray):
            state_indices = np.fromiter(map(self.card_to_index, cards))
        else:
            state_indices = self.card_to_index(cards)
        self.state[state_indices] = new_state

    @staticmethod
    def _get_score(card: Card) -> int:
        """Return the score of the given card.

        Args:
            card (Card): Card to return the score for.

        Returns:
            int: Score of the card.
        """
        is_heart = card.suit == Card.SUIT_HEART
        if is_heart:
            return 1

        is_queen_of_spades = (card.suit == Card.SUIT_SPADE
                              and card.rank == HeartsGame.RANK_QUEEN)
        if is_queen_of_spades:
            return 13

        return 0

    @staticmethod
    def _has_score(card: Card) -> bool:
        """Return whether the given card has a score greater than zero.

        Args:
            card (Card): Card to return whether it has a score for.

        Returns:
            bool: Whether the card has a score greater than zero.
        """
        return HeartsGame._get_score(card) > 0

    def _first_index_without_score(self, hand: List[Card]) -> Optional[int]:
        """Return the index of the first card that has a score or `None`.

        Args:
            hand (List[Card]): Cards to index into.

        Returns:
            Optional[int]: Index of the first card with a score. `None`
                if no card has a score.
        """
        for (i, card) in enumerate(hand):
            if self._has_score(card):
                continue
            return i
        return None

    def _first_index_without_score_with_suit(
            self,
            hand: List[Card],
            suit: int,
    ) -> Optional[int]:
        """Return the index of the first card without a score that has the
        given suit or `None`.

        Args:
            hand (List[Card]): Cards to index into.
            suit (int): Numerical value of a suit.

        Returns:
            Optional[int]: Index of the first card without a score with
                the given suit. `None` if all cards have a score or no
                card has the given suit.
        """
        for (i, card) in enumerate(hand):
            if card.suit != suit or self._has_score(card):
                continue
            return i
        return None

    def _first_index_without_hearts(self, hand: List[Card]) -> Optional[int]:
        """Return the index of the first non-hearts card or `None`.

        Args:
            hand (List[Card]): Cards to index into.

        Returns:
            Optional[int]: Index of the first card with a suit other
                than hearts. `None` if all cards are hearts cards.
        """
        for (i, card) in enumerate(hand):
            if card.suit == Card.SUIT_HEART:
                continue
            return i
        return None

    def _first_index_with_suit(
            self,
            hand: List[Card],
            suit: int,
    ) -> Optional[int]:
        """Return the index of the first card with the given suit or `None`.

        Args:
            hand (List[Card]): Cards to index into.
            suit (int): Numerical value of a suit.

        Returns:
            Optional[int]: Index of the first card with the given suit.
                `None` if no card has the suit.
        """
        for (i, card) in enumerate(hand):
            if card.suit != suit:
                continue
            return i
        return None

    @staticmethod
    def _extract_action(
            action_card_tuple: Union[Tuple[int, Card], Tuple[int]],
    ) -> int:
        """Return the action from a tuple.

        Args:
            action_card_tuple (Union[Tuple[int, Card], Tuple[int]]):
                Tuple to extract the action from.

        Returns:
            int: Index in hand for which card to play.
        """
        return action_card_tuple[0]

    def get_legal_actions(self, player_index: int) -> List[int]:
        """Return all legal actions for the player with the given index.

        Args:
            player_index (int): Player index to query legal actions for.

        Returns:
            List[int]: Indices in hand for which cards are legal to play.
        """
        hand = self.hands[player_index]

        actions: Union[List[Tuple[int, Card]], List[Tuple[int]]]
        if (
                # No hearts or queen of spades in first trick.
                self.is_first_trick
        ):
            if player_index == self.leading_player_index:
                actions = list(filter(
                    lambda i_card: not self._has_score(i_card[1]),
                    enumerate(hand),
                ))
            else:
                actions = list(filter(
                    lambda i_card: (i_card[1].suit == self.leading_suit
                                    and not self._has_score(i_card[1])),
                    enumerate(hand),
                ))

        elif (
                # Can't start with hearts.
                not self.leading_hearts_allowed
                and player_index == self.leading_player_index
        ):
            actions = list(filter(
                lambda i_card: i_card[1].suit != Card.SUIT_HEART,
                enumerate(hand),
            ))

        elif (
                # Must follow suit.
                player_index != self.leading_player_index
        ):
            actions = list(filter(
                lambda i_card: i_card[1].suit == self.leading_suit,
                enumerate(hand),
            ))

        else:
            actions = list(map(lambda x: (x,), range(len(hand))))

        actions: List[int] = list(map(self._extract_action, actions))
        if len(actions) == 0:
            actions = list(range(len(hand)))
        return actions

    def _play_card(self, card_index: int) -> Card:
        """Play and return the card at the given index in hand of the
        active player.

        Also update the game state accordingly; however, do not
        distribute the trick.

        Low-level version of `play_card` without checks.

        Args:
            card_index (int): Index in hand of the card to play.

        Returns:
            Card: The card that was played.
        """
        hand = self.hands[self.active_player_index]
        card_to_play = hand.pop(card_index)
        self.table_cards.append(card_to_play)
        self._update_state(card_to_play,
                           self.on_table_state(self.active_player_index))

        if self.active_player_index == self.leading_player_index:
            self.leading_suit = card_to_play.suit

        self.active_player_index = \
            (self.active_player_index + 1) % self.num_players
        return card_to_play

    def _distribute_trick(self) -> Tuple[int, int]:
        """Distribute the cards on the table according to who won the trick.
        Return the index of the winner of the trick and the score
        obtained by them.

        Also update the game state accordingly.

        Returns:
            int: Index of the player that won the trick.
            int: Score of the cards obtained by the player that won
                the trick.
        """
        assert all(map(
            lambda hand: len(hand) == len(self.hands[0]),
            self.hands,
        )), \
            'all players must have same amount of cards for trick distribution'
        assert len(self.table_cards) == self.num_players, \
            'trick must be full for distribution'

        trick_winner_index = self._get_trick_winner()
        trick_score = self._score_cards(self.table_cards)
        self.scores[trick_winner_index] += trick_score

        self.collected[trick_winner_index].extend(self.table_cards)
        self._update_state(self.table_cards,
                           self.collected_state(trick_winner_index))
        self.table_cards.clear()

        self.leading_player_index = trick_winner_index
        self.active_player_index = self.leading_player_index
        self.is_first_trick = False
        self.prev_trick_score = trick_score
        return trick_winner_index, trick_score

    def play_card(
            self,
            action: int,
    ) -> Tuple[Card, bool, Optional[int], Optional[int]]:
        """Play the card at the given index in hand of the active player.
        Return the played card and additional information.

        If the action was actually illegal, play the first legal card
        in hand.

        Also update the game state accordingly.

        Args:
            action (int): Index in hand of the card to play.

        Returns:
            Card: The card that was played.
            bool: Whether the action was illegal and a card different
                from the one specified by the action was played.
            Optional[int]: Index of the player that won the trick or
                `None` if it is still ongoing.
            Optional[int]: Score of the cards obtained by the player
                that won the trick or `None` if it is still ongoing.
        """
        assert self._is_reset, \
            'please call `reset` before interacting with the game.'
        assert len(self.table_cards) < self.num_players, \
            'cannot play a card when trick is already full'
        hand = self.hands[self.active_player_index]

        if action >= len(hand):
            # Just play the last card.
            adjusted_action = -1
        else:
            adjusted_action = action
        card_to_play = hand[adjusted_action]

        if (
                # No hearts or queen of spades in first trick.
                self.is_first_trick
                and self._has_score(card_to_play)
        ):
            if self.active_player_index == self.leading_player_index:
                index_without_score = self._first_index_without_score(hand)
            else:
                index_without_score = \
                    self._first_index_without_score_with_suit(
                        hand, self.leading_suit)

            if index_without_score is not None:
                adjusted_action = index_without_score
            elif card_to_play.suit == Card.SUIT_HEART:
                self.leading_hearts_allowed = True

        elif (
                # Can't start with hearts.
                not self.leading_hearts_allowed
                and self.active_player_index == self.leading_player_index
                and card_to_play.suit == Card.SUIT_HEART
        ):
            index_without_heart = self._first_index_without_hearts(hand)

            if index_without_heart is not None:
                adjusted_action = index_without_heart
            else:
                self.leading_hearts_allowed = True

        elif (
                # Must follow suit.
                self.active_player_index != self.leading_player_index
                and card_to_play.suit != self.leading_suit
        ):
            index_with_suit = self._first_index_with_suit(
                hand, self.leading_suit)

            if index_with_suit is not None:
                adjusted_action = index_with_suit
            elif card_to_play.suit == Card.SUIT_HEART:
                self.leading_hearts_allowed = True

        card_to_play = self._play_card(adjusted_action)
        was_illegal = adjusted_action != action

        trick_winner_index: Optional[int]
        trick_score: Optional[int]
        if len(self.table_cards) == self.num_players:
            trick_winner_index, trick_score = self._distribute_trick()
        else:
            trick_winner_index, trick_score = (None, None)
        return card_to_play, was_illegal, trick_winner_index, trick_score

    def _get_trick_winner(self) -> int:
        """Return the index of the player that won the trick.

        Returns:
            int: Index of the player that won the trick.
        """
        max_rank = -1
        max_rank_index = None
        for (table_index, card) in enumerate(self.table_cards):
            if card.suit != self.leading_suit or card.rank <= max_rank:
                continue

            max_rank = card.rank
            max_rank_index = table_index
        assert max_rank_index is not None
        assert self.leading_player_index is not None
        return (max_rank_index + self.leading_player_index) % self.num_players

    @staticmethod
    def _score_cards(cards: List[Card]) -> int:
        """Return the accumulated score of the given cards.

        Args:
            cards (List[Card]): Cards to score.

        Returns:
            int: Accumulated score of the cards.
        """
        total_score = 0
        for card in cards:
            total_score += HeartsGame._get_score(card)
        return total_score

    def compute_final_scores(self) -> List[int]:
        """Compute and return the final scores of the game, taking into account
        shooting the moon.

        Returns:
            List[int]: The final scores of the game for each player,
                sorted by player indices.
        """
        # print('scores:', self.scores)
        for (player_index, score) in enumerate(self.scores):
            if score != self.max_score:
                continue

            # Someone has shot the moon.
            self.scores = [self.max_score] * self.num_players
            self.scores[player_index] = 0
            break

        # Return a copy to avoid surprises when stored scores change.
        return self.scores.copy()

    def compute_rankings(self) -> List[int]:
        """Compute and return the final rankings of the game.

        Players with the same score obtain the higher ranking.

        Requires the final scores to be computed (see
        `compute_final_scores`).

        Returns:
            List[int]: The final rankings of the game for each player,
                sorted by player indices.
        """
        rankings: List[Optional[int]] = [None] * self.num_players
        sorted_scores = sorted(
            zip(self.scores, range(self.num_players)),
            reverse=True,
        )

        same_score = []
        curr_ranking = self.num_players

        for (i, (score, player_index)) in enumerate(sorted_scores):
            if i + 1 < len(sorted_scores) and sorted_scores[i + 1][0] == score:
                same_score.append(player_index)
                curr_ranking -= 1
                continue
            elif len(same_score) > 0:
                for same_score_player_index in same_score:
                    rankings[same_score_player_index] = curr_ranking
                same_score.clear()

            rankings[player_index] = curr_ranking
            curr_ranking -= 1

        assert curr_ranking == 0, 'rankings were miscalculated'
        # print('rankings:', rankings)
        assert all(ranking is not None for ranking in rankings)
        rankings: List[int] = rankings  # type:ignore[assignment]
        return rankings

    def is_done(self) -> bool:
        """Return whether the game is over.

        Returns:
            bool: Whether the game is over.
        """
        return (
            all(map(lambda hand: len(hand) == 0, self.hands))
            and len(self.table_cards) == 0
        )

    def full_trick(
            self,
            actions: List[int],
    ) -> Tuple[np.ndarray, int, bool, Dict[str, Any]]:
        """Play a full trick according to the given actions. Return additional
        information about the trick.

        For each action, if it is illegal, play the first legal card.

        As the game state changes between actions, this function is only
        useful for debugging purposes.

        Args:
            actions (List[int]): Indices for the cards to play for each
                player, sorted by player indices. Action may be skipped
                if the player's card was already played (only relevant
                at the beginning of the game).

        Returns:
            np.ndarray: The card state vector after the trick.
            int: Index of the winner of the trick.
            bool: Whether the game is over now.
            Dict[str, Any]: Additional information such as which cards
                were played or which actions were illegal.
        """
        if not self.is_first_trick:
            assert all(map(
                lambda hand: len(hand) == len(self.hands[0]),
                self.hands,
            )), \
                'all players must have same amount of cards for full trick'

        leading_player_index = self.leading_player_index
        assert leading_player_index is not None

        cards: List[Optional[Card]] = [None] * self.num_players
        was_illegals: List[Optional[bool]] = [None] * self.num_players
        for player_index in range(
                leading_player_index,
                leading_player_index + self.num_players,
        ):
            if self.is_first_trick and player_index == leading_player_index:
                # We already force-play the first card.
                was_illegals[player_index] = False
                continue
            player_index = player_index % self.num_players
            action = actions[player_index]
            card, was_illegal, trick_winner_index, trick_score = \
                self.play_card(action)
            cards[player_index] = card
            was_illegals[player_index] = was_illegal

        assert trick_winner_index is not None and trick_score is not None, \
            'did not properly finish full trick'

        is_done = self.is_done()
        info: Dict[str, Any] = {
            'leading_player_index': leading_player_index,
            'cards': cards,
            'was_illegals': was_illegals,
            'trick_winner_index': trick_winner_index,
            'trick_score': trick_score,
        }

        if is_done:
            info['final_scores'] = self.compute_final_scores()
            info['final_rankings'] = self.compute_rankings()

        return self.state.copy(), trick_winner_index, is_done, info

    def reset(self) -> None:
        """Reset the game state.

        Due to the nature of the game, this also force-plays the card
        designating the starting player.

        Check the description of the `HeartsGame` class to find out how
        the starting player is chosen.
        """
        self.deck.reset()
        self.state[:] = self.STATE_UNKNOWN
        self.scores = [0] * self.num_players
        self.is_first_trick = True
        self.leading_hearts_allowed = False
        self.prev_trick_score = None
        # FIXME implement proper discarding (remove as many as needed from 2C, 2D, 3C, 2S)

        self.collected = [[] for _ in range(self.num_players)]
        self.hands.clear()
        for player_index in range(self.num_players):
            hand = self.deck.take(self.max_num_cards_on_hand)
            # Sort so we get deterministic behavior for illegal actions.
            hand.sort()
            self.hands.append(hand)
            self._update_state(hand, self.in_hand_state(player_index))
        assert all(map(
            lambda hand: len(hand) == len(self.hands[0]),
            self.hands,
        )), \
            'all players must have same amount of cards at start of game'
        self.table_cards.clear()

        card_index: Optional[int] = None
        self.leading_player_index = None
        for rank in range(Card.NUM_RANKS):
            # TODO Maybe use binary search here.
            card = Card(Card.SUIT_CLUB, rank)
            for (player_index, hand) in enumerate(self.hands):
                if card not in hand:
                    continue

                card_index = hand.index(card)
                self.leading_player_index = player_index
                break

            if self.leading_player_index is not None:
                break

        # We went through all clubs and _still_ haven't found a
        # starting player.
        assert self.leading_player_index is not None, (
            'could not find a club-suited card in player hands; '
            'please choose a more even player/deck distribution'
        )

        self.remaining_cards = self.deck.take(len(self.deck))
        assert (
            self.deck.size % self.num_players != 0
            or len(self.remaining_cards) == 0
        ), 'something went wrong during initialization'
        assert len(self.deck) == 0, \
            'deck must be empty at start of game'

        self.max_score = \
            self.MAX_SCORE - sum(map(self._get_score, self.remaining_cards))

        self.active_player_index = self.leading_player_index
        assert card_index is not None
        self._play_card(card_index)

        self._is_reset = True

    def __str__(self) -> str:
        output = []
        for (player_index, (hand, score, collected)) in enumerate(zip(
                self.hands,
                self.scores,
                self.collected,
        )):
            output.append('Player ')
            output.append(str(player_index + 1))
            if player_index == self.active_player_index:
                output.append(' (active)')
            output.append(': ')
            output.append(', '.join(str(card) for card in hand))
            output.append('\n   Collected (score ')

            output.append(str(score))
            output.append('): ')
            output.append(', '.join(str(card) for card in collected))
            output.append('\n')

        output.append('\nTable: ')
        output.append(', '.join(str(card) for card in self.table_cards))
        output.append('\nLeading hearts allowed: ')
        output.append('yes' if self.leading_hearts_allowed else 'no')

        return ''.join(output)
