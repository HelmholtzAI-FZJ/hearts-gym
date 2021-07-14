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

import bisect
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from hearts_gym.utils.typing import Seed
from .card_deck import Card, Deck


class HeartsGame:
    """A game of Hearts.

    Needs to be `reset` before being able to play.

    Rules are mostly implemented as specified by the modern rules from
    Morehead (2001) (ISBN: 9780451204844).

    - The game starts with the player with the two of clubs playing it.
    - In the first trick, players are not allowed to play penalty-scored
      cards (any hearts or the queen of spades).
    - Players may only lead with a hearts card once a heart has been
      played due to not being able to follow suit or when they only have
      hearts on their hand.
    - Two or more players with equal penalty score obtain the
      higher ranking.

    Each "hand" (terminology in the Wikipedia article for a single game
    from start to finish) is viewed independently; there is no state
    between games such as observing whether a player reached 100 points,
    then taking the player with the least amount of points as the
    winner.

    In addition and due to that, currently, there is no implementation
    for discarding ("passing") cards at the start of each game.

    When the deck size is 52 and not divisible by the number of players,
    remove as many of the following cards as necessary (and in that
    order):
    - two of clubs
    - two of diamonds
    - three of clubs
    - two of spades
    If the deck size is any other value, instead remove cards with the
    lowest ranks from the suits in the order given by
    `self.REMOVE_SUIT_ORDER` until the desired size is reached.

    We pick the starting player simply by finding the player with the
    lowest clubs card. This rule is not sufficiently specified in
    Morehead (2001).
    """

    NUM_GENERAL_STATES = 1
    """Amount of states that do not refer to a specific player."""

    STATE_UNKNOWN = 0
    """This card has not been seen."""

    MAX_PENALTY = 26
    """Maximum penalty score possibly reachable."""
    RANK_QUEEN = Card.RANKS.index('Q')

    # These are the numbers the rules state; the program is able to
    # handle 2 to 8 players without modification.
    MIN_NUM_PLAYERS = 3
    MAX_NUM_PLAYERS = 6

    REMOVE_CARDS = [
        Card(Card.SUIT_CLUB, Card.RANKS.index('2')),
        Card(Card.SUIT_DIAMOND, Card.RANKS.index('2')),
        Card(Card.SUIT_CLUB, Card.RANKS.index('3')),
        Card(Card.SUIT_SPADE, Card.RANKS.index('2')),
    ]
    """Cards to remove from a standard deck according to the
    implemented rules.
    """
    REMOVE_SUIT_ORDER = [
        Card.SUIT_CLUB,
        Card.SUIT_DIAMOND,
        Card.SUIT_SPADE,
        Card.SUIT_HEART,
    ]
    """In which order of suits to remove lowest-ranked cards if not
    using a standard 52-card deck.
    """

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
        assert self.MIN_NUM_PLAYERS <= num_players <= self.MAX_NUM_PLAYERS, (
            f'number of players must be between {self.MIN_NUM_PLAYERS} and '
            f'{self.MAX_NUM_PLAYERS} inclusively'
        )

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

        self.deck = Deck(Deck.MAX_SIZE, build_ordered=True, seed=seed)
        deck_size, removed_cards = self._remove_cards(deck_size, num_players)

        self.max_penalty = \
            self.MAX_PENALTY - sum(map(self.get_penalty, removed_cards))

        self.state = np.empty(deck_size, np.int8)
        """The state for each card."""
        self.hands: List[List[Card]] = []
        self.table_cards: List[Card] = []
        self._is_reset = False
        # self.reset()

        # Type hints
        self.penalties: List[int]
        self.collected: List[List[Card]]

        self.is_first_trick: bool
        self.leading_hearts_allowed: bool

        self.leading_player_index: Optional[int]
        """Index of the player that lead, i.e. started, the current trick."""
        self.active_player_index: int
        """Index of the currently active player."""
        self.leading_suit: int

        self.prev_hands: List[List[Card]]
        """The cards in hand in the previous trick for each player.

        Entries are empty lists if no trick has been played yet.
        """
        self.prev_played_cards: List[Optional[Card]]
        """The last card actively played by each player.

        Does not include the card that is force-played at the beginning
        of the game.

        Entries are `None` if no action has been taken yet, so also
        `None` after the initial card was force-played.
        """
        self.prev_table_cards: List[Card]
        """The last cards on the table.

        Empty if no trick has been distributed yet.
        """

        self.prev_collected: List[List[Card]]
        """All cards collected before the previous trick."""

        self.prev_was_illegals: List[Optional[bool]]
        """For each player, whether their previous action was illegal.

        Entries are `None` if no action has been taken yet.
        """
        self.prev_states: List[Optional[np.ndarray]]
        """State before the last action for each player.

        Entries are `None` if no action has been taken yet.
        """

        self.prev_was_first_trick: Optional[bool]
        """Whether the previous trick was the first one.

        `None` if no trick has been distributed yet.
        """
        self.prev_leading_hearts_allowed: List[Optional[bool]]
        """For each player, whether leading hearts were allowed for the
        previous action.

        Entries are `None` if no action has been taken yet.
        """

        self.prev_leading_suit: Optional[int]
        self.prev_leading_player_index: Optional[int]
        self.prev_trick_winner_index: Optional[int]
        self.prev_trick_penalty: Optional[int]

    def card_to_index(self, card: Card) -> int:
        """Return the index in the card state vector for the given card.

        Args:
            card (Card): Card to return the index for.

        Returns:
            int: Index into the card state vector.
        """
        return (
            card.rank
            + self._accumulated_cards_per_suit[card.suit]
            - self._accumulated_cards_per_suit[0]
        )

    def index_to_card(self, index: int) -> Card:
        """Return the card from a given index for the card state vector.

        Args:
            index (int): Index into the card state vector.

        Returns:
            Card: Card obtained from the card state vector index.
        """
        suit, num_accumulated = next(
            (index, num_cards)
            for num_cards in self._accumulated_cards_per_suit
            if index < num_cards
        )
        rank = index - (num_accumulated - self._accumulated_cards_per_suit[0])
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
    def get_penalty(card: Card) -> int:
        """Return the penalty score of the given card.

        Args:
            card (Card): Card to return the penalty score for.

        Returns:
            int: Penalty score of the card.
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
        return HeartsGame.get_penalty(card) > 0

    def _first_index_without_penalty(self, hand: List[Card]) -> Optional[int]:
        """Return the index of the first card that has a penalty
        or `None`.

        Args:
            hand (List[Card]): Cards to index into.

        Returns:
            Optional[int]: Index of the first card with a penalty score.
                `None` if no card has a penalty.
        """
        for (i, card) in enumerate(hand):
            if self.has_penalty(card):
                continue
            return i
        return None

    def _first_index_without_penalty_with_suit(
            self,
            hand: List[Card],
            suit: int,
    ) -> Optional[int]:
        """Return the index of the first card without a penalty that has
        the given suit or `None`.

        Args:
            hand (List[Card]): Cards to index into.
            suit (int): Numerical value of a suit.

        Returns:
            Optional[int]: Index of the first card without a penalty
                score with the given suit. `None` if all cards have a
                penalty or no card has the given suit.
        """
        for (i, card) in enumerate(hand):
            if card.suit != suit or self.has_penalty(card):
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

    def _remove_cards(
            self,
            deck_size: int,
            num_players: int,
    ) -> Tuple[int, List[Card]]:
        """Permanently remove cards from the game's deck so they match
        the desired size and number of players, meaning all players get
        the same amount of cards. Return the size of the remaining deck
        and the removed cards.

        When the deck size is 52 and not divisible by the number of
        players, remove as many of the following cards as necessary (and
        in that order):
        - two of clubs
        - two of diamonds
        - three of clubs
        - two of spades
        If the deck size is any other value, instead remove cards with
        the lowest ranks from the suits in the order given by
        `self.REMOVE_SUIT_ORDER` until the desired size is reached.

        Args:
            deck_size (int): Desired size of the deck.
            num_players (int): Number of players in the game.

        Returns:
            int: Amount of cards remaining in the deck.
            List[Card]]: Cards that were removed.
        """
        num_removed_cards = deck_size % num_players
        if deck_size != 52:
            num_larger_suits = deck_size % Card.NUM_SUITS
            min_num_cards_per_suit = deck_size // Card.NUM_SUITS
            smallest_larger_suit = Card.NUM_SUITS - num_larger_suits

            # Cards we remove to reach the desired deck size.
            removed_cards = [
                Card(suit, rank)
                for suit in self.REMOVE_SUIT_ORDER
                for rank in range(
                        (
                            Card.NUM_RANKS
                            - (
                                min_num_cards_per_suit
                                + (suit >= smallest_larger_suit)
                            )
                        ),
                )
            ]

            # Cards we remove so all players have the same amount.
            for i in range(num_removed_cards):
                suit_index = (smallest_larger_suit + i) % Card.NUM_SUITS
                suit = self.REMOVE_SUIT_ORDER[suit_index]
                card = Card(
                    suit,
                    (
                        Card.NUM_RANKS
                        - (
                            min_num_cards_per_suit
                            + (suit >= smallest_larger_suit)
                        )
                        + (smallest_larger_suit + i) // Card.NUM_SUITS
                    ),
                )
                removed_cards.append(card)
        else:
            removed_cards = self.REMOVE_CARDS[:num_removed_cards]

        self.deck.remove(removed_cards)
        self._cards_per_suit = [
            Card.NUM_RANKS - sum(
                1
                for _ in filter(lambda card: card.suit == suit, removed_cards)
            )
            for suit in range(Card.NUM_SUITS)
        ]
        self._accumulated_cards_per_suit = \
            list(itertools.accumulate(self._cards_per_suit))
        return deck_size - num_removed_cards, removed_cards

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
                    lambda i_card: not self.has_penalty(i_card[1]),
                    enumerate(hand),
                ))
            else:
                actions = list(filter(
                    lambda i_card: (i_card[1].suit == self.leading_suit
                                    and not self.has_penalty(i_card[1])),
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
        self.prev_hands[self.active_player_index] = hand.copy()
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
        """Distribute the cards on the table according to who won
        the trick.
        Return the index of the winner of the trick and the penalty
        score obtained by them.

        Also update the game state accordingly.

        Returns:
            int: Index of the player that won the trick.
            int: Penalty score of the cards obtained by the player that
                won the trick.
        """
        assert all(map(
            lambda hand: len(hand) == len(self.hands[0]),
            self.hands,
        )), \
            'all players must have same amount of cards for trick distribution'
        assert len(self.table_cards) == self.num_players, \
            'trick must be full for distribution'

        trick_winner_index = self._get_trick_winner()
        trick_penalty = self.penalize_cards(self.table_cards)
        self.penalties[trick_winner_index] += trick_penalty

        self.prev_collected[trick_winner_index] = \
            self.collected[trick_winner_index].copy()
        self.collected[trick_winner_index].extend(self.table_cards)
        self._update_state(self.table_cards,
                           self.collected_state(trick_winner_index))
        self.prev_table_cards = self.table_cards.copy()
        self.table_cards.clear()

        self.prev_leading_suit = self.leading_suit
        self.prev_leading_player_index = self.leading_player_index
        self.leading_player_index = trick_winner_index
        self.active_player_index = self.leading_player_index
        self.prev_was_first_trick = self.is_first_trick
        self.is_first_trick = False
        self.prev_trick_winner_index = trick_winner_index
        self.prev_trick_penalty = trick_penalty
        return trick_winner_index, trick_penalty

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
            Optional[int]: Penalty score of the cards obtained by the
                player that won the trick or `None` if it is
                still ongoing.
        """
        assert self._is_reset, \
            'please call `reset` before interacting with the game.'
        assert len(self.table_cards) < self.num_players, \
            'cannot play a card when trick is already full'
        self.prev_states[self.active_player_index] = self.state.copy()
        self.prev_leading_hearts_allowed[self.active_player_index] = \
            self.leading_hearts_allowed
        hand = self.hands[self.active_player_index]

        if action < 0 or action >= len(hand):
            # Just play the last card.
            adjusted_action = -1
        else:
            adjusted_action = action
        card_to_play = hand[adjusted_action]

        if (
                # No hearts or queen of spades in first trick.
                self.is_first_trick
                and self.has_penalty(card_to_play)
        ):
            if self.active_player_index == self.leading_player_index:
                index_without_penalty = self._first_index_without_penalty(hand)
            else:
                index_without_penalty = \
                    self._first_index_without_penalty_with_suit(
                        hand, self.leading_suit)

            if index_without_penalty is not None:
                adjusted_action = index_without_penalty
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
        self.prev_played_cards[self.active_player_index] = card_to_play
        self.prev_was_illegals[self.active_player_index] = was_illegal

        trick_winner_index: Optional[int]
        trick_penalty: Optional[int]
        if len(self.table_cards) == self.num_players:
            trick_winner_index, trick_penalty = self._distribute_trick()
        else:
            trick_winner_index, trick_penalty = (None, None)
        return card_to_play, was_illegal, trick_winner_index, trick_penalty

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

    def has_shot_the_moon(
            self,
            player_index: int,
    ) -> bool:
        """Return whether the given player has shot the moon.

        Requires the final penalty scores to be computed (see
        `self.compute_final_penalties`).

        Args:
            player_index (int): Index of the player to query whether
                they shot the moon for.

        Returns:
            bool: Whether the given player has shot the moon.
        """
        return (
            self.penalties[player_index] == 0
            and self.is_done()
            and (
                self.penalties[(player_index + 1)
                               % self.num_players]
                == self.max_penalty
            )
            and (
                self.penalties[(player_index - 1)
                               % self.num_players]
                == self.max_penalty
            )
        )

    @staticmethod
    def penalize_cards(cards: List[Card]) -> int:
        """Return the accumulated penalty score of the given cards.

        Args:
            cards (List[Card]): Cards to penalize.

        Returns:
            int: Accumulated penalty score of the cards.
        """
        total_penalty = 0
        for card in cards:
            total_penalty += HeartsGame.get_penalty(card)
        return total_penalty

    def compute_final_penalties(self) -> List[int]:
        """Compute and return the final penalty scores of the game,
        taking into account shooting the moon.

        Returns:
            List[int]: The final penalty scores of the game for each
                player, sorted by player indices.
        """
        # print('penalties:', self.penalties)
        for (player_index, penalty) in enumerate(self.penalties):
            if penalty != self.max_penalty:
                continue

            # Someone has shot the moon.
            self.penalties = [self.max_penalty] * self.num_players
            self.penalties[player_index] = 0
            break

        # Return a copy to avoid surprises when stored penalties change.
        return self.penalties.copy()

    def compute_rankings(self) -> List[int]:
        """Compute and return the final rankings of the game.

        Players with the same penalty score obtain the higher ranking.

        Requires the final penalties to be computed (see
        `compute_final_penalties`).

        Returns:
            List[int]: The final rankings of the game for each player,
                sorted by player indices.
        """
        rankings: List[Optional[int]] = [None] * self.num_players
        sorted_penalties = sorted(
            zip(self.penalties, range(self.num_players)),
            reverse=True,
        )

        same_penalty = []
        curr_ranking = self.num_players

        for (i, (penalty, player_index)) in enumerate(sorted_penalties):
            if (
                    i + 1 < len(sorted_penalties)
                    and sorted_penalties[i + 1][0] == penalty
            ):
                same_penalty.append(player_index)
                curr_ranking -= 1
                continue
            elif len(same_penalty) > 0:
                for same_penalty_player_index in same_penalty:
                    rankings[same_penalty_player_index] = curr_ranking
                same_penalty.clear()

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
            card, was_illegal, trick_winner_index, trick_penalty = \
                self.play_card(action)
            cards[player_index] = card
            was_illegals[player_index] = was_illegal

        assert trick_winner_index is not None and trick_penalty is not None, \
            'did not properly finish full trick'

        is_done = self.is_done()
        info: Dict[str, Any] = {
            'leading_player_index': leading_player_index,
            'cards': cards,
            'was_illegals': was_illegals,
            'trick_winner_index': trick_winner_index,
            'trick_penalty': trick_penalty,
        }

        if is_done:
            info['final_penalties'] = self.compute_final_penalties()
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
        self.penalties = [0] * self.num_players
        self.is_first_trick = True
        self.leading_hearts_allowed = False

        self.prev_hands = [[] for _ in range(self.num_players)]
        self.prev_played_cards = [None] * self.num_players
        self.prev_table_cards = []
        self.prev_collected = [[] for _ in range(self.num_players)]
        self.prev_was_illegals = [None] * self.num_players
        self.prev_states = [None] * self.num_players
        self.prev_was_first_trick = None
        self.prev_leading_hearts_allowed = [None] * self.num_players

        self.prev_trick_winner_index = None
        self.prev_trick_penalty = None
        self.prev_leading_suit = None
        self.prev_leading_player_index = None

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
        for rank in range(
                Card.NUM_RANKS - self._cards_per_suit[Card.SUIT_CLUB],
                Card.NUM_RANKS,
        ):
            card = Card(Card.SUIT_CLUB, rank)
            for (player_index, hand) in enumerate(self.hands):
                card_index = bisect.bisect_left(hand, card)
                if card_index >= len(hand) or hand[card_index] != card:
                    continue

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

        assert self.deck.size % self.num_players == 0, \
            'something went wrong during initialization'
        assert len(self.deck) == 0, \
            'deck must be empty at start of game'

        self.active_player_index = self.leading_player_index
        assert card_index is not None
        self._play_card(card_index)
        # We explicitly want the played card to still be set to `None`.
        # Refers to `self.prev_played_cards`.
        self.prev_was_illegals[self.leading_player_index] = False

        self._is_reset = True

    def __str__(self) -> str:
        output = []
        for (player_index, (hand, penalty, collected)) in enumerate(zip(
                self.hands,
                self.penalties,
                self.collected,
        )):
            output.append('Player ')
            output.append(str(player_index + 1))
            if player_index == self.active_player_index:
                output.append(' (active)')
            output.append(': ')
            output.append(', '.join(str(card) for card in hand))
            output.append('\n   Collected (penalty ')

            output.append(str(penalty))
            output.append('): ')
            output.append(', '.join(str(card) for card in collected))
            output.append('\n')

        output.append('\nTable: ')
        output.append(', '.join(str(card) for card in self.table_cards))
        output.append('\nLeading hearts allowed: ')
        output.append('yes' if self.leading_hearts_allowed else 'no')

        return ''.join(output)
