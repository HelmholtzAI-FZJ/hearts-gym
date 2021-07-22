import numpy as np
from typing import Optional, Tuple, Iterable

from hearts_gym.envs.card_deck import Card


class Probability(float):
    def __init__(self, value) -> None:
        if not 0 <= value <= 1:
            raise ValueError(f"Invalid probability value of {value} encountered.")
        super().__init__()


class Certainty:
    """The NEVER and ALWAYS can be treated as probabilities!"""
    MAYBE = -1
    NEVER = Probability(0)
    ALWAYS = Probability(1)


ALWAYS = Certainty.ALWAYS
MAYBE = Certainty.MAYBE
NEVER = Certainty.NEVER


class DeepState:
    def __init__(self, game) -> None:
        self.game = game
        self.cards_on_hand = np.array(self.game.hand)
        self.cards_on_table = np.array(self.game.table_cards)
        self.unseen_cards = self.game.unknown_cards
        self.cards_by_others = [c for c in self.unseen_cards if c not in self.cards_on_hand]
        self.legal_indices_to_play = np.array(self.game.get_legal_actions())
        self.legal_cards_to_play = [self.cards_on_hand[i] for i in self.legal_indices_to_play]
        self.penalty_on_table = np.array([self.game.get_penalty(c) for c in self.cards_on_table])
        self.penalty_of_action_cards = np.array([self.game.get_penalty(c) for c in self.legal_cards_to_play])
        self.T = len(self.cards_on_table)
        self.H = len(self.cards_on_hand)
        self.A = len(self.legal_indices_to_play)

        assert np.shape(self.legal_cards_to_play) == (self.A,), f"shape was {np.shape(self.penalty_of_action_cards)}"
        assert np.shape(self.legal_indices_to_play) == (self.A,), f"shape was {np.shape(self.legal_indices_to_play)}"
        assert np.shape(self.penalty_on_table) == (self.T,), f"shape was {np.shape(self.penalty_on_table)}"
        assert np.shape(self.penalty_of_action_cards) == (self.A,), f"shape was {np.shape(self.penalty_of_action_cards)}"
        super().__init__()

    def calculate_get_avoid_probabilities(self) -> Tuple[np.ndarray, np.ndarray]:
        p_get_trick = np.repeat(np.nan, self.A)
        p_avoid_trick = np.repeat(np.nan, self.A)
        for a, c in enumerate(self.legal_cards_to_play):
            # Would this card get the trick?
            gets = gets_trick(c, self.cards_on_table, self.cards_by_others)
            p_get_trick[a] = gets
            p_avoid_trick[a] = 1 - gets

        assert not any(np.isnan(p_get_trick))
        assert not any(np.isnan(p_avoid_trick))
        return p_get_trick, p_avoid_trick


def gets_trick(card: Card, table_cards: Iterable[Card], cards_by_others: Iterable[Card]) -> Probability:
    """
    Determines the certainty of a given `card` "winning" a trick,
    based on cards on the table and cards of other players.
    """
    lead_suit = table_cards[0].suit if len(table_cards) else card.suit
    lead_rank = table_cards[0].rank if len(table_cards) else card.rank

    if card.suit != lead_suit:
        return NEVER
    # We are qualified based on suit!

    # We have established that our card has the correct suit.
    # Are there cards on the table that are ranked higher?
    table_cards_that_beat_this = filter_cards_above(table_cards, card.suit, card.rank)
    if table_cards_that_beat_this:
        return NEVER

    if len(table_cards) == 3:
        # We are playing the last card! This is easy.
        # 1. There are no table cards that are higher.
        # 2. There are no cards still coming in.
        return ALWAYS

    # Based on unplayed cards by competitors...
    alien_cards_that_beat_this = filter_cards_above(cards_by_others, card.suit, card.rank)
    if alien_cards_that_beat_this:
        if len(cards_by_others) < 4:
            # This is the last tick, so that one competitor card that beats us WILL be played.
            return NEVER
        # More than one round is left and there are _some_ competitor cards that could beat us.
        # The following probability calculation is less than ideal in at least the following ways:
        # - We don't have probabilities of other players holding onto certain cards, that could further
        #   constrain the potential outcomes.
        # - We don't know which rules the other agents use for their decision making.
        # - Because we don't know which players hold onto the remaining cards, we can't filter down to only the
        #   cards that could "legally" be played.
        # - The "unplayed cards" also include cards by players that have already played in this trick.
        
        # The tradeoff:
        # Assume that the upcoming cards will be drawn randomly from the remaining cards.
        return p_gets_trick(
            n_others_higher=len(alien_cards_that_beat_this),
            n_others_lower=len(cards_by_others) - len(alien_cards_that_beat_this),
            n_incoming=4 - len(table_cards) - 1,
        )

    # So
    # 1. No cards on the table that can beat this one.
    # 2. No competitor cards that can beat this one.
    # 👉 This card always takes the trick.
    return ALWAYS


def filter_cards_above(cards: Iterable[Card], ref_suit:int, ref_rank: int) -> Tuple[Card]:
    """Returns all elements from `cards` that have the same suit, and a higher rank."""
    return tuple(
        c
        for c in cards
        if c.suit == ref_suit and c.rank > ref_rank
    )


def p_gets_trick(
    n_others_higher: int,
    n_others_lower: int,
    n_incoming: int,
) -> Probability:
    if n_incoming == 0:
        raise ValueError("Useless probability calculation. Probabily an upstream logic error.")
    p_lower = n_others_lower / (n_others_lower + n_others_higher)
    if p_lower == 1:
        return ALWAYS
    if p_lower == 0:
        return NEVER
    return p_lower ** n_incoming


def expected_inbound_penalty(
    card_penalties: np.ndarray,
    n_inbound: int,
) -> np.ndarray:
    """Elementwise contributions of penalty after drawing without replacement `n_inbound` times."""
    card_penalties = np.array(card_penalties)
    p_drawn = n_inbound / len(card_penalties)
    return card_penalties * p_drawn
