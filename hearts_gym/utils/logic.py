import pytest
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


def gets_trick(card: Card, table_cards: Iterable[Card], cards_by_others: Iterable[Card]) -> Certainty:
    """
    Determines the certainty of a given `card` "winning" a trick,
    based on cards on the table and cards of other players.
    """
    lead_suit = table_cards[0].suit if len(table_cards) else card.suit
    lead_rank = table_cards[0].rank if len(table_cards) else card.rank

    # Based on the cards already included in the tick...
    table_cards_that_beat_this = filter_cards_that_get(table_cards, lead_suit, lead_rank)
    if table_cards_that_beat_this:
        return NEVER

    # Based on unplayed cards by competitors...
    alien_cards_that_beat_this = filter_cards_that_get(cards_by_others, lead_suit, lead_rank)
    if alien_cards_that_beat_this:
        if len(cards_by_others) < 4:
            # This is the last tick, so that one competitor card that beats us WILL be played.
            return NEVER
        # More than one round is left and there are _some_ competitor cards that could beat us.
        # For exact probabilities we'll need a different function.
        # TODO: Calculate probs separately and return them instead of MAYBE.
        return MAYBE

    # So
    # 1. No cards on the table that can beat this one.
    # 2. No competitor cards that can beat this one.
    # 👉 This card always takes the trick.
    return ALWAYS


def filter_cards_that_get(cards: Iterable[Card], lead_suit:Optional[int], lead_rank: Optional[int]) -> Tuple[Card]:
    """Returns all elements from `cards` that would "win" over a certain lead."""
    return tuple(
        c
        for c in cards
        if c.suit == lead_suit and c.rank > lead_rank
    )

