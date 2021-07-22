"""
Primitive classes for classical card game decks.
"""

import bisect
import random
from typing import List, Union

from hearts_gym.utils.typing import Seed

unicode_level = 1
"""Default for how advanced the unicode used for printing cards
should be.
"""


class Card:
    """A standard French playing card with a fixed amount of suits
    and ranks.

    Internally, the suit and rank are stored as integer values.
    String representations may be obtained by using these values on
    class values such as `SUITS` or `RANKS`.

    Comparisons are implemented based on the commonly used values of
    suits and ranks in poker (alphabetical order). This means that
    clubs < diamonds < hearts < spades. After the suit comparison, the
    ranks are compared.
    """

    __slots__ = ['suit', 'rank']

    NUM_SUITS = 4

    SUIT_CLUB = 0
    SUIT_DIAMOND = 1
    SUIT_HEART = 2
    SUIT_SPADE = 3

    NUM_RANKS = 13
    # We assign value 0 for rank 2 up to value 8 for rank 10. The upper
    # ranks start from value 9 for the jack up to value 11 for the king.
    # Finally, the highest-valued card with value 12 is the ace.
    MAX_RANK = 12

    SUITS = ['C', 'D', 'H', 'S']
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    UNICODE_SUITS = ['♣', '♢', '♡', '♠']
    UNICODE_CARDS_START = [0x1f0d1, 0x1f0c1, 0x1f0b1, 0x1f0a1]

    def __init__(self, suit: int, rank: int) -> None:
        """Construct a card with the suit and rank.

        Args:
            suit (int): Numerical value for a suit.
            rank (int): Numerical value for a rank.
        """
        assert 0 <= suit < self.NUM_SUITS
        assert 0 <= rank <= self.MAX_RANK
        self.suit = suit
        self.rank = rank

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented

        return self.suit == other.suit and self.rank == other.rank

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented

        return (self.suit < other.suit
                or self.suit == other.suit and self.rank < other.rank)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented

        return (self.suit < other.suit
                or self.suit == other.suit and self.rank <= other.rank)

    def as_str(self, unicode_level: int = unicode_level) -> str:
        """Return self as a string.

        Args:
            unicode_level (int): How advanced unicode to use.

        Returns:
            str: String representation of self.
        """
        if unicode_level == 0:
            return self.SUITS[self.suit] + self.RANKS[self.rank]
        elif unicode_level == 1:
            return self.UNICODE_SUITS[self.suit] + self.RANKS[self.rank]
        else:
            unicode_start = self.UNICODE_CARDS_START[self.suit]
            if self.rank == self.MAX_RANK:
                # Ace has rank 0 for unicode.
                unicode_rank_offset = 0
            elif self.rank > 9:
                # Avoid knight card.
                unicode_rank_offset = self.rank + 2
            else:
                # Ace has rank 0 for unicode, so offset by one.
                unicode_rank_offset = self.rank + 1
            return chr(unicode_start + unicode_rank_offset)

    def __str__(self) -> str:
        return self.as_str(unicode_level)

    def __repr__(self) -> str:
        return 'Card(' + str(self.suit) + ', ' + str(self.rank) + ')'

    def __hash__(self) -> int:
        return hash((self.suit, self.rank))


class Deck:
    """A standard playing card deck."""
    MAX_SIZE = 52

    def __init__(
            self,
            size: int,
            build_ordered: bool,
            seed: Seed = None,
    ) -> None:
        """Construct a pre-shuffled card deck with a given size.

        If the size is not divisible by the number of suits, meaning
        that not all suits have the same number of cards, the
        lowest-ranked cards are removed from the suits
        - club,
        - diamond,
        - heart,
        - spade,
        in that order until the desired size is reached.

        Args:
            size (int): How many cards should be in the deck.
            build_ordered (bool): Whether lower-ranked cards should be
                discarded. If `False`, discard any cards (only relevant
                if `size` is less than `Deck.MAX_SIZE`).
            seed (Seed): Random number generator seed.
        """
        assert 0 < size <= self.MAX_SIZE, \
            f'deck size must be in (0, {self.MAX_SIZE}]'

        self._rng = random.Random(seed)
        self._build_ordered = build_ordered

        # This many suits will have one card more than the others
        # if `build_ordered`.
        num_larger_suits = size % Card.NUM_SUITS
        self._divides_evenly = num_larger_suits == 0

        if not build_ordered:
            self._used_cards = [
                Card(suit, rank)
                for suit in range(Card.NUM_SUITS)
                for rank in range(Card.NUM_RANKS)
            ]
        else:
            min_num_cards_per_suit = size // Card.NUM_SUITS
            smallest_larger_suit = Card.NUM_SUITS - num_larger_suits
            self._used_cards = [
                Card(suit, rank)
                for suit in range(Card.NUM_SUITS)
                for rank in range(
                        (
                            Card.NUM_RANKS
                            - (
                                min_num_cards_per_suit
                                + (suit >= smallest_larger_suit)
                            )
                        ),
                        Card.NUM_RANKS,
                )
            ]

        self.size = size
        self.reset()

    def __len__(self) -> int:
        """Return how many cards are left in the deck.

        Returns:
            int: How many cards are left in the deck.
        """
        return len(self._deck)

    def __str__(self) -> str:
        return '[' + ', '.join(str(card) for card in self._deck) + ']'

    def __repr__(self) -> str:
        return str(self._deck)

    def reset(self) -> None:
        """Re-build and shuffle the deck."""
        self._deck = self._build_cards()
        if self._build_ordered:
            self.shuffle_deck()

    def _build_cards(self) -> List[Card]:
        """Return a list of cards.

        Returns:
            List[Card]: A list of cards. If `self._build_ordered`, these
                are not shuffled.
        """
        if not self._build_ordered:
            return self._rng.sample(self._used_cards, self.size)
        return self._used_cards.copy()

    def shuffle_deck(self) -> None:
        """Shuffle the deck."""
        self._rng.shuffle(self._deck)

    def take(self, n: int = 1) -> List[Card]:
        """Remove and return `n` cards from the deck.

        Args:
            n (int): How many cards to take.

        Returns:
            List[Card]: `n` cards that are now removed from the deck.
        """
        cards = self._deck[:n]
        del self._deck[:n]
        return cards

    def remove(self, cards: Union[List[Card], Card]) -> None:
        """Permanently remove the given cards from the deck.

        Permanently means that the cards will not re-appear after
        resetting the deck. Cards to remove that are already not in the
        deck are ignored.

        Args:
            cards (Union[List[Card], Card]): Cards to permanently remove
                from the deck.
        """
        if not isinstance(cards, list):
            cards = [cards]

        for card in cards:
            remove_index = bisect.bisect_left(self._used_cards, card)
            if (
                    remove_index >= len(self._used_cards)
                    or self._used_cards[remove_index] != card
            ):
                continue
            del self._used_cards[remove_index]
            self.size -= 1
