import pytest

from hearts_gym.envs.hearts_game import Card
from hearts_gym.utils.logic import Probability, Certainty, ALWAYS, NEVER, MAYBE, filter_cards_above, gets_trick, p_gets_trick

# Suits
A, B, C, D = 0, 1, 2, 3

# Cards
A2 = Card(suit=A, rank=0)
A3 = Card(suit=A, rank=1)
A4 = Card(suit=A, rank=2)
A5 = Card(suit=A, rank=3)
A6 = Card(suit=A, rank=4)
A7 = Card(suit=A, rank=5)
A8 = Card(suit=A, rank=6)
A9 = Card(suit=A, rank=7)
A10 = Card(suit=A, rank=8)
AJ = Card(suit=A, rank=9)
AQ = Card(suit=A, rank=10)
AK = Card(suit=A, rank=11)
AA = Card(suit=A, rank=12)

B2 = Card(suit=B, rank=0)
B3 = Card(suit=B, rank=1)
B4 = Card(suit=B, rank=2)
B5 = Card(suit=B, rank=3)
B6 = Card(suit=B, rank=4)
B7 = Card(suit=B, rank=5)
B8 = Card(suit=B, rank=6)
B9 = Card(suit=B, rank=7)
B10 = Card(suit=B, rank=8)
BJ = Card(suit=B, rank=9)
BQ = Card(suit=B, rank=10)
BK = Card(suit=B, rank=11)
BA = Card(suit=B, rank=12)

C2 = Card(suit=C, rank=0)
C3 = Card(suit=C, rank=1)
C4 = Card(suit=C, rank=2)
C5 = Card(suit=C, rank=3)
C6 = Card(suit=C, rank=4)
C7 = Card(suit=C, rank=5)
C8 = Card(suit=C, rank=6)
C9 = Card(suit=C, rank=7)
C10 = Card(suit=C, rank=8)
CJ = Card(suit=C, rank=9)
CQ = Card(suit=C, rank=10)
CK = Card(suit=C, rank=11)
CA = Card(suit=C, rank=12)

D2 = Card(suit=D, rank=0)
D3 = Card(suit=D, rank=1)
D4 = Card(suit=D, rank=2)
D5 = Card(suit=D, rank=3)
D6 = Card(suit=D, rank=4)
D7 = Card(suit=D, rank=5)
D8 = Card(suit=D, rank=6)
D9 = Card(suit=D, rank=7)
D10 = Card(suit=D, rank=8)
DJ = Card(suit=D, rank=9)
DQ = Card(suit=D, rank=10)
DK = Card(suit=D, rank=11)
DA = Card(suit=D, rank=12)


def test_types():
    assert MAYBE == -1
    assert NEVER == 0
    assert ALWAYS == 1
    assert isinstance(Certainty.ALWAYS, Probability)
    assert isinstance(Certainty.NEVER, Probability)
    assert isinstance(Probability(1/2), Probability)
    with pytest.raises(ValueError):
        Probability(-3)
    with pytest.raises(ValueError):
        Probability(2)
    pass


def test_filter_cards_above():
    assert filter_cards_above(
        cards=[],
        ref_suit=D2.suit,
        ref_rank=D2.rank,
    ) == tuple()

    assert filter_cards_above(
        cards=[A4, A6, A10, B5],
        ref_suit=A5.suit,
        ref_rank=A5.rank,
    ) == (A6, A10)

    assert filter_cards_above(
        cards=[A4, A6, A10, B5],
        ref_suit=C2.suit,
        ref_rank=C2.rank,
    ) == tuple()

    assert filter_cards_above(
        cards=[A4, D6],
        ref_suit=A4.suit,
        ref_rank=A4.rank,
    ) == tuple()
    pass


def test_gets_trick_as_first():
    assert gets_trick(
        card=C10,
        table_cards=[],
        cards_by_others=[AK, BA, C9],
    ) is ALWAYS

    assert gets_trick(
        card=C10,
        table_cards=[],
        cards_by_others=[AK, BA, CJ],
    ) is NEVER

    assert gets_trick(
        card=C10,
        table_cards=[],
        cards_by_others=[AK, BA, CJ, C8],
    ) == (3/4 * 3/4 * 3/4)
    pass

def test_gets_trick_as_intermediate():
    assert gets_trick(
        card=A5,
        table_cards=[A4, D6],
        cards_by_others=[D6, A3, A2, C5, CK],
    ) is ALWAYS

    assert gets_trick(
        card=A5,
        table_cards=[A7, D6],
        cards_by_others=[D6, A3, A2, C5, CK],
    ) is NEVER

    assert gets_trick(
        card=BA,
        table_cards=[A7, D6],
        cards_by_others=[B2],
    ) is NEVER

    assert gets_trick(
        card=A5,
        table_cards=[A7, D6],
        cards_by_others=[A9],
    ) is NEVER

    assert gets_trick(
        card=A5,
        table_cards=[A2, D6],
        cards_by_others=[A9, A3, B2, C7],
    ) == 3/4
    pass


def test_gets_trick_as_last():
    assert gets_trick(
        card=B10,
        table_cards=[A4, CK, D6],
        cards_by_others=[D6, A3, A2, C5],
    ) is NEVER

    assert gets_trick(
        card=A7,
        table_cards=[A4, CK, D6],
        cards_by_others=[D6, A3, CA, C5],
    ) is ALWAYS

    assert gets_trick(
        card=A7,
        table_cards=[A4, CK, D6],
        cards_by_others=[D6, A3, A8, C5],
    ) is ALWAYS

    assert gets_trick(
        card=A7,
        table_cards=[A4, CK, D6],
        cards_by_others=[A3, A8, C5],
    ) is ALWAYS
    pass


def test_p_gets_trick():
    assert p_gets_trick(0, 5, 2) == 1
    assert p_gets_trick(5, 0, 2) == 0
    with pytest.raises(ValueError, match="Useless"):
        p_gets_trick(5, 3, 0)
    assert p_gets_trick(5, 5, 1) == 0.5
    assert p_gets_trick(2, 2, 2) == 0.25
    pass
