import pytest

from hearts_gym.envs.hearts_game import Card
from hearts_gym.utils.logic import Probability, Certainty, ALWAYS, NEVER, MAYBE


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


def test_filter_cards_that_get():
    # TODO: implement
    pass


def test_gets_trick():
    # TODO: implement
    pass
