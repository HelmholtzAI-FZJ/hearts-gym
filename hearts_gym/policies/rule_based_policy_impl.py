"""
A hard-coded Hearts policy implementation that yields rule-based actions
for each state/observation.

Rule-based in this context means fixed behavior according to pre-defined
rules (e.g. "always play the second legal card in hand").
"""
from hearts_gym.envs.card_deck import Card
from typing import List, Optional, Iterable, Tuple
from ray.rllib.utils.typing import TensorType

from hearts_gym.utils.typing import Action
from .deterministic_policy_impl import DeterministicPolicyImpl

import enum
import textwrap
import pathlib
import numpy as np

from hearts_gym.utils.logic import Probability, Certainty, gets_trick, ALWAYS, NEVER, MAYBE, p_gets_trick, DeepState


class LoggingMixin:
    handles = {}

    def log(self, message):
        if not id(self) in self.handles:
            fp = pathlib.Path("logs", f"{self.__class__.__name__}_{id(self)}.log")
            LoggingMixin.handles[id(self)] = fp.open("w")
        logfile = LoggingMixin.handles[id(self)]
        logfile.write(message)
        logfile.flush()
        return

class RuleBasedPolicyImpl(DeterministicPolicyImpl, LoggingMixin):
    """A rule-based policy implementation yielding deterministic actions
    for given observations.

    The policy has access to an observed `game` that is built from the
    observations. This observed game supports common methods to build a
    deterministic policy. If the observed game is missing an operation,
    it can be implemented from scratch due to having access to the raw
    observations. Please see `hearts_gym.policies.ObservedGame` for more
    information.

    The observed game is expected to be updated from elsewhere.
    """
    pass


logfile = pathlib.Path("logs", f"logfile_{os.getpid()}.log").open("w")


class RulebasedV2(RuleBasedPolicyImpl):
    def compute_action(self, obs: TensorType) -> Action:
        ds = DeepState(self.game)
        p_get_trick, p_avoid_trick = ds.calculate_get_avoid_probabilities()

        # What would be the penalties of the possible actions?
        penalty_lower_bound = sum(ds.penalty_on_table) + p_get_trick * ds.penalty_of_action_cards
        # TODO: Determine maximum penalty of the incoming cards
        # TODO: Determine expected penalty

        # Penalties are int-valued, so we can use values <1 to sort actions of the same penalty based on card "value":
        action_card_value = np.linspace(0.1, 0, ds.A)
        # TODO: Write helper function to determine action card values with heuristics.

        action_index = None
        ######## HEURISTICS ########
        if sum(ds.penalty_on_table) > 0 and any(p_avoid_trick == 1):
            # There's already a penalty, but we can avoid it so let's do that.
            # Fight off with the most-penalized card that defends successfully.
            action_index = np.argmin(penalty_lower_bound + action_card_value)
            # TODO: Choose the defense card based on penalty/value:
            # ☑ Defend with ♠Q if we can.
            # + Defend with a ♥️ (higher is better, except the ♥️A)
            # + Defend with ♣️ or diamond
            # + Don't defend with a ♠ unless we have ♠Q ourselves
        if action_index is None:
            # There's NO penalty on the table, OR we can't ALWAYS avoid the trick.
            # Take the action that minimizes the penalties and loss of valuable cards.
            action_index = np.argmin(penalty_lower_bound + action_card_value)

        action_card = ds.legal_cards_to_play[action_index]

        self.log(textwrap.dedent(f"""
        table  : {ds.cards_on_table}
        actions: {ds.legal_cards_to_play}
        p_get  : {p_get_trick.tolist()}
        p_avoid: {p_avoid_trick.tolist()}
        action : Card({action_card.suit}, {action_card.rank})
        """))

        return ds.legal_indices_to_play[action_index]



class RulebasedV1(RuleBasedPolicyImpl):
    def compute_action(self, obs: TensorType) -> Action:
        ds = DeepState(self.game)

        highest_rank_on_table = max(c.rank for c in ds.cards_on_table) if len(ds.cards_on_table) else -1

        # Probabilities of certain outcomes (NaN means unknown).
        p_get_trick = np.repeat(np.nan, ds.A)
        p_avoid_trick = np.repeat(np.nan, ds.A)
        for a, c in enumerate(ds.legal_cards_to_play):
            # Can this card definitely fight off the trick?
            if (
                c.suit != self.game.leading_suit
                or (c.rank < highest_rank_on_table)
            ):
                p_get_trick[a] = 0
                p_avoid_trick[a] = 1
                continue
            # Can this card definitely get the trick?
            elif (
                c.suit == self.game.leading_suit
                and c.rank > highest_rank_on_table
                and len(ds.cards_on_table) == self.game.num_players - 1
            ):
                p_get_trick[a] = 1
                p_avoid_trick[a] = 0
            # Oh boi, it's getting complicated.
            else:
                # TODO: try to find the get/avoid probabilities based on which cards are still in the game
                pass

        ######## HEURISTICS ########
        if not all(np.isnan(p_avoid_trick)):
            # There's already a penalty, but we can avoid it let's do that.
            a = sum(ds.penalty_on_table) > 0
            b = any(p_avoid_trick == 1)
            if a and b:
                # Fight off with the most-penalized card that defends successfully.
                return ds.legal_indices_to_play[np.nanargmax(p_avoid_trick * ds.penalty_of_action_cards)]


        if not all(np.isnan(p_get_trick)):
            # What would be the penalties of the possible actions?
            penalty_outcome = sum(ds.penalty_on_table) + p_get_trick * ds.penalty_of_action_cards
            assert np.shape(penalty_outcome) == (ds.A,), f"penalty_outcome.shape was {np.shape(penalty_outcome)}"
            if any(penalty_outcome == 0):
                # There's no penalty, and we can take the tick without taking a penalty 🎉
                return np.random.choice(ds.legal_indices_to_play[penalty_outcome == 0])

        # Heuristics did not conclude. Let's make an unpredictable move! 😈
        return np.random.choice(ds.legal_indices_to_play)


