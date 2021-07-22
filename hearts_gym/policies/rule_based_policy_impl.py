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
import os

from hearts_gym.utils.logic import Probability, Certainty, gets_trick, ALWAYS, NEVER, MAYBE, p_gets_trick

class RuleBasedPolicyImpl(DeterministicPolicyImpl):
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


class RulebasedNext(DeterministicPolicyImpl):
    def compute_action(self, obs: TensorType) -> Action:
        # Collect some observations about the current round
        cards_on_hand = np.array(self.game.hand)
        cards_on_table = np.array(self.game.table_cards)
        unseen_cards = self.game.unknown_cards
        cards_by_others = [c for c in unseen_cards if c not in cards_on_hand]
        legal_indices_to_play = np.array(self.game.get_legal_actions())
        legal_cards_to_play = [cards_on_hand[i] for i in legal_indices_to_play]
        penalty_on_table = np.array([self.game.get_penalty(c) for c in cards_on_table])
        penalty_of_action_cards = np.array([self.game.get_penalty(c) for c in legal_cards_to_play])

        # Some constants for easier iterations or slicing
        T = len(cards_on_table)
        H = len(cards_on_hand)
        A = len(legal_indices_to_play)

        assert np.shape(legal_cards_to_play) == (A,), f"shape was {np.shape(penalty_of_action_cards)}"
        assert np.shape(legal_indices_to_play) == (A,), f"shape was {np.shape(legal_indices_to_play)}"
        assert np.shape(penalty_on_table) == (T,), f"shape was {np.shape(penalty_on_table)}"
        assert np.shape(penalty_of_action_cards) == (A,), f"shape was {np.shape(penalty_of_action_cards)}"

        # Probabilities of certain outcomes (NaN means unknown).
        p_get_trick = np.repeat(np.nan, A)
        p_avoid_trick = np.repeat(np.nan, A)
        for a, c in enumerate(legal_cards_to_play):
            # Would this card get the trick?
            gets = gets_trick(c, cards_on_table, cards_by_others)
            p_get_trick[a] = gets
            p_avoid_trick[a] = 1 - gets


        assert not any(np.isnan(p_get_trick))
        assert not any(np.isnan(p_avoid_trick))

        action_index = None
        ######## HEURISTICS ########
        if sum(penalty_on_table) > 0 and any(p_avoid_trick == 1):
            # There's already a penalty, but we can avoid it so let's do that.
            # Fight off with the most-penalized card that defends successfully.
            action_index = np.argmax(p_avoid_trick * penalty_of_action_cards)
            # TODO: Choose the defense card based on penalty/value:
            # + Defend with ♠Q if we can.
            # + Defend with a ♥️ (higher is better, except the ♥️A)
            # + Defend with ♣️ or diamond
            # + Don't defend with a ♠ unless we have ♠Q ourselves
        if action_index is None:
            # What would be the penalties of the possible actions?
            penalty_outcome = sum(penalty_on_table) + p_get_trick * penalty_of_action_cards
            # TODO: Consider penalties of incoming cards.
            # TODO: Calculate a second vector [0-1] to describe the "value" of action cards for future rounds.

            # Take the action that minimizes the expected penalties.
            action_index = np.argmin(penalty_outcome)

        action_card = legal_cards_to_play[action_index]

        logfile.write(textwrap.dedent(f"""
        table  : {cards_on_table}
        actions: {legal_cards_to_play}
        p_get  : {p_get_trick.tolist()}
        p_avoid: {p_avoid_trick.tolist()}
        action : Card({action_card.suit}, {action_card.rank})
        """))
        logfile.flush()

        return legal_indices_to_play[action_index]



class RulebasedPrevious(DeterministicPolicyImpl):
    def compute_action(self, obs: TensorType) -> Action:
        # Collect some observations about the current round
        cards_on_hand = np.array(self.game.hand)
        cards_on_table = np.array(self.game.table_cards)
        legal_indices_to_play = np.array(self.game.get_legal_actions())
        legal_cards_to_play = [cards_on_hand[i] for i in legal_indices_to_play]
        penalty_on_table = np.array([self.game.get_penalty(c) for c in cards_on_table])
        penalty_of_action_cards = np.array([self.game.get_penalty(c) for c in legal_cards_to_play])
        highest_rank_on_table = max(c.rank for c in cards_on_table) if len(cards_on_table) else -1

        # Some constants for easier iterations or slicing
        T = len(cards_on_table)
        H = len(cards_on_hand)
        A = len(legal_indices_to_play)

        assert np.shape(legal_cards_to_play) == (A,), f"shape was {np.shape(penalty_of_action_cards)}"
        assert np.shape(legal_indices_to_play) == (A,), f"shape was {np.shape(legal_indices_to_play)}"
        assert np.shape(penalty_on_table) == (T,), f"shape was {np.shape(penalty_on_table)}"
        assert np.shape(penalty_of_action_cards) == (A,), f"shape was {np.shape(penalty_of_action_cards)}"

        # Probabilities of certain outcomes (NaN means unknown).
        p_get_trick = np.repeat(np.nan, A)
        p_avoid_trick = np.repeat(np.nan, A)
        for a, c in enumerate(legal_cards_to_play):
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
                and len(cards_on_table) == self.game.num_players - 1
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
            a = sum(penalty_on_table) > 0
            b = any(p_avoid_trick == 1)
            if a and b:
                # Fight off with the most-penalized card that defends successfully.
                return legal_indices_to_play[np.nanargmax(p_avoid_trick * penalty_of_action_cards)]


        if not all(np.isnan(p_get_trick)):
            # What would be the penalties of the possible actions?
            penalty_outcome = sum(penalty_on_table) + p_get_trick * penalty_of_action_cards
            assert np.shape(penalty_outcome) == (A,), f"penalty_outcome.shape was {np.shape(penalty_outcome)}"
            if any(penalty_outcome == 0):
                # There's no penalty, and we can take the tick without taking a penalty 🎉
                return np.random.choice(legal_indices_to_play[penalty_outcome == 0])

        # Heuristics did not conclude. Let's make an unpredictable move! 😈
        return np.random.choice(legal_indices_to_play)


