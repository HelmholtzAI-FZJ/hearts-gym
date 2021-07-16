"""
A hard-coded Hearts policy implementation that yields rule-based actions
for each state/observation.

Rule-based in this context means fixed behavior according to pre-defined
rules (e.g. "always play the second legal card in hand").
"""

from ray.rllib.utils.typing import TensorType

from hearts_gym.utils.typing import Action
from .deterministic_policy_impl import DeterministicPolicyImpl


class RuleBasedPolicyImpl(DeterministicPolicyImpl):
    """A rule-based policy implementation yielding deterministic actions
    for given observations.

    The policy has access to a `mock_game` that is built from the
    observations. This mock game supports common methods to build a
    deterministic policy. If the mock game is missing an operation, it
    can be implemented from scratch due to having access to the raw
    observations. Please see `hearts_gym.policies.MockGame` for
    more information.

    The mock game is expected to be updated from elsewhere.
    """

    def compute_action(self, obs: TensorType) -> Action:
        raise NotImplementedError('please implement the rule-based agent')
