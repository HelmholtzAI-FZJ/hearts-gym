"""
A hard-coded Hearts policy implementation that yields deterministic
actions for each state/observation.
"""

from ray.rllib.utils.typing import TensorType

from hearts_gym.utils.typing import Action
from .observed_game import ObservedGame


class DeterministicPolicyImpl:
    """A hard-coded Hearts policy implementation that yields
    deterministic actions for each state/observation.

    The policy has access to an observed `game` that is built from the
    observations. This observed game supports common methods to build a
    deterministic policy. If the observed game is missing an operation,
    it can be implemented from scratch due to having access to the raw
    observations. Please see `hearts_gym.policies.ObservedGame` for more
    information.

    The observed game is expected to be updated from elsewhere.
    """

    def __init__(self, observed_game: ObservedGame) -> None:
        """Construct a deterministic policy implementation that acts in
        the given observed game.

        Args:
            observed_game (ObservedGame): Observed game to act in.
        """
        self.game = observed_game

    def compute_action(self, obs: TensorType) -> Action:
        """Compute a deterministic action for the given observations.

        The internal observed game is expected to have been updated from
        elsewhere.

        Args:
            obs (TensorType): Observation from the environment to
                compute an action for. May be safely ignored due to the
                observed game implementing more sensible access to the
                observed data.

        Returns:
            Action: Action to execute given the observation.
        """
        raise NotImplementedError(
            f'please implement the `compute_action` method of class '
            f'`{self.__class__.__name__}`'
        )
