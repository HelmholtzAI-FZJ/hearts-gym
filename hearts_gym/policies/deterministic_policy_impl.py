"""
A hard-coded Hearts policy implementation that yields deterministic
actions for each state/observation.
"""

from ray.rllib.utils.typing import TensorType

from hearts_gym.utils.typing import Action
from .mock_game import MockGame


class DeterministicPolicyImpl:
    """A hard-coded Hearts policy implementation that yields
    deterministic actions for each state/observation.

    The policy has access to a mock `game` that is built from the
    observations. This mock game supports common methods to build a
    deterministic policy. If the mock game is missing an operation, it
    can be implemented from scratch due to having access to the raw
    observations. Please see `hearts_gym.policies.MockGame` for
    more information.

    The mock game is expected to be updated from elsewhere.
    """

    def __init__(self, mock_game: MockGame) -> None:
        """Construct a deterministic policy implementation that observes
        the given mock game.

        Args:
            mock_game (MockGame): Mock game to observe.
        """
        self.game = mock_game

    def compute_action(self, obs: TensorType) -> Action:
        """Compute a deterministic action for the given observations.

        The internal mock game is expected to have been updated from
        elsewhere.

        Args:
            obs (TensorType): Observation from the environment to
                compute an action for. May be safely ignored due to the
                mock game implementing more sensible access to the
                observed data.

        Returns:
            Action: Action to execute given the observation.
        """
        raise NotImplementedError(
            f'please implement the `compute_action` method of class '
            f'`{self.__class__.__name__}`'
        )
