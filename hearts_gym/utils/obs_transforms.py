"""
Functionality for custom transformations on observations.
"""

from typing import Any, List
from uuid import UUID

import numpy as np

from .typing import Observation


class ObsTransform:
    """Abstract class for transformations on observations.

    Subclasses must implement the `transform` method.
    """

    def transform(self, obs: Any, player_index: int, uuid: UUID) -> Any:
        """Return a new observation after applying a transformation.

        Args:
            obs (Any): Observation to transform.
            player_index (int): Index of the player observing.
            uuid (UUID): A uniquely identifying ID of the game
                being observed.

        Returns:
            Any: Transformed observation.
        """
        raise NotImplementedError(
            f'please override the `transform` method '
            f'for {self.__class__.__name__}'
        )

    def __call__(self, obs: Any, uuid: UUID) -> Any:
        """Return a new observation after applying the
        `transform` function.

        Args:
            obs (Any): Observation to transform.
            uuid (UUID): A uniquely identifying ID of the game
                being observed.

        Returns:
            Any: Transformed observation.
        """
        return self.transform(obs, uuid)


def apply_obs_transforms(
        obs_transforms: List[ObsTransform],
        obs: Observation,
        player_index: int,
        uuid: UUID,
) -> Any:
    """Return the given observation received after applying all
    given transformations.

    Args:
        obs_transforms (List[ObsTransform]): Transformations to apply.
        obs (Observation): The observation before
            any transformation.
        player_index (int): Index of the player observing.
        uuid (UUID): A uniquely identifying ID of the game
            being observed.

    Returns:
        Any: An observation received after applying
            the transformations.
    """
    cards = obs['cards']
    if not isinstance(cards, np.ndarray):
        obs['cards'] = np.array(cards)

    for transform in obs_transforms:
        obs = transform(obs, player_index, uuid)
    return obs
