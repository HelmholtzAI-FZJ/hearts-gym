"""
Vectorized multi-agent Heats environment using multiple processes for
additional speed.
"""

from multiprocessing.pool import ThreadPool
from typing import Any, Iterator, List, Tuple

from hearts_gym.envs.hearts_env import (
    Action,
    HeartsEnv,
    MultiInfo,
    MultiIsDone,
    MultiObservation,
    MultiReward,
)
from hearts_gym import utils
from hearts_gym.utils.mock_pool import MockPool


class VecHeartsEnv(HeartsEnv):
    """Vectorized multi-agent Hearts environment.

    The list of environments is sharded along a given number of processes.
    """

    def __init__(
            self,
            envs: List[HeartsEnv],
            num_procs: int = utils.get_num_cpus() - 1,
    ) -> None:
        """Construct a vectorized Hearts environment over the
        given environments.

        Args:
            envs (List[HeartsEnv]): The environments to act in
                in parallel.
            num_procs (int): Amount of processes to use for parallel
                vectorized processing. If 0 or 1, do not start
                extra processes.
        """
        self.num_envs = len(envs)
        self._envs = envs
        self._first_env = envs[0]

        self._pool: ThreadPool
        if num_procs <= 1:
            self._pool = MockPool()
        else:
            self._pool = ThreadPool(processes=num_procs)

    def __getattr__(self, name: str) -> Any:
        """Return the attribute with the given name from the first
        environment we interact with.

        Args:
            name (str): Name of the attribute to retrieve from
                the environment.

        Returns:
            Any: Value of the retrieved attribute.
        """
        return getattr(self._first_env, name)

    def __len__(self) -> int:
        """Return the amount of environments processed in parallel.

        Returns:
            int: Amount of environments processed in parallel.
        """
        return self.num_envs

    def __getitem__(self, key: int) -> HeartsEnv:
        """Return the hearts environment at the given position.

        Args:
            key (int): Index of the hearts environment to return.

        Returns:
            HeartsEnv: The environment at the given index."""
        return self._envs[key]

    def __iter__(self) -> Iterator:
        """Return an iterator over the environments this
        vectorizes over.

        Returns:
            Iterator: Iterator over the internal environments.
        """
        return iter(self._envs)

    def get_envs(self) -> List[HeartsEnv]:
        """Return a list of the environments this vectorizes over.

        Returns:
            List[HeartsEnv]: List of internal environments.
        """
        return self._envs

    def terminate_pool(self) -> None:
        """Terminate the thread pool."""
        self._pool.terminate()
        self._pool.join()

    def step(  # type: ignore[override]
            self,
            actions: Iterator[Action],
    ) -> List[Tuple[
        MultiObservation,
        MultiReward,
        MultiIsDone,
        MultiInfo,
    ]]:
        """Take a step in each environment, using the actions in order.
        Return each environment's information in order.

        Args:
            actions (Iterator[Action]): Actions to execute, one for
                each environment.

        Returns:
            List[Tuple[
                MultiObservation,
                MultiReward,
                MultiIsDone,
                MultiInfo,
            ]]: Environment information after stepping, one for
                each environment.
        """
        data = self._pool.starmap(
            lambda env, action: env.step({env.active_player_index: action}),
            zip(self._envs, actions),
        )
        assert len(data) == self.num_envs, \
            'amount of actions did not match amount of environments'
        return data

    def reset(self) -> List[MultiObservation]:  # type: ignore[override]
        """Reset all environments and return the observations.

        Returns:
            List[MultiObservation]: Environment observation after
                resetting, one for each environment.
        """
        return self._pool.map(lambda env: env.reset(), self._envs)
