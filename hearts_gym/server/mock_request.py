"""
Client request acting like a socket.

"Sends" random actions to the server.
"""

import random
import socket
from typing import List, Optional

from hearts_gym import HeartsEnv
from hearts_gym.envs.card_deck import Seed
from hearts_gym.server import utils as server_utils


class MockRequest(socket.socket):
    """Client request acting like a socket.

    "Sends" random actions to the server, simulating a randomly acting
    client agent.
    """

    def __init__(
            self,
            envs: List[HeartsEnv],
            player_index: int,
            seed: Seed = None,
    ) -> None:
        """Construct a mock request interacting with the
        given environments.

        Args:
            envs (List[HeartsEnv]): Environments the clients acts in.
            player_index (int): Index of the client in the environment.
            seed (Seed): Random number generator seed for action sampling.
        """
        self._envs = envs
        self._player_index = player_index
        self._rng = random.Random(seed)

        self._ok_msg = server_utils.OK_MSG

    def sendall(self, bytes: bytes, flags: int = 0) -> None:
        """Overridden with NOP for API compatibility."""
        pass

    def get_actions(self) -> bytes:
        """Return random legal action for the environments the client
        interacts with.

        Args:
            bufsize (int): Ignored.
            flags (int): Ignored.

        Returns:
            bytes: Random legal actions.
        """
        # Also catches an uninitialized game.
        if self._envs[0].game.is_done():
            return self._ok_msg

        actions = [
            self._rng.choice(env.get_legal_actions())
            for env in self._envs
            if env.active_player_index == self._player_index
        ]
        return server_utils.encode_actions(actions)

    def recv(self, bufsize: int, flags: int = 0) -> bytes:
        return self.get_actions()

    def settimeout(self, value: Optional[float]) -> None:
        """Overridden with NOP for API compatibility."""
        pass

    def gettimeout(self) -> None:
        """Overridden with NOP for API compatibility."""
        pass
