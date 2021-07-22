"""
TCP socket server to host Hearts games.
"""

import logging
import math
from multiprocessing.pool import ThreadPool
import os
import socket
from socketserver import BaseRequestHandler, BaseServer, TCPServer
from threading import RLock, Thread
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from gym.utils import seeding
import numpy as np

from hearts_gym import utils
from hearts_gym.envs.hearts_env import HeartsEnv
from hearts_gym.utils.typing import (
    Action,
    GymSeed,
    MultiInfo,
    MultiIsDone,
    MultiObservation,
    MultiReward,
)
from hearts_gym.envs.hearts_game import HeartsGame
from hearts_gym.envs.vec_hearts_env import VecHeartsEnv
from hearts_gym.server import utils as server_utils
from hearts_gym.server.mock_request import MockRequest
from hearts_gym.server.client import Client
from hearts_gym.server.utils import Address, Request

SERVER_ADDRESS = '127.0.0.1'
"""Address to host the server at.

"127.0.0.1" or "localhost" will host a local server.
"""
PORT = 6087
"""Port to use for the server."""


def next_power(value: int, base: int) -> int:
    """Return the next power of a given base for a given value.

    Args:
        value (int): Value to get the next power for.
        base (int): Base of the powers.

    Returns:
        int: The next power of the given base after value.
    """
    prev_pow_exp = int(math.log(value, base))
    next_pow = base ** (prev_pow_exp + 1)
    return next_pow


class HeartsServer(TCPServer):
    """TCP server to host Hearts games.

    Will wait until enough players have joined and then continually run
    games, keeping track of several statistics.

    When players leave during games or players have been waiting for too
    long, simulated agents are inserted.

    Games are played in parallel; this means clients will receive
    different batch sizes due to the nature of the game.
    """

    OK_TIMEOUT_SEC = 2
    """How long clients have to respond with an 'OK' message until they are
    automatically disconnected.
    """
    SETUP_OK_TIMEOUT_SEC = 20
    """How long clients have to respond with an 'OK' message to the metadata
    message until they are automatically disconnected after.

    We give clients more time here so they can get set up.
    """

    PRINT_INTERVAL_SEC = 10
    """How long to wait between messages to a waiting client."""

    RANDOM_AGENT_NAME = b'__RAND'
    """Name that will result in a randomly acting agent."""

    def __init__(
            self,
            server_address: Address,
            RequestHandlerClass: Callable[
                [Request, Address, BaseServer],
                BaseRequestHandler,
            ],
            *,
            num_players: int = 4,
            deck_size: int = 52,
            game: Optional[HeartsGame] = None,
            mask_actions: bool = HeartsEnv.MASK_ACTIONS_DEFAULT,
            seed: GymSeed = None,
            num_parallel_games: int = 1024,
            num_procs: int = utils.get_num_cpus() - 1,
            max_num_games: Optional[int] = None,
            accept_repeating_client_addresses: bool = True,
            wait_duration_sec: Optional[int] = None,
            bind_and_activate: bool = True,
    ) -> None:
        """Construct a Hearts server.

        See also `TCPServer.__init__`.

        Args:
            server_address (Address): Address and port to host the
                server at.
            RequestHandlerClass (Callable[
                [Request, Address, BaseServer],
                BaseRequestHandler,
            ]): Request handler class to use.
            num_players (int): Amount of players. Only used if `game` is
                not `None`.
            deck_size (int): Amount of cards in the deck. Only used if
                `game` is not `None`.
            game (Optional[HeartsGame]): A pre-initialized game simulator.
            mask_actions (bool): Whether to enable action masking,
                parameterizing the action space.
            seed (GymSeed): Random number generator base seed.
            num_parallel_games (int): How many games to play in parallel.
                This is also approximately the batch size for the
                observations times four. The batch size may be anywhere
                between zero and this number.
            num_procs (int): How many processes to use for playing the
                parallel games.
            max_num_games (Optional[int]): After how many games to
                automatically disconnect all clients. If `None`, keep
                connected indefinitely.
            accept_repeating_client_addresses (bool): Whether clients
                are allowed to connect multiple times from the same
                address (only changing the port they connect from).
            wait_duration_sec (Optional[int]): How long to wait after the
                first player has connected until the remaining spots are
                filled with randomly acting agents. If `None`,
                wait indefinitely.
            bind_and_activate (bool): Whether to automatically bind and
                activate the server upon construction.
        """
        assert num_parallel_games > 0, 'must have at least one game'
        assert num_procs > 0, 'must have at least one process'
        assert (
            max_num_games is None
            or max_num_games % num_parallel_games == 0
        ), (
            'maximum number of games must be divisible by number of '
            'parallel games'
        )

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(os.getenv('LOG_LEVEL', 'NOTSET').upper())

        if num_procs > num_parallel_games:
            num_procs = num_parallel_games
            self.logger.warning(
                f'Warning: set `num_procs = {num_parallel_games}`; '
                f'cannot have more processes than games.'
            )

        self._accept_repeating_client_address = \
            accept_repeating_client_addresses
        self._wait_duration_sec = wait_duration_sec
        self._max_num_clients = num_players

        # TODO Allow setting max batch size per client
        self.num_parallel_games = num_parallel_games
        self.max_num_games = max_num_games

        self._client_change_lock = RLock()
        self._waiter_threads: Dict[int, Thread] = {}

        self.clients: Dict[int, Client] = {}

        self.is_closed = False
        self.needs_reset = True
        self.num_games = 0
        self.stats: List[Tuple[List[int], List[int]]] = []
        self.num_illegals: List[int] = [0] * num_players
        self.total_penalties = [0] * num_players
        self.total_placements = [[0] * num_players for _ in range(num_players)]

        self.envs = VecHeartsEnv(
            [
                HeartsEnv(
                    num_players=num_players,
                    deck_size=deck_size,
                    game=game,
                    mask_actions=mask_actions,
                    seed=self._add_to_seed(seed, i),
                )
                for i in range(self.num_parallel_games)
            ],
            num_procs=num_procs,
        )

        super().__init__(
            server_address,
            RequestHandlerClass,
            bind_and_activate,
        )
        self.print_log(f'Server started on {server_address}.')

    def print_log(self, message: str, log_level: int = logging.INFO):
        """Print and log the given message.

        Args:
            message (str): What to print and log.
            log_level (int): Logging level indicating the importance of
                the log.
        """
        print(message)
        self.logger.log(log_level, message)

    @staticmethod
    def _add_to_seed(seed: GymSeed, integer: int) -> GymSeed:
        """Return the seed with an integer added to it.

        Used to obtain different seeds.

        Args:
            seed (GymSeed): Base random number generator seed.
            integer (int): Integer to modify the seed with.

        Returns:
            GymSeed: Seed modified according to the given integer.
        """
        if isinstance(seed, int):
            return seed + integer

        if isinstance(seed, str):
            return str(integer) + seed

        if seed is None:
            return seed

        raise TypeError('unknown seed type')

    def _has_client_address(self, client_address: Address) -> bool:
        """Return whether the given client address is already registered.

        Note that this ignores the port portion of the address.

        Args:
            client_address (Address): Client address to query.

        Returns:
            bool: Whether the client address is already registered.
        """
        registered_addresses = (
            client.address[0]
            for client in self.clients.values()
        )
        return client_address[0] in registered_addresses

    def verify_request(  # type: ignore[override]
            self,
            request: Request,
            client_address: Address,
    ) -> bool:
        self.logger.info(f'Verifying {client_address}...')
        with self._client_change_lock:
            if (
                    len(self.clients) >= self._max_num_clients
                    or (
                        not self._accept_repeating_client_address
                        and self._has_client_address(client_address)
                    )
            ):
                try:
                    data = server_utils.encode_data('Game is full already.')
                    request.sendall(data)
                except Exception:
                    pass
                self.logger.info('Rejected.')
                return False

        self.logger.info('Accepted.')
        return True

    def find_free_index(self) -> Optional[int]:
        """Return the first free index or `None` if the maximum number
        of players has already been reached.

        Returns:
            Optional[int]: First free index or `None` if there is none.
        """
        return next(
            (
                i
                for i in range(self._max_num_clients)
                if i not in self.clients
            ),
            None,
        )

    def register_client(
            self,
            request: Request,
            client_address: Address,
            player_index: Optional[int] = None,
    ) -> Optional[Client]:
        """Register the given client and return it if successful.
        Otherwise, return `None`.

        Args:
            request (Request): Socket/request of the client.
            client_address (Address): Address of the client.
            player_index (Optional[int]): Index to register the player
                at; if `None`, take first free one.

        Returns:
            Optional[Client]: The registered client or `None`.
        """
        if isinstance(request, tuple):
            request = request[1]
        with self._client_change_lock:
            # Sanity check just in case.
            if (
                    len(self.clients) >= self._max_num_clients
                    or (
                        not self._accept_repeating_client_address
                        and self._has_client_address(client_address)
                    )
            ):
                return None

            if player_index is None:
                player_index = self.find_free_index()
            assert player_index is not None

            client = Client(player_index, request, client_address)
            self.clients[player_index] = client
            return client

    def register_bot(
            self,
            client_index: Optional[int] = None,
    ) -> Optional[Client]:
        """Register a simulated agent and return it if successful.
        Otherwise, return `None`.

        Args:
            client_index (Optional[int]): Index to register the
                simulated agent at. If `None`, use the first free index.

        Returns:
            Optional[Client]: The registered client or `None`.
        """
        if client_index is None:
            client_index = self.find_free_index()
        assert client_index is not None

        client = self.register_client(
            MockRequest(
                self.envs.get_envs(),
                client_index,
                seed=seeding.hash_seed(),
            ),
            ('mock-client', client_index),
            client_index,
        )

        if client is not None:
            self.logger.info(f'Registered bot at index {client_index}.')
        return client

    def shutdown_request(  # type: ignore[override]
            self,
            request: Union[Request, MockRequest],
    ) -> None:
        if isinstance(request, MockRequest):
            return
        super().shutdown_request(request)  # type: ignore[misc]

    def unregister_client(
            self,
            client: Client,
            replace_with_bot: bool,
    ) -> None:
        """Unregister the given client.

        Args:
            client (Client): Client to unregister.
            replace_with_bot (bool): Whether to replace a lost client with a
                simulated agent.
        """
        with self._client_change_lock:
            client_index = client.player_index

            self.shutdown_request(client.request)  # type: ignore[attr-defined]
            client.is_registered = False
            del self.clients[client_index]
            if client_index in self._waiter_threads:
                del self._waiter_threads[client_index]

            if replace_with_bot:
                self.register_bot(client_index)

    def _receive_shard(
            self,
            client: Client,
            max_receive_bytes: int,
            timeout_sec: int,
            replace_with_bot: bool,
            client_error_msg: str,
    ) -> Optional[bytes]:
        """Return a message received from the client in a failsafe way.
        If something went wrong, return `None`.

        Args:
            client (Client): Client to receive the data from.
            max_receive_bytes (int): How many bytes to receive
                at maximum.
            timeout_sec (int): How long to wait for the message at
                maximum.
            replace_with_bot (bool): Whether to replace a lost client
                with a simulated agent.
            client_error_msg (str): Error message to send to the client
                upon failure.

        Returns:
            Optional[bytes]: The message received or `None` if there was
                an error.
        """
        request = client.request
        prev_timeout = request.gettimeout()
        request.settimeout(timeout_sec)

        try:
            data = request.recv(max_receive_bytes)
            if data == b'' or data is None:
                raise ValueError('received empty data')
        except socket.timeout:
            request.settimeout(prev_timeout)
            self.send_failable(client, client_error_msg)
            self.logger.warning(
                f'Client {client.address} did not respond in time.')
            self.unregister_client(client, replace_with_bot)
            return None
        except Exception:
            self.logger.warning(f'Lost client {client.address}.')
            self.unregister_client(client, replace_with_bot)
            return None

        request.settimeout(prev_timeout)
        return data

    def _receive_msg_length(
            self,
            client: Client,
            max_receive_bytes: int,
            timeout_sec: int,
            replace_with_bot: bool,
            client_error_msg: str,
    ) -> Optional[Tuple[int, bytes]]:
        """Return the expected length of a message received from the
        client in a failsafe way. If something went wrong,
        return `None`.

        To be more efficient, receive more data than necessary. Any
        additional data is returned.

        Args:
            client (Client): Client to receive the data from.
            max_receive_bytes (int): How many bytes to receive
                at maximum.
            timeout_sec (int): How long to wait for the message at
                maximum.
            replace_with_bot (bool): Whether to replace a lost client
                with a simulated agent.
            client_error_msg (str): Error message to send to the client
                upon failure.

        Returns:
            Optional[Tuple[int, bytes]]: Amount of bytes in the rest of
                the message and the extraneous part of message data
                received, or `None` if there was an error.
        """
        data_shard = self._receive_shard(
            client,
            max_receive_bytes,
            timeout_sec,
            replace_with_bot,
            client_error_msg,
        )
        if data_shard is None:
            return None
        total_num_received_bytes = len(data_shard)
        data = [data_shard]
        length_end = data_shard.find(server_utils.MSG_LENGTH_SEPARATOR)

        while (
                length_end == -1
                and (
                    total_num_received_bytes
                    < server_utils.MAX_MSG_PREFIX_LENGTH
                )
        ):
            data_shard = self._receive_shard(
                client,
                max_receive_bytes,
                timeout_sec,
                replace_with_bot,
                client_error_msg,
            )
            if data_shard is None:
                return None
            total_num_received_bytes += len(data_shard)
            data.append(data_shard)
            length_end = data_shard.find(server_utils.MSG_LENGTH_SEPARATOR)

        if length_end == -1:
            self.send_failable(
                client,
                (
                    f'Please prefix messages with unknown length with '
                    f'their length and '
                    f'"{server_utils.MSG_LENGTH_SEPARATOR.decode()}".'
                )
            )
            self.logger.warning(
                f'Client {client.address} did not send message length. '
                f'Closing connection...'
            )
            self.unregister_client(client, replace_with_bot)
            return None

        length_end += total_num_received_bytes - len(data_shard)
        data = b''.join(data)
        try:
            msg_length = int(data[:length_end])
        except ValueError:
            self.send_failable(
                client,
                (
                    f'Please prefix messages with unknown length with '
                    f'only their length and '
                    f'"{server_utils.MSG_LENGTH_SEPARATOR.decode()}".'
                )
            )
            self.logger.warning(
                f'Client {client.address} sent garbled message length. '
                f'Closing connection...'
            )
            self.unregister_client(client, replace_with_bot)
            return None

        extra_data = data[length_end + len(server_utils.MSG_LENGTH_SEPARATOR):]

        return msg_length, extra_data

    def receive_name(
            self,
            client: Client,
            timeout_sec: Optional[int] = None,
    ) -> bool:
        """Wait for a message containing a name from the given client.
        Return whether the message was correctly received.

        Args:
            client (Client): Client to receive the name from.
            timeout_sec (Optional[int]): How long to wait at maximum. If
                `None`, use `HeartsServer.OK_TIMEOUT_SEC`.

        Returns:
            bool: Whether a name was correctly received in the allowed
                time frame.
        """
        if timeout_sec is None:
            timeout_sec = self.OK_TIMEOUT_SEC

        receive_msg_length_result = self._receive_msg_length(
                client,
                Client.MAX_NAME_BYTES,
                timeout_sec,
                False,
                (
                    f'Please send a name with a length of '
                    f'{Client.MAX_NAME_BYTES} bytes at maximum; '
                    f'closing connection...'
                ),
        )
        if receive_msg_length_result is None:
            return False
        data_shard: Optional[bytes]

        msg_length, data_shard = receive_msg_length_result
        if msg_length > Client.MAX_NAME_BYTES:
            self.send_failable(
                client,
                'Declared name length is too long.',
            )
            self.logger.warning(
                f'Client {client.address} declared too long name. '
                f'Closing connection...'
            )
            self.unregister_client(client, False)
            return False

        total_num_received_bytes = len(data_shard)
        data = [data_shard]
        while total_num_received_bytes < msg_length:
            data_shard = self._receive_shard(
                client,
                Client.MAX_NAME_BYTES - total_num_received_bytes,
                timeout_sec,
                False,
                (
                    f'Please send a name with a length of '
                    f'{Client.MAX_NAME_BYTES} bytes at maximum; '
                    f'closing connection...'
                ),
            )
            if data_shard is None:
                return False
            total_num_received_bytes += len(data_shard)
            data.append(data_shard)

        if total_num_received_bytes != msg_length:
            self.send_failable(
                client,
                'Message had a different length than declared.',
            )
            self.logger.warning(
                f'Client {client.address} declared different message length. '
                f'Closing connection...'
            )
            self.unregister_client(client, False)
            return False

        data = b''.join(data)
        # We assume the client does not want to set a name.
        if data == server_utils.OK_MSG:
            return True
        if data == self.RANDOM_AGENT_NAME:
            self.unregister_client(client, True)
            return True

        with self._client_change_lock:
            client.set_name(data)
            other_names = [
                other_client.name
                for other_client in self.clients.values()
                if other_client is not client
            ]
            i = 2
            while client.name in other_names:
                client.set_name(data + f' ({i})'.encode())
                i += 1

        self.logger.info(
            f'Client {client.address} is now called "{client.name}".')
        return True

    def _receive_ok(
            self,
            client: Client,
            timeout_sec: Optional[int],
            replace_with_bot: bool,
    ) -> bool:
        """Wait for an 'OK' message from the given client. Return
        whether the message was correctly received.

        Upon error, optionally replace the client with a
        simulated agent.

        Args:
            client (Client): Client to receive the 'OK' from.
            timeout_sec (Optional[int]): How long to wait at maximum. If
                `None`, use `HeartsServer.OK_TIMEOUT_SEC`.
            replace_with_bot (bool): Whether to replace a lost client
                with a simulated agent.

        Returns:
            bool: Whether the message was correctly received in the
                allowed time frame.
        """
        if timeout_sec is None:
            timeout_sec = self.OK_TIMEOUT_SEC

        data_shard = self._receive_shard(
            client,
            len(server_utils.OK_MSG),
            timeout_sec,
            replace_with_bot,
            (
                f'Please respond with "{server_utils.OK_MSG.decode()}"; '
                f'closing connection...'
            ),
        )
        if data_shard is None:
            return False
        total_num_received_bytes = len(data_shard)
        data = [data_shard]
        while total_num_received_bytes < len(server_utils.OK_MSG):
            data_shard = self._receive_shard(
                client,
                len(server_utils.OK_MSG) - total_num_received_bytes,
                timeout_sec,
                replace_with_bot,
                (
                    f'Please respond with "{server_utils.OK_MSG.decode()}"; '
                    f'closing connection...'
                ),
            )
            if data_shard is None:
                return False
            total_num_received_bytes += len(data_shard)
            data.append(data_shard)

        data = b''.join(data)
        if data == server_utils.OK_MSG:
            return True

        self.send_failable(
            client,
            (
                f'Please respond with "{server_utils.OK_MSG.decode()}"; '
                f'closing connection...'
            ),
        )
        self.unregister_client(client, replace_with_bot)
        return False

    def receive_ok(
            self,
            client: Client,
            timeout_sec: Optional[int] = None,
    ) -> bool:
        """Wait for an 'OK' message from the given client. Return
        whether the message was correctly received.

        Args:
            client (Client): Client to receive the 'OK' from.
            timeout_sec (Optional[int]): How long to wait at maximum. If
                `None`, use `HeartsServer.OK_TIMEOUT_SEC`.

        Returns:
            bool: Whether the message was correctly received in the
                allowed time frame.
        """
        return self._receive_ok(client, timeout_sec, False)

    def receive_ok_replacing(
            self,
            client: Client,
            timeout_sec: Optional[int] = None,
    ) -> bool:
        """Wait for an 'OK' message from the given client. Return
        whether the message was correctly received.

        Upon error, replace the client with a simulated agent.

        Args:
            client (Client): Client to receive the 'OK' from.
            timeout_sec (Optional[int]): How long to wait at maximum. If
                `None`, use `HeartsServer.OK_TIMEOUT_SEC`.

        Returns:
            bool: Whether the message was correctly received in the
                allowed time frame.
        """
        return self._receive_ok(client, timeout_sec, True)

    def _send_hello(
            self,
            client: Client,
    ) -> None:
        """Receive a name, then send a greeting and server metadata
        message to the given client.

        Args:
            client (Client): Client to receive a name from and send
                hello to.
        """
        max_num_clients = self._max_num_clients
        num_clients = len(self.clients)

        message = (
            f'{client.name} connected to server; '
            f'{num_clients}/{max_num_clients} players connected'
        )

        if num_clients < max_num_clients:
            message = message + '...'
        else:
            message = message + '!'

        self.send_failable(client, message)

        if not self.receive_ok(client):
            return

        env = self.envs[0]
        metadata = {
            'player_index': client.player_index,
            'num_players': env.num_players,
            'deck_size': env.deck_size,
            'mask_actions': env.mask_actions,
            'max_num_games': self.max_num_games,
            'num_parallel_games': self.num_parallel_games,
        }

        self.send_failable(client, metadata)
        self.receive_ok(client, self.SETUP_OK_TIMEOUT_SEC)

    def _send_failable(
            self,
            client: Client,
            data: Any,
            replace_with_bot: bool,
    ) -> bool:
        """Send the given data to the given client, handling failure cases.
        Return whether the message was correctly sent.

        Upon error, optionally replace the client with a simulated agent.

        Args:
            client (Client): Client to send the data to.
            data (Any): Data to send to the client.
            replace_with_bot (bool): Whether to replace a lost client with a
                simulated agent.

        Returns:
            bool: Whether the data was correctly sent to the client.
        """
        try:
            if not isinstance(data, bytes):
                data = server_utils.encode_data(data)
            client.request.sendall(data)
            return True
        except Exception:
            self.logger.warning(f'Lost client {client.address}.')
            self.unregister_client(client, replace_with_bot)
            return False

    def send_failable(
            self,
            client: Client,
            data: Any,
    ) -> bool:
        """Send the given data to the given client, handling failure cases.
        Return whether the message was correctly sent.

        Args:
            client (Client): Client to send the data to.
            data (Any): Data to send to the client.

        Returns:
            bool: Whether the data was correctly sent to the client.
        """
        return self._send_failable(client, data, False)

    def send_failable_replacing(
            self,
            client: Client,
            data: Any,
    ) -> bool:
        """Send the given data to the given client, handling failure cases.
        Return whether the message was correctly sent.

        Upon error, replace the client with a simulated agent.

        Args:
            client (Client): Client to send the data to.
            data (Any): Data to send to the client.

        Returns:
            bool: Whether the data was correctly sent to the client.
        """
        return self._send_failable(client, data, True)

    def fill_most_remaining(self) -> None:
        """Fill most remaining free spots with randomly acting agents,
        keeping one spot free.
        """
        with self._client_change_lock:
            client_index = self.find_free_index()
            while len(self.clients) < self._max_num_clients - 1:
                self.register_bot(client_index)
                client_index = self.find_free_index()

    def _wait_for_players(
            self,
            client: Client,
            is_first_client: bool,
    ) -> None:
        """Let the given client wait until enough clients have connected.
        During the waiting, periodically send messages.

        Supposed to be started in a new thread.

        See also `self._start_waiter_thread`.

        Args:
            client (Client): Client that waits.
            is_first_client (bool): Whether this is the first client
                that connected.
        """
        max_num_clients = self._max_num_clients
        num_clients = len(self.clients)

        start_time = time.time()
        last_print_time = start_time
        while not self.is_closed and client.is_registered:
            prev_num_clients = num_clients
            num_clients = len(self.clients)

            if prev_num_clients != num_clients:
                message = f'{num_clients}/{max_num_clients} players connected'
                if num_clients >= max_num_clients:
                    self.send_failable(client, message + '!')
                    self.receive_ok(client)
                    break
                send_success = self.send_failable(client, message + '...')
                if (
                        not send_success
                        or not self.receive_ok(client)
                ):
                    break

            time.sleep(0.1)
            curr_time = time.time()

            if (
                    self._wait_duration_sec is not None
                    and is_first_client
                    and curr_time - start_time > self._wait_duration_sec
            ):
                self.fill_most_remaining()

                def simulate_client():
                    with server_utils.create_client() as tmp_client:
                        tmp_client.connect(self.server_address)
                        server_utils.send_name(
                            tmp_client, self.RANDOM_AGENT_NAME.decode())
                        time.sleep(self.PRINT_INTERVAL_SEC)

                Thread(target=simulate_client).start()

                self.print_log('Filled remaining spots with bots.')
                continue

            if curr_time - last_print_time < self.PRINT_INTERVAL_SEC:
                continue

            self.print_log('Waiting...', logging.DEBUG)
            last_print_time = curr_time
            send_success = self.send_failable(
                client, 'Waiting for more players...')
            if (
                    not send_success
                    or not self.receive_ok(client)
            ):
                break

    def _start_waiter_thread(
            self,
            client: Client,
    ) -> None:
        """Let the given client wait until enough clients have connected in
        another thread.

        See also `self._wait_for_players`.

        Args:
            client (Client): Client that should wait.
        """
        if not client.is_registered:
            return
        self.logger.info(f'Starting waiter thread for {client.address}...')
        with self._client_change_lock:
            thread = Thread(
                target=self._wait_for_players,
                args=(client, len(self.clients) == 1),
            )
            self._waiter_threads[client.player_index] = thread
        thread.start()
        self.logger.info('Thread started.')

    def _join_waiters(self) -> None:
        """Block until all waiter threads have completed."""
        for thread in self._waiter_threads.values():
            thread.join()
        # No need to worry about locking anymore.
        self._waiter_threads.clear()

    def process_request(  # type: ignore[override]
            self,
            request: Request,
            client_address: Address,
    ) -> None:
        self.logger.info(f'Registering {client_address}...')
        client = self.register_client(request, client_address)
        if client is None:
            self.logger.warning('Failed.')
            self.shutdown_request(request)  # type: ignore[attr-defined]
            return

        self.print_log(
            f'Registered {client_address} at index {client.player_index}.')

        successful = self.receive_name(client)
        if not successful:
            return

        if client.is_registered:
            self._send_hello(client)

        if len(self.clients) < self._max_num_clients:
            self._start_waiter_thread(client)
            return

        self._join_waiters()

        # We lost a client while joining threads.
        if len(self.clients) < self._max_num_clients:
            with self._client_change_lock:
                for client in self.clients.values():
                    self._start_waiter_thread(client)
            return

        self.print_log('Starting game loop...')
        self.finish_request(
            client.request,  # type: ignore[arg-type]
            client.address,
        )

    def server_close(self) -> None:
        self.is_closed = True
        self._join_waiters()
        super().server_close()


class HeartsRequestHandler(BaseRequestHandler):
    MAX_BYTES_PER_SEPARATE_ACTION = 2
    """One individual action can be this many bytes long."""
    MAX_BYTES_PER_ACTION_UNPADDED = (
        MAX_BYTES_PER_SEPARATE_ACTION + len(server_utils.ACTION_SEPARATOR))
    """One action can be `MAX_BYTES_PER_SEPARATE_ACTION` bytes long.
    In addition, we have the comma as a separator.
    """
    MAX_BYTES_PER_ACTION = next_power(MAX_BYTES_PER_ACTION_UNPADDED, 2)
    """`MAX_BYTES_PER_SEPARATE_ACTION_UNPADDED` padded towards a power
    of two.
    """
    assert (
        math.log(MAX_BYTES_PER_ACTION, 2) % 1 == 0
    ), '`MAX_BYTES_PER_ACTION` must be a power of two'

    PARSE_FAIL_TOLERANCE = 2
    ACTION_TIMEOUT_SEC = 1
    OK_TIMEOUT_SEC = 2

    @staticmethod
    def calculate_max_receive_bytes(num_parallel_games: int) -> int:
        """Return the maximum amount of bytes that may be sensibly
        received from a client.

        Args:
            num_parallel_games (int): Amount of games played in parallel.

        Returns:
            int: Maximum amount of bytes that will be received from
                a client.
        """
        return num_parallel_games * HeartsRequestHandler.MAX_BYTES_PER_ACTION

    def setup(self) -> None:
        assert isinstance(self.server, HeartsServer), \
            'received unknown server type'
        self.server: HeartsServer
        self.max_receive_bytes = \
            self.calculate_max_receive_bytes(self.server.num_parallel_games)
        self._max_shard_receive_bytes = min(
            self.max_receive_bytes,
            server_utils.MAX_RECEIVE_BYTES,
        )
        self.max_prefix_len = (
            len(str(self.max_receive_bytes))
            + len(server_utils.MSG_LENGTH_SEPARATOR)
        )
        for client in self.server.clients.values():
            client.request.settimeout(self.ACTION_TIMEOUT_SEC)

        num_players = len(self.server.clients)
        self._communicators = ThreadPool(processes=num_players)

    def _replace_with_bot(self, player_index: int) -> Tuple[Client, bytes]:
        """Replace the client at the given index with a randomly acting
        agent. Return the replacement client and its actions.

        Args:
            player_index (int): Which player/client to replace.

        Returns:
            Client: Client to receive data from in the future.
            bytes: Random actions in message form.
        """
        client = self.server.clients[player_index]
        self.server.unregister_client(client, True)
        # Get random actions from newly registered bot.
        client = self.server.clients[player_index]
        assert isinstance(client.request, MockRequest), \
            'replacing with bot failed'
        data = client.request.get_actions()
        return client, data

    def _receive_shard(
            self,
            player_index: int,
            client: Client,
    ) -> Tuple[Client, bytes, bool]:
        """Return a message received from the client in a failsafe way.

        Args:
            player_index (int): Which player we are getting the
                message from.
            client (Client): Client to receive the data from.

        Returns:
            Client: The original client or its replacement.
            bytes: The message or a default if there was an error.
            bool: Whether the message was received without an error.
        """
        try:
            data = client.request.recv(self._max_shard_receive_bytes)
            if data == b'' or data is None:
                raise ValueError('received empty data')
        except Exception:
            self.server.print_log(
                f'Lost client {client.address}.', logging.WARNING)
            client, data = self._replace_with_bot(player_index)
            return client, data, False

        return client, data, True

    def _receive_msg_length(
            self,
            player_index: int,
            client: Client,
    ) -> Tuple[Client, int, bytes]:
        """Return the expected length of a message received from the
        client in a failsafe way.

        To be more efficient, receive more data than necessary. Any
        additional data is returned.

        Args:
            player_index (int): Which player we are getting the
                message from.
            client (Client): Client to receive the data from.

        Returns:
            Client: The original client or its replacement.
            int: Amount of bytes in the rest of the message or a default
                if there was an error.
            bytes: Extraneous part of message data or a default if there
                was an error.
        """
        client, data_shard, successful = self._receive_shard(
            player_index, client)
        total_num_received_bytes = len(data_shard)
        data = [data_shard]
        length_end = data_shard.find(server_utils.MSG_LENGTH_SEPARATOR)
        while (
                successful
                and length_end == -1
                and total_num_received_bytes < self.max_prefix_len
        ):
            client, data_shard, successful = self._receive_shard(
                player_index, client)
            total_num_received_bytes = len(data_shard)
            data.append(data_shard)
            length_end = data_shard.find(server_utils.MSG_LENGTH_SEPARATOR)

        if not successful:
            total_num_received_bytes = len(data_shard)
            data = [data_shard]

        if length_end == -1:
            self.server.send_failable(
                client,
                (
                    f'Please prefix actions with their length and '
                    f'"{server_utils.MSG_LENGTH_SEPARATOR.decode()}".'
                )
            )
            self.server.logger.warning(
                f'Client {client.address} did not send action length. '
                f'Closing connection...'
            )
            client, data_shard = self._replace_with_bot(player_index)

            total_num_received_bytes = len(data_shard)
            data = [data_shard]
            length_end = data_shard.find(server_utils.MSG_LENGTH_SEPARATOR)

        length_end += total_num_received_bytes - len(data_shard)
        data = b''.join(data)
        try:
            msg_length = int(data[:length_end])
        except ValueError:
            self.server.send_failable(
                client,
                (
                    f'Please prefix actions with only their length and '
                    f'"{server_utils.MSG_LENGTH_SEPARATOR.decode()}".'
                )
            )
            self.server.logger.warning(
                f'Client {client.address} sent garbled action length. '
                f'Closing connection...'
            )
            client, data = self._replace_with_bot(player_index)

            total_num_received_bytes = len(data)
            length_end = data.find(server_utils.MSG_LENGTH_SEPARATOR)

        extra_data = data[length_end + len(server_utils.MSG_LENGTH_SEPARATOR):]

        return client, msg_length, extra_data

    def _parse_message(
            self,
            player_index: int,
            client: Client,
    ) -> List[Action]:
        """Parse a message received by the given client.

        Args:
            player_index (int): Which player we are getting the
                message from.
            client (Client): Client to receive the message from.

        Returns:
            List[Action]: Actions contained in the message or a default
                if there was an error.
        """
        for _ in range(self.PARSE_FAIL_TOLERANCE + 1):
            client, msg_length, data_shard = self._receive_msg_length(
                player_index, client)

            successful = True
            total_num_received_bytes = len(data_shard)
            data = [data_shard]
            while (
                    successful
                    and total_num_received_bytes < msg_length
            ):
                client, data_shard, successful = self._receive_shard(
                    player_index, client)
                total_num_received_bytes += len(data_shard)
                data.append(data_shard)

            if successful and total_num_received_bytes != msg_length:
                self.server.send_failable(
                    client,
                    'Actions had a different length than declared.',
                )
                self.server.logger.warning(
                    f'Client {client.address} declared different actions '
                    f'length. Closing connection...'
                )
                client, data_shard = self._replace_with_bot(player_index)
                successful = False

            if not successful:
                data_shard = data_shard.partition(
                    server_utils.MSG_LENGTH_SEPARATOR)[2]
                total_num_received_bytes = len(data_shard)
                data = [data_shard]

            data = b''.join(data)

            self.server.logger.debug(f'Received data:\n{data.decode()}')
            try:
                actions = server_utils.decode_actions(data)
            except Exception:
                self.server.logger.warning('Error parsing data; ignoring...')
                self.server.logger.warning(f'Data that errored:\n{str(data)}')
                self.server.send_failable_replacing(
                    client,
                    (
                        f'Actions were malformed. Please submit at maximum '
                        f'{self.max_receive_bytes} bytes which are your '
                        f'actions (a comma-separated list of integers) as a '
                        f'string. Do not encode the message in any other '
                        f'form. To submit no action, submit just a comma.'
                    ),
                )
                continue
            return actions

        if not self.server.envs.mask_actions:
            return [
                0
                for env in self.server.envs
                if env.active_player_index == player_index
            ]
        return [
            env.get_legal_actions()[0]
            for env in self.server.envs
            if env.active_player_index == player_index
        ]

    def _parse_messages(self) -> List[List[Action]]:
        """Receive and parse messages for each client in parallel.
        Return the parsed actions.

        Returns:
            List[List[Action]]: Actions contained in the messages.
        """
        num_players = len(self.server.clients)
        clients = self.server.clients

        return self._communicators.starmap(
            self._parse_message,
            ((i, clients[i]) for i in range(num_players)),
        )

    @staticmethod
    def _tree_map(func: Callable[[Any], Any], tree: Any) -> Any:
        """Recursively map the given function over the given
        tree-like object.

        Note that strings are assumed to be atomic.

        Contrary to what the name suggests, the functionality is only
        very basic, supporting only a few Python primitive types.

        Args:
            func (Callable[[Any], Any]): Function to map over the tree.
            tree (Any): Tree-like to map over.

        Returns:
            Any: The tree-like with the mapping applied.
        """
        if isinstance(tree, dict):
            return {key: HeartsRequestHandler._tree_map(func, value)
                    for (key, value) in tree.items()}
        if isinstance(tree, list):
            return [HeartsRequestHandler._tree_map(func, value)
                    for value in tree]
        if isinstance(tree, tuple):
            return tuple(HeartsRequestHandler._tree_map(func, value)
                         for value in tree)
        return func(tree)

    @staticmethod
    def _to_primitive(data: Any) -> Any:
        """Convert the given data to a primitive Python type.

        Contrary to what the name suggests, the functionality is
        very basic.

        Args:
            data (Any): Object to convert to a primitive.

        Returns:
            Any: Primitive representation of `data`.
        """
        if isinstance(data, np.ndarray):
            return list(map(HeartsRequestHandler._to_primitive, data))
        if isinstance(data, np.integer):
            return int(data)
        if isinstance(data, np.floating):
            return float(data)
        if hasattr(data, '__dict__'):
            return vars(data)
        if hasattr(data, '__slots__'):
            return tuple(
                getattr(data, slot)
                if hasattr(data, slot)
                else None
                for slot in data.__slots__
            )
        return data

    def _encode_data(self, data: Any) -> bytes:
        """Return the given data encoded as a message between client
        and server.

        Args:
            data (Any): Data to encode for sending.

        Returns:
            bytes: Encoded data.
        """
        self.server.logger.debug(f'Data before tree map:\n{data}')
        data = self._tree_map(self._to_primitive, data)
        self.server.logger.debug(f'Data after tree map:\n{data}')
        return server_utils.encode_data(data)

    def _send_shard(
            self,
            player_index: int,
            data: Union[
                List[Tuple[int, MultiObservation]],
                List[Tuple[
                    int,
                    MultiObservation,
                    MultiReward,
                    MultiIsDone,
                    MultiInfo,
                ]],
            ],
    ) -> None:
        """Send the given data to the client corresponding to the
        given index.

        Args:
            player_index (int): Index of the client the shard should be
                sent towards.
            data (Union[
                List[Tuple[int, MultiObservation]],
                List[Tuple[
                    int,
                    MultiObservation,
                    MultiReward,
                    MultiIsDone,
                    MultiInfo,
                ]],
            ]): Data to send to the client.
        """
        data: bytes = self._encode_data(data)
        self.server.logger.debug(f'Sending to {player_index}:\n{str(data)}')
        client = self.server.clients[player_index]
        self.server.send_failable_replacing(client, data)

    def _distribute_return_data(
            self,
            return_data: Union[
                List[MultiObservation],
                List[Tuple[
                    MultiObservation,
                    MultiReward,
                    MultiIsDone,
                    MultiInfo,
                ]],
            ],
    ) -> None:
        """Distribute the given data among the connected clients.
        Send the partitioned data to each client in parallel.

        Distributing means to partition the data so that each client
        receives the data meant for it.

        Args:
            return_data (Union[
                List[MultiObservation],
                List[Tuple[
                    MultiObservation,
                    MultiReward,
                    MultiIsDone,
                    MultiInfo,
                ]],
            ]): Environment information received from the parallely
                processed environments.
        """
        num_players = len(self.server.clients)
        distributed_data: List[List[
            Union[
                Tuple[
                    int,
                    MultiObservation,
                ],
                Tuple[
                    int,
                    MultiObservation,
                    MultiReward,
                    MultiIsDone,
                    MultiInfo,
                ],
            ],
        ]] = [[] for _ in range(num_players)]

        for (i, (data, env)) in enumerate(zip(return_data, self.server.envs)):
            active_player_index = env.active_player_index
            active_player_data = distributed_data[active_player_index]
            if isinstance(data, tuple):
                active_player_data.append((i,) + data)
            else:
                active_player_data.append((i, data))

        self._communicators.starmap(
            self._send_shard,
            enumerate(distributed_data),
        )

    def _order_player_actions(
            self,
            player_actions: List[List[Action]],
    ) -> Iterator[Action]:
        """Return an iterator over the given actions for each player,
        sorted so each action matches the corresponding environment.

        Args:
            player_actions (List[List[Action]]): List of actions for
                each player, sorted by player indices.

        Returns:
            Iterator[Action]: Flattened iterator over the actions,
                sorted so the order corresponds to the order
                of environments.
        """
        offsets = [0] * len(player_actions)
        for env in self.server.envs:
            active_player_index = env.active_player_index
            offset = offsets[active_player_index]
            offsets[active_player_index] += 1
            yield player_actions[active_player_index][offset]

    @staticmethod
    def is_done(num_games: int, max_num_games: Optional[int]) -> bool:
        """Return whether the desired number of games have been played..

        Returns:
            bool: Whether the desired number of games have been played.
        """
        return max_num_games is not None and num_games >= max_num_games

    def _is_done(self) -> bool:
        """Return whether the server should disconnect its clients.

        Returns:
            bool: Whether the server should disconnect its clients.
        """
        return (
            self.is_done(self.server.num_games, self.server.max_num_games)
            # When we only have simulated agents left, we can just quit.
            or all(
                isinstance(client.request, MockRequest)
                for client in self.server.clients.values()
            )
        )

    def _index_to_name(self, player_index: int) -> str:
        """Return the name of the player with the given index.

        Args:
            player_index (int): Index of the player to query the
                name for.

        Returns:
            str: Name of the player.
        """
        return self.server.clients[player_index].name

    def handle(self) -> None:
        self.server.num_games = 0
        self.server.stats.clear()
        num_players = len(self.server.clients)
        for i in range(num_players):
            self.server.num_illegals[i] = 0
            self.server.total_penalties[i] = 0
            self.server.total_placements[i] = [0] * num_players

        envs = self.server.envs
        clients = self.server.clients

        while not self._is_done():
            if self.server.needs_reset:
                init_return_data: List[MultiObservation] = envs.reset()
                self.server.needs_reset = False

                self._distribute_return_data(init_return_data)
                del init_return_data

            player_actions = self._parse_messages()
            actions_iter = self._order_player_actions(player_actions)

            return_data: List[Tuple[
                MultiObservation,
                MultiReward,
                MultiIsDone,
                MultiInfo,
            ]] = envs.step(actions_iter)

            for data in return_data:
                obs, reward, is_done, info = data
                # return_data = {
                #     'obs': obs,
                #     'reward': reward,
                #     'is_done': is_done,
                #     'info': info,
                # }
                first_key = next(iter(info.keys()))
                single_info = info[first_key]
                prev_active_player_index = \
                    single_info['prev_active_player_index']
                self.server.num_illegals[prev_active_player_index] += \
                    single_info['was_illegal']

            game_is_done = is_done['__all__']
            if not game_is_done:
                self._distribute_return_data(return_data)
                continue

            self.server.needs_reset = True
            self.server.num_games += self.server.num_parallel_games

            for data in return_data:
                _, _, _, info = data
                first_key = next(iter(info.keys()))
                single_info = info[first_key]
                final_penalties = single_info['final_penalties']
                final_rankings = single_info['final_rankings']

                self.server.stats.append((
                    final_penalties,
                    final_rankings,
                ))
                for (i, penalty) in enumerate(final_penalties):
                    self.server.total_penalties[i] += penalty
                for (i, ranking) in enumerate(final_rankings):
                    self.server.total_placements[i][ranking - 1] += 1

            self.server.print_log(f'Num games: {self.server.num_games}')
            if self.server.num_parallel_games == 1:
                if 2 not in final_rankings:
                    winner_indices = [
                        i
                        for (i, ranking) in enumerate(final_rankings)
                        if ranking == 1
                    ]
                    self.server.print_log(f'Winners: {winner_indices}')
                else:
                    self.server.print_log(f'Winner: {final_rankings.index(1)}')

            self.server.logger.info(
                f'Total penalties: {self.server.total_penalties}')
            self.server.logger.info(
                f'Total placements: {self.server.total_placements}')
            results_table = utils.create_results_table(
                self.server.total_penalties,
                self.server.total_placements,
                self._index_to_name,
                self.server.num_illegals,
            )
            print(results_table)
            results_table: bytes = server_utils.encode_data(
                '\n' + results_table)

            return_data: List[Tuple[  # type: ignore[no-redef]
                int,
                MultiObservation,
                MultiReward,
                MultiIsDone,
                MultiInfo,
            ]] = [(i,) + data for (i, data) in enumerate(return_data)]
            return_data: bytes = (  # type: ignore[no-redef]
                self._encode_data(return_data)
            )
            self.server.logger.debug('Return data:', return_data)

            self._communicators.map(
                lambda client: self.server.send_failable_replacing(
                    client, return_data),
                (clients[i] for i in range(num_players)),
            )
            self._communicators.map(
                lambda client: self.server.receive_ok_replacing(
                    client, self.OK_TIMEOUT_SEC),
                (clients[i] for i in range(num_players)),
            )
            self._communicators.map(
                lambda client: self.server.send_failable_replacing(
                    client, results_table),
                (clients[i] for i in range(num_players)),
            )
            self._communicators.map(
                lambda client: self.server.receive_ok_replacing(
                    client, self.OK_TIMEOUT_SEC),
                (clients[i] for i in range(num_players)),
            )

    def finish(self) -> None:
        self.server.print_log('Finishing...')
        self._communicators.terminate()
        self._communicators.join()
        self.server.needs_reset = True

        clients = self.server.clients

        # Clean up all requests.
        for client in clients.values():
            self.server.shutdown_request(  # type: ignore[attr-defined]
                client.request)

        clients.clear()

        self.server.print_log('Done.')
