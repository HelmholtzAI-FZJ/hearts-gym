"""
Evaluate a local agent on a remote server.
"""

from argparse import ArgumentParser, Namespace
from json import JSONDecodeError
from pathlib import Path
import socket
import sys
from typing import Any, List, Optional

import ray
from ray.rllib.utils.typing import TensorType

from hearts_gym import utils
from hearts_gym.envs.hearts_env import Reward
from hearts_gym.server import utils as server_utils
from hearts_gym.server.hearts_server import (
    Client,
    HeartsRequestHandler,
    SERVER_ADDRESS,
    PORT,
)
from hearts_gym.policies import RandomPolicy, RuleBasedPolicy

MAX_RECEIVE_BYTES = 8192
ENV_NAME = 'Hearts-v0'
LEARNED_POLICY_ID = 'learned'


def parse_args() -> Namespace:
    """Parse command line arguments for evaluating an agent against
    a server.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = ArgumentParser()

    parser.add_argument(
        'checkpoint_path',
        type=str,
        help='Path of model checkpoint to load for evaluation.',
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Name to register',
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='PPO',
        help='Model algorithm to use.',
    )
    parser.add_argument(
        '--framework',
        type=str,
        default=utils.DEFAULT_FRAMEWORK,
        help='Framework used for training.',
    )

    parser.add_argument(
        '--server_address',
        type=str,
        default=SERVER_ADDRESS,
        help='Server address to connect to.',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=PORT,
        help='Server port to connect to.',
    )

    return parser.parse_args()


def _is_done(num_games: int, max_num_games: Optional[int]) -> bool:
    """Return whether the desired number of games have been played..

    Returns:
        bool: Whether the desired number of games have been played.
    """
    return HeartsRequestHandler.is_done(num_games, max_num_games)


def _receive_data_shard(
        client: socket.socket,
        max_receive_bytes: int,
) -> bytes:
    """Return a message received from the server in a failsafe way.

    If the server stopped, exit the program.

    Args:
        client (socket.socket): Socket of the client.
        max_receive_bytes (int): Number of bytes to receive at maximum.

    Returns:
        Any: Message data received.
    """
    try:
        data = client.recv(max_receive_bytes)
    except Exception:
        print('Unable to receive data from server.')
        raise

    if data == b'' or data is None:
        print('Server stopped. Exiting...')
        sys.exit(0)

    return data


def receive_data(
        client: socket.socket,
        max_receive_bytes: int,
        max_total_receive_bytes: int,
) -> Any:
    """Return data received from the server in a failsafe way.

    If the server stopped, exit the program. If the message could not be
    decoded, return an error message string.

    Args:
        client (socket.socket): Socket of the client.
        max_receive_bytes (int): Number of bytes to receive at maximum
            per message shard.
        max_total_receive_bytes (int): Number of bytes to receive at
            maximum, that is, summed over all message shards.

    Returns:
        Any: Data received or an error message string if there
            were problems.
    """
    data: List[bytes] = []

    data_shard = _receive_data_shard(client, max_receive_bytes)
    total_num_received_bytes = len(data_shard)
    data.append(data_shard)
    while (
            len(data_shard) == max_receive_bytes
            and total_num_received_bytes <= max_total_receive_bytes
    ):
        data_shard = _receive_data_shard(client, max_receive_bytes)
        total_num_received_bytes += len(data_shard)
        data.append(data_shard)

    data = b''.join(data)
    try:
        data = server_utils.decode_data(data)
    except JSONDecodeError as ex:
        print('Failed decoding:', data)
        print('Error message:', str(ex))
        return '[See decoding error message.]'
    return data


def wait_for_data(
        client: socket.socket,
        max_receive_bytes: int,
        max_total_receive_bytes: int,
) -> Any:
    """Continually receive data from the server the given client is
    connected to.

    Whenever the data received is a string, print it and receive
    data again.

    Args:
        client (socket.socket): Socket of the client.
        max_receive_bytes (int): Number of bytes to receive at maximum
            per message shard.
        max_total_receive_bytes (int): Number of bytes to receive at
            maximum per single message, that is, summed over all
            message shards of a single message.

    Returns:
        Any: Non-string data received.
    """
    data = receive_data(client, max_receive_bytes, max_total_receive_bytes)
    while isinstance(data, str):
        server_utils.send_ok(client)
        print('Server says:', data)
        data = receive_data(client, max_receive_bytes, max_total_receive_bytes)
    return data


def main() -> None:
    """Connect to a server and play games using a loaded model."""
    args = parse_args()
    name = args.name
    if name is not None:
        Client.check_name_length(name.encode())

    ray.init()

    algorithm = args.algorithm
    checkpoint_path = Path(args.checkpoint_path)

    with server_utils.create_client() as client:
        client.connect((args.server_address, args.port))
        print('Connected to server.')
        server_utils.send_name(client, name)

        metadata = wait_for_data(client, MAX_RECEIVE_BYTES, MAX_RECEIVE_BYTES)
        player_index = metadata['player_index']
        num_players = metadata['num_players']
        deck_size = metadata['deck_size']
        mask_actions = metadata['mask_actions']
        max_num_games = metadata['max_num_games']
        num_parallel_games = metadata['num_parallel_games']

        print(f'Positioned at index {player_index}.')

        max_total_receive_bytes = MAX_RECEIVE_BYTES * num_parallel_games
        # We only get strings as keys.
        str_player_index = str(player_index)

        env_config = {
            'num_players': num_players,
            'deck_size': deck_size,
            'mask_actions': mask_actions,
        }
        obs_space, act_space = utils.get_spaces(ENV_NAME, env_config)

        model_config = {
            'use_lstm': False,
        }

        config = {
            'env': ENV_NAME,
            'env_config': env_config,
            'model': model_config,
            'explore': False,
            'multiagent': {
                'policies_to_train': [],
                'policies': {
                    LEARNED_POLICY_ID: (None, obs_space, act_space, {}),
                    'random': (RandomPolicy, obs_space, act_space,
                               {'mask_actions': mask_actions}),
                    'rulebased': (RuleBasedPolicy, obs_space, act_space,
                                  {'mask_actions': mask_actions}),
                },
                'policy_mapping_fn': lambda _: LEARNED_POLICY_ID,
            },
            'num_gpus': utils.get_num_gpus(args.framework),
            'num_workers': 0,
            'framework': args.framework,
        }
        utils.maybe_set_up_masked_actions_model(algorithm, config)

        agent = utils.load_agent(algorithm, str(checkpoint_path), config)
        server_utils.send_ok(client)

        num_iters = 0
        num_games = 0
        while not _is_done(num_games, max_num_games):
            # FIXME convert to list; then, mask states and prev_* according to env
            # FIXME itertools.compress may help
            # states = [utils.get_initial_state(agent, LEARNED_POLICY_ID)]
            # prev_actions: List[Optional[TensorType]] = [None]
            # prev_rewards: List[Optional[Reward]] = [None]
            states: Optional[List[TensorType]] = None  # should be a dict like obs
            prev_actions: Optional[List[TensorType]] = None
            prev_rewards: Optional[List[Reward]] = None

            while True:
                data = wait_for_data(
                    client, MAX_RECEIVE_BYTES, max_total_receive_bytes)

                if len(data) == 0:
                    # We have no observations; send no actions.
                    try:
                        client.sendall(b',')
                    except Exception:
                        print('Unable to send data to server.')
                        raise
                    continue

                if not isinstance(data[0], list):
                    obss = data
                else:
                    (obss, rewards, is_dones, infos) = zip(*data)
                    rewards = [reward[str_player_index] for reward in rewards]
                    prev_rewards = rewards

                    if is_dones[0]['__all__']:
                        break
                assert all(str_player_index in obs for obs in obss), \
                    'received wrong data'
                obss = [obs[str_player_index] for obs in obss]
                # print('Received', len(obss), 'observations.')

                actions, states, _ = utils.compute_actions(
                    agent,
                    obss,
                    states,
                    prev_actions,
                    prev_rewards,
                    policy_id=LEARNED_POLICY_ID,
                    full_fetch=True,
                )
                # print('Actions:', actions)

                try:
                    client.sendall(server_utils.encode_actions(actions))
                except Exception:
                    print('Unable to send data to server.')
                    raise

                prev_actions = actions

            server_utils.send_ok(client)
            num_games += num_parallel_games
            num_iters += 1

            if num_iters % 100 == 0:
                print('Played', num_games, 'games.')

    ray.shutdown()


if __name__ == '__main__':
    main()
