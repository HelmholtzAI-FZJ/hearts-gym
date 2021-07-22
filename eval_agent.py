"""
Evaluate a local agent on a remote server.
"""

from argparse import ArgumentParser, Namespace
from json import JSONDecodeError
from pathlib import Path
import pickle
import socket
import sys
from typing import Any, Dict, List, Optional, Tuple
import uuid
from uuid import UUID
import zlib

import numpy as np
import ray
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils.typing import PolicyID, TensorType, TrainerConfigDict
from ray.tune.result import EXPR_PARAM_PICKLE_FILE
from ray.tune.trainable import Trainable

import configuration as conf
from configuration import ENV_NAME, LEARNED_POLICY_ID
from hearts_gym import HeartsEnv, utils
from hearts_gym.server import utils as server_utils
from hearts_gym.server.hearts_server import (
    Client,
    HeartsRequestHandler,
    HeartsServer,
    SERVER_ADDRESS,
    PORT,
)
from hearts_gym.utils import ObsTransform
from hearts_gym.utils.typing import Observation, Reward

SERVER_TIMEOUT_SEC = HeartsServer.PRINT_INTERVAL_SEC + 5


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
        nargs='?',
        default=conf.checkpoint_path,
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
        default=conf.algorithm,
        help='Model algorithm to use.',
    )
    parser.add_argument(
        '--framework',
        type=str,
        default=conf.framework,
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
    parser.add_argument(
        '--policy_id',
        type=PolicyID,
        default=LEARNED_POLICY_ID,
        help='ID of the policy to evaluate.',
    )

    return parser.parse_args()


def _assert_same_envs(
        config: TrainerConfigDict,
        server_metadata: Dict[str, Any],
) -> None:
    """Raise an error when the environment configuration in the given
    configuration does not match the one from the server.

    Args:
        config (TrainerConfigDict): Local configuration.
        server_metadata (Dict[str, Any]): Server configuration metadata.
    """
    load_env_name = utils.get_default(config, 'env', COMMON_CONFIG)
    assert load_env_name == ENV_NAME, (
        f'loaded agent was trained on different environment '
        f'({load_env_name}); please change `ENV_NAME` in `configuration.py` '
        f'if this is fine'
    )

    env_config = utils.get_default(config, 'env_config', COMMON_CONFIG)
    for attr in ['num_players', 'deck_size']:
        # We just expect these to be set.
        load_attr = env_config.get(attr, None)
        server_attr = server_metadata[attr]
        assert load_attr == server_attr, (
            f'environment model was trained on does not match server '
            f'environment: {attr} does not match '
            f'({load_attr} != {server_attr})'
        )


def configure_remote_eval(
        config: TrainerConfigDict,
        policy_id: PolicyID,
) -> TrainerConfigDict:
    """Return the given configuration modified so it has settings useful
    for remote evaluation.

    Args:
        config (TrainerConfigDict): RLlib configuration to set up
            for evaluation.

    Returns:
        TrainerConfigDict: Evaluation configuration based on the
            given one.
    """
    eval_config = utils.configure_eval(config)
    eval_config['num_workers'] = 0

    multiagent_config = utils.get_default(
        eval_config, 'multiagent', COMMON_CONFIG).copy()
    eval_config['multiagent'] = multiagent_config
    multiagent_config['policy_mapping_fn'] = lambda _: policy_id

    return eval_config


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


def _receive_msg_length(
        client: socket.socket,
        max_receive_bytes: int,
) -> Tuple[int, bytes]:
    """Return the expected length of a message received from the server
    in a failsafe way.

    To be more efficient, receive more data than necessary. Any
    additional data is returned.

    If the server stopped, exit the program.

    Args:
        client (socket.socket): Socket of the client.
        max_receive_bytes (int): Number of bytes to receive at maximum
            per message shard.

    Returns:
        int: Amount of bytes in the rest of the message.
        bytes: Extraneous part of message data received.
    """
    data_shard = _receive_data_shard(client, max_receive_bytes)
    total_num_received_bytes = len(data_shard)
    data = [data_shard]
    length_end = data_shard.find(server_utils.MSG_LENGTH_SEPARATOR)
    while (
            length_end == -1
            and total_num_received_bytes < server_utils.MAX_MSG_PREFIX_LENGTH
    ):
        data_shard = _receive_data_shard(client, max_receive_bytes)
        total_num_received_bytes += len(data_shard)
        data.append(data_shard)
        length_end = data_shard.find(server_utils.MSG_LENGTH_SEPARATOR)

    assert length_end != -1, 'server did not send message length'

    length_end += total_num_received_bytes - len(data_shard)
    data = b''.join(data)
    msg_length = int(data[:length_end])
    extra_data = data[length_end + len(server_utils.MSG_LENGTH_SEPARATOR):]

    return msg_length, extra_data


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
    msg_length, data_shard = _receive_msg_length(client, max_receive_bytes)
    assert msg_length < max_total_receive_bytes, 'message is too long'

    total_num_received_bytes = len(data_shard)
    data = [data_shard]
    while total_num_received_bytes < msg_length:
        data_shard = _receive_data_shard(client, max_receive_bytes)
        total_num_received_bytes += len(data_shard)
        data.append(data_shard)

    assert total_num_received_bytes == msg_length, \
        'message does not match length'

    data = b''.join(data)
    try:
        data = server_utils.decode_data(data)
    except (JSONDecodeError, zlib.error) as ex:
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


def _take_indices(data: List[Any], indices: List[int]) -> List[Any]:
    """Return the elements obtained by indexing into the given data
    according to the given indices.

    Args:
        data (List[Any]): List to multi-index.
        indices (List[int]): Indices to use; are used in the
            order they are given in.

    Returns:
        List[Any]: Elements obtained by multi-indexing into the
            given data.
    """
    return [data[i] for i in indices]


def _update_indices(
        values: List[Any],
        indices: List[int],
        new_values: List[Any],
) -> None:
    """Update the given list of values with new elements according to
    the given indices.

    Args:
        values (List[Any]): List to multi-update.
        indices (List[int]): Indices to use; are used in the
            order they are given in.
        new_values (List[Any]): Updated values; one for each index.
    """
    assert len(indices) == len(new_values), \
        'length of indices to update and values to update with must match'
    for (i, new_val) in zip(indices, new_values):
        values[i] = new_val


def _update_states(
        agent: Trainable,
        states: List[List[TensorType]],
        indices: List[int],
        new_states: List[List[TensorType]],
):
    model_config = utils.get_default(agent.config, 'model', COMMON_CONFIG)

    if (
            utils.get_default(
                model_config, 'use_attention', MODEL_DEFAULTS)
            or (
                (
                    utils.get_default(
                        model_config, 'custom_model', MODEL_DEFAULTS)
                    is not None
                )
                and model_config.get(
                    'custom_model', '').endswith('_attn')
            )
    ):
        for (i, new_state) in zip(indices, new_states):
            for (j, (prev_state, state)) in enumerate(
                    zip(states[i], new_state),
            ):
                states[i][j] = np.vstack((prev_state[1:], state))
    else:
        _update_indices(states, indices, new_states)


def _transform_observations(
        obs_transforms: List[ObsTransform],
        remove_action_mask: bool,
        has_action_mask: bool,
        observations: List[Observation],
        uuids: List[UUID],
) -> None:
    """Modify the given observations in-place, applying the given
    transformations and stripping them of the action mask as desired.

    Args:
        obs_transforms (List[ObsTransform]): Transformations to apply to
            the raw observations (that is, without the action mask).
        remove_action_mask (bool): Whether to strip the observations of
            their action mask.
        has_action_mask (bool): Whether the observation contain an
            action mask.
        observations (List[Observation]): Observations to transform.
        uuids (List[UUID]): Uniquely identifying IDs of the games
            being observed.
    """
    if has_action_mask:
        for (i, (obs, uuid_)) in enumerate(zip(observations, uuids)):
            observations[i][HeartsEnv.OBS_KEY] = utils.apply_obs_transforms(
                obs_transforms,
                obs[HeartsEnv.OBS_KEY],
                0,
                uuid_,
            )
    else:
        for (i, (obs, uuid_)) in enumerate(zip(observations, uuids)):
            observations[i] = utils.apply_obs_transforms(
                obs_transforms,
                obs,
                0,
                uuid_,
            )

    if remove_action_mask:
        for (i, obs) in enumerate(observations):
            observations[i] = obs[HeartsEnv.OBS_KEY]


def main() -> None:
    """Connect to a server and play games using a loaded model."""
    args = parse_args()
    assert (
        args.checkpoint_path is not None
        or args.policy_id is not LEARNED_POLICY_ID
    ), 'need a checkpoint for the learned policy'
    name = args.name
    if name is not None:
        Client.check_name_length(name.encode())

    algorithm = args.algorithm
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        assert checkpoint_path.exists(), 'checkpoint file does not exist'
        assert checkpoint_path.is_file(), \
            'please pass the checkpoint file, not its directory'
        checkpoint_path.resolve(True)
        params_path = checkpoint_path.parent.parent / EXPR_PARAM_PICKLE_FILE
        has_params = params_path.is_file()
    else:
        has_params = False

    ray.init()

    with server_utils.create_client() as client:
        client.connect((args.server_address, args.port))
        client.settimeout(SERVER_TIMEOUT_SEC)
        print('Connected to server.')
        server_utils.send_name(client, name)

        metadata = wait_for_data(
            client,
            server_utils.MAX_RECEIVE_BYTES,
            server_utils.MAX_RECEIVE_BYTES,
        )
        player_index = metadata['player_index']
        num_players = metadata['num_players']
        deck_size = metadata['deck_size']
        mask_actions = metadata['mask_actions']
        max_num_games = metadata['max_num_games']
        num_parallel_games = metadata['num_parallel_games']

        print(f'Positioned at index {player_index}.')

        max_total_receive_bytes = \
            server_utils.MAX_RECEIVE_BYTES * num_parallel_games
        # We only get strings as keys.
        str_player_index = str(player_index)

        if conf.allow_pickles and has_params:
            with open(params_path, 'rb') as params_file:
                config = pickle.load(params_file)

            assert (
                args.policy_id
                in utils.get_default(
                    utils.get_default(config, 'multiagent', COMMON_CONFIG),
                    'policies',
                    COMMON_CONFIG['multiagent'],
                )
            ), (
                'cannot find policy ID in loaded configuration; '
                'please configure `args.policy_id`'
            )
            _assert_same_envs(config, metadata)
            print('Loaded configuration for checkpoint; to disable, set '
                  '`allow_pickles = False` in `configuration.py`.')
        else:
            env_config = {
                'num_players': num_players,
                'deck_size': deck_size,
                # We allow the user to set their own here so they may
                # use a non-action-masked model even though the
                # environment is action-masked.
                'mask_actions': mask_actions and conf.mask_actions,
            }

            config = {
                **conf.config,
                'env_config': env_config,
                'framework': args.framework,
            }
        config = configure_remote_eval(config, args.policy_id)
        utils.maybe_set_up_masked_actions_model(algorithm, config)

        agent = utils.create_agent(algorithm, config)
        if args.checkpoint_path:
            agent = utils.load_agent(algorithm, str(checkpoint_path), config)

        server_utils.send_ok(client)
        remove_action_mask = (
            mask_actions
            and not utils.get_default(config, 'env_config', COMMON_CONFIG).get(
                'mask_actions', HeartsEnv.MASK_ACTIONS_DEFAULT)
        )
        obs_transforms = utils.get_default(
            config, 'env_config', COMMON_CONFIG).get('obs_transforms', [])

        num_iters = 0
        num_games = 0
        while not _is_done(num_games, max_num_games):
            uuids = [uuid.uuid4() for _ in range(num_parallel_games)]
            states: List[List[TensorType]] = [
                utils.get_initial_state(agent, args.policy_id)
                for _ in range(num_parallel_games)
            ]
            prev_actions: List[Optional[TensorType]] = \
                [None] * num_parallel_games
            prev_rewards: List[Optional[Reward]] = \
                [None] * num_parallel_games

            while True:
                data = wait_for_data(
                    client,
                    server_utils.MAX_RECEIVE_BYTES,
                    max_total_receive_bytes,
                )

                if len(data) == 0:
                    # We have no observations; send no actions.
                    server_utils.send_actions(client, [])

                if len(data[0]) < 4:
                    (indices, obss) = zip(*data)
                else:
                    (indices, obss, rewards, is_dones, infos) = zip(*data)
                    rewards = [
                        reward[str_player_index]
                        for reward in rewards
                    ]
                    _update_indices(prev_rewards, indices, rewards)

                    if is_dones[0]['__all__']:
                        break
                assert all(str_player_index in obs for obs in obss), \
                    'received wrong data'
                obss = [obs[str_player_index] for obs in obss]
                _transform_observations(
                    obs_transforms,
                    remove_action_mask,
                    mask_actions,
                    obss,
                    _take_indices(uuids, indices),
                )
                # print('Received', len(obss), 'observations.')

                masked_prev_actions = _take_indices(prev_actions, indices)
                masked_prev_rewards = _take_indices(prev_rewards, indices)
                actions, new_states, _ = utils.compute_actions(
                    agent,
                    obss,
                    _take_indices(states, indices),
                    (
                        masked_prev_actions
                        if None not in masked_prev_actions
                        else None
                    ),
                    (
                        masked_prev_rewards
                        if None not in masked_prev_rewards
                        else None
                    ),
                    policy_id=args.policy_id,
                    full_fetch=True,
                )
                # print('Actions:', actions)

                server_utils.send_actions(client, actions)

                _update_states(agent, states, indices, new_states)
                _update_indices(prev_actions, indices, actions)

            server_utils.send_ok(client)
            num_games += num_parallel_games
            num_iters += 1

            if num_games % 128 == 0:
                print('Played', num_games, 'games.')

    ray.shutdown()


if __name__ == '__main__':
    main()
