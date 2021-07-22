"""
Start a server for remote agent evaluation.
"""

import argparse
import logging

from hearts_gym import HeartsEnv, utils
from hearts_gym.server.hearts_server import (
    HeartsServer,
    HeartsRequestHandler,
    SERVER_ADDRESS,
    PORT,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_players',
        default=4,
        type=int,
        help='Number of players in the game.',
    )
    parser.add_argument(
        '--deck_size',
        default=52,
        type=int,
        help='Number of cards in the deck.',
    )
    parser.add_argument(
        '--mask_actions',
        default=HeartsEnv.MASK_ACTIONS_DEFAULT,
        type=utils.parse_bool,
        help='Whether to apply action masking.',
    )
    parser.add_argument(
        '--seed',
        default=None,
        type=int,
        help='Random number generator base seed.',
    )
    parser.add_argument(
        '--num_parallel_games',
        default=1024,
        type=int,
        help=(
            'Number of games to play in parallel. '
            'Approximately also the batch size times four.'
        ),
    )
    parser.add_argument(
        '--num_procs',
        default=utils.get_num_cpus() - 1,
        type=int,
        help='How many processes to use for playing the parallel games.',
    )
    parser.add_argument(
        '--max_num_games',
        default=None,
        type=int,
        help=(
            'Number of games to play in total before disconnecting clients. '
            'By default, play with the same clients forever.'
        ),
    )
    parser.add_argument(
        '--accept_repeating_client_addresses',
        default=True,
        type=utils.parse_bool,
        help=(
            'Whether clients can connect from the same address more than once.'
        ),
    )
    parser.add_argument(
        '--wait_duration_sec',
        default=None,
        type=int,
        help='How long to wait until filling with randomly acting agents.',
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


def main() -> None:
    """Start a server for remote agent evaluation."""
    args = parse_args()
    logging.basicConfig()
    with HeartsServer(
            (args.server_address, args.port),
            HeartsRequestHandler,
            num_players=args.num_players,
            deck_size=args.deck_size,
            mask_actions=args.mask_actions,
            seed=args.seed,
            num_parallel_games=args.num_parallel_games,
            num_procs=args.num_procs,
            max_num_games=args.max_num_games,
            accept_repeating_client_addresses=(
                args.accept_repeating_client_addresses
            ),
            wait_duration_sec=args.wait_duration_sec,
    ) as server:
        try:
            server.serve_forever()
        except Exception:
            pass
        finally:
            server.envs.terminate_pool()


if __name__ == '__main__':
    main()
