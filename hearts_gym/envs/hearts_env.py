"""
A multi-agent Gym-like environment for learning the game of Hearts
(specifically, Black Lady).
"""

from contextlib import closing
from io import StringIO
import sys
from typing import Any, List, Optional, TextIO, Tuple, Union
import uuid

from gym import spaces
from gym.utils import seeding
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from hearts_gym.utils.obs_transforms import apply_obs_transforms, ObsTransform
from hearts_gym.utils.typing import (
    GymSeed,
    MultiAction,
    MultiInfo,
    MultiIsDone,
    MultiObservation,
    MultiReward,
)
from .hearts_game import HeartsGame


class HeartsEnv(MultiAgentEnv):
    """A multi-agent Gym-like Hearts environment.

    See `HeartsGame` for the underlying game simulator.

    For multi-agent support, environment information is returned as a
    dictionary mapping from player indices to the according information
    for that player.

    Observations are player index-independent. Optionally, action
    masking may be enabled to parameterize the action space so that – if
    properly set up – an agent may choose only legal actions. This is
    not supported by all reinforcement learning algorithms.

    The main function of interest for optimizing an agent is the
    `RewardFunction` in `./hearts_gym/envs/reward_function.py`.

    See also `MultiAgentEnv` for a description of
    multi-agent environments.
    """

    MASK_ACTIONS_DEFAULT = True

    NUM_GENERAL_OBSERVATION_STATES = 2
    """Amount of states that do not refer to a specific player."""

    STATE_UNKNOWN = 0
    """This card has not been seen."""
    STATE_ON_HAND = 1
    """This card is on the player's hand."""

    OBS_KEY = 'obs'
    """Key in the observation dictionary for the standard observations.
    Only used when action masking is enabled.
    """
    ACTION_MASK_KEY = 'action_mask'
    """Key in the observation dictionary for the action mask.
    Only used when action masking is enabled.
    """

    # Gym metadata
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(
            self,
            *,
            num_players: int = 4,
            deck_size: int = 52,
            game: Optional[HeartsGame] = None,
            mask_actions: bool = MASK_ACTIONS_DEFAULT,
            seed: GymSeed = 0,
            obs_transforms: List[ObsTransform] = [],
    ) -> None:
        """Construct a Hearts environment with a strong random seed.

        Optional action masking may be enabled to parameterize the
        action space so that – if properly set up – an agent may choose
        only legal actions. This is not supported by all reinforcement
        learning algorithms. This changes the observation space.

        Args:
            num_players (int): Amount of players. Only used if `game` is
                not `None`.
            deck_size (int): Amount of cards in the deck. Only used if
                `game` is not `None`.
            game (Optional[HeartsGame]): A pre-initialized
                game simulator.
            mask_actions (bool): Whether to enable action masking,
                parameterizing the action space.
            seed (GymSeed): Random number generator base seed.
            obs_transforms (List[ObsTransform]): Transformations to
                apply to the observations.
        """
        seed = self.seed(seed)[0]
        if game is None:
            game = HeartsGame(
                num_players=num_players, deck_size=deck_size, seed=seed)
        self.game = game
        self.mask_actions = mask_actions
        self._obs_transforms = obs_transforms

        # Each card can either be
        #    0: unknown
        #    1: on our hand
        #    2, ..., 1 + num_players:
        #       on the table (i.e. part of the current trick),
        #       played by player at "clockwise" index from own + 2
        #    2 + num_players, ..., 1 + 2 * num_players:
        #       collected by player at "clockwise" index from own + 3
        self.num_observation_states = \
            self.NUM_GENERAL_OBSERVATION_STATES + self.game.num_players * 2

        # It's important that all other keys in the dictionary are
        # ordered below these ones. Otherwise the model and policies
        # will act up.
        obs_space = {
            # The whole deck with possible individual states as
            # described above.
            'cards': spaces.Box(
                low=0,
                high=self.num_observation_states - 1,
                shape=self.game.state.shape,
                dtype=self.game.state.dtype,
            ),
            # Whether players are allowed to start a trick with a
            # heart-suited card.
            #
            # This is only allowed once any player was unable to follow
            # the leading suit and instead played a hearts card.
            # However, a player may always start with hearts if they
            # don't have any other suit.
            'leading_hearts_allowed': spaces.Discrete(2),
        }
        ordered_keys = iter(sorted(obs_space.keys()))
        first_ordered_key = next(ordered_keys)
        second_ordered_key = next(ordered_keys)
        assert (
            first_ordered_key == 'cards'
            and second_ordered_key == 'leading_hearts_allowed'
        ), (
            f'first two keys in the first definition of `obs_space` must be '
            f"'cards' and 'leading_hearts_allowed', in that order (was "
            f'{first_ordered_key} and {second_ordered_key})'
        )

        if mask_actions:
            # Same as above, all other keys in the dictionary must be
            # ordered below these ones.
            obs_space = {self.OBS_KEY: spaces.Dict(obs_space)}
            obs_space[self.ACTION_MASK_KEY] = spaces.Box(
                low=0,
                high=1,
                shape=(self.game.max_num_cards_on_hand,),
                dtype=np.int8,
            )
            first_ordered_key = next(iter(sorted(obs_space.keys())))
            assert first_ordered_key == self.ACTION_MASK_KEY, (
                f'first key in observation space must be '
                f'`HeartsEnv.ACTION_MASK_KEY` (was {first_ordered_key})'
            )

        self.observation_space = spaces.Dict(obs_space)

        self.action_space = spaces.Discrete(self.game.max_num_cards_on_hand)

        from .reward_function import RewardFunction
        self.reward_function = RewardFunction(self)

    def seed(self, seed: GymSeed = None) -> List[int]:
        """Return a strong seed for this environment's random
        number generator(s).

        See also `gym.Env.seed`.

        Args:
            seed (GymSeed): Base random number generator seed. Used to
                create a stronger seed.

        Returns:
            List[int]: List of seeds used in this environment's random
                number generator(s).
        """
        return [seeding.create_seed(seed)]

    @staticmethod
    def on_table_state(
            player_index_offsets: Union[int, np.ndarray],
    ) -> Union[int, np.ndarray]:
        """Return the states for cards put on the table by the players
        with the given index offsets from a certain agent.

        The index offsets should assume wrapping, so with 4 players in
        total, the offset from the player with index 3 to player index 0
        is 1.

        Args:
            player_index_offsets (Union[int, np.ndarray]): Index offsets
                with relation to the agent of the players that put cards
                on the table.

        Returns:
            Union[int, np.ndarray]: States for cards put on the table
                by the given players.
        """
        return HeartsEnv.NUM_GENERAL_OBSERVATION_STATES + player_index_offsets

    @staticmethod
    def collected_state(
            player_index_offsets: Union[int, np.ndarray],
            num_players: int,
    ) -> Union[int, np.ndarray]:
        """Return the states for cards collected (picked up by winning a
        trick) by the players with the given index offsets from a
        certain agent.

        The index offsets should assume wrapping, so with 4 players in
        total, the offset from the player with index 3 to player index 0
        is 1.

        Args:
            player_index_offsets (Union[int, np.ndarray]): Index offsets
                with relation to the agent of the players that have
                collected cards.
            num_players (int): Amount of players in the game.

        Returns:
            Union[int, np.ndarray]: States for cards collected by the
                given players.
        """
        return (
            HeartsEnv.NUM_GENERAL_OBSERVATION_STATES
            + num_players
            + player_index_offsets
        )

    @staticmethod
    def get_offset_indices(
            player_indices: np.ndarray,
            offset_from_player_index: int,
            num_players: int,
    ) -> np.ndarray:
        """Return the given indices converted to offset indices for the
        player with the given index.

        Args:
            player_indices (np.ndarray): Indices to convert to
                offset indices.
            offset_from_player_index (int): Index of the player the
                indices are being offset from.
            num_players (int): Amount of players in the game.

        Returns:
            np.ndarray: The given indices as offset indices from the
                given player index.
        """
        return (player_indices - offset_from_player_index) % num_players

    def _game_state_to_obs(self, player_index: int) -> Any:
        """Return all necessary game state information as a player
        index-independent observation for the player with the given
        index.

        The information is not perfect, so the player only knows the
        state of cards it has 'seen'. However, we are assuming perfect
        memory. The complete history of the game is also not included in
        the state.

        Args:
            player_index (int): Index of the player to get an
                observation for.

        Returns:
            Any: The observation with all known information of the
                given player.
        """
        cards_state = self.game.state.copy()

        lowest_on_table_state = self.game.on_table_state(0)
        highest_on_table_state = self.game.on_table_state(
            self.game.num_players - 1)

        lowest_in_hand_state = self.game.in_hand_state(0)
        highest_in_hand_state = self.game.in_hand_state(
            self.game.num_players - 1)

        lowest_collected_state = self.game.collected_state(0)
        highest_collected_state = self.game.collected_state(
            self.game.num_players - 1)

        in_own_hand_state = self.game.in_hand_state(player_index)

        on_table_indices = (
            (cards_state >= lowest_on_table_state)
            & (cards_state <= highest_on_table_state)
        )
        in_hand_indices = (
            (cards_state >= lowest_in_hand_state)
            & (cards_state <= highest_in_hand_state)
        )
        collected_indices = (
            (cards_state >= lowest_collected_state)
            & (cards_state <= highest_collected_state)
        )
        in_own_hand_indices = cards_state == in_own_hand_state
        if self.game.STATE_UNKNOWN != self.STATE_UNKNOWN:
            state_unknown_indices = cards_state == self.game.STATE_UNKNOWN

        on_table_player_indices = (
            cards_state[on_table_indices]
            - lowest_on_table_state
        )
        collected_player_indices = (
            cards_state[collected_indices]
            - lowest_collected_state
        )

        cards_state[on_table_indices] = self.on_table_state(
            self.get_offset_indices(
                on_table_player_indices,
                player_index,
                self.game.num_players,
            ),
        )
        cards_state[in_hand_indices] = self.STATE_UNKNOWN
        cards_state[collected_indices] = self.collected_state(
            self.get_offset_indices(
                collected_player_indices,
                player_index,
                self.game.num_players,
            ),
            self.game.num_players,
        )
        cards_state[in_own_hand_indices] = self.STATE_ON_HAND
        if self.game.STATE_UNKNOWN != self.STATE_UNKNOWN:
            cards_state[state_unknown_indices] = self.STATE_UNKNOWN

        obs = {
            'cards': cards_state,
            'leading_hearts_allowed': self.game.leading_hearts_allowed,
        }
        obs = apply_obs_transforms(
            self._obs_transforms, obs, player_index, self.uuid)

        if self.mask_actions:
            obs = {self.OBS_KEY: obs}
            action_mask = np.zeros(
                self.observation_space[self.ACTION_MASK_KEY].shape)
            legal_actions = self.game.get_legal_actions(player_index)
            action_mask[legal_actions] = 1
            obs[self.ACTION_MASK_KEY] = action_mask

        return obs

    def get_legal_actions(self) -> List[int]:
        """Return all legal actions for the active player.

        Returns:
            List[int]: Indices for which cards in hand are legal to play.
        """
        return self.game.get_legal_actions(self.game.active_player_index)

    @property
    def num_players(self) -> int:
        """Amount of players in the game."""
        return self.game.num_players

    @property
    def deck_size(self) -> int:
        """Amount of cards in the deck."""
        return self.game.deck.size

    @property
    def active_player_index(self) -> int:
        """Index of the currently active player."""
        return self.game.active_player_index

    def step(
            self,
            action_dict: MultiAction,
    ) -> Tuple[MultiObservation, MultiReward, MultiIsDone, MultiInfo]:
        """Take a step in the environment, playing the card given by the action
        in `action_dict` for the agent specified by the key in it.
        Return observations from ready agents.

        The return values only concern the active player unless the game
        is over at which point every player gets their terminal
        observation.

        See also `MultiAgentEnv.step`.

        Args:
            action_dict (MultiAction): A dictionary mapping agent IDs to
                actions. As only one player is active at a time, it
                should contain only one key – the index of the
                active player.

        Returns:
            MultiObservation: Observations for each ready agent.
            MultiReward: Reward values for each ready agent. If the
                episode is just started, the value will be `None`.
            MultiIsDone: Whether each ready agent is done. The special
                key "__all__" (required) is used to indicate environment
                termination.
            MultiInfo: Optional info values for each agent ID.

        """
        leading_player_index = self.game.leading_player_index
        active_player_index = self.game.active_player_index
        assert active_player_index == next(iter(action_dict.keys()))

        card, was_illegal, trick_winner_index, trick_penalty = \
            self.game.play_card(action_dict[active_player_index])
        if self.mask_actions and was_illegal:
            print('actions should not be illegal when masking is on')

        trick_is_done = trick_winner_index is not None
        game_is_done = trick_is_done and self.game.is_done()
        if game_is_done:
            ready_player_indices = list(range(self.game.num_players))
            final_penalties = self.game.compute_final_penalties()
            final_rankings = self.game.compute_rankings()
        else:
            next_active_player_index = self.game.active_player_index
            ready_player_indices = [next_active_player_index]

        obs: MultiObservation = {}
        reward: MultiReward = {}
        is_done: MultiIsDone = {'__all__': game_is_done}
        info: MultiInfo = {}

        for ready_player_index in ready_player_indices:
            obs[ready_player_index] = \
                self._game_state_to_obs(ready_player_index)

            player_reward = self.reward_function(
                ready_player_index,
                active_player_index,
                trick_winner_index is not None,
            )
            reward[ready_player_index] = player_reward

            is_done[ready_player_index] = \
                len(self.game.hands[ready_player_index]) == 0

            player_info = {
                'prev_active_player_index': active_player_index,
                'active_player_index': self.game.active_player_index,
                'card': card,
                'was_illegal': was_illegal,
                'leading_player_index': leading_player_index,
                'trick_winner_index': trick_winner_index,
                'trick_penalty': trick_penalty,
            }
            if game_is_done:
                player_info['final_penalties'] = final_penalties
                player_info['final_rankings'] = final_rankings

            info[ready_player_index] = player_info

        # print('agent', active_player_index, 'took a step')
        # if trick_is_done:
        #     print('trick winner:', trick_winner_index)
        #     print('rewarding', ready_player_indices)
        # print('next agent:', self.game.active_player_index)
        return obs, reward, is_done, info

    def reset(self) -> MultiObservation:
        """Reset the environment. Return observations for ready agents. Due to
        the nature of the game, the first card is already force-played.
        Also due to the game being single player-turn-based, the
        observation only concerns the active player.

        See also `MultiAgentEnv.reset`.

        Returns:
            MultiObservation: Observation for the active player.
        """
        self.game.reset()
        self.uuid = uuid.uuid4()
        next_active_player_index = self.game.active_player_index
        obs = {
            next_active_player_index: (
                self._game_state_to_obs(next_active_player_index)),
        }
        return obs

    def _write_io_stream(self, io_stream: TextIO) -> int:
        """Write the internal game simulator as a string to the given
        I/O stream. Return the number of characters written.

        Args:
            io_stream (TextIO): Where to write to.

        Returns:
            int: Number of characters written.
        """
        return io_stream.write(str(self.game))

    def render(self, mode: str = 'human') -> Any:
        """Render the environment in a given mode. Return the representation or
        `None`, depending on the chosen mode.

        See also `gym.Env.render` for a more detailed description.

        To extend, more modes may be implemented. The return type may
        need to be adjusted.

        Args:
            mode (str): How to render the environment.

        Returns:
            Any: Representation of the rendered environment.
        """
        if mode == 'human' or mode == 'ansi':
            io_stream = StringIO() if mode == 'ansi' else sys.stdout
            self._write_io_stream(io_stream)

            if mode == 'ansi':
                with closing(io_stream):
                    return io_stream.getvalue()  # type: ignore[attr-defined]
        else:
            raise NotImplementedError('unsupported render mode')
        return None

    def close(self) -> None:
        """Clean up the environment.

        See also `gym.Env.close`.
        """
        pass
