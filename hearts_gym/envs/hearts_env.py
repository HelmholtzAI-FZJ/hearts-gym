"""
A multi-agent Gym-like environment for learning the game of Hearts
(specifically, Black Lady).
"""

from contextlib import closing
from io import StringIO
import sys
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

from gym import spaces
from gym.utils import seeding
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .hearts_game import HeartsGame

GymSeed = Union[int, str, None]
"""Seed type as supported by the `gym` library."""
Real = Union[int, float, np.integer, np.floating]
"""Real number type."""

AgentId = int
"""ID of an agent.

For this environment, these are the corresponding player indices.
"""
Action = int
"""An action giving the index of which card in hand to play."""
MultiAction = Dict[AgentId, Action]

Observation = Dict[str, Any]
MultiObservation = Dict[AgentId, Observation]
Reward = Real
MultiReward = Dict[AgentId, Reward]
IsDone = bool
MultiIsDone = Dict[Union[AgentId, str], IsDone]
Info = Dict[str, Any]
MultiInfo = Dict[AgentId, Info]


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
            mask_actions: bool = True,
            seed: GymSeed = 0,
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
            game (Optional[HeartsGame]): A pre-initialized game simulator.
            mask_actions (bool): Whether to enable action masking,
                parameterizing the action space.
            seed (GymSeed): Random number generator base seed.
        """
        seed = self.seed(seed)[0]
        if game is None:
            game = HeartsGame(
                num_players=num_players, deck_size=deck_size, seed=seed)
        self.game = game
        self.mask_actions = mask_actions

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
        if mask_actions:
            obs_space = {self.OBS_KEY: spaces.Dict(obs_space)}
            # It's important that all other keys in the dictionary are
            # ordered below this. Otherwise the model and policies will
            # act up.
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
    def on_table_state(player_index_offset: int) -> int:
        """Return the state for a card put on the table by the player with the
        given index offset from a certain agent.

        The index offset should assume wrapping, so with 4 players in
        total, the offset from the player with index 3 to player index 0
        is 1.

        Args:
            player_index_offset (int): Index offset with relation to the
                agent of the player that put the card on the table.

        Returns:
            int: State for a card put on the table by a given player.
        """
        return HeartsEnv.NUM_GENERAL_OBSERVATION_STATES + player_index_offset

    def collected_state(self, player_index_offset: int) -> int:
        """Return the state for a card collected (picked up by winning a trick)
        by the player with the given index offset from a certain agent.

        The index offset should assume wrapping, so with 4 players in
        total, the offset from the player with index 3 to player index 0
        is 1.

        Args:
            player_index_offset (int): Index offset with relation to the
                agent of the player that has collected the card.

        Returns:
            int: State for a card put on the table by a given player.
        """
        return (self.NUM_GENERAL_OBSERVATION_STATES
                + self.game.num_players
                + player_index_offset)

    def _game_state_to_obs(self, player_index: int) -> Observation:
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
            Observation: The observation with all known information of
                the given player.
        """
        cards_state = self.game.state.copy()

        first_on_table_state = self.game.on_table_state(0)
        last_on_table_state = self.game.on_table_state(
            self.game.num_players - 1)

        first_in_hand_state = self.game.in_hand_state(0)
        last_in_hand_state = self.game.in_hand_state(
            self.game.num_players - 1)

        first_collected_state = self.game.collected_state(0)
        last_collected_state = self.game.collected_state(
            self.game.num_players - 1)

        in_own_hand_state = self.game.in_hand_state(player_index)

        on_table_indices = (
            (cards_state >= first_on_table_state)
            & (cards_state <= last_on_table_state)
        )
        in_hand_indices = (
            (cards_state >= first_in_hand_state)
            & (cards_state <= last_in_hand_state)
        )
        collected_indices = (
            (cards_state >= first_collected_state)
            & (cards_state <= last_collected_state)
        )
        in_own_hand_indices = cards_state == in_own_hand_state
        if self.game.STATE_UNKNOWN != self.STATE_UNKNOWN:
            state_unknown_indices = cards_state == self.game.STATE_UNKNOWN

        cards_state[on_table_indices] = (
            self.on_table_state(
                (
                    cards_state[on_table_indices]
                    - first_on_table_state
                    - player_index
                ) % self.game.num_players
            )
        )
        cards_state[in_hand_indices] = self.STATE_UNKNOWN
        cards_state[collected_indices] = (
            self.collected_state(
                (
                    cards_state[collected_indices]
                    - first_collected_state
                    - player_index
                ) % self.game.num_players
            )
        )
        cards_state[in_own_hand_indices] = self.STATE_ON_HAND
        if self.game.STATE_UNKNOWN != self.STATE_UNKNOWN:
            cards_state[state_unknown_indices] = self.STATE_UNKNOWN

        obs = {
            'cards': cards_state,
            'leading_hearts_allowed': self.game.leading_hearts_allowed,
        }
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

    def compute_reward(
            self,
            player_index: int,
            prev_active_player_index: int,
            trick_winner_index: Optional[int],
            trick_score: Optional[int],
    ) -> Reward:
        """Return the reward for the player with the given index.

        It is important to keep in mind that most of the time, the
        arguments are unrelated to the player getting their reward. This
        is because agents receive their reward only when it is their
        next turn, not right after their turn. Due to this peculiarity,
        it is encouraged to use `self.game.prev_played_cards`,
        `self.game.prev_was_illegals`, and others.

        Args:
            player_index (int): Index of the player to return the reward
                for. This is most of the time _not_ the player that took
                the action.
            prev_active_player_index (int): Index of the previously
                active player that took the action. In other words, the
                active player index before the action was taken.
            trick_winner_index (Optional[int]): Index of the player that
                won the trick or `None` if it is still ongoing.
            trick_score (Optional[int]): Score of the cards obtained by
                the player that won the trick or `None` if it is
                still ongoing.

        Returns:
            Reward: Reward for the player with the given index.
        """
        if self.game.prev_was_illegals[player_index]:
            return -self.game.max_score * self.game.max_num_cards_on_hand

        card = self.game.prev_played_cards[player_index]

        if card is None:
            # The agent did not take a turn until now; no information
            # to provide.
            return 0

        if (
                trick_winner_index is not None
                and self.has_shot_the_moon(player_index)
        ):
            return self.game.max_score * self.game.max_num_cards_on_hand

        # score = self.game.scores[player_index]

        # if self.game.is_done():
        #     return -score

        prev_trick_winner_index = self.game.leading_player_index
        if prev_trick_winner_index == player_index:
            assert self.game.prev_trick_score is not None
            return -self.game.prev_trick_score
        return 1
        # return -score

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
                trick_winner_index,
                trick_penalty,
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
