"""
Helpers for evaluating an agent in a multi-agent setting using RLlib.
"""

import contextlib
import itertools
import os
import pickle
from tempfile import NamedTemporaryFile
import time
from typing import Any, Callable, Optional, Tuple

import numpy as np
import ray
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.rollout import RolloutSaver
from ray.rllib.utils.spaces import space_utils
from ray.rllib.utils.typing import (
    List,
    PolicyID,
    TensorType,
    TrainerConfigDict,
)
from ray.tune.trainable import Trainable

from . import common as utils
from hearts_gym.utils.typing import (
    AgentId,
    Info,
    MultiInfo,
    MultiIsDone,
    Reward,
)

__all__ = [
    'EvalResults',
    'configure_eval',
    'get_initial_state',
    'get_initial_states',
    'compute_actions',
    'evaluate',
    'create_results_table',
]

EvalResults = Tuple[List[int], List[List[int]], int, int, float]


def configure_eval(
        config: TrainerConfigDict,
) -> TrainerConfigDict:
    """Return the given configuration modified so it has settings useful
    for evaluation.

    The returned dictionary is a copy, but not a deepcopy, of the given
    dictionary. While this method guarantees `config` is not modified,
    be careful about further modifications.

    Args:
        config (TrainerConfigDict): RLlib configuration to set up
            for evaluation.

    Returns:
        TrainerConfigDict: Evaluation configuration based on the
            given one.
    """
    eval_config = config.copy()
    eval_config['explore'] = False

    multiagent_config = utils.get_default(
        eval_config, 'multiagent', COMMON_CONFIG).copy()
    eval_config['multiagent'] = multiagent_config
    multiagent_config['policies_to_train'] = []

    return eval_config


def get_initial_state(
        agent: Trainable,
        policy_id: PolicyID,
) -> List[TensorType]:
    """Return the initial recurrent state for the given agent policy.

    Unlike the `ray.rllib.policy.Policy.get_initial_state` method, this
    correctly handles attention states.

    Args:
        agent (Trainable): Reinforcement learning trainer/agent.
        policy_id (PolicyID): ID of the policy to query for.

    Returns:
        List[TensorType]: Initial recurrent state for the given
            agent policy.
    """
    policy = agent.get_policy(policy_id)
    model_config = utils.get_default(agent.config, 'model', COMMON_CONFIG)

    state = policy.get_initial_state()
    if (
            state
            or not utils.get_default(
                model_config, 'use_attention', MODEL_DEFAULTS)
            and (
                (
                    utils.get_default(
                        model_config, 'custom_model', MODEL_DEFAULTS)
                    is None
                )
                or not model_config.get('custom_model', '').endswith('_attn')
            )
    ):
        # No attention; use standard API.
        return state

    # Manually calculate attention state.
    # See
    # https://github.com/ray-project/ray/issues/14548#issuecomment-793667603
    # and
    # https://github.com/ray-project/ray/issues/14548#issuecomment-801969493.
    view_reqs = policy.view_requirements
    states = []
    for i in itertools.count(0):
        key = 'state_in_' + str(i)
        if key not in view_reqs:
            break

        view_req = view_reqs[key]
        state = np.zeros(
            shape=(
                (utils.get_default(
                    model_config,
                    'attention_memory_inference',
                    MODEL_DEFAULTS,
                ),)
                + view_req.space.shape
            ),
            dtype=view_req.space.dtype,
        )
        states.append(state)
    return states


def get_initial_states(
        agent: Trainable,
        policy_mapping_fn: Callable[[AgentId], PolicyID],
        num_players: int,
) -> List[List[TensorType]]:
    """Return the initial recurrent state for each player with their
    corresponding agent policy.

    See also `get_initial_state`.

    Args:
        agent (Trainable): Reinforcement learning trainer/agent.
        policy_mapping_fn (Callable[[AgentId], PolicyID]): Function
            mapping player indices to policy IDs.
        num_players (int): Amount of players in the game.

    Returns:
        List[List[TensorType]]]: Initial recurrent state for each player
            according to their policy. Sorted by player indices.
    """
    states = []
    for agent_id in range(num_players):
        policy_id = policy_mapping_fn(agent_id)
        state = get_initial_state(agent, policy_id)
        states.append(state)
    return states


def compute_actions(self,
                    observations,
                    state=None,
                    prev_action=None,
                    prev_reward=None,
                    info=None,
                    policy_id=DEFAULT_POLICY_ID,
                    full_fetch=False,
                    explore=None):
    """Computes an action for the specified policy on the local Worker.

    Note that you can also access the policy object through
    self.get_policy(policy_id) and call compute_actions() on it directly.

    Args:
        observation (obj): observation from the environment.
        state (dict): RNN hidden state, if any. If state is not None,
            then all of compute_single_action(...) is returned
            (computed action, rnn state(s), logits dictionary).
            Otherwise compute_single_action(...)[0] is returned
            (computed action).
        prev_action (obj): previous action value, if any
        prev_reward (int): previous reward, if any
        info (dict): info object, if any
        policy_id (str): Policy to query (only applies to multi-agent).
        full_fetch (bool): Whether to return extra action fetch results.
            This is always set to True if RNN state is specified.
        explore (bool): Whether to pick an exploitation or exploration
            action (default: None -> use self.config["explore"]).

    Returns:
        any: The computed action if full_fetch=False, or
        tuple: The full output of policy.compute_actions() if
            full_fetch=True or we have an RNN-based Policy.
    """
    # Preprocess obs and states
    stateDefined = state is not None
    policy = self.get_policy(policy_id)
    filtered_obs, filtered_state = [], []
    for (i, ob) in enumerate(observations):
        worker = self.workers.local_worker()
        preprocessed = worker.preprocessors[policy_id].transform(ob)
        filtered = worker.filters[policy_id](preprocessed, update=False)
        filtered_obs.append(filtered)
        if state is None:
            continue
        elif len(state) > i:
            filtered_state.append(state[i])
        else:
            filtered_state.append(policy.get_initial_state())

    # Batch obs and states
    obs_batch = np.stack(filtered_obs)
    if state is None:
        state = []
    else:
        state = list(zip(*filtered_state))
        state = [np.stack(s) for s in state]

    # Batch compute actions
    actions, states, infos = policy.compute_actions(
        obs_batch,
        state,
        prev_action,
        prev_reward,
        info,
        clip_actions=self.config["clip_actions"],
        explore=explore)

    # Unbatch actions for the environment
    atns, actions = space_utils.unbatch(actions), []
    for atn in atns:
        actions.append(atn)

    # Unbatch states into a dict
    unbatched_states = []
    for idx in range(len(observations)):
        unbatched_states.append([s[idx] for s in states])

    # Return only actions or full tuple
    if stateDefined or full_fetch:
        return actions, unbatched_states, infos
    else:
        return actions


def _get_num_players(config: TrainerConfigDict) -> int:
    env_config = utils.get_default(config, 'env_config', COMMON_CONFIG)
    num_players = env_config.get('num_players', 4)
    return num_players


def _setup_eval_vars(num_players: int) -> EvalResults:
    total_penalties = [0] * num_players
    total_placements = [[0] * num_players for _ in range(num_players)]

    num_actions = 0
    num_illegal = 0

    test_start_time = time.perf_counter()

    return (
        total_penalties,
        total_placements,
        num_actions,
        num_illegal,
        test_start_time,
    )


def _eval_stable(
        agent: Trainable,
        env_name: str,
        eval_config: TrainerConfigDict,
        num_test_games: int,
        learned_agent_id: int,
) -> EvalResults:
    num_players = _get_num_players(eval_config)
    (
        total_penalties,
        total_placements,
        num_actions,
        num_illegal,
        test_start_time,
    ) = _setup_eval_vars(num_players)

    with NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile_name = tmpfile.name

    with RolloutSaver(
            tmpfile_name,
            target_episodes=num_test_games,
            save_info=True,
    ) as saver, contextlib.redirect_stdout(None):
        ray.rllib.rollout.rollout(
            agent,
            env_name,
            None,
            num_test_games,
            saver,
        )

    with open(tmpfile_name, 'rb') as tmpfile:
        rollouts = pickle.load(tmpfile)

    # Clean up temporary file
    os.unlink(tmpfile_name)

    for ep in range(num_test_games):
        for rollout in rollouts[ep]:
            info = rollout[-1]
            info = info[next(iter(info.keys()))]
            if info['prev_active_player_index'] != learned_agent_id:
                continue
            num_actions += 1
            num_illegal += info['was_illegal']

        final_rollout = rollouts[ep][-1]
        # We could use any index here since all agents get the same
        # final observation.
        info = final_rollout[-1][learned_agent_id]

        for (i, penalty) in enumerate(info['final_penalties']):
            total_penalties[i] += penalty
        for (i, ranking) in enumerate(info['final_rankings']):
            total_placements[i][ranking - 1] += 1

    test_duration = time.perf_counter() - test_start_time
    return (
        total_penalties,
        total_placements,
        num_actions,
        num_illegal,
        test_duration,
    )


def _eval_unstable(
        agent: Trainable,
        env_name: str,
        eval_config: TrainerConfigDict,
        num_test_games: int,
        learned_agent_id: int,
) -> EvalResults:
    num_players = _get_num_players(eval_config)
    eval_policy_mapping_fn = \
        eval_config['multiagent']['policy_mapping_fn']

    (
        total_penalties,
        total_placements,
        num_actions,
        num_illegal,
        test_start_time,
    ) = _setup_eval_vars(num_players)

    # TODO use batching for more speed; see `eval_agent.py`

    model_config = utils.get_default(eval_config, 'model', COMMON_CONFIG)
    make_env = utils.get_registered_env(env_name)
    env = make_env(utils.get_default(eval_config, 'env_config', COMMON_CONFIG))

    for i in range(num_test_games):
        states = get_initial_states(
            agent, eval_policy_mapping_fn, num_players)
        prev_actions: List[Optional[TensorType]] = [None] * num_players
        prev_rewards: List[Optional[Reward]] = [None] * num_players

        obs = env.reset()
        is_done: MultiIsDone = {'__all__': False}
        while not is_done['__all__']:
            assert len(obs) == 1, 'encountered multiple ready agents'
            agent_id = next(iter(obs.keys()))
            policy_id = eval_policy_mapping_fn(agent_id)
            action, state, _ = agent.compute_action(
                obs[agent_id],
                states[agent_id],
                prev_actions[agent_id],
                prev_rewards[agent_id],
                policy_id=policy_id,
                full_fetch=True,
            )
            info: MultiInfo
            obs, reward, is_done, info = env.step({agent_id: action})

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
                for (i, prev_state) in enumerate(states[agent_id]):
                    states[agent_id][i] = np.vstack((prev_state[1:], state[i]))
            else:
                states[agent_id] = state
            prev_actions[agent_id] = action
            next_agent_id = env.active_player_index
            prev_rewards[next_agent_id] = reward[next_agent_id]

            info: Info = info[next(iter(info.keys()))]
            if info['prev_active_player_index'] != learned_agent_id:
                continue
            num_actions += 1
            num_illegal += info['was_illegal']

        for (i, penalty) in enumerate(info['final_penalties']):
            total_penalties[i] += penalty
        for (i, ranking) in enumerate(info['final_rankings']):
            total_placements[i][ranking - 1] += 1

    test_duration = time.perf_counter() - test_start_time
    return (
        total_penalties,
        total_placements,
        num_actions,
        num_illegal,
        test_duration,
    )


def evaluate(
        use_stable_method: bool,
        agent: Trainable,
        env_name: str,
        eval_config: TrainerConfigDict,
        num_test_games: int,
        learned_agent_id: int,
) -> EvalResults:
    # Unstable method is a faster, re-implemented version. That may
    # sometimes even offer better support.
    if use_stable_method:
        return _eval_stable(
            agent,
            env_name,
            eval_config,
            num_test_games,
            learned_agent_id,
        )
    else:
        return _eval_unstable(
            agent,
            env_name,
            eval_config,
            num_test_games,
            learned_agent_id,
        )


def _strlen(x: Any) -> int:
    """Return the length of the string-representation of the
    given object.

    Args:
        x (Any): Object to get string length of.

    Returns:
        int: Length of the string-representaton of the given object.
    """
    return len(str(x))


def create_results_table(
        total_penalties: List[int],
        total_placements: List[List[int]],
        policy_mapping_fn: Callable[[AgentId], PolicyID],
        num_illegals: Optional[List[int]] = None,
) -> str:
    """Return a string table summarizing the given results.

    Args:
        total_penalties (List[int]): Total penalties for each player,
            sorted by player index.
        total_placements (List[List[int]]): Total amount of each ranking
            sorted by ranking for each player, sorted by player index.
        policy_mapping_fn (Callable[[AgentId], PolicyID]): Function
            mapping agent IDs to policy IDs.

    Returns:
        str: Table-formatted string summarizing the given results.
    """
    num_players = len(total_penalties)
    agent_names = [policy_mapping_fn(i) for i in range(num_players)]

    header = ['policy']
    header.extend(f'# rank {i}' for i in range(1, num_players + 1))
    header.append('total penalty')
    if num_illegals is not None:
        header.append('# illegal actions')

    longest_agent_name = max(map(_strlen, agent_names))
    longest_placements = [
        max(map(_strlen, map(lambda x: x[i], total_placements)))
        for i in range(num_players)
    ]
    longest_penalty = max(map(_strlen, total_penalties))

    longest_in_cols = [longest_agent_name]
    longest_in_cols.extend(longest_placements)
    longest_in_cols.append(longest_penalty)
    if num_illegals is not None:
        longest_num_illegals = max(map(_strlen, num_illegals))
        longest_in_cols.append(longest_num_illegals)
    longest_in_cols: List[int] = list(map(max, zip(  # type: ignore[arg-type]
        longest_in_cols,
        map(_strlen, header),
    )))

    row_formatter = ''.join([
        '| ',
        f'{{:<{longest_in_cols[0]}}}',
        ' | ',
        ' | '.join(f'{{:>{width}}}' for width in longest_in_cols[1:]),
        ' |',
    ])
    header_separator = ''.join([
        '|',
        '+'.join('-' * (width + 2) for width in longest_in_cols),
        '|',
    ])

    table = [
        row_formatter.format(*header),
        header_separator,
    ]

    table_values = [
        agent_names,
        total_placements,
        total_penalties,
    ]
    if num_illegals is not None:
        table_values.append(num_illegals)

    for row in zip(*table_values):
        name, placements = row[:2]
        table.append(row_formatter.format(name, *placements, *row[2:]))

    return '\n'.join(table)
