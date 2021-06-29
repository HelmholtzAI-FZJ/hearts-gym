"""
Helpers for evaluating an agent in a multi-agent setting using RLlib.
"""

import contextlib
import itertools
import os
import pickle
from tempfile import NamedTemporaryFile
import time
from typing import Callable, Optional, Tuple

import numpy as np
import ray
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

from hearts_gym.envs.hearts_env import (
    AgentId,
    Info,
    MultiInfo,
    MultiIsDone,
    Reward,
)
import hearts_gym.utils.common as utils

__all__ = [
    'EvalResults',
    'configure_eval',
    'get_initial_state',
    'get_initial_states',
    'compute_actions',
    'evaluate',
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

    multiagent_config = eval_config.get('multiagent', {}).copy()
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
    model_config = agent.config.get('model', {})

    state = policy.get_initial_state()
    if (
            state
            or not model_config.get('use_attention', False)
            and (
                model_config.get('custom_model', None) is None
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
                (model_config.get('attention_memory_inference', 50),)
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
    env_config = config.get('env_config', {})
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

    # FIXME use batching for more speed

    model_config = eval_config.get('model', {})
    make_env = utils.get_registered_env(env_name)
    env = make_env(eval_config.get('env_config', {}))

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
                    model_config.get('use_attention', False)
                    or (
                        model_config.get('custom_model', None) is not None
                        and model_config.get(
                            'custom_model', '').endswith('_attn')
                    )
            ):
                for (i, state) in enumerate(states[agent_id]):
                    states[agent_id][i] = np.vstack((state[1:], state[i]))
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
