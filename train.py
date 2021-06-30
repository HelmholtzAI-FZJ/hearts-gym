"""
Train and evaluate an agent in a multi-agent setting using RLlib.
"""

import os
from typing import Any, Callable, List

import ray
from ray import tune
from ray.rllib.utils.typing import PolicyID, TrainerConfigDict

import hearts_gym
from hearts_gym import utils
from hearts_gym.envs.card_deck import Seed
from hearts_gym.envs.hearts_env import AgentId
from hearts_gym.policies import RandomPolicy, RuleBasedPolicy

LEARNED_AGENT_ID = 0
"""Agent ID of the learned policy."""
LEARNED_POLICY_ID = 'learned'

# FIXME argument parsing


def configure_eval(
        config: TrainerConfigDict,
        seed: Seed,
        policy_mapping_fn: Callable[[AgentId], PolicyID],
        reset_workers: bool,
) -> TrainerConfigDict:
    """Return the given configuration modified so it has settings useful
    for evaluation.

    Args:
        config (TrainerConfigDict): RLlib configuration to set up
            for evaluation.
        seed (Seed): Random number generator base seed for evaluation.
        policy_mapping_fn (Callable[[AgentId], PolicyID]): Policy
            mapping for evaluation.
        reset_workers (bool): Whether workers were reset and can be used
            for evaluation.

    Returns:
        TrainerConfigDict: Evaluation configuration based on the
            given one.
    """
    eval_config = utils.configure_eval(config)

    env_config = eval_config.get('env_config', {}).copy()
    eval_config['env_config'] = env_config
    env_config['seed'] = seed

    multiagent_config = eval_config.get('multiagent', {}).copy()
    multiagent_config['multiagent'] = multiagent_config
    multiagent_config['policy_mapping_fn'] = policy_mapping_fn

    eval_config['num_gpus'] = (
        utils.get_num_gpus(eval_config.get('framework', 'tf'))
        if reset_workers
        else 0
    )
    eval_config['num_workers'] = (
        utils.get_num_cpus() - 1
        if reset_workers
        else 0
    )

    # These settings did not play nice with the stable
    # evaluation method.
    # eval_config['evaluation_num_workers'] = utils.get_num_cpus() - 1
    # eval_config['evaluation_num_episodes'] = 1
    # eval_config['evaluation_config'] = env_config

    return eval_config


def _strlen(x: Any) -> int:
    """Return the length of the string-representation of the
    given object.

    Args:
        x (Any): Object to get string length of.

    Returns:
        int: Length of the string-representaton of the given object.
    """
    return len(str(x))


def print_results_table(
        total_penalties: List[int],
        total_placements: List[List[int]],
        policy_mapping_fn: Callable[[AgentId], PolicyID],
) -> None:
    """Print a table summarizing the given results.

    Args:
        total_penalties (List[int]): Total penalties for each player,
            sorted by player index.
        total_placements (List[List[int]]): Total amount of each ranking
            sorted by ranking for each player, sorted by player index.
        policy_mapping_fn (Callable[[AgentId], PolicyID]): Function
            mapping agent IDs to policy IDs.
    """
    num_players = len(total_penalties)
    agent_names = [policy_mapping_fn(i) for i in range(num_players)]

    header = ['policy']
    header.extend(f'# rank {i}' for i in range(1, num_players + 1))
    header.append('total penalty')

    longest_agent_name = max(map(_strlen, agent_names))
    longest_placements = [
        max(map(_strlen, map(lambda x: x[i], total_placements)))
        for i in range(num_players)
    ]
    longest_penalty = max(map(_strlen, total_penalties))

    longest_in_cols = [longest_agent_name]
    longest_in_cols.extend(longest_placements)
    longest_in_cols.append(longest_penalty)
    longest_in_cols = list(map(max, zip(  # type: ignore[arg-type]
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

    print(row_formatter.format(*header))
    print(header_separator)
    for (name, placements, penalty) in zip(
            agent_names,
            total_placements,
            total_penalties,
    ):
        print(row_formatter.format(name, *placements, penalty))


def main() -> None:
    """Train and evaluate an agent in a multi-agent setting using RLlib."""

    # This most likely does not work due to RLlib issues.
    reset_workers = False

    if reset_workers:
        utils.fix_ray_shutdown()

    ray.init()

    # "tf", "torch", or "jax", whichever is available (in that order).
    framework = utils.DEFAULT_FRAMEWORK

    # Environment config

    num_players = 4
    deck_size = 52
    seed = 0
    mask_actions = True

    policy_mapping_fn = create_policy_mapping(
        'all_learned',
        # 'one_learned_rest_random',
        LEARNED_AGENT_ID,
        LEARNED_POLICY_ID,
        'random',
        'rulebased',
    )

    # Test config

    eval_seed = seed + 1
    num_test_games = 5000
    eval_policy_mapping_fn = create_policy_mapping(
        'one_learned_rest_random',
        LEARNED_AGENT_ID,
        LEARNED_POLICY_ID,
        'random',
        'rulebased',
    )

    # Unstable method is a faster, re-implemented version. Due to that,
    # it may sometimes even offer better support.
    use_stable_method = False

    # RLLib config

    algorithm = 'PPO'
    # Use `None` to not load a checkpoint.
    checkpoint_path = None

    env_name = 'Hearts-v0'
    env_config = {
        'num_players': num_players,
        'deck_size': deck_size,
        'seed': seed,
        'mask_actions': mask_actions,
    }

    obs_space, act_space = utils.get_spaces(env_name, env_config)

    opt_metric = 'episode_reward_mean'
    opt_mode = 'max'

    stop_config = {
        'timesteps_total': 2000000,
    }

    scheduler = tune.schedulers.FIFOScheduler()
    # scheduler = tune.schedulers.ASHAScheduler(
    #     time_attr='timesteps_total',
    # )

    model_config = {
        # 'fcnet_hiddens': tune.grid_search([
        #     [64, 64],
        #     [64, 128],
        #     [128, 64],
        #     [128, 128],
        #     [256, 128],
        #     [128, 256],
        #     [256, 256],
        #     [512, 256],
        #     [256, 512],
        #     [512, 512],
        # ]),
        # 'fcnet_activation': tune.grid_search(['tanh', 'relu']),

        # 'use_lstm': tune.grid_search([True, False]),
        # 'use_attention': True,
        'max_seq_len': deck_size // num_players,
        'custom_model': None,
    }

    config = {
        'env': env_name,
        'env_config': env_config,
        'model': model_config,
        'multiagent': {
            'policies_to_train': [LEARNED_POLICY_ID],
            'policies': {
                LEARNED_POLICY_ID: (None, obs_space, act_space, {}),
                'random': (RandomPolicy, obs_space, act_space,
                           {'mask_actions': mask_actions}),
                'rulebased': (RuleBasedPolicy, obs_space, act_space,
                              {'mask_actions': mask_actions}),
            },
            'policy_mapping_fn': policy_mapping_fn,
        },
        'num_gpus': utils.get_num_gpus(framework),
        'num_workers': utils.get_num_cpus() - 1,
        'framework': framework,

        # 'lr': tune.loguniform(1e-5, 1e-1),
        # 'gamma': tune.uniform(0.9, 1.0),
        # 'sgd_minibatch_size': tune.grid_search(
        #     [32, 64, 128, 256, 512, 1024]),
    }
    utils.maybe_set_up_masked_actions_model(algorithm, config)

    if any(
            eval_policy_mapping_fn(agent_id) == LEARNED_POLICY_ID
            for agent_id in range(num_players)
    ):
        assert eval_policy_mapping_fn(LEARNED_AGENT_ID) == LEARNED_POLICY_ID, \
            'agent index does not match policy with name "learned"'
    else:
        print('Warning: you are not evaluating a learned policy; '
              'modify `eval_policy_mapping` to change this')

    assert checkpoint_path is None or os.path.isfile(checkpoint_path), \
        'please pass the checkpoint file, not its directory'

    analysis = tune.run(
        algorithm,
        stop=stop_config,
        config=config,
        metric=opt_metric,
        mode=opt_mode,
        local_dir='./results',
        checkpoint_at_end=True,
        scheduler=scheduler,
        # resume=True,  # Resume the most recent run.
        restore=checkpoint_path,
    )

    # Testing

    best_cp = analysis.get_best_checkpoint(
        trial=analysis.get_best_trial(opt_metric),
        metric=opt_metric,
    )
    # last_cp = analysis.get_last_checkpoint()
    print('best cp:', best_cp)

    if reset_workers:
        # FIXME Even with a reset, the workers are not properly cleaned up.

        # Reset so we free the workers
        ray.shutdown()
        ray.init()

        # Re-register our stuff
        hearts_gym.register_envs()
        utils.maybe_set_up_masked_actions_model(algorithm, config)

    eval_config = configure_eval(
        config,
        eval_seed,
        eval_policy_mapping_fn,
        reset_workers,
    )
    agent = utils.load_agent(algorithm, best_cp, eval_config)

    (
        total_penalties,
        total_placements,
        num_actions,
        num_illegal,
        test_duration,
    ) = utils.evaluate(
        use_stable_method,
        agent,
        env_name,
        eval_config,
        num_test_games,
        LEARNED_AGENT_ID,
    )

    print('testing took', test_duration, 'seconds')
    print(f'# illegal action (player {LEARNED_AGENT_ID}):',
          num_illegal, '/', num_actions)
    print(f'# illegal action ratio (player {LEARNED_AGENT_ID}):',
          'NaN' if num_actions == 0 else num_illegal / num_actions)
    print_results_table(
        total_penalties, total_placements, eval_policy_mapping_fn)

    ray.shutdown()


if __name__ == '__main__':
    main()
