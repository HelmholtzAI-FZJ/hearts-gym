"""
Train and evaluate an agent in a multi-agent setting using RLlib.
"""

import os
from typing import Callable

import ray
from ray import tune
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.utils.typing import PolicyID, TrainerConfigDict

import configuration as conf
from configuration import (
    ENV_NAME,
    LEARNED_AGENT_ID,
    LEARNED_POLICY_ID,
    RANDOM_POLICY_ID,
)
import hearts_gym
from hearts_gym import utils
from hearts_gym.utils.typing import AgentId, Seed


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

    env_config = utils.get_default(
        eval_config, 'env_config', COMMON_CONFIG).copy()
    eval_config['env_config'] = env_config
    env_config['seed'] = seed

    multiagent_config = utils.get_default(
        eval_config, 'multiagent', COMMON_CONFIG).copy()
    eval_config['multiagent'] = multiagent_config
    multiagent_config['policy_mapping_fn'] = policy_mapping_fn

    policies_config = utils.get_default(
        multiagent_config, 'policies', COMMON_CONFIG['multiagent']).copy()
    multiagent_config['policies'] = policies_config
    if RANDOM_POLICY_ID in policies_config:
        random_policy = policies_config[RANDOM_POLICY_ID]
        random_policy_config = random_policy[3]
        random_policy_config = {
            **random_policy_config,
            'seed': seed,
        }
        policies_config[RANDOM_POLICY_ID] = \
            random_policy[:3] + (random_policy_config,) + random_policy[4:]

    eval_config['num_gpus'] = (
        utils.get_num_gpus(
            utils.get_default(eval_config, 'framework', COMMON_CONFIG))
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


def main() -> None:
    """Train and evaluate an agent in a multi-agent setting using RLlib."""

    # This most likely does not work due to RLlib issues.
    reset_workers = False
    """Experimental flag to be able to use workers for evaluation."""

    if reset_workers:
        utils.fix_ray_shutdown()

    ray.init()

    utils.maybe_set_up_masked_actions_model(conf.algorithm, conf.config)

    if any(
            conf.eval_policy_mapping_fn(agent_id) == LEARNED_POLICY_ID
            for agent_id in range(conf.num_players)
    ):
        assert (
            conf.eval_policy_mapping_fn(LEARNED_AGENT_ID) == LEARNED_POLICY_ID
        ), 'agent index does not match policy with name "learned"'
    else:
        print('Warning: you are not evaluating a learned policy; '
              'modify `eval_policy_mapping` to change this')

    assert (
        conf.checkpoint_path is None
        or os.path.isfile(conf.checkpoint_path)
    ), 'please pass the checkpoint file, not its directory'

    analysis = tune.run(
        conf.algorithm,
        stop=conf.stop_config,
        config=conf.config,
        metric=conf.opt_metric,
        mode=conf.opt_mode,
        local_dir=conf.RESULTS_DIR,
        checkpoint_at_end=True,
        scheduler=conf.scheduler,
        resume=conf.resume,
        restore=conf.checkpoint_path,
    )

    # Testing

    best_cp = analysis.get_best_checkpoint(
        trial=analysis.get_best_trial(conf.opt_metric),
        metric=conf.opt_metric,
    )
    # last_cp = analysis.get_last_checkpoint()
    print('Using best checkpoint for evaluation:', best_cp)

    if reset_workers:
        # TODO Even with a reset, the workers are not properly cleaned up.

        # Reset so we free the workers
        ray.shutdown()
        ray.init()

        # Re-register our stuff
        hearts_gym.register_envs()
        utils.maybe_set_up_masked_actions_model(conf.algorithm, conf.config)

    eval_config = configure_eval(
        conf.config,
        conf.eval_seed,
        conf.eval_policy_mapping_fn,
        reset_workers,
    )
    agent = utils.load_agent(conf.algorithm, best_cp, eval_config)

    print('Running', conf.num_test_games, 'test games...')
    (
        total_penalties,
        total_placements,
        num_actions,
        num_illegal,
        test_duration,
    ) = utils.evaluate(
        conf.use_stable_method,
        agent,
        ENV_NAME,
        eval_config,
        conf.num_test_games,
        LEARNED_AGENT_ID,
    )

    print('testing took', test_duration, 'seconds')
    print(f'# illegal action (player {LEARNED_AGENT_ID}):',
          num_illegal, '/', num_actions)
    print(f'# illegal action ratio (player {LEARNED_AGENT_ID}):',
          'NaN' if num_actions == 0 else num_illegal / num_actions)
    print(utils.create_results_table(
        total_penalties, total_placements, conf.eval_policy_mapping_fn))

    ray.shutdown()


if __name__ == '__main__':
    main()
