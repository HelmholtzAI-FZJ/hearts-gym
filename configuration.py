from typing import Dict, List, Optional

from ray import tune

from hearts_gym import utils
from hearts_gym.utils.obs_transforms import ObsTransform

assert utils.DEFAULT_FRAMEWORK is not None

RESULTS_DIR = './results'

ENV_NAME = 'Hearts-v0'

LEARNED_AGENT_ID = 0
"""Agent ID of the learned policy."""

LEARNED_POLICY_ID = 'learned'
RANDOM_POLICY_ID = 'random'
RULEBASED_POLICY_ID = 'rulebased'


allow_pickles = True
"""Whether to allow loading parameter pickle files.

When set to `True`, this is a security hole when receiving untrusted
checkpoints due to arbitrary code execution. Trade-off between safety
and convenience.
"""

# By default: "tf", "torch", or "jax", whichever is available (in that order).
framework: str = utils.DEFAULT_FRAMEWORK

custom_rulebased_policies: Dict[str, type] = {}
"""Dictionary of custom rule-based policies.

Mapping from policy IDs to classes (not class instances!) implementing
`hearts_gym.policies.deterministic_policy_impl.DeterministicPolicyImpl`.
"""

obs_transforms: List[ObsTransform] = []


# Environment config

num_players = 4
deck_size = 52
seed = 0
mask_actions = True

# The following is a simple example for a custom `policy_mapping_fn`
# for a four-player game:
#
# def policy_mapping_fn(player_index):
#     return {
#         0: LEARNED_AGENT_ID,
#         1: RANDOM_AGENT_ID,
#         2: RULEBASED_AGENT_ID,
#         3: LEARNED_AGENT_ID,
#     }[player_index]
policy_mapping_fn = utils.create_policy_mapping(
    'all_learned',
    # 'one_learned_rest_random',
    LEARNED_AGENT_ID,
    LEARNED_POLICY_ID,
    RANDOM_POLICY_ID,
    RULEBASED_POLICY_ID,
)

random_policy_seed = None


# Test config

eval_seed = seed + 1
num_test_games = 5000
eval_policy_mapping_fn = utils.create_policy_mapping(
    'one_learned_rest_random',
    LEARNED_AGENT_ID,
    LEARNED_POLICY_ID,
    RANDOM_POLICY_ID,
    RULEBASED_POLICY_ID,
)

use_stable_method = False
"""Whether to use RLlib's implementation ('stable') or a
re-implementation ('unstable') for model evaluation.

The unstable method is a faster, re-implemented version. Due to that,
it may sometimes even offer better support.
"""


# RLLib config

algorithm = 'PPO'
checkpoint_path: Optional[str] = None
"""Path of a checkpoint to load. Use `None` to not load a checkpoint."""
resume = False
"""Whether to resume the most recent run."""

env_config = {
    'num_players': num_players,
    'deck_size': deck_size,
    'seed': seed,
    'mask_actions': mask_actions,
    'obs_transforms': obs_transforms,
}

model_config = {
    # 'use_lstm': True,
    # 'use_attention': True,
    'max_seq_len': deck_size // num_players,
    'custom_model': None,
}


# Tune config

opt_metric: str = 'episode_reward_mean'
opt_mode: str = 'max'

stop_config = {
    'timesteps_total': 2000000,
}

scheduler = tune.schedulers.FIFOScheduler()

config = {
    'env': ENV_NAME,
    'env_config': env_config,
    'model': model_config,
    'multiagent': {
        'policies_to_train': [LEARNED_POLICY_ID],
        'policies': {
            **utils.default_policies(
                ENV_NAME,
                env_config,
                LEARNED_POLICY_ID,
                RANDOM_POLICY_ID,
                RULEBASED_POLICY_ID,
                random_policy_seed,
            ),
            **utils.create_custom_rulebased_policies(
                ENV_NAME,
                env_config,
                custom_rulebased_policies,
            ),
        },
        'policy_mapping_fn': policy_mapping_fn,
    },
    'num_gpus': utils.get_num_gpus(framework),
    'num_workers': utils.get_num_cpus() - 1,
    'framework': framework,

    # 'lr': 3e-4,
    # 'gamma': 0.999,
    # 'sgd_minibatch_size': 512,
}
