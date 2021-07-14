import numpy as np
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override

from hearts_gym.envs import HeartsEnv
from hearts_gym.envs.card_deck import Card


class RuleBasedPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mask_actions = self.config.get('mask_actions', True)
        self._mask_actions = mask_actions

        self._setup_variables()

    def _setup_variables(self):
        original_space = self.observation_space.original_space

        if self._mask_actions:
            original_obs_space = original_space[HeartsEnv.OBS_KEY]

            action_mask_space = original_space[HeartsEnv.ACTION_MASK_KEY]
            self._action_mask_len = np.prod(action_mask_space.shape)
        else:
            original_obs_space = original_space

        self._num_cards = np.prod(original_obs_space['cards'].shape)
        self._cards_per_suit = self._num_cards // Card.NUM_SUITS

    def _split_obs_and_mask(self, obs_batch):
        action_mask = obs_batch[:, :self._action_mask_len]
        sans_action_mask = obs_batch[:, self._action_mask_len:]
        return sans_action_mask, action_mask

    def _index_to_card(self, index):
        suit = index // self._cards_per_suit
        rank = index - suit * self._cards_per_suit
        return Card(suit, rank)

    def _suit_from_index(self, index):
        return index // self._cards_per_suit

    @override(Policy)
    def compute_actions(
            self,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch=None,
            **kwargs,
    ):
        if isinstance(obs_batch, list):
            obs_batch = np.array(obs_batch)
        if self._mask_actions:
            obs_batch, action_masks = self._split_obs_and_mask(obs_batch)
            raise NotImplementedError('please implement the rule-based agent')
        else:
            raise NotImplementedError('please implement the rule-based agent')

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
            self,
            actions,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
    ):
        return np.ones(size=(len(obs_batch), 1))

    @override(Policy)
    def get_weights(self):
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights):
        """No weights to set."""
        pass
