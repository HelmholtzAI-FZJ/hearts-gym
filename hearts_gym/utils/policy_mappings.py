from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.utils.typing import PolicyID

from hearts_gym.utils.typing import Any, AgentId, Optional, PolicyMappingFn


def create_policy_mapping(
        policy_mapping_name: str,
        learned_agent_id: AgentId,
        learned_policy_id: PolicyID,
        random_policy_id: PolicyID,
        rulebased_policy_id: PolicyID,
) -> PolicyMappingFn:
    """Return a policy mapping function where the given learned agent ID
    may be treated specially.

    `policy_mapping_name` may be one of the following:

    - 'one_learned_rest_random': `learned_agent_id` uses the learned
      policy, the rest is random.
    - 'one_learned_rest_rulebased': `learned_agent_id` uses the learned
      policy, the rest is rule-based.
    - 'all_learned': All agents use the learned policy.
    - 'all_random': All agents use the randomly acting policy.
    - 'all_rulebased': All agents use the rule-based policy.

    Args:
        policy_mapping_name (str): Name of the policy mapping to create.
        learned_agent_id (AgentId): ID or index of the 'main' learned
            agent, i.e. the one that is always kept, also
            for evaluation.
        learned_policy_id (PolicyID): ID of the learned policy.
        random_policy_id (PolicyID): ID of the random policy.
        rulebased_policy_id (PolicyID): ID of the rule-based policy.

    Returns:
        PolicyMappingFn: Desired policy mapping function.
    """

    if policy_mapping_name == 'one_learned_rest_random':
        def policy_mapping_one_learned_rest_random(
                agent_id: AgentId,
                episode: Optional[Episode],
                worker: Optional[RolloutWorker],
                **kwargs: Any,
        ) -> PolicyID:
            """Return the ID for a learned policy for the agent with
            `learned_agent_id`, otherwise for a randomly acting policy.

            Args:
                agent_id (AgentId): Agent ID to get the policy for.

            Returns:
                PolicyID: ID of the policy for the queried agent.
            """
            if agent_id == learned_agent_id:
                return learned_policy_id
            return random_policy_id

        return policy_mapping_one_learned_rest_random
    elif policy_mapping_name == 'one_learned_rest_rulebased':
        def policy_mapping_one_learned_rest_rulebased(
                agent_id: AgentId,
                episode: Optional[Episode],
                worker: Optional[RolloutWorker],
                **kwargs: Any,
        ) -> PolicyID:
            """Return the ID for a learned policy for the agent with
            `learned_agent_id`, otherwise for a rule-based policy.

            Args:
                agent_id (AgentId): Agent ID to get the policy for.

            Returns:
                PolicyID: ID of the policy for the queried agent.
            """
            if agent_id == learned_agent_id:
                return learned_policy_id
            return rulebased_policy_id

        return policy_mapping_one_learned_rest_rulebased
    elif policy_mapping_name == 'all_learned':
        def policy_mapping_all_learned(
                agent_id: AgentId,
                episode: Optional[Episode],
                worker: Optional[RolloutWorker],
                **kwargs: Any,
        ) -> PolicyID:
            """Always return a learned policy.

            Returns:
                PolicyID: A learned policy.
            """
            return learned_policy_id

        return policy_mapping_all_learned
    elif policy_mapping_name == 'all_random':
        def policy_mapping_all_random(
                agent_id: AgentId,
                episode: Optional[Episode],
                worker: Optional[RolloutWorker],
                **kwargs: Any,
        ) -> PolicyID:
            """Always return a randomly acting policy.

            Returns:
                PolicyID: A randomly acting policy.
            """
            return random_policy_id

        return policy_mapping_all_random
    elif policy_mapping_name == 'all_rulebased':
        def policy_mapping_all_rulebased(
                agent_id: AgentId,
                episode: Optional[Episode],
                worker: Optional[RolloutWorker],
                **kwargs: Any,
        ) -> PolicyID:
            """Always return a rule-based policy.

            Returns:
                PolicyID: A policy acting with hardcoded rules.
            """
            return rulebased_policy_id

        return policy_mapping_all_rulebased
    else:
        raise NotImplementedError(
            f'unknown policy mapping: {policy_mapping_name}')
