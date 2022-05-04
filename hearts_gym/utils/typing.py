from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.utils.typing import PolicyID

Seed = Union[None, int, float, str, bytes, bytearray]

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
# This is mapping to `Any` due to observation transformations.
MultiObservation = Dict[AgentId, Any]
Reward = Real
MultiReward = Dict[AgentId, Reward]
IsDone = bool
MultiIsDone = Dict[Union[AgentId, str], IsDone]
Info = Dict[str, Any]
MultiInfo = Dict[AgentId, Info]

PolicyMappingFn = Callable[
    [AgentId, Optional[Episode], Optional[RolloutWorker]],
    PolicyID,
]
