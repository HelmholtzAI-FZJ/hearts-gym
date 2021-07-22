from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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
