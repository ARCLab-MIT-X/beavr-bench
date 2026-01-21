"""Base classes and factory for task rules."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import draccus
import mujoco
import numpy as np

from beavr_bench.schemas import (
    TaskRulesConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """Result from rule computation.

    Note:
        The 'info' dictionary must contain only JSON-serializable native Python types
        (int, float, str, bool, list, dict). Avoid passing NumPy types (np.int64,
        np.float32, np.ndarray) directly as they may crash network service servers.
    """

    reward: float
    terminated: bool  # Episode should end (success or failure)
    truncated: bool  # Episode hit max steps
    info: dict[str, Any]  # Extra info (use RuleInfoKey for keys)


class BaseRules(ABC):
    """Abstract base class for task rules.

    Caches MuJoCo body/site IDs at init for fast lookup during step.
    """

    _registry: dict[type[TaskRulesConfig], type[BaseRules]] = {}

    @classmethod
    def register(cls, config_type: type[TaskRulesConfig]):
        """Decorator to register a rule implementation for a specific config type."""

        def wrapper(rule_cls: type[BaseRules]):
            cls._registry[config_type] = rule_cls
            return rule_cls

        return wrapper

    def __init__(self, config: TaskRulesConfig, model: mujoco.MjModel):
        self.config = config
        self.model = model
        self._step_count = 0

    @abstractmethod
    def compute(self, data: mujoco.MjData) -> RuleResult:
        """Compute reward and termination for current state."""
        pass

    @abstractmethod
    def randomize_scene(self, data: mujoco.MjData, np_random: np.random.Generator) -> None:
        """Randomize object positions at episode start using the provided RNG.

        Default implementation does nothing. Subclasses should override if
        they require specific object randomization.
        """
        pass

    def reset(self) -> None:
        """Reset rule state at episode start."""
        self._step_count = 0


def create_rules(config: TaskRulesConfig | dict | None, model: mujoco.MjModel) -> BaseRules | None:
    """Factory function to create rules from config.

    Args:
        config: Task rules configuration (dataclass, dict, or None)
        model: Compiled MuJoCo model

    Returns:
        Rules instance, or None if no config provided
    """
    if config is None:
        return None

    # Resolve config if it's a dict (e.g., from YAML)
    if isinstance(config, dict):
        config = draccus.decode(TaskRulesConfig, config)

    # Use the registry to find the right implementation class
    rule_class = BaseRules._registry.get(type(config))

    if rule_class is None:
        raise ValueError(f"No rule implementation registered for config type: {type(config)}")

    return rule_class(config, model)
