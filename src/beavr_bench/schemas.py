"""Beavr-Bench schemas and constants.

This module contains all dataclasses, enums, and constants needed by beavr-bench.
Inlined to avoid external dependencies on beavr-configs.
"""

from dataclasses import dataclass
from enum import Enum, StrEnum

import draccus


class SceneId(StrEnum):
    """Available scene identifiers - NEVER use string literals for scene names."""

    PICK_PLACE = "scene_pickplace"
    SHELL_GAME = "scene_shell_game"
    VANISHING_BLUEPRINT = "scene_vanishing_blueprint"
    SERVER_SWAP = "scene_serverswap"


class ObservationKey(StrEnum):
    """Standard keys for observations."""

    AGENT_POS = "agent_pos"
    OBS_STATE = "observation.state"


class MetadataKey(StrEnum):
    """JSON metadata keys for baked scene files."""

    NAME = "name"
    ROBOT_INDICES = "robot_indices"
    INITIAL_CTRL = "initial_ctrl"
    CAMERAS = "cameras"
    ROBOTS = "robots"
    QPOS = "qpos"
    CTRL = "ctrl"
    FULL_ID = "full_id"
    CONFIG_NAME = "config_name"
    SIDE = "side"
    OBS_KEY = "obs_key"
    WIDTH = "width"
    HEIGHT = "height"


# LeRobot compatibility constants
RENDER_FPS_KEY = "render_fps"
DEFAULT_RENDER_FPS = 30


@dataclass
class CameraConfig:
    """Camera configuration for scene rendering."""

    name: str
    obs_key: str
    width: int = 640
    height: int = 480

    @property
    def shape_hwc(self) -> tuple[int, int, int]:
        """Return shape as (height, width, channels)."""
        return (self.height, self.width, 3)

    @property
    def shape_chw(self) -> tuple[int, int, int]:
        """Return shape as (channels, height, width)."""
        return (3, self.height, self.width)


class TaskRuleType(str, Enum):
    """Available rule implementations."""

    REACH = "reach"  # Success when object reaches goal zone
    PICK_PLACE = "pick_place"  # Success when object picked and placed
    SHELL_GAME = "shell_game"  # Success when correct cup is lifted
    VANISHING_BLUEPRINT = "vanishing_blueprint"  # Success when the colored blocks are stacked correctly
    SERVER_SWAP = "server_swap"  # Success when failing drive is swapped


class RuleInfoKey(str, Enum):
    """Typed keys for RuleResult.info dictionary."""

    IS_SUCCESS = "is_success"  # bool: Task goal achieved (LeRobot compatible)
    DROPPED = "dropped"  # bool: Object fell below threshold
    DISTANCE = "distance"  # float: Current distance to goal
    STEPS = "steps"  # int: Current step count
    STATE = "state"  # str: Current rule state machine state
    FAILING_SLOT = "failing_slot"  # int: Index of the failing slot (ServerSwap)


@dataclass(frozen=True)
class TaskRulesConfig(draccus.ChoiceRegistry):
    """Base configuration for task success/failure rules.

    Linked to a Scene via the scene YAML's `task_rules:` section.
    """

    # Common fields for all rules
    max_steps: int = 400  # Maximum steps before truncation
    reward_scale: float = 1.0  # Scalar multiplier for all rewards


@TaskRulesConfig.register_subclass(TaskRuleType.REACH)
@dataclass(frozen=True)
class ReachRuleConfig(TaskRulesConfig):
    """Success when target object reaches goal zone."""

    target_object: str = ""  # MuJoCo body name to track
    goal_site: str = ""  # MuJoCo site name for the goal zone
    success_distance: float = 0.05  # Distance threshold (m) for success
    fail_on_drop: bool = True  # Terminate if object is dropped
    drop_height_threshold: float = 0.1  # Height (m) below which object is "dropped"
    box_randomization_range: float = 0.09  # Random offset (m) for object placement
    distance_reward: bool = True  # Enable shaped reward based on dist to goal
    max_steps: int = 350


@TaskRulesConfig.register_subclass(TaskRuleType.SHELL_GAME)
@dataclass(frozen=True)
class ShellGameRuleConfig(TaskRulesConfig):
    """Success when correct cup is lifted."""

    target_object: str = "cup_b"  # Cup that starts with the ball
    goal_site: str | None = None  # Shell game usually doesn't use a goal site
    shuffle_seed: int | None = (
        None  # Seed for deterministic shuffle (None = random) (Set this to shuffle_seed = 2026 during eval)
    )
    shuffle_min_swaps: int = 2  # Minimum number of cup swaps per episode
    shuffle_max_swaps: int = 5  # Maximum number of cup swaps per episode
    cup_names: tuple[str, ...] = ("cup_a", "cup_b", "cup_c")  # Cup body names
    lift_height_threshold: float = 0.05  # Height (m) above table for "lifted"
    max_steps: int = 400


@TaskRulesConfig.register_subclass(TaskRuleType.VANISHING_BLUEPRINT)
@dataclass(frozen=True)
class VanishingBlueprintRuleConfig(TaskRulesConfig):
    """Success when blocks are stacked in the correct blueprint order."""

    shuffle_seed: int | None = None  # Seed for deterministic order (None = random)
    block_names: tuple[str, ...] = (
        "orange_block",
        "skyblue_block",
        "bluishgreen_block",
    )
    hologram_names: tuple[str, ...] = (
        "holo_orange",
        "holo_skyblue",
        "holo_bluishgreen",
    )
    build_zone_site: str = "build_zone_site"  # Site marking the build area
    show_duration: float = 10.0  # Seconds to show hologram
    stack_tolerance: float = 0.05  # XY tolerance for "aligned" blocks (m)
    z_spacing: float = 0.04  # Vertical spacing (m), should match block height
    max_steps: int = 750


@TaskRulesConfig.register_subclass(TaskRuleType.SERVER_SWAP)
@dataclass(frozen=True)
class ServerSwapRuleConfig(TaskRulesConfig):
    """Success when failing drive is swapped with replacement."""

    failing_slot_seed: int | None = None  # Seed for selecting failing slot (None = random)
    sled_names: tuple[str, ...] = ("sled0", "sled1", "sled2", "sled3")
    led_material_names: tuple[str, ...] = (
        "led_slot0",
        "led_slot1",
        "led_slot2",
        "led_slot3",
    )
    replacement_sled: str = "sled4"  # Spare sled on parts bench
    cue_duration: float = 5.0  # Seconds to show orange LED before vanish
    swap_tolerance: float = 0.1  # Tolerance (m) for sled placement
    max_steps: int = 750
