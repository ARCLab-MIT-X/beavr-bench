"""Registry utilities for beavr-bench.

Handles Gymnasium and LeRobot environment registration.
"""

import json
import logging
from dataclasses import dataclass, field

from beavr_bench.schemas import (
    MetadataKey,
    ObservationKey,
    SceneId,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LeRobot Integration
# =============================================================================

try:
    import gymnasium
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.envs.configs import EnvConfig
    from lerobot.utils.constants import ACTION, OBS_STATE

    @EnvConfig.register_subclass("beavr")
    @dataclass
    class BeavrEnv(EnvConfig):
        """LeRobot environment config for beavr-bench baked scenes.

        This config enables using lerobot-eval with beavr-bench:

        ```bash
        beavr-eval \\
            --policy.path=/path/to/checkpoint \\
            --env.type=beavr \\
            --env.scene=scene_pickplace \\
            --eval.n_episodes=10
        ```
        """

        scene: str = "scene_pickplace"
        """Scene name to load (e.g., 'scene_pickplace', 'scene_shell_game')."""

        fps: int = 30
        """Simulation frames per second."""

        episode_length: int = 1000
        """Maximum episode length in steps."""

        seed: int | None = None
        """Global environment seed for deterministic randomization."""

        # LeRobot requires features to be defined for policy compatibility
        # These are populated in __post_init__ based on the baked scene JSON
        features: dict[str, PolicyFeature] = field(default_factory=dict)
        features_map: dict[str, str] = field(default_factory=dict)

        def __post_init__(self):
            """Build features and features_map from baked scene JSON."""
            from beavr_bench.sim import get_assets_root

            root = get_assets_root()
            json_path = root / f"{self.scene}.json"

            if not json_path.exists():
                raise FileNotFoundError(f"Baked scene metadata not found: {json_path}")

            with open(json_path) as f:
                meta = json.load(f)

            # Calculate action and state dimensions from robot indices
            # We sort robot names to match TeleopTask initialization logic
            state_dim = 0
            action_dim = 0
            robot_indices = meta[MetadataKey.ROBOT_INDICES]
            for robot_name in sorted(robot_indices.keys()):
                indices = robot_indices[robot_name]
                state_dim += len(indices.get(MetadataKey.QPOS, []))
                action_dim += len(indices.get(MetadataKey.CTRL, []))

            # Define features and features_map
            # features: uses environment-internal keys (matches Task's observation keys)
            # features_map: maps environment keys to policy keys
            from lerobot.utils.constants import OBS_IMAGES

            self.features = {
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
                ObservationKey.AGENT_POS.value: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            }
            self.features_map = {
                ACTION: ACTION,
                ObservationKey.AGENT_POS.value: OBS_STATE,
            }

            # Add camera features from baked metadata
            for cam in meta[MetadataKey.CAMERAS]:
                obs_key = cam[MetadataKey.OBS_KEY]
                # LeRobot expects (channels, height, width)
                shape = (3, cam[MetadataKey.HEIGHT], cam[MetadataKey.WIDTH])
                self.features[obs_key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
                # Map environment key (e.g. 'camera_egocentric') to policy key (e.g. 'observation.images.camera_egocentric')
                self.features_map[obs_key] = f"{OBS_IMAGES}.{obs_key}"

            # Update episode length from task rules if present
            if "task_rules" in meta and meta["task_rules"]:
                self.episode_length = meta["task_rules"].get("max_steps", self.episode_length)

        @property
        def gym_id(self) -> str:
            """Gym environment ID for this scene."""
            return f"beavr_bench/Teleop-{self.scene}-v0"

        @property
        def gym_kwargs(self) -> dict:
            """Keyword arguments passed to gym.make()."""
            return {
                "scene_name": self.scene,
                "seed": self.seed,
            }

    LEROBOT_AVAILABLE = True

except ImportError:
    # LeRobot not installed
    LEROBOT_AVAILABLE = False


# =============================================================================
# Gymnasium Registration
# =============================================================================


def register_beavr_envs() -> None:
    """Register all beavr-bench environments with Gymnasium.

    Registers environment variants for each baked scene.
    """
    for scene_id in SceneId:
        env_id = f"beavr_bench/Teleop-{scene_id.value}-v0"
        if env_id not in gymnasium.registry:
            gymnasium.register(
                id=env_id,
                entry_point="beavr_bench.envs.tasks:make_teleop_env",
                kwargs={"scene_name": scene_id.value},
                # We don't set max_episode_steps here; it's handled by the environment's
                # task rules (BaseRules.compute returns truncated=True).
            )
