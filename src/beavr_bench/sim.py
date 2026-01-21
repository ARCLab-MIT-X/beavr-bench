"""Lightweight scene loader for beavr-bench.

Loads pre-baked scenes from static XML and JSON files without requiring beavr-sim.
"""

import json
import logging
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import draccus
import mujoco
import numpy as np

from beavr_bench.schemas import (
    CameraConfig,
    MetadataKey,
    ReachRuleConfig,
    ServerSwapRuleConfig,
    ShellGameRuleConfig,
    TaskRulesConfig,
    VanishingBlueprintRuleConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class SimpleRobotId:
    """Minimal robot identifier."""

    full_id: str
    config_name: str
    side: str


@dataclass
class SimpleRobotInfo:
    """Minimal robot info container."""

    robot_id: SimpleRobotId


@dataclass
class BakedScene:
    """Minimal scene representation loaded from pre-baked XML + JSON.

    This class provides the same interface expected by TeleopTask,
    but loads from static assets instead of building procedurally.
    """

    name: str
    model: mujoco.MjModel
    robots: list[SimpleRobotInfo]
    cameras: list[CameraConfig]
    _robot_indices: dict[str, dict[str, np.ndarray]]
    _initial_ctrl: np.ndarray
    task_rules_config: TaskRulesConfig | None = field(default=None)

    def apply_initial_positions(self, data: mujoco.MjData) -> None:
        """Apply initial joint positions to MjData.

        Uses the cached robot indices to map initial_ctrl values to qpos.
        """
        for robot in self.robots:
            full_id = robot.robot_id.full_id
            indices = self._robot_indices.get(full_id, {})
            ctrl_idx = indices.get(MetadataKey.CTRL, np.array([]))
            qpos_idx = indices.get(MetadataKey.QPOS, np.array([]))

            # Apply initial ctrl values as initial qpos
            for i, q_idx in enumerate(qpos_idx):
                if i < len(ctrl_idx):
                    ctrl_index = ctrl_idx[i]
                    if ctrl_index < len(self._initial_ctrl):
                        data.qpos[q_idx] = self._initial_ctrl[ctrl_index]


def get_assets_root() -> Path:
    """Get the root directory for baked scene assets."""
    return Path(str(files("beavr_bench.assets.scenes")))


def list_available_scenes() -> list[str]:
    """List all available scene names (without extension)."""
    root = get_assets_root()
    return sorted(p.stem for p in root.glob("*.xml"))


def load_baked_scene(scene_name: str) -> BakedScene:
    """Load a pre-baked scene from XML and JSON files.

    Args:
        scene_name: Name of the scene (e.g., 'scene_pickplace')

    Returns:
        BakedScene instance ready for use with TeleopTask

    Raises:
        FileNotFoundError: If the scene files don't exist
    """
    root = get_assets_root()
    xml_path = root / f"{scene_name}.xml"
    json_path = root / f"{scene_name}.json"

    if not xml_path.exists():
        available = list_available_scenes()
        raise FileNotFoundError(f"Scene '{scene_name}' not found. Available scenes: {available}")

    # Load MuJoCo model from XML path.
    # Meshes are resolved automatically because they are in a 'meshes/'
    # folder relative to the XML, and the XML contains <compiler meshdir="meshes"/>.
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    logger.info(f"Loaded MuJoCo model from {xml_path}")

    # Load metadata from JSON
    with open(json_path) as f:
        meta = json.load(f)

    # Parse robot_indices (convert lists back to numpy arrays)
    robot_indices: dict[str, dict[str, np.ndarray]] = {}
    for robot_id, idx_map in meta[MetadataKey.ROBOT_INDICES].items():
        robot_indices[robot_id] = {k: np.array(v, dtype=np.int32) for k, v in idx_map.items()}

    # Parse robots
    robots = [
        SimpleRobotInfo(
            robot_id=SimpleRobotId(
                full_id=r[MetadataKey.FULL_ID],
                config_name=r[MetadataKey.CONFIG_NAME],
                side=r[MetadataKey.SIDE],
            )
        )
        for r in meta[MetadataKey.ROBOTS]
    ]

    # Parse cameras
    cameras = [
        CameraConfig(
            name=c[MetadataKey.NAME],
            obs_key=c[MetadataKey.OBS_KEY],
            width=c[MetadataKey.WIDTH],
            height=c[MetadataKey.HEIGHT],
        )
        for c in meta[MetadataKey.CAMERAS]
    ]

    initial_ctrl = np.array(meta[MetadataKey.INITIAL_CTRL])

    # Parse task rules (with fallback for legacy baked scenes)
    task_rules = None
    if "task_rules" in meta:
        try:
            task_rules = draccus.decode(TaskRulesConfig, meta["task_rules"])
        except Exception as e:
            logger.warning(f"Failed to decode task rules from JSON: {e}")

    if task_rules is None:
        # Fallback mapping for standalone benchmark
        if scene_name == "scene_serverswap":
            task_rules = ServerSwapRuleConfig()
        elif scene_name == "scene_shell_game":
            task_rules = ShellGameRuleConfig()
        elif scene_name == "scene_vanishing_blueprint":
            task_rules = VanishingBlueprintRuleConfig()
        elif scene_name == "scene_pickplace":
            task_rules = ReachRuleConfig(target_object="box", goal_site="bin_site")

    return BakedScene(
        name=meta[MetadataKey.NAME],
        model=model,
        robots=robots,
        cameras=cameras,
        _robot_indices=robot_indices,
        _initial_ctrl=initial_ctrl,
        task_rules_config=task_rules,
    )
