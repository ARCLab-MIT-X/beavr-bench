"""Base task class for simulation scenarios.

Gymnasium-compatible tasks for LeRobot evaluation.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import mujoco
import numpy as np

from beavr_bench.envs.rules import BaseRules, create_rules
from beavr_bench.schemas import (
    DEFAULT_RENDER_FPS,
    RENDER_FPS_KEY,
    CameraConfig,
    ObservationKey,
    RuleInfoKey,
    SceneId,
)

if TYPE_CHECKING:
    from beavr_bench.sim import BakedScene

logger = logging.getLogger(__name__)


class BaseTask(gym.Env, ABC):
    """Abstract base class for simulation tasks.

    Inherits from gymnasium.Env for LeRobot/lerobot-eval compatibility.
    Subclasses should define observation_space and action_space.

    Thread Safety:
        MuJoCo's mjData is not thread-safe. This class provides a `self.lock`
        (RLock) that protects all mjData access. External code accessing
        task.data directly must acquire task.lock first.
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        model: mujoco.MjModel | None = None,
        control_hz: int = 30,
    ):
        """Initialize the task with a MuJoCo model.

        Args:
            xml_path: Path to the MJCF XML file (mutually exclusive with model)
            model: Pre-composed MjModel (mutually exclusive with xml_path)
            control_hz: Control frequency in Hz (default: 30)
        """
        if model is not None:
            self.model = model
            self.xml_path = None

        elif xml_path is not None:
            self.xml_path = Path(xml_path)
            if not self.xml_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.xml_path}")
            self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))

        else:
            raise ValueError("Either xml_path or model must be provided")

        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.viewer = None

        # Configure timing
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz
        self.metadata = {RENDER_FPS_KEY: control_hz}

        # Compute substeps
        sim_dt = self.model.opt.timestep
        self.n_substeps = int(round(self.dt / sim_dt))
        if self.n_substeps < 1:
            self.n_substeps = 1

    def init_viewer(self, key_callback: Callable[[int], None] | None = None):
        """Initialize the MuJoCo viewer (call after model is loaded).

        Args:
            key_callback: Optional callback for keyboard events.
                         Receives GLFW keycode on key press.
        """
        import mujoco.viewer as mjviewer

        self.viewer = mjviewer.launch_passive(
            self.model,
            self.data,
            key_callback=key_callback,
        )

        # Enable world coordinat frame visualization (shows X=Red, Y=Green, Z=Blue at origin)
        # self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

        logger.info("👁️  MuJoCo viewer launched with world frame enabled")

    @abstractmethod
    def get_observation(self) -> dict[str, dict[str, np.ndarray]]:
        """Get current observation from simulation.

        Returns:
            Dictionary mapping robot_name -> state dict (qpos, qvel, etc.)
        """
        pass

    @abstractmethod
    def apply_action(self, actions: dict[str, np.ndarray]) -> None:
        """Apply actions to the simulation.

        Args:
            actions: Dictionary mapping robot_name -> target_qpos
        """
        pass

    def step_physics(self, actions: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
        """Perform a single physics step (1/physics_hz).

        Args:
            actions: Dictionary mapping robot_name -> target_qpos

        Returns:
            Observations after stepping
        """
        with self.lock:
            self.apply_action(actions)
            mujoco.mj_step(self.model, self.data)
            return self.get_observation()

    def step(self, actions: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
        """Step the simulation for one control period (1/control_hz).

        Loops n_substeps physics steps internally.

        Args:
            actions: Dictionary mapping robot_name -> target_qpos

        Returns:
            Observations after stepping
        """
        for _ in range(self.n_substeps):
            obs = self.step_physics(actions)
        return obs

    def step_with_rules(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[dict[str, dict[str, np.ndarray]], float, bool, bool, dict]:
        """Step simulation for one physics tick and return RL-style result.

        Default implementation has no reward or termination logic.
        """
        obs = self.step_physics(actions)
        return obs, 0.0, False, False, {}

    def render(self):
        """Sync the viewer if active."""
        if self.viewer is not None:
            self.viewer.sync()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset simulation to initial state.

        Args:
            seed: Optional random seed
            options: Optional reset options

        Returns:
            Tuple of (observation, info). Subclasses should override to return actual observations.
        """
        super().reset(seed=seed)

        with self.lock:
            mujoco.mj_resetData(self.model, self.data)

            # Try to load "home" keyframe if it exists
            if self.model.nkey > 0:
                for i in range(self.model.nkey):
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
                    if name == "home":
                        mujoco.mj_resetDataKeyframe(self.model, self.data, i)
                        logger.info("🔑 Loaded 'home' keyframe")
                        break

            mujoco.mj_forward(self.model, self.data)

        return {}, {}

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class TeleopTask(BaseTask):
    """Teleoperation task for controlling robots via network.

    Gymnasium-compatible: use reset() and step(action) for RL.
    For network teleop (PhysicsLoop), use get_observation() and step_with_rules().

    Camera configuration comes from the scene's `cameras` section in the YAML.
    """

    metadata = {RENDER_FPS_KEY: DEFAULT_RENDER_FPS}

    def __init__(
        self,
        scene: BakedScene,
        cameras: list[CameraConfig] | None = None,
        control_hz: int = 30,
    ):
        """Initialize teleoperation task.

        Args:
            scene: Composed Scene object with cached robot indices
            cameras: Override camera configurations. If None, uses scene.cameras.
            control_hz: Control frequency in Hz (default: 30)
        """
        super().__init__(model=scene.model, control_hz=control_hz)

        self._scene = scene
        self.robots = scene.robots
        self._robot_indices = scene._robot_indices

        # Use provided cameras or fall back to scene's camera config
        self._cameras = cameras if cameras is not None else scene.cameras

        # Initialize per-camera renderers with appropriate dimensions
        self._renderers: dict[str, mujoco.Renderer] = {}
        for cam_cfg in self._cameras:
            # Validate camera exists in model
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_cfg.name)
            if cam_id == -1:
                available = [
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                    for i in range(self.model.ncam)
                ]
                raise ValueError(
                    f"Camera '{cam_cfg.name}' not found in scene. Available cameras: {available}"
                )
            self._renderers[cam_cfg.name] = mujoco.Renderer(
                self.model, height=cam_cfg.height, width=cam_cfg.width
            )

        # Sorted robot names for deterministic observation/action ordering
        self._sorted_robot_names = sorted(
            r.robot_id.full_id for r in self.robots if r.robot_id.full_id in self._robot_indices
        )

        # Compute total state and action dimensions
        self._state_dim = 0
        self._action_dim = 0
        self._robot_slices: dict[str, tuple[int, int]] = {}  # robot -> (start, end) in flat vector

        offset = 0
        for robot_name in self._sorted_robot_names:
            indices = self._robot_indices[robot_name]
            qpos_len = len(indices.get("qpos", []))
            ctrl_len = len(indices.get("ctrl", []))
            self._state_dim += qpos_len
            self._action_dim += ctrl_len
            self._robot_slices[robot_name] = (offset, offset + ctrl_len)
            offset += ctrl_len

        # Define Gymnasium spaces (LeRobot-compatible)
        obs_dict = {
            ObservationKey.AGENT_POS.value: gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32
            ),
        }

        # Add camera spaces using configured obs_key
        pixels_space = {}
        for cam_cfg in self._cameras:
            # Use obs_key (e.g., 'camera_egocentric') to match LeRobot conventions
            pixels_space[cam_cfg.obs_key] = gym.spaces.Box(
                low=0, high=255, shape=cam_cfg.shape_hwc, dtype=np.uint8
            )

        if pixels_space:
            obs_dict["pixels"] = gym.spaces.Dict(pixels_space)

        self.observation_space = gym.spaces.Dict(obs_dict)
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._action_dim,), dtype=np.float32
        )

        # Store last targets for hold behavior
        self._last_targets: dict[str, np.ndarray] = {}

        # Create rules from scene config (optional)
        self._rules: BaseRules | None = create_rules(scene.task_rules_config, self.model)

    def _max_episode_steps(self) -> int:
        """Maximum steps for this task, pulled from rules config.

        Required by LeRobot's rollout utility via env.call().
        """
        if self._rules:
            return self._rules.config.max_steps
        return 500

    # -------------------------------------------------------------------------
    # Gymnasium-compatible interface (for LeRobot/RL)
    # -------------------------------------------------------------------------

    def _get_agent_pos(self) -> np.ndarray:
        """Get flat agent_pos observation (LeRobot format).

        Returns:
            Concatenated qpos for all robots in sorted order.
        """
        parts = []
        for robot_name in self._sorted_robot_names:
            indices = self._robot_indices[robot_name]
            qpos_idx = indices.get("qpos", [])
            if len(qpos_idx) > 0:
                parts.append(self.data.qpos[qpos_idx])
        return np.concatenate(parts).astype(np.float32) if parts else np.array([], dtype=np.float32)

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Get all observations including camera images.

        Returns:
            Dictionary with 'agent_pos' and configured camera observations.
            Camera images are keyed by their obs_key (e.g., 'observation.images.overhead').
        """
        obs = {ObservationKey.AGENT_POS.value: self._get_agent_pos()}

        for cam_cfg in self._cameras:
            renderer = self._renderers[cam_cfg.name]
            renderer.update_scene(self.data, camera=cam_cfg.name)
            # MuJoCo returns (H, W, 3) in [0, 255] uint8.
            # LeRobot expects (3, H, W) for visual features.
            image = renderer.render()

            if "pixels" not in obs:
                obs["pixels"] = {}

            # Use obs_key (e.g., 'camera_egocentric') to match Registry features
            obs["pixels"][cam_cfg.obs_key] = image.copy()

        return obs

    def _split_flat_action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        """Split flat action vector into per-robot dict.

        Args:
            action: Flat action vector (from Gym interface)

        Returns:
            Dictionary mapping robot_name -> action segment
        """
        result = {}
        for robot_name in self._sorted_robot_names:
            start, end = self._robot_slices[robot_name]
            result[robot_name] = action[start:end]
        return result

    # -------------------------------------------------------------------------
    # PhysicsLoop/Network interface (backwards compatible)
    # -------------------------------------------------------------------------

    def get_observation(self) -> dict[str, dict[str, np.ndarray]]:
        """Get per-robot observations for network publishing.

        Used by PhysicsLoop for sending state to teleop clients.

        Returns:
            Dictionary mapping robot_name -> {"qpos": array}
        """
        obs = {}
        for robot_name in self._sorted_robot_names:
            indices = self._robot_indices.get(robot_name, {})
            qpos_idx = indices.get("qpos", [])
            if len(qpos_idx) > 0:
                obs[robot_name] = {
                    "qpos": self.data.qpos[qpos_idx].copy(),
                }
        return obs

    def apply_action(self, actions: dict[str, np.ndarray]) -> None:
        """Apply target positions to robot actuators.

        Args:
            actions: Dictionary mapping robot_name -> target_qpos
        """
        for robot in self.robots:
            full_id = robot.robot_id.full_id
            indices = self._robot_indices.get(full_id, {})
            ctrl_idx = indices.get("ctrl", [])
            if len(ctrl_idx) == 0:
                continue

            # Get target (new action or last target for hold)
            if full_id in actions:
                target = actions[full_id]
                self._last_targets[full_id] = target
            else:
                target = self._last_targets.get(full_id)

            if target is not None:
                num_act = len(ctrl_idx)
                if len(target) < num_act:
                    logger.warning(f"Action mismatch for {full_id}: Expected {num_act}, got {len(target)}")
                # Apply using indexed assignment (fast)
                self.data.ctrl[ctrl_idx] = target[:num_act]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset simulation (Gymnasium-compatible).

        Args:
            seed: Optional random seed
            options: Optional reset options

        Returns:
            Tuple of (observation, info) where observation is {"agent_pos": array}
        """
        # Call super().reset() which handles self.np_random
        super().reset(seed=seed, options=options)

        with self.lock:
            # 1. MuJoCo reset - uses default qpos0
            mujoco.mj_resetData(self.model, self.data)

            # 2. Apply initial joint positions using named joint lookups
            self._scene.apply_initial_positions(self.data)

            # 3. Apply pre-computed control defaults
            self.data.ctrl[:] = self._scene._initial_ctrl

            # 4. Reset rules state (must happen before randomize_scene)
            if self._rules is not None:
                self._rules.reset()

            # 5. Randomize scene using task rules and current RNG
            if self._rules is not None:
                self._rules.randomize_scene(self.data, self.np_random)

            # 6. Clear stored targets
            self._last_targets.clear()

            # 7. Forward kinematics
            mujoco.mj_forward(self.model, self.data)

        # Return LeRobot-compatible observation
        obs = self._get_observations()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Step simulation for one control period (Gymnasium-compatible).

        Args:
            action: Flat action vector (concatenated per-robot actions)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Split flat action into per-robot dict
        actions_dict = self._split_flat_action(action)

        # BaseTask.step handles the substepping loop
        _ = super().step(actions_dict)

        # Flatten observations for Gym/LeRobot
        obs = self._get_observations()

        with self.lock:
            # Compute rules (if configured)
            if self._rules is not None:
                result = self._rules.compute(self.data)
                info = result.info
                if RuleInfoKey.IS_SUCCESS not in info:
                    info[RuleInfoKey.IS_SUCCESS] = result.terminated and result.reward > 0
                return obs, result.reward, result.terminated, result.truncated, info

        return obs, 0.0, False, False, {RuleInfoKey.IS_SUCCESS: False}

    def render(self) -> np.ndarray | None:
        """Render the environment for video recording.

        Also syncs the native MuJoCo viewer if active.

        Returns:
            HWC uint8 numpy array from the first camera, or None if no cameras.
        """
        # Sync native viewer if active (critical for visual updates)
        if self.viewer is not None:
            self.viewer.sync()

        if not self._cameras:
            return None

        # Render all configured cameras and tile them
        images = []
        for cam_cfg in self._cameras:
            renderer = self._renderers[cam_cfg.name]
            renderer.update_scene(self.data, camera=cam_cfg.name)
            images.append(renderer.render().copy())

        if len(images) == 1:
            return images[0]

        # Tile images into a grid
        n = len(images)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        h, w, c = images[0].shape
        grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)

        for i, img in enumerate(images):
            r, c_idx = divmod(i, cols)
            grid[r * h : (r + 1) * h, c_idx * w : (c_idx + 1) * w, :] = img

        return grid

    # -------------------------------------------------------------------------
    # PhysicsLoop interface (uses dict actions, returns nested observations)
    # -------------------------------------------------------------------------

    def step_with_rules(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],  # obs
        float,  # reward
        bool,  # terminated
        bool,  # truncated
        dict,  # info
    ]:
        """Step simulation for one physics tick (for PhysicsLoop).

        Args:
            actions: Dictionary mapping robot_name -> target_qpos

        Returns:
            Tuple of (robot_observations, reward, terminated, truncated, info)
        """
        # BaseTask.step_physics handles the single tick
        obs = self.step_physics(actions)

        with self.lock:
            if self._rules is not None:
                result = self._rules.compute(self.data)
                return (
                    obs,
                    result.reward,
                    result.terminated,
                    result.truncated,
                    result.info,
                )

        return obs, 0.0, False, False, {}


def make_teleop_env(
    scene_name: str = SceneId.PICK_PLACE,
    cameras: list[CameraConfig] | None = None,
    **kwargs,
):
    """Factory function for TeleopTask.

    Args:
        scene_name: Name of the scene to load (default: "scene_pickplace")
        cameras: Override camera configs. If None, uses cameras from baked scene.
        **kwargs: Additional arguments (unused)

    Returns:
        TeleopTask instance

    Example:
        >>> # Uses cameras from baked scene
        >>> env = make_teleop_env("scene_pickplace")

        >>> # Override cameras
        >>> from beavr_bench.schemas import CameraConfig
        >>> cameras = [CameraConfig(name="overhead", obs_key="observation.images.overhead")]
        >>> env = make_teleop_env("scene_pickplace", cameras=cameras)
    """
    from beavr_bench.sim import load_baked_scene

    scene = load_baked_scene(scene_name)
    return TeleopTask(scene=scene, cameras=cameras)
