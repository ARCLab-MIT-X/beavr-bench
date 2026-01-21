"""Reach rules implementation."""

import logging

import mujoco
import numpy as np

from beavr_bench.schemas import (
    ReachRuleConfig,
    RuleInfoKey,
)

from .base import BaseRules, RuleResult

logger = logging.getLogger(__name__)


@BaseRules.register(ReachRuleConfig)
class ReachRules(BaseRules):
    """Success when target object reaches goal zone."""

    def __init__(self, config: ReachRuleConfig, model: mujoco.MjModel):
        super().__init__(config, model)
        self.config: ReachRuleConfig = config  # Type narrowing
        self._prev_distance: float | None = None

        # Cache target ID - try site first, then body
        self._target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.target_object)
        self._target_body_id = -1
        if self._target_site_id == -1:
            self._target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.target_object)
            if self._target_body_id == -1:
                raise ValueError(f"Target object '{config.target_object}' not found as body or site")

        # Cache goal site ID
        self._goal_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.goal_site)
        if self._goal_site_id == -1:
            raise ValueError(f"Goal site '{config.goal_site}' not found in model")

    def compute(self, data: mujoco.MjData) -> RuleResult:
        self._step_count += 1

        # Get positions using cached IDs
        if self._target_site_id != -1:
            obj_pos = data.site_xpos[self._target_site_id].copy()
        else:
            obj_pos = data.xpos[self._target_body_id].copy()

        goal_pos = data.site_xpos[self._goal_site_id].copy()
        distance = float(np.linalg.norm(obj_pos - goal_pos))

        # Check success
        success = distance < self.config.success_distance

        # Check failure (dropped)
        dropped = self.config.fail_on_drop and obj_pos[2] < self.config.drop_height_threshold

        # Check truncation (max steps)
        truncated = self._step_count >= self.config.max_steps

        # Compute reward
        reward = 0.0
        if success:
            reward = 10.0 * self.config.reward_scale
        elif dropped:
            reward = -5.0 * self.config.reward_scale
        elif self.config.distance_reward and self._prev_distance is not None:
            # Shaped reward: positive for getting closer
            reward = (self._prev_distance - distance) * self.config.reward_scale

        self._prev_distance = distance

        return RuleResult(
            reward=reward,
            terminated=success or dropped,
            truncated=truncated,
            info={
                RuleInfoKey.IS_SUCCESS: success,
                RuleInfoKey.DROPPED: dropped,
                RuleInfoKey.DISTANCE: distance,
                RuleInfoKey.STEPS: self._step_count,
            },
        )

    def randomize_scene(self, data: mujoco.MjData, np_random: np.random.Generator) -> None:
        """Randomize box position at episode start."""
        # Find the target object's joint.
        # Try {target_object}_joint first, then fallback to any joint attached to the body.
        joint_name = f"{self.config.target_object}_joint"
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

        if jnt_id == -1 and self._target_body_id != -1:
            # Look for any joint belonging to this body
            for i in range(self.model.njnt):
                if self.model.jnt_bodyid[i] == self._target_body_id:
                    jnt_id = i
                    break

        if jnt_id != -1:
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            init_x = self.model.qpos0[qpos_adr]
            init_y = self.model.qpos0[qpos_adr + 1]
            rand_range = self.config.box_randomization_range

            data.qpos[qpos_adr] = init_x + np_random.uniform(-rand_range, rand_range)
            data.qpos[qpos_adr + 1] = init_y + np_random.uniform(-rand_range, rand_range)
            logger.debug(f"Randomized {self.config.target_object} (range={rand_range})")

    def reset(self) -> None:
        super().reset()
        self._prev_distance = None
