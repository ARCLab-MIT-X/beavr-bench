"""Vanishing blueprint rules implementation."""

import logging

import mujoco
import numpy as np

from beavr_bench.schemas import (
    RuleInfoKey,
    VanishingBlueprintRuleConfig,
)

from .base import BaseRules, RuleResult

logger = logging.getLogger(__name__)


@BaseRules.register(VanishingBlueprintRuleConfig)
class VanishingBlueprintRules(BaseRules):
    """Vanishing blueprint rules: robot must replicate a memorized stack order."""

    STATE_SHOWING = "showing"
    STATE_TESTING = "testing"

    def __init__(self, config: VanishingBlueprintRuleConfig, model: mujoco.MjModel):
        super().__init__(config, model)
        self.config: VanishingBlueprintRuleConfig = config  # Type narrowing

        # Get block body IDs
        self._block_names = list(config.block_names)
        self._block_body_ids = []
        for name in self._block_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id == -1:
                raise ValueError(f"Block body '{name}' not found")
            self._block_body_ids.append(body_id)

        # Get hologram mocap IDs
        self._hologram_names = list(config.hologram_names)
        self._hologram_mocap_ids = []
        for name in self._hologram_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id == -1:
                raise ValueError(f"Hologram body '{name}' not found")

            # Get the mocap ID for this body
            mocap_id = model.body_mocapid[body_id]
            if mocap_id == -1:
                raise ValueError(f"Hologram body '{name}' is not a mocap body! Add mocap='true' to XML.")

            self._hologram_mocap_ids.append(mocap_id)

        # Get build zone site
        self._build_zone_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.build_zone_site)
        if self._build_zone_site_id == -1:
            raise ValueError(f"Build zone site '{config.build_zone_site}' not found")

        # Get table geom ID for contact checking
        self._table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")

        # Get block geom IDs (assuming one geom per body, which matches XML)
        self._block_geom_ids = []
        for body_id in self._block_body_ids:
            # The first geom of the body
            geom_id = model.body_geomadr[body_id]
            if geom_id == -1:
                raise ValueError(f"Block body {body_id} has no geoms?")
            self._block_geom_ids.append(geom_id)

        # State tracking
        self._state = self.STATE_SHOWING
        self._blueprint_order: list[int] = []  # Indices into block_names (bottom to top)
        self._success = False

        # Initial blueprint selection (will be overwritten by randomize_scene in TeleopTask.reset)
        self._select_blueprint(np.random.default_rng(config.shuffle_seed))

        logger.info(f"Vanishing Blueprint: Showing hologram for {config.show_duration}s...")

    def _select_blueprint(self, rng: np.random.Generator) -> None:
        """Generate a random stack order (bottom to top)."""
        indices = list(range(len(self._block_names)))
        rng.shuffle(indices)
        self._blueprint_order = indices
        order_names = [self._block_names[i] for i in self._blueprint_order]
        logger.info(f"Blueprint order (bottom to top): {order_names}")

    def _position_holograms(self, data: mujoco.MjData, visible: bool) -> None:
        """Position hologram blocks to show the blueprint or hide them."""
        build_zone_pos = data.site_xpos[self._build_zone_site_id].copy()

        for stack_idx, block_idx in enumerate(self._blueprint_order):
            mocap_id = self._hologram_mocap_ids[block_idx]
            if visible:
                # Stack holograms above build zone
                # Each block is 4cm (0.04m) tall, so spacing is 0.04m
                z_offset = 0.02 + stack_idx * self.config.z_spacing  # Start at half block height
                data.mocap_pos[mocap_id] = [
                    build_zone_pos[0],
                    build_zone_pos[1],
                    build_zone_pos[2] + z_offset,
                ]
            else:
                # Hide below the scene
                data.mocap_pos[mocap_id] = [0, 0, -1]

    def _check_stack_order(self, data: mujoco.MjData) -> bool:
        """Check if physical blocks are stacked in the correct order."""
        build_zone_pos = data.site_xpos[self._build_zone_site_id].copy()
        tolerance = self.config.stack_tolerance

        # Get block positions
        block_positions = []
        for body_id in self._block_body_ids:
            pos = data.xpos[body_id].copy()
            block_positions.append(pos)

        # Check each block in the blueprint order
        for stack_idx, block_idx in enumerate(self._blueprint_order):
            pos = block_positions[block_idx]

            # Check XY alignment with build zone
            dx = abs(pos[0] - build_zone_pos[0])
            dy = abs(pos[1] - build_zone_pos[1])
            if dx > tolerance or dy > tolerance:
                return False

            # The build_zone_pos is in world frame, so expected_z is also world frame.
            expected_z = build_zone_pos[2] + (self.config.z_spacing / 2) + stack_idx * self.config.z_spacing
            dz = abs(pos[2] - expected_z)
            if dz > tolerance:
                # print(f"DEBUG: {self._block_names[block_idx]} dz={dz:.4f} > {tolerance}")
                return False

        return True

    def compute(self, data: mujoco.MjData) -> RuleResult:
        self._step_count += 1
        sim_time = data.time

        # --- State Machine ---
        if self._state == self.STATE_SHOWING:
            # Show holograms in blueprint order
            self._position_holograms(data, visible=True)

            if sim_time >= self.config.show_duration:
                self._state = self.STATE_TESTING
                self._update_holograms(data)
                logger.info("Hologram vanished! Replicate the stack.")

        elif self._state == self.STATE_TESTING:
            # Hide holograms (keep them hidden)
            self._update_holograms(data)

            # Check for success
            if not self._success and self._check_stack_order(data):
                self._success = True
                logger.info("SUCCESS! Stack matches blueprint!")

        # --- Compute reward and termination ---
        truncated = self._step_count >= self.config.max_steps
        reward = 10.0 * self.config.reward_scale if self._success else 0.0

        return RuleResult(
            reward=reward,
            terminated=self._success,
            truncated=truncated,
            info={
                RuleInfoKey.IS_SUCCESS: self._success,
                "state": self._state,
                "blueprint_order": [self._block_names[i] for i in self._blueprint_order],
                RuleInfoKey.STEPS: self._step_count,
            },
        )

    def reset(self) -> None:
        super().reset()
        self._state = self.STATE_SHOWING
        self._success = False
        # Do NOT generate blueprint here; randomize_scene will do it.
        logger.info(f"Reset: SHOWING for {self.config.show_duration}s...")

    def _update_holograms(self, data: mujoco.MjData) -> None:
        """Update hologram positions based on current state."""
        visible = self._state == self.STATE_SHOWING
        self._position_holograms(data, visible=visible)

    def randomize_scene(self, data: mujoco.MjData, np_random: np.random.Generator) -> None:
        """Generate a new blueprint and update hologram positions."""
        self._select_blueprint(np_random)
        self._update_holograms(data)
