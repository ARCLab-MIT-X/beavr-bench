"""Server swap rules implementation.

Transient failure signal memory challenge:
1. CUE (0-5s): One drive shows orange LED, others green
2. VANISHED (5s+): All LEDs turn green, looking identical
3. TESTING: Robot must swap the correct (memorized) drive
"""

import logging
from enum import Enum

import mujoco
import numpy as np

from beavr_bench.schemas import (
    RuleInfoKey,
    ServerSwapRuleConfig,
)

from .base import BaseRules, RuleResult

logger = logging.getLogger(__name__)


class ServerSwapState(str, Enum):
    """State machine states for ServerSwap rules."""

    CUE = "cue"  # Showing orange LED on failing slot
    VANISHED = "vanished"  # All LEDs green, robot must remember
    TESTING = "testing"  # Robot is actively working


# Okabe-Ito Color Palette (RGBA)
OKABE_BLUISH_GREEN = (0.0, 0.620, 0.451, 1.0)  # #009E73 - OK state
OKABE_ORANGE = (0.902, 0.624, 0.0, 1.0)  # #E69F00 - Failing cue


@BaseRules.register(ServerSwapRuleConfig)
class ServerSwapRules(BaseRules):
    """Server swap rules: robot must remember and swap the failing drive."""

    def __init__(self, config: ServerSwapRuleConfig, model: mujoco.MjModel):
        super().__init__(config, model)
        self.config: ServerSwapRuleConfig = config  # Type narrowing

        # Cache LED material IDs
        self._led_material_names = list(config.led_material_names)
        self._led_material_ids: list[int] = []
        for name in self._led_material_names:
            mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if mat_id == -1:
                raise ValueError(f"LED material '{name}' not found")
            self._led_material_ids.append(mat_id)

        # Cache slot positions (inferred from LED positions as sleds are now removed)
        self._slot_positions: list[np.ndarray] = []
        for name in self._led_material_names:
            # We assume LEDs are named 'led0', 'led1', etc. as per registry keys in config
            geom_name = name.replace("led_slot", "led")
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id == -1:
                raise ValueError(f"LED geom '{geom_name}' not found")

            # Slot center is relative to LED position: x_offset=0, y_offset=0.05 (matching original sled placement)
            led_pos = model.geom_pos[geom_id]
            slot_pos = np.array([led_pos[0], led_pos[1] + 0.05, led_pos[2]])
            self._slot_positions.append(slot_pos)

        # Cache replacement sled body ID
        self._replacement_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, config.replacement_sled
        )
        if self._replacement_body_id == -1:
            raise ValueError(f"Replacement sled '{config.replacement_sled}' not found")

        # State tracking
        self._state = ServerSwapState.CUE
        self._failing_slot_idx: int = 0
        self._success = False
        self._failure = False

        # Initial failing slot selection (will be overwritten by randomize_scene in TeleopTask.reset)
        self._select_failing_slot(np.random.default_rng(config.failing_slot_seed))

        logger.info(
            f"ServerPlacement: Target slot {self._failing_slot_idx} "
            f"will show orange LED for {config.cue_duration}s"
        )

    def _select_failing_slot(self, rng: np.random.Generator) -> None:
        """Select which slot is the goal using the provided RNG."""
        self._failing_slot_idx = int(rng.integers(0, len(self._led_material_names)))

    def _set_led_color(self, model: mujoco.MjModel, slot_idx: int, rgba: tuple) -> None:
        """Set the LED color for a specific slot."""
        mat_id = self._led_material_ids[slot_idx]
        model.mat_rgba[mat_id] = rgba

    def _set_all_leds_green(self, model: mujoco.MjModel) -> None:
        """Set all LEDs to green (OK) state."""
        for slot_idx in range(len(self._led_material_ids)):
            self._set_led_color(model, slot_idx, OKABE_BLUISH_GREEN)

    def _check_placement_success(self, data: mujoco.MjData) -> bool:
        """Check if the placement was performed correctly.

        Success if the replacement sled is in the target slot position.
        """
        target_pos = self._slot_positions[self._failing_slot_idx]
        tolerance = self.config.swap_tolerance

        # Check if replacement sled is in the target slot
        replacement_pos = data.xpos[self._replacement_body_id]
        dist = np.linalg.norm(replacement_pos[:2] - target_pos[:2])
        # Also check vertical alignment to ensure it's in the right slot level
        vertical_dist = abs(replacement_pos[2] - target_pos[2])

        return dist < tolerance and vertical_dist < 0.05

    def _check_wrong_placement(self, data: mujoco.MjData) -> bool:
        """Check if the sled was placed in the wrong slot.

        Failure if the sled is placed in any slot OTHER than the target one.
        """
        tolerance = self.config.swap_tolerance
        replacement_pos = data.xpos[self._replacement_body_id]

        for idx, target_pos in enumerate(self._slot_positions):
            if idx == self._failing_slot_idx:
                continue

            dist = np.linalg.norm(replacement_pos[:2] - target_pos[:2])
            vertical_dist = abs(replacement_pos[2] - target_pos[2])

            if dist < tolerance and vertical_dist < 0.05:
                logger.info(f"FAILURE: Sled placed in wrong slot {idx}")
                return True

        return False

    def compute(self, data: mujoco.MjData) -> RuleResult:
        self._step_count += 1
        sim_time = data.time

        # --- State Machine ---
        if self._state == ServerSwapState.CUE:
            self._update_leds()

            if sim_time >= self.config.cue_duration:
                self._state = ServerSwapState.VANISHED
                self._update_leds()
                logger.info("ServerSwap: LED vanished! All drives look identical now.")

        elif self._state == ServerSwapState.VANISHED:
            # Transition to testing state after a brief moment
            self._state = ServerSwapState.TESTING
            logger.info("ServerSwap: Testing active. Swap the failing drive.")

        elif self._state == ServerSwapState.TESTING:
            # Keep all LEDs green
            self._update_leds()

            # Check for success
            if not self._success and not self._failure:
                if self._check_placement_success(data):
                    self._success = True
                    logger.info("SUCCESS! Sled placed in target slot.")
                elif self._check_wrong_placement(data):
                    self._failure = True
                    logger.info("FAILURE! Sled placed in wrong slot.")

        # --- Compute reward and termination ---
        truncated = self._step_count >= self.config.max_steps

        reward = 0.0
        if self._success:
            reward = 10.0 * self.config.reward_scale
        elif self._failure:
            reward = -5.0 * self.config.reward_scale

        return RuleResult(
            reward=reward,
            terminated=self._success or self._failure,
            truncated=truncated,
            info={
                RuleInfoKey.IS_SUCCESS: self._success,
                RuleInfoKey.DROPPED: self._failure,
                RuleInfoKey.STATE: self._state.value,
                RuleInfoKey.FAILING_SLOT: self._failing_slot_idx,
                RuleInfoKey.STEPS: self._step_count,
            },
        )

    def reset(self) -> None:
        super().reset()
        self._state = ServerSwapState.CUE
        self._success = False
        self._failure = False
        # LEDs will be updated in randomize_scene which is called by TeleopTask.reset
        logger.info(f"Reset: CUE phase for {self.config.cue_duration}s")

    def _update_leds(self) -> None:
        """Update LED colors in the model based on current state."""
        if self._state == ServerSwapState.CUE:
            for idx in range(len(self._led_material_ids)):
                color = OKABE_ORANGE if idx == self._failing_slot_idx else OKABE_BLUISH_GREEN
                self._set_led_color(self.model, idx, color)
        else:
            self._set_all_leds_green(self.model)

    def randomize_scene(self, data: mujoco.MjData, np_random: np.random.Generator) -> None:
        """Select a new failing slot using the provided RNG."""
        self._select_failing_slot(np_random)
        self._update_leds()
        logger.info(f"Randomized failing slot: {self._failing_slot_idx}")
