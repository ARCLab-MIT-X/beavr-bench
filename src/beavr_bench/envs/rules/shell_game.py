"""Shell game rules implementation."""

import logging

import mujoco
import numpy as np

from beavr_bench.schemas import (
    RuleInfoKey,
    ShellGameRuleConfig,
)

from .base import BaseRules, RuleResult

logger = logging.getLogger(__name__)


@BaseRules.register(ShellGameRuleConfig)
class ShellGameRules(BaseRules):
    """Shell game rules: robot must lift the correct cup after shuffle."""

    STATE_SHOWING = "showing"
    STATE_COVERING = "covering"
    STATE_SHUFFLING = "shuffling"
    STATE_TESTING = "testing"

    def __init__(self, config: ShellGameRuleConfig, model: mujoco.MjModel):
        super().__init__(config, model)
        self.config: ShellGameRuleConfig = config  # Type narrowing

        # Get cup body and joint info
        self._cup_names = list(config.cup_names)
        self._cup_body_ids = []
        self._cup_qpos_adrs = []
        self._cup_qvel_adrs = []

        for name in self._cup_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
            if body_id == -1 or jnt_id == -1:
                raise ValueError(f"Cup {name} or its joint not found")
            self._cup_body_ids.append(body_id)
            self._cup_qpos_adrs.append(model.jnt_qposadr[jnt_id])
            self._cup_qvel_adrs.append(model.jnt_dofadr[jnt_id])

        # Get ball body and joint info
        self._ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        if self._ball_body_id == -1 or ball_jnt_id == -1:
            raise ValueError("Ball or its joint not found")
        self._ball_qpos_adr = model.jnt_qposadr[ball_jnt_id]
        self._ball_qvel_adr = model.jnt_dofadr[ball_jnt_id]

        # Store ball's initial position from the model (defined in XML)
        self._ball_initial_pos = model.body_pos[self._ball_body_id].copy()

        # The cup that started with the ball
        self._ball_cup_idx = self._cup_names.index(config.target_object)

        # Timings (in seconds)
        self._timer_showing = 3.0
        self._timer_covering = 1.0
        self._swaps_per_sec = 1.2

        # State tracking
        self._state = self.STATE_SHOWING
        self._state_start_time = 0.0
        self._current_swap_idx = 0
        self._cup_lifted = None
        self._success = False
        self._failure = False

        # Slots: x-positions on table
        self._slot_positions: list[np.ndarray] = []
        self._cup_to_slot = list(range(len(self._cup_names)))

        self._table_z = 0.445

        # Initialize shuffle state (will be populated by randomize_scene)
        self._shuffle_sequence: list[tuple[int, int]] = []
        self._target_slot_idx = 0
        self._num_swaps = 0

        logger.info(f"Shell Game: Showing ball for {self._timer_showing}s")

    def _generate_shuffle(self, rng: np.random.Generator) -> None:
        """Generate deterministic slot swaps and compute final ball slot."""
        self._num_swaps = int(rng.integers(self.config.shuffle_min_swaps, self.config.shuffle_max_swaps + 1))
        # Per user requirement: the ball always starts with the target cup
        ball_slot = self._ball_cup_idx
        self._shuffle_sequence = []
        for _ in range(self._num_swaps):
            s1, s2 = rng.choice([0, 1, 2], size=2, replace=False)
            self._shuffle_sequence.append((int(s1), int(s2)))
            if ball_slot == s1:
                ball_slot = s2
            elif ball_slot == s2:
                ball_slot = s1
        self._target_slot_idx = int(ball_slot)

    def _set_cup_pos(self, data: mujoco.MjData, cup_idx: int, pos: np.ndarray):
        """Override cup position and kill velocity."""
        qadr = self._cup_qpos_adrs[cup_idx]
        vadr = self._cup_qvel_adrs[cup_idx]
        data.qpos[qadr : qadr + 3] = pos
        data.qvel[vadr : vadr + 3] = 0

    def _sync_ball(self, data: mujoco.MjData):
        """Synchronize ball position with its cup."""
        cup_qadr = self._cup_qpos_adrs[self._ball_cup_idx]
        ball_qadr = self._ball_qpos_adr
        # Snap ball XY to cup XY
        data.qpos[ball_qadr : ball_qadr + 2] = data.qpos[cup_qadr : cup_qadr + 2]
        # Keep ball Z at initial height from XML
        data.qpos[ball_qadr + 2] = self._ball_initial_pos[2]
        # Kill velocity
        data.qvel[self._ball_qvel_adr : self._ball_qvel_adr + 6] = 0

    def compute(self, data: mujoco.MjData) -> RuleResult:
        self._step_count += 1
        sim_time = data.time

        # Initialize slot positions on first step
        if self._step_count == 1:
            for i in range(len(self._cup_names)):
                pos = data.xpos[self._cup_body_ids[i]].copy()
                pos[2] = self._table_z
                self._slot_positions.append(pos)
            self._initial_cup_heights = [float(data.xpos[bid][2]) for bid in self._cup_body_ids]

        # --- State Machine for Animation ---
        if self._state == self.STATE_SHOWING:
            for i in range(len(self._cup_names)):
                pos = self._slot_positions[i].copy()
                pos[2] = 0.55
                self._set_cup_pos(data, i, pos)

            # Ball stays put at center (Slot 1), using initial position from XML
            ball_qadr = self._ball_qpos_adr
            data.qpos[ball_qadr : ball_qadr + 3] = self._ball_initial_pos
            data.qvel[self._ball_qvel_adr : self._ball_qvel_adr + 6] = 0

            if sim_time >= self._timer_showing:
                self._state = self.STATE_COVERING
                self._state_start_time = sim_time
                logger.info("Covering ball...")

        elif self._state == self.STATE_COVERING:
            dur = sim_time - self._state_start_time
            t = min(1.0, dur / self._timer_covering)
            z = 0.55 - (t * (0.55 - self._table_z))
            for i in range(len(self._cup_names)):
                pos = self._slot_positions[i].copy()
                pos[2] = z
                self._set_cup_pos(data, i, pos)

            self._sync_ball(data)

            if t >= 1.0:
                self._state = self.STATE_SHUFFLING
                self._state_start_time = sim_time
                logger.info(f"Shuffling ({self._num_swaps} swaps)...")

        elif self._state == self.STATE_SHUFFLING:
            swap_dur = 1.0 / self._swaps_per_sec
            elapsed = sim_time - self._state_start_time
            swap_idx = int(elapsed // swap_dur)

            if swap_idx < len(self._shuffle_sequence):
                # Update mapping when transitioning to a NEW swap (previous swap just finished)
                while self._current_swap_idx < swap_idx:
                    # Apply the mapping for the swap that just completed
                    s1, s2 = self._shuffle_sequence[self._current_swap_idx]
                    c1 = self._cup_to_slot.index(s1)
                    c2 = self._cup_to_slot.index(s2)
                    self._cup_to_slot[c1], self._cup_to_slot[c2] = s2, s1
                    self._current_swap_idx += 1

                s1, s2 = self._shuffle_sequence[swap_idx]
                t = (elapsed % swap_dur) / swap_dur

                c1_idx = self._cup_to_slot.index(s1)
                c2_idx = self._cup_to_slot.index(s2)

                p1 = self._slot_positions[s1]
                p2 = self._slot_positions[s2]
                mid = (p1 + p2) / 2.0

                ctrl1 = mid.copy()
                ctrl1[0] += 0.1
                ctrl2 = mid.copy()
                ctrl2[0] -= 0.1

                pos1 = (1 - t) ** 2 * p1 + 2 * (1 - t) * t * ctrl1 + t**2 * p2
                pos2 = (1 - t) ** 2 * p2 + 2 * (1 - t) * t * ctrl2 + t**2 * p1
                self._set_cup_pos(data, c1_idx, pos1)
                self._set_cup_pos(data, c2_idx, pos2)

                for i in range(len(self._cup_names)):
                    if i != c1_idx and i != c2_idx:
                        self._set_cup_pos(data, i, self._slot_positions[self._cup_to_slot[i]])

                self._sync_ball(data)
            else:
                while self._current_swap_idx < len(self._shuffle_sequence):
                    s1, s2 = self._shuffle_sequence[self._current_swap_idx]
                    c1 = self._cup_to_slot.index(s1)
                    c2 = self._cup_to_slot.index(s2)
                    self._cup_to_slot[c1], self._cup_to_slot[c2] = s2, s1
                    self._current_swap_idx += 1

                self._state = self.STATE_TESTING
                logger.info("Teleop active! Find the ball.")
                self._initial_cup_heights = [float(data.xpos[bid][2]) for bid in self._cup_body_ids]
                self._sync_ball(data)

        # --- Success/Failure Detection (only in TESTING state) ---
        lifted_slot = None

        if self._state == self.STATE_TESTING:
            for i, body_id in enumerate(self._cup_body_ids):
                current_z = data.xpos[body_id][2]
                delta_z = current_z - self._initial_cup_heights[i]
                if delta_z > self.config.lift_height_threshold:
                    lifted_slot = self._cup_to_slot[i]
                    if self._cup_lifted is None:
                        self._cup_lifted = lifted_slot
                        logger.info(
                            f"FIRST LIFT: cup={i}, slot={lifted_slot}, target={self._target_slot_idx}"
                        )
                        if lifted_slot == self._target_slot_idx:
                            self._success = True
                            logger.info("SUCCESS! Lifted correct cup!")
                        else:
                            self._failure = True
                            logger.info(f"FAILURE! Wrong cup. Expected slot {self._target_slot_idx}")
                    break

        terminated = self._success or self._failure
        truncated = self._step_count >= self.config.max_steps
        reward = (10.0 if self._success else -5.0 if self._failure else 0.0) * self.config.reward_scale

        return RuleResult(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                RuleInfoKey.IS_SUCCESS: self._success,
                RuleInfoKey.DROPPED: self._failure,
                "state": self._state,
                "target_slot": self._target_slot_idx,
                "lifted_slot": lifted_slot,
            },
        )

    def reset(self) -> None:
        super().reset()
        self._state = self.STATE_SHOWING
        self._state_start_time = 0.0
        self._current_swap_idx = 0
        self._cup_lifted = None
        self._success = False
        self._failure = False
        self._cup_to_slot = list(range(len(self._cup_names)))
        # Do NOT generate shuffle here; randomize_scene will do it.
        logger.info(f"Reset: SHOWING for {self._timer_showing}s...")

    def get_shuffle_sequence(self) -> list[tuple[int, int]]:
        return self._shuffle_sequence.copy()

    def randomize_scene(self, data: mujoco.MjData, np_random: np.random.Generator) -> None:
        """Generate a new shuffle sequence."""
        self._generate_shuffle(np_random)
        logger.info(
            f"Shuffle randomized: {len(self._shuffle_sequence)} swaps, target slot {self._target_slot_idx}"
        )
