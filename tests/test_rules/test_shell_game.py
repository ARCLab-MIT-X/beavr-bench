import mujoco
import numpy as np
import pytest

from beavr_bench.envs.rules.shell_game import ShellGameRules
from beavr_bench.schemas import RuleInfoKey, ShellGameRuleConfig


def advance_to_state(rules, data, model, target_state, max_calls=400):
    """Helper to advance the state machine by incrementing time and calling compute."""
    for _ in range(max_calls):
        result = rules.compute(data)
        if result.info["state"] == target_state:
            return result
        # Step time slightly to trigger transitions
        data.time += 0.05
        # CRITICAL: Update xpos after compute might have set qpos (during shuffle)
        # and so that the rule sees updated xpos in the next call.
        mujoco.mj_forward(model, data)
    current = result.info["state"]
    raise RuntimeError(f"Failed to reach state {target_state}. Current state: {current}")


@pytest.fixture
def shell_game_setup():
    xml = """
    <mujoco>
        <worldbody>
            <body name="cup_a" pos="0 0.1 0">
                <joint type="free" name="cup_a_joint"/>
                <geom type="cylinder" size="0.05 0.05" mass="1"/>
            </body>
            <body name="cup_b" pos="0 0 0">
                <joint type="free" name="cup_b_joint"/>
                <geom type="cylinder" size="0.05 0.05" mass="1"/>
            </body>
            <body name="cup_c" pos="0 -0.1 0">
                <joint type="free" name="cup_c_joint"/>
                <geom type="cylinder" size="0.05 0.05" mass="1"/>
            </body>
            <body name="ball" pos="0 0 0">
                <joint type="free" name="ball_joint"/>
                <geom type="sphere" size="0.02" mass="0.1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    config = ShellGameRuleConfig(
        target_object="cup_b",
        cup_names=("cup_a", "cup_b", "cup_c"),
        shuffle_min_swaps=1,
        shuffle_max_swaps=1,
        max_steps=2000,
    )
    rules = ShellGameRules(config, model)

    # Initialize randomization
    rng = np.random.default_rng(42)
    rules.randomize_scene(data, rng)
    mujoco.mj_forward(model, data)

    return rules, data, model


def test_shell_game_initial_state(shell_game_setup):
    rules, data, model = shell_game_setup
    result = rules.compute(data)
    assert result.info["state"] == ShellGameRules.STATE_SHOWING


def test_shell_game_state_transitions(shell_game_setup):
    rules, data, model = shell_game_setup
    advance_to_state(rules, data, model, ShellGameRules.STATE_COVERING)
    advance_to_state(rules, data, model, ShellGameRules.STATE_SHUFFLING)
    advance_to_state(rules, data, model, ShellGameRules.STATE_TESTING)


def test_shell_game_success(shell_game_setup):
    rules, data, model = shell_game_setup
    result = advance_to_state(rules, data, model, ShellGameRules.STATE_TESTING)

    target_slot = result.info["target_slot"]
    cup_to_slot = rules._cup_to_slot
    cup_idx = cup_to_slot.index(target_slot)

    # Lift the correct cup
    qadr = rules._cup_qpos_adrs[cup_idx]
    mujoco.mj_forward(model, data)
    data.qpos[qadr + 2] += 0.1
    mujoco.mj_forward(model, data)

    result = rules.compute(data)
    assert result.info[RuleInfoKey.IS_SUCCESS]
    assert result.terminated


def test_shell_game_failure(shell_game_setup):
    rules, data, model = shell_game_setup
    result = advance_to_state(rules, data, model, ShellGameRules.STATE_TESTING)

    target_slot = result.info["target_slot"]
    wrong_slot = (target_slot + 1) % 3
    cup_to_slot = rules._cup_to_slot
    cup_idx = cup_to_slot.index(wrong_slot)

    qadr = rules._cup_qpos_adrs[cup_idx]
    mujoco.mj_forward(model, data)
    data.qpos[qadr + 2] += 0.1
    mujoco.mj_forward(model, data)

    result = rules.compute(data)
    assert not result.info[RuleInfoKey.IS_SUCCESS]
    assert result.info[RuleInfoKey.DROPPED]
    assert result.terminated
