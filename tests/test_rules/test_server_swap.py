import mujoco
import numpy as np
import pytest

from beavr_bench.envs.rules.server_swap import (
    OKABE_BLUISH_GREEN,
    OKABE_ORANGE,
    ServerSwapRules,
    ServerSwapState,
)
from beavr_bench.schemas import RuleInfoKey, ServerSwapRuleConfig


@pytest.fixture
def server_swap_setup():
    """Create a model with LED materials and sled bodies for swap tests."""
    xml = """
    <mujoco>
        <asset>
            <material name="led_slot0" rgba="0 1 0 1"/>
            <material name="led_slot1" rgba="0 1 0 1"/>
        </asset>
        <worldbody>
            <geom name="led0" pos="0 0 0" material="led_slot0" size="0.01"/>
            <geom name="led1" pos="0.2 0 0" material="led_slot1" size="0.01"/>
            <body name="sled4" pos="0 0 0">
                <joint type="free"/>
                <geom type="box" size="0.02 0.02 0.02" mass="1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    config = ServerSwapRuleConfig(
        led_material_names=("led_slot0", "led_slot1"),
        sled_names=("sled0", "sled1"),  # These aren't checked in logic, just replacement_sled
        replacement_sled="sled4",
        cue_duration=1.0,
        swap_tolerance=0.05,
    )
    rules = ServerSwapRules(config, model)
    return rules, data, model


def test_server_swap_cue_phase(server_swap_setup):
    """Test that orange LED is shown on the failing slot during cue."""
    rules, data, model = server_swap_setup
    failing_idx = rules._failing_slot_idx

    rules.compute(data)

    # Check materials
    mat_id = rules._led_material_ids[failing_idx]
    assert np.allclose(model.mat_rgba[mat_id], OKABE_ORANGE)

    other_idx = 1 - failing_idx
    other_mat_id = rules._led_material_ids[other_idx]
    assert np.allclose(model.mat_rgba[other_mat_id], OKABE_BLUISH_GREEN)


def test_server_swap_transition_to_vanished(server_swap_setup):
    """Test that LEDs turn green after cue duration."""
    rules, data, model = server_swap_setup

    # Fast forward time
    data.time = 1.1  # cue_duration is 1.0

    rules.compute(data)
    assert rules._state == ServerSwapState.VANISHED

    # All LEDs should be green
    for mat_id in rules._led_material_ids:
        assert np.allclose(model.mat_rgba[mat_id], OKABE_BLUISH_GREEN)


def test_server_swap_success(server_swap_setup):
    """Test correct placement results in success."""
    rules, data, model = server_swap_setup
    failing_idx = rules._failing_slot_idx
    target_pos = rules._slot_positions[failing_idx]

    # Move to vanished/testing state
    data.time = 1.1
    rules.compute(data)
    rules.compute(data)
    assert rules._state == ServerSwapState.TESTING

    # Move replacement sled to target pos
    data.qpos[:3] = target_pos
    mujoco.mj_forward(model, data)

    result = rules.compute(data)
    assert result.info[RuleInfoKey.IS_SUCCESS]
    assert result.terminated


def test_server_swap_failure(server_swap_setup):
    """Test wrong placement results in failure."""
    rules, data, model = server_swap_setup
    wrong_idx = 1 - rules._failing_slot_idx
    wrong_pos = rules._slot_positions[wrong_idx]

    # Move to testing state
    data.time = 1.1
    rules.compute(data)
    rules.compute(data)

    # Move replacement sled to wrong slot
    data.qpos[:3] = wrong_pos
    mujoco.mj_forward(model, data)

    result = rules.compute(data)
    assert not result.info[RuleInfoKey.IS_SUCCESS]
    assert result.info[RuleInfoKey.DROPPED]
    assert result.terminated
