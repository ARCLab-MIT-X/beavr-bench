import mujoco
import numpy as np
import pytest

from beavr_bench.envs.rules.vanishing_blueprint import VanishingBlueprintRules
from beavr_bench.schemas import RuleInfoKey, VanishingBlueprintRuleConfig


def advance_to_state(rules, data, model, target_state, max_calls=100):
    """Helper to advance the state machine by incrementing time and calling compute."""
    for _ in range(max_calls):
        result = rules.compute(data)
        if result.info["state"] == target_state:
            return result
        # Step time slightly
        data.time += 0.1
        mujoco.mj_forward(model, data)
    current = result.info["state"]
    raise RuntimeError(f"Failed to reach state {target_state}. Current state: {current}")


@pytest.fixture
def vanishing_blueprint_setup():
    xml = """
    <mujoco>
        <worldbody>
            <body name="orange_block" pos="0 0 0">
                <joint type="free" name="orange_joint"/>
                <geom type="box" size="0.02 0.02 0.02" mass="1"/>
            </body>
            <body name="skyblue_block" pos="0 0 0">
                <joint type="free" name="skyblue_joint"/>
                <geom type="box" size="0.02 0.02 0.02" mass="1"/>
            </body>
            <body name="bluishgreen_block" pos="0 0 0">
                <joint type="free" name="bluishgreen_joint"/>
                <geom type="box" size="0.02 0.02 0.02" mass="1"/>
            </body>

            <body name="holo_orange" pos="0 0 -1" mocap="true">
                <geom type="box" size="0.02 0.02 0.02" contype="0" conaffinity="0"/>
            </body>
            <body name="holo_skyblue" pos="0 0 -1" mocap="true">
                <geom type="box" size="0.02 0.02 0.02" contype="0" conaffinity="0"/>
            </body>
            <body name="holo_bluishgreen" pos="0 0 -1" mocap="true">
                <geom type="box" size="0.02 0.02 0.02" contype="0" conaffinity="0"/>
            </body>

            <site name="build_zone_site" pos="0.5 0.5 0.445" size="0.1"/>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    config = VanishingBlueprintRuleConfig(
        show_duration=1.0,
        max_steps=100,
    )
    rules = VanishingBlueprintRules(config, model)

    # Initialize randomization
    rng = np.random.default_rng(42)
    rules.randomize_scene(data, rng)
    mujoco.mj_forward(model, data)

    return rules, data, model


def test_vanishing_blueprint_initial_state(vanishing_blueprint_setup):
    rules, data, model = vanishing_blueprint_setup
    result = rules.compute(data)
    assert result.info["state"] == VanishingBlueprintRules.STATE_SHOWING


def test_vanishing_blueprint_transition(vanishing_blueprint_setup):
    rules, data, model = vanishing_blueprint_setup
    advance_to_state(rules, data, model, VanishingBlueprintRules.STATE_TESTING)
    result = rules.compute(data)  # noqa: F841
    for mocap_id in rules._hologram_mocap_ids:
        assert data.mocap_pos[mocap_id][2] == -1


def test_vanishing_blueprint_success(vanishing_blueprint_setup):
    rules, data, model = vanishing_blueprint_setup

    advance_to_state(rules, data, model, VanishingBlueprintRules.STATE_TESTING)

    blueprint_order = rules._blueprint_order
    build_zone_pos = data.site_xpos[rules._build_zone_site_id].copy()

    for stack_idx, block_idx in enumerate(blueprint_order):
        body_id = rules._block_body_ids[block_idx]
        jnt_idx = model.body_jntadr[body_id]
        qadr = model.jnt_qposadr[jnt_idx]

        expected_z = build_zone_pos[2] + 0.02 + stack_idx * rules.config.z_spacing
        data.qpos[qadr : qadr + 3] = [build_zone_pos[0], build_zone_pos[1], expected_z]

    mujoco.mj_forward(model, data)
    result = rules.compute(data)

    assert result.info[RuleInfoKey.IS_SUCCESS]
    assert result.terminated
