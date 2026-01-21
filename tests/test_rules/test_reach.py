import mujoco
import pytest

from beavr_bench.envs.rules.reach import ReachRules
from beavr_bench.schemas import ReachRuleConfig, RuleInfoKey


@pytest.fixture
def reach_test_setup():
    """Create a minimal model with a site and a body for reaching tests."""
    xml = """
    <mujoco>
        <worldbody>
            <body name="target" pos="0 0 0">
                <joint type="free" name="target_joint"/>
                <geom type="box" size="0.02 0.02 0.02" mass="1"/>
            </body>
            <site name="goal" pos="0.1 0.1 0.5" size="0.01"/>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    config = ReachRuleConfig(
        target_object="target",
        goal_site="goal",
        success_distance=0.05,
        fail_on_drop=True,
        drop_height_threshold=0.1,
        max_steps=10,
    )
    rules = ReachRules(config, model)
    return rules, data, model


def test_reach_success(reach_test_setup):
    """Test that placing the object at the goal results in success."""
    rules, data, model = reach_test_setup

    # Move target to goal using joint qpos
    # Free joint qpos: [x, y, z, qw, qx, qy, qz]
    data.qpos[:3] = [0.1, 0.1, 0.5]

    # Update positions
    mujoco.mj_forward(model, data)

    result = rules.compute(data)
    assert result.info[RuleInfoKey.IS_SUCCESS]
    assert result.terminated
    assert result.reward > 0


def test_reach_dropped(reach_test_setup):
    """Test that dropping the object below threshold results in failure."""
    rules, data, model = reach_test_setup

    # Move target below drop threshold
    data.qpos[:3] = [0, 0, 0.05]  # Below 0.1

    mujoco.mj_forward(model, data)

    result = rules.compute(data)
    assert result.info[RuleInfoKey.DROPPED]
    assert result.terminated
    assert result.reward < 0


def test_reach_truncation(reach_test_setup):
    """Test that exceeding max steps results in truncation."""
    rules, data, model = reach_test_setup

    # Step 10 times (max_steps is 10)
    for _ in range(9):
        rules.compute(data)

    result = rules.compute(data)
    assert result.truncated is True
    assert result.info[RuleInfoKey.STEPS] == 10
