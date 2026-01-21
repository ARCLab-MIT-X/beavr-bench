import mujoco
import numpy as np
import pytest

from beavr_bench.sim import load_baked_scene


def test_available_scenes(available_scenes):
    """Verify that we have scenes in the assets folder."""
    assert len(available_scenes) > 0
    assert "scene_pickplace" in available_scenes


@pytest.mark.parametrize(
    "scene_name", ["scene_pickplace", "scene_serverswap", "scene_shell_game", "scene_vanishing_blueprint"]
)
def test_scene_loading(scene_name):
    """Verify that each baked scene can be loaded without error."""
    scene = load_baked_scene(scene_name)
    assert scene is not None
    assert isinstance(scene.model, mujoco.MjModel)
    assert len(scene.robots) > 0


@pytest.mark.parametrize(
    "scene_name", ["scene_pickplace", "scene_serverswap", "scene_shell_game", "scene_vanishing_blueprint"]
)
def test_physics_sanity(scene_name):
    """Verify that the physics doesn't explode on the first few steps."""
    scene = load_baked_scene(scene_name)
    data = mujoco.MjData(scene.model)

    # Reset and step a few times
    mujoco.mj_resetData(scene.model, data)
    scene.apply_initial_positions(data)

    for _ in range(10):
        mujoco.mj_step(scene.model, data)
        # Check for NaN or Inf in velocities
        assert not np.any(np.isnan(data.qvel))
        assert not np.any(np.isinf(data.qvel))
        # Velocities should be reasonably small initially
        assert np.max(np.abs(data.qvel)) < 100.0


@pytest.mark.parametrize(
    "scene_name", ["scene_pickplace", "scene_serverswap", "scene_shell_game", "scene_vanishing_blueprint"]
)
def test_camera_existence(scene_name):
    """Verify that all cameras in metadata exist in the XML model."""
    scene = load_baked_scene(scene_name)
    for cam_cfg in scene.cameras:
        cam_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_cfg.name)
        assert cam_id != -1, f"Camera {cam_cfg.name} not found in model {scene_name}"
