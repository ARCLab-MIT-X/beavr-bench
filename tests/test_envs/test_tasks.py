import gymnasium as gym
import numpy as np
import pytest

from beavr_bench.registry import register_beavr_envs

# Ensure environments are registered
register_beavr_envs()


class NoPixelDeterminismWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Wrapper that removes 'pixels' from observation for strict determinism checks.

    MuJoCo rendering can have 1-pixel noise which fails Gymnasium's strict equality checks.
    We verify semantic determinism (agent_pos) strictly and pixel determinism with tolerance
    in separate tests.
    """

    def __init__(self, env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        # Modify observation space to remove pixels
        spaces = {k: v for k, v in self.env.observation_space.spaces.items() if k != "pixels"}
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, obs):
        return {k: v for k, v in obs.items() if k != "pixels"}


@pytest.mark.parametrize(
    "scene_name", ["scene_pickplace", "scene_serverswap", "scene_shell_game", "scene_vanishing_blueprint"]
)
def test_gym_api_compliance(scene_name):
    """Verify that the environment complies with the Gymnasium API."""
    from gymnasium.utils.env_checker import check_env

    env_id = f"beavr_bench/Teleop-{scene_name}-v0"
    env = gym.make(env_id)

    # Wrap to allow check_env to pass its strict determinism check
    # check_env checks both reset() and step() determinism
    wrapped_env = NoPixelDeterminismWrapper(env.unwrapped)

    # Gymnasium check_env captures warnings and errors
    # Note: skip_render_check=True because we don't necessarily have a display in CI
    check_env(wrapped_env, skip_render_check=True)
    env.close()


@pytest.mark.parametrize(
    "scene_name", ["scene_pickplace", "scene_serverswap", "scene_shell_game", "scene_vanishing_blueprint"]
)
def test_reset_determinism(scene_name):
    """Verify that resetting with the same seed results in the same initial observation."""
    env_id = f"beavr_bench/Teleop-{scene_name}-v0"
    env = gym.make(env_id)

    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)

    # Check agent_pos (flat vector) - must be EXACTLY deterministic
    assert np.array_equal(obs1["agent_pos"], obs2["agent_pos"]), f"agent_pos mismatch in {scene_name}"

    # Check pixels if present - allow for 1.0 pixel jitter
    if "pixels" in obs1:
        for cam_key in obs1["pixels"]:
            diff = obs1["pixels"][cam_key].astype(float) - obs2["pixels"][cam_key].astype(float)
            max_diff = np.max(np.abs(diff))
            assert max_diff <= 1.0, f"Pixel mismatch > 1.0 in {scene_name} camera {cam_key}"

    env.close()


@pytest.mark.parametrize(
    "scene_name", ["scene_pickplace", "scene_serverswap", "scene_shell_game", "scene_vanishing_blueprint"]
)
def test_step_determinism(scene_name):
    """Verify that stepping with the same action after the same reset is deterministic."""
    env_id = f"beavr_bench/Teleop-{scene_name}-v0"
    env = gym.make(env_id)

    # First sequence
    env.reset(seed=42)
    action = env.action_space.sample()
    obs1, _, _, _, _ = env.step(action)

    # Second sequence
    env.reset(seed=42)
    obs2, _, _, _, _ = env.step(action)

    # Check agent_pos
    assert np.array_equal(obs1["agent_pos"], obs2["agent_pos"]), f"Step agent_pos mismatch in {scene_name}"

    # Check pixels
    if "pixels" in obs1:
        for cam_key in obs1["pixels"]:
            diff = obs1["pixels"][cam_key].astype(float) - obs2["pixels"][cam_key].astype(float)
            max_diff = np.max(np.abs(diff))
            assert max_diff <= 1.0, f"Step pixel mismatch > 1.0 in {scene_name} camera {cam_key}"

    env.close()


@pytest.mark.parametrize(
    "scene_name", ["scene_pickplace", "scene_serverswap", "scene_shell_game", "scene_vanishing_blueprint"]
)
def test_observation_space_sample(scene_name):
    """Verify that observations match the defined observation space."""
    env_id = f"beavr_bench/Teleop-{scene_name}-v0"
    env = gym.make(env_id)

    obs, _ = env.reset(seed=42)

    # Check observation space structure
    assert env.observation_space.contains(obs), f"Observation for {scene_name} not in observation_space"

    # Step once and check again
    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    assert env.observation_space.contains(obs), (
        f"Observation after step for {scene_name} not in observation_space"
    )

    env.close()
