import numpy as np

from beavr_bench.envs.tasks import make_teleop_env
from beavr_bench.schemas import RuleInfoKey, SceneId


def test_server_swap_seed_determinism():
    # Same seed should produce the same failing slot across separate env instances
    env1 = make_teleop_env(scene_name=SceneId.SERVER_SWAP, seed=2026)
    obs1, reward1, terminated1, truncated1, info1 = env1.step(np.zeros(env1.action_space.shape))
    env1.close()

    env2 = make_teleop_env(scene_name=SceneId.SERVER_SWAP, seed=2026)
    obs2, reward2, terminated2, truncated2, info2 = env2.step(np.zeros(env2.action_space.shape))
    env2.close()

    assert info1[RuleInfoKey.FAILING_SLOT] == info2[RuleInfoKey.FAILING_SLOT]


def test_vanishing_blueprint_seed_determinism():
    # Same seed should produce the same blueprint order across separate env instances
    env1 = make_teleop_env(scene_name=SceneId.VANISHING_BLUEPRINT, seed=2026)
    obs1, reward1, terminated1, truncated1, info1 = env1.step(np.zeros(env1.action_space.shape))
    env1.close()

    env2 = make_teleop_env(scene_name=SceneId.VANISHING_BLUEPRINT, seed=2026)
    obs2, reward2, terminated2, truncated2, info2 = env2.step(np.zeros(env2.action_space.shape))
    env2.close()

    assert info1["blueprint_order"] == info2["blueprint_order"]
