"""Evaluation entry point for beavr-bench.

Wraps lerobot-eval to ensure beavr-bench environments are registered.
All standard lerobot-eval arguments are supported.

Example:
    beavr-eval \
        --policy.path=arclabmit/mobile_xarm7_act_beavrsim_serverswap_model \
        --env.type=beavr \
        --env.scene=scene_serverswap \
        --eval.batch_size=25 \
        --eval.n_episodes=1000
"""

from lerobot.scripts.lerobot_eval import main

import beavr_bench  # noqa: F401

if __name__ == "__main__":
    main()
