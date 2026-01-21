"""Training entry point for beavr-bench.

Wraps lerobot-train to ensure beavr-bench environments are registered.
All standard lerobot-train arguments are supported.

Example:
    beavr-train --policy.path=... --env.type=beavr --env.scene=scene_pickplace
"""

from lerobot.scripts.lerobot_train import main

import beavr_bench  # noqa: F401

if __name__ == "__main__":
    main()
