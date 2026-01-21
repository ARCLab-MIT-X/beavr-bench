"""Training entry point for beavr-bench.

Wraps lerobot-train to ensure beavr-bench environments are registered.
All standard lerobot-train arguments are supported.

Example:
    beavr-train \
        --policy.type=act \
        --dataset.repo_id=arclabmit/xarm7_beavrsim_shellgame_dataset \
        --policy.repo_id=arclabmit/xarm7_act_beavrsim_shellgame_model \
        --dataset.video_backend=pyav \
        --env.type=beavr \
        --env.scene=scene_serverswap \
        --eval.batch_size=25 \
        --wandb.enable true \
        --job_name xarm7_act_beavrsim_shellgame_model \
"""

from lerobot.scripts.lerobot_train import main

import beavr_bench  # noqa: F401

if __name__ == "__main__":
    main()
