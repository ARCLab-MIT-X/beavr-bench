"""
Uploads evaluation results from an eval_info.json to a Hugging Face Model Card.
Updates both the YAML metadata and the Markdown README.

Usage:
    python upload_eval.py --eval-json <path_to_json> --repo-id <user/repo>
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, ModelCard

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_markdown_table(metrics: dict, timestamp: str) -> str:
    """Creates a nicely formatted Markdown section for the README."""
    table = [
        "## Evaluation Results",
        f"*Evaluated on {timestamp}*",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Success Rate** | {metrics['pc_success']:.1f}% |",
        f"| **Average Reward** | {metrics['avg_sum_reward']:.3f} |",
        f"| **Max Reward (Avg)** | {metrics['avg_max_reward']:.3f} |",
        f"| **Episodes** | {metrics['n_episodes']} |",
        f"| **Eval Speed** | {metrics['eval_ep_s']:.2f} s/ep |",
        "| **Seed** | 26 |",
        "",
        "> [!TIP]",
        "> Detailed per-episode results can be found in [eval/eval_info.json](./eval/eval_info.json).",
        "",
    ]
    return "\n".join(table)


def main():
    parser = argparse.ArgumentParser(description="Upload evaluation results to HF Hub")
    parser.add_argument("--eval-json", type=str, required=True, help="Path to eval_info.json")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Repo ID (e.g. user/model)")
    args = parser.parse_args()

    eval_path = Path(args.eval_json)
    if not eval_path.exists():
        logger.error(f"File not found: {args.eval_json}")
        return

    # 1. Load Data
    with open(eval_path) as f:
        data = json.load(f)

    overall = data.get("overall", {})
    if not overall:
        logger.error("Could not find 'overall' metrics in JSON.")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    repo_id = args.repo_id

    # 2. Update YAML Metadata (for HF indexing)
    logger.info(f"Fetching model card for {repo_id}...")
    card = ModelCard.load(repo_id)

    # Use dictionary-style access for 'model-index' as it is the literal YAML key
    if "model-index" not in card.data:
        card.data["model-index"] = []

    eval_entry = {
        "task": {"type": "robotics", "name": "Robotic Manipulation"},
        "dataset": {"name": "beavr_sim", "type": "simulation"},
        "metrics": [
            {"type": "success_rate", "value": overall["pc_success"], "name": "Success Rate"},
            {"type": "reward", "value": overall["avg_sum_reward"], "name": "Avg Reward"},
        ],
    }

    # Update the results list for this model
    model_name = repo_id.split("/")[-1]
    found = False
    for entry in card.data["model-index"]:
        if entry.get("name") == model_name:
            entry["results"] = [eval_entry]
            found = True
            break
    if not found:
        card.data["model-index"].append({"name": model_name, "results": [eval_entry]})

    # 3. Update Markdown Content
    new_table_content = generate_markdown_table(overall, now)

    # Use markers to allow overwriting old results
    start_marker = "<!-- EVAL_RESULTS_START -->"
    end_marker = "<!-- EVAL_RESULTS_END -->"
    wrapped_content = f"\n{start_marker}\n{new_table_content}\n{end_marker}\n"

    content = card.content
    if start_marker in content and end_marker in content:
        # Replace existing section
        logger.info("Found existing evaluation section, updating it...")
        pre = content.split(start_marker)[0]
        post = content.split(end_marker)[1]
        card.content = pre + wrapped_content + post
    else:
        # Append to end
        logger.info("No existing evaluation section found, appending to end...")
        card.content = content.strip() + "\n\n" + wrapped_content

    # 4. Upload Files
    api = HfApi()

    logger.info(f"Uploading eval_info.json to {repo_id}/eval/...")
    api.upload_file(
        path_or_fileobj=str(eval_path), path_in_repo="eval/eval_info.json", repo_id=repo_id, repo_type="model"
    )

    logger.info(f"Pushing updated README.md to {repo_id}...")
    card.push_to_hub(repo_id)

    logger.info("Done! View your results at: https://huggingface.co/" + repo_id)


if __name__ == "__main__":
    main()
