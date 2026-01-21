# 🦫 BEAVR Sim

## GPU-Accelerated RL & Imitation Learning for Robotic Manipulation

![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![MuJoCo](https://img.shields.io/badge/MuJoCo-Menagerie-orange)
![LeRobot](https://img.shields.io/badge/LeRobot-Compatible-yellow)
![GPU](https://img.shields.io/badge/GPU-Acceleration-green)

**BEAVR Sim** is a high-performance simulation benchmark suite designed to test and evaluate physical AI algorithms. It unifies state-of-the-art tools like **[MuJoCo](https://github.com/google-deepmind/mujoco)**, **[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)**, **[Isaac Lab](https://github.com/mujocolab/mjlab)**, and **[LeRobot](https://github.com/huggingface/lerobot)** into a single, cohesive platform for robotic learning.

Whether you are researching imitation learning, reinforcement learning, or simply need a simulation environment for your robot, BEAVR Sim provides the performance needed to iterate quickly.

---

## Key Features

- **GPU Acceleration**: Run parallel environments on a single GPU for massive throughput.
- **Unified API**: Seamless integration with **LeRobot** for training and evaluating policies.
- **Robot & Scene Support**: Robots and scenes are pre-built using the **MuJoCo Menagerie** library in **MuJoCo**.
- **Advanced Tasks**: Includes pre-built scenes testing memory, precision, and dexterity.
- **IL & RL Ready**: 4 pre-built scenes with **[100 human demonstrations each](https://huggingface.co/collections/arclabmit/beavr-sim)** collected via [BEAVR-teleop](https://github.com/ARCLab-MIT/beavr-bot). Each scene is compliant with [gymnasium](https://github.com/Farama-Foundation/Gymnasium) and follows its own ruleset.

---

## Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- CUDA-capable GPU (recommended)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ARCLab-MIT/beavr-bench.git
cd beavr-bench

# 2. Install dependencies with uv
uv sync

# 3. Verify installation
uv run beavr-eval --help
```

---

## Demonstrations

We include 4 main scenes designed to test physical intelligence, ranging from simple manipulation to complex memory-based challenges.

<table>
  <tr>
    <th width="50%">Scene</th>
    <th width="50%">Description</th>
  </tr>
  <tr>
    <td align="center">
      <h3>Pick and Place</h3>
      <img src="media/pickplace.gif" width="100%" alt="Pick and Place">
    </td>
    <td>
      <p><strong>Simple manipulation task demonstrating object grasping and placement.</strong></p>
      <p>The system learns to pick up objects from randomized positions and place them at a target location.</p>
      <p>
        <code>Manipulation</code> <code>Grasping</code> <code>IL</code>
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <h3>Shell Game</h3>
      <img src="media/shellgame.gif" width="100%" alt="Shell Game">
    </td>
    <td>
      <p><strong>Advanced memory and tracking task.</strong></p>
      <p>The robot must track the cup containing a concealed object. Designed to test object permanence and memory amid occlusions.</p>
      <p>
        <code>Tracking</code> <code>Memory</code> <code>IL</code>
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <h3>Server Swap (Overhead)</h3>
      <img src="media/serverswap_overhead.gif" width="100%" alt="Server Swap Overhead">
    </td>
    <td>
      <p><strong>Mobile manipulation and precision assembly.</strong></p>
      <p>Top-down perspective where the robot coordinates its arm and mobile base to insert server components into the correct slot indicated by a transient LED.</p>
      <p>
        <code>Egocentric</code> <code>Precision</code> <code>Assembly</code>
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <h3>Server Swap (Egocentric)</h3>
      <img src="media/serverswap_ego.gif" width="100%" alt="Server Swap Ego">
    </td>
    <td>
      <p><strong>First-person manipulation perspective.</strong></p>
      <p>A closer look at the detailed interaction from the robot's viewpoint during the server module replacement task.</p>
      <p>
        <code>Egocentric</code> <code>Precision</code> <code>Assembly</code>
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <h3>Vanishing Blueprint</h3>
      <img src="media/vanishing_blueprint.gif" width="100%" alt="Vanishing Blueprint">
    </td>
    <td>
      <p><strong>Sequential memory and stacking task.</strong></p>
      <p>A blueprint is shown for 5 seconds before vanishing. The robot must recall the correct order of objects to stack them successfully.</p>
      <p>
        <code>Stacking</code> <code>Memory</code> <code>Configuration</code>
      </p>
    </td>
  </tr>
</table>

### Datasets

Access our pre-collected demonstration datasets for imitation learning research on HuggingFace.

[🤗 **View Datasets on HuggingFace**](https://huggingface.co/collections/arclabmit/beavr-sim)

---

## Usage

BEAVR Sim is fully compatible with LeRobot scripts.

### Training a Policy

```bash
beavr-train \
    --policy.type=act \
    --dataset.repo_id=arclabmit/xarm7_beavrsim_shellgame_dataset \
    --policy.repo_id=arclabmit/xarm7_act_beavrsim_shellgame_model \
    --dataset.video_backend=pyav \
    --env.type=beavr \
    --env.scene=scene_serverswap \
    --eval.batch_size=25 \
    --wandb.enable=true \
    --job_name=xarm7_act_beavrsim_shellgame_model
```

### Evaluating a Policy

```bash
beavr-eval \
    --policy.path=arclabmit/xarm7_act_beavrsim_shellgame_model \
    --env.scene=scene_shellgame \
    --eval.n_episodes=50
```

---

## Contributing

We welcome contributions! Whether it's adding a new robot from the Menagerie, designing a new task, or fixing a bug.

Please check out our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Citations

If you use BEAVR Sim in your research, please cite the following works:

### BEAVR-teleop

```bibtex
@misc{posadasnava2025beavr,
  title         = {BEAVR: Bimanual, multi-Embodiment, Accessible, Virtual Reality Teleoperation System for Robots},
  author        = {Alejandro Posadas-Nava and Alejandro Carrasco and Richard Linares},
  year          = {2025},
  eprint        = {2508.09606},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  note          = {Accepted for presentation at ICCR 2025, Kyoto},
  url           = {https://arxiv.org/abs/2508.09606}
}
```

### LeRobot

```bibtex
@misc{cadene2024lerobot,
  author = {Cadene, Remi, et al.},
  title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
  howpublished = "\url{https://github.com/huggingface/lerobot}",
  year = {2024}
}
```

### MJLab

```bibtex
@software{Zakka_mjlab_Isaac_Lab_2025,
  author = {Zakka, Kevin and Yi, Brent and Liao, Qiayuan and Le Lay, Louis},
  title = {{mjlab: Isaac Lab API, powered by MuJoCo-Warp, for RL and robotics research.}},
  url = {https://github.com/mujocolab/mjlab},
  year = {2025}
}
```

### MuJoCo Menagerie

```bibtex
@software{menagerie2022github,
  author = {Zakka, Kevin and Tassa, Yuval and {MuJoCo Menagerie Contributors}},
  title = {{MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo}},
  url = {http://github.com/google-deepmind/mujoco_menagerie},
  year = {2022},
}
```

### Gymnasium

```bibtex
@article{towers2024gymnasium,
  title={Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  author={Towers, Mark and Kwiatkowski, Ariel and Terry, Jordan and Balis, John U and De Cola, Gianluca and Deleu, Tristan and Goul{\~a}o, Manuel and Kallinteris, Andreas and Krimmel, Markus and KG, Arjun and others},
  journal={arXiv preprint arXiv:2407.17032},
  year={2024}
}
```

---

<p align="center">
  Built at <a href="https://github.com/ARCLab-MIT">MIT's ARCLab</a> for Space Robotics Research
</p>
