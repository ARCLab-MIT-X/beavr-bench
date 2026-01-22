# 🦫 BEAVR Bench

## GPU-Accelerated RL & Imitation Learning for Robotic Manipulation

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![CI](https://github.com/ARCLab-MIT/beavr-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/ARCLab-MIT/beavr-bench/actions/workflows/ci.yml)
[![GitHub Stars](https://img.shields.io/github/stars/ARCLab-MIT/beavr-bench?style=social)](https://github.com/ARCLab-MIT/beavr-bench)
[![arXiv](https://img.shields.io/badge/arXiv-2508.09606-b31b1b.svg)](https://arxiv.org/abs/2508.09606)
[![HuggingFace](https://img.shields.io/badge/🤗-Datasets-yellow)](https://huggingface.co/collections/arclabmit/beavr-sim)

**BEAVR Bench** is a simulation benchmark suite designed to test and evaluate physical AI algorithms.

It unifies state-of-the-art tools like **[MuJoCo](https://github.com/google-deepmind/mujoco)**, **[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)**, **[Isaac Lab](https://github.com/mujocolab/mjlab)**, and **[LeRobot](https://github.com/huggingface/lerobot)** into a single, cohesive benchmarking platform for robotic learning.

Whether you are researching imitation learning, reinforcement learning, BEAVR Bench provides the performance needed to iterate quickly.

---

## 📑 Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Docker](#docker)
- [Usage](#usage)
- [Releases](#releases)
- [Getting Help](#getting-help)
- [Contributing](#contributing)
- [License](#license)
- [Citations](#citations)

---

## Key Features

- **GPU Acceleration**: Run parallel environments on a single GPU for massive throughput.
- **Unified API**: Seamless integration with **LeRobot** for training and evaluating policies.
- **Robot & Scene Support**: Robots and scenes are pre-built using the **MuJoCo Menagerie** library in **MuJoCo**.
- **Advanced Tasks**: Includes pre-built scenes testing memory, precision, and dexterity.
- **IL & RL Ready**: 4 pre-built scenes with **[100 human demonstrations each](https://huggingface.co/collections/arclabmit/beavr-sim)** collected via [BEAVR-teleop](https://github.com/ARCLab-MIT/beavr-bot). Each scene is compliant with [gymnasium](https://github.com/Farama-Foundation/Gymnasium) and follows its own ruleset.

---

## Installation

### Installation Prerequisites

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

## Docker

BEAVR Bench can be run with Docker.

### Docker Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support)

### Building the Image

```bash
docker build -t beavr-bench .
```

### Running with GPU Support

To run tests inside the container:

```bash
docker run --rm --gpus all \
    -e MUJOCO_GL=egl \
    beavr-bench
```

To run a specific script:

```bash
docker run --rm --gpus all \
    -e MUJOCO_GL=egl \
    beavr-bench uv run beavr-eval --help
```

---

## Demonstrations

We include 4 main scenes designed to test physical intelligence. For a detailed breakdown of rules and task definitions, see [**SCENES.md**](docs/SCENES.md).

<div align="center">
<table>
  <tr>
    <td align="center">
      <b>Pick and Place</b><br>
      <img src="media/videos/pickplace.gif" width="400px">
    </td>
    <td align="center">
      <b>Shell Game</b><br>
      <img src="media/videos/shellgame.gif" width="400px">
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Server Swap</b><br>
      <img src="media/videos/serverswap.gif" width="400px">
    </td>
    <td align="center">
      <b>Vanishing Blueprint</b><br>
      <img src="media/videos/vanishing_blueprint.gif" width="400px">
    </td>
  </tr>
</table>
</div>

> **Note:** Access our pre-collected demonstration datasets for imitation learning research on HuggingFace:
> [🤗 **View Datasets on HuggingFace**](https://huggingface.co/collections/arclabmit/beavr-sim)

---

## Usage

BEAVR Bench is fully compatible with LeRobot scripts.

### Training a Policy

```bash
beavr-train \
    --policy.type=act \
    --dataset.repo_id=arclabmit/xarm7_beavrsim_shellgame_dataset \
    --policy.repo_id=arclabmit/xarm7_act_beavrsim_shellgame_model \
    --dataset.video_backend=pyav \
    --env.type=beavr \
    --env.scene=scene_shellgame \
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

## Releases

This project uses automated releases via GitHub Actions.

### Triggering a Release

To create a new release:

1. Update the version in `pyproject.toml`.
2. Commit and push the change.
3. Create and push a new tag:

```bash
git tag v1.0.1
git push origin v1.0.1
```

The GitHub Action will automatically build the package and create a new GitHub Release with the artifacts attached.

---

## Contributing

We welcome contributions! Whether it's adding a new robot from the Menagerie, designing a new task, or fixing a bug.

Please check out our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Getting Help

Need assistance? Here's how to get support:

- **📖 Documentation**: Check out [SCENES.md](docs/SCENES.md) for detailed task descriptions.
- **🐛 Bug Reports**: [Open an issue](https://github.com/ARCLab-MIT/beavr-bench/issues/new) on GitHub.
- **💬 Discussions**: Join the conversation in [GitHub Discussions](https://github.com/ARCLab-MIT/beavr-bench/discussions).
- **📧 Contact**: Reach out to the maintainers at [alexposadas24@gmail.com](mailto:alexposadas24@gmail.com).

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## Citations

If you use BEAVR Bench in your research, please cite the following works:

### BEAVR Teleop

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
  Built with ❤️ at <a href="https://github.com/ARCLab-MIT"><b>MIT's ARCLab</b></a> for Space Robotics Research
</p>
