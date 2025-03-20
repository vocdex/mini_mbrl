# 🔥 MiniMBRL

MiniMBRL is a minimal implementation of model-based reinforcement learning algorithms. The goal is to implement the most basic version of the algorithms to understand the core concepts and to build on top of them.

<div align="center">
    <img src="mini_mbrl/02_world_models/plots/car_racing.gif" width="800" alt="Pong Wars" />
</div>

## Installation

1. Install and activate a new python3.8 virtualenv:
```sh
virtualenv mbrl_venv --python=python3.8
source mbrl_venv/bin/activate
```

2. Install requirements:
```sh
pip install "gymnasium[all]"
pip install mujoco
```

## Implementation Status

Below is a structured list of algorithms that will be implemented, organized by category:

### Dyna-style Algorithms

| Paper | Link | Personal Notes | Status |
|-------|------|---------------|--------|
| Dyna: An Integrated Architecture for Learning, Planning, and Reacting | [Link](https://dl.acm.org/doi/pdf/10.1145/122344.122377) | [Dyna Notes](notes/01_Dyna.pdf) | Done |
| World Models | [Link](https://arxiv.org/pdf/1803.10122.pdf) | - | Ongoing|
| PlaNet: Planning Network | [Link](https://arxiv.org/pdf/1811.04551.pdf) | - | Planned |
| Dream to Control: Learning Behaviors by Latent Imagination (Dreamer V1) | [Link](https://arxiv.org/abs/1912.01603) | - | Planned |
| Mastering Atari with Discrete World Models (Dreamer V2) | [Link](https://arxiv.org/pdf/2010.02193) | - | Planned |
| DayDreamer: World Models for Physical Robot Learning | [Link](https://arxiv.org/pdf/2206.14176) | - | Planned |
| Planning to Explore via Self-Supervised World Models | [Link](https://arxiv.org/pdf/2005.05960) | - | Planned |

### MPC-based Algorithms

| Paper | Link | Personal Notes | Status |
|-------|------|---------------|--------|
| Neural Network Dynamics for Model-Based Deep Reinforcement Learning | [Link](https://arxiv.org/pdf/1708.02596) | - | Planned |
| Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models | [Link](https://arxiv.org/abs/1805.12114) | - | Planned |

### Newer Papers

| Paper | Link | Personal Notes | Status |
|-------|------|---------------|--------|
| Transformers are Sample-Efficient World Models | [Link](https://openreview.net/pdf?id=vhFu1Acb0xb) | - | Planned |
| The Benefits of Model-Based Generalization in Reinforcement Learning (ICML 2024) | [Link](https://arxiv.org/pdf/2211.02222) | - | Planned |
| Facing Off World Model Backbones: RNNs, Transformers, and S4 (NeurIPS 2023) | [Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6c65eb9b56719c1aa45ff73874de317-Paper-Conference.pdf) | - | Planned |

### Exploration-focused Algorithms

| Paper | Link | Personal Notes | Status |
|-------|------|---------------|--------|
| Curiosity-driven Exploration by Self-supervised Prediction | [Link](https://pathak22.github.io/noreward-rl/resources/icml17.pdf) | - | Planned |
| Curious exploration via structured world models yields zero-shot object manipulation | [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/98ecdc722006c2959babbdbdeb22eb75-Paper-Conference.pdf) | - | Planned |
| SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models | [Link](https://openreview.net/pdf?id=dHNVY5qMiP) | - | Planned |

### Language-oriented Algorithms

| Paper | Link | Personal Notes | Status |
|-------|------|---------------|--------|
| Learning to Model the World with Language | [Link](https://arxiv.org/abs/2308.01399) | - | Planned |
| Language-Guided World Models: A Model-Based Approach to AI Control | [Link](https://arxiv.org/abs/2402.01695) | - | Planned |

### Other Algorithms

| Paper | Link | Personal Notes | Status |
|-------|------|---------------|--------|
| Contrastive Learning of Structured World Models | [Link](https://arxiv.org/pdf/1911.12247) | - | Planned |
