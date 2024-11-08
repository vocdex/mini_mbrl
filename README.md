
# 🔥 MiniMBRL
MiniMBRL is a minimal implementation of model-based reinforcement learning algorithms. The goal is to implement the most basic version of the algorithms to understand the core concepts and to build on top of them.


## Status
Here is the list of algorithms that I plan to implement:
### General:
#### Dyna-style algorithms:
- [Dyna: An Integrated Architecture for Learning, Planning, and Reacting](https://dl.acm.org/doi/pdf/10.1145/122344.122377). Paper Notes: [Dyna](notes/01_Dyna.pdf)
- [World Models](https://arxiv.org/pdf/1803.10122.pdf)
- [PlaNet: Planning Network](https://arxiv.org/pdf/1811.04551.pdf)
-  [Dream to Control: Learning Behaviors by Latent Imagination(Dreamer V1)](https://arxiv.org/abs/1912.01603)
- [Mastering Atari with Discrete World Models(Dreamer V2)](https://arxiv.org/pdf/2010.02193)
- [DayDreamer: World Models for Physical Robot Learning](https://arxiv.org/pdf/2206.14176)
- [Planning to Explore via Self-Supervised World Models](https://arxiv.org/pdf/2005.05960)

#### MPC-based algorithms:
- [Neural Network Dynamics for Model-Based Deep Reinforcement Learning](https://arxiv.org/pdf/1708.02596)
-  [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114)

#### Newer papers:
- [Transformers are Sample-Efficient World Models](https://openreview.net/pdf?id=vhFu1Acb0xb)
- [The Benefits of Model-Based Generalization in Reinforcement Learning](https://arxiv.org/pdf/2211.02222) (ICML 2024)
- [Facing Off World Model Backbones: RNNs, Transformers, and S4](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6c65eb9b56719c1aa45ff73874de317-Paper-Conference.pdf) (NeurIPS 2023)
#### Exploration-focused:
- [Curiosity-driven Exploration by Self-supervised Prediction](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)
- [Curious exploration via structured world models yields zero-shot object manipulation](https://proceedings.neurips.cc/paper_files/paper/2022/file/98ecdc722006c2959babbdbdeb22eb75-Paper-Conference.pdf)
- [SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models](https://openreview.net/pdf?id=dHNVY5qMiP)
#### Language-oriented:
- [Learning to Model the World with Language](https://arxiv.org/abs/2308.01399)
- [Language-Guided World Models: A Model-Based Approach to AI Control](https://arxiv.org/abs/2402.01695)

#### Other:
- [Contrastive Learning of Structured World Models](https://arxiv.org/pdf/1911.12247)

## Installation
1. Install and activate a new python3.8 virtualenv:
```sh
virtualenv mbrl_venv --python=python3.8
```
```sh
source mbrl_venv/bin/activate
```
#### Requirements
```sh
pip install "gymnasium[all]"
pip install mujoco
```
