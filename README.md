
# 🔥 MiniMBRL
MiniMBRL is a minimal model-based reinforcement learning framework.

# Development details
## 21.12.2023
We need to start simple. The goal here is to implement minimally possible MBRL algorithm in one script. Later, we can build on top of it.




We need a policy that can be used to collect data. This policy can be a random policy, or a policy that will be trained with model-based RL. We will use a random policy for now.

Now, since we have a policy, we can collect trajectories and write them to a replay buffer. We will use a simple replay buffer for now.

## Installation
1. Install and activate a new python3.8 virtualenv:
```sh
virtualenv mbrl_venv --python=python3.8
```
```sh
source mbrl_venv/bin/activate
```

## Features

MiniMBRL aims to provide minimal implementations of the commonly used model-based RL algorithms. It is designed to be easy to use and easy to extend.

### `mini_mbrl.Controller`

```python
from mini_mbrl.controller import Controller
controller = Controller(algo="mpc_with_cem", env=env, model=model, **kwargs)
action = controller.get_action(observation)
```
## Next Steps
Implement the following at this order
1. Alternating between model learning and planning loop
2. Gradient-free planning algorithms: Random Shooting, Cross-Entropy Method, iCEM
3. Closed-loop planning with MPC, and Open-loop
4. Model learning: deterministic vs probabilistic models, ensembles, uncertainty estimation(aleatoric vs epistemic)
5. Gradient-based optimization methods.

Papers:
- [Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning](https://arxiv.org/pdf/1708.02596.pdf)
- [Deep Dynamics Models for Learning Dexterous Manipulation](https://proceedings.mlr.press/v100/nagabandi20a/nagabandi20a.pdf)
- [PILCO: A Model-Based and Data-Efficient Approach to Policy Search](https://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf)
- [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/pdf/1805.12114.pdf)
- Dreamer papers ...



## Questions

Please file an [issue on Github](https://github.com/vocdex/mini_mbrl/issues).
