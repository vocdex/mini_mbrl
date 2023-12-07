
# 🔥 MiniMBRL

MiniMBRL is a minimal model-based reinforcement learning framework.

## Installation

```sh
pip install mini_mbrl
```

## Features

MiniMBRL aims to provide minimal implementations of the commonly used model-based RL algorithms. It is designed to be easy to use and easy to extend.

### `mini_mbrl.Controller`

```python
from mini_mbrl.controller import Controller
controller = Controller(algo="mpc_with_cem", env=env, model=model, **kwargs)
action = controller.get_action(observation)
```
```


## Questions

Please file an [issue on Github](https://github.com/vocdex/mini_mbrl/issues).