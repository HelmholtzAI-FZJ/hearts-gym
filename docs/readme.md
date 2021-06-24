<link rel="stylesheet" href="style.css">

# Hearts Gym Documentation

This document serves as a quick overview on usage and features of the
Hearts Gym.

## Introduction

An multi-agent environment to train agents on playing the
[Hearts](https://en.wikipedia.org/wiki/Black_Lady) card game.

Also includes a client-server architecture to remotely evaluate local
agents.

The rules are mostly implemented as specified by the modern rules from
Morehead (2001). For a more detailed description of the rules and
differences from the original, execute the following:

```shell
python -m pydoc hearts_gym.envs.hearts_game.HeartsGame
```

## Installing

Supported Python versions are shown in `setup.py` under the
`python_requires` argument. Clone this repository so you can easily
modify it, replacing `<repo-uri>` with the URI of this repo.

```shell
git clone <repo-uri>
cd hearts-gym
# If `python3` is not found, try `python`.
# If the `venv` module is not found, please install it.
python3 -m venv --system-site-packages ./env
# On Unix:
source ./env/bin/activate
# On Windows:
.\env\Scripts\activate
```

Install at least one of
[PyTorch](https://pytorch.org/get-started/locally/) or
[TensorFlow](https://www.tensorflow.org/install) as your deep learning
framework (RLlib also has experimental
[JAX](https://github.com/google/jax#installation) support if you feel
adventurous).

After installing a deep learning framework, in the root directory of
the repository clone, execute:

```shell
python -m pip install --user --upgrade pip
python -m pip install -e .
```

## Usage

You will need to execute the following line each time you start a new
shell. This will activate the Python virtual environment we are using:

```shell
# On Unix:
source ./env/bin/activate
# On Windows:
.\env\Scripts\activate
```

### Training

We use [RLlib](https://docs.ray.io/en/master/rllib.html) with the
recommended [Tune](https://docs.ray.io/en/master/tune/index.html) API
to manage training experiments.

The main script for starting a training run is `train.py`, started
like this:

```shell
python train.py
```

If everything worked correctly, you should see a table summarizing
test results of your learned agent against other agents printed on
your terminal. If you see the table, you can ignore any other errors
displayed by Ray. The table looks something like this:

```python
[...]
(pid=10101) SystemExit: 1  # Can be ignored.
[...]
testing took 98.7654321 seconds
# illegal action (player 0): 0 / 63754
# illegal action ratio (player 0): 0.0
| policy  | # rank 1 | # rank 2 | # rank 3 | # rank 4 | total penalty |
|---------+----------+----------+----------+----------+---------------|
| learned |    [...] |    [...] |    [...] |    [...] |         [...] |
| random  |    [...] |    [...] |    [...] |    [...] |         [...] |
| random  |    [...] |    [...] |    [...] |    [...] |         [...] |
| random  |    [...] |    [...] |    [...] |    [...] |         [...] |
```

In `train.py`, you will find lots of [configuration options which are
described here](#configuration). Results including configuration and
checkpoints are saved in the `results` directory by default. After
training, your agent is automatically evaluated as well.

To optimize your agent, the main thing you want to modify is the
`hearts_gym.RewardFunction.compute_reward` method in
`hearts_gym/envs/reward_function.py` with which you can shape the
reward function for your agent, adjusting its behaviour. Variables and
functions that may help you during this step are described in
[`docs/reward-shaping.md`](./reward-shaping.md).

You should not modify the observations of the environment because we
rely on their structure for remote interaction. If you do decide to
modify them, you need to apply the same transformations in the
`eval_agent.py` script so that the observations received from the
remote server match what your trained model expects.

### Evaluation

Aside from the local evaluation in `train.py`, you can start a server
and connect to it with different clients. You may want to configure
the variables `SERVER_ADDRESS` and `PORT` in
`hearts_gym/envs/hearts_server.py` to obtain sensible defaults.

To start the server, execute the following:

```shell
python start_server.py --num_parallel_games 16
```

To connect to the server for evaluation, execute the following:

```shell
python eval_agent.py <checkpoint_path> \
    --name <name> --algorithm <algo> --framework <framework>
```

Replace `<name>` with a name you want to have displayed,
`<checkpoint_path>` with the path to a checkpoint, `<algo>` with the
name of the algorithm you used for training the agent, and
`<framework>` with the configuration string of the framework you used
to train it. Here is an example:

```shell
python eval_agent.py results/PPO/PPO_Hearts-v0_00000_00000_0_1970-01-01_00-00-00/checkpoint_000002/checkpoint-2 \
    --name '🂭-collector' --algorithm PPO --framework torch
```

Since the server will wait until enough players are connected, you
should either execute the `eval_agent.py` script multiple times in
different shells or allow the server to use simulated agents. When a
client disconnects during games, they will be replaced with a randomly
acting agent.

The evaluation statistics are currently not communicated to the
clients, so either log them on the client or check the server output
for more information.

### Configuration

In the `train.py` script, you will find several configuration options
and dictionaries such as `stop_config`, `model_config` or the main
`config`. These are used to configure RLlib; possible options and
default values can be found at the following locations:

| Configuration               | Textual                                                                                                      | Code                                                                             |
|-----------------------------|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `env_config`                | `python -m pydoc hearts_gym.HeartsEnv.__init__`                                                              | `hearts_gym/envs/hearts_env.py`                                                  |
| `model_config`              | [RLlib Models](https://docs.ray.io/en/master/rllib-models.html#default-model-config-settings)                | `ray/rllib/models/catalog.py`                                                    |
| `config`                    | [RLlib Training](https://docs.ray.io/en/master/rllib-training.html#common-parameters)                        | `ray/rllib/agents/trainer.py`                                                    |
| Algorithm-specific `config` | [RLlib Algorithms](https://docs.ray.io/en/master/rllib-algorithms.html) (bottom of each algorithm's section) | `ray/rllib/agents/<algo>/<algo>.py` (replace `<algo>` with the algorithm's name) |
| `stop_config`               | [Tune Guide](https://docs.ray.io/en/latest/tune/user-guide.html#stopping-trials)                             | `ray/python/ray/tune/tune.py`                                                    |

### Development

As Ray takes quite some time to initialize, for a faster development
workflow, you can use [the `mypy`
typechecker](https://github.com/python/mypy). To check types for the
`train.py` script, execute the following:

```shell
mypy train.py
```

`mypy` gives several helpful hints; types not matching may be an
indicator for an issue.

This is completely optional and not required in any way to work on the
code. Whether you want to use type hints and type checking is entirely
up to your preference.

## Miscellaneous Information

### Supported Algorithms

[See the list of RLlib
algorithms.](https://docs.ray.io/en/master/rllib-algorithms.html)

You can filter algorithms via the following rules:

1. We act in a discrete action space, so we require "Discrete Actions → Yes".
2. We have a multi-agent environment, so we require "Multi-Agent →
   Yes".
3. Action masking is supported for algorithms with the "Discrete
   Actions → Yes +parametric" label.
4. [Auto-wrapping models with an
   LSTM](https://docs.ray.io/en/master/rllib-models.html#built-in-auto-lstm-and-auto-attention-wrappers)
   requires "Model Support" for "+RNN, +LSTM auto-wrapping"
5. [Auto-wrapping models with an Attention
   function](https://docs.ray.io/en/master/rllib-models.html#built-in-auto-lstm-and-auto-attention-wrappers)
   requires "Model Support" for "+LSTM auto-wrapping, +Attention".

### Notes on Action Masking

Some models require special settings or a re-implementation to support
action masking. For example, when using the DQN algorithm, the
`hiddens` configuration option is automatically set to an empty list
when setting up action masking. When you try out another algorithm
with action masking and it fails for a weird reason, you may have to
modify its settings or re-implement it alltogether with explicit
support.

### Notes on Auto-Wrapping and Action Masking

When action masking is enabled (`mask_actions = True`) together with
model auto-wrapping (whether `model['use_lstm']` or
`model['use_attention']` does not matter), you will notice the
respective auto-wrapping configuration option will be set to `False`
during training setup.

This is required for our action masking wrappers to work; however, the
`model['custom_model']` configuration option will have either `_lstm`
or `_attn` appended to it when the model is wrapped in an LSTM or
Attention function, respectively.

Remember, this behavior only occurs when action masking is enabled!

## References

- General:
	- https://docs.python.org/3/
	- https://github.com/ray-project/ray
- Hearts:
	- https://en.wikipedia.org/wiki/Black_Lady
	- https://en.wikipedia.org/wiki/Microsoft_Hearts
- Environments:
	- https://docs.ray.io/en/latest/rllib-env.html
	- https://github.com/openai/gym
	- https://docs.ray.io/en/latest/rllib-models.html
- Policies:
	- https://docs.ray.io/en/latest/rllib-concepts.html
- Models:
	- https://docs.ray.io/en/latest/rllib-models.html
