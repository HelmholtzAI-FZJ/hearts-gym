<link rel="stylesheet" href="style.css">

# Hearts Gym Documentation

This document serves as a quick overview on usage and features of the
Hearts Gym.

## Introduction

An multi-agent environment to train agents on playing the
[Hearts](https://en.wikipedia.org/wiki/Black_Lady) card game.

Also includes a client-server architecture to remotely evaluate local
agents.

Finally, any number of hard-coded baseline agents may be implemented
with ease.

The rules are mostly implemented as specified by the modern rules from
Morehead (2001) (ISBN: 9780451204844). For a more detailed description
of the rules and differences from the original, execute the following:

```shell
python -m pydoc hearts_gym.envs.hearts_game.HeartsGame
```

## Installing

Supported Python versions are shown in `setup.py` under the
`python_requires` argument. If your system does not have the correct
version (it will complain at some point during the installation), you
can [use the Conda installation instructions](#conda-installation).

### Environment Setup

Clone this repository so you can easily modify it, replacing
`<repo-uri>` with the URI of this repo.

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

# Do not use `python3` from this point onward!
```

### Installing Requirements

Install at least one of
[PyTorch](https://pytorch.org/get-started/locally/) or
[TensorFlow](https://www.tensorflow.org/install) as your deep learning
framework (RLlib also has experimental
[JAX](https://github.com/google/jax#installation) support if you feel
adventurous).

After installing a deep learning framework, in the root directory of
the repository clone, execute:

```shell
python -m pip install --upgrade pip
python -m pip install -e .
```

You are done! [Head over to the usage section](#usage).

### Conda Installation

Note that if you are coming from the standard installation
instructions, you **must not** anymore use the virtual environment
that was created (via `venv`). So please execute `deactivate` to
deactivate the environment and delete the `env` directory. You will
also have to re-install the requirements later on.

To install a Python version different from your system's, below you
can find instructions for
[Miniconda](https://docs.conda.io/en/latest/miniconda.html). If you
already have the `conda` command available, you do not need to install
Miniconda.

After installing Miniconda, execute the following:

```shell
conda create -n hearts-gym python=3.8 zlib
conda activate hearts-gym
```

You now have a Conda environment for development. However, you still
need to [install the requirements](#installing-requirements).

## Usage

You will need to execute one of the following lines each time you
start a new shell. This will activate the Python virtual environment
we are using.

- If you used the standard installation instructions:

  ```shell
  # On Unix:
  source ./env/bin/activate
  # On Windows:
  .\env\Scripts\activate
  ```

- If you used the Conda installation instructions:

  ```
  conda activate hearts-gym
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

Common errors at this point:

- If you encounter memory errors, the simplest solution is to set a
  lower number of worker processes (`'num_workers'` in the `config`
  dictionary). By default, all CPUs and all GPUs are used.
- If your operating system complains about a low `ulimit`, please
  execute `ulimit -n 8192` (or whatever your operating system
  recommends) after activating your environment each time.
- If you encounter GPU errors, make sure your CUDA and cuDNN versions
  match the ones expected by your deep learning framework. You may
  also set `num_gpus` in the `config` dictionary to 0 to forego these
  troubles for a small loss in speed.

If everything worked correctly, you should see a table summarizing
test results of your learned agent against other agents printed on
your terminal. If you see the table, you can ignore any other errors
displayed by Ray. The table looks something like this:

```python
[...]
# On Unix:
(pid=10101) SystemExit: 1  # Can be ignored.
# On Windows:
(pid=10101) Windows fatal exception: access violation.  # Can be ignored.
[...]
testing took 1.23456789 seconds
# illegal action (player 0): 0 / 52
# illegal action ratio (player 0): 0.0
| policy  | # rank 1 | # rank 2 | # rank 3 | # rank 4 | total penalty |
|---------+----------+----------+----------+----------+---------------|
| learned |        1 |        0 |        0 |        0 |             0 |
| random  |        0 |        1 |        0 |        0 |             5 |
| random  |        0 |        1 |        0 |        0 |             5 |
| random  |        0 |        0 |        0 |        1 |            16 |
```

The table lists the policy, the number of placements in each rank, and
the accumulated penalty over all test games for each player. As you
can see from this example with a single test game, players with the
same penalty score get the highest of their rankings.

A central role in `train.py` is played by the file `configuration.py`.
`configuration.py` contains [lots of configuration options which are
described here](#configuration). Results including configuration and
checkpoints are saved in the `results` directory by default. You can
list directories containing checkpoints with `python
show_checkpoint_dirs.py`. When you want to [share your checkpoints,
check out the corresponding section](#sharing-checkpoints). After
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
python eval_agent.py --name <name> --algorithm <algo> <checkpoint_path>
```

Replace `<name>` with a name you want to have displayed, `<algo>` with
the name of the algorithm you used for training the agent, and
`<checkpoint_path>` with the path to a checkpoint. The rest of the
configuration is loaded from the `params.pkl` file next to the
checkpoint's directory; if that file is missing, you have to configure
`configuration.py` according to the checkpoint you are loading. Here
is an example:

```shell
python eval_agent.py --name '🂭-collector' --algorithm PPO results/PPO/PPO_Hearts-v0_00000_00000_0_1970-01-01_00-00-00/checkpoint_000002/checkpoint-2
```

Since the server will wait until enough players are connected, you
should either execute the `eval_agent.py` script multiple times in
different shells or allow the server to use simulated agents. When a
client disconnects during games, they will be replaced with a randomly
acting agent.

To evaluate another policy, you do not need to supply a checkpoint.
Instead, give its policy ID using `--policy_id <policy-id>`, replacing
`<policy-id>` with the ID of the policy to evaluate.

### Configuration

In `configuration.py`, you will find several configuration options and
dictionaries such as `stop_config`, `model_config` or the main
`config`. These are used to configure RLlib; possible options and
default values can be found at the following locations:

| Configuration               | Textual                                                                                                      | Code                                                                             |
|-----------------------------|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `env_config`                | `python -m pydoc hearts_gym.HeartsEnv.__init__`                                                              | `hearts_gym/envs/hearts_env.py`                                                  |
| `model_config`              | [RLlib Models](https://docs.ray.io/en/master/rllib-models.html#default-model-config-settings)                | `ray/rllib/models/catalog.py`                                                    |
| `config`                    | [RLlib Training](https://docs.ray.io/en/master/rllib-training.html#common-parameters)                        | `ray/rllib/agents/trainer.py`                                                    |
| Algorithm-specific `config` | [RLlib Algorithms](https://docs.ray.io/en/master/rllib-algorithms.html) (bottom of each algorithm's section) | `ray/rllib/agents/<algo>/<algo>.py` (replace `<algo>` with the algorithm's name) |
| `stop_config`               | [Tune Guide](https://docs.ray.io/en/latest/tune/user-guide.html#stopping-trials)                             | `ray/python/ray/tune/tune.py`                                                    |

### Rule-based Agents

There is no pre-existing rule-based agent; the default one may be
implemented in the file
`hearts_gym/policies/rule_based_policy_impl.py` by implementing the
`compute_action` method. It is available under the policy ID
"rulebased" by default.

For more information on this topic including what to look out for and
how to implement multiple rule-based agents, refer to
[`docs/rule-based-policies.md`](./rule-based-policies.md)

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

#### Debugging

For less clutter and an easier debugging setup, set the `num_gpus` and
`num_workers` configuration values to 0.

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

### Sharing Checkpoints

To share a checkpoint, you need the whole directory containing the
checkpoint file (as listed by `python show_checkpoint_dirs.py`). In
addition, you may want to share the `params.pkl` file next to the
directory containing the checkpoint to share its configuration as
well.

### Monitoring Training with TensorBoard

RLlib automatically creates
[TensorBoard](https://www.tensorflow.org/tensorboard) summaries,
allowing you to monitor statistics of your models during (or after)
training. Start it with the following line:

```shell
tensorboard --logdir results
```

Note that usage of this is completely optional; TensorBoard is not an
installation requirement.

### Security

Be default, the `eval_agent.py` script automatically loads parameters
used for model training from a `pickle` file so the script does not
have to be re-configured for each checkpoint. If you obtain
checkpoints that include a `params.pkl` file from an untrusted source
and load them, arbitrary code may be executed.

To avoid this security issue, set `allow_pickles = False` in
`configuration.py`. Note that you then have to configure
`configuration.py` for each checkpoint you want to load in
`eval_agent.py` so the configuration matches.

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
