# Hearts Gym

A multi-agent environment to train agents on playing the
[Hearts](https://en.wikipedia.org/wiki/Black_Lady) card game.

Also includes a client-server architecture to remotely evaluate local
agents.

This README file is just a brief introduction to getting started;
please [check out the
documentation](https://hearts-gym.readthedocs.io/en/latest/readme.html)
for more information.

## Getting Started

### Environment Setup

These minimal instructions assume you are using a Unix-based operating
system. [The
documentation](https://hearts-gym.readthedocs.io/en/latest/readme.html#environment-setup)
has instructions for other operating systems and catches more failure
cases. If you encounter problems, please check there.

Set up a Python environment:

```shell
git clone https://github.com/HelmholtzAI-FZJ/hearts-gym.git
cd hearts-gym
python3 -m venv --system-site-packages ./env
# On Unix:
source ./env/bin/activate
```

### Installing Requirements

Install at least one of
[PyTorch](https://pytorch.org/get-started/locally/) or
[TensorFlow](https://www.tensorflow.org/install) as your deep learning
framework (RLlib also has experimental
[JAX](https://github.com/google/jax#installation) support if you feel
adventurous).

For example to install TensorFlow:

```shell
python -m pip install --upgrade pip
python -m pip install --upgrade tensorflow
```

After installing a deep learning framework, in the root directory of
the repository clone, execute:

```shell
python -m pip install --upgrade pip
python -m pip install -e .
```

You are done!

## Training

```shell
source ./env/bin/activate
python train.py
```

If everything worked correctly, you should see a table summarizing
test results of your learned agent against other agents printed on
your terminal. If you see the table, you can ignore any other errors
displayed by Ray. If you don't see the table, check out [the
documentation](https://hearts-gym.readthedocs.io/en/latest/readme.html#training)
for common errors or submit an issue.

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

Afterwards, modify `configuration.py` to adapt the training to your
needs. Again, more help can be found in [the
documentation](https://hearts-gym.readthedocs.io/en/latest/readme.html#configuration).

## Evaluation

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
