<link rel="stylesheet" href="style.css">

# Hearts Environment

This document describes the Hearts environment in detail.

## Observations

Observations are "normalized", or position-independent. This means
that encountering the same game state at different indices will always
lead to the same observation. While the observations are perfect in
the memory sense, they do not keep track of the game's history.

The default observations the environment returns when action masking
is enabled are of the following form:

```python
{
    HeartsEnv.OBS_KEY: {
        'cards': <vector of integer card states>,
        'leading_hearts_allowed': <boolean>,
    },
    HeartsEnv.ACTION_MASK_KEY: <vector of booleans>,
}
```

When action masking is not enabled, the environment only returns the
observations under the `HeartsEnv.OBS_KEY`. Now, the individual
observations are described in more detail.

### Cards

Each entry in this vector indicates the state for a particular card,
where states are integer values described in more detail below. The
vector is of the same size as the deck. With the amount of players,
the amount of states, or the range of allowed values, changes. See
`hearts_env.envs.HeartsGame.card_to_index` for the order of the cards
in the vector. While the positions of the cards do not change, their
state does.

As explained above, we do not track the game's history: only the
current position of each card is known, meaning we do not observe past
states, such as which player had a certain card on hand or when it was
played.

#### Card States

| Name                                             | Amount                  | Description                                                     |
|--------------------------------------------------|-------------------------|-----------------------------------------------------------------|
| `HeartsEnv.STATE_UNKNOWN`                        | 1                       | Has not been seen.                                              |
| `HeartsEnv.STATE_ON_HAND`                        | 1                       | Is on the player's hand.                                        |
| `HeartsEnv.on_table_state(player_index_offset)`  | `HeartsEnv.num_players` | Was put on the table by the player with the given index offset. |
| `HeartsEnv.collected_state(player_index_offset)` | `HeartsEnv.num_players` | Was collected by the player with the given index offset.        |

#### Index Offsets

Index offsets offer position-independent location information. They
are positive values in the interval `[0, num_players)`. Index offsets
answer the question "how many clockwise steps do I need to take from
my position to reach a player with a given index?". For the following
example, keep in mind that player indices (which are different from
index offsets) start at 0, so index 3 is the largest in a game with 4
players.

In a game with 4 players, the index offset from the player at index 3
to the player at index 0 is 1. Index offsets cannot be negative and
simply assume that player indices wrap around. The following table
lists all indices you reach given an index offsets for the player at
index 3 in a 4-player game.

| Index Offset | Index |
|--------------|-------|
| 0            | 3     |
| 1            | 0     |
| 2            | 1     |
| 3            | 2     |

The formula for getting an index offset given a starting position (3
in the example above) and target position (0 in the example above) in
code is:

```python
def index_offset(start_index, target_index, num_players):
    # Python's modulo implementation will always return a positive
    # number here.
    return (target_index - start_index) % num_players

    # Alternative, implementation-independent formulation with only
    # positive values:
    # return (num_players + target_index - start_index) % num_players
```

The formula for getting a target position given a starting position
and an index offset can easily be derived:

```python
def target_position(start_index, index_offset, num_players):
    return (start_index + index_offset) % num_players
```

### Leading Hearts Allowed

This binary value indicates whether it is allowed to lead a trick with
a hearts card. The implemented rules only allow this once a hearts
card has been discarded due to not being able to follow suit.

### Action Mask

The action mask is a vector of binary values with the length of the
maximum amount of cards in hand, indicating whether playing the card
at that same location is a legal action. A 1 indicates legality while
a 0 indicates an illegal, or masked, action.

When the player has less than the maximum amount of cards in hand, the
upper portion of the action mask vector is obviously filled with 0.
This is because we cannot select cards to play that do not exist.

## Actions

Actions are a single integer value indicating which card to play. More
explicitly, an action is the index of the card in hand to play.

### Illegal Actions

When an illegal action is encountered, the first legal card in hand is
deterministically played. This means illegal actions are not
inherently bad from the perspective of an agent; their behaviour can
be learned just as well. However, the amount of illegal actions is
still kept track of during evaluation as they are assumed to be
undesired. An agent that achieves the same results as another but
which executes fewer illegal actions wins in terms of an arbitrary
metric.

With action masking enabled and an algorithm supporting it, the chance
of encountering an illegal action is infinitesimally small. Due to
numerical stability, completely preventing actions by assigning a
probability of zero to them is not possible. As the log probability is
modified, negative infinities would have to be inserted possibly
causing trouble.
