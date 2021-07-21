<link rel="stylesheet" href="style.css">

# Hearts Environment

This document gives some pointers for helpful variables and functions
for reward shaping.

A peculiarity in multi-agent environments is that agents are rewarded
much later than they take their action, that is, an agent receives its
reward along with the observation to take its next action. This means
that we cannot use present information but need to work with past
state. This is why below, several variables are prefixed with a `prev`
for 'previous', referring to when the agent was active last.

## Useful Variables and Functions available to `RewardFunction`

What follows is a list of useful attributes accessible to the
`RewardFunction` that will help for reward shaping. In the list,
`self` refers to the `RewardFunction` instance.

Please see the respective docstrings for a detailed description of
each of these. Docstrings for variables are triple-quoted strings
_below_ them.

- Find attributes beginning with `self.env` in the file
  `hearts_gym/envs/hearts_env.py` under `HeartsEnv`.
- Find attributes beginning with `self.game` in the file
  `hearts_gym/envs/hearts_game.py` under `HeartsGame`.

| Attribute                               | Summary                                                                |
|-----------------------------------------|------------------------------------------------------------------------|
| `self.game.num_players`                 | Number of players.                                                     |
| `self.game.deck_size`                   | Number of cards in the deck.                                           |
| `self.game.prev_hands`                  | Cards in hands; use player index for retrieval.                        |
| `self.game.prev_played_cards`           | Cards actively played; use player index for retrieval.                 |
| `self.game.prev_table_cards`            | Cards on the table.                                                    |
| `self.game.prev_collected`              | All cards collected; use player index for retrieval.                   |
| `self.game.collected`                   | All cards collected after the action; use player index for retrieval.  |
| `self.game.penalties`                   | Penalty scores; use player index for retrieval.                        |
| `self.game.prev_was_illegals`           | Wether actions were illegal; use player index for retrieval.           |
| `self.game.prev_states`                 | Card state vector; use player index for retrieval.                     |
| `self.game.prev_was_first_trick`        | Wether it is the first trick of the game.                              |
| `self.game.prev_leading_hearts_allowed` | Wether leading with hearts is allowed; use player index for retrieval. |
| `self.game.prev_leading_suit`           | Leading suit.                                                          |
| `self.game.prev_leading_player_index`   | Index of the player that lead the trick.                               |
| `self.game.prev_trick_winner_index`     | Index of the player that won the trick.                                |
| `self.game.prev_trick_penalty`          | Trick penalty.                                                         |
| `self.game.get_penalty`                 | Return the penalty score of a given card.                              |
| `self.game.has_penalty`                 | Return whether the given card has a penalty score greater than zero.   |
| `self.game.has_shot_the_moon`           | Return whether the given player shot the moon.                         |
