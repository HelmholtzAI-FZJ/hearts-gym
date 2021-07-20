<link rel="stylesheet" href="style.css">

# Rule-based Policies

This document goes more in-depth into implementing rule-based agents
and what to look out for when doing so.

## Implementing the Default Rule-based Policy

As no pre-existing rule-based agent is provided, the default one has
to be implemented in the file
`hearts_gym/policies/rule_based_policy_impl.py` before it is able to
be used. Simply implement the `compute_action` method to return a
single action and the agent will work. It is available under the
policy ID `RULEBASED_POLICY_ID` in `configuration.py`. This is also
the rule-based policy referred to by any `policy_mapping_fn` with
`rulebased` in the name.

### Observed Games

Any rule-based policy has access to an `ObservedGame` under the `game`
member variable. The observed game provides several utility functions that
may be useful to implement the policy.

For each observation, the observed game recreates the game state as
viewed by the observing player (the agent being implemented). Due to
only having limited knowledge of the game and working with the
provided observations, some variables have to be treated with care.
For example, the specially labeled variables `offset_collected` and
`offset_penalties` are not ordered by player indices but instead by
index offsets ([see index offsets in
`docs/environment.md`](./environment.md#index-offsets) for an
explanation of these). The cards on the table, available under
`table_cards`, are simply ordered by time of placement.

Other differences between a standard `HeartsGame` and an
`ObservedGame` include different and fewer variables and
functionalities. For example, as an observed game only has access to
the information one player has, there is only one `hand` but a list of
`unknown_cards` whose location is not known to the observing player.

## Implementing Other Rule-based Policies

Any number of rule-based agents may exist in parallel; simply create a
new deterministic policy implementation by following these steps:

1. Subclass
   `hearts_gym.policies.deterministic_policy_impl.DeterministicPolicyImpl`.
2. Implement a `compute_action` method with the same signature as
   given by the superclass.
3. Add the newly implemented class to the `custom_rulebased_policies`
   dictionary in `configuration.py`, for example like this:

   ```python
   # Note we are adding the class, not an instance of it.
   custom_rulebased_policies: Dict[str, type] = {
       'my_new_policy': MyNewPolicyImpl,
   }
   ```
4. Create a new `policy_mapping_fn` that includes the new policy ID
   by mapping a player index (the agent ID) to it.
