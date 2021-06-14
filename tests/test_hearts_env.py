import random
import unittest

import numpy as np

from hearts_gym import HeartsEnv
from hearts_gym.envs.card_deck import Deck


class MockDeck(Deck):
    def __init__(self, deck, deck_cards):
        self.__deck = deck
        self.__deck_cards = deck_cards

    def reset(self):
        self.__deck._deck = self.__deck_cards.copy()

    def __getattr__(self, name):
        return getattr(self.__deck, name)


class TestCommon(unittest.TestCase):
    def assert_tree_equal(self, a, b, depth=0):
        self.assertEqual(type(a), type(b))

        if isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())

            for (key, aa) in a.items():
                bb = b[key]
                self.assert_tree_equal(aa, bb, depth + 1)
            return
        elif isinstance(a, np.ndarray):
            self.assertEqual(a.shape, b.shape)

            for (aa, bb) in zip(a, b):
                self.assert_tree_equal(aa, bb, depth + 1)
            return
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))

            for (aa, bb) in zip(a, b):
                self.assert_tree_equal(aa, bb, depth + 1)
            return

        self.assertEqual(a, b)

    def test_same_state_random(self):
        seed = 0

        for mask_actions in [True, False]:
            states = []
            env = HeartsEnv(seed=seed, mask_actions=mask_actions)
            game = env.game

            deck = game.deck
            deck_cards = None
            deck_cycle_size = deck.size // 4

            for player_index in range(4):
                player_states = []
                rng = random.Random(seed + 1)

                if deck_cards is None:
                    deck_cards = deck._deck
                else:
                    # print(player_index, deck_cards)
                    # print(deck_cards[:deck_cycle_size])
                    # print(deck_cards[deck_cycle_size:])
                    # print()
                    deck_cards = (
                        deck_cards[-deck_cycle_size:]
                        + deck_cards[:-deck_cycle_size]
                    )
                game.deck = MockDeck(deck, deck_cards)

                for ep in range(10):
                    env.reset()
                    obs = env._game_state_to_obs(player_index)
                    player_states.append({
                        'obs': obs,
                        'game.state': game.state.copy(),
                        'hands': [hand.copy() for hand in game.hands],
                        'collected': [coll.copy()
                                      for coll in game.collected],
                        'table': game.table_cards.copy(),
                        'deck_cards': deck_cards.copy(),
                    })

                    for _ in range(13):
                        leading_index = game.leading_player_index
                        for i in range(leading_index, leading_index + 4):
                            if game.is_first_trick and i == leading_index:
                                continue
                            i = i % 4
                            legal_actions = game.get_legal_actions(i)
                            action = rng.choice(legal_actions)
                            game.play_card(action)

                            obs = env._game_state_to_obs(player_index)
                            player_states.append({
                                'obs': obs,
                                'game.state': game.state.copy(),
                                'hands': [hand.copy() for hand in game.hands],
                                'collected': [coll.copy()
                                              for coll in game.collected],
                                'table': game.table_cards.copy(),
                                'deck_cards': deck_cards.copy(),
                            })

                states.append(player_states)

            cmp_len = len(states[0])
            for player_states in states[1:]:
                self.assertEqual(len(player_states), cmp_len)

            cmp_states = states[0]

            for (i, cmp_state) in enumerate(cmp_states):
                for (player_index, player_states) in enumerate(states[1:]):
                    player_index += 1

                    player_state = player_states[i]
                    try:
                        self.assert_tree_equal(player_state['obs'],
                                               cmp_state['obs'])
                    except AssertionError:
                        print(f'a game state (player index {player_index})',
                              player_state)
                        print('b game state (player index 0)',
                              cmp_state)
                        raise


if __name__ == '__main__':
    unittest.main()
