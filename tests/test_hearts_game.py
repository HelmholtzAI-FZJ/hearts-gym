import random
import unittest

from hearts_gym.envs.card_deck import Card
from hearts_gym.envs.hearts_game import HeartsGame


class TestCommon(unittest.TestCase):
    def test_full_trick(self):
        game = HeartsGame(seed=0)
        game.reset()
        # print(game.hands)
        results = game.full_trick([0] * 4)
        # print(results)

    def test_full_game(self):
        game = HeartsGame(seed=0)
        game.reset()
        for _ in range(13):
            results = game.full_trick([0] * 4)
            (state, winner_index, is_done, info) = results
            self.assertEqual(game.leading_player_index, winner_index)
        self.assertTrue(is_done)

    def test_card_to_index(self):
        game = HeartsGame()
        deck = game.deck
        deck._deck.sort()

        self.assertEqual(deck.size, deck.MAX_SIZE)
        self.assertEqual(len(deck), deck.size)
        for (i, card) in enumerate(deck.take(len(deck))):
            card_index = game.card_to_index(card)
            self.assertEqual(card_index, i)

    def test_states_ordered(self):
        game = HeartsGame()

        states = [game.STATE_UNKNOWN]

        for player_index in range(game.num_players):
            states.append(game.on_table_state(player_index))

        for player_index in range(game.num_players):
            states.append(game.in_hand_state(player_index))

        for player_index in range(game.num_players):
            states.append(game.collected_state(player_index))

        self.assertEqual(len(states), game.num_states)
        for (i, val) in enumerate(states):
            self.assertEqual(val, i)

    def assert_state_matches(self, game):
        state = game.state
        indices = []

        # Unknown (remaining)
        for card in game.remaining_cards:
            card_index = game.card_to_index(card)
            indices.append(card_index)
            card_state = state[card_index]

            self.assertEqual(card_state, game.STATE_UNKNOWN)

        # Table
        for (i, card) in enumerate(game.table_cards):
            player_index = (
                game.active_player_index - len(game.table_cards) + i
            ) % game.num_players

            card_index = game.card_to_index(card)
            indices.append(card_index)
            card_state = state[card_index]

            target_card_state = game.on_table_state(player_index)
            self.assertEqual(target_card_state, 1 + player_index)
            self.assertEqual(card_state, target_card_state)

        # Hands
        for (player_index, hand) in enumerate(game.hands):
            for card in hand:
                card_index = game.card_to_index(card)
                indices.append(card_index)
                card_state = state[card_index]

                target_card_state = game.in_hand_state(player_index)
                self.assertEqual(
                    target_card_state,
                    1 + game.num_players + player_index,
                )
                self.assertEqual(card_state, target_card_state)

        # Collected
        for (player_index, collection) in enumerate(game.collected):
            for card in collection:
                card_index = game.card_to_index(card)
                indices.append(card_index)
                card_state = state[card_index]

                target_card_state = game.collected_state(player_index)
                self.assertEqual(
                    target_card_state,
                    1 + game.num_players * 2 + player_index,
                )
                self.assertEqual(card_state, target_card_state)

        indices.sort()
        self.assertEqual(len(indices), len(state))
        for (i, card_index) in enumerate(indices):
            self.assertEqual(card_index, i)

    def test_states_randomly(self):
        seed = 0

        game = HeartsGame(seed=seed)
        rng = random.Random(seed + 1)

        for ep in range(10000):
            game.reset()
            self.assert_state_matches(game)
            for _ in range(13):
                leading_index = game.leading_player_index
                for i in range(leading_index, leading_index + 4):
                    if game.is_first_trick and i == leading_index:
                        continue
                    i = i % 4
                    legal_actions = game.get_legal_actions(i)
                    action = rng.choice(legal_actions)
                    game.play_card(action)
                    self.assert_state_matches(game)

            self.assertTrue(game.is_done())

    def test_legality_randomly(self):
        seed = 0

        game = HeartsGame(seed=seed)
        rng = random.Random(seed + 1)

        for ep in range(10000):
            game.reset()
            for _ in range(13):
                leading_index = game.leading_player_index
                for i in range(leading_index, leading_index + 4):
                    if game.is_first_trick and i == leading_index:
                        continue
                    i = i % 4
                    action = rng.randint(0, 13)
                    legal_actions = game.get_legal_actions(i)
                    hand = game.hands[i]
                    _, was_illegal, _, _ = game.play_card(action)

                    self.assertEqual(
                        action not in legal_actions, was_illegal,
                        f'\naction: {action}\nlegal_actions: {legal_actions}\n'
                        f'hand: {hand}\ntable: {game.table_cards}\n'
                        f'is_first_trick: {game.is_first_trick}\n'
                        f'leading_hearts_allowed: '
                        f'{game.leading_hearts_allowed}',
                    )

            self.assertTrue(game.is_done())

    def test_rankings_randomly(self):
        seed = 0

        game = HeartsGame(seed=seed)
        rng = random.Random(seed + 1)

        total_penalties = [0] * 4
        total_placements = [[0] * 4 for _ in range(4)]

        for _ in range(10000):
            game.reset()
            for _ in range(13):
                leading_index = game.leading_player_index
                for i in range(leading_index, leading_index + 4):
                    if game.is_first_trick and i == leading_index:
                        continue
                    i = i % 4
                    legal_actions = game.get_legal_actions(i)
                    action = rng.choice(legal_actions)
                    game.play_card(action)

            self.assertTrue(game.is_done())

            final_penalties = game.compute_final_penalties()
            final_rankings = game.compute_rankings()
            for (i, penalty) in enumerate(final_penalties):
                total_penalties[i] += penalty
            for (i, ranking) in enumerate(final_rankings):
                total_placements[i][ranking - 1] += 1

        print(total_penalties)
        print(total_placements)

    def test_hand_sorted(self):
        game = HeartsGame(seed=0)
        game.reset()
        hand = [
            Card(0, 2),
            Card(3, 2),
            Card(0, 3),
            Card(3, 3),
            Card(2, 4),
            Card(2, 7),
            Card(3, 4),
            Card(3, 8),
            Card(0, 10),
            Card(1, 9),
            Card(3, 10),
            Card(2, 11),
            Card(2, 12),
        ]
        sorted_hand = [
            Card(0, 2),
            Card(0, 3),
            Card(0, 10),
            Card(1, 9),
            Card(2, 4),
            Card(2, 7),
            Card(2, 11),
            Card(2, 12),
            Card(3, 2),
            Card(3, 3),
            Card(3, 4),
            Card(3, 8),
            Card(3, 10),
        ]
        hand.sort()
        self.assertEqual(len(hand), len(sorted_hand))
        for (a, b) in zip(hand, sorted_hand):
            self.assertEqual(a, b)

    def test_string(self):
        game = HeartsGame(seed=0)
        game.reset()
        print(str(game))

    def test_no_matching_suit(self):
        pass

    def test_leading_heart(self):
        pass


class TestIllegal(unittest.TestCase):
    def test_bleeding_first(self):
        game = HeartsGame(seed=0)
        game.reset()
        game.full_trick([0] * 4)
        pass

    def test_bleeding_other(self):
        pass

    def test_no_matching_suit(self):
        pass

    def test_leading_heart(self):
        pass


if __name__ == '__main__':
    unittest.main()
