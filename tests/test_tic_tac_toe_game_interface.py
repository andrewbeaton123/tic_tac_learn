
import unittest
from unittest.mock import MagicMock
from tic_tac_learn.src.game_interfaces.tic_tac_toe_game_interface import TicTacToeGameInterface

class TestTicTacToeGameInterface(unittest.TestCase):

    def setUp(self):
        """Set up a new TicTacToeGameInterface for each test."""
        mock_config = MagicMock()
        mock_config.get_allowed_players.return_value = [1, 2]
        self.game = TicTacToeGameInterface(current_player=1, config_manager=mock_config)

    def test_initial_state(self):
        """Test that the game initializes to an empty board."""
        self.assertEqual(self.game.get_state(), (0, 0, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(self.game.current_player, 1)
        self.assertFalse(self.game.is_game_over())

    def test_reset(self):
        """Test that the reset method clears the board."""
        self.game.make_move(0)
        self.game.reset()
        self.assertEqual(self.game.get_state(), (0, 0, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(self.game.current_player, 1)

    def test_make_move(self):
        """Test making a move and its effect on the state."""
        self.game.make_move(0)
        self.assertEqual(self.game.get_state(), (1, 0, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(self.game.current_player, 2)
        
        self.game.make_move(1)
        self.assertEqual(self.game.get_state(), (1, 2, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(self.game.current_player, 1)

    def test_get_possible_actions(self):
        """Test that possible actions are correctly identified."""
        self.assertEqual(self.game.get_possible_actions(), [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.game.make_move(0)
        self.game.make_move(1)
        self.assertEqual(self.game.get_possible_actions(), [2, 3, 4, 5, 6, 7, 8])

    def test_win_condition(self):
        """Test a simple win condition by making moves."""
        # Player 1 wins with moves: 0, 1, 2
        self.game.make_move(0)  # P1
        self.game.make_move(3)  # P2
        self.game.make_move(1)  # P1
        self.game.make_move(4)  # P2
        self.game.make_move(2)  # P1 wins

        self.assertTrue(self.game.is_game_over())
        self.assertEqual(self.game.get_reward(player_id=1), 1.0)
        self.assertEqual(self.game.get_reward(player_id=2), -1.0)

    def test_draw_condition(self):
        """Test a draw condition by making moves."""
        # A known sequence of moves that results in a draw
        draw_moves = [4, 0, 2, 6, 1, 7, 3, 5, 8]
        for move in draw_moves:
            self.game.make_move(move)

        self.assertTrue(self.game.is_game_over())
        self.assertEqual(self.game.get_reward(player_id=1), 0.5)
        self.assertEqual(self.game.get_reward(player_id=2), 0.5)

    def test_incomplete_game_reward(self):
        """Test that the reward is 0 for a game that is not over."""
        self.game.make_move(0)
        self.assertFalse(self.game.is_game_over())
        self.assertEqual(self.game.get_reward(player_id=1), 0)

if __name__ == '__main__':
    unittest.main()
