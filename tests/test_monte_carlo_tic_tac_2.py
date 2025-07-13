import unittest
from unittest.mock import patch
from tic_tac_learn.monte_carlo_learning import MonteCarloAgent
from tic_tac_learn.src.control import Config_2_MC
from tic_tac_learn.src.game_interfaces.tic_tac_toe_game_interface import TicTacToeGameInterface



class TestMonteCarloTicTac2(unittest.TestCase):

    def setUp(self):
        self.config_manager = Config_2_MC()
        self.agent = MonteCarloAgent(epsilon=0.1, all_possible_states=[(0, 0, 0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1, 1, 1)], config_manager=self.config_manager)

    def test_add_training_games_total(self):
        # Adds ten training games using the method and then ensures that
        #ensures that the games have been added properly
        self.agent._add_to_training_games_total(10)
        self.assertEqual(self.agent.agent_training_games_total, 10)

        with self.assertRaises(TypeError):
            #Checks that there is a type error riased when a string is passed
            # to the add training games 
            self.agent._add_to_training_games_total("10")

        with self.assertRaises(ValueError):
            #Checks that there is a value error raised when a 
            #negative number is passed 
            self.agent._add_to_training_games_total(-10)

    def test_agent_training_games_total_setter(self):
        #sets the  total number of games in the agent  to 20
        #then asserts that the trianing games total was set
        self.agent.agent_training_games_total = 20
        self.assertEqual(self.agent.agent_training_games_total, 20)
        

        with self.assertRaises(TypeError):
            #chekcs that the training games errors when string is passed
            self.agent.agent_training_games_total = "20"

        with self.assertRaises(ValueError):
            #checks that there is an error  raised when 
            #a negative is passed to training games total 
            self.agent.agent_training_games_total = -20

    def test_check_q_value_space_exists(self):
        #runs the check q values exits method 
        #asserts that the length of the q_values space is greater than  0
        # this is part of the process in the q vallue sppace method
        self.agent.check_q_value_space_exists()
        self.assertGreater(len(self.agent.q_values), 0)

    @patch('tic_tac_learn.monte_carlo_learning.monte_carlo_tic_tac_2.TicTacToeGameInterface')
    def test_generate_q_value_space(self, MockTicTacToeGameInterface):
        # Configure the mock to return valid moves
        MockTicTacToeGameInterface.return_value.get_valid_moves.return_value = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        self.agent.generate_q_value_space()
        self.assertGreater(len(self.agent.q_values), 0)
        self.assertIn((0, 0, 0, 0, 0, 0, 0, 0, 0), self.agent.q_values)
        self.assertIn(0, self.agent.q_values[(0, 0, 0, 0, 0, 0, 0, 0, 0)].keys())

    def test_load_q_values(self):
        #sets the q values for the empty board for 2 of the moves
        #asserst that the q values are set corectly
        q_values = {tuple([0]*9): {0: 1, 1: 2}}
        self.agent.load_q_values(q_values)
        self.assertEqual(self.agent.q_values, q_values)

    @patch('tic_tac_learn.monte_carlo_learning.monte_carlo_tic_tac_2.TicTacToeGameInterface')
    def test_generate_returns_space_only(self, MockTicTacToeGameInterface):
        # Configure the mock to return valid moves
        MockTicTacToeGameInterface.return_value.get_valid_moves.return_value = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        self.agent.generate_returns_space_only()
        self.assertIn((tuple([0]*9), 0), self.agent.returns)

    def test_check_q_values(self):
        #creates q values structure and sets the blank board 
        #triggers q check whihc looks for 0 or none
        self.agent.q_values = {'000000000': {0: 0, 1: None}}
        self.assertTrue(self.agent.check_q_values())

        self.agent.q_values = {'000000000': {0: 1}}
        self.assertFalse(self.agent.check_q_values())

    def test_get_state(self):
        class MockEnv:
            def get_state(self):
                return (0, 0, 0, 0, 0, 0, 0, 0, 0)

        env = MockEnv()
        state = self.agent.get_state(env)
        self.assertEqual(state, (0, 0, 0, 0, 0, 0, 0, 0, 0))

    def test_calculate_reward(self):
        class MockEnv:
            def is_game_over(self):
                return True
            def get_winner(self):
                return 1

        env = MockEnv()
        reward = self.agent.calculate_reward(env)
        self.assertEqual(reward, 1)

    def test_associate_reward_with_game_state(self):
        # sets up  state action  reward for 2  games and sets them into the q_values  
        # asserts that sum of  the two rewards are assigned to the given board state
        state_action_reward = [('000000000', 0, 1), ('000000000', 0, 2)]
        result = self.agent.associate_reward_with_game_state(state_action_reward)
        self.assertEqual(result, [('000000000', 0, 3), ('000000000', 0, 2)])

    def test_update(self):
        #Test that the q values can be properly updated
        self.agent.q_values = {'000000000': {0: 0}}
        self.agent.returns = {('000000000', 0): []}
        reward_game_states = [('000000000', 0, 1)]
        self.agent.update(reward_game_states)
        self.assertEqual(self.agent.q_values['000000000'][0], 1)

        

if __name__ == '__main__':
    unittest.main()


