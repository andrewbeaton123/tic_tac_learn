import unittest
import numpy as np
from tic_tac_learn.monte_carlo_learning import MonteCarloAgent



class TestMonteCarloTicTac2(unittest.TestCase):

    def setUp(self):
        #creates an agent that only has the possible states of filled or unfilled
        #The setup method is run first by unittest
        self.agent = MonteCarloAgent(epsilon=0.1, all_possible_states=['000000000', '111111111'])

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

    def test_generate_q_value_space(self):
        #runs the generate q value space, checks that the length is greater than 0
        # checks that  the  blank board is part of the space
        # checks that the q value for the empty board space is set to 0 
        self.agent.generate_q_value_space()
        self.assertGreater(len(self.agent.q_values), 0)
        self.assertIn('000000000', self.agent.q_values)
        self.assertIn(0, self.agent.q_values['000000000'])

    def test_load_q_values(self):
        #sets the q values for the empty board for 2 of the moves
        #asserst that the q values are set corectly
        q_values = {'000000000': {0: 1, 1: 2}}
        self.agent.load_q_values(q_values)
        self.assertEqual(self.agent.q_values, q_values)

    def test_generate_returns_space_only(self):
        # calls the generator for the reward space
        self.agent.generate_returns_space_only()
        #ensures that the blank board space key and associated 
        self.assertIn(('000000000', 0), self.agent.returns)

    def test_check_q_values(self):
        #creates q values structure and sets the blank board 
        #triggers q check whihc looks for 0 or none
        self.agent.q_values = {'000000000':  None}
        self.assertTrue(self.agent.check_q_values())

        # sets the blank board to 1 and then triggers check q values 
        #which should be 0 or None
        self.agent.q_values = {'000000000': {0: 1}}
        #self.assertFalse(self.agent.check_q_values())

    def test_get_state(self):
        class MockEnv:
            board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        env = MockEnv()
        state = self.agent.get_state(env)
        self.assertEqual(state.tolist(), [0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_calculate_reward(self):
        #creates  a mock class that declares the game over 
        #and assigns the winner to 1 
        # then calculates the reward and asserts that thee 
        # reward result is 1 
        class MockEnv:
            def is_game_over(self):
                return True
            winner = 1

        env = MockEnv()
        reward = self.agent.calculate_reward(env)
        self.assertEqual(reward, 1)

    def test_associate_reward_with_game_state(self):
        # sets up  state action  reward for 2  games and sets them into the q_values  
        # asserts that sum of  the two rewards are assigned to the given board state
        state_action_reward = [('000000000', 0, 1), ('000000000', 0, 2)]
        result = self.agent.associate_reward_with_game_state(state_action_reward)
        self.assertEqual(result, ('000000000', 0, 3))

    def test_update(self):
        #Test that the q values can be properly updated
        self.agent.q_values = {'000000000': {0: 0}}
        self.agent.returns = {('000000000', 0): []}
        reward_game_states = [('000000000', 0, 1)]
        self.agent.update(reward_game_states)
        self.assertEqual(self.agent.q_values['000000000'][0], 1)

if __name__ == '__main__':
    unittest.main()


