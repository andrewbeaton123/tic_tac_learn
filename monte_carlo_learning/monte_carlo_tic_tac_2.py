import numpy as np
import copy
import pickle
from game.game_2 import TicTacToe
import random
from multi_processing_tools.multi_process_controller import multi_process_controller
import logging

class MonteCarloAgent:
    def __init__(self, epsilon, all_possible_states):
        self.epsilon = epsilon
        self.q_values = {}
        self.returns = {}
        self.all_possible_states = all_possible_states
        
        for state in self.all_possible_states:
            self.q_values[state] = {}
            board = np.reshape(list(state),(3,3))
            env = TicTacToe(1,board)
            for action in range(len(env.get_valid_moves())):  # 9 possible actions in a Tic Tac Toe game
                self.q_values[state][action] = 0
                self.returns[(state, action)] = []

    def load_q_values(self, q_values):
        self.q_values = q_values.copy()
    def check_q_values(self) -> bool:
        
        # Check if all q_values are either 0 or None
        
         for key, value in self.q_values.items():
            if not np.all(np.logical_or(value == 0, value is None)):
                return False
            return True
    def get_state(self, env):
        return env.board.reshape(-1)
    def train(self, episodes):
        
        
            
        self.learn(TicTacToe(1),episodes) ## Old Random implementationrandom.choice([1,2])
        if self.check_q_values():
            logging.info(f"In training of monte carlo models - Q values are all 0 or nan")
            logging.info((next(iter(self.q_values.items()))[1]))
    
    def get_action(self, env):
        state = tuple(self.get_state(env))
        if env.current_player == 1:
            if np.random.rand() < self.epsilon:
                return np.random.choice(list(self.q_values[state].keys()))
            else:
                logging.debug("get action - selecting move from q values")
                logging.debug(f"get action - {self.q_values[state]}")
                logging.debug("get action -  and  q states ")
                return max(self.q_values[state], key=self.q_values[state].get)
        else:
                return np.random.choice(len(env.get_valid_moves()))

    def update(self, env, state, action, reward):
        self.returns[(state, action)].append(reward)
        self.q_values[state][action] = np.mean(self.returns[(state, action)])

    def learn(self, env, num_episodes):
        for _ in range(num_episodes):
            env.reset()
            state_action_reward = []
            while not env.is_game_over():
                action = self.get_action(env) 
                old_state = tuple(self.get_state(env))
                logging.debug(f" valid moves are {env.get_valid_moves()}")
                logging.debug(f" move is {action}")
                env.make_move(*env.get_valid_moves()[action])
                reward =self.calculate_reward(env)
                #This is old version of above
                #-1 if env.is_game_over() and env.winner != 1 else 0
                state_action_reward.append((old_state, action, reward))
            
            for state, action, _ in state_action_reward:
                first_idx = next(i for i, (st, ac, _) in enumerate(state_action_reward) if st == state and ac == action)
                G = sum(reward for _, _, reward in state_action_reward[first_idx:])
                self.update(env, state, action, G)

    def serialize(self, filename="montecarlo.pkl"):
        with open(filename, "wb") as file:
            pickle.dump(self, file)


    def calculate_reward(self, env):
        if env.is_game_over():
            if env.winner == 1:
                return 1  # Player 1 wins
            elif env.winner == 2:
                return -1  # Player 2 wins
            else:
                return 0  # Draw
        return 0  # Continue playing, no immediate reward
    

    @classmethod
    def deserialize(cls, filename="montecarlo.pkl"):
        with open(filename, "rb") as file:
            return pickle.load(file)
    
    # Testing the agents against a random oponent 
    def play_x_test_games(self,
                          num_games : int) -> tuple[int,int]:
        """
        Function to play 'x_games' number of Tic Tac Toe games and count the number of wins and draws.

        Args:
            num_games  (int): The number of games to be played.

        Returns:
            (int, int): A tuple containing the number of wins and draws respectively.

        """

        wins = 0
        draws = 0

        # Loop to play 'x_games' number of games
        for _ in range(num_games ):

            # Create a TicTacToe environment with a random player
            env = TicTacToe(random.choice([1, 2]))

            # Loop until the game is over
            while not env.is_game_over():

                # Decide the action based on the current player
                if env.current_player == 1:
                    action = self.get_action(env)
                else:
                    action = np.random.choice(len(env.get_valid_moves()))

                # Make the selected move
                env.make_move(*env.get_valid_moves()[action])

            # Increment the wins and draws count based on the game outcome
            if env.winner == 1:
                wins += 1
            elif env.winner == 0:
                draws += 1

        # Return the final count of wins and draws
        return wins, draws
    


    def test(self
            ,num_tests:int =10000
            ,cores:int=4) -> (int,int):
        
        """Plays num_tests games of tic tac toe using the monte_carlo
        agent gready process against a random oponent
        

        Args:

            agent (_type_, optional): The monte carlo based agent to play vs the 
            random oponent 
            num_tests int = The number of games to be played Defaults to 10000, 
            cores:int=4 = The number of cores to split the games  across

        Returns:
            Tuple (int, int): total wins by the agent, total draws 
        """
        
            #print(f"Agent won {wins} out of 10,000 games.")
            #print(f"Draws: {draws}")
        test_count_pc= int(num_tests/cores)
        logging.debug("testing_agent - starting test for combined q's")
        test_config = [(test_count_pc) for _ in range(cores) ]
        total_wins, total_draws= 0,0
        res_test_games = multi_process_controller(self.play_x_test_games,test_config,cores)
        for multi_return_single in res_test_games:
            wins_run, draw_run= multi_return_single
            #logging.info(wins_run,draw_run)
            total_wins += wins_run
            total_draws += draw_run
        return total_wins,total_draws
        