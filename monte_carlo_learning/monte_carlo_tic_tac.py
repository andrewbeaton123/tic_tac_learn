import numpy as np 
import logging
import random
from game.game_2 import TicTacToe
from multi_processing_tools.multi_process_controller import multi_process_controller
class MonteCarloAgent:
    def __init__(self, epsilon, all_possible_states):
        self.epsilon = epsilon
        self.q_values = {}
        self.returns = {}
        self.all_possible_states = all_possible_states
        self.standard_q_length = 9

    def to_serializable(self):
        """
        Converts the MonteCarloAgent object to a serializable dictionary.

        Returns:
        - dict: A serializable dictionary representation of the MonteCarloAgent object.
        """
        return {
            'epsilon': self.epsilon,
            'q_values': {str(k): v for k, v in self.q_values.items()},
            'returns': self.returns,
            'all_possible_states': self.all_possible_states,
        }
    
    
    def __check_q_values(self) -> bool:
        
        # Check if all q_values are either 0 or None
        
         for key, value in self.q_values.items():
            if not np.all(np.logical_or(value == 0, value is None)):
                return False
            return True
       
            
    def initialize_q_values(self):
        """
        Initializes the Q-values for all possible states.

        This method iterates over all possible states and initializes the Q-values for each state.

        Returns:
        - None
        """
        for state in self.all_possible_states:
            state_str = tuple(state.board.flatten())
            valid_moves = state.get_valid_moves()
            self.q_values[state_str] = np.zeros(len(valid_moves)).tolist()
        
        #self.check_q_value_lengths("initialize_q_values")
            
    def epsilon_greedy_policy(self, state):
        """
        Trains a single episode of the Monte Carlo agent.

        This method plays a single episode of the game using the epsilon-greedy policy for action selection.
        It updates the Q-values and returns for each state-action pair encountered during the episode.

        Parameters:
        - env: The game environment.

        Returns:
        - tuple: A tuple containing the updated Q-values and returns.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(state.get_valid_moves()))
        state_str = tuple(state.board.flatten())
        if state_str not in self.q_values:
            self.q_values[state_str] = np.zeros(len(state.get_valid_moves()))
        return np.argmax(self.q_values[state_str])


    def train_episode(self, env):
        
        episode_data = []
        while not env.is_game_over():
            if env.current_player == 1:
                action = self.epsilon_greedy_policy(env)
            else:
                action = np.random.choice(len(env.get_valid_moves()))
            
            env.make_move(*env.get_valid_moves()[action])
            reward = -1 if env.is_game_over() and env.winner != 1 else 0
            episode_data.append((env, action, reward))
        episode_data.pop()
        G = 0
        for i in reversed(range(len(episode_data))):
            state, action, reward = episode_data[i]
            
            state_str = tuple(state.board.flatten())
            G = reward + self.epsilon * G
            if not any(x[0].board.flatten().tolist() == state.board.flatten().tolist() and x[1] == action for x in episode_data[0:i]):
                if state_str not in self.returns:
                    self.returns[state_str] = {}
                if action not in self.returns[state_str]:
                    self.returns[state_str][action] = []
                self.returns[state_str][action].append(G)
                
                
                if state_str not in self.q_values or  not isinstance(self.q_values[state_str],np.ndarray):
                    self.q_values[state_str] = np.zeros( len(env.get_valid_moves()))
                elif len(self.q_values[state_str]) < len(env.get_valid_moves()):
                    self.q_values[state_str] += np.zeros((len(env.get_valid_moves()) - len(self.q_values[state_str])))
                
                

                if len(env.get_valid_moves()) > 0:
                    
                    try:
                        self.q_values[state_str][action] = np.mean(self.returns[state_str][action])
                    except IndexError:
                        logging.error(type(self.q_values[state_str][0]))
                        logging.error(type(np.mean(self.returns[state_str][action])))
                        
                        #logging.error(np.mean(self.returns[state_str][action]))
                        logging.error(f" Error in the meaning of values action,{action} and {state_str} with{self.returns[state_str]} ")
                        logging.error(f" There are {len(env.get_valid_moves())} Valid moves")
                        logging.error(f"Values for this action are currently {self.q_values[state_str][0]}")
                    

    def train(self, episodes):
        
        for episode in range(episodes):
            logging.debug(f"monte carlo tick tac train episode starting - {episode+1}")
            self.train_episode(TicTacToe(random.choice([1,2])))
            logging.debug(f"monte carlo tick tac train episode ended -  {episode+1}")
        if self.__check_q_values():
            logging.info(f"In training of monte carlo models - Q values are all 0 or nan")
            logging.info((next(iter(self.q_values.items()))[1]))
        #self.check_q_value_lengths("train")
        # Test your training logic...
    
    # Testing the agents against a random oponent 
    def play_x_test_games(self,
                          num_games : int) -> (int, int):
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
                    action = self.epsilon_greedy_policy(env)
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
        
        