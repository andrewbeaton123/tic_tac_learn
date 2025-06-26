import numpy as np
import logging
import random
import mlflow.pyfunc
import pandas as pd 


from typing import Dict, List
from tic_tac_toe_game.game import TicTacToe
from tic_tac_learn.src import errors
from tic_tac_learn.multi_processing_tools.multi_process_controller import multi_process_controller

class MonteCarloAgent(mlflow.pyfunc.PythonModel):

    def __init__(self, epsilon : float, all_possible_states: list):
        self.epsilon : float = epsilon
        self.q_values : Dict = {}
        self.returns : Dict = {}
        self.all_possible_states : list = all_possible_states

        # This is to server as a training game count between different training
        #runs for the same agent
        self._agent_training_games_total : int = 0
    
    def convert_valid_move_to_board_index(self, 
                                          move_tuple: tuple) -> list : 
        """
        Converts a tuple of valid move coordinates to their corresponding board indices.
        Each move is represented as a (row, column) tuple, and the board is assumed to be a 3x3 grid.
        The board index is calculated as (row * 3 + column) for each move.
        Args:
            move_tuple (tuple): A tuple containing (row, column) pairs representing valid moves.
        Returns:
            list: A list of integers representing the corresponding board indices for each move.
        """
        

        return [int(move[0]*3 + move[1]) for move in move_tuple ]
    

    def _add_to_training_games_total(self, training_games: int ) -> None: 
        """Adds a number of training games to the total count.

        This method updates the internal counter of total training games played by the agent.

        Args:
            training_games: The number of training games to add.

        Raises:
            TypeError: If training_games is not an integer.
            ValueError: If training_games is negative.

        """
        if not isinstance(training_games, int):
            raise TypeError("training_games must be an integer.")
        if training_games < 0:
            raise ValueError("training_games cannot be negative.")
        self._agent_training_games_total += training_games
     
    @property
    def agent_training_games_total(self) -> int : 
        """Gets the total number of training games played by the agent.

        Returns:
            The total number of training games as an integer.
        """
        return self._agent_training_games_total

    @agent_training_games_total.setter
    def agent_training_games_total(self, games_previously_played: int ) -> None: 
        """Sets the total number of training games played by the agent.

        This setter allows for directly setting the total number of games, for instance,
        when loading a previously saved state.

        Args:
            games_previously_played: The total number of games played to set.

        Raises:
            TypeError: If games_previously_played is not an integer.
            ValueError: If games_previously_played is negative.
        """
        if not isinstance(games_previously_played, int):
            raise TypeError("games_previously_played must be an integer.")
        
        if games_previously_played < 0:
            raise ValueError("games_previously_played cannot be negative.")
        
        self._agent_training_games_total = games_previously_played


    def check_q_value_space_exists(self) -> None :
        """Checks if the q_values dict is empty, if it is 
        it populates the dict
        """
        logging.debug(f"The q vlaue dict length on checkigng was {len(self.q_values)}")

        if len(self.q_values) == 0:

            self.generate_q_value_space()


    def generate_q_value_space(self) -> None:
        """
        Initializes a Q-value space by iterating over all possible game states,
        initializing empty dictionaries to store Q-values and returns for each state.

        For each state, a Tic Tac Toe environment is created, and Q-values are initialized
        for all 9 possible actions (i.e., moves).

        Args:
            None

        Returns:
            None

        Side Effects:
            Populates self.q_values with empty dictionaries for each state,
                and initializes self.returns as an empty list for each (state, action) pair.
        """

        for state in self.all_possible_states:
            self.q_values[state] = {}
            board = np.reshape(list(state), (3, 3))
            env = TicTacToe(1, board)
            valid_moves = env.get_valid_moves()
            for move in valid_moves:
                action_index = int(move[0]*3 + move[1])  # ensure Python int
                self.q_values[state][action_index] = 0
                self.returns[(state, action_index)] = []

    def load_q_values(self, q_values):
        """
        Loads pre-trained Q-values into the agent.

        Args:
            q_values (dict): A dictionary containing the pre-trained Q-values.
        """
        self.q_values = q_values.copy()
        
        
    def generate_returns_space_only(self) -> None:
        """
        Creates the returns space for all possible moves. 
        This should be used when loading in q states to 
        ensure  that the returns space is also available
        """
        for state in self.all_possible_states:
            board = np.reshape(list(state), (3, 3))
            env = TicTacToe(1, board)
            valid_moves = env.get_valid_moves()
            for move in valid_moves:
                action_index = move[0]*3 + move[1]
                self.returns[(state, action_index)] = []

    def check_q_values(self) -> bool:
        """
        Checks if all Q-values are either 0 or None.

        Returns:
            bool: True if all Q-values are 0 or None, False otherwise.
        """
        # Check if all q_values are either 0 or None
        
        for key, value in self.q_values.items():

            if not np.all([np.logical_or(x ==0 , x is None )  for x in value]):
                return False
            return True
    

   
    def get_state(self, env):
        """
        Gets the current state of the environment.

        Args:
            env (TicTacToe): The TicTacToe environment.

        Returns:
            tuple: A tuple representing the current state of the environment.
        """
        return tuple(int(x) for x in env.board.reshape(-1))
    

    def train(self, episodes, starting_player: int = 1):
        """
        Trains the agent for a given number of episodes.

        Args:
            episodes (int): The number of episodes to train for.
        """
        if starting_player >=1 and starting_player<=2:
            self.learn(TicTacToe(starting_player),episodes) ## Old Random implementationrandom.choice([1,2])
        else:
            raise errors.OutOfBoundsPlayerChoice()
    
    def check_q_values_not_nan(self) -> None :

        if self.check_q_values():
            logging.info(f"In training of monte carlo models - Q values are all 0 or nan")
            logging.info((next(iter(self.q_values.items()))[1]))

    
    def load_context(self, context):
        """
        Load model artifacts, such as pre-trained Q-values, if saved in the MLflow model.
        """
        q_values_path = context.artifacts.get("q_values")
        if q_values_path:
            self.q_values = pd.read_pickle(q_values_path)

    def predict (self, context, model_input) -> List: 
        """
        Generate predictions for input data.

        Args:
            context: MLflow context (unused here).
            model_input: A pandas DataFrame containing the game state as input.

        Returns:
            A pandas DataFrame with the predicted actions.
        """
        # Convert input DataFrame into the required format

        # Ensure model_input is a pandas DataFrame
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("model_input must be a pandas DataFrame")
        
        game_states = model_input.to_numpy()
        predictions = []

        for state_array in game_states:
            state = tuple(state_array)
            env = TicTacToe(1, np.reshape(state, (3, 3)))

            if not env.is_game_over():
                action = self.get_action(env)
                predictions.append(action)
            else:

                predictions.append(None)  # Invalid move as the game is over

        return predictions
        
            
    def take_turn(self, env: TicTacToe) -> tuple[TicTacToe,int] :  
        
        # take turn is something that happens inside of learn
        action = self.get_action(env) 
        valid_moves = env.get_valid_moves()
        valid_boad_indices = [int(idx) for idx in self.convert_valid_move_to_board_index(valid_moves)]
        action = int(action)
        move_index = valid_boad_indices.index(action)
        logging.debug(f" valid moves are {env.get_valid_moves()}")
        logging.debug(f" move is {action}")
        env.make_move(*valid_moves[move_index])
        return env, action
    
    def get_action(self, env):
        """
        Selects an action based on the current state and epsilon-greedy exploration.

        Args:
            env (TicTacToe): The TicTacToe environment.

        Returns:
            int: The selected action.
        """
        state = tuple(self.get_state(env))
        valid_moves = env.get_valid_moves()
        valid_indices = [int(move[0]*3 + move[1]) for move in valid_moves]  # ensure Python int

        if env.current_player == 1:
            if np.random.rand() < self.epsilon:
                action = np.random.choice(valid_indices)
            else:
                # Only consider Q-values for valid actions
                q_vals = {a: self.q_values[state][a] for a in valid_indices}
                action = max(q_vals, key=q_vals.get)
                
            return action  # index in valid_moves
        else:
            return np.random.choice(valid_indices)

    def update(self,reward_game_states: list[tuple]):
        """
        Updates the Q-values based on the given state, action, and reward.

        Args:
            
            state (tuple): The current state of the environment.
            action (int): The selected action.
            reward (float): The reward received.
        """
        for state, action, total_reward in reward_game_states:
            action = int(action)
            self.returns[(state, action)].append(total_reward)
            self.q_values[state][action] = np.mean(self.returns[(state, action)])  

    def associate_reward_with_game_state(self, state_action_reward: list[tuple]) -> list[tuple]:
        reward_game_states = []
        for idx, (state, action, _) in enumerate(state_action_reward):
            G = sum(reward for _, _, reward in state_action_reward[idx:])
            reward_game_states.append((state, action, G))
        return reward_game_states
            
    def calculate_reward(self, env):
        """
        Calculates the reward based on the current game state.

        Args:
            env (TicTacToe): The TicTacToe environment.

        Returns:
            float: The reward.
        """
        if env.is_game_over():
            if env.winner == 1:
                return 1  # Player 1 wins
            elif env.winner == 2:
                return -1  # Player 2 wins
            else:
                return -0.1  # Draw
        return 0  # Continue playing, no immediate reward
    

    def learn(self, env, num_episodes):
        """
        Learns to play Tic Tac Toe through Monte Carlo tree search.

        Args:
            env (TicTacToe): The TicTacToe environment.
            num_episodes (int): The number of episodes to learn for.
        """
        for _ in range(num_episodes):
            env.reset()
            state_action_reward = []
            
            while not env.is_game_over():
                old_state = tuple(self.get_state(env))

                env, action  = self.take_turn(env)
                reward =self.calculate_reward(env)

                #This is old version of above
                #-1 if env.is_game_over() and env.winner != 1 else 0
                state_action_reward.append((old_state, action, reward))

            summed_rewards_with_states = self.associate_reward_with_game_state(state_action_reward)
            self.update(summed_rewards_with_states)
        self._add_to_training_games_total(num_episodes)
        
    def test(self
            ,num_tests:int =10000
            ,cores:int=2) -> tuple[int,int]:
        
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
    
    
    def play_x_test_games(self, num_games: int) -> tuple[int, int]:
        wins = 0
        draws = 0

        for _ in range(num_games):
            env = TicTacToe(random.choice([1, 2]))
            while not env.is_game_over():
                if env.current_player == 1:
                    action = self.get_action(env)
                    valid_moves = env.get_valid_moves()
                    valid_boad_indices = [int(idx) for idx in self.convert_valid_move_to_board_index(valid_moves)]
                    action = int(action)
                    move_index = valid_boad_indices.index(action)
                    env.make_move(*valid_moves[move_index])
                else:
                    valid_moves = env.get_valid_moves()
                    move = random.choice(valid_moves)
                    env.make_move(*move)

            if env.winner == 1:
                wins += 1
            elif env.winner == 0:
                draws += 1

        return wins, draws
