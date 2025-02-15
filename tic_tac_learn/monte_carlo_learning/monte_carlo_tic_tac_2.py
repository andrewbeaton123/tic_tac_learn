import numpy as np
import logging
import random
import mlflow.pyfunc
import pandas as pd 


from typing import Dict
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
            board = np.reshape(list(state),(3,3))
            env = TicTacToe(1,board)
            for action in range(len(env.get_valid_moves())):  # 9 possible actions in a Tic Tac Toe game
                self.q_values[state][action] = 0
                self.returns[(state, action)] = []

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
            board = np.reshape(list(state), (3,3))
            env = TicTacToe(1,board)
            for aciton in range(len(env.get_valid_moves())):
                self.returns[(state,aciton)] = []

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
        return env.board.reshape(-1)
    

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

    def predict (self, context, model_input): 
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

        return pd.DataFrame(predictions, columns=["predicted_action"])
        
            
    def take_turn(self, env: TicTacToe) -> tuple[TicTacToe,int] :  
        
        # take turn is something that happens inside of learn
        action = self.get_action(env) 
        logging.debug(f" valid moves are {env.get_valid_moves()}")
        logging.debug(f" move is {action}")
        env.make_move(*env.get_valid_moves()[action])  
        return (env, action)
    
    def get_action(self, env):
        """
        Selects an action based on the current state and epsilon-greedy exploration.

        Args:
            env (TicTacToe): The TicTacToe environment.

        Returns:
            int: The selected action.
        """
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

    def update(self,reward_game_states: list[tuple]):
        """
        Updates the Q-values based on the given state, action, and reward.

        Args:
            
            state (tuple): The current state of the environment.
            action (int): The selected action.
            reward (float): The reward received.
        """
        for state,action , total_reward  in reward_game_states:
            self.returns[(state, action)].append(total_reward)
            self.q_values[state][action] = np.mean(self.returns[(state, action)])  

    def associate_reward_with_game_state(self,state_action_reward:list[tuple]) -> list[tuple] :
        """gets the total reward for a game state and stores this in a list of tuples

        Args:
            state_action_reward (list[tuple]): un summed rewards 

        Returns:
            list[tuple]: Correctly summed rewards based on the satse and action taken 
        """
        reward_game_states =  []
        for state, action, _ in state_action_reward:
            first_idx = next(i for i, (st, ac, _) in enumerate(state_action_reward) if st == state and ac == action)
            #total reward
            G = sum(reward for _, _, reward in state_action_reward[first_idx:])
        #moved this indent in to here from inside the for loop
        #I think this  is correct but I am not  100% sure
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
                return 0  # Draw
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
            ,cores:int=2) -> (int,int):
        
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
