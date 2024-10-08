# This code is a general plug and play of the behaviour of the 
#monte carlo methods
import torch
import torch.nn as nn
import torch.nn as nn

import numpy as np
import pickle
import logging

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)  # Input layer 9, hidden layer 128 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)  # Hidden layer 64 neurons
        self.out = nn.Linear(64, 9)  # Output layer 9 (Q-values for 9 actions)

        # Initialize weights and biases
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)  

        torch.nn.init.xavier_uniform_(self.out.weight)  

        torch.nn.init.zeros_(self.out.bias)
    

    def load_q_values(self, q_values):
        self.q_values = q_values.copy()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values
    
    
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


    def get_action(self, env):
        state = torch.tensor(self.get_state(env)).float()  # Convert to tensor and float
        q_values = self.net(state)  # Get Q-values from neural network
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
            """
            Updates the Q-values based on the given state, action, and reward.

            Args:
                env (TicTacToe): The TicTacToe environment.
                state (tuple): The current state of the environment.
                action (int): The selected action.
                reward (float): The reward received.
            """

            # Calculate target Q-value
            next_state = tuple(self.get_state(env))
            if env.is_game_over():
                target_q_value = reward
            else:
                next_state_tensor = torch.tensor(next_state).float()
                next_q_values = self.net(next_state_tensor)
                target_q_value = reward + self.gamma * next_q_values.max().item()

            # Update Q-values
            state_tensor = torch.tensor(state).float()
            q_values = self.net(state_tensor)
            q_values[action] = target_q_value

            # Calculate loss and update network
            loss = nn.MSELoss()(q_values.unsqueeze(0), target_q_value.unsqueeze(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def serialize(self, filename="montecarlo.pkl"):
        """
        Serializes the agent to a file.

        Args:
            filename (str, optional): The filename to save the agent to. Defaults to "montecarlo.pkl".
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)


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
      

    @classmethod
    def deserialize(cls, filename="montecarlo.pkl"):
        """
        Deserializes an agent from a file.

        Args:
            filename (str, optional): The filename to load the agent from. Defaults to "montecarlo.pkl".

        Returns:
            MonteCarloAgent: The deserialized agent.
        """
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
        