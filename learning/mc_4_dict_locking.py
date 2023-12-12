import numpy as np
import logging 
from tqdm import tqdm
import  pickle
import random

from joblib import Parallel, delayed, parallel_backend, wrap_non_picklable_objects
logging.basicConfig(level="INFO")

from threading import Lock
from itertools import product
# Monte Carlo Control Agent

class TicTacToe:
    def __init__(self,starting_player:int, board =None):
        
        self.current_player = starting_player # Player 1 starts
        self.winner =0 # currently a draw 
        if board is None:
            self.board = np.zeros((3, 3))  # 3x3 Tic Tac Toe board
        else :
            self.board = board

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.check_winner()
            self.current_player = 3 - self.current_player  # Switch players (1 -> 2, 2 -> 1)
        else:
            raise ValueError("Invalid move")

    def check_winner(self):
        for player in [1, 2]:
            # Check rows, columns, and diagonals for a win
            if np.any(np.all(self.board == player, axis=0)) or \
               np.any(np.all(self.board == player, axis=1)) or \
               np.all(np.diag(self.board) == player) or \
               np.all(np.diag(np.fliplr(self.board)) == player):
                self.winner = player
                return player
        if len(self.get_valid_moves()) == 0:
            return 0  # Draw
        return None  # Game is ongoing

    def is_game_over(self):
        return self.check_winner() is not None

    def print_board(self):
        for row in self.board:
            print(" | ".join(["X" if cell == 1 else "O" if cell == 2 else " " for cell in row]))
            print("-" * 9)

    def step(self, action):
        if not self.is_game_over() and action in self.get_valid_moves():
            row, col = action
            self.make_move(row, col)
            if self.is_game_over():
                if self.winner == 1:
                    return


# Monte Carlo Control Agent

class MonteCarloAgent:
    def __init__(self, epsilon, all_possible_states):
        self.epsilon = epsilon
        self.q_values = {}
        self.returns = {}

    def initialize_q_values(self):
        for state in all_possible_states:
            state_str = tuple(state.board.flatten())
            valid_moves = state.get_valid_moves()
            self.q_values[state_str] = np.zeros(len(valid_moves))

    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(state.get_valid_moves()))
        state_str = str(state.board.flatten())
        action_values = self.q_values.get(state_str, np.zeros(len(state.get_valid_moves())))
        return np.argmax(action_values)


    def train_episode(self, episodes, cores, lock):
    # Create a list to store the state-Q-value pairs and returns for each episode
        episode_data = []
        episode_returns = [] 
        for _ in tqdm(range(episodes)):
            # Initialize the state
            env = TicTacToe(random.choice([1, 2]))
            state_str =None
            # Play the game until it is over
            while not env.is_game_over():
                if env.current_player == 1:
                    action = self.epsilon_greedy_policy(env)
                else:
                    action = np.random.choice(len(env.get_valid_moves()))
                env.make_move(*env.get_valid_moves()[action])

            # Update the Q-values and returns for the episode
            state_str = tuple(env.board.flatten())
            logging.debug(f"monte_carlo_Aget- Train_episode- before saving q value game state is..." )
            logging.debug(env.board)
            #logging.debug(env.board.flatten())
            #logging.debug(env.print_board())
            #logging.debug(self.q_values.keys())

            #try:
            episode_data.append((state_str, (self.q_values[state_str])))
            #except  Exception:
            #    episode_data.append((state_str, np.zeros(len(env.get_valid_moves()))))

            # Update the Q-values dictionary
            for state_str, q_values in episode_data:
                for action, value in enumerate(q_values):
                    self.q_values[state_str][action] = sum(
                        episode_returns[i]
                        for i in range(len(episode_returns))
                        if state_str == episode_data[i][0]
                        and episode_data[i][1][action] == value
                    )

        # Return the updated Q-values and returns dictionaries
        return self.q_values, self.returns

    def train(self, episodes, cores):
        # Create a lock to synchronize access to the q_values and returns dictionaries
        lock = Lock()

        # Create a list of futures to store the results of the parallel training processes
        futures = []

        with Parallel(n_jobs=cores, backend='threading') as pool:
            delayed_train_episode = delayed(self.train_episode)

            # Make the instance method picklable
            delayed_train_episode = wrap_non_picklable_objects(delayed_train_episode)

            # Use the map function to execute the train_episode in parallel
            futures.extend(pool(delayed_train_episode(episodes // cores, cores, lock) for _ in range(cores)))
        # Get the results of the parallel training
        for future in futures:
            q_values, returns = future

            # Lock the dictionaries before updating them
            lock.acquire()

            # Update the q_values dictionary
            for state_str, state_q_values in q_values.items():
                self.q_values[state_str] = state_q_values

            # Update the returns dictionary
            for state_str, state_returns in returns.items():
                self.returns[state_str] = state_returns

            # Unlock the dictionaries
            lock.release()
        # After all the worker processes have finished training their episodes
        # Print the agent's win rate and draws from 10,000 games against a random opponent
        wins = 0
        draws = 0
        for _ in range(10000):
            env = TicTacToe(random.choice([1,2]))
            while not env.is_game_over():
                if env.current_player == 1:
                    action = self.epsilon_greedy_policy(env)
                else:
                    action = np.random.choice(len(env.get_valid_moves()))
                env.make_move(*env.get_valid_moves()[action])

            if env.winner == 1:
                wins += 1
            elif env.winner == 0:
                draws += 1

        print(f"Agent won {wins} out of 10,000 games.")
        print(f"Draws: {draws}")

    def save(self,filename:str):
        # Ask the user for the filename
        filename = filename

        # Save the class to the file using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(filename):
        # Load the class from the file using pickle
        with open(filename, 'rb') as f:
            loaded_agent = pickle.load(f)
def generate_all_states():
    states = []
    for player_marks in product([0, 1, 2], repeat=9):  # 0 represents an empty cell
        board = np.array(player_marks).reshape((3, 3))
        tic_tac_toe_instance = TicTacToe(1,board)
        states.append(tic_tac_toe_instance)
    return states

if __name__ == "__main__":
    episodes = 10000
    cores = 4
    epsilon = 0.1

    # Initialize Q-values for all possible state-action pairs
    all_possible_states = generate_all_states()

    rl_model = MonteCarloAgent(epsilon, all_possible_states)
    rl_model.initialize_q_values()

    rl_model.train(episodes,cores)
    rl_model.save(f"rl_mc_tic_tac_toe_model_{episodes}_{cores}.pkl")