import numpy as np
from joblib import Parallel, delayed
from threading import Lock
from itertools import product
import random 
from tqdm import tqdm

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


class MonteCarloAgent:
    def __init__(self, epsilon, all_possible_states):
        self.epsilon = epsilon
        self.q_values = {}
        self.returns = {}
        self.all_possible_states = all_possible_states

    def initialize_q_values(self):
        for state in self.all_possible_states:
            state_str = tuple(state.board.flatten())
            valid_moves = state.get_valid_moves()
            self.q_values[state_str] = np.zeros(len(valid_moves))

    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(state.get_valid_moves()))
        state_str = tuple(state.board.flatten())
        action_values = self.q_values.get(state_str, np.zeros(len(state.get_valid_moves())))
        return np.argmax(action_values)

    def train_episode(self, env):
        episode_data = []
        episode_returns = []
        
        while not env.is_game_over():
            if env.current_player == 1:
                action = self.epsilon_greedy_policy(env)
            else:
                action = np.random.choice(len(env.get_valid_moves()))
            env.make_move(*env.get_valid_moves()[action])

        state_str = tuple(env.board.flatten())
        episode_data.append((state_str, self.q_values[state_str]))

        for state_str, q_values in episode_data:
            for action, value in enumerate(q_values):
                self.q_values[state_str][action] = sum(
                    episode_returns[i]
                    for i in range(len(episode_returns))
                    if state_str == episode_data[i][0]
                    and episode_data[i][1][action] == value
                )

        return self.q_values, self.returns

    def train(self, episodes, cores):
        Parallel(n_jobs=cores)(
            delayed(self.train_episode)(TicTacToe(random.choice([1, 2])))
            for _ in range(episodes)
        )

        # Test your training logic...
        wins = 0
        draws = 0
        for _ in range(10000):
            env = TicTacToe(random.choice([1, 2]))
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

if __name__ == "__main__":
    episodes = 10000
    cores = 4
    epsilon = 0.1

    # Initialize Q-values for all possible state-action pairs
    def generate_all_states():
        states = []
        for player_marks in tqdm(product([0, 1, 2], repeat=9)):  # 0 represents an empty cell
            board = np.array(player_marks).reshape((3, 3))
            tic_tac_toe_instance = TicTacToe(1,board)
            states.append(tic_tac_toe_instance)
        return states
    

    all_possible_states = generate_all_states()
    rl_model = MonteCarloAgent(epsilon, all_possible_states)
    rl_model.initialize_q_values()
    rl_model.train(episodes, cores)