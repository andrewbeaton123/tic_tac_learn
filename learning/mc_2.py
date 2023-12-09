import numpy as np
from tqdm import tqdm
import random
class MonteCarloAgent:
    def __init__(self, epsilon=0.1):
        self.q_values = {}  # Dictionary to store state-action values
        self.returns = {}   # Dictionary to store returns for each state-action pair
        self.epsilon = epsilon  # Epsilon for epsilon-greedy policy

    def initialize_q_values(self, all_possible_states):
        for state in all_possible_states:
            state_str = str(state.board.flatten())
            valid_moves = state.get_valid_moves()
            self.q_values[state_str] = np.zeros(len(valid_moves))

    def epsilon_greedy_policy(self, state):
        state_str = str(state.board.flatten())
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.q_values[state_str]))  # Random action
        else:
            try:
                return np.argmax(self.q_values[state_str])  # Greedy action
            except Exception:
                return np.random.choice(len(self.q_values[state_str])) 

    def train(self, episodes, env):
        for episode in tqdm(range(episodes), colour="green"):
            trajectory = []  # List to store the trajectory (state, action, reward)

            # Randomly determine the first player
            first_player = random.randint(0, 1)

            if first_player == 1:
                env = TicTacToe()
                state = env
                done = False

                while not done:
                    if state.current_player == 1:
                        action = self.epsilon_greedy_policy(env)
                        action = env.get_valid_moves()[action]
                        board_state, reward, done = env.step(action)

                    else:
                        action = random.choice(len(state.get_valid_moves()))
                        action = env.get_valid_moves()[action]
                        board_state, reward, done = env.step(action)

                    trajectory.append((env, action_index, reward))

            else:
                # The second player goes first in this case
                # Randomly determine which player to play as
                player = random.randint(0, 1)

                if player == 1:
                    # Agent plays as 1st player
                    env = TicTacToe()
                    state = env
                    done = False

                    while not done:
                        if state.current_player == 1:
                            action = self.epsilon_greedy_policy(env)
                            action = env.get_valid_moves()[action]
                            board_state, reward, done = env.step(action)

                        else:
                            action = random.choice(len(state.get_valid_moves()))
                            action = env.get_valid_moves()[action]
                            board_state, reward, done = env.step(action)

                        trajectory.append((env, action_index, reward))

                else:
                    # Agent plays as second player
                    env = TicTacToe()
                    state = env
                    done = False

                    while not done:
                        action = random.choice(len(state.get_valid_moves()))
                        action = env.get_valid_moves()[action]
                        board_state, reward, done = env.step(action)

                        if state.current_player == 1:
                            action = self.epsilon_greedy_policy(env)
                            action = env.get_valid_moves()[action]
                            board_state, reward, done = env.step(action)

                        trajectory.append((env, action_index, reward))

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 3x3 Tic Tac Toe board
        self.current_player = 1  # Player 1 starts
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.winner = None

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
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
            self.winner = 0  # Draw
            return 0
        return None  # Game is ongoing

    def is_game_over(self):
        return self.check_winner() is not None

    def step(self, action):
        if not self.is_game_over() and action in self.get_valid_moves():
            row, col = action
            self.make_move(row, col)
            if self.is_game_over():
                if self.winner == 1:
                    return self.board, 1, True  # Player 1 wins
                elif self.winner == 2:
                    return self.board, -1, True  # Player 2 wins
                else:
                    return self.board, 0, True  # Draw
            else:
                return self.board, 0, False  # Game continues
        else:
            return self.board, -1, True  # Invalid move or game already over

# Usage:
if __name__ == "__main__":
    state_size = 9  # State size for Tic Tac Toe
    agent = MonteCarloAgent(epsilon=0.1)

    # Generate all possible initial game states
    all_possible_states = []
    for row in range(3):
        for col in range(3):
            for player in [1, 2]:
                initial_state = TicTacToe()
                initial_state.make_move(row, col)
                initial_state.current_player = player
                all_possible_states.append(initial_state)

    
    # Initialize Q-values for all possible states
    agent.initialize_q_values(all_possible_states)

    print(agent.q_values)
    exit()
    
    # Train the agent using the Monte Carlo method
    episodes = 10000
    env = TicTacToe()
    agent.train(episodes, env)

    # After training, you can use the Q-values to play Tic Tac Toe.
    # Example: Let the agent play against a random opponent for evaluation.
    wins = 0
    for _ in range(1000):
        env = TicTacToe()
        while not env.is_game_over():
            if env.current_player == 1:
                action = agent.epsilon_greedy_policy(env)
            else:
                action = np.random.choice(len(env.get_valid_moves()))
            env.step(action)
        if env.check_winner() == 1:
            wins += 1

    print(f"Agent won {wins} out of {episodes} games.")