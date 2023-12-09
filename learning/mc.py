import numpy as np
import logging 
from tqdm import tqdm
import  pickle as pkl
logging.basicConfig(level="INFO")

# Monte Carlo Control Agent

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))  # 3x3 Tic Tac Toe board
        self.current_player = 1  # Player 1 starts
        self.winner =0 # currently a draw 

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
                    return self.board, 1, True  # Player 1 wins
                elif self.winner == 2:
                    return self.board, -1, True  # Player 2 wins
                else:
                    return self.board, 0, True  # Draw
            else:
                return self.board, 0, False  # Game continues
        else:
            return self.board, -1, True  # Invalid move or game already over


class MonteCarloAgent:
    def __init__(self):
        self.q_values = {}  # Dictionary to store state-action values
        self.returns = {}    # Dictionary to store returns for each state-action pair
        self.epsilon = 0.1  # Epsilon for epsilon-greedy policy

    def initialize_q_values(self, all_possible_states):
        for state in all_possible_states:
            state_str = str(state.board.flatten())
            valid_moves = state.get_valid_moves()
            self.q_values[state_str] = np.zeros(len(valid_moves))

    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            logging.debug("rand was less than epsilon, selecting a random action")
            return np.random.choice(len(state.get_valid_moves()))  # Random action
        else:
            state_str = str(state.board.flatten())
            action_values = self.q_values.get(state_str, np.zeros(9))  # Initialize Q-values if not seen before
            logging.debug(f"Action values are {action_values}")
            logging.debug(f"Selected action is {np.argmax(action_values)}")
            return np.argmax(action_values)  # Greedy action

    def train(self, episodes):
        for episode in tqdm(range(episodes),colour="green"):
            trajectory = []  # List to store the trajectory (state, action, reward)

            env = TicTacToe()
            state = env
            done = False

            while not done:
                #print(f"valid moves are  {state.get_valid_moves()}")
                action_index = self.epsilon_greedy_policy(env)
                action = env.get_valid_moves()[action_index]
                logging.debug(f"Action is :{action}")
                board_state, reward, done = env.step(action)

                if str(state.board.flatten()) not in self.q_values.keys():
                    self.q_values[str(state.board.flatten())]= np.zeros(9)
                trajectory.append((env, action_index, reward))

                #state = next_state

            # Update Q-values using the returns from the episode
            returns = 0
            logging.debug(self.q_values.keys())

            for t in reversed(range(len(trajectory))):
                state, action, reward = trajectory[t]
                
                state_str = str(state.board.flatten())
                
                #action_tuple = tuple(action)
                returns = reward + returns
                self.returns[(state_str,action)] = self.returns.get((state_str, action), 0) + 1
                self.q_values[state_str][action] += (returns - self.q_values[state_str][action]) / self.returns[(state_str, action)]

if __name__ == "__main__":
    agent = MonteCarloAgent()
    episodes = 250000
    
    # Initialize Q-values for all possible state-action pairs
    # Generate all possible initial game states
    # all_possible_states = []
    # for row in range(3):
    #     for col in range(3):
    #         for player in [1, 2]:
    #             initial_state = TicTacToe()
    #             initial_state.make_move(row, col)
    #             initial_state.current_player = player
    #             all_possible_states.append(initial_state)
    
    # # Initialize Q-values for all possible states
    # agent.initialize_q_values(all_possible_states)
    logging.info(agent.q_values.keys())

    logging.info(len(agent.q_values))


    agent.train(episodes)
    
    #with open('MC_agent_1.pickle', 'wb') as handle:
    #    pkl.dump(agent, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    # After training, you can use the Q-values to play Tic Tac Toe.
    # Example: Let the agent play against a random opponent for evaluation.
    wins = 0
    draws= 0
    for _ in range(10000):
        env = TicTacToe()
        while not env.is_game_over():
            if env.current_player == 1:
                action = agent.epsilon_greedy_policy(env)
            else:
                action = np.random.choice(len(env.get_valid_moves()))
            env.make_move(*env.get_valid_moves()[action])
        if env.check_winner() == 1:
            wins += 1
        elif env.check_winner() ==0:
            draws +=1

    print(f"Agent won {wins} out of 100000 games.")
    print(f"Games drawn {draws}")
