"""
An agent that performs Monte Carlo Q-learning using the game interface.
"""

import random
from collections import defaultdict
from tic_tac_learn.game_interfaces.game_interface_abc import GameInterface
import logging


def _create_nested_q_table():
    """
    A helper function to create a nested defaultdict for the Q-table.
    This is defined at the top level to be pickleable by multiprocessing.
    """
    return defaultdict(float)

class MontecarloQlearningAgent:
    """
    A Monte Carlo Q-learning agent that interacts with a game environment
    through the GameInterface.

    Attributes:
        game_interface (GameInterface): The game environment to interact with.
        player_id (int): The ID of the player this agent represents.
        learning_rate (float): The learning rate (alpha) for Q-value updates.
        discount_factor (float): The discount factor (gamma) for future rewards.
        exploration_rate (float): The exploration rate (epsilon) for the epsilon-greedy policy.
        q_table (defaultdict): The Q-table storing state-action values.
    """

    def __init__(self, 
                 game_interface: GameInterface, 
                 player_id: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 exploration_rate: float = 0.1,
                 initial_q_table: defaultdict | None = None):
        """
        Initializes the MontecarloQlearningAgent.

        Args:
            game_interface (GameInterface): The game environment.
            player_id (int): The ID of the player.
            learning_rate (float, optional): The learning rate. Defaults to 0.1.
            discount_factor (float, optional): The discount factor. Defaults to 0.9.
            exploration_rate (float, optional): The exploration rate. Defaults to 0.1.
            initial_q_table (defaultdict, optional): An optional initial Q-table to start with.
        """
        self.game_interface = game_interface
        self.player_id = player_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        if initial_q_table is not None:
            self.q_table = initial_q_table
        else:
            self.q_table = defaultdict(_create_nested_q_table)

    def _choose_best_action(self, state: tuple) -> int:
        """
        Chooses the best action from the Q-table (exploitation).

        Args:
            state (tuple): The current state of the game.

        Returns:
            int: The best action.
        """
        q_values = self.q_table[state]
        if not q_values:
            # If no Q-values for this state, choose a random action
            possible_actions = self.game_interface.get_possible_actions()
            return random.choice(possible_actions)
        
        max_q_value = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q_value]
        return random.choice(best_actions)

    def choose_action(self, state: tuple) -> int:
        """
        Chooses an action based on the epsilon-greedy policy.

        Args:
            state (tuple): The current state of the game.

        Returns:
            int: The chosen action.
        """
        if random.uniform(0, 1) < self.exploration_rate:
            # Explore: choose a random valid action
            possible_actions = self.game_interface.get_possible_actions()
            return random.choice(possible_actions)
        else:
            # Exploit: choose the best action from the Q-table
            return self._choose_best_action(state)

    def update_q_table(self, episode_history: list, reward: float):
        """
        Updates the Q-table based on the experiences of a single episode.

        Args:
            episode_history (list): A list of (state, action) tuples from the episode.
            reward (float): The reward received at the end of the episode.
        """
        g = 0
        for state, action in reversed(episode_history):
            g = self.discount_factor * g + reward
            old_q_value = self.q_table[state][action]
            self.q_table[state][action] = old_q_value + self.learning_rate * (g - old_q_value)

    def train(self, num_episodes: int) -> defaultdict:
        """
        Trains the agent for a specified number of episodes and returns the Q-table.

        The agent plays against an opponent that always chooses the best move
        from this agent's own Q-table (self-play).

        Args:
            num_episodes (int): The number of episodes to train for.

        Returns:
            defaultdict: The trained Q-table.
        """
        for episode in range(num_episodes):
            self.game_interface.reset()
            episode_history = []
            
            while not self.game_interface.is_game_over():
                if self.game_interface.current_player == self.player_id:
                    state = self.game_interface.get_state()
                    action = self.choose_action(state)
                    episode_history.append((state, action))
                    self.game_interface.make_move(action)
                else:
                    # Opponent's turn: choose the best action from the Q-table
                    if not self.game_interface.is_game_over():
                        state = self.game_interface.get_state()
                        opponent_action = self._choose_best_action(state)
                        self.game_interface.make_move(opponent_action)

            reward = self.game_interface.get_reward(self.player_id)
            self.update_q_table(episode_history, reward)

            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed.")
        
        return self.q_table


def merge_q_tables(q_tables: list[defaultdict], merge_strategy: str = 'max') -> defaultdict:
    """
    Merges multiple Q-tables using the specified strategy.

    Args:
        q_tables (list[defaultdict]): List of Q-tables to merge
        merge_strategy (str): Strategy to use ('max', 'avg', 'weighted_avg')

    Returns:
        defaultdict: Merged Q-table
    """
    if not q_tables:
        logging.warning("No Q-tables provided for merging")
        return defaultdict(_create_nested_q_table)

    if merge_strategy not in ['max']:
        logging.error(f"Merge Q tables recieved an invalid merge strategy : {str(merge_strategy)}")
        raise ValueError(f"Merge Q tables recieved an invalid merge strategy : {str(merge_strategy)}")
    
    merged_q_table = defaultdict(_create_nested_q_table)
    merge_stats = {
        'total_states': 0,
        'total_actions': 0,
        'max_q_value': float('-inf'),
        'min_q_value': float('inf')
    }

    if merge_strategy == 'max':
        # Take maximum Q-value for each state-action pair
        for q_table in q_tables:
            for state, actions in q_table.items():
                for action, q_value in actions.items():
                    current_q = merged_q_table[state][action]
                    merged_q_table[state][action] = max(current_q, q_value)
                    
                    # Update statistics
                    merge_stats['max_q_value'] = max(merge_stats['max_q_value'], q_value)
                    merge_stats['min_q_value'] = min(merge_stats['min_q_value'], q_value)

    else:  # 'avg' or 'weighted_avg'
        state_action_counts = defaultdict(lambda: defaultdict(int))
        
        for q_table in q_tables:
            for state, actions in q_table.items():
                for action, q_value in actions.items():
                    merged_q_table[state][action] += q_value
                    state_action_counts[state][action] += 1

        # Calculate averages and collect statistics
        for state, actions in merged_q_table.items():
            merge_stats['total_states'] += 1
            for action, total_q_value in actions.items():
                merge_stats['total_actions'] += 1
                count = state_action_counts[state][action]
                if count > 0:  # Protect against division by zero
                    avg_q_value = total_q_value / count
                    merged_q_table[state][action] = avg_q_value
                    
                    merge_stats['max_q_value'] = max(merge_stats['max_q_value'], avg_q_value)
                    merge_stats['min_q_value'] = min(merge_stats['min_q_value'], avg_q_value)

    # Log merge statistics
    logging.info(f"Q-table merge completed: {merge_stats}")
    
    return merged_q_table
