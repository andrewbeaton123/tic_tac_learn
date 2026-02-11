
"""
Orchestrates the parallel training of Monte Carlo agents using the multi_process_controller.
"""
import logging
import mlflow
import pickle
from tqdm import tqdm
import numpy as np
import random
import time
import os
from collections import defaultdict

from tic_tac_learn.control import Config_2_MC
from tic_tac_learn.execution.multi_process_controller import multi_process_controller
from tic_tac_learn.agents.monte_carlo_q_learning import MontecarloQlearningAgent, merge_q_tables, _create_nested_q_table
from tic_tac_learn.game_interfaces.tic_tac_toe_game_interface import TicTacToeGameInterface
from tic_tac_learn.control.learning_rate_decay.decay_rate_calulator import e_decay

def add_exploration_noise(q_table: defaultdict, noise_scale=0.1) -> defaultdict:
    """Add random noise to Q-table values to encourage exploration."""
    noisy_q_table = defaultdict(_create_nested_q_table)
    for state in q_table:
        for action in q_table[state]:
            noise = np.random.normal(0, noise_scale)
            noisy_q_table[state][action] = q_table[state][action] + noise
    return noisy_q_table


def training_worker(config: dict) -> defaultdict:
    """
    This is the target function for each process in the pool.
    It creates an agent, runs its training loop, and returns the learned Q-table.
    """
    # 1. Set up the environment for this worker
    num_episodes = config.get("num_episodes", 1000)
    player_id = config.get("player_id", 1)
    current_q_table = config.get("current_q_table", defaultdict(_create_nested_q_table))
    current_learning_rate = config.get("learning_rate", 0.1)

    conf = Config_2_MC() # The config is a singleton, so this retrieves the instance

    
    # Each process needs its own game interface instance
    game_interface = TicTacToeGameInterface(
        current_player=player_id,
        allowed_players=conf.get_allowed_players()
    )

    # 2. Create the agent, using parameters from the shared config instance
    agent = MontecarloQlearningAgent(
        game_interface=game_interface,
        player_id=player_id,
        learning_rate=current_learning_rate,
        discount_factor=conf.discount_factor,
        exploration_rate=conf.exploration_rate,
        initial_q_table=current_q_table
    )

    # 3. Run the training and return the resulting Q-table
    logging.info(f"Worker starting training for {num_episodes} episodes with LR {current_learning_rate:.4f}.")
    q_table = agent.train(num_episodes)
    logging.info(f"Worker finished training.")
    return q_table

def test_agent(q_table: defaultdict, 
               num_test_games: int, 
               player_id: int, 
               config_manager: Config_2_MC) -> dict:
    """
    Tests the performance of the agent with the given Q-table.
    The agent plays against a random opponent.

    Args:
        q_table (defaultdict): The Q-table to test.
        num_test_games (int): The number of games to play for testing.
        player_id (int): The ID of the player whose performance is being tested.
        config_manager (Config_2_MC): The configuration manager.

    Returns:
        dict: A dictionary containing win, loss, and draw counts.
    """
    conf = Config_2_MC()
    wins = 0
    losses = 0
    draws = 0

    # Set a seed for reproducibility during testing
    random.seed(42) 

    test_game_interface = TicTacToeGameInterface(current_player=player_id,
                                                allowed_players=conf.get_allowed_players())
    
    test_agent_instance = MontecarloQlearningAgent(
        game_interface=test_game_interface,
        player_id=player_id,
        learning_rate=0.0, # Not used in testing
        discount_factor=config_manager.discount_factor, 
        exploration_rate=0.0 # No exploration during testing
    )
    test_agent_instance.q_table = q_table # Assign the provided Q-table for testing

    for i in range(num_test_games):
        test_game_interface.reset()
        
        # Determine who starts the game randomly for testing fairness
        current_test_player = random.choice([1, 2]) # Assuming players are 1 and 2
        test_game_interface.current_player = current_test_player # Set initial player

        while not test_game_interface.is_game_over():
            if test_game_interface.current_player == player_id:
                state = test_game_interface.get_state()
                action = test_agent_instance._choose_best_action(state) # Always exploit
                test_game_interface.make_move(action)
                if test_game_interface.is_game_over():
                    break
            else:
                # Random opponent
                possible_actions = test_game_interface.get_possible_actions()
                if not possible_actions:
                    logging.warning(f"Test game {i+1}: No possible actions for random opponent, but game not over. Board: {test_game_interface.get_state()}")
                    break # Exit loop to prevent infinite loop
                random_action = random.choice(possible_actions)
                test_game_interface.make_move(random_action)
                if test_game_interface.is_game_over():
                    break
        
        winner = test_game_interface.get_winner()
        
        # Add detailed logging for each test game result
        if (i + 1) % (num_test_games // 10) == 0 or i == 0: # Log every 10% or first game
            logging.debug(f"Test Game {i+1} - Final Board: {test_game_interface.get_state()}")
            logging.debug(f"Test Game {i+1} - GameInterface Winner: {winner}")
            logging.debug(f"Test Game {i+1} - Underlying Game Winner: {test_game_interface.game.winner}")
            logging.debug(f"Test Game {i+1} - Game Over: {test_game_interface.is_game_over()}")
            logging.debug(f"Test Game {i+1} - Valid Moves Left: {test_game_interface.get_valid_moves()}")

        # Add new logging here to debug draw issue
        if test_game_interface.is_game_over():
            logging.debug(f"Game {i+1} ended. Winner: {winner}, Game Over: {test_game_interface.is_game_over()}, Valid Moves: {test_game_interface.get_valid_moves()}")
            if winner == 0:
                logging.debug(f"Game {i+1} identified as a DRAW.")
            elif winner == player_id:
                logging.debug(f"Game {i+1} identified as a WIN for player {player_id}.")
            else:
                logging.debug(f"Game {i+1} identified as a LOSS for player {player_id}.")

        if winner == player_id:
            wins += 1
        elif winner == 0: # Assuming 0 is a draw
            draws += 1
        else:
            losses += 1
            
    return {"wins": wins, "losses": losses, "draws": draws}

def run_parallel_training(conf: Config_2_MC):
    """
    Sets up and executes the multi-process training run with step-by-step testing.
    """
    master_q_table =add_exploration_noise(defaultdict(_create_nested_q_table))
    total_games_played = 0

    # Calculate games per step for each core
    games_per_step_per_core = int(conf.total_games / conf.steps / conf.cores)
    
    # Initial learning rate
    current_learning_rate = conf.learning_rate_start

    for step in range(conf.steps):
        logging.info(f"\n--- Starting Training Step {step + 1}/{conf.steps} ---")
        step_start_time = time.time()

        # Prepare configurations for workers for this step
        training_configs = []
        for _ in range(conf.cores):
            training_configs.append({
                "player_id": conf.training_player, 
                "num_episodes": games_per_step_per_core,
                "current_q_table": master_q_table, # Pass the current master Q-table
                "learning_rate": current_learning_rate
            })

        # Execute Parallel Training for this step
        list_of_q_tables_from_step = multi_process_controller(
            func=training_worker,
            configs=training_configs,
            cores=conf.cores
        )

        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        total_games_in_step = games_per_step_per_core * conf.cores
        total_games_played += total_games_in_step

        logging.info(f"Step {step + 1} training finished.")

        # Merge Results from this step into the master Q-table
        logging.info(f"Merging Q-tables from step {step + 1}...")
        master_q_table = merge_q_tables(list_of_q_tables_from_step)
        
        logging.info(f"Master Q-table has {len(master_q_table)} states after step {step + 1}.")

        # Performance Metrics for this step
        # Ensure step_duration is not negative or zero
        if step_duration <= 0:
            logging.warning(f"Step {step + 1} duration is non-positive ({step_duration:.2f}s). Setting games_per_second to 0.")
            games_per_second_step = 0.0
        else:
            games_per_second_step = total_games_in_step / step_duration

        logging.info(f"Step {step + 1} duration: {step_duration:.2f} seconds")
        logging.info(f"Games per second (step {step + 1}): {games_per_second_step:.2f}")
        mlflow.log_metric("step_duration_seconds", step_duration, step=step)
        mlflow.log_metric("games_per_second_step", games_per_second_step, step=step)
        mlflow.log_metric("total_games_played", total_games_played, step=step)
        mlflow.log_metric("current_learning_rate", current_learning_rate, step=step)

        # Agent Testing for this step
        num_test_games = conf.test_games_per_step
        logging.info(f"Starting agent testing for step {step + 1} with {num_test_games} games...")
        test_results = test_agent(master_q_table, num_test_games, player_id=1, config_manager=conf)

        win_percentage = (test_results["wins"] / num_test_games) * 100
        loss_percentage = (test_results["losses"] / num_test_games) * 100
        draw_percentage = (test_results["draws"] / num_test_games) * 100

        logging.info(f'Step {step + 1} Test Results: Wins={test_results["wins"]}, Losses={test_results["losses"]}, Draws={test_results["draws"]}')
        logging.info(f"Step {step + 1} Win Percentage: {win_percentage:.2f}%")
        logging.info(f"Step {step + 1} Loss Percentage: {loss_percentage:.2f}%")
        logging.info(f"Step {step + 1} Draw Percentage: {draw_percentage:.2f}%")

        mlflow.log_metric("test_win_percentage", win_percentage, step=step)
        mlflow.log_metric("test_loss_percentage", loss_percentage, step=step)
        mlflow.log_metric("test_draw_percentage", draw_percentage, step=step)
        logging.info(f"Test results for step {step + 1} logged to MLflow.")

        # Log the Q-table as an artifact for this step
        q_table_artifact_dir = "q_tables"
        os.makedirs(q_table_artifact_dir, exist_ok=True)
        q_table_path_step = os.path.join(q_table_artifact_dir, f"q_table_step_{step + 1}.pkl")
        try:
            with open(q_table_path_step, "wb") as f:
                pickle.dump(dict(master_q_table), f)
            mlflow.log_artifact(q_table_path_step, artifact_path="q_tables")
            logging.info(f"Logged Q-table for step {step + 1} to MLflow.")
        except Exception as e:
            logging.error(f"Failed to save or log Q-table artifact for step {step + 1}: {e}")

        # Update learning rate for the next step (simple linear decay example)
        if step < conf.frozen_learning_rate_steps:
            # Flat learning rate phase
            pass
        else:
            # Decay phase
            current_learning_rate = max(conf.learning_rate_min,
                                        current_learning_rate - conf.learning_rate_decay_rate)
            logging.info(f"Updated learning rate to {current_learning_rate:.4f} for next step.")

            

    logging.info("\n--- All Training Steps Completed ---")

    # Final logging (optional, as each step is logged)
    mlflow.log_metric("final_q_table_size", len(master_q_table))
