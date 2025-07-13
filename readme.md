# Tic Tac Toe Game with Reinforcement Learning


![Alt Text](https://i.makeagif.com/media/4-24-2016/N2q-9R.gif)

This repository contains a Tic Tac Toe game that uses reinforcement learning techniques to train the game player. The ML approach uses Q-values, epsilon greedy selections, and multi-threading to learn and improve its gameplay.

## Reinforcement Learning

The code learns by playing the game multiple times and updating the Q-values based on the outcomes of the games. An epsilon greedy selection strategy is used to balance exploration and exploitation during the learning process.

A key architectural improvement is the introduction of a **Game Interface Abstraction**, which decouples the core game logic from the reinforcement learning agent. This allows for easier integration of different game types in the future.

The learning process now leverages **flexible multiprocessing** by utilizing shared Q-value and return dictionaries across multiple processes. This enables concurrent game simulations and direct updates to the shared learning state, significantly speeding up training and improving the efficiency of the multi-threaded learning.

## Configuration

The training parameters are managed through the `Config_2_MC` class. Key configurable parameters include:
- `total_games`: Total number of games to simulate for training.
- `steps`: Number of training steps.
- `cores`: Number of CPU cores to utilize for parallel game simulations.
- `learning_rate_start`: Initial learning rate.
- `learning_rate_min`: Minimum learning rate.
- `learning_rate_scaling`: Factor for learning rate decay.
- `test_games_per_step`: Number of games to simulate for testing agent performance at each step.
- `learning_rate_flat_games`: Number of games for which the learning rate remains flat.
- `experiment_name`: MLflow experiment name.
- `run_name`: MLflow run name.
- `custom_model_name`: Name for the saved MLflow model artifact.

## Current Status

This code is a work in progress. Recent significant improvements include:
- Implementation of a game interface abstraction for better modularity.
- Refactoring of the multiprocessing system for enhanced flexibility and concurrent training.
- Removal of direct `numpy` dependencies from core agent and game interface logic.
- Resolution of various import errors and runtime exceptions, leading to a more stable training environment.