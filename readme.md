# Tic Tac Learn: A Reinforcement Learning Framework

![Alt Text](https://i.makeagif.com/media/4-24-2016/N2q-9R.gif)

This repository contains a framework for training agents to play games like Tic Tac Toe using reinforcement learning. It features a modular architecture, parallel training capabilities, and configurable run settings.

## Key Features

- **Game Interface Abstraction**: A core design principle is the `GameInterface`, an abstract base class that decouples the learning agents from the specific game logic. This allows the same agent to learn different games, provided a compatible interface is created.
- **Monte Carlo Q-learning Agent**: A new, fully implemented `MontecarloQlearningAgent` that learns to play games by interacting with the `GameInterface`. It uses an epsilon-greedy policy for action selection and learns from episode rollouts.
- **Parallel Training**: The framework uses a high-performance, parallel training strategy. Multiple instances of the agent are run on different CPU cores, and their learned Q-tables are merged after training is complete. This "Combine Post-Training" approach maximizes throughput by eliminating inter-process communication during the training loops.
- **Configuration via YAML**: All key parameters for the Monte Carlo simulation are managed in the `config.yml` file, allowing for easy experimentation without code changes.

## How It Works

### 1. Configuration

The primary settings for a training run are defined in `config.yml` under the `monte_carlo_settings` section. This includes parameters like the number of games, learning rate, and MLflow tracking details.

```yaml
# Example from config.yml
monte_carlo_settings:
  run_name: "My First Run"
  total_games: 100000
  experiment_name: "Tic Tac Dev"
  steps: 10
  cores: 4
  decay: 
    type: LINEAR
    params: 
      learning_rate_frozen_steps: 1
      learning_rate_scaling: 1 
      learning_rate_inital: 0.7
      learning_rate_min: 0.01
  test_games_per_step: 3000
  # ... and other parameters
```

### 2. Initialization

When `main.py` is executed, it uses `ConfigManager` to load the settings from `config.yml` into a configuration object.

### 3. Parallel Training

The application uses Python's `multiprocessing` module to achieve parallel training. Here is the workflow:

1.  A pool of worker processes is created (one for each CPU core specified in the config).
2.  Each worker process creates its own instance of the `MontecarloQlearningAgent` and a game interface.
3.  Each agent trains independently for a set number of episodes, running game simulations at full speed without any communication overhead.
4.  Upon completion, each agent returns its learned Q-table (a dictionary of state-action values).

### 4. Merging Q-tables

After all worker processes have finished, the main process collects the list of individual Q-tables. A `merge_q_tables` function then combines them into a single, master Q-table by averaging the Q-values for each state-action pair that was learned by multiple agents.

### 5. Logging

The entire run, including parameters and the final merged Q-table (as an artifact), is logged to MLflow for tracking and analysis.

## How to Run

1.  **Install Dependencies**: 
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure the Run**: Edit the `monte_carlo_settings` in `config.yml` to define your training parameters.
3.  **Run the Training**: 
    ```bash
    python main.py
    ```

## Current Status

This code is a work in progress. Recent significant improvements include:
- Implementation of the `MontecarloQlearningAgent`.
- A robust and performant parallel training strategy ("Combine Post-Training").
- Centralized configuration management using `config.yml`.
- Refactoring of the core configuration class to be more flexible.
