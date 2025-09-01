# Tic Tac Toe Game with Reinforcement Learning

![Alt Text](https://i.makeagif.com/media/4-24-2016/N2q-9R.gif)

This repository contains a Tic Tac Toe game that uses reinforcement learning techniques to train the game player. The ML approach uses Q-values, epsilon greedy selections, and multi-threading to learn and improve its gameplay.

## Configuration

The application uses YAML-based configuration files located in environment-specific folders under the `config/` directory. The environment is selected using the `TICLEARN_ENV` environment variable.

### Configuration Structure

```
config/
├── development/
│   └── config.yaml
├── production/
│   └── config.yaml
└── debug/
    └── config.yaml
```

### Example Configuration (config.yaml)

```yaml
name: "Tic Tac Learn 0.1.2"
mlflow_name: "mlflow_name"
level: "TRAINING"
total_games: 1_000_000_000  # 1e9
steps: 4
agent_reload: None

training:
  cores: 3
  learning_rate_start: 0.8
  learning_rate_min: 0.001
  learning_rate_scaling: 1
  test_games_per_step: 30000
  learning_rate_flat_games: 200_000_000  # 20% of total_games
```

### Running with Different Configurations

To run the application with a specific configuration:

```bash
# Development configuration
export TICLEARN_ENV=development
python main.py

# Production configuration
export TICLEARN_ENV=production
python main.py
```

### Docker Usage

Using Docker with specific configurations:

```bash
# Build the image
docker build -t tic_tac_learn:latest .

# Run with development configuration
docker run -e TICLEARN_ENV=development tic_tac_learn:latest

# Run with production configuration
docker run -e TICLEARN_ENV=production tic_tac_learn:latest
```



## Training Sessions

Different training configurations are now managed through YAML files instead of code. Examples include:

- **Debug Session**: Use `config/debug/config.yaml` with smaller game counts
- **Warmup Session**: Use `config/warmup/config.yaml` with moderate game counts
- **Production Training**: Use `config/production/config.yaml` for full training runs

## Current Status

This is a work in progress. The configuration system has been updated to use YAML files and environment variables for better maintainability and deployment flexibility.
