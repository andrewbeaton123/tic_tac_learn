# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Trains a Monte Carlo Q-learning agent to play Tic Tac Toe with parallel multiprocessing, then logs the trained model to MLflow. Two sibling repos provide key parts (installed as git dependencies in `pyproject.toml`):
- `tic_tac_toe_game`: the game engine (`TicTacToe`)
- `tic_tac_toe_model`: `TicTacToeModel`, an `mlflow.pyfunc` model. It's the shared contract between this training repo and the serving layer. Changes to how the model is saved or what its signature looks like belong in that repo, not here.

## Environment & commands

Use the project `.venv` only. Never install packages into the base/system Python.

```bash
poetry install --with test                         # deps + pytest/pytest-cov
.venv/bin/python -m tic_tac_learn                  # run training (or: python -m tic_tac_learn.main)
.venv/bin/python -m pytest                         # all tests
.venv/bin/python -m pytest tests/agents/test_monte_carlo_q_learning.py::TestQTableMerging::test_merge_empty_returns_defaultdict
```

- Run commands from the repo root. `main.py` loads `tic_tac_learn/config.yml` by relative path, and `load_config` creates `q_tables/`, `logs/`, and `models/` relative to the working directory.
- The MLflow server is optional. The top-level `tracking:` block in `config.yml` controls it: `log_mlflow`, `mlflow_tracking_uri` (falls back to `$MLFLOW_TRACKING_URI`), `on_unavailable: warn | fail`, and `health_check_timeout_seconds`. To run offline, set `log_mlflow: false`, or leave `on_unavailable: warn` and let the health check fall back.
- `tests/test_config_manager.py` imports the legacy `tic_tac_learn.src.config_management` using relative `..` imports, so it's expected to fail. As of the MLflow decoupling work, 6 other tests also fail on the current code (q-table merge `avg`, exponential decay, and `TicTacToeModel.get_model_signature` raising `TypeError` under mlflow 3.12).

## Architecture (active execution path)

1. **`main.py`**: loads config via `control/factory.py::load_config`, builds a tracker with `tracking.create_tracker(conf.runner)`, then runs inside `tracker.start_run()`: it logs `monte_carlo_settings` as params and calls `run_parallel_training(conf, tracker)`.
2. **`control/factory.py` + `control/schemas.py`**: parse `tic_tac_learn/config.yml` into a frozen `MonteCarloConfig` (`env` / `agent` / `runner` sub-configs, plus `raw_config`). `MonteCarloConfig` exposes flat convenience properties (`conf.total_games`, `conf.learning_rate_start`, and so on) that the rest of the code relies on. `RunnerConfig` also holds the tracking settings parsed from `tracking:`.
3. **`monte_carlo_learning/flow_control/run_monte_carlo.py`**: the core loop. For each of `steps` steps:
   - Sends the current master Q-table to `cores` workers through `execution/multi_process_controller.py` (a `multiprocessing.Pool`, `imap_unordered`).
   - Each `training_worker` builds its own `TicTacToeGameInterface` and `MontecarloQlearningAgent`, trains for `total_games / steps / cores` episodes, and returns its Q-table.
   - `merge_q_tables` combines the results. The default strategy is `'max'`, not the averaging that the readme describes. `'avg'` is also supported.
   - Tests the merged table against a random opponent.
   - Reports metrics through the tracker and saves a per-step `.pkl` Q-table to `q_tables/`, which is also logged as a tracker artifact.
   - Decays the learning rate after the `learning_rate_frozen_steps` steps.

   At the end, it wraps the master table in `TicTacToeModel` and always saves it to `models/<experiment>_<run>_<timestamp>/Q_values.safetensors`, then calls `tracker.log_model`. This module never imports `mlflow`.
4. **`tracking/tracker.py`**: the only place MLflow tracking calls are made.
   - `ExperimentTracker` ABC, with two implementations: `NullTracker` (no-op) and `MlflowTracker`.
   - `create_tracker` runs a quick `/health` check against http(s) URIs and applies `on_unavailable`.
   - `MlflowTracker` routes every call through `_call()`. The first failure is logged and turns tracking off for the rest of the run, so a lost server never stalls or kills training. Keep new tracking calls behind this interface.
5. **`agents/monte_carlo_q_learning.py`**: epsilon-greedy MC agent. The Q-table is a `defaultdict(_create_nested_q_table)`, mapping state tuple → action → value. The nested-default factory must stay a module-level function so the table can be pickled across processes.
6. **`game_interfaces/`**: `GameInterface` ABC in `game_interface_abc.py` decouples agents from game logic. `TicTacToeGameInterface` adapts `tic_tac_toe_game`. To support a new game, add another implementation. The `games:` section of `config.yml` already lists placeholders (chess, poker, etc.).
7. **`control/learning_rate_decay/`**: `decay_from_name(type, step, conf)` dispatches to constant/linear/exponential decay (the name is case-insensitive). `config.yml` selects it under `monte_carlo_settings.decay`.

## Legacy / dead code

These aren't on the active path. Don't extend them without checking first:
- `tic_tac_learn/src/`
- `config_management/`
- `control/config_class_v2_MC.py` (the old singleton config)
- `control/mlflow/`
- `results_saving/`
- `game_interfaces/game_interface_factory.py` (broken imports)
- `config/*/config.yaml`

The live config is `tic_tac_learn/config.yml`.

## Git workflow

Never push to `master`. Work on the issue/feature branch (e.g. `andrewbeaton123/issueNN`) and let the user merge.

A hook in `.claude/settings.json` runs `.claude/hooks/guard.sh` before every Bash command. It blocks pushes to `master`/`main`, commits while on `master`/`main`, and `pip`/`conda` installs that don't target `.venv`. If a command is blocked, fix the command. Don't try to get around the guard.
