# Standard library imports
import logging
import time
from multiprocessing import Manager

# Third-party imports
import mlflow
from tqdm import tqdm

# Local application imports
from tic_tac_learn.monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent
from tic_tac_learn.multi_processing_tools.multi_process_controller import multi_process_controller
from tic_tac_learn.src.control import Config_2_MC
from tic_tac_learn.src.control.mlflow.log_named_tuple_as_params import log_named_tuple_as_params
from tic_tac_learn.src.control.run_variables import RunVariableCreator
from tic_tac_learn.multi_processing_tools.game_batch_processor import process_game_batch
from tic_tac_learn.monte_carlo_learning.tracking_tools import log_in_progress_mc_model
from tic_tac_learn.src.results_saving.save_controller import save_results_core, save_path_generator
from tic_tac_learn.src.result_plotter.plot_step_info import plot_step_info
from tic_tac_learn.monte_carlo_learning.learning_rate_scaling import learning_rate_scaling
import tic_tac_learn.src.errors as errors


def _check_pre_run_calculations(conf: Config_2_MC) -> None:
    """
    Checks if pre-run calculations have been performed in the configuration.

    Args:
        conf: The configuration object.

    Raises:
        PreRunCalculationsNotComplete: If pre-run calculations are not complete.
    """
    if conf.frozen_learning_rate_steps is None:
        raise errors.PreRunCalculationsNotComplete("Pre-run calculations have not been performed!")


def _initialize_shared_data(manager: Manager, conf: Config_2_MC, all_possible_states: list) -> tuple[dict, dict]:
    """
    Initializes and populates shared Q-value and returns dictionaries.

    Args:
        manager: The multiprocessing Manager instance.
        conf: The configuration object.
        all_possible_states: A list of all possible game states.

    Returns:
        A tuple containing the shared Q-values dictionary and shared returns dictionary.
    """
    shared_q_values = manager.dict()
    shared_returns = manager.dict()

    # Create a temporary agent to initialize the Q-value space
    initial_agent = MonteCarloAgent(conf.learning_rate_start, all_possible_states, conf)
    initial_agent.check_q_value_space_exists()

    # Copy the initialized Q-values and returns to the shared dictionaries
    for state, actions in initial_agent.q_values.items():
        shared_q_values[state] = actions
    for (state, action), returns_list in initial_agent.returns.items():
        shared_returns[(state, action)] = returns_list

    return shared_q_values, shared_returns


def _run_training_step(
    episode_num: int,
    conf: Config_2_MC,
    run_var: RunVariableCreator,
    shared_q_values: dict,
    shared_returns: dict,
    current_learning_rate: float,
    games_per_step: int,
    all_possible_states: list
) -> float:
    """
    Executes a single training step, including multiprocessing and metric logging.

    Args:
        episode_num: The current episode number.
        conf: The configuration object.
        run_var: The RunVariableCreator instance.
        shared_q_values: The shared Q-values dictionary.
        shared_returns: The shared returns dictionary.
        current_learning_rate: The learning rate for the current step.
        games_per_step: Number of games to simulate in this step.
        all_possible_states: A list of all possible game states.

    Returns:
        The updated learning rate for the next step.
    """
    t_before_train = time.time()

    # Prepare configurations for multiprocessing
    games_per_core = int(games_per_step / conf.cores)
    configs = [
        (current_learning_rate, shared_q_values, shared_returns, conf, games_per_core, all_possible_states)
        for _ in range(conf.cores)
    ]

    logging.info(f"Current learning rate is : {current_learning_rate}")
    multi_process_controller(process_game_batch, configs, conf.cores)

    t_after_train = time.time()
    time_taken_to_train = round(t_after_train - t_before_train, 6) + 1e-9
    games_per_sec = round(games_per_step / time_taken_to_train)
    run_var.training_rate.append(games_per_sec)

    logging.debug(f"Trained {games_per_step} games over {conf.cores} cores in {time_taken_to_train} seconds")
    logging.info(f"Training at {games_per_sec} g/s")

    run_var.last_e_total += games_per_step

    # Update learning rate
    new_learning_rate = learning_rate_scaling(current_learning_rate, run_var.last_e_total, episode_num)

    # Test agent performance
    test_agent = MonteCarloAgent(0.0, all_possible_states, conf)  # Epsilon 0 for greedy testing
    test_agent.q_values = shared_q_values
    test_agent.returns = shared_returns

    total_wins, total_draws = test_agent.test(conf.test_games_per_step, conf.cores)

    print(f"For Episodes :{run_var.last_e_total}")
    print(f"Winrate is {round((total_wins / conf.test_games_per_step) * 100)}%")
    print(f"Games drawn {total_draws}")

    log_in_progress_mc_model(test_agent, run_var.last_e_total, bool(run_var.last_e_total >= conf.total_games))

    run_var.overall_res[run_var.last_e_total] = (
        current_learning_rate,
        total_wins,
        total_draws,
        conf.test_games_per_step,
    )

    mlflow.log_metric("In Progress Win Rate", (total_wins / conf.test_games_per_step) * 100, step=episode_num)
    mlflow.log_metric("In Progress Draw Rate", (total_draws / conf.test_games_per_step) * 100, step=episode_num)
    mlflow.log_metric(
        "In Progress Loss Rate",
        ((conf.test_games_per_step - (total_draws + total_wins)) / conf.test_games_per_step) * 100,
        step=episode_num,
    )
    mlflow.log_metric("In Progress Games Per Second", games_per_sec, step=episode_num)

    return new_learning_rate


def _log_final_metrics(conf: Config_2_MC, total_wins: int, total_draws: int, final_learning_rate: float) -> None:
    """
    Logs final MLflow metrics after training is complete.

    Args:
        conf: The configuration object.
        total_wins: Total wins in the final test.
        total_draws: Total draws in the final test.
        final_learning_rate: The learning rate at the end of training.
    """
    mlflow.log_metric("Final Win Rate", (total_wins / conf.test_games_per_step) * 100)
    mlflow.log_metric("Final Draw Rate", (total_draws / conf.test_games_per_step) * 100)
    mlflow.log_metric(
        "Final Loss Rate",
        ((conf.test_games_per_step - (total_draws + total_wins)) / conf.test_games_per_step) * 100,
    )
    mlflow.log_metric("Final Learning Rate", final_learning_rate)


def _save_and_plot_results(
    run_var: RunVariableCreator, run_initial_rate: float, test_agent: MonteCarloAgent, conf: Config_2_MC
) -> None:
    """
    Saves results and logs plots to MLflow.

    Args:
        run_var: The RunVariableCreator instance.
        run_initial_rate: The initial learning rate of the run.
        test_agent: The final trained Monte Carlo agent.
        conf: The configuration object.
    """
    save_path = save_path_generator(run_var, run_initial_rate)
    save_results_core(run_var, save_path, run_initial_rate, test_agent, conf)

    plots_figures = plot_step_info(run_var, save_path)
    for title, fig in plots_figures.items():
        mlflow.log_figure(fig, f"{title}.png")


def multi_core_monte_carlo_learning(all_possible_states: list) -> dict:
    """
    Performs Monte Carlo learning on multiple cores using the specified configuration.

    The results of this learning process will be saved in the folder:
        "../results/mc/current/frozen"

    Args:
        all_possible_states: A list of all possible board states for the Tic Tac Toe game.

    Returns:
        A dictionary containing the overall results of the training run.
    """
    conf = Config_2_MC()
    log_named_tuple_as_params(conf)
    _check_pre_run_calculations(conf)

    with Manager() as manager:
        shared_q_values, shared_returns = _initialize_shared_data(manager, conf, all_possible_states)

        run_var = RunVariableCreator(
            all_possible_states,
            {},  # overall results dict
            {},  # The combined q levels for each model (not used with shared dicts)
            0,  # number of episodes trained so far this run
            [],  # Training rate log (games per second across all cores)
        )

        current_learning_rate = conf.learning_rate_start
        games_per_step = int(conf.total_games / conf.steps)

        for episode_num in tqdm(range(1, conf.total_games + 1, games_per_step)):
            current_learning_rate = _run_training_step(
                episode_num,
                conf,
                run_var,
                shared_q_values,
                shared_returns,
                current_learning_rate,
                games_per_step,
                all_possible_states
            )

        # Final testing and logging
        final_test_agent = MonteCarloAgent(0.0, all_possible_states, conf)
        final_test_agent.q_values = shared_q_values
        final_test_agent.returns = shared_returns
        final_wins, final_draws = final_test_agent.test(conf.test_games_per_step, conf.cores)

        _log_final_metrics(conf, final_wins, final_draws, current_learning_rate)
        _save_and_plot_results(run_var, conf.learning_rate_start, final_test_agent, conf)

    return run_var.overall_res
