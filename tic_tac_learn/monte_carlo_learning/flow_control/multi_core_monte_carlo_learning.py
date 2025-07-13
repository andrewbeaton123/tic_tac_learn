#third part imports
import logging
import time
import mlflow
from tqdm import tqdm
from multiprocessing import Manager

#first part imports 

#learning method imports 
from tic_tac_learn.monte_carlo_learning.combine_q_value_dict import combine_q_values,combine_returns_max
from tic_tac_learn.monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent

#tooling imports
from tic_tac_learn.src.control import Config_2_MC
from tic_tac_learn.multi_processing_tools.multi_process_controller import multi_process_controller
from tic_tac_learn.src.results_saving.save_controller import save_results_core,save_path_generator
from tic_tac_learn.src.control.mlflow.create_experiment import create_mlflow_experiment
from tic_tac_learn.src.control.mlflow.log_named_tuple_as_params  import log_named_tuple_as_params
from tic_tac_learn.src.result_plotter.plot_step_info import plot_step_info
from tic_tac_learn.src.control.run_variables import RunVariableCreator
from tic_tac_learn.multi_processing_tools.game_batch_processor import process_game_batch

#relative  imports 
from ..learning_rate_scaling import learning_rate_scaling

#tracking imports 
from tic_tac_learn.monte_carlo_learning.tracking_tools import log_in_progress_mc_model
import tic_tac_learn.src.errors as errors

def mc_computed_check(conf: Config_2_MC):
    """
    This function is used to check for pre-run calculations, in order to run the code. 
    """   
    if conf.frozen_learning_rate_steps == None:
        raise errors.PreRunCalculationsNotComplete("Pre run calculations have not been  perfomred !")

def multi_core_monte_carlo_learning(all_possible_states):
    """
    This function is used to perform monte carlo learning on multiple cores using the 
    config file "Config_2_MC". It is not a general purpose function and can only be used with this 
    specific config file.  The results of this learning process will be saved in the folder:
        "../results/mc/current/frozen"
    """   
    conf = Config_2_MC()
    log_named_tuple_as_params(conf)
    mc_computed_check(conf)

    with Manager() as manager:
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

        # The main_agent will now use these pre-populated shared dictionaries
        main_agent = MonteCarloAgent(conf.learning_rate_start, all_possible_states, conf)
        main_agent.q_values = shared_q_values
        main_agent.returns = shared_returns

        #~~~~~~~~~~~~~~~~~~~
            #Create Variables for the run
            #~~~~~~~~~~~~~~~~~~~
        run_var = RunVariableCreator(all_possible_states,# get a list of all possible board states are tic tac toe game 
                            {}, #overall results dict
                            {},# The combined q levels for each model
                            0,# number of episodes trained so far this run
                            []# Training rate log, how many games per second
                            #across all cores 
                            )
        
       
        
        run_var.last_e_total: int  = 0
        run_inital_rate : float  = conf.learning_rate_start
        rate : float  = conf.learning_rate_start
        games_per_step: int = int(conf.total_games /conf.steps)

        #break the overall game count into steps 
        for episodes in tqdm(range(1,conf.total_games,games_per_step)):
            t_before_train = time.time()
            
            # Prepare configurations for multiprocessing
            configs = [(rate, shared_q_values, shared_returns, conf,  int(games_per_step/conf.cores) , all_possible_states) for _ in range(conf.cores)]
            
            logging.info(f"Current learning rate is : {rate}")
            multi_core_returns = multi_process_controller(process_game_batch,
                                                            configs,
                                                            conf.cores)
            
            t_after_train = time.time()

            time_taken_to_train = round(t_after_train - t_before_train,6)+1e-9
            

            games_per_sec: int = round(games_per_step/ time_taken_to_train)

            run_var.training_rate.append(games_per_sec)

            logging.debug(f"Trained {games_per_step} games over {conf.cores} cores in {time_taken_to_train} seconds")
            logging.info(f"Training at {games_per_sec} g/s")

            run_var.last_e_total += games_per_step

            # setting epsilon to 0 so that there ar eno random moves in the test 
            test_agent = MonteCarloAgent(0.0, all_possible_states, conf)
            test_agent.q_values = shared_q_values
            test_agent.returns = shared_returns # Not strictly needed for testing, but good for consistency

            total_wins, total_draws = test_agent.test(
                                                        conf.test_games_per_step,
                                                        conf.cores)
            
            print(f"For Episodes :{run_var.last_e_total}")
            
            print(f"Winrate is {round((total_wins/conf.test_games_per_step)*100)}%")
            print(f"Games drawn {total_draws}")
            #logs the 
            log_in_progress_mc_model(test_agent, 
                                     run_var.last_e_total, 
                                     bool(run_var.last_e_total>= conf.total_games) )
            
            run_var.overall_res[run_var.last_e_total] = (rate,
                                                        total_wins,
                                                        total_draws,
                                                        conf.test_games_per_step)
            
            mlflow.log_metric("In Progress Win Rate", (total_wins/conf.test_games_per_step)*100 , step = episodes)
            mlflow.log_metric("In Progress Draw Rate", (total_draws/conf.test_games_per_step)*100 , step = episodes)
            mlflow.log_metric("In Progress Loss Rate", ((conf.test_games_per_step - (total_draws +total_wins))/conf.test_games_per_step)*100, step = episodes )
            mlflow.log_metric("In Progress Games Per Second",games_per_sec, step = episodes)
            
        mlflow.log_metric("Final Win Rate", (total_wins/conf.test_games_per_step)*100 )
        mlflow.log_metric("Final Draw Rate", (total_draws/conf.test_games_per_step)*100 )
        mlflow.log_metric("Final Loss Rate", ((conf.test_games_per_step - (total_draws +total_wins))/conf.test_games_per_step)*100 )
        mlflow.log_metric("Final Learning Rate", rate)

        save_path = save_path_generator(run_var, rate)
        save_results_core(run_var,
                        save_path,
                        run_inital_rate,
                        test_agent,
                        conf)
        
        plots_figures = plot_step_info(run_var,save_path)

        for title, fig in plots_figures.items():
            mlflow.log_figure(fig,f"{title}.png")
        return run_var.overall_res