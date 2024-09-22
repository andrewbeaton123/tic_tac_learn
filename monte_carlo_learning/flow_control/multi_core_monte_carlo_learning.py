#third part imports
import logging
import time
import mlflow
import numpy as np

from tqdm import tqdm

#first part imports 

#learning method imports 
from monte_carlo_learning.combine_q_value_dict import combine_q_values
from monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent

#tooling imports
from src.control import Config_2_MC
from multi_processing_tools.multi_process_controller import multi_process_controller
from src.results_saving.save_controller import save_results_core,save_path_generator
from src.control.mlflow.create_experiment import create_mlflow_experiment
from src.control.mlflow.log_named_tuple_as_params  import log_named_tuple_as_params
from src.result_plotter.plot_step_info import plot_step_info
from src.control.run_variables import RunVariableCreator

#relative  imports 
from .agent_creation import mc_create_run_instance, setup_mc_class


def multi_core_monte_carlo_learning(all_possible_states):
    #{run_name}_starting_lr_{config.learning_rate[0]}_steps_{steps}_total_games_{total_games}
    conf = Config_2_MC
    log_named_tuple_as_params(conf)
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
    
    for rate in tqdm(conf.learning_rate, colour="green"):
    
        run_var.last_e_total = 0
        run_inital_rate : float  = rate
        # perform training using a single learning rate 
        for episodes in tqdm(range(1,conf.total_games,conf.steps)):#range(100000,1000000,100000):
            
            if episodes != 1:
                combined_agent = MonteCarloAgent(rate, all_possible_states)
                combined_agent.load_q_values(combined_q_values)
                agents = [combined_agent for core in range(conf.cores)]
            else :
                logging.info("Setting all possible states")
                #logging.debug(run_var.all_possible_states)
                logging.debug(len(run_var.all_possible_states))
                agents = [setup_mc_class((run_var.all_possible_states,rate)
                                         )  for core in range (conf.cores)]
            

            logging.debug(f"main - Starting {episodes}")
            
            # steps to be given to each core
            __steps_pc = int((conf.steps/conf.cores)+1)
            logging.debug(f"Main - steps are {__steps_pc}")
            configs = [(agents[_c],__steps_pc) for _c in range(conf.cores)]
            
            #_-__-__-__-__-__-__-__-__-__-__-_
            logging.debug("main - Finished generating configs ")
            logging.debug(f"main - cofig length is : {len(configs)}")
            logging.debug(f"main - episodes configs are {[e_s[0] for e_s in configs]}")
            #_-__-__-__-__-__-__-__-__-__-__-_
            
            t_before_train = time.time()

            # learning rate scaling
            if run_var.last_e_total >= conf.learning_rate_flat_games : 
                
                rate -= conf.learning_rate_decay_rate# 0.1 #learning_rate_change*25
                if rate < conf.learning_rate_min: 
                    rate = conf.learning_rate_min
            mlflow.log_metric("Learning Rate", rate)

            logging.info(f"Current learning rate is : {rate}")
            multi_core_returns = multi_process_controller(mc_create_run_instance,
                                                            configs,
                                                            conf.cores)
            
            t_after_train = time.time()

            time_taken_to_train = round(t_after_train - t_before_train)+1e-9

            games_per_sec= round(conf.steps/ time_taken_to_train)

            run_var.training_rate.append(games_per_sec)
            mlflow.log_metric("Games Per Second", games_per_sec)

            #_-__-__-__-__-__-__-__-__-__-__-_                    
            logging.debug(f"Trained {conf.steps} games over {conf.cores} cores in {time_taken_to_train} seconds")
            logging.info(f"Training at {round(conf.steps/ time_taken_to_train)} g/s")
            logging.debug(f"main - multi core training returned {type(multi_core_returns)}")
            logging.debug(f"main - multi core training single returned {type(multi_core_returns[0])}")
            logging.debug("main- staring q vlaue combination")
            #_-__-__-__-__-__-__-__-__-__-__-_

            
            #for mc_return_single in multi_core_returns:

            
            # Combine the Q values from multiple agents
            agents = multi_core_returns # Replace with your actual agents
            combined_q_values = combine_q_values(agents)

            # Load the combined Q values into a new agent
            combined_agent = MonteCarloAgent(rate, all_possible_states)
            combined_agent.load_q_values(combined_q_values)
            #_-__-__-__-__-__-__-__-__-__-__-_            
            logging.debug("main- finshed q vlaue combination")
            #_-__-__-__-__-__-__-__-__-__-__-_
            run_var.last_e_total +=sum([e_s[1] for e_s in configs])

            agent_to_test = combined_agent

            #TODO This functionality exists inside the fo the agent already
            total_wins, total_draws = agent_to_test.test(
                                                        conf.test_games_per_step,
                                                        conf.cores)
            
            print(f"For Episodes :{run_var.last_e_total}")
            
            
            #print(f"Agent won {total_wins} out of {config.test_games} games.")
            print(f"Winrate is {round((total_wins/conf.test_games_per_step)*100)}%")
            print(f"Games drawn {total_draws}")

            run_var.overall_res[run_var.last_e_total] = (rate,
                                                        total_wins,
                                                        total_draws,
                                                        conf.test_games_per_step)
            

        mlflow.log_metric("Final Win Rate", (total_wins/conf.test_games_per_step)*100 )
        mlflow.log_metric("Final Draw Rate", (total_draws/conf.test_games_per_step)*100 )
        mlflow.log_metric("Final Loss Rate", ((conf.test_games_per_step - (total_draws +total_wins))/conf.test_games_per_step)*100 )
        mlflow.log_metric("Final Learning Rate", rate)

        save_path = save_path_generator(run_var, rate)
        save_results_core(run_var,
                        save_path,
                        conf,
                        run_inital_rate,
                        agent_to_test)
        
        plots_figures = plot_step_info(run_var,save_path)

        for title, fig in plots_figures.items():
            mlflow.log_figure(fig,f"{title}.png")
        return run_var.overall_res