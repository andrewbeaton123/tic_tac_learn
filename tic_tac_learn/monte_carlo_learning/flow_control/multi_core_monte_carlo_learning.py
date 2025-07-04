#third part imports
import logging
import time
import mlflow
from tqdm import tqdm

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

#relative  imports 
from .agent_creation import mc_create_run_instance, setup_mc_class
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
        if episodes != 1:
            if conf.agent_reload == None:
                #Normal processing for learning by loading in stats and q values
                combined_agent = MonteCarloAgent(rate, run_var.all_possible_states)
                combined_agent.load_q_values(combined_q_values)
                combined_agent.generate_returns_space_only()
                agents = [combined_agent for core in range(conf.cores)]
            else:
                #Call a function that loads an agent from pickle 
                agents = [conf.agent_reload for core in range(conf.cores)]
        else :
            #specific setup for first time run 
            logging.debug(len(run_var.all_possible_states))
            agents = [setup_mc_class((run_var.all_possible_states,rate)
                                        )  for core in range (conf.cores)]
        

        logging.debug(f"main - Starting {episodes}")
        # steps to be given to each core
        __steps_pc: int = int((games_per_step/conf.cores)+1)
        configs = [(agents[_c],__steps_pc) for _c in range(conf.cores)]
        

        #_-__-__-__-__-__-__-__-__-__-__-_
        logging.debug(f"Main - steps per core are  {__steps_pc}")
        logging.debug("main - Finished generating configs ")
        logging.debug(f"main - cofig length is : {len(configs)}")
        logging.debug(f"main - episodes configs are {[e_s[0] for e_s in configs]}")
        #_-__-__-__-__-__-__-__-__-__-__-_
        
       

        # learning rate scaling
        rate = learning_rate_scaling(rate,run_var.last_e_total,episodes)
        

        logging.info(f"Current learning rate is : {rate}")
        multi_core_returns = multi_process_controller(mc_create_run_instance,
                                                        configs,
                                                        conf.cores)
        
        t_after_train = time.time()

        time_taken_to_train = round(t_after_train - t_before_train,6)+1e-9
        

        games_per_sec: int = round(games_per_step/ time_taken_to_train)

        run_var.training_rate.append(games_per_sec)

        #_-__-__-__-__-__-__-__-__-__-__-_                    
        logging.debug(f"Trained {games_per_step} games over {conf.cores} cores in {time_taken_to_train} seconds")
        logging.info(f"Training at {games_per_sec} g/s")
        logging.debug(f"main - multi core training returned {type(multi_core_returns)}")
        logging.debug(f"main - multi core training single returned {type(multi_core_returns[0])}")
        logging.debug("main- staring q vlaue combination")
        #_-__-__-__-__-__-__-__-__-__-__-_

        
        

        
        # Combine the Q values from multiple agents
        agents = multi_core_returns # Replace with your actual agents
        combined_q_values = combine_returns_max(agents)#combine_q_values(agents)

        # Load the combined Q values into a new agent
        combined_agent = MonteCarloAgent(rate, run_var.all_possible_states)
        combined_agent.load_q_values(combined_q_values)
        
        # setting epsilon to 0 so that there ar eno random moves in the test 
        test_agent = MonteCarloAgent(0.0, run_var.all_possible_states)
        test_agent.load_q_values(combined_q_values)
        #_-__-__-__-__-__-__-__-__-__-__-_            
        logging.debug("main- finshed q vlaue combination")
        #_-__-__-__-__-__-__-__-__-__-__-_
        run_var.last_e_total +=sum([e_s[1] for e_s in configs])

        
        #TODO This functionality exists inside the fo the agent already
        total_wins, total_draws = test_agent.test(
                                                    conf.test_games_per_step,
                                                    conf.cores)
        
        print(f"For Episodes :{run_var.last_e_total}")
        
        
        #print(f"Agent won {total_wins} out of {config.test_games} games.")
        print(f"Winrate is {round((total_wins/conf.test_games_per_step)*100)}%")
        print(f"Games drawn {total_draws}")
        #logs the 
        log_in_progress_mc_model(combined_agent, 
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
                    combined_agent,
                    conf)
    
    plots_figures = plot_step_info(run_var,save_path)

    for title, fig in plots_figures.items():
        mlflow.log_figure(fig,f"{title}.png")
    return run_var.overall_res