import numpy as np
import logging 
logging.basicConfig(level="INFO")
import pickle as pkl
import time
import mlflow
# Monte Carlo Control Agent
from tqdm import tqdm
from src.control.config_class import ConfigClass
from src.control.run_variables import RunVariableCreator
from multiprocessing import Pool
from datetime import datetime
from typing import Dict

from src.game.get_all_states import generate_all_states
#from game.test_mc_models import test_agent_tic_tac_toe
from monte_carlo_learning.combine_q_value_dict import combine_q_values
from monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent
from multi_processing_tools.multi_process_controller import multi_process_controller

from src.results_saving.save_controller import save_results_core
from src.control.mlflow.create_experiment import create_mlflow_experiment
from src.control.mlflow.log_named_tuple_as_params  import log_named_tuple_as_params

mlflow.set_tracking_uri("http://192.168.1.159:5000")
def mc_create_run_instance(args) -> tuple[int,Dict]:
    agent, episodes_in = args
    agent.train(episodes_in)
    return agent

def setup_mc_class(args) -> MonteCarloAgent:
    all_states,lr = args
    agent= MonteCarloAgent(lr,all_states)
    #agent.initialize_q_values()
    return agent


def main():

        #~~~~~~~~~~~~~~~~~~~
        #Overall run settings 
        #~~~~~~~~~~~~~~~~~~~
        total_games = 1200000
        steps = 30.0
        config = ConfigClass(4,# cores
                             round(total_games/steps),#steps per run

                             total_games, # total runs to create a model from
                             9508,#How many games to test with
                             [0.9],# learning rates 
                            "Pre_training_test",
                            round(0.86/steps,4)#"reduced decay rate and lower bounds for LR_min_0_01_subing_0.001"
                             )
        


        #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
        #End of User editable variables 
        #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
        
        all_possible_states = generate_all_states()
        # Generate all possible states
        #all_possible_states = 
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
        
        #TODO extract this code out and try and  make a base repeatable 
        experiment_name  = "tic tac toe learning - Work Show"
        mlflow.set_experiment(experiment_name)
        run_name = "ml_flow_testing"
        with mlflow.start_run(run_name=f"{run_name}_starting_lr_{config.learning_rate[0]}_steps_{steps}_total_games_{total_games}"):
            log_named_tuple_as_params(config)
            for rate in tqdm(config.learning_rate, colour="green"):
            
                run_var.last_e_total = 0
                run_inital_rate : float  = rate
                # perform training using a single learning rate 
                for episodes in tqdm(range(1,config.total_training_games,config.steps)):#range(100000,1000000,100000):
                    
                    if episodes != 1:
                        combined_agent = MonteCarloAgent(rate, all_possible_states)
                        combined_agent.load_q_values(combined_q_values)
                        agents = [combined_agent for core in range(config.cores)]
                    else :
                        logging.info("Setting all possible states")
                        #logging.debug(run_var.all_possible_states)
                        logging.debug(len(run_var.all_possible_states))
                        agents = [setup_mc_class((run_var.all_possible_states,rate))  for core in range (config.cores)]
                    

                    logging.debug(f"main - Starting {episodes}")
                    
                    # steps to be given to each core
                    __steps_pc = int((config.steps/config.cores)+1)
                    logging.debug(f"Main - steps are {__steps_pc}")
                    configs = [(agents[_c],__steps_pc) for _c in range(config.cores)]
                    
                    #_-__-__-__-__-__-__-__-__-__-__-_
                    logging.debug("main - Finished generating configs ")
                    logging.debug(f"main - cofig length is : {len(configs)}")
                    logging.debug(f"main - episodes configs are {[e_s[0] for e_s in configs]}")
                    #_-__-__-__-__-__-__-__-__-__-__-_
                    
                    t_before_train = time.time()
                    
                    # learning rate scaling
                    if run_var.last_e_total != 0 : 
                        
                        rate -= config.lr_decay# 0.1 #learning_rate_change*25
                        if rate < 0.03: 
                            rate = 0.03


                    logging.info(f"Current learning rate is : {rate}")
                    multi_core_returns = multi_process_controller(mc_create_run_instance,
                                                                    configs,
                                                                    config.cores)
                    
                    t_after_train = time.time()

                    time_taken_to_train = round(t_after_train - t_before_train)+1e-9

                    games_per_sec= round(config.steps/ time_taken_to_train)

                    run_var.training_rate.append(games_per_sec)

                    #_-__-__-__-__-__-__-__-__-__-__-_                    
                    logging.debug(f"Trained {config.steps} games over {config.cores} cores in {time_taken_to_train} seconds")
                    logging.info(f"Training at {round(config.steps/ time_taken_to_train)} g/s")
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
                                                                config.test_games,
                                                                config.cores)
                    
                    print(f"For Episodes :{run_var.last_e_total}")
                    
                    
                    #print(f"Agent won {total_wins} out of {config.test_games} games.")
                    print(f"Winrate is {round((total_wins/config.test_games)*100)}%")
                    print(f"Games drawn {total_draws}")

                    run_var.overall_res[run_var.last_e_total] = (rate,
                                                                total_wins,
                                                                total_draws,
                                                                config.test_games)
                mlflow.log_metric("Final Win Rate", (total_wins/config.test_games)*100 )
                mlflow.log_metric("Final Draw Rate", (total_draws/config.test_games)*100 )
                mlflow.log_metric("Final Loss Rate", ((config.test_games - (total_draws +total_wins))/config.test_games)*100 )
                mlflow.log_metric("Final Learning Rate", rate)
                save_results_core(run_var,
                                rate,
                                config,
                                run_inital_rate,
                                agent_to_test)

        return run_var.overall_res


if __name__ == "__main__":
   res = main()