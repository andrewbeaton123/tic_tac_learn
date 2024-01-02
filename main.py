import numpy as np
import logging 
logging.basicConfig(level="DEBUG")
import pickle as pkl
import time
# Monte Carlo Control Agent
from tqdm import tqdm
from control.config_class import ConfigClass
from control.run_variables import RunVariableCreator
from multiprocessing import Pool
from datetime import datetime
from typing import Dict
from src.file_mangement.directory_creator import create_directory
from game.get_all_states import generate_all_states
#from game.test_mc_models import test_agent_tic_tac_toe
from monte_carlo_learning.combine_q_value_dict import update_q_values
from monte_carlo_learning.monte_carlo_tic_tac import MonteCarloAgent
from monte_carlo_learning.super_carlo_tic_tac import SuperCarloAgent
from multi_processing_tools.multi_process_controller import multi_process_controller

def mc_create_run_instance(args) ->(int,Dict):
    agent, episodes_in = args
    agent.train(episodes_in)
    return episodes_in,agent.q_values

def setup_mc_class(args) -> MonteCarloAgent:
    all_states,lr = args
    agent= MonteCarloAgent(lr,all_states)
    #agent.initialize_q_values()
    return agent

def setup_sc_class(args) -> SuperCarloAgent:
    q_values ,lr = args
    return SuperCarloAgent(q_values,lr)

def main():

        #~~~~~~~~~~~~~~~~~~~
        #Overall run settings 
        #~~~~~~~~~~~~~~~~~~~
        
        config = ConfigClass(5,# cores
                             5,#steps per run
                             50, # total runs to create a model from
                             700,#How many games to test with
                             [0.1,0.01,0.001]# learning rates 
                             )
        


        #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
        #End of User editable variables 
        #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
        
        
       #~~~~~~~~~~~~~~~~~~~
        #Create Variables for the run
        #~~~~~~~~~~~~~~~~~~~
        run_var = RunVariableCreator(generate_all_states(),# get a list of all possible board states are tic tac toe game 
                           {}, #overall results dict
                           {},# The combined q levels for each model
                           0,# number of episodes trained so far this run
                           []# Training rate log, how many games per second
                           #across all cores 
                           )
        
        #TODO extract this code out and try and  make a base repeatable 
        
        
        for rate in tqdm(config.learning_rate, colour="green"):
            run_var.last_e_total = 0
            
            # perform training using a single learning rate 
            for episodes in tqdm(range(1,config.total_training_games,config.steps)):#range(100000,1000000,100000):
                
                if not run_var.combined_q_values: 
                    logging.info("Setting all possible states")
                    agents = [setup_mc_class((run_var.all_possible_states,rate))  for core in range (config.cores)]
                else:

                    logging.info(" Loading q values into monte carlo agents")
                    agents = [setup_sc_class((run_var.combined_q_values,rate))  for core in range (config.cores)]
                # Split overall run numbers into checkpoint models 

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
                mc_create_run_instance(configs[0])
                exit()
                multi_core_returns = multi_process_controller(mc_create_run_instance,
                                                                configs,
                                                                config.cores)
                
                t_after_train = time.time()

                time_taken_to_train = round(t_after_train-t_before_train)+1e-9

                games_per_sec= round(config.steps/ time_taken_to_train)

                run_var.training_rate.append(games_per_sec)

                #_-__-__-__-__-__-__-__-__-__-__-_                    
                logging.debug(f"Trained {config.steps} games over {config.cores} cores in {time_taken_to_train} seconds")
                logging.info(f"Training at {round(config.steps/ time_taken_to_train)} g/s")
                logging.debug(f"main - multi core training returned {type(multi_core_returns)}")
                logging.debug(f"main - multi core training single returned {type(multi_core_returns[0])}")
                logging.debug("main- staring q vlaue combination")
                #_-__-__-__-__-__-__-__-__-__-__-_

                
                for mc_return_single in multi_core_returns:

                    episodes, q_values = mc_return_single
                    
                    logging.debug(f"main -q length {len((q_values).keys())}")
                    run_var.combined_q_values = update_q_values(q_values,run_var.combined_q_values)
                    
                #_-__-__-__-__-__-__-__-__-__-__-_            
                logging.debug("main- finshed q vlaue combination")
                #_-__-__-__-__-__-__-__-__-__-__-_
                run_var.last_e_total +=sum([e_s[1] for e_s in configs])

                agent_to_test = SuperCarloAgent(run_var.combined_q_values,0.1)

                #TODO This functionality exists inside the fo the agent already
                total_wins, total_draws = agent_to_test.test(
                                                            config.test_games,
                                                            config.cores)
                
                print(f"For Episodes :{run_var.last_e_total}")

                print(f"Agent won {total_wins} out of {config.test_games} games.")

                print(f"Games drawn {total_draws}")

                run_var.overall_res[run_var.last_e_total] = (rate,total_wins,total_draws)

            save_time= datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

            dir_save = f".//runs//games-{run_var.last_e_total}_learning_rate-{rate}_{save_time}//"

            create_directory(dir_save)

            with open(f"{dir_save}//latest_overall_results_{run_var.last_e_total}_lr_{rate}.pkl", "wb") as f :
                pkl.dump(run_var.overall_res,f)
            
            with open(f"{dir_save}//Combination_super_carlo_{run_var.last_e_total}_lr_{rate}.pkl","wb") as f2:
                pkl.dump(agent_to_test, f2)

        return run_var.overall_res


if __name__ == "__main__":
   res = main()