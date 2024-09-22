
import logging 
logging.basicConfig(level="INFO")
import pickle as pkl

import mlflow
# Monte Carlo Control Agent
from tqdm import tqdm
from src.control.config_class import ConfigClass
from src.control.run_variables import RunVariableCreator
from multiprocessing import Pool
from datetime import datetime



#from game.test_mc_models import test_agent_tic_tac_toe
# from monte_carlo_learning.combine_q_value_dict import combine_q_values
# from monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent
# from multi_processing_tools.multi_process_controller import multi_process_controller

# from src.results_saving.save_controller import save_results_core,save_path_generator
# from src.control.mlflow.create_experiment import create_mlflow_experiment
# from src.control.mlflow.log_named_tuple_as_params  import log_named_tuple_as_params
# from src.result_plotter.plot_step_info import plot_step_info

from src.control import Config_2_MC
from src.control.setup import pre_run_calculations_tasks

mlflow.set_tracking_uri("http://192.168.1.159:5000")

def main():

        #~~~~~~~~~~~~~~~~~~~
        #Overall run settings for Monte Carlo  
        #~~~~~~~~~~~~~~~~~~~
        conf = Config_2_MC
        conf.total_games = int(2e6)
        conf.steps = 1
        conf.cores= 1
        conf.learning_rate= 0.65
        conf.learning_rate_min = 0.01
        conf.learning_rate_scaling = 1
        conf.learning_rate_flat_games = conf.total_games* 0.2
        
        
        
        #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
        #End of User editable variables 
        #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
        
        
        
        all_possible_states = pre_run_calculations_tasks()
       
        #TODO extract this code out and try and  make a base repeatable 
      
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"{conf.run_name}"):
              pass


if __name__ == "__main__":
   res = main()
