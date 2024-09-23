
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



from src.control import Config_2_MC
from src.control.setup import pre_run_calculations_tasks
from monte_carlo_learning.flow_control import multi_core_monte_carlo_learning

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
      e_name = conf.experiment_name
      r_name = None
      mlflow.set_experiment(experiment_name = f"{e_name}")
      
      with mlflow.start_run(run_name=f"{e_name}"):
            multi_core_monte_carlo_learning(all_possible_states)

              
if __name__ == "__main__":
   res = main()
