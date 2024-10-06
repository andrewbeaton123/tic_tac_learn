#Module imports
import logging 
import pickle as pkl
import mlflow




#Part module imports
from tqdm import tqdm
from multiprocessing import Pool
from datetime import datetime

#local imports
from src.control.config_class import ConfigClass
from src.control.run_variables import RunVariableCreator
from src.control import Config_2_MC
from src.control.setup import pre_run_calculations_tasks
from monte_carlo_learning.flow_control import multi_core_monte_carlo_learning

# confiig basics 
mlflow.set_tracking_uri("http://192.168.1.159:5000")
logging.basicConfig(level="INFO")


def main():

      #~~~~~~~~~~~~~~~~~~~
      #Overall run settings for Monte Carlo  
      #~~~~~~~~~~~~~~~~~~~
      conf = Config_2_MC()
      conf.run_name = " Testing Monte Carlo Agent Refactor"
      conf.total_games = int(2e6)
      conf.steps = 50
      conf.cores= 12
      conf.learning_rate_start= 0.8
      conf.learning_rate_min = 0.01
      conf.learning_rate_scaling = 1.2
      conf.test_games_per_step = 3000
      conf.learning_rate_flat_games = conf.total_games* 0.01
      
      
      
      #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
      #End of User editable variables 
      #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
      
      
      all_possible_states = pre_run_calculations_tasks()
      
      #TODO extract this code out and try and  make a base repeatable 
      
      mlflow.set_experiment(experiment_name = f"{conf.experiment_name}")
      
      with mlflow.start_run(run_name=f"{conf.run_name}"):
            multi_core_monte_carlo_learning(all_possible_states)

              
if __name__ == "__main__":
   res = main()
