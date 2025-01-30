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
mlflow.set_tracking_uri("http://homelab.mlflow")#("http://192.168.1.159:5000")
logging.basicConfig(level="INFO")


def main():

      #~~~~~~~~~~~~~~~~~~~
      #Overall run settings for Monte Carlo  
      #~~~~~~~~~~~~~~~~~~~
      #TODO: Test that the models in ml flow predicts work fine with the game.
      #Make new repo that is the game using a trained model to pick the best move
      # Need to think if I should split the game out as its going  to be something I would  need
      # to be consistent for the models to use. The game would then be imported into the training code here 
      # and then would be imported into the place where you can play against the model.
      conf = Config_2_MC()
      conf.run_name = " Testing agent refactor "
      conf.total_games = int(1e6)
      conf.experiment_name= "Tic Tac Learn"
      conf.steps = 10
      conf.cores= 8
      conf.learning_rate_start= 0.8
      conf.learning_rate_min = 0.01
      conf.learning_rate_scaling = 1.2
      conf.test_games_per_step = 3000
      conf.learning_rate_flat_games = conf.total_games* 0.01

      conf.custom_model_name = "Tiny Test Run dev"
      
      
      
      #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
      #End of User editable variables 
      #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
      
      
      #TODO extract this code out and try and  make a base repeatable 
      
      mlflow.set_experiment(experiment_name = f"{conf.experiment_name}")
      
      with mlflow.start_run(run_name=f"{conf.run_name}"):
            multi_core_monte_carlo_learning(pre_run_calculations_tasks())

              
if __name__ == "__main__":
   res = main()
