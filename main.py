#Module imports
import logging 
import mlflow



#local imports

from tic_tac_learn.src.control import Config_2_MC
from tic_tac_learn.src.control.setup import pre_run_calculations_tasks
from tic_tac_learn.monte_carlo_learning.flow_control import multi_core_monte_carlo_learning

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
      conf.run_name = "Prod - 10e6 - draw negative reward - larger lr scaling low lr start "
      conf.total_games = int(10e6)
      conf.experiment_name= "Tic Tac Learn"
      conf.steps = 20
      conf.cores= 8
      conf.learning_rate_start= 0.2
      conf.learning_rate_min = 0.01
      conf.learning_rate_scaling = 5.5
      conf.test_games_per_step = 3000
      conf.learning_rate_flat_games = conf.total_games* 0.01

      conf.custom_model_name = "Final Model as artifact"
      
      
      
      #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
      #End of User editable variables 
      #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
      
      
      #TODO extract this code out and try and  make a base repeatable 
      
      mlflow.set_experiment(experiment_name = f"{conf.experiment_name}")
      
      with mlflow.start_run(run_name=f"{conf.run_name}"):
            multi_core_monte_carlo_learning(pre_run_calculations_tasks())

              
if __name__ == "__main__":
   res = main()
