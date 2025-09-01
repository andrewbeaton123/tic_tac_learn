#Module imports
import logging 
import mlflow
import os
from datetime import datetime
from pathlib import Path

#local imports

from tic_tac_learn.src.config_management import ConfigManager
from tic_tac_learn.src.control.setup import pre_run_calculations_tasks
from tic_tac_learn.monte_carlo_learning.flow_control import multi_core_monte_carlo_learning

# confiig basics 
mlflow.set_tracking_uri("http://homelab.mlflow")#("http://192.168.1.159:5000")


# Setup logging
def setup_logging():
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"tic_tac_learn_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging to file: {log_file}")


def main():

      #~~~~~~~~~~~~~~~~~~~
      #Overall run settings for Monte Carlo  
      #~~~~~~~~~~~~~~~~~~~
      #TODO: Test that the models in ml flow predicts work fine with the game.
      #Make new repo that is the game using a trained model to pick the best move
      # Need to think if I should split the game out as its going  to be something I would  need
      # to be consistent for the models to use. The game would then be imported into the training code here 
      # and then would be imported into the place where you can play against the model.


       # Setup logging first
      setup_logging()
    
      config_manager = ConfigManager()
      conf = config_manager.config
      
      # Log which config is being used
      logging.info(f"Using configuration: {os.getenv('TICLEARN_ENV', 'development')}")

      #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
      #End of User editable variables 
      #~~~~~~~~~~~~~~~~~~~-----------------~~~~~~~~~~~~~~~~~~~
      
      
      #TODO extract this code out and try and  make a base repeatable 
      
      mlflow.set_experiment(experiment_name = f"{conf.name}")
      
      with mlflow.start_run(run_name=f"{conf.run_name}"):
            multi_core_monte_carlo_learning(pre_run_calculations_tasks())

              
if __name__ == "__main__":
   res = main()
