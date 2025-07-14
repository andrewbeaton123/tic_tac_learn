#Module imports
import logging 
import sys

logging.basicConfig(level="DEBUG")
# Configure logging to ensure DEBUG messages are shown
log_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Clear existing handlers to prevent duplicate output
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Add a StreamHandler to output to stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

import mlflow
import yaml

#local imports
from tic_tac_learn.src.control import Config_2_MC
from tic_tac_learn.monte_carlo_learning.flow_control.run_monte_carlo import run_parallel_training

# confiig basics 
mlflow.set_tracking_uri("http://homelab.mlflow")


def load_config(config_path: str) -> dict:
    """Loads a YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main execution function."""
    print("--- Starting main() function ---")

    # 1. Load Configuration from file
    logging.info("Loading configuration from config.yml...")
    print("DEBUG: Before loading config.yml")
    config_data = load_config('config.yml')
    print("DEBUG: After loading config.yml")
    mc_settings = config_data.get('monte_carlo_settings', {})
    print("DEBUG: After getting monte_carlo_settings")

    # 2. Populate the singleton Config object
    print("DEBUG: Before creating Config_2_MC instance")
    conf = Config_2_MC()
    print("DEBUG: After creating Config_2_MC instance")
    conf.load_from_dict(mc_settings)
    print("DEBUG: After loading from dict")
    conf.pre_run_calculations() # Ensure calculated properties are set
    print("DEBUG: After pre_run_calculations")
    logging.info(f"Configuration loaded for run: {conf.run_name}")

    # 3. Set up MLflow Experiment
    print("DEBUG: Before setting MLflow experiment")
    mlflow.set_experiment(experiment_name=conf.experiment_name)
    print("DEBUG: After setting MLflow experiment")

    # 4. Start the MLflow run and execute training
    print("DEBUG: Before starting MLflow run")
    with mlflow.start_run(run_name=conf.run_name) as run:
        print("DEBUG: Inside MLflow run context")
        logging.info(f"MLflow run started (ID: {run.info.run_id})")
        mlflow.log_params(mc_settings) # Log all the settings
        
        # This is the main call to our new orchestration function
        print("DEBUG: Before calling run_parallel_training")
        run_parallel_training(conf)
        print("DEBUG: After calling run_parallel_training")
        
        logging.info("Training run finished successfully.")

    print("--- Exiting main() function ---")

if __name__ == "__main__":
   main()
