#Module imports
import logging 
import sys

logging.basicConfig(level="INFO")
# Configure logging to ensure DEBUG messages are shown
log_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

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
from tic_tac_learn.control.factory import load_config as load_mc_config
from tic_tac_learn.monte_carlo_learning.flow_control.run_monte_carlo import run_parallel_training

# confiig basics 
mlflow.set_tracking_uri("http://homelab.mlflow")


def load_config(config_path: str) -> dict:
    """Loads a YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main execution function."""
    # 1. Load Configuration from file
    logging.info("Loading configuration from config.yml...")

    config_path ='tic_tac_learn/config.yml'
    conf = load_mc_config(config_path)

    mlflow.set_experiment(experiment_name=conf.runner.experiment_name)
    with mlflow.start_run(run_name=conf.runner.run_name) as run:
        logging.info(f"Configuration loaded for run: {conf.runner.run_name}")
        logging.info(f"MLflow run started (ID: {run.info.run_id})")
        
        # Log all settings
        mlflow.log_params(conf.raw_config.get("monte_carlo_settings", {}))

        # 4. Execute Training
        run_parallel_training(conf)
        
        logging.info("Training run finished successfully.")

if __name__ == "__main__":
   main()
