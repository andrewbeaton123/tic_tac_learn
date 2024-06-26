from datetime import datetime
import pickle as pkl
import logging
from src.control.run_variables import RunVariableCreator
from src.control.config_class import ConfigClass
from monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent

from src.file_mangement.directory_creator import create_directory


def save_results_core(
                      run_var: RunVariableCreator,
                      rate: float,
                      config: ConfigClass,
                    run_inital_rate: float,
                    agent_to_test : MonteCarloAgent ) -> None : 
            """
            Save the results to files.

            Parameters:
            run_var (RunVariableCreator): The run variable creator object.
            rate (float): The rate value.
            config (ConfigClass): The configuration class object.
            run_inital_rate (float): The initial rate value.
            agent_to_test (MonteCarloAgent): The Monte Carlo agent object.

            Returns:
            None
            """
            save_time : str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            dir_save = f".//runs//games-{run_var.last_e_total}_learning_rate-{rate}_{save_time}//"

            create_directory(dir_save)

            run_metrics_path = f"{dir_save}//{config.run_name}_latest_overall_results_{run_var.last_e_total}_lr_{run_inital_rate}.pkl"
            with open(run_metrics_path, "wb") as f :
                logging.info(f"Run metrics stored in : {run_metrics_path} \n")
                pkl.dump(run_var.overall_res,f)


            combo_model_path =f"{dir_save}//{config.run_name}_Combination_super_carlo_{run_var.last_e_total}_lr_{run_inital_rate}.pkl"
            with open(combo_model_path,"wb") as f2:
                logging.info(f"Combonation model stored in : {combo_model_path} \n")
                pkl.dump(agent_to_test, f2)
