
from multiprocessing import Pool
import logging

def multi_process_controller(func,configs,cores:int):
    """
    A function that executes a given function in parallel using multiple processes.

    Parameters:
    - func: The function to be executed in parallel.
    - configs: A list of configurations or inputs for the function.
    - cores: The number of processes to be used for parallel execution.

    Returns:
    - A list of results returned by the function for each configuration.
    """
    res_retun: list = []
    logging.debug(f"multi_process_controller- cores are {cores}")
    logging.debug(f"multi_process_controller- configs are {configs}")
    logging.debug(f"multi_process_controller- function is {func}")

    
    with Pool(cores) as pool:
        logging.debug("multi process controler thread create")
                     
        for  func_return in pool.imap_unordered(func,configs):
            res_retun.append(func_return)
    return res_retun