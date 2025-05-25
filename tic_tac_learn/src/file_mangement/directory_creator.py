import os 
import logging 
from tic_tac_learn.src.errors import SaveDirectoryAlreadyExistsError
def create_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If it doesn't exist, create it
        os.makedirs(directory_path)
        logging.info(f"Directory '{directory_path}' created.")
    else:
        raise SaveDirectoryAlreadyExistsError(f"Directory '{directory_path}' already exists.")
