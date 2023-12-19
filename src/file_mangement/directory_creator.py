import os 
import logging 
def create_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If it doesn't exist, create it
        os.makedirs(directory_path)
        logging.info(f"Directory '{directory_path}' created.")
    else:
        logging.ERROR(f"Directory '{directory_path}' already exists.")
