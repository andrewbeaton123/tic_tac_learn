from typing import Dict
import numpy as np


def  update_q_values(new_q_values: Dict, 
                                        current_q_values: Dict )-> Dict:
                    """Takes in two dicts and creates a new record if it
                    was not in the dict and if it was already present 
                    then sum the values as a float64

                    Args:
                        new_q_values (Dict): dict with the new values to be
                        added to the orginal dict

                        current_q_values (Dict): Existing dict which we want to
                        have the new values added into it

                    Returns:
                        Dict: Combination of the two dicts with creation
                        if the key was not found and summation if it was
                        found
                    """
                    for state_str, values in new_q_values.items():
                        
                        if state_str not in current_q_values:

                            current_q_values[state_str] = np.array(values).astype("float64")
                        else:
                            current_q_values[state_str] += np.array(values).astype("float64")
                    return current_q_values