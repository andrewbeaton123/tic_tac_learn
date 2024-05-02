from typing import Dict
import numpy as np
import logging


def combine_q_values(agents):
    combined_q_values = {}
    
    # Initialize combined_q_values with the structure of q_values from the first agent
    for state, actions in agents[0].q_values.items():
        combined_q_values[state] = {}
        for action, value in actions.items():
            combined_q_values[state][action] = 0

    # Sum the Q values from all agents
    for agent in agents:
        for state, actions in agent.q_values.items():
            for action, value in actions.items():
                combined_q_values[state][action] += value

    # Divide by the number of agents to get the average
    num_agents = len(agents)
    for state, actions in combined_q_values.items():
        for action, value in actions.items():
            combined_q_values[state][action] /= num_agents

    return combined_q_values





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
                    res_q_values = current_q_values.copy()
                    for state_str, values in new_q_values.items():
                        
                        if state_str not in res_q_values:

                            res_q_values[state_str] = np.array(values)#.astype("float64")
                        else:
                            #logging.debug(f"update q values - Adding new values  {np.array(values)}")
                            #logging.debug(f"update q values - Current values {res_q_values[state_str]}")
                            
                            # Assuming res_q_values[state_str] is of shape (4,) and np.array(values) is of shape (6,)
                             # Pad the shorter array with zeros to make them compatible
                            if res_q_values[state_str].shape[0] < np.array(values).shape[0]:
                                res_q_values[state_str] = np.pad(res_q_values[state_str], (0,
                                                                                            np.array(values).shape[0] - res_q_values[state_str].shape[0]
                                                                                            ), mode='constant')
                                pad_corrected_q =values
                            elif  np.array(values).shape[0] < res_q_values[state_str].shape[0] :
                                
                                pad_corrected_q= np.pad(np.array(values), (0,
                                                        res_q_values[state_str].shape[0] - np.array(values).shape[0]
                                                         ), mode='constant')
                            else:
                                pad_corrected_q =  values
                                

                            res_q_values[state_str] =np.add(res_q_values[state_str] ,
                                                        np.array(pad_corrected_q))#.astype("float64")) 
                    return res_q_values