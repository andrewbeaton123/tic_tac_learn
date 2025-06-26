from typing import Dict
import numpy as np
import logging


def combine_q_values(agents):

    combined_q_values ={}
    # Initialize combined_q_values with the structure of q_values from the first agent
    # For example, if q_values = {a: [1, 2], b: [3, 4]},
    # then combined_q_values = {a: [0, 0], b: [0, 0]}

    for state, actions in agents[0].q_values.items():
        combined_q_values[state] = {}

        # Initialize all values to zero
        for action, value in actions.items():
            combined_q_values[state][int(action)] = 0


    # Sum the Q values from all agents
    # For example, if q_values = {a: [1, 2], b: [3, 4]},
    # then combined_q_values = {a: [1, 2], b: [3, 4]}

    for agent in agents:
        for state, actions in agent.q_values.items():
            for action, value in actions.items():
                combined_q_values[state][int(action)] += value


    # Divide by the number of agents to get the average
    num_agents = len(agents)

    # For example, if q_values = {a: [1, 2], b: [3, 4]},
    # then combined_q_values = {a: [0.5, 1], b: [1.5, 2]}

    for state, actions in combined_q_values.items():
        for action, value in actions.items():
            combined_q_values[state][int(action)] /= num_agents

    return combined_q_values

def combine_returns(agents):
    combined_returns = {}
    for agent in agents:
        for (state, action), returns_list in agent.returns.items():
            key = (tuple(int(x) for x in state), int(action))
            if key not in combined_returns:
                combined_returns[key] = []
            combined_returns[key].extend(returns_list)
    # Now compute Q-values as the mean of returns
    combined_q_values = {}
    for (state, action), returns_list in combined_returns.items():
        if state not in combined_q_values:
            combined_q_values[state] = {}
        combined_q_values[state][action] = np.mean(returns_list) if returns_list else 0.0
    return combined_q_values

def combine_returns_max(agents):
    combined_q_values = {}
    for agent in agents:
        for state, actions in agent.q_values.items():
            if state not in combined_q_values:
                combined_q_values[state] = {}
            for action, value in actions.items():
                action = int(action)
                if action not in combined_q_values[state]:
                    combined_q_values[state][action] = value
                else:
                    combined_q_values[state][action] = max(combined_q_values[state][action], value)
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