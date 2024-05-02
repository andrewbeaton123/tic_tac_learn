
from recordclass import RecordClass
from typing import Dict

class RunVariableCreator(RecordClass):
    """Creates the variables that will be used during the run

    Args:
        NamedTuple (_type_): Variables that are needed for the 
        run in  a named tuple 
    """
    all_possible_states: list
    overall_res: Dict
    combined_q_values: Dict
    last_e_total: int
    training_rate: list
