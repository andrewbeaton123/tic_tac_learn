from src.control import Config_2
from src.game.get_all_states import generate_all_states
from src.control import Config_2

def pre_run_calculations_tasks():
    conf = Config_2()
    conf.pre_run_calculations()
    all_possible_states = generate_all_states()

    return all_possible_states