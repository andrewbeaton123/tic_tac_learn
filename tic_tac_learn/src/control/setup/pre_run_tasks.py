from src.control import Config_2_MC
from tic_tac_toe_game.get_all_states import generate_all_states
def pre_run_calculations_tasks():
    conf = Config_2_MC()
    conf.pre_run_calculations()
    all_possible_states = generate_all_states()

    return all_possible_states