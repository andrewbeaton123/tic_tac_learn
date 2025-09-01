from tic_tac_learn.src.config_management import ConfigManager
from tic_tac_toe_game.get_all_states import generate_all_states
def pre_run_calculations_tasks():
    conf = ConfigManager().config
    conf.pre_run_calculations()
    all_possible_states = generate_all_states()

    return all_possible_states