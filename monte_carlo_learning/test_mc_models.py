from tic_tac_toe_game.game import TicTacToe
import logging
from main import MonteCarloAgent, SuperCarloAgent,multi_process_controller

def test_agent_tic_tac_toe(agent:SuperCarloAgent|MonteCarloAgent
                            ,num_tests:int =10000
                            ,cores:int=4) -> (int,int):
    """Plays num_tests games of tic tac toe using the monte_carlo
    agent gready process against a random oponent
    

    Args:

        agent (_type_, optional): The monte carlo based agent to play vs the 
        random oponent 
        num_tests int = The number of games to be played Defaults to 10000, 
        cores:int=4 = The number of cores to split the games  across

    Returns:
        Tuple (int, int): total wins by the agent, total draws 
    """
    test_count_pc= int(num_tests/cores)
    logging.debug("testing_agent - starting test for combined q's")
    test_config = [(test_count_pc,agent) for _ in range(cores) ]
    total_wins, total_draws= 0,0
    res_test_games = multi_process_controller(agent,test_config,cores)
    for multi_return_single in res_test_games:
        wins_run, draw_run= multi_return_single
        #logging.info(wins_run,draw_run)
        total_wins += wins_run
        total_draws += draw_run
    return total_wins,total_draws
