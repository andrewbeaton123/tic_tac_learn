import numpy as np
import logging 
import  os
logging.basicConfig(level="DEBUG")
import random
import pickle as pkl
import time
# Monte Carlo Control Agent
from tqdm import tqdm

from typing import Dict
from game.game_2 import TicTacToe
from multiprocessing import Pool
from datetime import datetime
from src.file_mangement.directory_creator import create_directory
from game.get_all_states import generate_all_states
#TODO move agents into their own files in the src structure
class MonteCarloAgent:
    def __init__(self, epsilon, all_possible_states):
        self.epsilon = epsilon
        self.q_values = {}
        self.returns = {}
        self.all_possible_states = all_possible_states


    def to_serializable(self):
        return {
            'epsilon': self.epsilon,
            'q_values': {str(k): v for k, v in self.q_values.items()},
            'returns': self.returns,
            'all_possible_states': self.all_possible_states,
        }
    def initialize_q_values(self):
        for state in self.all_possible_states:
            state_str = tuple(state.board.flatten())
            valid_moves = state.get_valid_moves()
            self.q_values[state_str] = np.zeros(len(valid_moves)).tolist()

    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(state.get_valid_moves()))
        state_str = tuple(state.board.flatten())
        action_values = self.q_values.get(state_str, np.zeros(len(state.get_valid_moves())))#.tolist()
        return np.argmax(action_values)

    def train_episode(self, env):
        episode_data = []
        episode_returns = []
        
        while not env.is_game_over():
            if env.current_player == 1:
                action = self.epsilon_greedy_policy(env)
            else:
                action = np.random.choice(len(env.get_valid_moves()))
            env.make_move(*env.get_valid_moves()[action])

        state_str = tuple(env.board.flatten())
        episode_data.append((state_str, self.q_values[state_str]))

        for state_str, q_values in episode_data:
            for action, value in enumerate(q_values):
                self.q_values[state_str][action] = sum(
                    episode_returns[i]
                    for i in range(len(episode_returns))
                    if state_str == episode_data[i][0]
                    and episode_data[i][1][action] == value
                )

        return self.q_values, self.returns

    def train(self, episodes):
        for episode in range(episodes):
            self.train_episode(TicTacToe(random.choice([1,2])))
        # Test your training logic...
        
    def test(self):
        wins = 0
        draws = 0
        for _ in range(10000):
            env = TicTacToe(random.choice([1, 2]))
            while not env.is_game_over():
                if env.current_player == 1:
                    action = self.epsilon_greedy_policy(env)
                else:
                    action = np.random.choice(len(env.get_valid_moves()))
                env.make_move(*env.get_valid_moves()[action])

            if env.winner == 1:
                wins += 1
            elif env.winner == 0:
                draws += 1
    
        print(f"Agent won {wins} out of 10,000 games.")
        print(f"Draws: {draws}")

class SuperCarloAgent(MonteCarloAgent):
    def __init__(self,q_values:Dict,epsilon: float):
        self.q_values = q_values
        self.epsilon = epsilon
    def to_serializable(self):
        return {
            'q_values': {str(k): v for k, v in self.q_values.items()},
            'epsilon': self.epsilon
        }
    

#TODO add in the ability to get a random board state in and play the game from there
# this will balance the training set better
#TODO move these 3 controller functions to their own files
def    mc_create_run_instance(args) ->(int,Dict):

    """single run instances of the monte carlo agent training
    takes the number of episodes to train over 
    A list of all possible game states and a learning rate

    Args:
        args (int,list,float): number of episodes, 
        all states of the board, learning rate
       

    Returns:
        Tuple(int,Dict): the number of episodes that were used to train
        resultant q values 
    """
    #print("mc_create_run_instance - run")
    episodes_in,all_states,lr = args
    agent= MonteCarloAgent(lr,all_states)
    agent.initialize_q_values()
    agent.train(episodes_in)
    #rint("mc_create_run_instance - finish")
    return episodes_in,agent.q_values

def test_agent(args:(int ,SuperCarloAgent|MonteCarloAgent)) -> (int,int):
    """_summary_

    Args:
        tests (_type_): _description_
        int (_type_): _description_

    Returns:
        _type_: _description_
    """
    tests,agent= args
    wins , draws = 0 , 0
    for _ in range(tests):
        env = TicTacToe(random.choice([0,1]))
        while not env.is_game_over():
            if env.current_player == 1:
                action = agent.epsilon_greedy_policy(env)
            else:
                action = np.random.choice(len(env.get_valid_moves()))
            env.make_move(*env.get_valid_moves()[action])
        if env.check_winner() == 1:
            wins += 1
        elif env.check_winner() ==0:
            draws +=1
    return wins,draws 

def multi_process_controller(func,configs,cores:int):
    res_retun: list = []
    with Pool(cores) as pool:
        logging.debug("multi process controler thread create")
                     
        for  func_return in pool.imap_unordered(func,configs):
            res_retun.append(func_return)
    return res_retun

#TODO Move this testing to the correct place in src
def test_agent_tic_tac_toe(agent:SuperCarloAgent|MonteCarloAgent
                                  ,num_tests:int =10000,
                                  cores:int=4) -> (int,int):
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
    res_test_games = multi_process_controller(test_agent,test_config,cores)
    for multi_return_single in res_test_games:
        wins_run, draw_run= multi_return_single
        #logging.info(wins_run,draw_run)
        total_wins += wins_run
        total_draws += draw_run
    return total_wins,total_draws


def main():
        
        
        # get a list of all possible board states are tic tac toe game 
        # instances 
        all_possible_states = generate_all_states()
        # create two datastructures that will be used for overall results
        overall_res = {}
        combined_q_values = {}
        last_e_total = 0

        #~~~~~~~~~~~~~~~~~~~
        #Overall run settings 
        #~~~~~~~~~~~~~~~~~~~
        cores= 8
        step = 250000
        total_training_games=1000000
        test_games = 50000
        training_rate = []
        learning_rate = [0.1,0.01, 0.001]

        for rate in tqdm(learning_rate, colour="green"):
            # perform training using a single learning rate 
            for episodes in tqdm(range(1,total_training_games,step)):#range(100000,1000000,100000):
                # Split overall run numbers into checkpoint models 

                logging.debug(f"main - Starting {episodes}")
                
                # steps to be given to each core
                __steps_pc = int(step/cores)
                configs = [(__steps_pc, all_possible_states,rate) for _ in range(cores)]
                #_-__-__-__-__-__-__-__-__-__-__-_
                logging.debug("main - Finished generating configs ")
                logging.debug(f"main - cofig length is : {len(configs)}")
                logging.debug(f"main - episodes configs are {[e_s[0] for e_s in configs]}")
                #_-__-__-__-__-__-__-__-__-__-__-_
                if [e_s[0] for e_s in configs] != [0,0,0]:
                    t_before_train = time.time()
                    multi_core_returns = multi_process_controller(mc_create_run_instance,configs,cores)
                    t_after_train = time.time()
                    time_taken_to_train = round(t_after_train-t_before_train)
                    games_per_sec= round(step/ time_taken_to_train)
                    training_rate.append(games_per_sec)

                    #_-__-__-__-__-__-__-__-__-__-__-_                    
                    logging.debug(f"Trained {step} games over {cores} cores in {time_taken_to_train} seconds")
                    logging.info(f"Training at {round(step/ time_taken_to_train)} g/s")
                    logging.debug(f"main - multi core training returned {type(multi_core_returns)}")
                    logging.debug(f"main - multi core training single returned {type(multi_core_returns[0])}")
                    logging.debug("main- staring q vlaue combination")
                    #_-__-__-__-__-__-__-__-__-__-__-_
                    for mc_return_single in multi_core_returns:
                        episodes, q_values = mc_return_single
                        logging.debug(f"main -q length {len((q_values).keys())}")
                        for state_str, values in q_values.items():
                            if state_str not in combined_q_values:
                                combined_q_values[state_str] = np.array(values).astype("float64")
                            else:
                                combined_q_values[state_str] += np.array(values).astype("float64")
                    #_-__-__-__-__-__-__-__-__-__-__-_            
                    logging.debug("main- finshed q vlaue combination")
                    #_-__-__-__-__-__-__-__-__-__-__-_
                    last_e_total +=sum([e_s[0] for e_s in configs])
                agent_to_test = SuperCarloAgent(combined_q_values,0.1)

                total_wins, total_draws = test_agent_tic_tac_toe(agent_to_test,test_games,cores)
                print(f"For Episodes :{last_e_total}")

                print(f"Agent won {total_wins} out of {test_games} games.")
                print(f"Games drawn {total_draws}")
                overall_res[episodes] = (rate,total_wins,total_draws)
            save_time= datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            dir_save = f".//runs//games-{last_e_total}_learning_rate-{rate}_{save_time}//"
            create_directory(dir_save)
            with open(f"{dir_save}//latest_overall_results_{last_e_total}_lr_{rate}.pkl", "wb") as f :
                pkl.dump(overall_res,f)
            
            with open(f"{dir_save}//Combination_super_carlo_{last_e_total}_lr_{rate}.pkl","wb") as f2:
                pkl.dump(agent_to_test, f2)

        return overall_res


if __name__ == "__main__":
   res = main()