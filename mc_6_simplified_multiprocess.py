import numpy as np
import logging 
import  pickle as pkl
logging.basicConfig(level="DEBUG")
import random
import logging 
import pickle as pkl
# Monte Carlo Control Agent
from itertools import product
from typing import Dict
from game.game_2 import TicTacToe
from multiprocessing import Pool
import time 
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
    


def    mc_create_run_instance(args):
        print("mc_create_run_instance - run")
        episodes_in,all_states = args
        agent= MonteCarloAgent(0.1,all_states)
        agent.initialize_q_values()
        agent.train(episodes_in)
        print("mc_create_run_instance - finish")
        return episodes_in,agent.q_values

def main():
        def generate_all_states():
            states = []
            logging.debug("Geenerate_all_states  is starting ")
            for player_marks in product([0, 1, 2], repeat=9):  # 0 represents an empty cell
                board = np.array(player_marks).reshape((3, 3))
                tic_tac_toe_instance = TicTacToe(1,board)
                states.append(tic_tac_toe_instance)
            logging.debug("Geenerate_all_states  is finishing ")
            return states
        

        all_possible_states = generate_all_states()
        
        
        logging.debug(f"main - length of all states is {len(all_possible_states)}")
        """  def create_run_mc(runs:int):
            mc_objects =( mc_create_run_instance() for r in range(runs) )
            return mc_objects
        """
        overall_res = {}
        combined_q_values = {}
        cores= 3
        last_e_total = 0
        for episodes in range(100000,1000000,50000):#range(100000,1000000,100000):
            logging.debug(f"main - Starting {episodes}")
            
            new_episodes  = episodes - last_e_total
            core_episodes =   int(new_episodes/cores)
            configs = [(core_episodes, all_possible_states) for _ in range(cores)]
            logging.debug("main - Finished generating configs ")
            #runs = create_run_mc(10)
            logging.debug(f"main - cofig length is : {len(configs)}")
            logging.debug(f"main - episodes configs are {[e_s[0] for e_s in configs]}")
            if [e_s[0] for e_s in configs] != [0,0,0]:
                start = time.process_time()
                with Pool(cores) as pool: 
                    
                    logging.debug("main - within pool")
                    results = pool.imap_unordered(mc_create_run_instance,configs)
                    logging.debug("main - finished pool ")
                    logging.debug(type(results))
                    logging.debug(results._index)

                    
                    for runs, q_values in results:
                        print("")
                    print(f"Episodes time taken {time.process_time() - start}")
                # Combine Q-values
                
                logging.debug("main- staring q vlaue combination")
                
                for episodes, q_values in results:
                    logging.debug(f"main -q length {len((q_values).keys())}")
                    for state_str, values in q_values.items():
                        if state_str not in combined_q_values:
                            combined_q_values[state_str] = np.array(values)
                        else:
                            combined_q_values[state_str] += np.array(values)
                logging.debug("main- fished q vlaue combination")
                last_e_total +=sum([e_s[0] for e_s in configs])
            agent = SuperCarloAgent(combined_q_values,0.1)
            wins = 0
            draws= 0
            logging.debug("main - starting test for combined q's")
            for _ in range(10000):
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

            print(f"For Episodes : {episodes}")

            print(f"Agent won {wins} out of 100000 games.")
            print(f"Games drawn {draws}")
            overall_res[episodes] = (wins,draws)
            
        with open("latest_overall_results.pkl", "wb") as f :
            pkl.dump(overall_res,f)
        
        with open("Combination_super_carlo.pkl","wb") as f2:
            pkl.dump(agent, f2)

        return overall_res


if __name__ == "__main__":
   res = main()