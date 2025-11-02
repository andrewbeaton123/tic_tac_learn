import logging
import mlflow


from tic_tac_learn.game_interfaces.tic_tac_toe_game_interface import TicTacToeGameInterface
from collections import namedtuple

# Define the named tuple
Config_2_MC = namedtuple(
    "Config_2_MC",
    [
        "run_name",
        "game_interface",
        "total_games",
        "experiment_name",
        "steps",
        "cores",
        "learning_rate_start",
        "learning_rate_end",
        "epsilon_start",
        "epsilon_end",
        "reward_draw",
        "q_table_path",
        "save_q_table_steps",
        "save_q_table_final",
        "save_results_video",
        "save_results_plot",
        "log_mlflow",
        "mlflow_tracking_uri",
        "video_dir",
        "plot_dir",
        "q_table_dir",
        "log_dir",
        "video_callable_frequency",
        "test_game_frequency",
        "test_games_n",
        "test_game_interface",
        "test_epsilon",
        "log_file",
        "merge_strategy",
        "learning_rate_flat_games",
        "frozen_learning_rate_steps"
    ],
)

class Config_2_MC:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config_2_MC, cls).__new__(cls, *args, **kwargs)

            cls._instance.experiment_name = "Tic Tac Learn Default Config"
            cls._instance.run_name = "Default Run"
            cls._instance.total_games = 200
            cls._instance.steps = 10
            cls._instance.cores = 1
            cls._instance._learning_rate_start = 1
            cls._instance._learning_rate_min = 0.01
            cls._instance._learning_rate_scaling = 1

            cls._instance._test_games_per_step = 1000
            cls._instance._discount_factor = 0.9
            cls._instance._exploration_rate = 0.1


            cls._instance.frozen_learning_rate_steps = None 
            cls._instance._games_per_step = None
            cls._instance._learning_rate_decay_rate = None
            cls._instance.learning_rate_flat_games: int|None   = None
            cls._instance._agent_reload = None
            cls._instance.custom_model_name = "DefaultModelName" 

        
        return cls._instance
    
    def get_allowed_players(self) -> tuple[int, int]:
        return (1, 2)

    def load_from_dict(self, config_dict: dict):
        """Loads configuration from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def pre_run_calculations(self): 
        # Calculations from user defined variables to code format
        # These must be run before the config class is used
        logging.info("Starting Monte Carlo Pre run calculations.")

        # Ensure steps is not zero to prevent division by zero
        if self.steps == 0:
            logging.error("Config Error: 'steps' cannot be zero. Setting to 1.")
            self.steps = 1

        # Calculate games per step
        self.games_per_step = self.total_games / self.steps

        # Calculate frozen learning rate steps
        # Ensure games_per_step is not zero to prevent division by zero
        if self.games_per_step == 0:
            logging.error("Config Error: 'games_per_step' is zero. Setting frozen_learning_rate_steps to 1.")
            self.frozen_learning_rate_steps = 1
        else:
            self.frozen_learning_rate_steps = int(self.learning_rate_flat_games / self.games_per_step)
        
        # Ensure frozen_learning_rate_steps is at least 1 if it's a positive value
        if self.frozen_learning_rate_steps < 1 and self.learning_rate_flat_games > 0:
            self.frozen_learning_rate_steps = 1
            
        elif self.learning_rate_flat_games == 0:
            self.frozen_learning_rate_steps = 0 # No flat phase

        logging.debug(f"MC config frozen steps :{self.learning_rate_flat_games}")

        # Calculate learning rate decay rate
        decay_steps = self.steps - self.frozen_learning_rate_steps
        if decay_steps <= 0: # Prevent division by zero or negative steps for decay
            logging.warning("Config Warning: Decay steps are zero or negative. Learning rate will not decay.")
            self.learning_rate_decay_rate = 0.0
        else:
            self.learning_rate_decay_rate = round(self.learning_rate_scaling *
                                                        (self.learning_rate_start -
                                                        self.learning_rate_min
                                                        ) / decay_steps, 4)
        
        logging.info("Monte Carlo Pre run calculations finished.")
        logging.debug(f"frozen_learning_rate_steps = {self.frozen_learning_rate_steps}")
        logging.debug(f"games_per_step = {self.games_per_step}")
        logging.debug(f"learning_rate_decay_rate = {self.learning_rate_decay_rate}")
    
    def log_to_mlflow(self):
        """
        Logs all configuration attributes to MLflow in organized parameter groups.
        """
        

        # Group related parameters
        param_groups = {
            "training": {
                "total_games": self.total_games,
                "steps": self.steps,
                "cores": self.cores,
                "learning_rate_flat_games": self.learning_rate_flat_games
            },
            "learning_rates": {
                "start": self.learning_rate_start,
                "min": self.learning_rate_min,
                "scaling": self.learning_rate_scaling,
                "decay_rate": getattr(self, 'learning_rate_decay_rate', None),
                "frozen_steps": self.frozen_learning_rate_steps
            },
            "testing": {
                "games_per_step": self.test_games_per_step,
                "discount_factor": self.discount_factor,
                "exploration_rate": self.exploration_rate
            },
            "experiment": {
                "name": self.experiment_name,
                "run_name": self.run_name,
                "custom_model_name": self.custom_model_name
            }
        }

        # Log each parameter group with prefix
        for group_name, params in param_groups.items():
            for param_name, value in params.items():
                if value is not None:  # Only log non-None values
                    mlflow.log_param(f"{group_name}.{param_name}", value)

        # Log calculated properties
        if hasattr(self, '_games_per_step'):
            mlflow.log_param("calculated.games_per_step", self._games_per_step)
    

    @property
    def custom_model_name(self) -> str:
        """str: Gets the custom model name."""
        return self._custom_model_name
    
    @custom_model_name.setter
    def custom_model_name(self, value: str) -> None:
        """Sets the custom model name.
        
        Args:
            value (str): The new custom model name.
        """
        self._custom_model_name = value

        

    @property
    def agent_reload(self) :
        return self._agent_reload
    
    @agent_reload.setter
    def agent_reload(self, agent_object): 
        """
        This sets the agent reload varialbe if it is 
        of the type montecarlo agent 
        """
        self._agent_reload = agent_object
    
    @property
    def frozen_learning_rate_steps(self) -> int:
        """int: Gets the number of steps with frozen learning rate."""
        return self._frozen_learning_rate_steps

    @frozen_learning_rate_steps.setter
    def frozen_learning_rate_steps(self, value: int) -> None:
        """Sets the number of steps with frozen learning rate.
        
        Args:
            value (int): The new number of frozen learning rate steps.
        """
        self._frozen_learning_rate_steps = value
    
    @property
    def run_name(self) -> str:
        """str: Gets the experiment name."""
        return self._run_name

    @run_name.setter
    def run_name(self, value: str) -> None:
        """Sets the experiment name.
        
        Args:
            value (str): The new experiment name.
        """
        self._run_name = value



    @property
    def experiment_name(self) -> str:
        """str: Gets the experiment name."""
        return str(self._experiment_name)

    @experiment_name.setter
    def experiment_name(self, value: str) -> None:
        """Sets the experiment name.
        
        Args:
            value (str): The new experiment name.
        """
        self._experiment_name = value

    @property
    def total_games(self) -> int:
        """int: Gets the total number of games."""
        return self._total_games

    @total_games.setter
    def total_games(self, value: int) -> None:
        """Sets the total number of games.
        
        Args:
            value (int): The new total games value.
        """
        self._total_games = value

    @property
    def steps(self) -> int:
        """int: Gets the number of steps."""
        return self._steps

    @steps.setter
    def steps(self, value: int) -> None:
        """Sets the number of steps.
        
        Args:
            value (int): The new steps value.
        """
        self._steps = value

    @property
    def cores(self) -> int:
        """int: Gets the number of CPU cores to use."""
        return self._cores

    @cores.setter
    def cores(self, value: int) -> None:
        """Sets the number of CPU cores to use.
        
        Args:
            value (int): The new number of CPU cores.
        """
        self._cores = value

    @property
    def learning_rate_start(self) -> float:
        """float: Gets the learning rate."""
        return self._learning_rate_start

    @learning_rate_start.setter
    def learning_rate_start(self, value: float) -> None:
        """Sets the learning rate.
        
        Args:
            value (float): The new learning rate value.
        """
        self._learning_rate_start = value

    @property
    def learning_rate_min(self) -> float:
        """float: Gets the minimum learning rate."""
        return self._learning_rate_min

    @learning_rate_min.setter
    def learning_rate_min(self, value: float) -> None:
        """Sets the minimum learning rate.
        
        Args:
            value (float): The new minimum learning rate.
        """
        self._learning_rate_min = value

    @property
    def learning_rate_scaling(self) -> float:
        """float: Gets the learning rate scaling factor."""
        return self._learning_rate_scaling

    @learning_rate_scaling.setter
    def learning_rate_scaling(self, value: float) -> None:
        """Sets the learning rate scaling factor.
        
        Args:
            value (float): The new scaling factor for learning rate.
        """
        self._learning_rate_scaling = value


    @property
    def test_games_per_step(self) -> int:
        """int: Gets the number of test games per step."""
        return self._test_games_per_step

    @test_games_per_step.setter
    def test_games_per_step(self, value: int) -> None:
        """Sets the number of test games per step.
        
        Args:
            value (int): The new number of test games per step.
        """
        self._test_games_per_step = value

    @property
    def discount_factor(self) -> float:
        """float: Gets the discount factor (gamma) for Q-learning."""
        return self._discount_factor

    @discount_factor.setter
    def discount_factor(self, value: float) -> None:
        """Sets the discount factor (gamma) for Q-learning.
        
        Args:
            value (float): The new discount factor.
        """
        self._discount_factor = value

    @property
    def exploration_rate(self) -> float:
        """float: Gets the exploration rate (epsilon) for epsilon-greedy policy."""
        return self._exploration_rate

    @exploration_rate.setter
    def exploration_rate(self, value: float) -> None:
        """Sets the exploration rate (epsilon) for epsilon-greedy policy.
        
        Args:
            value (float): The new exploration rate.
        """
        self._exploration_rate = value

    