import logging
from monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent

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
            cls._instance.learning_rate_start = 1
            cls._instance.learning_rate_min = 0.01
            cls._instance.learning_rate_scaliing = 1
            cls._instance.learning_rate_flat_games = 0.1* cls._instance.total_games

            cls._instance.test_games_per_step = 1000


            cls._instance._frozen_learning_rate_steps = None
            cls._instance._games_per_step = None
            cls._instance._learning_rate_decay_rate = None

            cls._instance._agent_reload = None
            cls._instance.custom_model_name = "DefaultModelName" 

        
        return cls._instance
    
    def pre_run_calculations(self): 
        #calculations  from user defined variables to code format
        # These must be run before the config class is used
        self._frozen_learning_rate_steps = (self._instance.learning_rate_flat_games /
                                            (self._instance.total_games  /self._instance.steps) )
        
        

        self.games_per_step = self._instance.total_games /self._instance.steps
        
        self.learning_rate_decay_rate = round( self._instance.learning_rate_scaliing*
                                                        (self._instance.learning_rate_start -
                                                        self._instance.learning_rate_min
                                                        )/(self._instance.steps - 
                                                            self.frozen_learning_rate_steps),4)
        
        if self._frozen_learning_rate_steps < 1: 
            self._frozen_learning_rate_steps =1
        logging.info("Monte Carlo Pre run calculations finished.")
        logging.debug(f"frozen_learning_rate_steps = {self.frozen_learning_rate_steps}")
        logging.debug(f"games_per_step = {self.games_per_step}")
        logging.debug(f"learning_rate_decay_rate = {self.learning_rate_decay_rate}")

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
        if not isinstance(agent_object, MonteCarloAgent) or isinstance(agent_object, type(None)):
            raise ValueError("The agent path must be a Monte Carlo Agent")
        
        self._agent_reload = agent_object
    
    @property
    def frozen_learning_rate_steps(self) -> str:
        """str: Gets the experiment name."""
        return self._frozen_learning_rate_steps
    
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
        return self._learning_rate

    @learning_rate_start.setter
    def learning_rate_start(self, value: float) -> None:
        """Sets the learning rate.
        
        Args:
            value (float): The new learning rate value.
        """
        self._learning_rate = value

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
    def learning_rate_flat_games(self) -> float:
        """float: Gets the number of games for which learning rate is flat."""
        return self._learning_rate_flat_games

    @learning_rate_flat_games.setter
    def learning_rate_flat_games(self, value: float) -> None:
        """Sets the number of games for which learning rate is flat.
        
        Args:
            value (float): The new number of flat learning rate games.
        """
        self._learning_rate_flat_games = value

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