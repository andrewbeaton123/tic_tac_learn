from abc import ABC


class ConfigBaseClass(ABC):

    def load_from_dict(self, config_dict: dict):
        """Loads configuration from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.config_dict = config_dict
    