
import logging 

from datetime import datetime
from ..config_management  import ConfigManager
cm = ConfigManager()

class AgentError(Exception):
    """ Base exception for all game errors"""
   
    def __init__(self, message,code=None):
        self.message = message
        self.code = code
        self.timestamp = datetime.now()
        super().__init__(self.message)
        self.log_error()

    def log_error(self):
        logging.error(f"[{self.timestamp}] - Agent Error ({self.code}) : {self.message}")



