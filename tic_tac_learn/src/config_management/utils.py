
import logging

from datetime import datetime


class ConfigError(Exception):

    def __init__(self, message,code=None):
        self.message = message
        self.code = code
        self.timestamp = datetime.now()
        self.log_error()
        super().__init__(self.message)

    def log_error(self):
        logging.error(f"[{self.timestamp}] - ConfigError ({self.code}) : {self.message}")

class MissingConfigError(ConfigError):
    def __init__(self, message="Required configuration is missing.",code = 1001):
        super().__init__(message,code)

class InvalidConfigValueError(ConfigError):
    def __init__(self, message="Configuration value is invalid.", code = 1002):
        super().__init__(message,code)

class ConfigTypeError(ConfigError):
    def __init__(self, message="Configuration type is incorrect.", code = 1003):
        super().__init__(message,code)