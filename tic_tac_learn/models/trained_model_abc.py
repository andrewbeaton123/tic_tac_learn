from abc import ABC
from typing import Any, Dict
from mlflow.models  import ModelSignature
from importlib.metadata import version

class TrainedModelABC(ABC):

    def __init__(self,
                 model_data : Any,
                 game_interface_type: str,
                 hyperparameters : Dict,
                 training_config : Dict,
                 meta_data: Dict ):
        
        self._meta_data = meta_data
        self._hyperparameters = hyperparameters
        self._training_config = training_config



    def predict (self,
                 state:Any):
        pass
    
    @property
    def meta_data(self) -> Dict:
        return self._meta_data
    
    @meta_data.setter
    def meta_data(self,value) :
        if not isinstance(value, dict):
            raise TypeError("Meta data must be a dict")
        self._meta_data= value


    def save(self) -> None :
        pass 
    
    def load(self) -> None : 
        pass 

    def to_dict(self) -> Dict : 
        pass
    
    @classmethod
    def from_dict() -> "TrainedModelABC":
        pass 

    def get_model_signature(self) -> ModelSignature:
        pass

    def get_input_example(self) -> Any: 
        pass 

    def get_model_flavor(self) -> str:
        # returns the way that the model should be treated
        return "pyfunc" 

    def get_artifact_path(self) -> str: 
        pass
    
    def get_model_uri(self) -> str: 
        pass

    def get_hyperparameters (self) -> Dict: 
        return self._hyperparameters

    def get_training_config(self) -> Dict : 
        version_config = {
            "package_version": version("tic_tac_learn")
        }
        
        return self._training_config | version_config

    def set_meta_data_field(self, key: str, value: Any) -> None:
        """Set a single metadata field."""
        self._meta_data[key] = value