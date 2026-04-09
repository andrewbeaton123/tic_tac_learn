from abc import ABC
from typing import Any, Dict

def TrainedModelABC(ABC):

    def __init__(self,
                 model_data : Any,
                 metadata: Dict,
                 game_interface_type: str):
        self._metadata = metadata




    def predict (self,
                 state:Any):
        pass
    
    @property
    def get_metadata(self) -> Dict:
        return self._metadata
    
    def set_meta_data_field(self,
                            key:str,
                            value: Any) -> None:
        
        self._metadata[key] = value


    def save(self) -> None :
        pass 
    
    def load(self) -> None : 
        pass 

    def to_dict() -> Dict : 
        pass

    @classmethod
    def from_dict() -> TrainedModelABC:
        pass 