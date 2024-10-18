# search_space.py

from pydantic import BaseModel, validator
from typing import Union, List
import yaml

class IntParam(BaseModel):
    name: str
    low: int
    high: int

class FloatParam(BaseModel):
    name: str
    low: float
    high: float

class CategoricalParam(BaseModel):
    name: str
    choices: List[Union[str, int, float]]

class SearchSpace(BaseModel):
    model_type: str
    int_params: List[IntParam] = []
    float_params: List[FloatParam] = []
    categorical_params: List[CategoricalParam] = []

    # @validator('*', pre=True, each_item=True)
    # def check_params(cls, v):
    #     if not isinstance(v, list):
    #         return [v]
    #     return v
    @validator('int_params', 'float_params', 'categorical_params', pre=True)
    def check_params(cls, v):
        if not isinstance(v, list):
            return [v]
        return v

def load_search_space(yaml_file):
    with open(yaml_file, 'r') as file:
        search_space_dict = yaml.safe_load(file)
    search_space = SearchSpace(**search_space_dict)
    return search_space
