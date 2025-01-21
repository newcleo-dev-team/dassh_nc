import numpy as np 
from abc import ABC, abstractmethod
from typing import Union

class PropertyClass(ABC):
    """
    Abstract class for property correlations
    """
    def __init__(self, prop: Union[str, None] = None):
        self.prop = prop
    
    def __call__(self, temperature: float):
        return getattr(self, self.prop)(temperature)
    
    @abstractmethod
    def density(self, T: float) -> float:
        pass
    
    @abstractmethod
    def thermal_conductivity(self, T: float) -> float:
        pass
    
    @abstractmethod
    def viscosity(self, T: float) -> float:
        pass
    
    @abstractmethod
    def heat_capacity(self, T: float) -> float:
        pass
    
    @abstractmethod
    def density_range(self) -> tuple[float]:
        pass

    @abstractmethod
    def thermal_conductivity_range(self) -> tuple[float]:
        pass

    @abstractmethod
    def viscosity_range(self) -> tuple[float]:
        pass

    @abstractmethod
    def heat_capacity_range(self) -> tuple[float]:
        pass
    