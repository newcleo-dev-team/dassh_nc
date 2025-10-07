import numpy as np
from dassh.correlations.properties_abs import PropertyClass
from typing import Union

class Mat_from_corr(PropertyClass):
    """
    Correlation object for sodium properties
    
    Parameters
    ----------
    prop : str
        Property name to be calculated
        
    Returns
    ----------
    float
        Property value
        
    Notes
    ----------
    Reference correlations can be found in:
    [1] O. J. Foust, Ed., Sodium-NaK Engineering Handbook 
    Volume I - Sodium Chemistry and Physical Properties, 
    Gordon and Breach, Science Publishers, Inc., 1972.
    """
    def density(self, T: float) -> float:
        """
        Computes NaK density
        Eq. (1.9) pag. 18 of [1]
        """
        rho_na = 1000 * self.__density_Na(T) 
        rho_k = 1000 * self.__density_K(T)
        N_k = 0.611138  # corresponding to 72.77 K weight fraction
        N_na = 1 - N_k
        v = (N_na/rho_na + N_k/rho_k)*1.003
        return 1/v 
    
    def thermal_conductivity(self, T: float) -> float:
        """
        Computes NaK thermal conductivity
        Eq. (1.53) pag. 46 of [1]
        """
        T = T - 273.15
        return (0.214 + 2.07e-4*T - 2.2e-7*T**2)*100
    
    def viscosity(self, T: float) -> float:
        """
        Computes NaK viscosity
        Eq. (1.18) and (1.19) pag.24 of [1]
        """
        rho = self.density(T) / 1000 # g/cm3
        if T <= 673.15:
            return 0.116*rho**(1/3) * np.exp(688*rho/T) / 1000
        else:
            return 0.082*rho**(1/3) * np.exp(979*rho/T) / 1000
    
    def heat_capacity(self, T: float) -> float:
        """
        Computes NaK heat capacity
        Eq. (1.59) pag. 53 of [1]
        """
        cp = 0.2320 - 8.82e-5 * T + 8.2e-8*T**2
        return cp*4186.8

    def enthalpy(self, T: float) -> float:
        """
        Computes NaK enthalpy
        Obtained by integrating the heat capacity correlation
        in Eq. (1.59) pag. 53 of [1]
        """
        return (971.3376*T - 0.18465*T**2 + 1.1443e-4*T**3)

    @property 
    def density_range(self) -> tuple[float]:
        return (323.15, 1423.15)
    
    @property 
    def thermal_conductivity_range(self) -> tuple[float]:
        return (323.15, 1173.15)
    
    @property 
    def viscosity_range(self) -> tuple[float]:
        return (323.15, 1423.15)
    
    @property 
    def heat_capacity_range(self) -> tuple[float]:
        return (323.15, 1423.15)
    
    @property
    def enthalpy_range(self) -> tuple[float]:
        return (323.15, 1423.15)
    
    def __density_K(self, T: float) -> float:
        T = T - 273.15
        return 0.8415 - 2.172e-4*T - 2.70e-8*T**2 + 4.77e-12*T**3
    
    def __density_Na(self, T: float) -> float:
        T = T - 273.15
        return 0.9501 - 2.2976e-4*T - 1.460e-8*T**2 + 5.638e-12*T**3