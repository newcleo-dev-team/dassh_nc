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
    [1] J. K. Fink and L. Leibowitz, "Thermodynamic and 
    Transport Properties of Sodium Liquid and Vapor," 
    ANL-RE-95-2, Argonne National Laboratory, 1995.
    """
    def density(self, T: float) -> float:
        """
        Compute density
        Eg. (1) pag. 86 of [1]
        """
        return 219 + 275.32*(1-T/2503.7) + 511.58*(1-T/2503.7)**0.5 
    
    def thermal_conductivity(self, T: float) -> float:
        """
        Compute thermal conductivity
        Eg. (1) pag. 181 of [1]
        """
        return 124.67 - 0.11381*T + 5.5226e-5*T**2 - 1.1842e-8*T**3
    
    def viscosity(self, T: float) -> float:
        """
        Compute viscosity
        Eq. (1) pag. 207 of [1]
        """
        return np.exp(-6.4406 - 0.3958 * np.log(T) + 556.835/T)
    
    def heat_capacity(self, T: float) -> float:
        """
        Compute heat capacity
        Eq. (39) pag. 29 of [1]
        """
        # CODATA equation for Na
        return (1.6582 - 8.4790e-4*T + 4.4541e-7*T**2 - 2992.6/T**2)*1000  # J/kg-K
    
    def enthalpy(self, T: float) -> float:
        """
        Compute enthalpy (relative to solid sodium at 298.15 K)
        Eq. (1) pag. 4 of [1]
        """
        return (-365.77 + 1.6582*T - 4.2395e-4*T**2 + 1.4847e-7*T**3 
                + 2992.6/T)*1e3  # J/kg
    
    @property 
    def density_range(self) -> tuple[float]:
        return (371, 2500)
    
    @property 
    def thermal_conductivity_range(self) -> tuple[float]:
        return (371, 2500)
    
    @property 
    def viscosity_range(self) -> tuple[float]:
        return (371, 2500)
    
    @property 
    def heat_capacity_range(self) -> tuple[float]:
        return (371, 2000)
    
    @property 
    def enthalpy_range(self) -> tuple[float]:
        return (371, 2000)