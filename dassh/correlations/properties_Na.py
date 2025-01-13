import numpy as np
from dassh.correlations.properties_abs import PropertyClass

class mat_from_corr(PropertyClass):
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
    
    def __init__(self, prop):
        self.prop = prop
    
    def __call__(self, temperature):
        return getattr(self, self.prop)(temperature)
    
    def density(self, T):
        """
        Computes sodium density
        Eg. (1) pag. 86 of [1]
        """
        return 219 + 275.32*(1-T/2503.7) + 511.58*(1-T/2503.7)**0.5 
    
    def thermal_conductivity(self, T):
        """
        Computes sodium thermal conductivity
        Eg. (1) pag. 181 of [1]
        """
        return 124.67 - 0.11381*T + 5.5226e-5*T**2 - 1.1842e-8*T**3
    
    def viscosity(self, T):
        """
        Computes sodium viscosity
        Eq. (1) pag. 207 of [1]
        """
        return np.exp(-6.4406 - 0.3958 * np.log(T) + 556.835/T)
    
    def heat_capacity(self, T):
        """
        Computes sodium heat capacity
        Eq. (39) pag. 29 of [1]
        """
        # CODATA equation for Na
        cp = 1.6582 - 8.4790e-4*T + 4.4541e-7*T**2 - 2992.6/T**2 
        return cp*1000  # J/kg-K
    @property 
    def density_range(self):
        return (371, 2500)
    @property 
    def thermal_conductivity_range(self):
        return (371, 2500)
    @property 
    def viscosity_range(self):
        return (371, 2500)
    @property 
    def heat_capacity_range(self):
        return (371, 2000)