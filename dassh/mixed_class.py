import numpy as np

class MixedClass:
    """Parent class for mixed convection models
    
    Notes
    -----
    This class is used to set up the attributes and methods needed for 
    mixed convection calculations in DASSH. It is not intended to be 
    instantiated directly, but rather to be inherited by other classes that 
    implement specific mixed convection models.
    
    Parameters
    ----------
    n_sc : int
        Number of subchannels. It is either the number of subchannels inside 
        the assembly or the number of subchannels in the gap
    
    """
    
    def __init__(self, n_sc):
        self._delta_P = 1.0 # Guess on pressure drop
        self._delta_v = 0.1 * np.ones(n_sc) # Guess on velocity variation
        self._delta_rho = np.ones(n_sc) # Guess on density variation
        # Initialize star quantities
        self._hstar = np.zeros(n_sc)
        self._vstar = np.zeros(n_sc)
        