from abc import ABC, abstractmethod
import numpy as np
from dassh._commons import GRAVITY_CONST
from dassh.material import Material

class MixedClass(ABC):
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
        Number of subchannels
    coolant_obj : DASSH Material object
        Coolant object for the subchannels
    """
    
    
    def __init__(self, n_sc: int, coolant_obj: Material):
        # Coolant object
        self._coolant: Material = coolant_obj
        # Set initial guesses 
        self._delta_P: float = 1.0 # Guess on pressure drop
        self._delta_v: np.ndarray = 0.1 * np.ones(n_sc) # Guess on velocity variation
        self._delta_rho: np.ndarray = np.ones(n_sc) # Guess on density variation
        # Initialize star quantities
        self._hstar: np.ndarray = np.zeros(n_sc)
        self._vstar: np.ndarray = np.zeros(n_sc)
        # Initialize pressure drop
        self._pressure_drop: float = 0.0 
        # Initialize coolant density in subchannels
        self.sc_properties['density'] = self._coolant.density * np.ones(n_sc) 
        # Initialize enthalpy array
        self._enthalpy: np.ndarray = self._coolant.convert_properties(
            density=self.sc_properties['density'])
        # Initialize subchannel velocities
        self._sc_vel: np.ndarray = np.zeros(n_sc)

        
    def _calc_momentum_coefficients(self, dz: float, 
                                    ff: np.ndarray, dh: np.ndarray,
                                    delta_v: np.ndarray) -> tuple[np.ndarray]:
        """
        Calculate coefficients for the momentum equation.
        
        Parameters
        ----------
        dz : float
            Axial step size (m)
        ff : np.ndarray
            Friction factor for each subchannel (-)
        dh : np.ndarray
            Hydraulic diameter for each subchannel (m) 
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
            
        Returns
        -------
        Tuple[np.ndarray]
            Container of the two following np.ndarrays:
            
            - EE coefficients 
            - FF coefficients
        """
        EE = (self._sc_vel + delta_v) * (self._sc_vel + delta_v - self._vstar) \
            + GRAVITY_CONST * dz / 2 + ff * dz / 16 / dh * \
                (2 * self._sc_vel + delta_v)**2 
        FF = self.sc_properties['density'] * ((2 + ff * dz / 2 / dh) * \
            self._sc_vel + (1 + ff * dz / 8 / dh) * delta_v - self._vstar)
        return EE, FF
    
    
    def _calc_energy_coefficients(self, delta_v: np.ndarray, 
                                  delta_rho: np.ndarray, 
                                  RR: np.ndarray) -> tuple[np.ndarray]:
        """
        Calculate coefficients for the energy equation.
        
        Parameters
        ----------
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
        delta_rho : np.ndarray
            Variation of the SC densities (kg/m^3)
        RR : np.ndarray
            Derivative of enthalpy w.r.t. density (J*m^3/kg^2)
            
        Returns
        -------
        Tuple[np.ndarray]
            Container of the two following np.ndarrays:
            
            - SS coefficients 
            - TT coefficients
        """
        SS = (self._sc_vel + delta_v) * (self._enthalpy - self._hstar + 
                                         RR * (self.sc_properties['density'] 
                                               + delta_rho))
        TT = self.sc_properties['density'] * (self._enthalpy - self._hstar)
        return SS, TT
    
    
    def _calc_continuity_coefficients(self, delta_v: np.ndarray,
                                      areas: np.ndarray) -> tuple[np.ndarray]:
        """
        Calculate coefficients for the continuity equation.
        
        Parameters
        ----------
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
        areas : np.ndarray
            Subchannel areas (m^2)
            
        Returns
        -------
        Tuple[np.ndarray]
            Container of the two following np.ndarrays:
            
            - C_rho coefficients 
            - C_v coefficients
        """
        return areas * (self._sc_vel + delta_v), \
            areas * self.sc_properties['density']
            
        
    def _calc_RR(self, drho: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of enthalpy w.r.t. density at constant 
        pressure, that is the RR coefficient
        
        Parameters
        ----------
        drho : np.ndarray
            Variation of the SC densities (kg/m^3)
            
        Returns
        -------
        np.ndarray
            Derivative of enthalpy w.r.t. density (J*m^3/kg^2)
            RR = dh / drho = [h(rho + drho) - h(rho)] / drho
        """
        return (self._coolant.convert_properties(
            density=self.sc_properties['density']+drho) 
                - self._enthalpy) / drho
    
    @abstractmethod
    def _calc_star_quantity():
        """Calculate the star quantities hstar and vstar"""
        pass
    

    @property
    def coolant(self) -> Material:
        """Get the coolant object"""
        return self._coolant
    
    @coolant.setter
    def coolant(self, value: Material):
        self._coolant = value