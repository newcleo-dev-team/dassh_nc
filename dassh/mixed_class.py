import numpy as np
import sys
from dassh._commons import GRAVITY_CONST

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
        Number of subchannels
    sc_area : np.ndarray
        Subchannel areas
    """
    
    
    def __init__(self, n_sc: int, coolant_obj):
        # Coolant object
        self.coolant = coolant_obj
        # Set initial guesses 
        self._delta_P = 1.0 # Guess on pressure drop
        self._delta_v = 0.1 * np.ones(n_sc) # Guess on velocity variation
        self._delta_rho = np.ones(n_sc) # Guess on density variation
        # Initialize star quantities
        self._hstar = np.zeros(n_sc)
        self._vstar = np.zeros(n_sc)
        # Initialize pressure drop
        self._pressure_drop = 0.0 
        # Initialize coolant density in subchannels
        self.sc_properties['density'] = self.coolant.density * np.ones(n_sc) 
        # Initialize enthalpy array
        self._enthalpy = self.coolant.convert_properties(
            density=self.sc_properties['density'])
        # Initialize subchannel velocities
        self._sc_vel = np.zeros(n_sc)
        
        
    def _calc_momentum_coefficients(self, nn: int, dz: float, 
                                    ff: np.ndarray, dh: np.ndarray,
                                    delta_v: np.ndarray) -> tuple[np.ndarray]:
        """
        Calculate Ei and Fi coefficients for the momentum equation
        
        Parameters
        ----------
        nn : int
            Number of coolant subchannels
        dz : float
            Axial step size (m)
        ff : np.ndarray
            Friction factor for each subchannel
        dh : np.ndarray
            Hydraulic diameter for each subchannel
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
            Enthalpy variation coefficient (J*m^3/kg^2)
            
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
            
    def _calc_h_v_star(self, delta_v: np.ndarray, delta_rho: np.ndarray, 
                       RR: np.ndarray, nn: int) -> None:
        """
        Update hstar and vstar
        
        Parameters
        ----------
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
        delta_rho : np.ndarray
            Variation of the SC densities (kg/m^3)
        RR : np.ndarray
            Enthalpy variation coefficient (J*m^3/kg^2)
        nn : int
            Number of coolant subchannels
            
        Notes
        -----
        Two options are available:
        1) Approximate hstar and vstar as the midpoint value of enthalpy
           and velocity (i.e., at z + dz/2) `h_mid` and `v_mid`
        2) Calculate hstar and vstar as per "Cheng, S.K., 1984. Constitutive 
           Correlations for wire-wrapped subchannel analysis under forced and
           mixed convection conditions (Ph.D. thesis). MIT."
        """
        h_mid = self._enthalpy + RR * delta_rho / 2
        v_mid = self._sc_vel + delta_v / 2
        # OPTION 1: Approximate hstar and vstar as midpoints
        if not self._accurate_star_quantities:
            self._hstar = h_mid
            self._vstar = v_mid
            return
        # OPTION 2: Calculate hstar and vstar as per Cheng 
        numerator_h = np.zeros(nn)
        numerator_v = np.zeros(nn)
        sum_den = np.zeros(nn)
        # Calculate delta_m for each subchannel
        vrho_1 = self.sc_properties['density'] * self._sc_vel 
        vrho_2 = (self.sc_properties['density'] + delta_rho) * \
            (self._sc_vel + delta_v) 
        delta_m = (vrho_2 - vrho_1) * \
            self.params['area'][self.subchannel.type[:nn]]
        # Iterate over subchannels and adjacent subchannels
        for i in range(nn):
            denominator = 0.0
            num_h = 0.0
            num_v = 0.0
            for k in range(3):
                j = self.ht['cond']['adj'][i][k]
                if i in self.ht['conv']['ind'][self.ht['conv']['type'] == 2] \
                    and k == 2:
                    continue
                # Calculate delta_m difference between adjacent subchannel
                xij = delta_m[i] - delta_m[j]
                # Calculate numerators and denominators
                num_h += self._calc_star_quantity_numerator(
                    h_mid[i], h_mid[j], xij)
                num_v += self._calc_star_quantity_numerator(
                    v_mid[i], v_mid[j], xij)
                denominator += np.abs(xij)
            numerator_h[i] = num_h
            numerator_v[i] = num_v
            sum_den[i] = denominator      
        # Calculate hstar and vstar
        sum_den = 2 * sum_den + sys.float_info.epsilon 
        self._hstar = numerator_h / sum_den
        self._vstar = numerator_v / sum_den