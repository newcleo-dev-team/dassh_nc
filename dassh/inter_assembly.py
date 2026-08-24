########################################################################
"""
date: 2026-07-xx
author: fpepe
Class to handle the inter-assembly models
"""
########################################################################
import numpy as np
from dassh.material import Material
from dassh.mixed_class import MixedClass
from dassh._commons import PROPS_NAME
from typing import Union


class InterAssembly(MixedClass):
    """Class to handle the inter-assembly models
    
    Parameters
    ----------
    model : str
        Inter-assembly gap model to use
        Options are: 'flow', 'no_flow', 'duct_average', and 'mixed_flow'
    n_sc : int
        Number of subchannels in the assembly
    gap_coolant : DASSH Material object
        Coolant object for the inter-assembly gap coolant
    Rcond : numpy.ndarray
        Thermal contact resistance between subchannels [m^2-K/W]
    sc_adj : numpy.ndarray
        Subchannel adjacency matrix
    conv_util : dict[str, Union[numpy.ndarray, list]]
        Dictionary of convection utility variables
    inv_sc_mfr : numpy.ndarray
        Inverse of the subchannel mass flow rate [s/kg]
    """    
    def __init__(self, model: str, n_sc: int,
                 gap_coolant: Material, Rcond: np.ndarray, 
                 sc_adj: np.ndarray, 
                 conv_util: dict[str, Union[np.ndarray, list]],
                 inv_sc_mfr: np.ndarray):
        
        self._model: str = model
        self._gap_coolant: Material = gap_coolant
        self._sc_adj: np.ndarray = sc_adj
        self._Rcond: np.ndarray = Rcond
        self._conv_util: dict[str, Union[np.ndarray, list]] = conv_util
        self._inv_sc_mfr: np.ndarray = inv_sc_mfr
        self.sc_properties: dict[str, np.ndarray] = {k: np.zeros(n_sc) 
                                                     for k in PROPS_NAME}
        if self._model == 'mixed_flow':
            MixedClass.__init__(self, n_sc, coolant_obj=gap_coolant)
        self._n_sc: int = n_sc

    def set_params(self, dz: float, t_duct: np.ndarray, 
                   coolant_gap_temp: np.ndarray, htc: np.ndarray, 
                   ff: Union[np.ndarray, None] = None):
        """
        Set non-constant parameters for the inter-assembly gap model
        
        Parameters
        ----------
        dz : float
            Axial mesh size [m]
        t_duct : numpy.ndarray
            Duct wall temperature [K]
        coolant_gap_temp : numpy.ndarray
            Inter-assembly gap coolant temperature [K]
        htc : numpy.ndarray
            Heat transfer coefficient between the duct wall and the 
            inter-assembly gap coolant [W/m^2-K]
        ff : Union[numpy.ndarray, None], optional
            Friction factor for the inter-assembly gap [-]
        """
        self._dz = dz
        self._t_duct = t_duct
        self._coolant_gap_temp = coolant_gap_temp
        self._htc = htc
        self._ff = ff
    

    def gap_model(self) -> np.ndarray:
        """Run the selected inter-assembly gap model to calculate the 
        temperature in the inter-assembly 
        
        Returns
        -------
        numpy.ndarray
            Temperature calculated in the inter-assembly gap coolant
        """
        if self._model in self.available_models:
            self.available_models[self._model]()
        return self._coolant_gap_temp
        
        
    def _flow_model(self):
        """Inter-assembly gap convection model
            
        Notes
        -----
        The contact resistance between the bulk liquid and the duct
        wall is calculated using a heat transfer coefficient based on
        the actual velocity of the interassembly gap flow
        """
        # CONVECTION TO/FROM DUCT WALL
        C = self._conv_util['const'] * self._htc[:, None]
        dT = C[:, 0] * (self._t_duct[tuple(self._conv_util['inds'][0])]
                        - self._coolant_gap_temp)
        dT += C[:, 1] * (self._t_duct[tuple(self._conv_util['inds'][1])]
                         - self._coolant_gap_temp)
        dT += C[:, 2] * (self._t_duct[tuple(self._conv_util['inds'][2])]
                         - self._coolant_gap_temp)

        # CONDUCTION TO/FROM OTHER COOLANT CHANNELS
        dT += (self._gap_coolant.thermal_conductivity * 
               np.sum((self._Rcond * (self._coolant_gap_temp[self._sc_adj - 1]
                                     - self._coolant_gap_temp[..., None])), 
                      axis=1))

        self._coolant_gap_temp += dT * self._dz * self._inv_sc_mfr \
            / self._gap_coolant.heat_capacity
        
        
    def _noflow_model(self):
        """Inter-assembly gap conduction model

        Notes
        -----
        Recommended for use when inter-assembly gap flow rate is so
        low that the the axial mesh requirement is intractably small.
        Assumes no thermal contact resistance between the duct wall
        and the coolant.
        """
        # CONDUCTION TO/FROM DUCT WALL
        R_conv = self._conv_util['const']

        # Lookup temperatures and mask as necessary
        T = R_conv[:, 0] * self._t_duct[tuple(self._conv_util['inds'][0])]
        T += R_conv[:, 1] * self._t_duct[tuple(self._conv_util['inds'][1])]
        T += R_conv[:, 2] * self._t_duct[tuple(self._conv_util['inds'][2])]
        # Get the total conduction resistance, which will go in the
        # denominator at the end
        C_conv = R_conv[:, 0] + R_conv[:, 1] + R_conv[:, 2]

        # CONDUCTION TO/FROM OTHER COOLANT CHANNELS
        R_cond = self._Rcond
        adj_ctemp = self._coolant_gap_temp[self._sc_adj - 1] * R_cond
        C_cond = R_cond[:, 0] + R_cond[:, 1] + R_cond[:, 2]

        # COMBINE AND APPLY TOTAL RESISTANCE DENOM
        T += adj_ctemp[:, 0] + adj_ctemp[:, 1] + adj_ctemp[:, 2]
        self._coolant_gap_temp = T / (C_cond + C_conv)


    def _duct_average_model(self):
        """Inter-assembly gap model that simply averages the adjacent
        duct wall surface temperatures

        Notes
        -----
        Recommended for use when inter-assembly gap flow rate is so
        low that the axial mesh requirement is intractably small.
        Assumes no thermal contact resistance between the duct wall
        and the coolant
        """
        # Lookup temperatures and mask as necessary
        T0 = self._t_duct[tuple(self._conv_util['inds'][0])]
        T1 = (self._t_duct[tuple(self._conv_util['inds'][1])]
              * self._conv_util['mask1'])
        T2 = (self._t_duct[tuple(self._conv_util['inds'][2])]
              * self._conv_util['mask2'])

        # Average nonzero values
        self._coolant_gap_temp = (np.sum((T0, T1, T2), axis=0)
                                  / np.count_nonzero((T0, T1, T2), axis=0))
        
        
    def _mixed_flow_model(self):
        """
        Inter-assembly gap model that uses a mixed convection model
        to calculate the inter-assembly gap coolant temperature
        """        
        self._solve_system()
        self._coolant_gap_temp = \
            self._coolant.convert_properties(enthalpy=self._enthalpy)       
        
    
    def _solve_system(self):
        """
        Solve the system of equations for the mixed convection model
        
        """
        # Use previous step deltas as initial guesses
        delta_rho0, delta_v0, delta_P0 = \
            self._copy_solution(self._delta_rho, self._delta_v, self._delta_P)
        # BUild known vector
        bb = self._build_vector()
        # Calculate initial RR using guess `delta_rho0`
        RR = self._calc_RR(delta_rho0)
        # Iterate to solve the system
        iter = 0
        err_vect = np.ones(3)
        while np.any(err_vect > 1e-3) and iter < 10:
            # Build matrix
            AA = self._build_matrix(self._dz, delta_v0, delta_rho0, RR, 
                                    self._n_sc)
            # Solve system
            xx = np.linalg.solve(AA, bb)
            # Extract deltas
            delta_rho = xx[0:2*self._n_sc:2]
            delta_v = xx[1:2*self._n_sc:2]
            delta_P = xx[-1]
            # Calculate errors
            err_vect = self._calc_error(np.dstack((delta_rho, delta_v)), 
                                        np.dstack((delta_rho0, delta_v0)),
                                        delta_P, delta_P0)
            # Update guesses for next iteration
            delta_rho0, delta_v0, delta_P0 = \
                self._copy_solution(delta_rho, delta_v, delta_P)
            # Recalculate RR 
            RR = self._calc_RR(delta_rho)
            # Update iteration counter
            iter += 1
        # Update deltas with converged values
        self._delta_rho, self._delta_v, self._delta_P = \
            self._copy_solution(delta_rho, delta_v, delta_P)
        # Update velocity, density, pressure drop adding convergence deltas
        self._sc_vel += self._delta_v
        self.sc_properties['density'] += self._delta_rho
        self._pressure_drop -= self._delta_P
        # Update enthalpy using converting density
        self._enthalpy = self._coolant.convert_properties(
            density=self.sc_properties['density'])
            
    
    def _build_vector(self) -> np.ndarray:
        pass
        
    def _calc_star_quantity(self, delta_v: np.ndarray, delta_rho: np.ndarray,
                            variable: str, 
                            RR: Union[np.ndarray, None] = None) -> np.ndarray:
        """
        Update hstar or vstar
        
        Parameters
        ----------
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
        delta_rho : np.ndarray
            Variation of the SC densities (kg/m^3)
        variable : str
            Indicate whether to calculate hstar or vstar; 
            options are 'h' or 'v'
        RR : Union[np.ndarray, None], optional
            Derivative of enthalpy with respect to density (J*m^3/kg^2);
            Only used for hstar calculation

        Returns
        -------
        np.ndarray
            Calculated star quantity for each subchannel

        Raises
        ------
        ValueError
            If `variable` is not 'h' or 'v'
        """   
        if variable != 'h' and variable != 'v':
            raise ValueError("Invalid variable for star quantity calculation.")
        if variable == 'h':
            return self._enthalpy + RR * delta_rho / 2
        return self._sc_vel + delta_v / 2


    @property
    def available_models(self):
        """
        dict[str, callable]: Dictionary of available inter-assembly gap models
        """
        return {
            "flow": self._flow_model,
            "no_flow": self._noflow_model,
            "duct_average": self._duct_average_model,
            "mixed_flow": self._mixed_flow_model,
            }