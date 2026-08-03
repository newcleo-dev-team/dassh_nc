########################################################################
"""
date: 2026-07-xx
author: fpepe
Class to handle the inter-assembly models
"""
########################################################################
import numpy as np
from dassh.material import Material
from typing import Union

class InterAssembly():
    """Class to handle the inter-assembly models
    
    Parameters
    ----------
    model : str
        Inter-assembly gap model to use
        Options are: 'flow', 'no_flow', 'duct_average'
    dz : float
        Axial mesh [m]
    t_duct : numpy.ndarray
        Duct wall temperature [K]
    coolant_gap_temp : numpy.ndarray
        Inter-assembly gap coolant temperature [K]
    gap_coolant : DASSH Material object
        Coolant object for the inter-assembly gap coolant
    Rcond : numpy.ndarray
        Thermal contact resistance between subchannels [m^2-K/W]
    sc_adj : numpy.ndarray
        Subchannel adjacency matrix
    conv_util : dict
        Dictionary of convection utility variables
    inv_sc_mfr : numpy.ndarray
        Inverse of the subchannel mass flow rate [s/kg]
    htc : numpy.ndarray
        Heat transfer coefficient between the duct wall and the inter-assembly 
        gap coolant [W/m^2-K]
    """    
    def __init__(self, model: str, dz: float, t_duct: np.ndarray, 
                 coolant_gap_temp: np.ndarray, 
                 gap_coolant: Material, Rcond: np.ndarray, 
                 sc_adj: np.ndarray, 
                 conv_util: dict[str, Union[np.ndarray, list]],
                 inv_sc_mfr: np.ndarray, htc: np.ndarray):
        
        self._model: str = model
        self._dz: float = dz
        self._t_duct: np.ndarray = t_duct
        self._coolant_gap_temp: np.ndarray = coolant_gap_temp
        self._gap_coolant: Material = gap_coolant
        self._sc_adj: np.ndarray = sc_adj
        self._Rcond: np.ndarray = Rcond
        self._htc: np.ndarray = htc
        self._conv_util: dict[str, Union[np.ndarray, list]] = conv_util
        self._inv_sc_mfr: np.ndarray = inv_sc_mfr
        

    def gap_model(self) -> np.ndarray:
        """Run the selected inter-assembly gap model to calculate the 
        temperature in the inter-assembly 
        
        Returns
        -------
        numpy.ndarray
            Temperature in the inter-assembly gap coolant
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
        
    @property
    def available_models(self):
        """Dictionary of available inter-assembly gap models"""
        return {
            "flow": self._flow_model,
            "no_flow": self._noflow_model,
            "duct_average": self._duct_average_model
            }