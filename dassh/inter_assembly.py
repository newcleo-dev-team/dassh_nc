
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from dassh.core import Core

class InterAssembly():
    """Class to handle the inter-assembly models
    
    Parameters
    ----------
    
    """

    @staticmethod
    def flow_model(coreobj: Core, dz: float, t_duct: np.ndarray) -> np.ndarray:
        """Inter-assembly gap convection model
        
        Parameters
        ----------
        coreobj : Core
            Core object containing the core geometry and properties
        dz : float
            Axial mesh height
        t_duct : numpy.ndarray
            Array of outer duct surface temperatures (K) for each
            assembly in the core (can be any length) on the inter-
            assembly gap subchannel mesh
        
        Returns
        -------
        numpy.ndarray
            Temperature change in the inter-assembly gap coolant
        """
        # CONVECTION TO/FROM DUCT WALL
        C = (coreobj._conv_util['const']
                * coreobj.coolant_gap_params['htc'][:, None])
        dT = C[:, 0] * (t_duct[tuple(coreobj._conv_util['inds'][0])]
                        - coreobj.coolant_gap_temp)
        dT += C[:, 1] * (t_duct[tuple(coreobj._conv_util['inds'][1])]
                            - coreobj.coolant_gap_temp)
        dT += C[:, 2] * (t_duct[tuple(coreobj._conv_util['inds'][2])]
                            - coreobj.coolant_gap_temp)

        # CONDUCTION TO/FROM OTHER COOLANT CHANNELS
        dT += (coreobj.gap_coolant.thermal_conductivity
                * np.sum((coreobj._Rcond *
                        (coreobj.coolant_gap_temp[coreobj._sc_adj - 1]
                            - coreobj.coolant_gap_temp[..., None])), axis=1))

        return (dT * dz * coreobj._inv_sc_mfr
                / coreobj.gap_coolant.heat_capacity)
        
    @staticmethod
    def noflow_model(coreobj: Core, t_duct: np.ndarray) -> np.ndarray:
        """Inter-assembly gap conduction model

        Parameters
        ----------
        coreobj : Core
            Core object containing the core geometry and properties
        t_duct : numpy.ndarray
            Array of outer duct surface temperatures (K) for each
            assembly in the core (can be any length) on the inter-
            assembly gap subchannel mesh

        Returns
        -------
        numpy.ndarray
            Temperature in the inter-assembly gap coolant

        Notes
        -----
        Recommended for use when inter-assembly gap flow rate is so
        low that the the axial mesh requirement is intractably small.
        Assumes no thermal contact resistance between the duct wall
        and the coolant.

        The contact resistance between the bulk liquid and the duct
        wall is calculated using a heat transfer coefficient based on
        the actual velocity of the interassembly gap flow

        """
        # CONVECTION TO/FROM DUCT WALL
        R_conv = coreobj._conv_util['const']

        # Lookup temperatures and mask as necessary
        T = R_conv[:, 0] * t_duct[tuple(coreobj._conv_util['inds'][0])]
        T += R_conv[:, 1] * t_duct[tuple(coreobj._conv_util['inds'][1])]
        T += R_conv[:, 2] * t_duct[tuple(coreobj._conv_util['inds'][2])]
        # Get the total convection resistance, which will go in the
        # denominator at the end
        C_conv = R_conv[:, 0] + R_conv[:, 1] + R_conv[:, 2]

        # CONDUCTION TO/FROM OTHER COOLANT CHANNELS
        R_cond = coreobj._Rcond
        adj_ctemp = coreobj.coolant_gap_temp[coreobj._sc_adj - 1] * R_cond
        C_cond = R_cond[:, 0] + R_cond[:, 1] + R_cond[:, 2]

        # COMBINE AND APPLY TOTAL RESISTANCE DENOM
        T += adj_ctemp[:, 0] + adj_ctemp[:, 1] + adj_ctemp[:, 2]
        return T / (C_cond + C_conv)
    
    
    @staticmethod
    def duct_average_model(coreobj: Core, t_duct: np.ndarray) -> np.ndarray:
        """Inter-assembly gap model that simply averages the adjacent
        duct wall surface temperatures

        Parameters
        ----------
        coreobj : Core
            Core object containing the core geometry and properties
        t_duct : numpy.ndarray
            Array of outer duct surface temperatures (K) for each
            assembly in the core (can be any length) on the inter-
            assembly gap subchannel mesh

        Returns
        -------
        numpy.ndarray
            Temperature in the inter-assembly gap coolant

        Notes
        -----
        Recommended for use when inter-assembly gap flow rate is so
        low that the the axial mesh requirement is intractably small.
        Assumes no thermal contact resistance between the duct wall
        and the coolant.

        The contact resistance between the bulk liquid and the duct
        wall is calculated using a heat transfer coefficient based on
        the actual velocity of the interassembly gap flow

        """
        # Lookup temperatures and mask as necessary
        T0 = t_duct[tuple(coreobj._conv_util['inds'][0])]
        T1 = (t_duct[tuple(coreobj._conv_util['inds'][1])]
                * coreobj._conv_util['mask1'])
        T2 = (t_duct[tuple(coreobj._conv_util['inds'][2])]
                * coreobj._conv_util['mask2'])

        # Average nonzero values
        return (np.sum((T0, T1, T2), axis=0)
                / np.count_nonzero((T0, T1, T2), axis=0))