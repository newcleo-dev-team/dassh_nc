import numpy as np
from dassh import Material 
from lbh15 import Lead
from _commons import ENTHALPY_COEFF, T_IN, MATERIAL, NEWTON_MAXITER, TOL


##############################################################################
#                              NEWTON'S METHOD 
##############################################################################
coolant = Material(MATERIAL, T_IN)

def newton_method(dh: np.ndarray, temp_coolant_int: np.ndarray) -> np.ndarray:
    """
    Find the temperature corresponding to a given enthalpy change and initial 
    temperature for the coolant using the Newton's method.
    
    Parameters
    ----------
    dh : numpy.ndarray
        Enthalpy change (J/kg)
    temp_coolant_int : numpy.ndarray
        Initial temperature of the coolant (K)
        
    Returns
    -------
    numpy.ndarray
        Final temperature of the coolant (K)
    """
    tref = temp_coolant_int.copy()
    TT = np.zeros(len(dh))
    for i in range(len(dh)):
        toll = TOL
        err = 1
        iter = 1
        while (err >= toll) and (iter < NEWTON_MAXITER):
            deltah = _calc_delta_h(temp_coolant_int[i], tref[i])
            coolant.update(tref[i])
            TT[i] = tref[i] + (dh[i] - deltah) / coolant.heat_capacity
            err = np.abs((TT[i]-tref[i]))
            tref[i] = TT[i] 
            iter += 1
    return TT

def _calc_delta_h(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """
        Calculate the enthalpy difference between two temperatures
        
        Parameters
        ----------
        T1 : numpy.ndarray
            Initial temperature (K)
        T2 : numpy.ndarray
            Final temperature (K)

        Returns
        -------
        numpy.ndarray
            Enthalpy difference (J/kg)
        """
        a, b, c, d = ENTHALPY_COEFF[coolant.name]
        return (a * (T2 - T1) \
                + b * (T2**2 - T1**2)
                + c * (T2**3 - T1**3)
                + d * (T2**(-1) - T1**(-1)))
        
##############################################################################
#                               LBH15 METHOD
##############################################################################

def lbh15_method(dh: np.ndarray, temp_coolant_int: np.ndarray) -> np.ndarray:
    """
    Use the lbh15 library to find the temperature corresponding to a given
    enthalpy change and initial temperature for the coolant.

    Parameters
    ----------
    dh : numpy.ndarray
        Enthalpy change (J/kg)
    temp_coolant_int : numpy.ndarray
        Initial temperature of the coolant (K)
        
    Returns
    -------
    numpy.ndarray
        Final temperature of the coolant (K)
    """
    TT = np.zeros(len(dh))
    for i in range(len(dh)):
        h_in = Lead(T = temp_coolant_int[i]).h
        TT[i] = Lead(h = h_in + dh[i]).T
    return TT


##############################################################################
#                               TABLE METHOD 
##############################################################################
def table_method(dh: np.ndarray, temp_coolant_int: np.ndarray, 
                 xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """
    Use interpolation of tabulated data to find the temperature 
    corresponding to a given enthalpy change and initial temperature
    for the coolant.
    
    Parameters
    ----------
    dh : numpy.ndarray
        Enthalpy change (J/kg)
    temp_coolant_int : numpy.ndarray
        Initial temperature of the coolant (K)
    xx : numpy.ndarray
        Temperature data (K)
    yy : numpy.ndarray
        Enthalpy data (J/kg)
        
    Returns
    -------
    numpy.ndarray
        Final temperature of the coolant (K)
    """
    h_in = np.interp(temp_coolant_int, xx, yy)   
    h_out = h_in + dh                         
    return np.interp(h_out, yy, xx)


def tab_method_from_hin(dh: np.ndarray, xx: np.ndarray, yy: np.ndarray, 
                        h_in: np.ndarray) -> np.ndarray:
    """
    Use interpolation of tabulated data to find the temperature 
    corresponding to a given enthalpy change and initial temperature
    for the coolant.
    
    Parameters
    ----------
    dh : numpy.ndarray
        Enthalpy change (J/kg)
    xx : numpy.ndarray
        Temperature data (K)
    yy : numpy.ndarray
        Enthalpy data (J/kg)
    h_in : numpy.ndarray
        Initial enthalpy of the coolant (J/kg)
        
    Returns
    -------
    numpy.ndarray
        Final temperature of the coolant (K)
    """
    h_out = h_in + dh
    return np.interp(h_out, yy, xx)


##############################################################################
#                               POLYNOMIUM
##############################################################################
def poly_method(dh: np.ndarray, T_coolant: np.ndarray, coeffs_T2h: np.ndarray,
                coeffs_h2T: np.ndarray) -> np.ndarray:
    """
    Function to calculate the temperature from enthalpy using an interpolating 
    polynomium
    
    Parameters
    ----------
    dh : np.ndarray
        Enthalpy difference array
    T_coolant : np.ndarray
        Coolant temperature array
    coeffs_T2h : np.ndarray
        Coefficients of the polynomium for T to h conversion
    coeffs_h2T : np.ndarray
        Coefficients of the polynomium for h to T conversion
        
    Returns
    -------
    np.ndarray
        Temperature array
    """
    h_coolant = np.polyval(coeffs_T2h, T_coolant)
    h = h_coolant + dh
    return np.polyval(coeffs_h2T, h)