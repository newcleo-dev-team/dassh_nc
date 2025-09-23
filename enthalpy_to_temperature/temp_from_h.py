import numpy as np
from dassh import Material 
from lbh15 import Lead

##################################################
#                 NEWTON'S METHOD 
##################################################

ENTHALPY_COEFF = {
    'lead': [176.2, -2.4615e-2, 5.147e-6, 1.524e6]
}

lead_dict = {'density': '11441 - 1.2795*T',
        'viscosity': '4.55e-4 * exp(1069/T)',
        'thermal_conductivity': '9.2 + 0.011*T',
        'heat_capacity': '176.2 - T * (4.923e-2 - 1.544e-5 * T) - 1.524e6 / T / T'}

coolant = Material('lead', 700)

def newton_method(dh: np.ndarray, temp_coolant_int: np.ndarray) -> np.ndarray:
    """
    Apply Newton's method to find the temperature corresponding to a given enthalpy change 
    and initial temperature for the coolant.
    
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
        toll = 1e-4
        err = 1
        iter = 1
        while (err >= toll) and (iter < 100):
            deltah = _calc_delta_h(temp_coolant_int[i], tref[i])
            coolant.update(tref[i])
            TT[i] = tref[i] + (dh[i] - deltah)/coolant.heat_capacity
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
        
######################################################
#                  LBH15 METHOD
######################################################

def lbh15_method(dh: np.ndarray, temp_coolant_int: np.ndarray) -> np.ndarray:
    """
    Use the lbh15 library to find the temperature corresponding to a given enthalpy change
    and initial temperature for the coolant.
    
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

######################################################
#               TABLE METHOD 
######################################################

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
    
    TT = np.zeros(len(dh))
    for i, temp in enumerate(temp_coolant_int):
        h_in = np.interp(temp, xx, yy)
        h_out = h_in + dh[i]
        TT[i] = np.interp(h_out, yy, xx)
    return TT