"""Module containing functions to evaluate the time effort and accuracy of the
   enthalpy to temperature conversion methods."""

from temp_from_h import table_method, poly_method
from _commons import DEG, DELTA_H, TEMP_COOLANT_INT, TOL, TIME_MAXITER, \
    TIME_MINITER
import time
import numpy as np
from typing import Callable, Any
from dassh import Material
import os


##############################################################################
#                                   TIME 
##############################################################################
def calc_elapsed_time(method: Callable[[Any], np.ndarray], 
                      data_path: str = None,
                      coeffs_T2h: np.ndarray = None, 
                      coeffs_h2T: np.ndarray = None,
                      coolant: Material = None) -> float:
    """
    Function to calculate the time effort needed for retrieving the output of 
    a single call of the function `method`
    
    Parameters
    ----------
    method : Callable[[Any], np.ndarray]
        Method for which the time effort is calculated
    data_path : str, optional
        Path to the data file 
        (used only for table_method), by default None
    coeffs_T2h : np.ndarray, optional
        Coefficients of the polynomium for T to h conversion 
        (used only for poly_method), by default None
    coeffs_h2T : np.ndarray, optional
        Coefficients of the polynomium for h to T conversion 
        (used only for poly_method), by default None
    coolant : Material, optional
        Material object (used only for newton_method), by default None
        
    Returns
    -------
    float
        Computational time
    """
    args = prepare_args(method, DELTA_H, TEMP_COOLANT_INT, data_path, 
                        coeffs_T2h, coeffs_h2T, coolant)
    start_time = time.time()
    method(*args)
    end_time = time.time()
    return end_time - start_time


def eval_time(method: Callable[[Any], np.ndarray], data_path: str = None,
              coeffs_T2h: np.ndarray = None, coeffs_h2T: np.ndarray = None,
              coolant: Material = None) -> float:
    """
    Function to estimate the average time effort of a function `method`
    
    Parameters
    ----------
    method : Callable[[Any], np.ndarray]
        Method for which the average time effort is evaluated
    data_path : str, optional
        Path to the data file (used only for table_method), by default None
    coeffs_T2h : np.ndarray, optional
        Coefficients of the polynomium for T to h conversion 
        (used only for poly_method), by default None
    coeffs_h2T : np.ndarray, optional
        Coefficients of the polynomium for h to T conversion 
        (used only for poly_method), by default None
    coolant : Material, optional
        Material object (used only for newton_method), by default None
        
    Returns
    -------
    float
        Average time effort
    """
    total_time = 0
    average_time = []
    i = 0
    err = [1.0]

    while (i < TIME_MAXITER):
        i += 1
        elapsed_time = calc_elapsed_time(method, data_path, coeffs_T2h, 
                                         coeffs_h2T, coolant)
        total_time += elapsed_time
        average_time.append((total_time / i))
        
        if i <= 1: 
            err.append(1)
            continue
    
        err.append(np.abs(average_time[i-2] - average_time[i-1]) 
                   / average_time[i-2])
        if i >= TIME_MINITER and all(e < TOL for e in err[-TIME_MINITER:]):         
            return average_time[-1]
        
    raise RuntimeError("Maximum number of iterations reached.")
    

##############################################################################
#                                 ACCURACY 
##############################################################################
def eval_accuracy(method: Callable[[Any], np.ndarray], reference: np.ndarray, 
                  data_path: str = None, coeffs_T2h: np.ndarray = None, 
                  coeffs_h2T: np.ndarray = None,
                  coolant: Material = None) -> np.ndarray:
    """
    Function to evaluate the accuracy of the function `method`
    
    Parameters
    ----------
    method : Callable[[Any], np.ndarray]
        Method for which the accuracy is evaluated
    reference : np.ndarray
        Reference data to compare against
    data_path : str, optional
        Path to the data file for the table method
        (used only for table_method), by default None
    coeffs_T2h : np.ndarray, optional
        Coefficients for the T to h conversion
        (used only for poly_method), by default None
    coeffs_h2T : np.ndarray, optional
        Coefficients for the h to T conversion
        (used only for poly_method), by default None
    coolant : Material, optional
        Material object (used only for newton_method), by default None

    Returns
    -------
    np.ndarray
        Array containing the relative error with respect to the reference data.
        Same dimension as `reference` minus one.
    """
    tt_ref = reference[:,0]
    hh_ref = reference[:,1]
    dh_ref = hh_ref[1:] - hh_ref[:-1]
    args = prepare_args(method, dh_ref, tt_ref[:-1], data_path, 
                        coeffs_T2h, coeffs_h2T, coolant)
    res = method(*args)
    return np.abs((res - tt_ref[1:]) / tt_ref[1:])


##############################################################################
#                              AUXILIARY
##############################################################################
def get_data_from_file(file_path: str) -> tuple[np.ndarray]:
    """
    Function to get data from the table read from `file_path`
    
    Parameters
    ----------
    file_path : str
        Path to the data file
        
    Returns
    -------
    tuple[np.ndarray]
        Tuple containing the temperature and enthalpy data
    """  
    data = np.genfromtxt(file_path, delimiter=',')
    return data[:,0], data[:,1]


def prepare_args(method: Callable[[Any], np.ndarray], dh: np.ndarray,
                 temp_int: np.ndarray, data_path: str = None,
                 coeffs_T2h: np.ndarray = None, coeffs_h2T: np.ndarray = None,
                 coolant: Material = None) -> tuple[np.ndarray]:
    """
    Function to prepare the arguments for the conversion methods
    
    Parameters
    ----------
    method : Callable[[Any], np.ndarray]
        Method for which the arguments are prepared
    data_path : str, optional
        Path to the data file 
        (used only for table_method), by default None
    coeffs_T2h : np.ndarray, optional
        Coefficients of the polynomium for T to h conversion 
        (used only for poly_method), by default None
    coeffs_h2T : np.ndarray, optional
        Coefficients of the polynomium for h to T conversion
        (used only for poly_method), by default None
    coolant : Material, optional
        Material object (used only for newton_method), by default None
        
    Returns
    -------
    tuple[np.ndarray]
        Tuple containing the arguments for `method`
    """
    if method == table_method:
        xx, yy = get_data_from_file(data_path)
        return (dh, temp_int, xx, yy)
    elif method == poly_method:
        return (dh, temp_int, coeffs_T2h, coeffs_h2T)
    else:
        return (dh, temp_int, coolant)


def table_method_summary(reference: np.ndarray, path: str) \
    -> dict[int, dict[str, float]]:
    """
    Function to summarize the results of a method for different dataset sizes
    
    Parameters
    ----------
    reference : np.ndarray
        Reference data to compare against
    path : str
        Path to the data file
    
    Returns
    -------
    dict[int, dict[str, float]]
        Dictionary containing the minimum, maximum and average errors, and the
        time effort for each dataset size
    """
    
    acc = eval_accuracy(table_method, reference, data_path=path)
    res_dict = {
        "emin": np.min(acc),
        "emax": np.max(acc),
        "eave": np.mean(acc),
        "time": eval_time(table_method, data_path=path)
    }
    return res_dict

def get_poly_results(reference: np.ndarray, data: np.ndarray
                     ) -> dict[str, np.ndarray]:
    """
    Function to get the results of the polynomial method for different degrees
    
    Parameters
    ----------
    reference : np.ndarray
        Reference data to compare against
    data : np.ndarray
        Data to be used for the polynomial fitting
        
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing the average errors and time efforts for
        different polynomial degrees
    """
    poly_results = {'eave_poly': np.array([]),
                    'time_poly': np.array([])}

    for dd in range(1, DEG):
        print(f"Evaluating polynomium degree {dd}...")
        coeffs_T2h = np.polyfit(data[:,0], data[:,1], deg=dd)
        coeffs_h2T = np.polyfit(data[:,1], data[:,0], deg=dd)
        
        errors = eval_accuracy(poly_method, reference, coeffs_T2h=coeffs_T2h, 
                            coeffs_h2T=coeffs_h2T)
        poly_results['eave_poly'] = np.append(poly_results['eave_poly'], 
                                            np.mean(errors))
        poly_results['time_poly'] = np.append(poly_results['time_poly'], 
                                            eval_time(poly_method, 
                                            coeffs_T2h=coeffs_T2h, 
                                            coeffs_h2T=coeffs_h2T))
    return poly_results