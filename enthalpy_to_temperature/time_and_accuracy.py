from temp_from_h import table_method, poly_method
from temp_from_h import tab_method_from_hin
from _commons import DELTA_H, SIZES, TEMP_COOLANT_INT, TOL, TIME_MAXITER, TIME_MINITER
import time
import numpy as np
from typing import Callable, Any
import os


##############################################################################
#                                   TIME 
##############################################################################
def calc_elapsed_time(method: Callable[[Any], np.ndarray], 
                      data_path: str = None,
                      coeffs_T2h: np.ndarray = None, 
                      coeffs_h2T: np.ndarray = None) -> float:
    """
    Function to calculate the computational time of a method
    
    Parameters
    ----------
    method : Callable[[Any], np.ndarray]
        Method for which the computational time is calculated
    data_path : str, optional
        Path to the data file 
        (used only for table_method), by default None
    coeffs_T2h : np.ndarray, optional
        Coefficients of the polynomium for T to h conversion 
        (used only for poly_method), by default None
    coeffs_h2T : np.ndarray, optional
        Coefficients of the polynomium for h to T conversion 
        (used only for poly_method), by default None
        
    Returns
    -------
    float
        Computational time
    """
    args = prepare_args(method, DELTA_H, TEMP_COOLANT_INT, data_path, 
                        coeffs_T2h, coeffs_h2T)
    
    start_time = time.time()
    _ = method(*args)
    end_time = time.time()
    return end_time - start_time


def eval_time(method: Callable[[Any], np.ndarray], data_path: str = None,
              coeffs_T2h: np.ndarray = None, coeffs_h2T: np.ndarray = None
              ) -> float:
    """
    Function to estimate the computational time of a method
    
    Parameters
    ----------
    method : Callable[[Any], np.ndarray]
        Method for wich the computational time is evaluated
    data_path : str, optional
        Path to the data file (used only for table_method), by default None
    coeffs_T2h : np.ndarray, optional
        Coefficients of the polynomium for T to h conversion 
        (used only for poly_method), by default None
    coeffs_h2T : np.ndarray, optional
        Coefficients of the polynomium for h to T conversion 
        (used only for poly_method), by default None
        
    Returns
    -------
    float
        Average computational time
    """
    num_iterations = TIME_MAXITER
    total_time = 0
    average_time = [0]
    i = 0
    err = [1.0]
    toll = TOL

    while (i < num_iterations):                
        elapsed_time = calc_elapsed_time(method, data_path, coeffs_T2h, 
                                         coeffs_h2T)
        total_time += elapsed_time
        average_time.append((total_time / (i+1))) 
        
        if i > 1:     
            err.append(np.abs(average_time[i-1] - average_time[i]) 
                       / average_time[i-1])
        else:
            err.append(1)
            
        if i >= TIME_MINITER and all(e < toll for e in err[-TIME_MINITER:]):            
            break
        
        i += 1
    return average_time[-1] 


##############################################################################
#                                 ACCURACY 
##############################################################################
def eval_accuracy(method: Callable[[Any], np.ndarray], reference: np.ndarray, 
                  data_path: str = None, coeffs_T2h: np.ndarray = None, 
                  coeffs_h2T: np.ndarray = None) -> np.ndarray:
    """
    Function to evaluate the accuracy of a methods
    
    Parameters
    ----------
    method : Callable[[Any], np.ndarray]
        Method for which the accuracy is evaluated
    reference : np.ndarray
        Reference data to compare against
    data_path : str, optional
        Path to the data file for the table method, by default None
    coeffs_T2h : np.ndarray, optional
        Coefficients for the T to h conversion, by default None
    coeffs_h2T : np.ndarray, optional
        Coefficients for the h to T conversion, by default None

    Returns
    -------
    np.ndarray
        Tuple containing the relative error with respect to the reference data
    """
    tt_ref = reference[:,0]
    hh_ref = reference[:,1]
    dh_ref = hh_ref[1:] - hh_ref[:-1]
    args = prepare_args(method, dh_ref, tt_ref[:-1], data_path, 
                        coeffs_T2h, coeffs_h2T)
    res = method(*args)
    
    return np.abs((res - tt_ref[1:]) / tt_ref[1:])


##############################################################################
#                              AUXILIARY
##############################################################################
def get_data_from_file(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get data from a file
    
    Parameters
    ----------
    file_path : str
        Path to the data file
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the temperature and enthalpy data
    """  
    data = np.genfromtxt(os.path.join('data', file_path), delimiter=',')
    xx = data[:,0]
    yy = data[:,1]
    return xx, yy


def prepare_args(method: Callable[[Any], np.ndarray], dh: np.ndarray,
                 temp_int: np.ndarray, data_path: str = None,
                 coeffs_T2h: np.ndarray = None, coeffs_h2T: np.ndarray = None
                 ) -> tuple:
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
        
    Returns
    -------
    tuple
        Tuple containing the arguments for the method
    """
    if method == table_method:
        xx, yy = get_data_from_file(data_path)
        args = (dh, temp_int, xx, yy)
    elif method == tab_method_from_hin:
        xx, yy = get_data_from_file(data_path)
        h_in = np.interp(temp_int, xx, yy)
        args = (dh, xx, yy, h_in)
    elif method == poly_method:
        args = (dh, temp_int, coeffs_T2h, coeffs_h2T)
    else:
        args = (dh, temp_int)
    return args


def summarize_results(method: Callable[[Any], np.ndarray], 
                      reference: np.ndarray, size: int,
                      ) -> dict[int, dict[str, float]]:
    """
    Function to summarize the results of a method for different dataset sizes
    
    Parameters
    ----------
    method : Callable[[Any], np.ndarray]
        Method for which the results are summarized
    reference : np.ndarray
        Reference data to compare against
    size : int
        Size of the dataset
        
    Returns
    -------
    dict[int, dict[str, float]]
        Dictionary containing the minimum, maximum and average error, and the
        computational time for each dataset size
    """
    path = f"lead_{size}.csv"
    acc = eval_accuracy(table_method, reference, data_path=path)
    res_dict= {
        "emin": np.min(acc),
        "emax": np.max(acc),
        "eave": np.mean(acc),
        "time": eval_time(table_method, data_path=path)
    }
    return res_dict