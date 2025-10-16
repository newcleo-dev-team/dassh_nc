"""Module with auxiliary functions for density to enthalpy conversion"""
import numpy as np 
from _commons import ENTHALPY_RANGE, MATERIAL_LBH, LBH15_PROP, PATH_TO_DATA, \
    DATA_EXTENSION, TEMP_STEP, REFERENCE_STEPS, REFERENCE_SUFFIX, DEG, \
        TIME_MAXITER, TIME_MINITER, TOL
from dassh.correlations import properties_Na, properties_NaK
from functools import partial
from typing import Type
import time
import os
import matplotlib.pyplot as plt
import warnings
from numpy.polynomial.polyutils import RankWarning

##############################################################################
#                            DATABASE GENERATION
##############################################################################
def check_material(material_name: str) -> None:
    """
    Check if the material name is recognized
    
    Parameters
    ----------
    material_name : str
        Name of the coolant. 
        Options are: 'lead', 'LBE', 'bismuth', 'NaK', 'sodium'
        
    Raises
    ----
    ValueError
        If the material name is not recognized
    """
    if material_name not in ENTHALPY_RANGE:
        raise ValueError(f"Material '{material_name}' not recognized.")
    
    
def generate_database(material_name: str, reference: bool = False) -> None:
    """
    Generate a database of density and enthalpy values for each fluid
    
    Parameters
    ----------
    material_name : str
        Name of the coolant. 
        Options are: 'lead', 'LBE', 'bismuth', 'NaK', 'sodium'
    reference : bool, optional
        If True, generate a reference database with 1000 points, 
        by default False
        
    Raises
    ----
    ValueError
        If the material name is not recognized
    """
    check_material(material_name)
    
    if reference:
        temp = np.linspace(*ENTHALPY_RANGE[material_name], REFERENCE_STEPS)
        file_name = material_name + REFERENCE_SUFFIX + DATA_EXTENSION
    else:
        temp = np.arange(*ENTHALPY_RANGE[material_name], TEMP_STEP)
        file_name = material_name + DATA_EXTENSION
        
    rho = np.zeros(len(temp))
    h = np.zeros(len(temp))
    corr_rho, corr_h = assign_correlations(material_name)
    for ii, T in enumerate(temp):
        rho[ii] = corr_rho(temperature = T)
        h[ii] = corr_h(temperature = T)

    np.savetxt(os.path.join(PATH_TO_DATA, file_name), np.vstack((rho, h)).T, 
               delimiter=',')


def assign_correlations(material_name: str) -> None:
    """
    Assign the density and enthalpy correlations based on the material name
    
    Parameters
    ----------
    material_name : str
        Name of the coolant. 
        Options are: 'lead', 'LBE', 'bismuth', 'NaK', 'sodium'
        
    Raises
    ----
    ValueError
        If the material name is not recognized
    """
    check_material(material_name)
    
    if material_name in MATERIAL_LBH:
        cool = MATERIAL_LBH[material_name](T = 700)
        return partial(prop_from_lbh15, cool=cool, 
                       prop='density'), \
            partial(prop_from_lbh15, cool=cool, 
                    prop='enthalpy')

    if material_name == 'sodium':
        return properties_Na.Mat_from_corr(prop='density'), \
            properties_Na.Mat_from_corr(prop='enthalpy')
    
    return properties_Na.Mat_from_corr(prop='density'), \
        properties_NaK.Mat_from_corr(prop='enthalpy')
        

def prop_from_lbh15(prop: str, cool: Type, temperature: float) -> float:
    """
    Get the property value from lbh15 library
    
    Parameters
    ----------
    prop : str
        Property name to be calculated
    cool : Type
        lbh15 class corresponding to the material
    temperature : float
        Temperature in K
        
    Returns
    -------
    float
        Property value
        
    Raises
    ----
    ValueError
        If the material name or property name is not recognized
    """    
    setattr(cool, 'T', temperature)
    return getattr(cool, LBH15_PROP[prop])


##############################################################################
#                            METHOD EVALUATION
##############################################################################
def eval_poly(reference: np.ndarray, data: np.ndarray) \
    -> tuple[dict[str, np.ndarray], int]:
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
    tuple[dict[str, np.ndarray], int]
        poly_results : dict[str, np.ndarray]
            Dictionary containing the average errors and time efforts for
            different polynomial degrees
        max_deg : int
            Maximum polynomial degree reached without RankWarning
    """
    poly_results = {'eave_poly': np.array([]),
                    'time_poly': np.array([])}

    for dd in range(1, DEG):
        print(f"Evaluating polynomium degree {dd}...")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', RankWarning)
                coeffs_rho2h = np.polyfit(data[:,0], data[:,1], deg=dd)
        except RankWarning:
            print(f"Polynomial degree {dd} issues a NumPy RankWarning. Stopping here.")
            break

        errors = eval_accuracy(reference, coeffs_rho2h=coeffs_rho2h)
        poly_results['eave_poly'] = np.append(poly_results['eave_poly'], 
                                              np.mean(errors))

        poly_results['time_poly'] = np.append(poly_results['time_poly'], 
                                            eval_time(coeffs_rho2h, 
                                                      reference[0,0]))
    return poly_results, dd-1


def eval_accuracy(reference: np.ndarray, coeffs_rho2h: np.ndarray) \
    -> np.ndarray:
    """
    Function to evaluate the accuracy of the function `method`
    
    Parameters
    ----------
    reference : np.ndarray
        Reference data to compare against
    coeffs_rho2h : np.ndarray
        Coefficients of the polynomium for rho to h conversion

    Returns
    -------
    np.ndarray
        Array containing the relative error with respect to the reference data.
    """
    rho_ref = reference[:,0]
    hh_ref = reference[:,1]

    res = poly_method(rho_ref, coeffs_rho2h)
    return np.abs((res - hh_ref) / hh_ref)


def calc_elapsed_time(coeffs_rho2h: float, rho: float) -> float:
    """
    Function to calculate the time effort needed for calculating the enthalpy
    from density using the function `poly_method`
    
    Parameters
    ----------
    coeffs_rho2h : float
        Coefficients of the polynomium for rho to h conversion
    rho : float
        Density of the coolant (kg/m3)

    Returns
    -------
    float
        Time effort in seconds
    """
    start_time = time.time()
    poly_method(rho, coeffs_rho2h)
    end_time = time.time()
    return end_time - start_time


def eval_time(coeffs_rho2h: np.ndarray, rho: float) -> float:
    """
    Function to estimate the average time effort of a function `method`
    
    Parameters
    ----------
    coeffs_rho2h : np.ndarray
        Coefficients of the polynomium for rho to h conversion
    rho : float
        Density of the coolant (kg/m3)

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
        elapsed_time = calc_elapsed_time(coeffs_rho2h, rho)
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
#                                POLY METHOD
##############################################################################
def poly_method(rho: np.ndarray, coeffs_rho2h: np.ndarray) -> np.ndarray:
    """
    Function to calculate the enthalpy from density using an interpolating 
    polynomium
    
    Parameters
    ----------
    rho: np.ndarray
        Density of the coolant (kg/m3)
    coeffs_rho2h : np.ndarray
        Coefficients of the polynomium for rho to h conversion

    Returns
    -------
    np.ndarray
        Enthalpy of the coolant (J/kg)
    """
    return np.polyval(coeffs_rho2h, rho)


##############################################################################
#                                PLOTTING
##############################################################################
def plot_polynomial_results(poly_results: dict[str, np.ndarray], 
                            material: str, max_deg: int) -> None:
    """
    Plot the evaluation of the polynomial method in terms of accuracy and
    time effort for different polynomial degrees
    
    Parameters
    ----------
    poly_results : dict[str, np.ndarray]
        Dictionary containing the evaluation results for different polynomial 
        degrees. The keys are "eave_poly" and "time_poly", representing the 
        average error and time effort, respectively, for polynomial 
        degrees from 1 to DEG-1
    material : str
        Material name
    max_deg : int
        Maximum polynomial degree reached without RankWarning
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(range(1, max_deg+1), poly_results['eave_poly'], '--o', 
                label='Ave. error polynomium')
    axs[0].set_xticks(range(1, max_deg+1))
    axs[0].set_xlim(1, max_deg)
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Polynomium Degree')
    axs[0].set_ylabel('Error [-]')
    axs[0].set_title('Accuracy')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(range(1, max_deg+1), poly_results['time_poly'], '--o', 
                label='Ave. time polynomium')
    axs[1].set_xlabel('Polynomium Degree')
    axs[1].set_xticks(range(1, max_deg+1))
    axs[1].set_xlim(1, max_deg)
    axs[1].set_ylabel('Time [s]')
    axs[1].set_title('Time Effort')
    axs[1].legend()
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[1].yaxis.get_offset_text().set_fontsize(10)
    axs[1].grid()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    plt.suptitle(material)
   # plt.savefig(os.path.join('results', material, 'polynomial_degree.png'))