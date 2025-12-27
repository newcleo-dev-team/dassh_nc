"""Module with common variables and functions for the solver evaluation"""
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import time


#############################################################################
# Variables
#############################################################################
TIME_MAXITER: int = 1e5
"""Maximum number of iterations for the time evaluation"""
TIME_MINITER: int = 300
"""Minimum number of iterations for the time evaluation"""
TOL: float = 1e-4
"""Tolerance for the time evaluation convergence"""
SC_ID: np.ndarray = np.array([1, 8, 7, 26, 27])
"""IDs of subchannels used for the test cases"""
EXP_RESULTS: np.ndarray = np.array([1.16390336, 1.07212222, 1.01669825, 
                                    0.89720512, 0.70937944])
"""Experimental results for the test cases"""
EXP_Z: float = 0.802 
"""Experimental axial position for the test cases (m)"""
OUTPUT_COOL_NAME: str = "temp_coolant_int.csv"
"""Name of the DASSH output file with the coolant temperatures"""
OUTPUT_NAME: str = "dassh.out"
"""Name of the DASSH output file"""
TOUT_POSITION: list[int] = [193, 4]
"""Position of the outlet temperature in the DASSH output file"""
T_IN: float = 473.25
"""Inlet temperature for the test cases (K)"""
MRK_DICT: dict = {
    'numpy': 's',
    'scipy': 'o',
    'greene': '^'
}
"""Marker dictionary for plotting different methods"""


#############################################################################
# Functions
#############################################################################
def calc_elapsed_time(path: str) -> float:
    """
    Function to calculate the time effort needed for retrieving the output of 
    a single call of the function `method`
    
    Parameters
    ----------
    path : str
        Path to the input file for the DASSH run
    
    Returns
    -------
    float
        Time effort in seconds
    """
    start_time = time.time()
    subprocess.run(["dassh", os.path.join(path, "input.txt")],
                   check=True, 
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL
                   )
    end_time = time.time()
    return end_time - start_time


def eval_time(path: str) -> float:
    """
    Function to estimate the average time effort of a function `method`
    
    Parameters
    ----------
    path : str
        Path to the input file for the DASSH run

        
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
        elapsed_time = calc_elapsed_time(path)
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


def get_results(path: str) -> np.ndarray:
    """
    Function to retrieve the DASSH results from a given input file path
    
    Parameters
    ----------
    path : str
        Path to the input file for the DASSH run
   
    Returns
    -------
    np.ndarray
        Array with the DASSH results for the specified subchannels
    """
    output_path = os.path.join(path, OUTPUT_COOL_NAME)
    zz = np.genfromtxt(output_path, delimiter=',')[:,1]
    temp = np.genfromtxt(output_path, delimiter=',')[:,3:]
    TT = []
    for sc in SC_ID:
        TT.append(np.interp(EXP_Z, zz, temp[:,sc-1], left=None, right=None))
    return TT


def read_Tout(path: str) -> float:
    """
    Read the outlet temperature from the DASSH output file
    
    Parameters
    ----------
    path : str
        Path to the output file folder
    """
    with open(os.path.join(path, OUTPUT_NAME)) as f:
        lines = f.readlines()
        line = lines[TOUT_POSITION[0]].strip()
        parts = line.split()
        return float(parts[TOUT_POSITION[1]])
    

def plot_results(res_dict):
    """
    Plot the results of the different methods against the experimental data
    
    Parameters
    ----------
    res_dict : dict
        Dictionary with the results for each method
    """
    plt.figure(figsize=(8,6))
    for method, results in res_dict.items():
        plt.plot([str(sc) for sc in SC_ID], results, marker=MRK_DICT[method], 
                 linestyle='--', label=f'{method} solver')
    
    plt.plot([str(sc) for sc in SC_ID], EXP_RESULTS, marker='x',
             color='k', label='Experimental data')
    plt.xlabel('SC ID')
    plt.ylabel(r'$\Delta T_{norm}$')
    plt.title('Solver Evaluation Against Experimental Data')
    plt.legend()
    plt.grid()
    plt.show()