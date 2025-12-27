"""Auxiliary module containing common variables and functions"""
import numpy as np
import os 
from pathlib import Path
import shutil
import subprocess

MM: list[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
"""Mass flowrates (kg/s)"""

SIMULATION_FOLDER: str = "simulations"
"""Folder to store simulation results"""

INPUT_FILE: str = "input.txt"
"""Name of the base input file"""

POWER_FILE: str = "power.csv"
"""Name of the base power file"""

OUTPUT_FILE: str = "dassh.out"
"""Name of the output file"""

BASE_INPUT_PATH: str = os.path.join('base_input', INPUT_FILE)
"""Path to the base input file"""

BASE_POWER_PATH: str = os.path.join('base_input', POWER_FILE)
"""Path to the base power file"""

LINES_TO_MODIFY: dict[str, str] = {
    "flowrate": "        xx09 = 1, 1, 1, FLOWRATE=13.88",
    "ff": "    ff_variable = False"
    }
"""Lines to be modified in the base input file"""

MODIFIED_LINES: dict[str, str] = {
    "flowrate": "        xx09 = 1, 1, 1, FLOWRATE=",
    "ff": "    ff_variable = True"
    }
"""Modified lines for the input file"""

LINE_NUM: int = 129
"""Line number in the output file containing the pressure drop"""

PART_NUM: int = 5
"""Part number in the line containing the pressure drop"""


def modify_input(base_input: str, ff_variable: bool, mfr: float) -> str:
    """
    Modify the base input file for the specific correlation function
    and mass flowrate

    Parameters
    ----------
    base_input : str
        Content of the base input file as a string
    ff_variable : bool
        Indicates whether friction factor is variable
    mfr : float
        Mass flowrate to set in the input file

    Returns
    -------
    str
        Modified input file content as a string
    """
    modified_input = base_input
    modified_input = modified_input.replace(LINES_TO_MODIFY['flowrate'],
                                            MODIFIED_LINES['flowrate'] + 
                                            f"{mfr}")
    if ff_variable:
        modified_input = modified_input.replace(LINES_TO_MODIFY['ff'],
                                                MODIFIED_LINES['ff'])
    return modified_input


def get_mass_flowrate_and_exp_results() -> np.ndarray:
    """
    Get the mass flowrate 
    
    Returns
    -------
    tuple[np.ndarray]
        Mass flowrates (kg/s) 
        Measured pressure drop (MPa) 
    """ 
    res = np.genfromtxt(os.path.join(SIMULATION_FOLDER, EXP_RESULTS),
                        delimiter=',')
    return res[:, 0], res[:, 1]


def create_dir_and_input(ii: int, ff_type: str, base_power: str, 
                         base_input: str, mfr: float) -> str:
    """
    Create a directory for the current combination of correlation functions
    and write the modified input and power files into it

    Parameters
    ----------
    ii : int
        Index of the current mass flowrate
    ff_type : str
        Type of friction factor ('ff_constant' or 'ff_variable')
    base_power : str
        Content of the base power file as a string
    base_input : str
        Content of the base input file as a string
    mfr : float
        Mass flowrate to set in the input file

    Returns
    -------
    str
        Path to the created directory
    """
    dir_name = os.path.join(SIMULATION_FOLDER, ff_type, str(ii))
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    Path(dir_name).mkdir(exist_ok=False, parents=True)
    if ff_type == 'ff_constant':
        modified_input = modify_input(base_input, False, mfr)
    else:
        modified_input = modify_input(base_input, True, mfr)
        
    with open(os.path.join(dir_name, INPUT_FILE), 'w') as f:
        f.write(modified_input)
    with open(os.path.join(dir_name, POWER_FILE), 'w') as f:
        f.write(base_power)
    return dir_name


def collect_results(output_path: str) -> float:
    """
    Collect the pressure drop from the output file
    
    Parameters
    ----------
    output_path : str
        Path to the output file
        
    Returns
    -------
    float
        Pressure drop (MPa)
    """
    with open(output_path) as f:
        lines = f.readlines()
        parts = lines[LINE_NUM].strip().split()
        return float(parts[PART_NUM])
    
    
def run_simulations(ff_type: str) -> np.ndarray:
    """
    Run all simulations for the different mass flowrates and 
    collect the pressure drop results
    
    Parameters
    ----------
    ff_type : str
        Type of friction factor ('ff_constant' or 'ff_variable')
        
    Returns
    -------
    np.ndarray
        Simulated pressure drops (MPa)
    """
    # Read base input and power files
    with open(os.path.join(BASE_INPUT_PATH), 'r') as f:
        base_input = f.read()
    with open(os.path.join(BASE_POWER_PATH), 'r') as f:
        base_power = f.read()
    deltaP_sim = np.zeros_like(MM)
    for ii, mfr in enumerate(MM):
        # Create simulation directory and input files
        dir_path = create_dir_and_input(ii, ff_type, base_power, 
                                        base_input, mfr)
        # Run DASSH simulation
        subprocess.run(["dassh", os.path.join(dir_path, "input.txt")],
                   check=True, 
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL
                   )
        # Collect results
        output_path = os.path.join(dir_path, OUTPUT_FILE)
        deltaP_sim[ii] = collect_results(output_path)
    return deltaP_sim