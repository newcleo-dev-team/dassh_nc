"""Common constants and paths for DASSH modules."""
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
"""Root folder of the dassh package."""
DATA_FOLDER = 'data'
"""Folder where data files are stored."""
h2T_COEFF_FILE = 'coeffs_h2T.csv'
"""File name for the enthalpy-to-temperature conversion coefficients."""
T2h_COEFF_FILE = 'coeffs_T2h.csv'
"""File name for the temperature-to-enthalpy conversion coefficients."""
SQRT3 = np.sqrt(3)
"""Square root of 3."""
SQRT3OVER3 = np.sqrt(3) / 3
"""Square root of 3 divided by 3."""
Q_P2SC = np.array([0.166666666666667, 0.25, 0.166666666666667])
"""Fraction of pin surface in contact with each type of subchannel."""