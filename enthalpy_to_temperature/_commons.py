"""Module with common imports and variables for the enthalpy to temperature 
   conversion methods."""
import numpy as np
from lbh15 import Lead, Bismuth, LBE

N_SC: int = 18
"""Number of subchannels in the assembly"""
DELTA_H: np.ndarray = 100.0 * np.ones(N_SC)  
"""Enthalpy change (J/kg)"""
T_IN: np.ndarray = 700.0 
"""Initial temperature of the coolant (K)"""
TEMP_COOLANT_INT: np.ndarray = T_IN * np.ones(N_SC)
"""Initial temperature of the coolant for each subchannel (K)"""
DB_SIZES: dict[str, list[int]] = {
    'lead': [1000, 2000, 3000, 4000, 5000],
    'sodium': [2500, 5000, 7500, 10000],
    'nak': [1000, 2000, 3000, 4000, 5000],
    'lbe': [1000, 2000, 3000, 4000, 5000],
    'bismuth': [1000, 2000, 3000, 4000, 5000]
}
"""Sizes of the tabulated databases for the table method"""
DEG: int = 18
"""Maximum degree of the polynomium for the poly method"""
ENTHALPY_COEFF: dict[str, list[float]] = {
    "sodium": [1.6582e3, -4.2395e-1, 1.4847e-4, 2.9926e6],
    "lead": [176.2, -2.4615e-2, 5.147e-6, 1.524e6],
    "lbe": [164.8, -1.97e-2, 4.167e-6, 4.56e5],
    "bismuth": [118.2, 2.967e-3, 0.0, -7.183e6],
    "nak": [971.3376, -0.18465, 1.1443e-4, 0.0]
}
"""Coefficients for the enthalpy change correlation"""
TOL: float = 1e-4
"""Tolerance for the Newton method"""
NEWTON_MAXITER: int = 100
"""Maximum number of iterations for the Newton method"""
TIME_MAXITER: int = 1e5
"""Maximum number of iterations for the time evaluation"""
TIME_MINITER: int = 300
"""Minimum number of iterations for the time evaluation"""
MATERIAL_LBH = {
            'lead': Lead,
            'bismuth': Bismuth,
            'lbe': LBE
        }
"""Dictionary mapping the material name to the corresponding lbh15 class"""
DB_PATH_SUFFIX: str = ".csv"
"""Suffix for the path to the data files"""