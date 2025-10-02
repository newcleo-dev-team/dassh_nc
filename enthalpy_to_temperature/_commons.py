"""Module with common imports and variables for the enthalpy to temperature 
   conversion methods."""
import numpy as np


N_SC: int = 18
"""Number of subchannels in the assembly"""
MATERIAL: str = 'lead'
"""Coolant material"""
DELTA_H: np.ndarray = 100.0 * np.ones(N_SC)  
"""Enthalpy change (J/kg)"""
T_IN: np.ndarray = 700.0 
"""Initial temperature of the coolant (K)"""
TEMP_COOLANT_INT: np.ndarray = T_IN * np.ones(N_SC)
"""Initial temperature of the coolant for each subchannel (K)"""
DB_SIZES: list[int] = [
    500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
    ]
"""Sizes of the tabulated databases for the table method"""
DEG: int = 18
"""Maximum degree of the polynomium for the poly method"""
ENTHALPY_COEFF = {
    'lead': [176.2, -2.4615e-2, 5.147e-6, 1.524e6] 
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
DB_PATH_PREFIX: str = "lead_"
"""Prefix for the path to the data files"""
DB_PATH_SUFFIX: str = ".csv"
"""Suffix for the path to the data files"""