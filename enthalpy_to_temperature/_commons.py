import numpy as np
"""Module with common imports and variables for the enthalpy to temperature conversion methods."""

MATERIAL: str = 'lead'
DELTA_H: np.ndarray = 100.0 * np.ones(18)  
T_IN: np.ndarray = 700.0 
TEMP_COOLANT_INT: np.ndarray = 700.0 * np.ones(18)
SIZES: list[int] = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
DEG: int = 18
ENTHALPY_COEFF = {
    'lead': [176.2, -2.4615e-2, 5.147e-6, 1.524e6]
}
TOL: float = 1e-4
NEWTON_MAXITER: int = 100
TIME_MAXITER: int = 1e4
TIME_MINITER: int = 300