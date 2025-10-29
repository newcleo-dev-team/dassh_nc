"""
Module with common values for density to enthalpy conversion
"""
from lbh15 import Lead, LBE, Bismuth
from typing import Type


ENTHALPY_RANGE: dict[str, tuple[float]] = {
    'lead': (600.7, 2000.0),
    'LBE': (400.1, 1926.9),
    'bismuth': (544.7, 1830.9),
    'NaK': (323.2, 1423.1),
    'sodium': (371.0, 2000.0)
}
"""Validity range of enthalpy correlation of each fluid"""

REFERENCE_STEPS: int = 5000
"""Number of points in the reference database"""

MATERIAL_LBH: dict[str, Type] = {
    'lead': Lead,
    'bismuth': Bismuth,
    'LBE': LBE}
"""Mapping between material names and lbh15 classes"""

LBH15_PROP: dict[str, str] = {
    'density': 'rho',
    'enthalpy': 'h'
}
"""Mapping between property names and lbh15 methods"""

PATH_TO_DATA: str = 'data'
"""Path to the folder containing the density-enthalpy databases"""

REFERENCE_SUFFIX: str = '_ref'
"""Suffix for reference database files"""

DATA_EXTENSION: str = '.csv'
"""Extension of the density-enthalpy database files"""

TEMP_STEP: float = 0.1
"""Temperature step for database generation"""

DEG: int = 15
"""Maximum degree of the polynomial fit"""

TIME_MAXITER: int = 1e5
"""Maximum number of iterations for the time evaluation"""

TIME_MINITER: int = 300
"""Minimum number of iterations for the time evaluation"""

TOL: float = 1e-4
"""Tolerance for the time evaluation convergence"""
