"""Common constants and paths for DASSH modules."""
import os
import numpy as np
from lbh15 import lead_properties, bismuth_properties, lbe_properties
from lbh15 import Lead, Bismuth, LBE
from types import ModuleType
from typing import Type

ROOT: str = os.path.dirname(os.path.abspath(__file__))
"""Root folder of the dassh package"""

DATA_FOLDER: str = 'data'
"""Folder where data files are stored"""

h2T_COEFF_FILE: str = 'coeffs_h2T.csv'
"""File name for the enthalpy-to-temperature conversion coefficients"""

rho2h_COEFF_FILE: str = 'coeffs_rho2h.csv'
"""File name for the density-to-enthalpy conversion coefficients"""

SQRT3: float = np.sqrt(3)
"""Square root of 3"""

SQRT3OVER3: float = np.sqrt(3) / 3
"""Square root of 3 divided by 3"""

Q_P2SC: np.ndarray = np.array([0.166666666666667, 0.25, 0.166666666666667])
"""Fraction of pin surface in contact with each type of subchannel"""

MATERIAL_LBH: dict[str, Type] = {
    'lead': Lead,
    'bismuth': Bismuth,
    'lbe': LBE}
"""Mapping between material names and lbh15 classes"""

PROP_LBH15: dict[str, ModuleType] = {'lead': lead_properties,
                                     'bismuth': bismuth_properties,
                                     'lbe': lbe_properties}
"""Mapping between material names and lbh15 property modules"""

LBH15_PROPERTIES: list[str] = ['rho', 'cp', 'mu', 'k']
"""Property names as used in lbh15"""

PROPS_NAME: list[str] = ['density', 'heat_capacity', 'viscosity', 
                         'thermal_conductivity']
"""Property names as used in dassh"""

PROPS_NAME_FULL: dict[str, str] = dict(zip(PROPS_NAME, LBH15_PROPERTIES))
"""Mapping between property names as used in dassh and in lbh15"""

BUILTIN_COOLANTS: list[str] = ['sodium', 'nak', 'lead', 'lbe', 'bismuth']
"""Builtin coolant materials"""

MATERIAL_NAMES: list[str] = BUILTIN_COOLANTS + ['potassium', 'water', 
                                                'ss304', 'ss316']
"""Material names built-in in dassh"""

AMBIENT_TEMPERATURE: float = 298.15
"""Default temperature [K]"""

GRAVITY_CONST: float = 9.81
"""Gravity acceleration [m/s^2]"""

MIX_CON_VERBOSE_OUTPUT: list[str] = [
    '----------------------------------------------------------------',
    'info', 'Iter.    Error density       Error velocity       Error pressure'
]
"""Verbose output header for mixed convection region solver"""

MC_MAX_ITER: int = 10
"""Maximum number of iterations for mixed convection region solver"""