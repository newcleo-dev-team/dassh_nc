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
    'Iter.   Error density      Error velocity     Error pressure'
]
"""Verbose output header for mixed convection region solver"""

MC_MAX_ITER: int = 10
"""Maximum number of iterations for mixed convection region solver"""

MIXED_CONV_PROP_TO_UPDATE: list[str] = ['viscosity', 'thermal_conductivity', 
                                        'heat_capacity']
"""Material properties to update in mixed convection solver"""
RHO2H_LBH15_COEFFS: dict[str, list[float]] = {
    'lead': [11441.0, 1.2795, 176.2, -2.4615e-2, 5.147e-6, 1.524e6, 600.6],
    'lbe': [11065.0, 1.293, 164.8, -1.97e-2, 4.167e-6, 4.56e5, 398.0],
    'bismuth': [10725, 1.22, 118.2, 2.967e-3, 0, -7.183e6, 544.6]
}
"""Coefficients for density to enthalpy conversion correlation functions for
lead, lbe, and bismuth. Format: [a0, a1, b0, b1, b2, b3, Tm] for the equations:

    density = a0 - a1*T 
    enthalpy = b0*(T-Tm) + b1*(T^2 - Tm^2) + b2*(T^3 - Tm^3) + b3*(1/T - 1/Tm)
 
 where Tm is the melting temperature of the material in K.
"""