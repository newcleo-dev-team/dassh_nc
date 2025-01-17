########################################################################
# Copyright 2021, UChicago Argonne, LLC
#
# Licensed under the BSD-3 License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a
# copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
########################################################################
"""
date: 2021-11-24
author: matz
Containers to hold and update material properties
"""
########################################################################
import os
import copy
import numpy as np
from dassh.logged_class import LoggedClass
from lbh15 import Lead, Bismuth, LBE
import sympy as sp


_ROOT = os.path.dirname(os.path.abspath(__file__))


_BUILTINS = ['ht9', 'ss316', 'ss304', 'd9', 'bismuth', 'lbe', 'lead',
             'nak', 'potassium', 'sodium', 'water' 'ht9_se2anl',
             'ht9_se2anl_425', 'sodium_se2anl', 'sodium_se2anl_425']


class Material(LoggedClass):
    """Container to hold and update material properties

    Parameters
    ----------
    name : str
        Material name; if no 'path_to_data' input provided, this must
        be one of the built-in DASSH materials
    temperature : float (optional)
        Temperature (K) at which to evaluate material properties
        (default = 298.15 K)
    from_file : str (optional)
        Path to user-defined material properties data or correlation
        coefficients (CSV)
    coeff_dict : dict (optional)
        User-defined material property correlation coefficients

    Notes
    -----
    Material properties data file requirements
    - Format must be comma-separated values (CSV)
    - The first row is the header, with "temperature" as the first
      column. The remaining columns can be in any order, but they
      depend on the type of material being defined:
        Coolant: "density", "viscosity", "heat_capacity",
                 "thermal_conductivity"
        Structural: "density", "heat_capacity", "thermal_conductivity"
    - Units:
        - Temperature: K
        - Density: kg/m^3
        - Viscosity: Pa-s
        - Heat capacity (Cp): J/kg-K
        - Thermal conductivity (k): W/m-K

    """
    MATERIAL_LBH = {
            'lead': Lead,
            'bismuth': Bismuth,
            'lbe': LBE
        }
    LBH15_PROPERTIES = ['rho', 'cp', 'mu', 'k']
    PROP_NAME = dict(zip(['density', 'heat_capacity', 'viscosity', 'thermal_conductivity'], LBH15_PROPERTIES)) 
    
    def __init__(self, name, temperature=298.15, from_file=None,
                 corr_dict=None, lbh15_correlations = None,
                 use_correlation = False):
        LoggedClass.__init__(self, 0, f'dassh.Material.{name}')
        self.name = name
        self.temperature = temperature
        # Read data into instance; use again to update properties later       
        if from_file:
            self.read_from_file(from_file)
        elif corr_dict:
            self._define_from_user_corr(corr_dict)
        elif use_correlation and self.name in Material.MATERIAL_LBH.keys():
            self._define_from_lbh15(lbh15_correlations) 
        elif use_correlation and self.name in ['sodium', 'nak']:
            self._define_from_correlation()
        else:
            try:
                self._define_from_table(None)
            except OSError:
                if self.name.lower() in globals():
                    self._define_from_user_corr(None)
                else:
                    msg = f'Cannot find properties for material {name}'
                    self.log('error', msg)

        # Update properties based on input temperature
        self.update(temperature)

    def read_from_file(self, path):
        """Determine whether a user-provided CSV file is providing
        correlation coefficients or tabulated data and read them in"""
        
        with open(path, 'r') as f:
            data = f.read()

        data = data.splitlines()
        # Tabulated data has cols ['temperature', prop 1, prop 2, ...]
        if '=' in data[0]:
            line1 = data[0].split('=')
        else:
            line1 = data[0].split(',')
            line1 = [l.lower() for l in line1]
        # print(line1)
        # print(line1[0] == 'temperature')
        # print('thermal_conductivity' in line1)
        if line1[0] == 'temperature' and 'thermal_conductivity' in line1:
            self._define_from_table(path)
        elif '=' in data[0]:
            self._define_from_user_corr(self._corr_from_file(path))
        else:  
            cdict = self._corr_from_file(path)    
            self._define_from_user_corr(cdict)
            
            
    def _define_from_table(self, path):
        """Define correlation by interpolation of data table"""
        if not path:
            path = os.path.join(_ROOT, 'data', self.name + '.csv')

        # data = pxd.read_csv(path, header=0)
        data = np.genfromtxt(path, skip_header=1, delimiter=',',
                             missing_values=0.0)
        # Get the columns with string parsing
        with open(path, 'r') as f:
            cols = f.read().splitlines()[0].split(',')[1:]

        # Check that all values are greater than zero, 
        # if file is provided by the user. 
        if np.any(data <= 0.0):
            msg = f'Non-positive values detected in material data {path}'
            self.log('error', msg)

        # define property attributes based on temperature
        # We store only the interpolations, not the data tables
        self._data = {}
        # Pull temperatures over which to interpolate; eliminate negative vals
        x = data[:, 0]         # temperatures over which to interpolate
        for i in range(len(cols)):
            y = data[:, i + 1]
            x2 = x[y > 0]  # Need to ignore zeros in dependent var
            y2 = y[y > 0]  # Now filter from dependent var
            if not np.all(np.diff(x) > 0):
                msg = f'Non strictly increasing temperature values detected in material data {path}'
                self.log('error', msg)
            self._data[cols[i]] = _MatInterp(x2, y2)
    
    def _define_from_lbh15(self, lbh15_correlations):
        """Define correlation by using lbh15"""    
        self._data = {}
        cool_lbh15 = Material.MATERIAL_LBH[self.name](T=self.temperature)
        
        for property in Material.PROP_NAME.keys():
            correlations = cool_lbh15.available_correlations(Material.LBH15_PROPERTIES)
            for property in Material.PROP_NAME.keys():
                corr_name = lbh15_correlations[Material.PROP_NAME[property]]
                if corr_name and corr_name not in correlations[Material.PROP_NAME[property]]:
                    msg = f'Correlation {corr_name} for {Material.PROP_NAME[property]} not available for {self.name}'
                    self.log('error', msg)
                if corr_name in correlations[Material.PROP_NAME[property]]:
                    cool_lbh15.change_correlation_to_use(Material.PROP_NAME[property], corr_name)
        for property in Material.PROP_NAME.keys():
            self._data[property] = _Matlbh15(Material.PROP_NAME[property], cool_lbh15)           
                                                                    
    @staticmethod
    def _coeff_from_table(path):
        """Read correlation coefficients from CSV file"""
        cdict = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.split(',')
                cdict[line[0]] = np.array([float(c) for c in line[1:]])
        return cdict
    
    @staticmethod
    def _corr_from_file(path):
        """Read correlation from file"""
        cdict = {}
        with open(path, 'r') as f:
            for line in f:
                if '=' in line:
                    line = line.split('=')
                    cdict[line[0].strip(' ')] = str(line[1])
                else:
                    line = line.split(',')
                    cdict[line[0]] = np.array([float(c) for c in line[1:]])
        return cdict
    
    def _define_from_user_corr(self, corr_dict):
        """Define correlation from array of polynomial coefficients"""
        if not corr_dict:
            corr_dict = globals()[self.name.lower()]
        self._data = {}
        for property in corr_dict.keys():
            if isinstance(corr_dict[property], str):
                try:
                    expr = sp.sympify(corr_dict[property])
                    corr_symbols = [str(fs) for fs in expr.free_symbols]
                except Exception as e:
                    msg = f'Invalid correlation for {self.name} {property}: {e}'
                    self.log('error', msg)
                if corr_symbols != ['T'] and corr_symbols != []:
                    msg = f'Correlation for {self.name} {property} contains invalid symbols'
                    self.log('error', msg)            
                self._data[property] = _MatUserCorr(property, expr) 
            else:
                self._data[property.lower()] = \
                    _MatPoly(corr_dict[property.lower()][::-1])
                
    def _define_from_correlation(self):
        """Define Na or NaK properties from correlation"""
        if self.name == 'sodium':
            import dassh.correlations.properties_Na as corr  
        elif self.name == 'nak':
            import dassh.correlations.properties_NaK as corr
        
        self._data = {}
        for property in self.PROP_NAME.keys():
            self._data[property] = corr.mat_from_corr(property)
    
    @property
    def name(self):
        return self._name

    @property
    def temperature(self):
        return self._temperature

    @property
    def heat_capacity(self):
        return self._heat_capacity

    @property
    def density(self):
        return self._density

    @property
    def thermal_conductivity(self):
        return self._thermal_conductivity

    @property
    def viscosity(self):
        return self._viscosity

    @property
    def beta(self):
        """Calculate volume expansion coefficient; used only in
        modified Grashof number calculation"""
        if hasattr(self, '_beta'):  # return constant property
            return self._beta
        else:  # otherwise, calculate two densities
            rho1 = self._data['density'](self.temperature)
            rho2 = self._data['density'](self.temperature - 1)
            beta = -1 * (rho1 - rho2) / rho1
            assert beta != 0.0
            return beta

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise TypeError('Material name must be of type "str"')
        self._name = name

    @temperature.setter
    def temperature(self, temperature):
        if any_nonpositive(temperature):
            msg = (f'Material "{self.name}" temperature must '
                   f'be > 0; given {temperature} K')
            self.log('error', msg)
        self._temperature = temperature

    @heat_capacity.setter
    def heat_capacity(self, heat_capacity):
        if any_nonpositive(heat_capacity):
            msg = (f'Material "{self.name}" heat capacity '
                   f'must be > 0; given {heat_capacity}')
            self.log('error', msg)
        self._heat_capacity = heat_capacity

    @density.setter
    def density(self, density):
        if any_nonpositive(density):
            msg = (f'Material "{self.name}" density '
                   f'must be > 0; given {density}')
            self.log('error', msg)
        self._density = density

    @thermal_conductivity.setter
    def thermal_conductivity(self, thermal_conductivity):
        if any_negative(thermal_conductivity):
            msg = (f'Material "{self.name}" thermal conductivity must '
                   f'be >= 0; given {thermal_conductivity}')
            self.log('error', msg)
        self._thermal_conductivity = thermal_conductivity

    @viscosity.setter
    def viscosity(self, viscosity):
        if any_nonpositive(viscosity):
            msg = (f'Material "{self.name}" dynamic viscosity '
                   f'must be > 0; given {viscosity}')
            self.log('error', msg)

        self._viscosity = viscosity

    @beta.setter
    def beta(self, beta):
        # Only used for constant-property materials
        self._beta = beta
        
    def update(self, temperature):
        """Update material properties based on new bulk temperature"""
        self.temperature = temperature
        for property in self._data.keys():
            setattr(self, property, self._data[property](temperature))
                
                
    def clone(self, new_temperature=None):
        """Create a clone of this material with a new temperature
        if requested"""
        clone = copy.copy(self)
        clone._temperature = copy.deepcopy(self._temperature)
        # clone._data = {k: v for k, v in self._data.items()}
        clone._data = copy.deepcopy(self._data)
        if new_temperature is not None:
            clone.update(new_temperature)
        return clone


class _MatInterp(object):
    """Interpolation object for material properties"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.const = False
        if np.all(y == y[0]):
            self.const = True

    def __call__(self, x):
        """Return the interpolated result"""
        if self.const:
            return self.y[0]
        else:
            return np.interp(x, self.x, self.y)


class _MatPoly(object):
    """Polynomial evaluator for material properties"""

    def __init__(self, coeffs):
        self.coeffs = coeffs
        # Flag if constant mat properties - just return, don't eval
        if len(coeffs) == 1:
            self.const = True
        else:
            self.const = False

    def __call__(self, x):
        """Evaluate the polynomial at the given x"""
        if self.const:
            return self.coeffs[0]
        else:
            y = 0.0
            for i, v in enumerate(self.coeffs):
                y *= x
                y += v
            return y

class _Matlbh15(object):
    """
    lbh15 object for material properties
    
    Parameters
    ----------
    prop: str
        Property to calculate
    cool_lbh15: lbh15 object
        lbh15 object representative of the liquid metal to which the property is related
    """
    def __init__(self, prop, cool_lbh15):
        self.prop = prop
        self.cool_lbh15 = cool_lbh15
    def __call__(self, temperature):
        if type(temperature) is np.ndarray:
            result = np.zeros(len(temperature))
            for ii in range(len(temperature)):
                setattr(self.cool_lbh15, 'T', temperature[ii])
                result[ii] = getattr(self.cool_lbh15, self.prop)
            return result
        setattr(self.cool_lbh15, 'T', temperature)
        return getattr(self.cool_lbh15, self.prop)
    

class _MatUserCorr(object):
    """
    User-defined correlation object for material properties
    
    Parameters
    ----------
    prop: str
        Property to calculate
    corr: str
        User-defined correlation to use
    """
    def __init__(self, prop: str, expr: str):
        self.prop: str = prop
        self.expr: str = expr
        self.T = sp.symbols('T')
    def __call__(self, temperature):
        if isinstance(temperature, np.ndarray):
            result = np.zeros(len(temperature))
            for ii in range(len(temperature)):
                result[ii] = float(self.expr.subs(self.T,temperature[ii]).evalf())
            return result
        return float(self.expr.subs(self.T,temperature).evalf())
    
class _MatTracker(object):
    """Keep track of changes in coolant properties to indicate when
    to update correlated parameters, if applicable

    Parameters
    ----------
    mat : DASSH Material object
        DASSH coolant object with temperature-dependent properties
    tol : float
        Relative change in any property above which correlated
        parameters should be updated

    Notes
    -----
    Material properties are updated every step. However, if material
    properties are constant, or if the temperature has changed very
    little such that the change in material properties is small, there
    is no need to perform the more costly updates to the correlated
    parameters (mixing, flowsplit, friction, HTC).

    This object tracks changes in material properties to limit the
    frequency with which correlated parameters are updated. This will
    limit computational expense in the regions where temperatures are
    not changing (e.g. in the radial and axial periphery of the core)
    and focus it in the regions where temperatures are changing more
    rapidly (in the fuel).

    """
    def __init__(self, mat, tol):
        """Initialize _MatTracker object"""
        self._dat0 = [mat.viscosity,
                      mat.density,
                      mat.heat_capacity,
                      mat.thermal_conductivity]
        self._dat = [0.0, 0.0, 0.0, 0.0]
        self._tol = tol
        self._count = 0
        self.recalculate_params = False

    def update(self, mat):
        """Update state, check whether to update correlated parameters"""
        self._dat = [mat.viscosity,
                     mat.density,
                     mat.heat_capacity,
                     mat.thermal_conductivity]
        for i in range(4):
            if abs(self._dat[i] - self._dat0[i]) / self._dat0[i] > self._tol:
                self.recalculate_params = True
                self._count += 1

    def reset(self):
        self._dat0 = self._dat
        self._dat = []
        self.recalculate_params = False


def any_negative(value):
    """Confirm that value(s) are nonnegative (zero allowed)"""
    try:
        return any(value < 0)
    except TypeError:
        return value < 0


def any_nonpositive(value):
    """Confirm that value(s) are positive (zero not allowed)"""
    try:
        return any(value <= 0)
    except TypeError:
        return value <= 0


########################################################################
# STRUCTURAL MATERIALS
# Only need thermal conductivity
########################################################################
# HT9: T < 1030 K; otherwise [12.027, 1.218e-2]
ht9 = {'thermal_conductivity': [17.622, 2.428e-2, -1.696e-5]}  # [1, 2]

# ----------------------------------------------------------------------
# Staninless steel 316
ss316 = {'thermal_conductivity': [6.308, 2.716e-2, -7.301e-6]}  # [1, 5]

# ----------------------------------------------------------------------
# Stainless steel 304
ss304 = {'thermal_conductivity': [8.116e-2, 1.618e-4]}  # [6]

# ----------------------------------------------------------------------
# D9 (294 < T < 1088 K)
d9 = {'thermal_conductivity': [8.25795, 1.94121e-2, -3.24027e-6]}  # [4]

# ----------------------------------------------------------------------
# HT9 (identical thermal conductivity to SE2ANL)
ht9_se2anl = {'thermal_conductivity': [23.663354319, 4.01774e-3]}

# ----------------------------------------------------------------------
# HT9 (identical thermal conductivity to SE2ANL; constant @ 425C
ht9_se2anl_425 = {'thermal_conductivity': [26.4683395]}

# # HT9
# # Parameter applicability:
# # Thermal conductivity: T < 1030 K; otherwise [12.027, 1.218e-2]
# #   From MFH: k_ht9 = [29.65, -6.668e-2, 2.184e-4, -2.527e-7, 9.621e-11]
# # Heat capacity: T < 800 K; otherwise [70.0, 0.6]
# # Density: 273.15 < T < 1073.15 K
# ht9 = {'thermal_conductivity': [17.622, 2.42e-2, -1.696e-5],  # [1, 2]
#        'heat_capacity': [416.66667, 0.166667],                # [1, 3]
#        'density': [7861.85705, -3.07e-4]                      # [4]
#        }
#
# # ----------------------------------------------------------------------
# # Staninless steel 316
# ss316 = {'thermal_conductivity': [6.308, 2.716e-2, -7.301e-6],  # [1, 5]
#          'heat_capacity': [428.6, 0.1816],                      # [1, 5]
#          'density': [8084.2, -0.42086, -3.8942e-5]              # [6]
#          }
#
# # ----------------------------------------------------------------------
# # Stainless steel 304
# ss304 = {'thermal_conductivity': [8.116e-2, 1.618e-4],  # [6]
#          'heat_capacity': [469.4448, 0.1348085],        # [6]
#          'density': [7984.1, -0.26506, -1.1580e-4]      # [6]
#          }
#
# # ----------------------------------------------------------------------
# # D9
# # Thermal conductivity: 294 < T < 1088 K
# # Heat capacity
# # Density: 573.15 < T < 1073.15
# d9 = {'thermal_conductivity': [8.25795, 1.94121e-2, -3.24027e-6],  # [4]
#       'heat_capacity': [431.0, 0.177],                             # [7]
#       'density': [8097.4545, 0.43]                                 # [4]
#       }
#
# # ----------------------------------------------------------------------
# # HT9 with identical thermal conductivity to SE2ANL
# ht9_se2anl = {'thermal_conductivity': [23.663354319, 4.01774e-3],
#               'heat_capacity': ht9['heat_capacity'],
#               'density': ht9['density']}

# ----------------------------------------------------------------------
# References
# ----------
# [1]  J. D. Hales et al. "BISON Theory Manual: The Equations Behind
#      Nuclear Fuel Analysis". INL/EXT-13-29930 Rev. 2. September 2015.
#      https://bison.inl.gov/SiteAssets/BISON_Theory_version_1_2.pdf
# [2]  L. Leibowitz and R.A. Blomquist. Thermal conductivity and
#      thermal expansion of stainless steels D9 and HT9.
#      International Journal of Thermophysics, 9(5):873–883, 1988.
# [3]  N. Yamanouchi, M. Tamura, H. Hayakawa, and T. Kondo.
#      Accumulation of engineering data for practical use of
#      reduced activation ferritic steel: 8%Cr-2%W-0.2%V-0.04%Ta-Fe.
#      J. Nucl. Mater., 191–194:822–826, 1992
# [4]  G. L. Hofman et al. "Metallic Fuels Handbook". ANL-NSE-3 (1989).
# [5]  Kenneth C. Mills. Recommended Values of Thermophysical
#      Properties for Selected Commercial Alloys. Woodhead
#      Publishing, 2002.
# [6]  C. S. Kim. "Thermophysical Properties of Stainless Steels".
#      ANL-75-55 (1975).
# [7]  A. Banerjee et. al. "High Temperature Heat Capacity of Alloy
#      D9 Using Drop Calorimetry Based Enthalpy Increment Measurements".
#      International Journal of Thermophysics, vol. 28, no. 1, (2007).
#      DOI: 10.1007/s10765-006-0136-0
#      https://link.springer.com/content/pdf/10.1007/s10765-006-0136-0.pdf
# SE2ANL correlations for sodium fixed at 425 degrees (C)

########################################################################
# COOLANTS
########################################################################
# Correlations from SE2ANL "prop.f" file
# Note that viscosity correlation is a polynomial fitted to the
# correlated values based on the non-polynomial SE2ANL equation; the
# two give reasonably close results between 250 - 900C
sodium_se2anl = {'thermal_conductivity': [109.7452, -0.064508, 1.173e-5],
                 # 'density': [950.1, -0.22976, -1.46e-5, 5.638e-9],
                 'density': [1011.654722241015, -0.220522050856835,
                             -1.92200591e-05, 5.638e-09],
                 'viscosity': [8.64078249e-03, -5.00659539e-05,
                               1.14361017e-07, -1.16194739e-10,
                               4.37629969e-14],
                 'heat_capacity': [1630.16, -0.832842, 0.0004625424]}


sodium_se2anl_425 = {'thermal_conductivity': [70.42623125],
                     'density': [850.2476796],
                     'viscosity': [0.000271272],
                     'heat_capacity': [1274.160732],
                     'beta': [0.00028122098689669706]}
#
# sodium_se2anl = {'thermal_conductivity': [109.7452, -0.064508, 1.173e-5],
#                  'density': [1011.654722241015, -0.220522050856835,
#                              -1.92200591e-05, 5.638e-9],
#                  'viscosity': [8.64078249e-03, -5.00659539e-05,
#                                1.14361017e-07, -1.16194739e-10,
#                                4.37629969e-14],
