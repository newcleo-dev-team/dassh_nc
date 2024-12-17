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
date: 2021-11-02
author: matz
Test DASSH material class
"""
########################################################################
import os
import pytest
import numpy as np
from dassh import Material


__PROPERTIES = ['density', 'heat_capacity','viscosity', 'thermal_conductivity']
__COOL_NAMES = ['lead', 'lbe', 'bismuth', 'sodium', 'nak']
__OUT_RANGE = {'lead': [600, 1301, 2021], 'lbe': [397, 1201, 1927], 'bismuth': [544, 1001, 1831]}

def __test_property(mat, t_range, correct_values, property):
    """
    Function to test properties
    """
    for i in np.arange(0,len(t_range)-1):
        mat.update(np.average([t_range[i], t_range[i+1]]))
        assert correct_values[i] == pytest.approx(getattr(mat, property))
        
def __extract_properties(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        p = __PROPERTIES.copy()
        if 'viscosity' not in lines[0]:
            p.remove('viscosity')
        pdict = {k: [] for k in p}
        for line in lines[1:]:  
            values = line.strip().split(',')  
            for i in range(0, len(p)):
                try:
                    pdict[p[i]].append(float(values[i]))
                except:
                    continue
    return pdict

def __get_temperature_range(name):
    lead_range =  range(650, 1300, 50)   # 600.60-1300 K
    lbe_range = range(400, 1200, 50)    
    bismuth_range = range(550, 1000, 50)
    sodium_range = range(400, 2400, 100)
    potassium_range = np.arange(373.15, 1173.15, 50)
    nak_range = np.arange(323.15, 1173.15, 50)
    ss304_range = np.arange(300, 1700, 100)
    ss316_range = np.arange(300, 1700, 100)
    return locals()[name + '_range']

def __generate_coefficients(n):
    c = {'thermal_conductivity': [10.0],
         'heat_capacity': [480.0],
         'density': [1200.0],
         'viscosity': [1.0],
         'beta': [0.002]
         }
    if n > 0:
        for i in range(1,n+1):
            for key in c.keys():
                c[key].append(i*0.001)
    return c

def __calc_expected_value(n, T):
    expected_value = [1200.0, 480.0, 1.0, 10.0, 0.002]
    if n > 0:
        for i in range(1,n+1):
            for p in range(5):
                expected_value[p] = expected_value[p] + 0.001*i*T**i
    return expected_value

def test_builtin_materials():
    """Test built in materials by coefficient dictionaries"""
    assert hasattr(Material('d9'), 'thermal_conductivity')
    assert hasattr(Material('ht9'), 'thermal_conductivity')
    
def test_material_from_coeff():
    # Define a custom dictionary and use it
    for n in range(1, 6):
        c = __generate_coefficients(n)
        print(n, c)
        m = Material('test_material', coeff_dict=c)
        for prop in ['thermal_conductivity', 'heat_capacity',
                    'density', 'viscosity']:
            assert hasattr(m, prop)
            # Try getting a value from the correlation; should be
            # a float, and should be greater than 0
            assert getattr(m, prop) > 0.0
            assert type(getattr(m, prop)) == float
            # Check the results for one of the values
        for T in range(100, 2000, 100):
            m.update(T)
            expected_values = __calc_expected_value(n, T)
            props = [getattr(m,p) for p in __PROPERTIES + ['beta']]
            assert props == pytest.approx(expected_values)


def test_material_from_table():
    """Try loading material properties stored exclusively in CSV
    tables in the data/ directory"""
    coolants = ['water', 'sodium', 'potassium', 'nak',
                'lead', 'bismuth', 'lbe']
    temps = [300.0, 500.0, 500.0, 500.0, 800.0, 800.0, 800.0]
    for i in range(len(coolants)):
        mat = Material(coolants[i], temps[i])
        for prop in ['thermal_conductivity', 'heat_capacity',
                     'density', 'viscosity']:
            assert hasattr(mat, prop)
            # Try getting a value from the correlation; should be
            # a float, and should be greater than 0
            assert getattr(mat, prop) > 0.0


def test_error_table_non_positive_val(testdir, caplog):
    """Test error when table has negative value"""
    f = os.path.join(testdir, 'test_inputs', 'custom_mat-3.csv')
    with pytest.raises(SystemExit):
        Material('badbad', from_file=f)     
    assert 'Non-positive or missing values detected in material data ' in caplog.text
    f0 = os.path.join(testdir, 'test_inputs', 'custom_mat-4.csv')
    with pytest.raises(SystemExit):
        Material('badbad', from_file=f0)     
    assert 'Non-positive or missing values detected in material data ' in caplog.text


def test_material_coeff_from_file(testdir):
    """Try loading material property correlation coeffs from CSV"""
    filepath = os.path.join(testdir, 'test_inputs', 'custom_mat.csv')
    m = Material('sodium', from_file=filepath)
    m.update(623.15)
    for prop in ['thermal_conductivity', 'heat_capacity',
                 'density', 'viscosity']:
        assert hasattr(m, prop)
        assert getattr(m, prop) > 0.0


def test_failed_material(caplog):
    """Make sure that the Material class fails with bad input"""
    with pytest.raises(SystemExit):
        Material('candycorn')
    assert 'material candycorn' in caplog.text


def test_bad_temperature(caplog):
    """Make sure Material throws error for negative temperatures"""
    m = Material('sodium')
    with pytest.raises(SystemExit):
        m.update(0.0)
    assert 'must be > 0; given' in caplog.text


def test_bad_property(caplog):
    """Make sure Material throws error for negative temperatures"""
    # Define a custom dictionary and use it
    c = {'thermal_conductivity': [0.05, 1.0, 2.0],
         'heat_capacity': [480.0, 0.5],
         'density': [1200.0, 0.2, 0.03],
         'viscosity': [300.0, -1.0],
         }
    m = Material('test', coeff_dict=c)
    with pytest.raises(SystemExit):
        m.update(400.0)
    assert 'viscosity must be > 0; given' in caplog.text

def test_table_with_missing_values(testdir, caplog):
    """Test that DASSH interpolates properties
    with missing or zero values"""
    filepath = os.path.join(testdir, 'test_inputs', 'custom_mat-2.csv')
    with pytest.raises(SystemExit):
        Material('missing_sodium', from_file=filepath)    
    assert 'Non-positive or missing values detected in material data ' in caplog.text

def test_interpolation():
    """
    Test interpolation of all properties for all materials
    """
    for mat in __COOL_NAMES + ['ss304', 'ss316', 'potassium']:
        temperature_range = __get_temperature_range(mat)
        properties = __extract_properties('tests/test_data/material_data/' + mat + '_properties.csv')
        m = Material(mat, temperature=temperature_range[0])
        for prop_name, correct_values in properties.items():
            __test_property(m, temperature_range, correct_values, prop_name)
            
def test_correlations_temperature_in_range():
    """
    Test use of lbh15 in calculating material properties for lead, lbe and bismuth
    and sodium/NaK correlations
    """
    for mat in __COOL_NAMES:
        temperature_range = __get_temperature_range(mat)
        properties = __extract_properties('tests/test_data/material_data_correlations/' + mat + '_properties.csv')
        m = Material(mat, temperature=temperature_range[0], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
        for prop_name, correct_values in properties.items():
            __test_property(m, temperature_range, correct_values, prop_name)  

     
def test_lbh15_temperature_outside_range():
    """
    Test use of lbh15 in calculating material properties for lead, lbe and bismuth:
    - below the melting temperature
    - outside the range of the correlation validity
    - above boiling temperature 
    """
    for mat in __OUT_RANGE.keys():    
        with pytest.raises(ValueError, match="Temperature must be larger than melting temperature"):
            m = Material(mat, temperature=__OUT_RANGE[mat][0], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
        with pytest.warns(UserWarning, match="The thermal conductivity is requested at temperature value"):
            m = Material(mat, temperature=__OUT_RANGE[mat][1], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
        with pytest.raises(ValueError, match="Temperature must be smaller than boiling temperature"):
            m = Material(mat, temperature=__OUT_RANGE[mat][2], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
            
def test_non_strict_monotonicity_in_table(testdir, caplog):
    """
    Test that an error is raised in case of non-strictly increasing temperatures in tables
    """
    f1 = os.path.join(testdir, 'test_inputs', 'mat_non_strict_1.csv')
    with pytest.raises(SystemExit):
        Material('badbad', from_file=f1)     
    assert 'Non strictly increasing temperature values detected in material data' in caplog.text
    f2 = os.path.join(testdir, 'test_inputs', 'mat_non_strict_2.csv')
    with pytest.raises(SystemExit):
        Material('badbad', from_file=f2)     
    assert 'Non strictly increasing temperature values detected in material data' in caplog.text