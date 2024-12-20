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
import json

with open("tests/test_data/material_class_data.json", "r") as json_file:
    file_data = json.load(json_file)


properties = file_data["properties"]
cool_names = file_data["cool_names"]
out_range = file_data["out_range"]
correlation_dict = file_data["correlation_dict"]
correlation_dict_wrong = file_data["correlation_dict_wrong"]



def property_test(mat: str, t_range: np.array, correct_values: list[float], property: str):
    """
    Function to test properties
    
    Parameters
    ----------
    mat : str
        Material name
    t_range : np.array
        Temperature range
    correct_values : list[float]
        Correct values for the properties
    property : str
        Property name
    """
    for i in np.arange(0,len(t_range)-1):
        mat.update(np.average([t_range[i], t_range[i+1]]))
        assert correct_values[i] == pytest.approx(getattr(mat, property))
        
def get_temperature_range(name: str):
    """
    Function to get validity temperature range
    
    Parameters
    ----------
    name : str
        Material name
    """
    temperature_range = np.arange(*file_data["temperature_range"][name])
    return temperature_range
        
def extract_properties(path: str):
    """
    Function to extract properties from csv file
    
    Parameters
    ----------
    path : str
        Path to the csv file
    """
    with open(path, 'r') as file:
        lines = file.readlines()
        p = properties.copy()
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
           
class TestCoefficients():
    """
    Class to test material properties as polynomials
    """
    def __generate_coefficients(self, n: int):
        """
        Method to generate coefficients for a polynomial
        
        Parameters
        ----------
        n : int
            Degree of the polynomial
        """
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
    
    def __calc_expected_value(self, n: int, T: float):
        """
        Method to calculate expected values for a polynomial evaluation
        
        Parameters
        ----------
        n : int
            Degree of the polynomial
        T : float
            Temperature
        """
        expected_value = [1200.0, 480.0, 1.0, 10.0, 0.002]
        if n > 0:
            for i in range(1,n+1):
                for p in range(5):
                    expected_value[p] = expected_value[p] + 0.001*i*T**i
        return expected_value
    
    def test_bad_temperature(self, caplog):
        """Make sure Material throws error for negative temperatures"""
        m = Material('sodium')
        with pytest.raises(SystemExit):
            m.update(0.0)
        assert 'must be > 0; given' in caplog.text

    def test_material_from_coeff(self):
        """Test material properties as polynomials"""
        # Define a custom dictionary and use it
        for n in range(1, 6):
            c = self.__generate_coefficients(n)
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
                expected_values = self.__calc_expected_value(n, T)
                props = [getattr(m,p) for p in properties + ['beta']]
                assert props == pytest.approx(expected_values)

    def test_material_coeff_from_file(self, testdir):
        """Try loading material property correlation coeffs from CSV"""
        filepath = os.path.join(testdir, 'test_inputs', 'custom_mat.csv')
        m = Material('sodium', from_file=filepath)
        m.update(623.15)
        for prop in ['thermal_conductivity', 'heat_capacity',
                    'density', 'viscosity']:
            assert hasattr(m, prop)
            assert getattr(m, prop) > 0.0
            
    def test_bad_property(self, caplog):
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
            
class TestBuiltInCorrelations():   
    """
    Class to test material properties using built-in correlations
    """
    def test_correlations_temperature_in_range(self):
        """
        Test use of lbh15 in calculating material properties for lead, lbe and bismuth
        and sodium/NaK correlations
        """
        for mat in cool_names:
            temperature_range = get_temperature_range(mat)
            properties = extract_properties('tests/test_data/material_data_correlations/' + mat + '_properties.csv')
            m = Material(mat, temperature=temperature_range[0], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
            for prop_name, correct_values in properties.items():
                property_test(m, temperature_range, correct_values, prop_name)  
                
    def test_lbh15_temperature_outside_range(self):
        """
        Test use of lbh15 in calculating material properties for lead, lbe and bismuth:
        - below the melting temperature
        - outside the range of the correlation validity
        - above boiling temperature 
        """
        for mat in out_range.keys():    
            with pytest.raises(ValueError, match="Temperature must be larger than melting temperature"):
                m = Material(mat, temperature= out_range[mat][0], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
            with pytest.warns(UserWarning, match="The thermal conductivity is requested at temperature value"):
                m = Material(mat, temperature= out_range[mat][1], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
            with pytest.raises(ValueError, match="Temperature must be smaller than boiling temperature"):
                m = Material(mat, temperature= out_range[mat][2], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
            
class TestTablesAndIntepolation():
    """
    Class to test material properties using tables and interpolation
    """
    def test_material_from_table(self):
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

    def test_error_table_non_positive_val(self, testdir, caplog):
        """Test error when table has negative value"""
        f = os.path.join(testdir, 'test_inputs', 'custom_mat-3.csv')
        with pytest.raises(SystemExit):
            Material('badbad', from_file=f)     
        assert 'Non-positive or missing values detected in material data ' in caplog.text
        f0 = os.path.join(testdir, 'test_inputs', 'custom_mat-4.csv')
        with pytest.raises(SystemExit):
            Material('badbad', from_file=f0)     
        assert 'Non-positive or missing values detected in material data ' in caplog.text
        
    def test_table_with_missing_values(self, testdir, caplog):
        """
        Test that DASSH interpolates properties
        with missing or zero values
        """
        filepath = os.path.join(testdir, 'test_inputs', 'custom_mat-2.csv')
        with pytest.raises(SystemExit):
            Material('missing_sodium', from_file=filepath)    
        assert 'Non-positive or missing values detected in material data ' in caplog.text
        
    def test_non_strict_monotonicity_in_table(self, testdir, caplog):
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
        
    def test_interpolation(self):
        """
        Test interpolation of all properties for all materials
        """
        for mat in cool_names + ['ss304', 'ss316', 'potassium']:
            temperature_range = get_temperature_range(mat)
            properties = extract_properties('tests/test_data/material_data/' + mat + '_properties.csv')
            m = Material(mat, temperature=temperature_range[0])
            for prop_name, correct_values in properties.items():
                property_test(m, temperature_range, correct_values, prop_name)    
        
class TestUserCorrelation():
    """
    Class to test user-defined correlations
    """
    def test_user_correlation_from_input(self):
        """
        Test that the user correlation is correctly parsed and used
        """
        c = correlation_dict
        m = Material('user_material', corr_dict = c)
        m.update(100)
        assert m.density == pytest.approx(229.2)
        assert m.thermal_conductivity == pytest.approx(113.779)
        assert m.heat_capacity == pytest.approx(1.2784)
        assert m.viscosity == pytest.approx(0.067543978521069)
        
    def test_user_correlation_from_file(self, testdir):
        """
        Test that the user correlation from file is correctly parsed and used
        """
        f = os.path.join(testdir, 'test_inputs', 'custom_mat-5.txt')
        m = Material('badbad', from_file=f)
        m.update(100)
        assert m.density == pytest.approx(229.2)
        assert m.thermal_conductivity == pytest.approx(113.779)
        assert m.heat_capacity == pytest.approx(1.2784)
        assert m.viscosity == pytest.approx(0.067543978521069)
        
    def test_wrong_user_correlation(self, caplog):
        """
        Test that an error is raised for wrong user correlation
        """
        c = correlation_dict_wrong
        with pytest.raises(SystemExit):
            Material('user_material', corr_dict = c)
        assert 'Correlation for user_material density contains invalid symbols' in caplog.text
        
    def test_wrong_user_correlation_from_file(self, testdir, caplog):
        """
        Test that an error is raised for wrong user correlation from file
        """
        f = os.path.join(testdir, 'test_inputs', 'custom_mat-6.txt')
        with pytest.raises(SystemExit):
            Material('wrong_corr', from_file=f)
        assert 'Correlation for wrong_corr thermal_conductivity contains invalid symbols' in caplog.text
        
class TestBuiltInDefinition():
    """
    Class to test built-in materials and failed material
    """
    def test_builtin_materials(self):
        """Test built in materials by coefficient dictionaries"""
        assert hasattr(Material('d9'), 'thermal_conductivity')
        assert hasattr(Material('ht9'), 'thermal_conductivity')
        
    def test_failed_material(self, caplog):
        """Make sure that the Material class fails with bad input"""
        with pytest.raises(SystemExit):
            Material('candycorn')
        assert 'material candycorn' in caplog.text