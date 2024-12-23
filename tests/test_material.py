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
import copy
from typing import List, Dict, Any
from dataclasses import dataclass
import lbh15

with open("tests/test_data/material_class_data.json", "r") as json_file:
    file_data = json.load(json_file)


@dataclass
class MaterialData:
    correlation_mat_names: List[str]
    material_names: List[str]
    temperature_range: Dict[str, np.array]
    correlation_dict: Dict[str, Any]
    correlation_dict_2: Dict[str, Any]
    correlation_dict_wrong: Dict[str, Any]
    expected_from_corr: Dict[str, List[float]]
    coeff: Dict[str, List[float]]
    bad_coeff: Dict[str, List[float]]
    lead_interp: Dict[str, List[float]]
    lbe_interp: Dict[str, List[float]]
    bismuth_interp: Dict[str, List[float]]
    sodium_interp: Dict[str, List[float]]
    nak_interp: Dict[str, List[float]]
    potassium_interp: Dict[str, List[float]]
    ss304_interp: Dict[str, List[float]]
    ss316_interp: Dict[str, List[float]]
    lead_corr: Dict[str, List[float]]
    lbe_corr: Dict[str, List[float]]
    bismuth_corr: Dict[str, List[float]]
    sodium_corr: Dict[str, List[float]]
    nak_corr: Dict[str, List[float]]
    properties_list = ['density', 'thermal_conductivity', 'heat_capacity', 'viscosity']
    mat_from_tables: Dict[str, float]
    out_range: Dict[str, List[float]]
    
@pytest.fixture
def material_data():
    """
    Setup test data for the material class
    """
    with open("tests/test_data/material_class_data.json", "r") as json_file:
        file_data = json.load(json_file)
        
    lead_bounds = lbh15.lead_properties.k().range
    lbe_bounds = lbh15.lbe_properties.k().range
    bismuth_bounds = lbh15.bismuth_properties.k().range
        
    out_range = {
        'lead': [lead_bounds[0]-1, lead_bounds[1]+1] + [2021.0],
        'lbe': [lbe_bounds[0]-1, lbe_bounds[1]+1] + [1927.0],
        'bismuth': [bismuth_bounds[0]-1, bismuth_bounds[1]+1] + [1831.0]
    }    
    
    data = MaterialData(
        correlation_mat_names=file_data["cool_names"],
        material_names=file_data["cool_names"] + ['ss304', 'ss316', 'potassium'],
        temperature_range=file_data["temperature_range"],
        correlation_dict=file_data["correlation_dict"],
        correlation_dict_2=file_data["correlation_dict_2"],
        correlation_dict_wrong=file_data["correlation_dict_wrong"],
        expected_from_corr=file_data["expected_from_corr"],
        coeff=file_data["coefficients"],
        bad_coeff=file_data["bad_coefficients"],
        lead_interp=file_data["lead_interp"],
        lbe_interp=file_data["lbe_interp"],
        bismuth_interp=file_data["bismuth_interp"],
        sodium_interp=file_data["sodium_interp"],
        nak_interp=file_data["nak_interp"],
        potassium_interp=file_data["potassium_interp"],
        ss304_interp=file_data["ss304_interp"],
        ss316_interp=file_data["ss316_interp"],
        lead_corr=file_data["lead_corr"],
        lbe_corr=file_data["lbe_corr"],
        bismuth_corr=file_data["bismuth_corr"],
        sodium_corr=file_data["sodium_corr"],
        nak_corr=file_data["nak_corr"],
        mat_from_tables=file_data["mat_from_tables"],
        out_range = out_range
    )
    
    yield data

def property_test(mat: str, t_range: np.array, correct_values: list[float], property: str):
    """
    Function that tests that values of a property of a certain material
    in a certain temperature range are correctly evaluated
    
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
        
def get_temperature_range(name: str, material_data: MaterialData):
    """
    Function to get validity temperature range
    
    Parameters
    ----------
    name : str
        Material name
    """
    temperature_range = np.arange(*material_data.temperature_range[name])
    return temperature_range  
           
class TestCoefficients():
    """
    Class to test material properties as polynomials
    """
    def __generate_coefficients(self, n: int, material_data: MaterialData):
        """
        Method to generate coefficients for a polynomial
        
        Parameters
        ----------
        n : int
            Degree of the polynomial
        """
        cc = copy.deepcopy(material_data.coeff)
        if n > 0:
            for i in range(1,n+1):
                for key in cc.keys():
                    cc[key].append(i*0.001)
        return cc
    
    def __calc_expected_value(self, n: int, T: float, material_data: MaterialData):
        """
        Method to calculate expected values for a polynomial evaluation
        
        Parameters
        ----------
        n : int
            Degree of the polynomial
        T : float
            Temperature
        """
        expected_value = {key: value[0] for key, value in material_data.coeff.items()}
        for i in range(0,n+1):   
            for p in material_data.properties_list + ['beta']:        
                expected_value[p] = expected_value[p] + 0.001*i*T**i     
        return expected_value
    
    def test_material_from_coeff(self, material_data: MaterialData):
        """Test material properties as polynomials"""
        # Define a custom dictionary and use it
        for n in range(1, 6):
            cc = self.__generate_coefficients(n, material_data)
            m = Material('test_material', corr_dict=cc)
            for prop in material_data.properties_list:
                assert hasattr(m, prop)
                # Try getting a value from the correlation; should be
                # a float, and should be greater than 0
                assert getattr(m, prop) > 0.0
                assert type(getattr(m, prop)) == float
            # Check the results for some of the values
            for T in range(500, 1000, 100):
                m.update(T)
                expected_values = self.__calc_expected_value(n, T, material_data)
                props = {p: getattr(m,p) for p in material_data.properties_list + ['beta']}
                assert props == pytest.approx(expected_values)

    def test_material_coeff_from_file(self, testdir, material_data: MaterialData):
        """Try loading material property correlation coeffs from CSV"""
        filepath = os.path.join(testdir, 'test_inputs', 'custom_mat.csv')
        m = Material('sodium', from_file=filepath)
        m.update(623.15)
        for prop in material_data.properties_list:
            assert hasattr(m, prop)
            assert getattr(m, prop) > 0.0
            
    def test_bad_property(self, caplog, material_data: MaterialData):
        """Make sure Material throws error for negative temperatures"""
        # Define a custom dictionary and use it
        cc = copy.deepcopy(material_data.bad_coeff)
        m = Material('test', corr_dict=cc)
        with pytest.raises(SystemExit):
            m.update(400.0)
        assert 'viscosity must be > 0; given' in caplog.text
            
class TestBuiltInCorrelations():   
    """
    Class to test material properties using built-in correlations
    """
    def test_correlations_temperature_in_range(self, material_data: MaterialData):
        """
        Test use of lbh15 in calculating material properties for lead, lbe and bismuth
        and sodium/NaK correlations
        """
        for mat in material_data.correlation_mat_names:
            temperature_range = get_temperature_range(mat, material_data)
            corr_name = f"{mat}_corr"
            properties = getattr(material_data, corr_name)
            m = Material(mat, temperature=temperature_range[0], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
            for prop_name, correct_values in properties.items():
                property_test(m, temperature_range, correct_values, prop_name)  
                
    def test_lbh15_temperature_outside_range(self, material_data: MaterialData):
        """
        Test use of lbh15 in calculating material properties for lead, lbe and bismuth:
        - below the melting temperature
        - outside the range of the correlation validity
        - above boiling temperature 
        """
        for mat in material_data.out_range.keys():    
            with pytest.raises(ValueError, match="Temperature must be larger than melting temperature"):
                m = Material(mat, temperature= material_data.out_range[mat][0], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
            with pytest.warns(UserWarning, match="The thermal conductivity is requested at temperature value"):
                m = Material(mat, temperature= material_data.out_range[mat][1], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
            with pytest.raises(ValueError, match="Temperature must be smaller than boiling temperature"):
                m = Material(mat, temperature= material_data.out_range[mat][2], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
            
class TestTablesAndIntepolation():
    """
    Class to test material properties using tables and interpolation
    """
    def test_material_from_table(self, material_data: MaterialData):
        """Try loading material properties stored exclusively in CSV
        tables in the data/ directory"""
        coolants = [k for k in material_data.mat_from_tables.keys()]
        temps = [v for v in material_data.mat_from_tables.values()]
        for i in range(len(coolants)):
            mat = Material(coolants[i], temps[i])
            for prop in material_data.properties_list:
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
        
    def test_interpolation(self, material_data: MaterialData):
        """
        Test interpolation of all properties for all materials
        """
        for mat in material_data.material_names:
            temperature_range = get_temperature_range(mat, material_data)
            interp_name = f"{mat}_interp"
            properties = getattr(material_data, interp_name)
            m = Material(mat, temperature=temperature_range[0])
            for prop_name, correct_values in properties.items():
                property_test(m, temperature_range, correct_values, prop_name)    
        
class TestUserCorrelation():
    """
    Class to test user-defined correlations
    """
    def test_user_correlation_from_input(self, material_data: MaterialData):
        """
        Test that the user correlation is correctly used
        """
        c = material_data.correlation_dict
        m = Material('user_material', corr_dict = c)
        TT = np.arange(100, 1100, 200)
        for i in range(len(TT)):
            m.update(TT[i])
            assert m.density == pytest.approx(material_data.expected_from_corr['density'][i])
            assert m.thermal_conductivity == pytest.approx(material_data.expected_from_corr['thermal_conductivity'][i])
            assert m.heat_capacity == pytest.approx(material_data.expected_from_corr['heat_capacity'][i])
            assert m.viscosity == pytest.approx(material_data.expected_from_corr['viscosity'][i])
        
    def test_user_correlation_from_file(self, testdir, material_data: MaterialData):
        """
        Test that the user correlation from file is correctly parsed and used
        """
        f = os.path.join(testdir, 'test_inputs', 'custom_mat-5.txt')
        m = Material('badbad', from_file=f)
        TT = np.arange(100, 1100, 200)
        for i in range(len(TT)):
            m.update(TT[i])
            assert m.density == pytest.approx(material_data.expected_from_corr['density'][i])
            assert m.thermal_conductivity == pytest.approx(material_data.expected_from_corr['thermal_conductivity'][i])
            assert m.heat_capacity == pytest.approx(material_data.expected_from_corr['heat_capacity'][i])
            assert m.viscosity == pytest.approx(material_data.expected_from_corr['viscosity'][i])
        
    def test_wrong_user_correlation(self, caplog, material_data: MaterialData):
        """
        Test that an error is raised for wrong user correlation
        """
        c = material_data.correlation_dict_wrong
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
    
    def test_user_correlation_and_coefficients(self, material_data: MaterialData):
        """
        Test that the user correlation is used when both user correlation and coefficients are provided
        """
        c = material_data.correlation_dict_2
        m = Material('user_material', corr_dict = c)
        
        TT = np.arange(100, 1100, 200)
        for i in range(len(TT)):
            m.update(TT[i])
            assert m.density == pytest.approx(material_data.expected_from_corr['density'][i])
            assert m.thermal_conductivity == pytest.approx(material_data.expected_from_corr['thermal_conductivity'][i])
            assert m.heat_capacity == pytest.approx(material_data.expected_from_corr['heat_capacity'][i])
            assert m.viscosity == pytest.approx(material_data.expected_from_corr['viscosity'][i])
        
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
        
    def test_bad_temperature(self, caplog):
        """Make sure Material throws error for negative temperatures"""
        m = Material('sodium')
        with pytest.raises(SystemExit):
            m.update(0.0)
        assert 'must be > 0; given' in caplog.text