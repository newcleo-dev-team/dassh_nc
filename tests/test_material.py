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
import copy
from typing import Dict, List, Union
from pytest import mat_data


def property_test(mat: str, t_values: np.array, correct_values: list[float], property: str):
    """
    Function that tests that values of a property of a certain material
    in a certain temperature range are correctly evaluated
    
    Parameters
    ----------
    mat : str
        Material name
    t_values : np.array
        Temperature range
    correct_values : list[float]
        Reference value for the property
    property : str
        Property name
    """
    for ind in range(len(t_values)-1):
        mat.update(np.average([t_values[ind], t_values[ind+1]]))
        assert correct_values[ind] == pytest.approx(getattr(mat, property))
        
def material_comparison(mat: str, use_corr: bool = False, 
                        lbh15_corr: Union[Dict[str, str], Dict[str, None]] = {'cp': None, 'k': None, 'rho': None, 'mu': None}):
    """ 
    Function to compare all material properties over validity range
    
    Parameters
    ----------
    mat : str
        Material name
    use_corr : bool, optional
        Whether to use correlations or not
        Default is False
    lbh15_corr : Dict[str, str], optional
        Dictionary with non-default correlations to use for each property 
    """
    strictest_temp_range = get_strictest_temp_range(mat)
    if use_corr:      
        if lbh15_corr['cp'] is not None:
            properties = mat_data.lead_corr_gurv
        else: 
            properties = getattr(mat_data, f"{mat}_corr")
    else:
        properties = getattr(mat_data, f"{mat}_interp")
    mat = Material(mat, temperature=strictest_temp_range[0], use_correlation = use_corr, 
                     lbh15_correlations = lbh15_corr)
    for prop_name, correct_values in properties.items():
        property_test(mat, strictest_temp_range, correct_values, prop_name)
                
def get_strictest_temp_range(name: str) -> np.ndarray:
    """
    Function to get the strictest validity temperature range
    among all properties
    
    Parameters
    ----------
    name : str
        Material name
        
    Returns
    -------
    strictest_temp_range : np.ndarray
        Array of temperatures in the strictest validity range 
    """
    return np.arange(*mat_data.strictest_temp_range[name])
           
class TestCoefficients():
    """
    Class to test material properties as polynomials
    """
    def __generate_coefficients(self, deg: int) -> Dict[str, List[float]]:
        """
        Method to generate coefficients for a polynomial
        
        Parameters
        ----------
        deg : int
            Degree of the polynomial
            
        Returns
        -------
        cc : Dict[str, List[float]]
            Dictionary with a list of coefficients for each property
        """
        cc = copy.deepcopy(mat_data.coeff)
        for ind in range(1,deg+1):
            for key in cc.keys():
                cc[key].append(ind*mat_data.mfact)
        return cc
    
    def __calc_expected_values(self, deg: int, T: float) -> Dict[str, float]:
        """
        Method to calculate expected values for a polynomial evaluation
        
        Parameters
        ----------
        deg : int
            Degree of the polynomial
        T : float
            Temperature
            
        Returns
        -------
        expected_values : Dict[str, float]
            Dictionary with expected values for each property
        """
        expected_values = {key: value[0] for key, value in mat_data.coeff.items()}
        for ind in range(1,deg+1):   
            for p in mat_data.properties_list_full:        
                expected_values[p] = expected_values[p] + mat_data.mfact*ind*T**ind    
        return expected_values
    
    def test_material_from_coeff(self):
        """Test material properties as polynomials"""
        # Define a custom dictionary and use it
        for n in range(6):
            cc = self.__generate_coefficients(n)
            mat = Material('test_material', corr_dict=cc)
            for prop in mat_data.properties_list_full:
                assert hasattr(mat, prop)
                assert type(getattr(mat, prop)) == float
            # Check the results for some of the values
            for T in range(*mat_data.coeff_test_values):
                mat.update(T)
                expected_values = self.__calc_expected_values(n, T)
                props = {prop_name: getattr(mat, prop_name) for prop_name in mat_data.properties_list_full}
                assert props == pytest.approx(expected_values)

    def test_material_coeff_from_file(self, testdir):
        """Try loading material property correlation coeffs from CSV"""
        filepath = os.path.join(testdir, 'test_inputs', 'custom_mat.csv')
        mat = Material('aasodium', mat_data.temperature_coeff_file, from_file=filepath)
        for prop in mat_data.properties_list:
            assert getattr(mat, prop) == pytest.approx(mat_data.expected_from_coeff[prop])
            
    def test_bad_property(self, caplog):
        """Make sure Material throws error for negative value of a property"""
        # Define a custom dictionary and use it
        cc = copy.deepcopy(mat_data.bad_coeff)
        with pytest.raises(SystemExit):
            mat = Material('test', mat_data.temperature_sodium_definition, corr_dict=cc)
        assert 'viscosity must be > 0; given' in caplog.text
            
class TestBuiltInCorrelations():   
    """
    Class to test material properties using built-in correlations
    """
    def test_correlations_results(self):
        """
        Test use of lbh15 in calculating material properties for lead, lbe and bismuth
        and sodium/NaK correlations
        """
        for mat in mat_data.correlation_mat_names:
            material_comparison(mat, use_corr = True) 
           
    def test_lbh15_non_default_correlation(self, caplog):
        """
        Test use of lbh15 in calculating material properties
        using non-default correlations
        """
        material_comparison('lead', use_corr = True, lbh15_corr = {'cp': mat_data.corr_names[0], 'k': None, 'rho': None, 'mu': None})
            
        with pytest.raises(SystemExit):
            mat_bad_corr = Material('lead', get_strictest_temp_range('lead')[0], use_correlation = True, 
                     lbh15_correlations = {'cp': mat_data.corr_names[1], 'k': None, 'rho': None, 'mu': None})
        assert f'Correlation {mat_data.corr_names[1]} for cp not available for lead' in caplog.text
             
    def test_lbh15_temperature_outside_range(self, caplog):
        """
        Test use of lbh15 in calculating material properties for lead, lbe and bismuth:
        - below the melting temperature
        - outside the range of the correlation validity
        - above boiling temperature 
        """
        for mat in mat_data.out_range.keys():    
            with pytest.raises(SystemExit):
                Material(mat, temperature= mat_data.out_range[mat][0], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
            assert "Temperature must be larger than melting temperature" in caplog.text
            Material(mat, temperature = mat_data.out_range[mat][1], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
            assert "The thermal conductivity is requested at temperature value" in caplog.text
            with pytest.raises(SystemExit):
                Material(mat, temperature= mat_data.out_range[mat][2], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
            assert "Temperature must be smaller than boiling temperature" in caplog.text
        Material('lbe', temperature = mat_data.out_range['lbe'][3], use_correlation = True, lbh15_correlations = {'cp': None, 'k': None, 'rho': None, 'mu': None})
        assert "is requested at temperature value" in caplog.text
        
    def test_sodium_nak_corr_out_range(self, caplog):
        """
        Test that an error is raised for temperature outside the range of correlations 
        for sodium and NaK
        """
        for mat in mat_data.corr_out_range.keys():
            with pytest.raises(SystemExit):
                Material(mat, temperature= mat_data.corr_out_range[mat][0])
            assert 'is below the minimum allowed value of the validity range ' in caplog.text
            with pytest.raises(SystemExit):
                Material(mat, temperature= mat_data.corr_out_range[mat][1])
            assert 'is above the maximum allowed value of the validity range ' in caplog.text
    
    def test_sodium_nak_corr_mid_range(self, caplog):
        """
        Test that a warning is raised for temperature above the validity range
        for a property, but below the maximum validity range
        """
        for mat in mat_data.corr_mid_range.keys():    
            Material(mat, temperature=mat_data.corr_mid_range[mat], use_correlation=True)
            assert 'is above the validity range' in caplog.text
                
class TestTablesAndIntepolation():
    """
    Class to test material properties using tables and interpolation
    """
    def test_error_table_non_positive_val(self, testdir, caplog):
        """Test error when table has negative value"""
        f = os.path.join(testdir, 'test_inputs', 'custom_mat-3.csv')
        with pytest.raises(SystemExit):
            Material('negative_prop_mat', from_file=f)     
        assert 'Non-positive values detected in material data ' in caplog.text
        f1 = os.path.join(testdir, 'test_inputs', 'custom_mat-2.csv')
        with pytest.raises(SystemExit):
            Material('zero_prop_mat', from_file=f1)    
        assert 'Non-positive values detected in material data ' in caplog.text
        
    def test_non_strict_monotonicity_in_table(self, testdir, caplog):
        """
        Test that an error is raised in case of non-strictly increasing temperatures in tables
        """
        f1 = os.path.join(testdir, 'test_inputs', 'mat_non_strict_1.csv')
        with pytest.raises(SystemExit):
            Material('not_monotonic_mat_1', from_file=f1)     
        assert 'Non strictly increasing temperature values detected in material data' in caplog.text
        f2 = os.path.join(testdir, 'test_inputs', 'mat_non_strict_2.csv')
        with pytest.raises(SystemExit):
            Material('not_monotonic_mat_2', from_file=f2)     
        assert 'Non strictly increasing temperature values detected in material data' in caplog.text
        f3 = os.path.join(testdir, 'test_inputs', 'mat_non_strict_3.csv')
        with pytest.raises(SystemExit):
            Material('not_monotonic_mat_3', from_file=f3)     
        assert 'Non strictly increasing temperature values detected in material data' in caplog.text
        
    def test_default_material_interpolation(self):
        """
        Test interpolation of all properties for all materials
        """
        for mat in mat_data.material_names:
            material_comparison(mat) 
            
    def test_interpolation_with_missing_value(self, testdir):
        """
        Tests that interpolation is correctly performed
        in case of missing value in a user-defined table
        """
        f = os.path.join(testdir, 'test_inputs', 'custom_missing.csv')
        mat = Material('test_mat', from_file=f)  
        mat.update(mat_data.interp_temperature)
        assert mat.density == pytest.approx(mat_data.interp_expected_values[0])   
        
        f = os.path.join(testdir, 'test_inputs', 'custom_missing-2.csv')
        mat = Material('test_mat', from_file=f)
        mat.update(mat_data.interp_temperature)
        assert mat.density == pytest.approx(mat_data.interp_expected_values[1])     
    
    def test_table_out_of_range(self, caplog):
        """
        Test that a warning is raised for temperature outside the range of the table
        """
        for mat in mat_data.table_out_range.keys():  
            with pytest.raises(SystemExit):  
                Material(mat, temperature= mat_data.table_out_range[mat][0])
            assert 'is below the minimum allowed value of the validity range ' in caplog.text
            with pytest.raises(SystemExit):
                Material(mat, temperature= mat_data.table_out_range[mat][1])
            assert 'is above the maximum allowed value of the validity range ' in caplog.text
            
    def test_table_mid_range(self, caplog):
        """
        Test that a warning is raised for temperature above the validity range
        for a property, but below the maximum validity range
        """
        for mat in mat_data.table_mid_range.keys():    
            Material(mat, temperature = mat_data.table_mid_range[mat][0])
            assert 'is above the validity range' in caplog.text
        Material('sodium', temperature = mat_data.table_mid_range['sodium'][1])
        assert 'is below the validity range' in caplog.text    
    
    def test_non_existing_property(self, testdir, caplog):
        """Make sure Material throws error for non-existing property in user-defined table"""
        f = os.path.join(testdir, 'test_inputs', 'non_exist_prop.csv')
        with pytest.raises(SystemExit):
            Material('additional_prop_mat', from_file=f)
        assert 'Property property5 not recognized' in caplog.text
        
               
class TestUserCorrelation():
    """
    Class to test user-defined correlations
    """
    def __user_correlation_comparison(self, mat: Material, TT: np.ndarray):
        """
        Function to compare properties calculated using the user correlation to expected values
        
        Parameters
        ----------
        mat : Material
            Material object
        TT : np.ndarray
            Range of temperatures over which the comparison is performed
        """
        for ind in range(len(TT)):
            mat.update(TT[ind])
            assert mat.density == pytest.approx(mat_data.expected_from_corr['density'][ind])
            assert mat.thermal_conductivity == pytest.approx(mat_data.expected_from_corr['thermal_conductivity'][ind])
            assert mat.heat_capacity == pytest.approx(mat_data.expected_from_corr['heat_capacity'][ind])
            assert mat.viscosity == pytest.approx(mat_data.expected_from_corr['viscosity'][ind])
            
    def test_user_correlation_from_input(self):
        """
        Test that the user correlation is correctly used
        """
        self.__user_correlation_comparison(
            Material('user_material', corr_dict = mat_data.correlation_dict),
            np.arange(*mat_data.user_corr_values))
        
    def test_user_correlation_from_file(self, testdir):
        """
        Test that the user correlation from file is correctly parsed and used
        """
        f = os.path.join(testdir, 'test_inputs', 'custom_mat-4.txt')
        self.__user_correlation_comparison(Material('from_file_material', from_file=f),
                                           np.arange(*mat_data.user_corr_values))
            
    def test_wrong_user_correlation(self, caplog):
        """
        Test that an error is raised for wrong user correlation
        """
        with pytest.raises(SystemExit):
            Material('user_material', corr_dict = mat_data.correlation_dict_wrong)
        assert 'Correlation for user_material density contains invalid symbols' in caplog.text
        
    def test_wrong_user_correlation_from_file(self, testdir, caplog):
        """
        Test that an error is raised for wrong user correlation from file
        """
        f = os.path.join(testdir, 'test_inputs', 'custom_mat-5.txt')
        with pytest.raises(SystemExit):
            Material('wrong_corr', from_file=f)
        assert 'Correlation for wrong_corr thermal_conductivity contains invalid symbols' in caplog.text
    
    def test_user_correlation_and_coefficients(self):
        """
        Test that the user correlation is used when both user correlation 
        and coefficients are provided for different properties
        """
        self.__user_correlation_comparison(Material('user_material', corr_dict = mat_data.correlation_dict_2),
                                           np.arange(*mat_data.user_corr_values))
        
class TestBuiltInDefinition():
    """
    Class to test built-in materials and failed material
    """
    def test_builtin_materials(self):
        """Test built in materials by coefficient dictionaries"""
        for mat in mat_data.built_in_coeff_mat:
            assert hasattr(Material(mat), 'thermal_conductivity')
        
    def test_failed_material(self, caplog):
        """
        Make sure that the Material class fails with a material name
        that is not in the built-in list and is not user-defined
        """ 
        with pytest.raises(SystemExit):
            Material('candycorn')
        assert 'material candycorn' in caplog.text
        
    def test_bad_temperature(self, caplog):
        """Make sure Material throws error for 0 or negative temperatures"""
        mat = Material('sodium', mat_data.temperature_sodium_definition)
        with pytest.raises(SystemExit):
            mat.update(0.0)
        assert 'must be > 0; given' in caplog.text
        with pytest.raises(SystemExit):
            mat.update(mat_data.negative_temperature)
            
class TestEnthalpyTemperatureConversion():
    """
    Class to test enthalpy-temperature conversion methods
    """       

    mat_dict: dict[str, Material] = {
        mat: Material(mat,temperature=mat_data.enthalpy['T1'], 
                      solve_enthalpy=True) \
                          for mat in mat_data.enthalpy['coeffs_h2T'].keys()}
    """Dictionary of Material objects with solve_enthalpy set to True"""
    
    def test_read_enthalpy_coefficients(self):
        """
        Test the _read_enthalpy_coefficients method of the Material class
        """
        for mat in mat_data.enthalpy['coeffs_h2T'].keys():
            mm = self.mat_dict[mat]
            assert mm.coeffs_h2T == \
                pytest.approx(mat_data.enthalpy['coeffs_h2T'][mat])

    def test_temp_from_enthalpy(self):
        """
        Test the temp_from_enthalpy method of the Material class
        """
        for mat in mat_data.enthalpy['h2'].keys():
            mm = self.mat_dict[mat]
            assert mm.temp_from_enthalpy(mat_data.enthalpy['h2'][mat]) == \
                pytest.approx(mat_data.enthalpy['T2'],
                              abs=mat_data.enthalpy['tol'])
                
    def test_enthalpy_from_temp(self):
        """
        Test the enthalpy_from_temp method of the Material class
        """
        for mat in mat_data.enthalpy['h2'].keys():
            mm = self.mat_dict[mat]
            assert mm.enthalpy_from_temp(mat_data.enthalpy['T2']) == \
                pytest.approx(mat_data.enthalpy['h2'][mat])
