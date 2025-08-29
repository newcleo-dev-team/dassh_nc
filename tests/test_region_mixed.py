"""
date: 2025-08-xx
author: fpepe-polito
Methods for mixed convection axial regions; to be used within Assembly objects
"""
########################################################################
import numpy as np
import pytest
import dassh
from pytest import rr_data

class TestConversions2Temperature():
    """
    Class to test the enthalpy to temperature  and the density to temperature
    conversions in the MixedRegion class
    """
    
    def test_calc_delta_h(self, simple_ctrl_rr_mixconv):
        """Test the _calc_delta_h method of RoddedRegion"""
        for mat in rr_data.enthalpy['delta_h'].keys():
            simple_ctrl_rr_mixconv.coolant.name = mat
            assert simple_ctrl_rr_mixconv._calc_delta_h(rr_data.enthalpy['T1'], rr_data.enthalpy['T2']) == \
                pytest.approx(rr_data.enthalpy['delta_h'][mat], abs=rr_data.enthalpy['tol'])

    def test_temp_from_enthalpy(self, simple_ctrl_rr_mixconv):
        """Test the _temp_from_enthalpy method of RoddedRegion"""
        simple_ctrl_rr_mixconv.temp['coolant_int'] = \
            np.array(rr_data.enthalpy['T1'] * np.ones(simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total']))

        for mat in rr_data.enthalpy['delta_h'].keys():
            simple_ctrl_rr_mixconv.coolant.name = mat
            simple_ctrl_rr_mixconv._delta_h = rr_data.enthalpy['delta_h'][mat] * \
            np.ones(simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total'])
            assert simple_ctrl_rr_mixconv._temp_from_enthalpy() == \
                pytest.approx(rr_data.enthalpy['T2'] * 
                              np.ones(simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total']),
                              abs=rr_data.enthalpy['tol'])

    def test_temp_from_enthalpy_zero_deltah(self, simple_ctrl_rr_mixconv):
        """Test that the _temp_from_enthalpy method of RoddedRegion with zero deltaH
        returns zero deltaT"""
        simple_ctrl_rr_mixconv._delta_h = np.zeros(simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total'])
        assert simple_ctrl_rr_mixconv._temp_from_enthalpy() == \
            pytest.approx(simple_ctrl_rr_mixconv.temp['coolant_int'],
                          abs=rr_data.enthalpy['tol'])
            
    def test_temp_from_density(self, simple_ctrl_rr_mixconv):
        """Test the _temp_from_density method of RoddedRegion"""
        for mat in rr_data.enthalpy['density_values'].keys():
            simple_ctrl_rr_mixconv.coolant.name = mat
            print(mat)
            assert simple_ctrl_rr_mixconv._T_from_rho(rr_data.enthalpy['density_values'][mat]) == \
                pytest.approx(rr_data.enthalpy['T1'], abs=rr_data.enthalpy['tol'])
