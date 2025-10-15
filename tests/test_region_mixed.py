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
from tests.test_region_rodded import mock_AssemblyPower

class TestEnergyBalance():
    """
    Class to test that energy balance is satisfied in the MixedRegion class
    """
    def _assign_parameters(self, rm: dassh.MixedRegion, temps: np.ndarray):
        """
        Method to initialize parameters and properties of the mixed region

        Parameters
        ----------
        rm : dassh.MixedRegion
            The mixed region object to initialize.
        temps : np.ndarray
            The temperature array to use for initialization.
        """
        rm.temp['coolant_int'] = temps
        vels = np.random.uniform(rr_data.enthalpy['vel'][0], 
                                rr_data.enthalpy['vel'][1], 
                                rm.subchannel.n_sc['coolant']['total'])
        rm._update_subchannels_properties(temps)
        ave_t = np.sum(rm.temp['coolant_int'] * 
                rm.params['area'][rm.subchannel.type[:rm.subchannel.n_sc['coolant']['total']]] *
                vels * rm.sc_properties['density']) / np.sum(
                rm.params['area'][rm.subchannel.type[:rm.subchannel.n_sc['coolant']['total']]] * 
                vels * rm.sc_properties['density'])
        rm._init_static_correlated_params(ave_t)
        rm._sc_vel = vels
        rm.coolant_int_params['ff'] = rr_data.enthalpy['ff']
        
    def test_EEX_MEX_balance(self, simple_ctrl_rr_mixconv: dassh.MixedRegion):
        """
        Test that the sum of the EEX (and MEX) contributions is zero
        
        Parameters
        ----------
        simple_ctrl_rr_mixconv : dassh.MixedRegion
            The mixed region object to test.
            
        Notes:
        ------
        The EEX term represents the exchange of energy between the subchannels
        that does not involve a net exchange of mass. It is due to conduction,
        turbulence and swirl velocity (if wire-wrapped). Since these phenomena 
        are all in the radial plane, the sum of all contributions should return 
        zero to conserve energy. 
        The MEX term is the analogous term in the momentum equation. 
        """
        temps = np.random.uniform(rr_data.enthalpy['T1'], 
                                  rr_data.enthalpy['T2'],
                                  simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total'])
        self._assign_parameters(simple_ctrl_rr_mixconv, temps)
        
        assert np.sum(simple_ctrl_rr_mixconv._calc_EEX(rr_data.enthalpy['dz']) \
                * simple_ctrl_rr_mixconv.params['area'][simple_ctrl_rr_mixconv.subchannel.type[
                    :simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total']]] \
                / rr_data.enthalpy['dz']) \
            == pytest.approx(0.0, abs=rr_data.enthalpy['tol_balance'])
            
        assert np.sum(simple_ctrl_rr_mixconv._calc_MEX(rr_data.enthalpy['dz']) \
                * simple_ctrl_rr_mixconv.params['area'][simple_ctrl_rr_mixconv.subchannel.type[
                    :simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total']]] \
                / rr_data.enthalpy['dz']) \
            == pytest.approx(0.0, abs=rr_data.enthalpy['tol_balance'])
            
    def test_axial_step_balance(self, simple_ctrl_rr_mixconv: dassh.MixedRegion):
        """
        Test that the axial step energy and mass balances are satisfied
        
        Parameters
        ----------
        simple_ctrl_rr_mixconv : dassh.MixedRegion
            The mixed region object to test.
        """
        q = mock_AssemblyPower(simple_ctrl_rr_mixconv) 
        temps = np.random.uniform(rr_data.enthalpy['T1'], 
                                  rr_data.enthalpy['T2'],
                                  simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total'])

        self._assign_parameters(simple_ctrl_rr_mixconv, temps)
        simple_ctrl_rr_mixconv._solve_system(rr_data.enthalpy['dz'], 
                                             2*rr_data.enthalpy['dz'], 
                                             q['pins'],
                                             q['cool'],
                                             ebal=False)
        ans = np.sum(q['pins'])*rr_data.enthalpy['dz'] + np.sum(q['cool'])*rr_data.enthalpy['dz'] 
        
        assert np.sum(simple_ctrl_rr_mixconv.sc_mfr*simple_ctrl_rr_mixconv._delta_h) \
               == pytest.approx(ans, abs=rr_data.enthalpy['tol'])
                        
        C_rho, C_v = simple_ctrl_rr_mixconv._calc_continuity_coefficients(simple_ctrl_rr_mixconv._sc_vel)
        
        assert np.sum(C_rho * simple_ctrl_rr_mixconv._delta_rho + C_v * simple_ctrl_rr_mixconv._delta_v) \
               == pytest.approx(0, abs=rr_data.enthalpy['tol'])

    def test_axial_step_zero_energy_balance(self, simple_ctrl_rr_mixconv: dassh.MixedRegion):
        """
        Test that the axial step energy balance is respected with zero input power
        
        Parameters
        ----------
        simple_ctrl_rr_mixconv: dassh.MixedRegion
            The mixed region object to test.
        """
        q = {'duct': None,
             'pins': None,
             'cool': None}
        temps = np.random.uniform(rr_data.enthalpy['T1'], 
                                  rr_data.enthalpy['T2'],
                                  simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total'])

        self._assign_parameters(simple_ctrl_rr_mixconv, temps)
        simple_ctrl_rr_mixconv._solve_system(rr_data.enthalpy['dz'], 
                                             2*rr_data.enthalpy['dz'], 
                                             q['pins'],
                                             q['cool'],
                                             ebal=False)
        assert np.sum(simple_ctrl_rr_mixconv.sc_mfr*simple_ctrl_rr_mixconv._delta_h) \
                        == pytest.approx(0, abs=rr_data.enthalpy['tol'])
        
        

        