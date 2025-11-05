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


class TestBalances():
    """
    Class to test that energy, mass, and momentum balances are satisfied 
    in the MixedRegion class
    """
    def _assign_parameters(self, rm: dassh.MixedRegion):
        """
        Initialize parameters and properties of the mixed region

        Parameters
        ----------
        rm : dassh.MixedRegion
            The mixed region object to initialize
        """
        temps = np.random.uniform(
            rr_data.enthalpy['T1'], rr_data.enthalpy['T2'],
            rm.subchannel.n_sc['coolant']['total'])
        # Assign temperatures and enthalpies
        rm.temp['coolant_int'] = temps
        rm._enthalpy = rm.coolant.enthalpy_from_temp(temps)
        # Update subchannel properties and initialize coolant parameters
        rm._update_subchannels_properties(temps)
        rm._init_static_correlated_params(np.mean(temps))
        rm.coolant_int_params['eddy'] = 1e-7
        rm.coolant_int_params['swirl'] = np.array([0, 1e-5, 1e-5])
        
    
    def _assert_energy_balance(self, q: dict[str, np.ndarray], 
                               mfr_1: np.ndarray, h_1: np.ndarray,
                               mfr_2: np.ndarray, h_2: np.ndarray, 
                               mr: dassh.MixedRegion):
        """
        Assert that the energy balance is satisfied between state 1 and state 2
        
        Parameters
        ----------
        q : dict[str, np.ndarray]
            Dictionary with power inputs from pins and coolant
        mfr_1 : np.ndarray
            Mass flow rates at state 1
        h_1 : np.ndarray
            Enthalpies at state 1
        mfr_2 : np.ndarray
            Mass flow rates at state 2
        h_2 : np.ndarray
            Enthalpies at state 2
        mr : dassh.MixedRegion
            The mixed region object to test
        """
        # Total power input from pins and coolant
        Q_in = (np.sum(q['pins']) + np.sum(q['cool'])) * rr_data.enthalpy['dz']
        # Delta(m2*h2 - m1*h1)
        delta_mh = np.sum(mfr_2 * h_2 - mfr_1 * h_1)
        # Error introduced by h_star, err = hstar * (m2 - m1)
        error_hstar = np.sum(mr.hstar * (mfr_2 - mfr_1))
        print('Delta(mh): ', delta_mh)
        print('Q_in: ', Q_in)
        print('Error introduced by h_star: ', error_hstar)
        assert np.sum(delta_mh - error_hstar) == \
            pytest.approx(Q_in, abs=rr_data.enthalpy['tol'])


    def _assert_mass_balance(self, mfr_1: np.ndarray, mfr_2: np.ndarray):
        """
        Assert that the mass balance is satisfied between state 1 and state 2
        
        Parameters
        ----------
        mfr_1 : np.ndarray
            Mass flow rates at state 1
        mfr_2 : np.ndarray
            Mass flow rates at state 2
        """
        print('mfr1: ', np.sum(mfr_1))
        print('mfr2: ', np.sum(mfr_2))
        assert np.sum(mfr_1) - np.sum(mfr_2) \
            == pytest.approx(0, abs=rr_data.enthalpy['tol'])
    
    
    def _assert_momentum_balance(self, v_1: np.ndarray, v_2: np.ndarray, 
                                 mr: dassh.MixedRegion, mfr_1: np.ndarray,
                                 mfr_2: np.ndarray):
        """
        Assert that the momentum balance is satisfied between state 1 and 
        state 2
        
        Parameters
        ----------
        v_1 : np.ndarray
            Velocities at state 1
        v_2 : np.ndarray
            Velocities at state 2
        mr : dassh.MixedRegion
            The mixed region object to test
        mfr_1 : np.ndarray
            Mass flow rates at state 1
        mfr_2 : np.ndarray
            Mass flow rates at state 2
        """
        Ai = mr.params['area'][
            mr.subchannel.type[
                :mr.subchannel.n_sc['coolant']['total']]]
        # Friction losses
        friction = 0.5 * mr.coolant_int_params['ff_i'] * \
            (mr._density - mr._delta_rho / 2) * \
                (mr._sc_vel - mr._delta_v/2)**2 / \
                    mr.params['de'][mr.subchannel.type[
                        :mr.subchannel.n_sc['coolant']['total']]] \
                            * rr_data.enthalpy['dz']
        # Gravity losses
        gravity = rr_data.enthalpy['gravity_const'] * \
            rr_data.enthalpy['dz'] * (mr._density - mr._delta_rho / 2)
        # Acceleration losses
        acceleration = (mfr_2 * v_2 - mfr_1 * v_1) / Ai
        # Error introduced by v_star, err = vstar * (m2 - m1) / A_i
        error_vstar = mr.vstar * (mfr_2 - mfr_1) / Ai
        
        print('total: ', gravity + friction + acceleration - error_vstar)
        print('Error introduced by v_star: ', error_vstar)
        print('gravity: ', gravity)
        print('friction: ', friction)
        print('acceleration: ', acceleration)
        print('delta_P: ', mr._delta_P)
        assert np.allclose(mr._delta_P + 
                           (gravity + friction + acceleration - error_vstar), 
                           0, atol=rr_data.enthalpy['tol'])
        
        
    def test_EEX_MEX_balance(self, simple_ctrl_rr_mixconv: dassh.MixedRegion):
        """
        Test that the sum of the EEX and MEX contributions is zero
        
        Parameters
        ----------
        simple_ctrl_rr_mixconv : dassh.MixedRegion
            The mixed region object to test
            
        Notes:
        ------
        The EEX term represents the exchange of energy between the subchannels
        that does not involve a net exchange of mass. It is due to conduction,
        turbulence and swirl velocity (if wire-wrapped). Since these phenomena 
        are all in the radial plane, the sum of all contributions should return 
        zero to conserve energy. 
        The MEX term is the analogous term in the momentum equation. 
        """
        self._assign_parameters(simple_ctrl_rr_mixconv)
        EEX, MEX = simple_ctrl_rr_mixconv._calc_EEX_MEX(rr_data.enthalpy['dz'])
        assert np.sum(EEX * simple_ctrl_rr_mixconv.params['area'][
            simple_ctrl_rr_mixconv.subchannel.type[
                :simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total']]] \
                / rr_data.enthalpy['dz']) \
            == pytest.approx(0.0, abs=rr_data.enthalpy['tol_balance'])
        assert np.sum(MEX * simple_ctrl_rr_mixconv.params['area'][
            simple_ctrl_rr_mixconv.subchannel.type[
                :simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total']]] \
                / rr_data.enthalpy['dz']) \
            == pytest.approx(0.0, abs=rr_data.enthalpy['tol_balance'])


    def test_axial_step_balance(self, 
                                simple_ctrl_rr_mixconv: dassh.MixedRegion):
        """
        Test that the axial step energy and mass balances are satisfied
        
        Parameters
        ----------
        simple_ctrl_rr_mixconv : dassh.MixedRegion
            The mixed region object to test
        """
        q = mock_AssemblyPower(simple_ctrl_rr_mixconv)
        self._assign_parameters(simple_ctrl_rr_mixconv)
        # Store mass flow rates, enthalpies and velocities at state 1
        mfr_1 = simple_ctrl_rr_mixconv.sc_mfr.copy()
        h_1 = simple_ctrl_rr_mixconv._enthalpy.copy()
        v_1 = simple_ctrl_rr_mixconv._sc_vel.copy()
        # Solve for state 2
        simple_ctrl_rr_mixconv._solve_system(rr_data.enthalpy['dz'], 
                                             rr_data.enthalpy['dz'], 
                                             q['pins'],
                                             q['cool'],
                                             ebal=False)
        # Store mass flow rates, enthalpies and velocities at state 2 
        mfr_2 = simple_ctrl_rr_mixconv.sc_mfr.copy()
        h_2 = simple_ctrl_rr_mixconv._enthalpy.copy()
        v_2 = simple_ctrl_rr_mixconv._sc_vel.copy()
        # Conservation of energy
        self._assert_energy_balance(q, mfr_1, h_1, mfr_2, h_2, 
                                    simple_ctrl_rr_mixconv)
        # Conservation of mass
        self._assert_mass_balance(mfr_1, mfr_2)
        # Conservation of momentum
        self._assert_momentum_balance(v_1, v_2, simple_ctrl_rr_mixconv,
                                      mfr_1, mfr_2)

        