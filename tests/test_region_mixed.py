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
from dassh._commons import rho2h_COEFF_FILE


def _assign_parameters(rm: dassh.MixedRegion):
        """
        Initialize parameters and properties of the mixed region

        Parameters
        ----------
        rm : dassh.MixedRegion
            The mixed region object to initialize
        """
        temps = np.random.uniform(rr_data.enthalpy['T1'], 
                                  rr_data.enthalpy['T2'],
                                  rm.subchannel.n_sc['coolant']['total'])
        # Assign temperatures and enthalpies
        rm.temp['coolant_int'] = temps
        rm.sc_properties['density'] = rm.coolant.density * \
            np.ones(rm.subchannel.n_sc['coolant']['total'])
        rm._enthalpy = rm.coolant.convert_properties(
            density=rm.sc_properties['density'])
        
        # Update subchannel properties and initialize coolant parameters
        rm._update_subchannels_properties(temps)
        rm._init_static_correlated_params(np.mean(temps))
        rm.coolant_int_params['eddy'] = rr_data.mixed['eddy']
        rm.coolant_int_params['swirl'] = np.array([0, rr_data.mixed['swirl'],
                                                   rr_data.mixed['swirl']])
        
class TestBalances():
    """
    Class to test that energy, mass, and momentum balances are satisfied 
    in the MixedRegion class
    """      
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
        error_hstar = np.sum(mr._hstar * (mfr_2 - mfr_1))
        print('Delta(mh): ', delta_mh)
        print('Q_in: ', Q_in)
        print('Error introduced by h_star: ', error_hstar)
        assert np.sum(delta_mh - error_hstar) == \
            pytest.approx(Q_in, abs=rr_data.mixed['tol'])


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
            == pytest.approx(0, abs=rr_data.mixed['tol'])
    
    
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
            (mr.sc_properties['density'] - mr._delta_rho / 2) * \
                (mr._sc_vel - mr._delta_v/2)**2 / \
                    mr.params['de'][mr.subchannel.type[
                        :mr.subchannel.n_sc['coolant']['total']]] \
                            * rr_data.enthalpy['dz']
        # Gravity losses
        gravity = rr_data.enthalpy['gravity_const'] * \
            rr_data.enthalpy['dz'] * (mr.sc_properties['density'] - 
                                      mr._delta_rho / 2)
        # Acceleration losses
        acceleration = (mfr_2 * v_2 - mfr_1 * v_1) / Ai
        # Error introduced by v_star, err = vstar * (m2 - m1) / A_i
        error_vstar = mr._vstar * (mfr_2 - mfr_1) / Ai
        
        print('total: ', gravity + friction + acceleration - error_vstar)
        print('Error introduced by v_star: ', error_vstar)
        print('gravity: ', gravity)
        print('friction: ', friction)
        print('acceleration: ', acceleration)
        print('delta_P: ', mr._delta_P)
        assert np.allclose(mr._delta_P,
                           - gravity - friction - acceleration + error_vstar, 
                           atol=rr_data.mixed['tol'])
        
        
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
        are all modelled as confined in the radial plane, the sum of all 
        contributions should return zero to conserve energy. 
        The MEX term is the analogous term in the momentum equation. 
        """
        _assign_parameters(simple_ctrl_rr_mixconv)
        EEX, MEX = simple_ctrl_rr_mixconv._calc_EEX_MEX(
            rr_data.enthalpy['dz'], 
            simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total'])
        assert np.sum(EEX * simple_ctrl_rr_mixconv.params['area'][
            simple_ctrl_rr_mixconv.subchannel.type[
                :simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total']]]) \
                / rr_data.enthalpy['dz'] \
            == pytest.approx(0.0, abs=rr_data.enthalpy['tol_balance'])
        assert np.sum(MEX * simple_ctrl_rr_mixconv.params['area'][
            simple_ctrl_rr_mixconv.subchannel.type[
                :simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total']]]) \
                / rr_data.enthalpy['dz'] \
            == pytest.approx(0.0, abs=rr_data.enthalpy['tol_balance'])


    def test_axial_step_balance(self, 
                                simple_ctrl_rr_mixconv: dassh.MixedRegion):
        """
        Test that the axial step energy, momentum and mass balances are
        satisfied
        
        Parameters
        ----------
        simple_ctrl_rr_mixconv : dassh.MixedRegion
            The mixed region object to test
        """
        q = mock_AssemblyPower(simple_ctrl_rr_mixconv)
        _assign_parameters(simple_ctrl_rr_mixconv)
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


class TestMethodsMixedRegion():
    """
    Class to test methods in the MixedRegion class
    """
    def _calc_and_test_star_quantities(self, rm: dassh.MixedRegion, 
                                       dv: np.ndarray, drho: np.ndarray, 
                                       RR: float, expected_hstar: np.ndarray,
                                       expected_vstar: np.ndarray) -> None:
        """
        Calculate and return the star quantities hstar and vstar
        
        Parameters
        ----------
        rm : dassh.MixedRegion
            The mixed region object to test
        dv : np.ndarray
            Velocity variations
        drho : np.ndarray
            Density variations
        RR : float
            RR coefficient
        expected_hstar : np.ndarray
            The expected hstar values
        expected_vstar : np.ndarray
            The expected vstar values
        """                              
        rm._calc_h_v_star(dv, drho, RR, rm.subchannel.n_sc['coolant']['total'])
        assert rm._hstar == pytest.approx(expected_hstar, 
                                         abs=rr_data.mixed['tol'])
        assert rm._vstar == pytest.approx(expected_vstar, 
                                         abs=rr_data.mixed['tol'])
    
    
    def test_calc_RR(self, simple_ctrl_rr_mixconv: dassh.MixedRegion):
        """
        Test the calculation of the RR coefficient
        
        Parameters
        ----------
        simple_ctrl_rr_mixconv : dassh.MixedRegion
            The mixed region object to test
        """
        nn = simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total']
        for mat in rr_data.mixed['drho'].keys():
            # Assign material and parameters (material name, initial density, 
            # initial enthalpy, rho2h conversion coeffs)
            simple_ctrl_rr_mixconv.coolant.name = mat
            simple_ctrl_rr_mixconv.sc_properties['density'] = \
                rr_data.enthalpy['density_values'][mat] * np.ones(nn)
            simple_ctrl_rr_mixconv.coolant._coeffs_rho2h = \
                simple_ctrl_rr_mixconv.coolant._read_coefficients(
                    rho2h_COEFF_FILE)
            simple_ctrl_rr_mixconv._enthalpy = \
                simple_ctrl_rr_mixconv.coolant.convert_properties(
                    density=simple_ctrl_rr_mixconv.sc_properties['density'])
                
            print('Material: ', simple_ctrl_rr_mixconv.coolant.name, mat)
            
            assert simple_ctrl_rr_mixconv._calc_RR(
                rr_data.mixed['drho'][mat] * np.ones(nn)) == \
                    pytest.approx(rr_data.mixed['RR'][mat], 
                                  rel=rr_data.mixed['tol'])
    
    def test_calc_star_quantities(self, 
                                  simple_ctrl_rr_mixconv: dassh.MixedRegion):
        """
        Test the calculation of the star quantities hstar and vstar
        
        Parameters
        ----------
        simple_ctrl_rr_mixconv : dassh.MixedRegion
            The mixed region object to test
        """
        _assign_parameters(simple_ctrl_rr_mixconv)
        # Test 1: test that hstar and vstar are calculated as mid cell values
        # by default
        expected_hstar = simple_ctrl_rr_mixconv._enthalpy + \
            rr_data.mixed['RR_star_test'] * rr_data.mixed['deltarho'] / 2
        expected_vstar = simple_ctrl_rr_mixconv._sc_vel + \
            rr_data.mixed['deltav'] / 2
        self._calc_and_test_star_quantities(
            simple_ctrl_rr_mixconv, rr_data.mixed['deltav'], 
            rr_data.mixed['deltarho'], rr_data.mixed['RR_star_test'], 
            expected_hstar, expected_vstar)   
            
        # Test 2: test that hstar and vstar are calculated as mass-weighted
        # averages when the accurate option is set to True
        simple_ctrl_rr_mixconv._accurate_star_quantities = True
        expected_hstar = rr_data.mixed['star']['h'] * np.ones(
            simple_ctrl_rr_mixconv.subchannel.n_sc['coolant']['total'])
        expected_vstar = rr_data.mixed['star']['v']
        self._calc_and_test_star_quantities(
            simple_ctrl_rr_mixconv, rr_data.mixed['deltav'], 
            rr_data.mixed['deltarho'], rr_data.mixed['RR_star_test'], 
            expected_hstar, expected_vstar)
        