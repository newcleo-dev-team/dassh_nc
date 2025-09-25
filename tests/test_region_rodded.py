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
date: 2022-07-06
author: matz
Test the DASSH RoddedRegion object (pin bundle)
"""
########################################################################
import numpy as np
import pytest
import copy
import dassh
from pytest import mat_data
from pytest import rr_data

def mock_AssemblyPower(rregion):
    """Fake a dictionary of linear power like that produced by the
    DASSH AssemblyPower object"""
    n_pin = rregion.n_pin
    n_duct_sc = rregion.subchannel.n_sc['duct']['total']
    n_cool_sc = rregion.subchannel.n_sc['coolant']['total']
    return {'pins': np.random.random(n_pin) * rr_data.mock_AP['pins'][0] 
            + rr_data.mock_AP['pins'][1],
            'duct': np.random.random(n_duct_sc) * rr_data.mock_AP['duct'][0] 
            + rr_data.mock_AP['duct'][1],
            'cool': np.random.random(n_cool_sc) + rr_data.mock_AP['cool']}
    

class TestMiscellaneous():
    """
    Class to test the flowsplit and the correlation assignment in the RoddedRegion
    """
    def test_rr_flowsplit_conservation(self, textbook_rr):
        """Test flowsplit mass conservation requirement"""
        textbook_rr._update_coolant_int_params(mat_data.water_temperature)
        total = (textbook_rr.coolant_int_params['fs'][0]
                * textbook_rr.params['area'][0]
                * textbook_rr.subchannel.n_sc['coolant']['interior'])
        total += (textbook_rr.coolant_int_params['fs'][1]
                * textbook_rr.params['area'][1]
                * textbook_rr.subchannel.n_sc['coolant']['edge'])
        total += (textbook_rr.coolant_int_params['fs'][2]
                * textbook_rr.params['area'][2]
                * textbook_rr.subchannel.n_sc['coolant']['corner'])
        total /= textbook_rr.bundle_params['area']
        assert np.abs(total - 1.0) <= rr_data.fs_tol
        

    def test_error_correlation_assignment(self, c_fuel_rr, caplog):
        """Make sure RoddedRegion fails if specified correlations 
        are not available"""
        # Gotta specify duct in the way it will be in the DASSH_Input
        duct_ftf = [item for sublist in c_fuel_rr.duct_ftf
                    for item in sublist]
        corr_dict = {'corr_friction': rr_data.corr_list[0],
                'corr_flowsplit': rr_data.corr_list[1],
                'corr_mixing': rr_data.corr_list[2],
                'corr_nusselt': rr_data.corr_list[3],
                'corr_shapefactor': None}
        passed = 0
        for corr in corr_dict.keys():
            tmp = copy.deepcopy(corr_dict)
            tmp[corr] = 'X'
            with pytest.raises(SystemExit):  
                dassh.RoddedRegion(
                    'conceptual_driver',
                    c_fuel_rr.n_ring,
                    c_fuel_rr.pin_pitch,
                    c_fuel_rr.pin_diameter,
                    c_fuel_rr.wire_pitch,
                    c_fuel_rr.wire_diameter,
                    c_fuel_rr.clad_thickness,
                    duct_ftf,
                    c_fuel_rr.int_flow_rate,
                    c_fuel_rr.coolant,
                    c_fuel_rr.duct,
                    None,
                    **tmp)
            passed += 1

        print(caplog.text)
        assert passed == len(corr_dict)
        
        
class TestGeometry():
    """
    Class to test the geometry of the RoddedRegion object
    """
    @pytest.mark.skip(reason='answers use non-costheta geom')
    def test_rr_geometry_params(self, textbook_rr):
        """Test the attributes of a RoddedRegion object; numerical values
        taken from Nuclear Systems II textbook (Todreas); Table 4-3
        page 159"""
        # print(dir(textbook_rr))
        # for x in textbook_rr.ht_consts:
        #     print(x)
        # Do the geometric attributes exist?
        # assert hasattr(textbook_rr, 'bare_params')
        assert hasattr(textbook_rr, 'params')
        assert hasattr(textbook_rr, 'bundle_params')

        # Note here that in this method I'm subtracting a little more area
        # because I'm accounting for the wire wrap angle. We're going to
        # delete that extra contribution in the test
        # cos_theta = np.cos(textbook_rr.params['theta'])
        # x = np.array([np.pi * textbook_rr.wire_diameter**2 / 8,
        #               np.pi * textbook_rr.wire_diameter**2 / 8,
        #               np.pi * textbook_rr.wire_diameter**2 / 24])
        # # Check the values for the individual coolant subchannels
        # # Subchannel area
        # ans = np.array([10.4600, 20.9838, 7.4896])
        # ans -= x * (1 / cos_theta - 1)
        # res = textbook_rr.params['area']
        # diff = res * 1e6 - ans
        # assert diff == pytest.approx(0, abs=1e-4)
        # # Wetted perimeter
        # ans = np.array([12.4690, 20.4070, 9.6562])
        # ans -= 4 * x
        # res = textbook_rr.params['wp'] + x * (1 / cos_theta - 1)
        # diff = res * 1e3 - ans
        # assert diff == pytest.approx(0, abs=1e-4)
        # # Hydraulic diameter
        # res = textbook_rr.params['de'] + x * (1 / cos_theta - 1)
        #
        # ans = np.array([3.3555, 4.1131, 3.1025])
        # diff = textbook_rr.params['de'] * 1e3 - ans
        # assert diff == pytest.approx(0, abs=1e-4)

        # Check the bulk values
        # Bundle flow area
        print(textbook_rr.bundle_params['area'] * 1e6, rr_data.geo_params_ans1)
        print(np.sqrt(3) * textbook_rr.duct_ftf[0][0]**2 / 2
            - rr_data.geo_params_nrods * np.pi * (textbook_rr.pin_diameter**2
                            + textbook_rr.wire_diameter**2) / 4)
        diff = textbook_rr.bundle_params['area'] * 1e6 - rr_data.geo_params_ans1
        assert diff == pytest.approx(0, abs=rr_data.geo_params_abs_tol1)  # <-- only given 1 dec.
        # Bundle de
        diff = textbook_rr.bundle_params['de'] * 1e3 - rr_data.geo_params_ans2
        assert diff == pytest.approx(0, abs=rr_data.geo_params_abs_tol2)  # only given 2 dec


    def test_rr_sc_areas(self, textbook_rr):
        """Test that the individual subchannel areas sum to the total"""
        tot = 0.0
        for i in range(textbook_rr.subchannel.n_sc['coolant']['total']):
            sc_type = textbook_rr.subchannel.type[i]
            tot += textbook_rr.params['area'][sc_type]
        assert pytest.approx(tot) == textbook_rr.bundle_params['area']


    def test_bypass_sc_areas(self, c_ctrl_rr):
        """Test that individual bypass subchannel areas sum to total"""
        tot = 0.0
        for i in range(c_ctrl_rr.n_bypass):
            start = (c_ctrl_rr.subchannel.n_sc['coolant']['total']
                    + c_ctrl_rr.subchannel.n_sc['duct']['total']
                    + i * (c_ctrl_rr.subchannel.n_sc['bypass']['total']
                            + c_ctrl_rr.subchannel.n_sc['duct']['total']))
            for j in range(c_ctrl_rr.subchannel.n_sc['bypass']['total']):
                sc_type = c_ctrl_rr.subchannel.type[start + j] - 5
                tot += c_ctrl_rr.bypass_params['area'][i][sc_type]
        assert pytest.approx(tot) == c_ctrl_rr.bypass_params['total area'][0]


    def test_double_duct_rr_geometry(self, c_ctrl_rr):
        """Test the geometry specifications of the double-ducted assembly
        (inner/outer ducts, bypass region)"""
        # Duct wall-corner lengths
        for d in range(c_ctrl_rr.n_duct):
            d_ftf = c_ctrl_rr.duct_ftf[d]
            for di in range(len(d_ftf)):
                hex_side = d_ftf[di] / np.sqrt(3)
                hex_perim = 6 * hex_side
                hex_edge_perim = \
                    (c_ctrl_rr.subchannel.n_sc['duct']['edge']
                    * c_ctrl_rr.L[1][1])
                hex_corner_perim = hex_perim - hex_edge_perim
                msg = 'duct: ' + str(d) + '; wall: ' + str(di)
                assert hex_corner_perim / 12 == \
                    pytest.approx(c_ctrl_rr.d['wcorner'][d][di], rr_data.double_duct_tol), msg
                    
        # duct areas
        for d in range(c_ctrl_rr.n_duct):
            assert c_ctrl_rr.duct_params['total area'][d] == \
                pytest.approx(c_ctrl_rr.subchannel.n_sc['duct']['edge']
                            * c_ctrl_rr.duct_params['area'][d][0]
                            + 6 * c_ctrl_rr.duct_params['area'][d][1],
                            rr_data.double_duct_tol)

        # bypass subchannel params
        assert c_ctrl_rr.bypass_params['total area'][0] == \
            pytest.approx(c_ctrl_rr.subchannel.n_sc['bypass']['edge']
                        * c_ctrl_rr.bypass_params['area'][0][0]
                        + 6 * c_ctrl_rr.bypass_params['area'][0][1],
                        rr_data.double_duct_tol)


    def test_single_pin_fail(self, coolant, structure):
        """Test that single pin assembly fails instantiation"""
        clad_thickness = rr_data.single_pin['clad_thickness_data'][0] * \
                         rr_data.single_pin['clad_thickness_data'][1] / 1e2 # cm -> m
        # rr = dassh.RoddedRegion('ctrl', n_ring, pin_pitch, pin_diameter,
        #                         wire_pitch, wire_diameter, clad_thickness,
        #                         duct_ftf, inlet_flow_rate, coolant, structure,
        #                         None, 'CTD', 'CTD', 'CTD', 'DB', byp_ff=0.1)
        with pytest.raises(SystemExit):
            dassh.RoddedRegion('ctrl', rr_data.single_pin['n_ring'], 
                               rr_data.single_pin['pin_pitch'], 
                               rr_data.single_pin['pin_diameter'],
                               clad_thickness,
                               rr_data.single_pin['wire_pitch'],
                               rr_data.single_pin['wire_diameter'],
                               rr_data.single_pin['duct_ftf'], 
                               rr_data.single_pin['inlet_flow_rate'],
                                coolant, structure,
                                None, None, rr_data.corr_list[0],
                                rr_data.corr_list[1], 
                                rr_data.corr_list[2],
                                rr_data.corr_list[3])


    @pytest.mark.skip(reason='No single pin functionality at the moment')
    def test_double_duct_single_pin_rr(self):
        """This is just to see if a special case will break things"""
        loc = (rr_data.loc[0], rr_data.loc[1])
        clad_thickness = rr_data.single_pin['clad_thickness_data'][0] * \
                         rr_data.single_pin['clad_thickness_data'][1] / 1e2 # cm -> m
        coolant_obj = dassh.Material('sodium')
        duct_obj = dassh.Material('ss316')
        asm = dassh.Assembly('ctrl', loc, rr_data.single_pin['n_ring'], 
                               rr_data.single_pin['pin_pitch'], 
                               rr_data.single_pin['pin_diameter'],
                               clad_thickness,
                               rr_data.single_pin['wire_pitch'],
                               rr_data.single_pin['wire_diameter'],
                               rr_data.single_pin['duct_ftf'],
                               coolant_obj,
                               duct_obj, 
                               rr_data.inlet_temp, 
                               rr_data.single_pin['inlet_flow_rate'],
                               rr_data.corr_list[0],
                               rr_data.corr_list[1], 
                               rr_data.corr_list[2])

        # asm._cleanup_1pin()
        assert all([np.all(asm.d[key] >= 0.0) for key in asm.d.keys()])
        assert all([np.all(asm.params[key] >= 0.0)
                    for key in asm.params.keys()])
        # assert all([np.all(asm.bare_params[key] >= 0.0)
        #             for key in asm.bare_params.keys()])
        assert all([np.all(asm.bundle_params[key] >= 0.0)
                    for key in asm.bundle_params.keys()])
        assert all([np.all(asm.bypass_params[key] >= 0.0)
                    for key in asm.bypass_params.keys()])
        assert all([np.all(asm.duct_params[key] >= 0.0)
                    for key in asm.duct_params.keys()])
        for i in range(len(asm.L)):
            for j in range(len(asm.L[i])):
                # assert np.all(np.array(asm.ht_consts[i][j]) >= 0.0)
                assert np.all(np.array(asm.L[i][j]) >= 0.0)


    def test_rr_duct_areas(self, textbook_rr):
        """."""
        print(textbook_rr.pin_pitch)
        print(textbook_rr.duct_params['thickness'])
        print(textbook_rr.subchannel.n_sc['duct']['edge'])
        print(textbook_rr.d['wcorner'][0])
        print(textbook_rr.duct_ftf[0])
        print(textbook_rr.duct_params['area'][0])
        assert textbook_rr.duct_params['total area'][0] == \
            pytest.approx(textbook_rr.subchannel.n_sc['duct']['edge']
                        * textbook_rr.duct_params['area'][0, 0]
                        + 6 * textbook_rr.duct_params['area'][0, 1], rr_data.duct_areas_tol)


    def test_thesis_rr_hydraulic_diam(self, thesis_asm_rr):
        """Test the MIT thesis assembly used for friction factor tests"""
        # Subchannel equivalent hydraulic diameter
        res = thesis_asm_rr.params['de'] * 1e3
        print('ans', rr_data.thesis_Dh['ans'])
        print('result', res)
        for i in range(len(res)):
            assert abs(100 * (res[i] - rr_data.thesis_Dh['ans'][i]) 
                       / rr_data.thesis_Dh['ans'][i]) < rr_data.thesis_Dh['tol']

        # Bundle average hydraulic diameter
        res2 = thesis_asm_rr.bundle_params['de'] * 1e3
        print('ans', rr_data.thesis_Dh['ans2'])
        print('result', res2)
        assert abs(100 * ((res2 - rr_data.thesis_Dh['ans2']) \
                / rr_data.thesis_Dh['ans2'])) < rr_data.thesis_Dh['tol']


    def test_error_pins_fit_in_duct(self, c_fuel_rr, caplog):
        """Test that the RoddedRegion object throws an error if the pins
        won't fit in the duct"""
        duct_ftf = [item for sublist in c_fuel_rr.duct_ftf
                    for item in sublist]
        with pytest.raises(SystemExit):
            dassh.RoddedRegion(
                'conceptual_fuel',
                c_fuel_rr.n_ring,
                c_fuel_rr.pin_pitch * 2,
                c_fuel_rr.pin_diameter,
                c_fuel_rr.wire_pitch,
                c_fuel_rr.wire_diameter,
                c_fuel_rr.clad_thickness,
                duct_ftf,
                c_fuel_rr.int_flow_rate,
                c_fuel_rr.coolant,
                c_fuel_rr.duct,
                None, None,
                rr_data.corr_list[0],
                rr_data.corr_list[1],
                rr_data.corr_list[2],
                rr_data.corr_list[3],
                None)
            assert 'Pins do not fit inside duct;' in caplog.text
    
        
class TestPower():
    """
    Class to test that the power is correctly delivered to the coolant 
    """
    def test_asm_zero_power(self, c_fuel_rr):
        """Test that the power put into the coolant subchannels is zero
        if the assigned power is zero"""
        pcoolant = np.zeros(c_fuel_rr.subchannel.n_sc['coolant']['total'])
        ppin = np.zeros(c_fuel_rr.n_pin)
        res = c_fuel_rr._calc_int_sc_power(ppin, pcoolant)
        zero_power = np.zeros(c_fuel_rr.subchannel.n_sc['coolant']['total'])
        assert np.array_equal(res, zero_power)  # all should be zero


    def test_rr_none_power(self, c_fuel_rr):
        """Test that correct power is delivered to subchannels if pin and/or
        coolant power is None"""
        # Both pin and coolant power are None: result is zero power
        res = c_fuel_rr._calc_int_sc_power(None, None)
        zero_power = np.zeros(c_fuel_rr.subchannel.n_sc['coolant']['total'])
        assert np.array_equal(res, zero_power)
        # Pin power is None: result is coolant power
        pcool = np.random.random(c_fuel_rr.subchannel.n_sc['coolant']['total'])
        res = c_fuel_rr._calc_int_sc_power(None, pcool)
        assert np.allclose(res, pcool)
        # Coolant power is None: result is distributed pin power as if no coolant
        # power is specified; use zero coolant power to check
        ppins = np.random.random(c_fuel_rr.subchannel.n_sc['coolant']['total'])
        res = c_fuel_rr._calc_int_sc_power(ppins, None)
        ans = c_fuel_rr._calc_int_sc_power(ppins, zero_power)
        assert np.allclose(res, ans)
        
        
    def test_coolant_pin_power(self, c_fuel_rr):
        """Test that the internal coolant subchannel power method reports
        the proper total power"""
        power = mock_AssemblyPower(c_fuel_rr)
        ans = np.sum(power['pins']) + np.sum(power['cool'])
        res = c_fuel_rr._calc_int_sc_power(power['pins'], power['cool'])
        assert np.sum(res) == pytest.approx(ans)
        
        
    def test_coolant_temp_roughly_qmcdt(self, c_fuel_rr):
        """Test that the change in interior subchannel coolant temperature
        over one axial step roughly approximates Q = mCdT"""
        tmp_asm = c_fuel_rr.clone()
        power = mock_AssemblyPower(c_fuel_rr)
        dz, _ = dassh.region_rodded.calculate_min_dz(tmp_asm, rr_data.inlet_temp,
                                                    rr_data.outlet_temp)

        ans = dz * (np.sum(power['pins']) + np.sum(power['cool']))

        tmp_asm._calc_coolant_int_temp(dz, power['pins'], power['cool'])
        dT = tmp_asm.temp['coolant_int'] - rr_data.inlet_temp
        tot = 0.0
        for i in range(len(tmp_asm.temp['coolant_int'])):
            tot += tmp_asm.params['area'][tmp_asm.subchannel.type[i]] * dT[i]
        dT = tot / tmp_asm.bundle_params['area']

        res = tmp_asm.int_flow_rate * tmp_asm.coolant.heat_capacity * dT

        print('dz (m): ' + str(dz))
        print('Power added (W): ' + str(ans))
        print('Mdot (kg/s): ' + str(tmp_asm.int_flow_rate))
        print('Cp (J/kgK): ' + str(tmp_asm.coolant.heat_capacity))
        print('dT (K): ' + str(dT))
        print('res (W): ' + str(res))
        assert ans == pytest.approx(res, rr_data.qmcdt_tol)
    
    
class TestCoolantIntTemperature():
    """
    Class to test the calculation of the coolant temperature 
    in the interior assembly region
    """
    def test_rr_temp_properties(self, c_fuel_rr):
        """Test that temperature property attributes return the correct
        structures and values. This is not limited to the interior subchannels,
        but here for simplicity"""

        # Coolant internal temperatures
        ans = np.ones(c_fuel_rr.subchannel.n_sc['coolant']['total']) * rr_data.inlet_temp
        assert np.array_equal(c_fuel_rr.temp['coolant_int'], ans)

        # Duct midwall temperatures
        ans = np.ones((1, c_fuel_rr.subchannel.n_sc['duct']['total'])) * rr_data.inlet_temp
        np.testing.assert_array_almost_equal(
            c_fuel_rr.temp['duct_mw'], ans, decimal=rr_data.rr_temp_decimal)

        # Duct outer surface temperatures
        ans = np.ones(c_fuel_rr.subchannel.n_sc['duct']['total']) * rr_data.inlet_temp
        np.testing.assert_array_almost_equal(
            c_fuel_rr.duct_outer_surf_temp, ans, decimal=rr_data.rr_temp_decimal)

        # Duct surface temperatures
        ans = rr_data.inlet_temp * np.ones((c_fuel_rr.n_duct, 2,
                                c_fuel_rr.subchannel.n_sc['duct']['total']))
        np.testing.assert_array_almost_equal(
            c_fuel_rr.temp['duct_surf'], ans)
    
    
    def test_rr_average_temperatures(self, textbook_active_rr):
        """Test that I can return average duct and coolant temperatures"""
        # temperature is from conftest.py
        assert textbook_active_rr.avg_coolant_int_temp == \
            pytest.approx(rr_data.avg_temp['ans'], rr_data.avg_temp['tol'])
        print(textbook_active_rr.duct_params['total area'])
        print(textbook_active_rr.subchannel.n_sc['duct'])
        print(textbook_active_rr.duct_params['area'])
        print(textbook_active_rr.d['wcorner'])
        print(textbook_active_rr.duct_ftf)
        print(textbook_active_rr.L[1][1])
        print(textbook_active_rr.avg_duct_mw_temp)
        assert textbook_active_rr.avg_duct_mw_temp == \
            pytest.approx(rr_data.avg_temp['ans'], rr_data.avg_temp['tol'])


    def test_rr_overall_average_temperatures(self, simple_ctrl_rr):
        """Test whether I can know the overall average coolant temp"""
        print(simple_ctrl_rr.avg_coolant_temp)
        assert simple_ctrl_rr.avg_coolant_temp == pytest.approx(rr_data.inlet_temp)
        
        
    def test_none_power_coolant_int_temp(self, c_fuel_rr):
        """Test that the internal coolant temperature calculation with None
        power for pins/coolant returns no temperature change"""
        T_in = c_fuel_rr.temp['coolant_int'].copy()
        c_fuel_rr._calc_coolant_int_temp(rr_data.none_pow_value, None, None)
        res = c_fuel_rr.temp['coolant_int'] - T_in
        assert np.allclose(res, 0.0)


    def test_zero_power_coolant_interior_adj_temp(self, c_fuel_rr):
        """Test that if only one subchannel has nonzero dT, only the
        adjacent channels are affected"""
        dz, _ = dassh.region_rodded.calculate_min_dz(c_fuel_rr,
                                                     rr_data.inlet_temp,
                                                     rr_data.outlet_temp)
        print('dz = ' + str(dz) + '\n')
        unperturbed_temperature = c_fuel_rr.temp['coolant_int'].copy()
        coolant_power = np.zeros(c_fuel_rr.subchannel.n_sc['coolant']['total'])
        pin_power = np.zeros(c_fuel_rr.n_pin)
        for sc in range(c_fuel_rr.subchannel.n_sc['coolant']['interior']):
            adj_sc = c_fuel_rr.subchannel.sc_adj[sc]

            # Perturb the temperature, calculate new temperatures, then
            # unperturb the temperature
            c_fuel_rr.temp['coolant_int'][sc] += rr_data.zero_pow_adj['perturb_temp']
            T_in = c_fuel_rr.temp['coolant_int'].copy()
            c_fuel_rr._calc_coolant_int_temp(rr_data.zero_pow_adj['z'], pin_power, coolant_power) 
            res = c_fuel_rr.temp['coolant_int'] - T_in
            c_fuel_rr.temp['coolant_int'] = T_in 
            c_fuel_rr.temp['coolant_int'][sc] -= rr_data.zero_pow_adj['perturb_temp']
            assert np.allclose(c_fuel_rr.temp['coolant_int'],
                               unperturbed_temperature)

            dT = []
            m = []
            for s in range(len(res)):  # only does coolant channels
                if s in adj_sc or s == sc:
                    s_type = c_fuel_rr.subchannel.type[s]
                    print(s, s_type, rr_data.inlet_temp, res[s])
                    dT.append(res[s])
                    m.append(c_fuel_rr.coolant_int_params['fs'][s_type]
                            * c_fuel_rr.int_flow_rate
                            * c_fuel_rr.params['area'][s_type]
                            / c_fuel_rr.bundle_params['area'])
                    assert res[s] != pytest.approx(0.0, abs=rr_data.zero_pow_adj['tol'])
                else:
                    assert res[s] == pytest.approx(0.0, abs=rr_data.zero_pow_adj['tol'])
            # Assert the balance
            mdT = [m[i] * dT[i] for i in range(len(dT))]
            print('dT: ' + str(dT))
            print('mdT: ' + str(mdT))
            print('bal: ' + str(sum(mdT)))
            print('\n')
            assert np.abs(sum(mdT)) == pytest.approx(0.0, abs=rr_data.zero_pow_adj['tol'])
          
            
    def test_coolant_temp_w_pin_power_indiv(self, c_fuel_rr):
        """Test that the internal coolant temperature calculation
        with no heat generation returns no temperature change"""
        tmp_asm = c_fuel_rr.clone()

        power = mock_AssemblyPower(c_fuel_rr)
        dz, _ = dassh.region_rodded.calculate_min_dz(tmp_asm, rr_data.inlet_temp, 
                                                      rr_data.outlet_temp)

        # Power added overall
        ans = dz * (np.sum(power['pins']) + np.sum(power['cool']))

        # Calculate new temperatures
        tmp_asm._calc_coolant_int_temp(dz, power['pins'], power['cool'])
        dT = tmp_asm.temp['coolant_int'] - rr_data.inlet_temp
        # Calculate Q = mCdT in each channel
        Q = 0.0
        for sc in range(len(dT)):

            sc_type = tmp_asm.subchannel.type[sc]
            mfr = (tmp_asm.coolant_int_params['fs'][sc_type]
                * tmp_asm.int_flow_rate
                * tmp_asm.params['area'][sc_type]
                / tmp_asm.bundle_params['area'])
            Q += mfr * tmp_asm.coolant.heat_capacity * dT[sc]

        print('dz (m): ' + str(dz))
        print('Power added (W): ' + str(ans))
        print('Mdot (kg/s): ' + str(tmp_asm.int_flow_rate))
        print('Cp (J/kgK): ' + str(tmp_asm.coolant.heat_capacity))
        print('Power result (W): ' + str(Q))
        assert ans == pytest.approx(Q)


    def test_zero_power_coolant_perturb_wall_temp(self, c_fuel_rr):
        """Test that if wall temperature is perturbed, only the adjacent
        edge/corner coolant subchannel has temperature change"""

        c_fuel_rr._update_coolant_int_params(rr_data.inlet_temp)
        dz, _ = dassh.region_rodded.calculate_min_dz(c_fuel_rr,
                                                     rr_data.inlet_temp, rr_data.outlet_temp)
        p_coolant = np.zeros(c_fuel_rr.subchannel.n_sc['coolant']['total'])
        p_pin = np.zeros(c_fuel_rr.n_pin)

        htc = c_fuel_rr.coolant_int_params['htc']
        cp = c_fuel_rr.coolant.heat_capacity
        mfr = (c_fuel_rr.int_flow_rate
               * c_fuel_rr.coolant_int_params['fs']
               * c_fuel_rr.params['area']
               / c_fuel_rr.bundle_params['area'])
        A = np.zeros(3)
        A[1] = c_fuel_rr.L[1][1] * dz
        # A[2] = c_fuel_rr.d['wcorner_m'][0] * 2 * dz
        A[2] = c_fuel_rr.d['wcorner'][0, 1] * 2 * dz
        # Loop over wall sc, perturb each one
        for w_sc in range(c_fuel_rr.subchannel.n_sc['duct']['total']):
            idx_sc_type = w_sc + c_fuel_rr.subchannel.n_sc['coolant']['total']
            adj_sc = c_fuel_rr.subchannel.sc_adj[idx_sc_type]

            # Perturb the duct temperature, calculate new coolant temps,
            # unperturb the duct temperature at the end
            # Index: first duct wall, inner surface
            c_fuel_rr.temp['duct_surf'][0, 0, w_sc] += rr_data.perturb_temp
            T_in = c_fuel_rr.temp['coolant_int'].copy()
            c_fuel_rr._calc_coolant_int_temp(dz, p_pin, p_coolant)
            res = c_fuel_rr.temp['coolant_int'] - T_in
            for s in range(len(res)):  # only does coolant channels
                if s in adj_sc:
                    s_type = c_fuel_rr.subchannel.type[s]
                    test = (A[s_type] * htc[s_type] * rr_data.perturb_temp
                            / mfr[s_type] / cp)
                   # if not res[s] == pytest.approx(test):
                    print('dz = ' + str(dz) + '\n')
                    print(c_fuel_rr.subchannel.n_sc)
                    print('htc expected: ' + str(htc))
                    print('cp expected: ' + str(cp))
                    print('fs expected: '
                        + str(c_fuel_rr.coolant_int_params['fs']))
                    print('wall sc: ' + str(idx_sc_type))
                    print('wall adj: ' + str(adj_sc))
                    print('wall temp: '
                        + str(c_fuel_rr.temp['duct_surf'][0, 0, w_sc]))
                    print('cool sc: ' + str(s))
                    print('cool sc type: ' + str(s_type))
                    print('cool in temp: '
                        + str(c_fuel_rr.temp['coolant_int'][s]))
                    print('cool out temp: ' + str(res[s] + rr_data.inlet_temp))
                    print('test: ' + str(test))
                    assert res[s] == pytest.approx(test)
                else:
                    assert res[s] == pytest.approx(0.0)

            # Unperturb the temperature
            c_fuel_rr.temp['coolant_int'] = T_in
            c_fuel_rr.temp['duct_surf'][0, 0, w_sc] -= rr_data.perturb_temp
       
        
    def test_zero_power_coolant_int_temp(self, c_fuel_rr):
        """Test that the internal coolant temperature calculation
        with no heat generation returns no temperature change"""
        pcoolant = np.zeros(c_fuel_rr.subchannel.n_sc['coolant']['total'])
        pin_power = np.zeros(c_fuel_rr.n_pin)
        T_in = c_fuel_rr.temp['coolant_int'].copy()
        c_fuel_rr._calc_coolant_int_temp(rr_data.zero_pow_cool_val, pin_power, pcoolant)
        res = c_fuel_rr.temp['coolant_int'] - T_in
        # Temperature should be unchanged relative to the previous level
        assert np.allclose(res, 0.0)
        
        
class TestDuctTemperature():
    """
    Class to test the calculation of the duct temperature 
    """
    def test_zero_power_duct_temp(self, c_fuel_rr):
        """Test that the internal coolant temperature calculation
        with no heat generation returns no temperature change"""
        gap_temps = (np.ones(c_fuel_rr.subchannel.n_sc['duct']['total'])
                    * c_fuel_rr.avg_coolant_int_temp)
        c_fuel_rr._update_coolant_int_params(rr_data.inlet_temp)
        gap_htc = c_fuel_rr.coolant_int_params['htc'][1:]
        gap_htc = gap_htc[
                c_fuel_rr.subchannel.type[
                c_fuel_rr.subchannel.n_sc['coolant']['interior']:
                c_fuel_rr.subchannel.n_sc['coolant']['total']] - 1]
        duct_power = np.zeros(c_fuel_rr.n_duct
                            * c_fuel_rr.subchannel.n_sc['duct']['total'])
        c_fuel_rr._calc_duct_temp(duct_power, gap_temps, gap_htc)
        # res_mw = zero_pow_asm.duct_midwall_temp
        # res_surf = zero_pow_asm.duct_surface_temp

        print('duct k (W/mK): ' + str(c_fuel_rr.duct.thermal_conductivity))
        print('duct thickness (m): ' + str(c_fuel_rr.duct_params['thickness']))

        # Temperature should be unchanged relative to the previous level
        assert np.allclose(c_fuel_rr.temp['duct_mw'], rr_data.inlet_temp)
        assert np.allclose(c_fuel_rr.temp['duct_surf'], rr_data.inlet_temp)


    def test_duct_temp_w_power_indiv(self, c_fuel_rr):
        """Test that duct temperature calculation gives reasonable result
        with power assignment and no temperature differential"""
        gap_temps = (np.ones(c_fuel_rr.subchannel.n_sc['duct']['total'])
                    * c_fuel_rr.avg_coolant_int_temp)
        c_fuel_rr._update_coolant_int_params(rr_data.inlet_temp)
        gap_htc = c_fuel_rr.coolant_int_params['htc'][1:]
        gap_htc = gap_htc[c_fuel_rr.subchannel.type[
                          c_fuel_rr.subchannel.n_sc['coolant']['interior']:
                          c_fuel_rr.subchannel.n_sc['coolant']['total']] - 1]
        dz, _ = dassh.region_rodded.calculate_min_dz(c_fuel_rr,
                                                     rr_data.inlet_temp,
                                                     rr_data.outlet_temp)

        # Power added overall
        power = mock_AssemblyPower(c_fuel_rr)
        ans = np.sum(power['duct'] * dz)

        # Calculate new temperatures
        c_fuel_rr._calc_duct_temp(power['duct'], gap_temps, gap_htc)
        dT_s = c_fuel_rr.temp['duct_surf'] - rr_data.inlet_temp

        # Steady state - all heat added to ducts is leaving ducts via
        # convection to the adjacent coolant
        surface_area = np.array([[c_fuel_rr.L[1][1] * dz,
                                c_fuel_rr.L[1][1] * dz],
                                [2 * c_fuel_rr.d['wcorner'][0, 1] * dz,
                                2 * c_fuel_rr.d['wcorner'][0, 1] * dz]])

        Q = 0.0
        start = c_fuel_rr.subchannel.n_sc['coolant']['total']
        for i in range(c_fuel_rr.n_duct):
            for sc in range(c_fuel_rr.subchannel.n_sc['duct']['total']):
                sc_type = c_fuel_rr.subchannel.type[sc + start] - 3
                htc = c_fuel_rr.coolant_int_params['htc'][1:][sc_type]
                qtmp_in = htc * dT_s[0, 0, sc] * surface_area[sc_type, 0]
                qtmp_out = htc * dT_s[0, 1, sc] * surface_area[sc_type, 1]
                # if sc == 0:
                #    print(q[sc], sc_type, htc, dT_s[0, 0, sc],
                #          qtmp_in, dT_s[0, 1, sc], qtmp_out)
                Q += qtmp_in + qtmp_out
        assert ans == pytest.approx(Q)


    def test_duct_temp_w_power_adiabatic(self, c_fuel_rr):
        """Test adiabatic flag for duct temp calculation"""
        # 2020-12-09: removed coolant parameter update from duct temp
        # method so I need to do it externally here...no biggie.
        c_fuel_rr._update_coolant_int_params(c_fuel_rr.avg_coolant_int_temp)
        p_duct = np.ones(c_fuel_rr.subchannel.n_sc['duct']['total']) * rr_data.duct_temp_mf
        t_gap = np.zeros(c_fuel_rr.subchannel.n_sc['duct']['total'])
        c_fuel_rr._calc_duct_temp(p_duct, t_gap, np.ones(2), True)
        # Not a "real" average but it'll do the trick
        print(np.average(c_fuel_rr.temp['duct_surf'][0, 0]))
        print(np.average(c_fuel_rr.temp['duct_mw'][0]))
        print(np.average(c_fuel_rr.temp['duct_surf'][0, 1]))
        # midwall temp should be higher than inner wall temp
        assert np.all(c_fuel_rr.temp['duct_mw'][0]
                    > c_fuel_rr.temp['duct_surf'][0, 0])
        # outer wall temp should be highest of all!
        assert np.all(c_fuel_rr.temp['duct_mw'][0]
                    < c_fuel_rr.temp['duct_surf'][0, 1])
    
    
class TestBypassTemperature():
    """
    Class to test the calculation of the bypass temperature 
    """
    def test_byp_coolant_temps_zero_dT(self, c_ctrl_rr):
        """Test that bypass coolant temperatures are unchanged when
        adjacent ducts have equal temperature"""
        print(c_ctrl_rr.n_bypass)
        print(c_ctrl_rr.temp['duct_surf'].shape)
        dT_byp = c_ctrl_rr._calc_coolant_byp_temp(rr_data.byp_val)
        assert np.allclose(dT_byp, 0.0)


    def test_bypass_perturb_wall_temps(self, c_ctrl_rr):
        """Test that perturbations in adjacent wall mesh cells affect
        only adjacent bypass coolant subchannels"""
        c_ctrl_rr._update_coolant_byp_params([rr_data.inlet_temp])
        dz, _ = dassh.region_rodded.calculate_min_dz(c_ctrl_rr,
                                                    rr_data.inlet_temp,
                                                    rr_data.outlet_temp)
        htc = c_ctrl_rr.coolant_byp_params['htc'][0]
        cp = c_ctrl_rr.coolant.heat_capacity
        mfr = (c_ctrl_rr.byp_flow_rate[0]
            * c_ctrl_rr.bypass_params['area'][0]
            / c_ctrl_rr.bypass_params['total area'][0])
        A = np.zeros((2, 2))
        A[0, 0] = c_ctrl_rr.L[5][5][0] * dz
        A[0, 1] = c_ctrl_rr.d['wcorner'][0, 1] * 2 * dz
        A[1, 0] = A[0, 0]
        A[1, 1] = c_ctrl_rr.d['wcorner'][1, 1] * 2 * dz

        surf = {0: 1, 1: 0}
        byp_types = {6: 'edge', 7: 'corner'}
        # Loop over wall sc, perturb each one
        for i in range(c_ctrl_rr.n_duct):
            for w_sc in range(c_ctrl_rr.subchannel.n_sc['duct']['total']):
                # index of the adjacent coolant subchannel
                wsc_type_idx = \
                    (w_sc
                    + c_ctrl_rr.subchannel.n_sc['coolant']['total']
                    + i * c_ctrl_rr.subchannel.n_sc['duct']['total']
                    + i * c_ctrl_rr.subchannel.n_sc['bypass']['total'])

                # find subchannels next to the wall
                adj_sc = c_ctrl_rr.subchannel.sc_adj[wsc_type_idx]

                # Perturb the duct temperature, calculate new coolant
                # temps; unperturb the duct temperature at the end
                # Index: i-th duct, surface (in: 0; out: 1), last position
                c_ctrl_rr.temp['duct_surf'][i, surf[i], w_sc] += rr_data.perturb_temp
                res = c_ctrl_rr._calc_coolant_byp_temp(dz)

                # Loop over all bypass coolant results - if the channel
                # is adjacent to the perturbed wall, we should know its
                # temperature; otherwise, its temperature should be
                # unchanged.
                for s in range(len(res[0])):
                    byp_idx = \
                        (s
                        + c_ctrl_rr.subchannel.n_sc['coolant']['total']
                        + c_ctrl_rr.subchannel.n_sc['duct']['total'])
                    if byp_idx in adj_sc:
                        # Determine what type of bypass channel you have
                        # s_type = c_ctrl_rr.subchannel.type[byp_idx] - 1
                        s_type = c_ctrl_rr.subchannel.type[byp_idx]
                        # Expected dT - only conv from the one wall=
                        # test = (A[i, s_type - 5]
                        #         * htc[s_type - 5]
                        #         * perturb_temp
                        #         / fr[s_type - 5] / cp)
                        test = (A[i, s_type - 5]
                                * htc[s_type - 5]
                                * rr_data.perturb_temp
                                / mfr[s_type - 5] / cp)
                        if not res[0, s] == pytest.approx(test):
                            print('dz = ' + str(dz))
                            print('byp sc: ' + str(byp_idx + 1))
                            # print('byp sc type: ' + str(byp_types[byp_idx + 1]))
                            print('byp sc type: ' + str(s_type + 1)
                                + '; ' + str(byp_types[s_type + 1]))
                            print('perturbed wall sc: ' + str(wsc_type_idx + 1))
                            print('perturbed wall adj: ' + str(adj_sc + 1))
                            print('htc expected: ' + str(htc[s]))
                            print('cp expected: ' + str(cp))
                            print('fr expected: ' + str(mfr[s]))
                            print('area: ' + str(A[i, s_type - 5]))
                            print('wall temp: '
                                + str(c_ctrl_rr
                                        .temp['duct_surf'][i, surf[i], s]))
                            print('byp in temp: '
                                + str(c_ctrl_rr
                                        .temp['coolant_byp'][0, -1, s]))
                            print('byp out temp: ' + str(res[1, 0, s]))
                            print('test: ' + str(test))
                            assert res[0, s] == pytest.approx(test)
                    else:
                        assert res[0, s] == pytest.approx(0.0)

                # Unperturb the temperature
                c_ctrl_rr.temp['duct_surf'][i, surf[i], w_sc] -= rr_data.perturb_temp
    
    
class TestAcceleratedMethod():
    """
    Class to test that the accelerated method for calculating the coolant 
    temperature gives the same results as in the previous versions of DASSH
    """
    def __calc_coolant_int_temp_old(self, rr_obj, consts, dz, pin_power, cool_power):
            """Calculate assembly coolant temperatures at next axial mesh

            Parameters
            ----------
            rr_obj : DASSH RoddedRegion object
                Rodded region object
            consts : list
                Multiplicative constants for heat transfer
            dz : float
                Axial step size (m)
            pin_power : numpy.ndarray
                Linear power generation (W/m) for each pin in the assembly
            cool_power : numpy.ndarray
                Linear power generation (W/m) for each coolant subchannel

            Returns
            -------
            numpy.ndarray
                Vector (length = # coolant subchannels) of temperatures
                (K) at the next axial level

            """

            # Power from pins and neutron/gamma reactions with coolant
            q = rr_obj._calc_int_sc_power(pin_power, cool_power)

            # PRECALCULATE CONSTANTS ---------------------------------------
            # Effective thermal conductivity
            keff = (rr_obj.coolant_int_params['eddy']
                    * rr_obj.coolant.density
                    * rr_obj.coolant.heat_capacity
                    + rr_obj._sf * rr_obj.coolant.thermal_conductivity)

            # This factor is in many terms; technically, the mass flow
            # rate is already accounted for in constants defined earlier
            mCp = rr_obj.coolant.heat_capacity * rr_obj.coolant_int_params['fs']

            # The mass flow rate term to be used as denominator for
            # the q term hasn't been calculated yet, 
            # so do that now (store this eventually)
            heat_added_denom = [rr_obj.int_flow_rate
                                * rr_obj.params['area'][i]
                                / rr_obj.bundle_params['area'] for i in range(3)]

            # Precalculate some other stuff
            conduction_consts = [[consts[i][j] * keff
                                for j in range(3)] for i in range(3)]
            convection_consts = [consts[i][i + 2]
                                * rr_obj.coolant_int_params['htc'][i]
                                for i in range(3)]
            swirl_consts = [rr_obj.coolant.density
                            * rr_obj.coolant_int_params['swirl'][i]
                            * rr_obj.d['pin-wall']
                            * rr_obj.bundle_params['area']
                            / rr_obj.coolant_int_params['fs'][i]
                            / rr_obj.params['area'][i]
                            / rr_obj.int_flow_rate
                            for i in range(3)]

            # Calculate the change in temperature in each subchannel
            dT = np.zeros(rr_obj.subchannel.n_sc['coolant']['total'])
            for sci in range(rr_obj.subchannel.n_sc['coolant']['total']):

                # The value of sci is the PYTHON indexing
                type_i = rr_obj.subchannel.type[sci]

                # Heat from adjacent fuel pins
                dT[sci] += q[sci] / heat_added_denom[type_i]

                for adj in rr_obj.subchannel.sc_adj[sci]:
                    # if adj == 0:
                    if adj == -1:
                        continue

                    # Adjacent cell type in PYTHON indexing
                    type_a = rr_obj.subchannel.type[adj]

                    # Conduction to/from adjacent coolant subchannels
                    if type_a <= 2:
                        dT[sci] += (conduction_consts[type_i][type_a]
                                    * (rr_obj.temp['coolant_int'][adj]
                                    - rr_obj.temp['coolant_int'][sci]))

                    # Convection to/from duct wall (type has to be 3 or 4)
                    else:
                        sc_wi = sci - (rr_obj.subchannel.n_sc['coolant']
                                                        ['interior'])
                        dT[sci] += (convection_consts[type_i]
                                    * (rr_obj.temp['duct_surf'][0, 0, sc_wi]
                                    - rr_obj.temp['coolant_int'][sci]))

                # Divide through by mCp
                dT[sci] /= mCp[type_i]

                # Swirl flow from adjacent subchannel; =0 for interior sc
                # The adjacent subchannel is the one the swirl flow is
                # coming from i.e. it's in the opposite direction of the
                # swirl flow itself. Recall that the edge/corner sub-
                # channels are indexed in the clockwise direction.
                # Example: Let sci == 26. The next subchannel in the clock-
                # wise direction is 27; the preceding one is 25.
                # - clockwise: use 25 as the swirl adjacent sc
                # - counterclockwise: use 27 as the swirl adjacent sc
                if type_i > 0:
                    dT[sci] += \
                        (swirl_consts[type_i]
                        * (rr_obj.temp['coolant_int']
                                    [rr_obj.subchannel.sc_adj[sci]
                                                            [rr_obj._adj_sw]]
                            - rr_obj.temp['coolant_int'][sci]))
            return dT * dz
        
        
    def __calc_coolant_byp_temp_old(self, rr_obj, dz, consts):
            """Calculate the coolant temperatures in the assembly bypass
            channels at the axial level j+1

            Parameters
            ----------
            rr_obj : DASSH RoddedRegion object
                Rodded region object
            dz : float
                Axial step size (m)
            consts : list
                Heat transfer constants
            Notes
            -----
            The coolant in the bypass channels is assumed to get no
            power from neutron/gamma heating (that contribution to
            coolant in the assembly interior is already small enough).

            """
            # Calculate the change in temperature in each subchannel
            dT = np.zeros((rr_obj.n_bypass,
                        rr_obj.subchannel.n_sc['bypass']['total']))
            for i in range(rr_obj.n_bypass):
                # starting index to lookup type is after all interior
                # coolant channels and all preceding duct and bypass
                # channels
                start = (rr_obj.subchannel.n_sc['coolant']['total']
                        + rr_obj.subchannel.n_sc['duct']['total']
                        + i * rr_obj.subchannel.n_sc['bypass']['total']
                        + i * rr_obj.subchannel.n_sc['duct']['total'])

                for sci in range(0, rr_obj.subchannel.n_sc['bypass']['total']):

                    # The value of sci is the PYTHON indexing
                    type_i = rr_obj.subchannel.type[sci + start]

                    # Heat transfer to/from adjacent subchannels
                    for adj in rr_obj.subchannel.sc_adj[sci + start]:
                        if adj == -1:
                            continue
                        type_a = rr_obj.subchannel.type[adj]

                        # Convection to/from duct wall
                        if 3 <= type_a <= 4:
                            if sci + start > adj:  # INTERIOR adjacent duct wall
                                byp_conv_const = \
                                    consts[type_i][type_a][i][0]
                                byp_conv_dT = \
                                    (rr_obj.temp['duct_surf'][i, 1, sci]
                                    - rr_obj.temp['coolant_byp'][i, sci])
                            else:  # EXTERIOR adjacent duct wall
                                byp_conv_const = \
                                    consts[type_i][type_a][i][1]
                                byp_conv_dT = \
                                    (rr_obj.temp['duct_surf'][i + 1, 0, sci]
                                    - rr_obj.temp['coolant_byp'][i, sci])

                            dT[i, sci] += \
                                (rr_obj.coolant_byp_params['htc'][i, type_i - 5]
                                * dz * byp_conv_const * byp_conv_dT
                                / rr_obj.coolant.heat_capacity)

                        # Conduction to/from adjacent coolant subchannels
                        else:
                            sc_adj = adj - start
                            dT[i, sci] += \
                                (rr_obj.coolant.thermal_conductivity
                                * dz
                                * consts[type_i][type_a][i]
                                * (rr_obj.temp['coolant_byp'][i, sc_adj]
                                    - rr_obj.temp['coolant_byp'][i, sci])
                                / rr_obj.coolant.heat_capacity)

            return dT
        
        
    def test_accelerated_coolant_sc_method_against_old(self, c_fuel_rr):
        """Confirm numpy coolant subchannel calculation gets same result
        as the old one (this let's me preserve the old just in case)"""
        OLD_HTCONSTS = dassh.region_rodded.calculate_ht_constants(c_fuel_rr)

        pin_power = rr_data.acc_met['pin_pow'][0] \
                    + rr_data.acc_met['pin_pow'][1] * np.random.random(c_fuel_rr.n_pin)
        cool_power = rr_data.acc_met['cool_pow'][0] \
                    + rr_data.acc_met['cool_pow'][1] * np.random.random(len(c_fuel_rr.temp['coolant_int']))
        dT_old = self.__calc_coolant_int_temp_old(c_fuel_rr,
            OLD_HTCONSTS, rr_data.acc_met['dz'], pin_power, cool_power)
        c_fuel_rr._calc_coolant_int_temp(rr_data.acc_met['dz'], pin_power, cool_power)
        dT = c_fuel_rr.temp['coolant_int'] - rr_data.inlet_temp
        print(np.average(dT))
        print('max abs diff: ', np.max(np.abs(dT - dT_old)))
        assert np.allclose(dT, dT_old)


    def test_accelerated_bypass_method_against_old(self, c_ctrl_rr):
        """Confirm that my changes to the bypass method maintain the same
        result as the old method"""
        OLD_HTCONSTS = dassh.region_rodded.calculate_ht_constants(c_ctrl_rr)

        for i in range((c_ctrl_rr.subchannel.n_sc['bypass']['total'])):
            duct_surf_temp = \
                (np.random.random(c_ctrl_rr.temp['duct_surf'].shape)
                + (rr_data.inlet_temp + i * 1.0))

        c_ctrl_rr.temp['duct_surf'] = duct_surf_temp
        dT_old = self.__calc_coolant_byp_temp_old(c_ctrl_rr, rr_data.acc_met['dz'], OLD_HTCONSTS)
        dT = c_ctrl_rr._calc_coolant_byp_temp(rr_data.acc_met['dz'])

        print(np.average(dT))
        print(np.average(dT_old))
        print('max abs diff: ', np.max(np.abs(dT - dT_old)))
        assert np.allclose(dT, dT_old)
    
    
class TestClone():
    """
    Class to test the clone method of the RoddedRegion object
    """
    def test_rr_clone_shallow(self, textbook_rr):
        """Test that assembly clone has correct shallow-copied attributes"""
        clone = textbook_rr.clone()
        assert id(clone) != id(textbook_rr)
        non_matches = []
        # Note: These attributes are immutable and therefore won't be
        # "deepcopied" to a new position:
        # 'n_ring', 'n_pin', 'pin_pitch', 'pin_diameter',
        # 'clad_thickness', 'wire_pitch', 'wire_diameter',
        # 'n_duct', 'n_bypass', 'kappa',
        # 'int_flow_rate',
        for attr in rr_data.clone_data['attr_list1']:
            id_clone = id(getattr(clone, attr))
            id_original = id(getattr(textbook_rr, attr))
            if id_clone == id_original:  # they should be the same
                continue
            else:
                non_matches.append(attr)
                print(attr, id_clone, id_original)
        assert len(non_matches) == 0


    def test_rr_clone_deep(self, textbook_rr):
        """Test that RoddedRegion clone has deep-copied attributes"""
        clone = textbook_rr.clone()
        assert id(clone) != id(textbook_rr)
        matches = []
        for attr in rr_data.clone_data['attr_list2']:
            id_clone = id(getattr(clone, attr))
            id_original = id(getattr(textbook_rr, attr))
            if id_clone != id_original:  # they should be different
                continue
            else:
                matches.append(attr)
                print(attr, id_clone, id_original)
        assert len(matches) == 0


    def test_assembly_clone_new_fr(self, textbook_rr):
        """Test behavior of clone method with new flowrate spec"""
        clone = textbook_rr.clone(rr_data.clone_data['fr'])
        assert clone.total_flow_rate != textbook_rr.total_flow_rate
        assert clone.total_flow_rate == rr_data.clone_data['fr']
        print(clone.total_flow_rate)
        print(textbook_rr.total_flow_rate)
        matches = []
        for attr in rr_data.clone_data['attr_list3']:
            id_clone = id(getattr(clone, attr))
            id_original = id(getattr(textbook_rr, attr))
            if id_clone != id_original:  # They should be different
                continue
            else:
                matches.append(attr)
                print(attr, id_clone, id_original)
        assert len(matches) == 0


class TestNonIsotropic():
    """
    Class to test the RoddedRegion class with non-isotropic properties
    """
    def test_update_subchannels_properties(self, simple_ctrl_rr_non_iso):
        """
        Test the _update_subchannels_properties method of RoddedRegion
        
        Parameters
        ----------
        simple_ctrl_rr_non_iso : dassh.region_rodded.RoddedRegion
            Simple RoddedRegion object with non-isotropic properties
        """
        simple_ctrl_rr_non_iso._update_subchannels_properties(
            np.array(rr_data.non_isotropic['sc_temps']))
        for prop in mat_data.properties_list:
            assert simple_ctrl_rr_non_iso.sc_properties[prop] == \
                pytest.approx(rr_data.non_isotropic[prop], 
                    rel = rr_data.non_isotropic['tol1'])


    def test_avg_coolant_int_temp(self, simple_ctrl_rr_non_iso):
        """
        Test the avg_coolant_int_temp method of RoddedRegion
        
        Parameters
        ----------
        simple_ctrl_rr_non_iso : dassh.region_rodded.RoddedRegion
            Simple RoddedRegion object with non-isotropic properties
        """
        simple_ctrl_rr_non_iso.temp['coolant_int'] = \
            np.array(rr_data.non_isotropic['sc_temps'])
        simple_ctrl_rr_non_iso._update_subchannels_properties(
            np.array(rr_data.non_isotropic['sc_temps']))
        assert simple_ctrl_rr_non_iso.avg_coolant_int_temp == \
            pytest.approx(rr_data.non_isotropic['Tavg_ans'], rel = rr_data.non_isotropic['tol2'])
       
        
    def test_non_isotropic_htc(self, simple_ctrl_rr_non_iso):
        """
        Test the _calculate_htc method of RoddedRegion
        
        Parameters
        ----------
        simple_ctrl_rr_non_iso : dassh.region_rodded.RoddedRegion
            Simple RoddedRegion object with non-isotropic properties
        """
        simple_ctrl_rr_non_iso._update_subchannels_properties(
            np.array(rr_data.non_isotropic['sc_temps']))
        simple_ctrl_rr_non_iso._calculate_htc()
        assert simple_ctrl_rr_non_iso.coolant_int_params['sc_htc'] == \
            pytest.approx(rr_data.non_isotropic['htc_ans'], rel = rr_data.non_isotropic['tol1'])
        
    def test_zero_power_coolant_interior_adj_temp_non_iso(self, simple_ctrl_rr_non_iso):
        """Test that if only one subchannel has nonzero dT, only the
        adjacent channels are affected"""
        unperturbed_temperature = simple_ctrl_rr_non_iso.temp['coolant_int'].copy()
       # simple_ctrl_rr_non_iso._update_subchannels_properties(simple_ctrl_rr_non_iso.temp['coolant_int'])
        coolant_power = np.zeros(simple_ctrl_rr_non_iso.subchannel.n_sc['coolant']['total'])
        pin_power = np.zeros(simple_ctrl_rr_non_iso.n_pin)
        for sc in range(simple_ctrl_rr_non_iso.subchannel.n_sc['coolant']['interior']):
            adj_sc = simple_ctrl_rr_non_iso.subchannel.sc_adj[sc]

            # Perturb the temperature, calculate new temperatures, then
            # unperturb the temperature
            simple_ctrl_rr_non_iso.temp['coolant_int'][sc] += rr_data.zero_pow_adj['perturb_temp']
            T_in = simple_ctrl_rr_non_iso.temp['coolant_int'].copy()
            simple_ctrl_rr_non_iso._update_subchannels_properties(simple_ctrl_rr_non_iso.temp['coolant_int'])
            simple_ctrl_rr_non_iso._calc_coolant_int_temp(rr_data.zero_pow_adj['z'], pin_power, coolant_power)
            res = simple_ctrl_rr_non_iso.temp['coolant_int'] - T_in
            simple_ctrl_rr_non_iso.temp['coolant_int'] = T_in
            simple_ctrl_rr_non_iso.temp['coolant_int'][sc] -= rr_data.zero_pow_adj['perturb_temp']
            assert np.allclose(simple_ctrl_rr_non_iso.temp['coolant_int'],
                               unperturbed_temperature)

            dT = []
            m = []
            cp = []
            for s in range(len(res)):  # only does coolant channels
                if s in adj_sc or s == sc:
                    s_type = simple_ctrl_rr_non_iso.subchannel.type[s]
                    print(s, s_type, rr_data.inlet_temp, res[s])
                    dT.append(res[s])
                    m.append(simple_ctrl_rr_non_iso.sc_mfr[s])
                    cp.append(simple_ctrl_rr_non_iso.sc_properties['heat_capacity'][s])
                    assert res[s] != pytest.approx(0.0, abs=rr_data.zero_pow_adj['tol'])
                else:
                    assert res[s] == pytest.approx(0.0, abs=rr_data.zero_pow_adj['tol'])
            # Assert the balance
            mdT = [cp[i]*m[i] * dT[i] for i in range(len(dT))]
            print('dT: ' + str(dT))
            print('mdT: ' + str(mdT))
            print('bal: ' + str(sum(mdT)))
            print('\n')
            print(simple_ctrl_rr_non_iso._calc_int_sc_power(pin_power, coolant_power))
            assert np.abs(sum(mdT)) == pytest.approx(0.0, abs=rr_data.zero_pow_adj['tol'])
          
            
    def test_coolant_temp_w_pin_power_indiv_non_iso(self, simple_ctrl_rr_non_iso):
        """Test that the internal coolant temperature calculation
        with no heat generation returns no temperature change"""
        tmp_asm = simple_ctrl_rr_non_iso.clone()

        power = mock_AssemblyPower(simple_ctrl_rr_non_iso)
        dz, _ = dassh.region_rodded.calculate_min_dz(tmp_asm, rr_data.inlet_temp, 
                                                      rr_data.outlet_temp)
        ans = dz * tmp_asm._calc_int_sc_power(power['pins'], power['cool'])
        # Calculate new temperatures
        tmp_asm._calc_coolant_int_temp(dz, power['pins'], power['cool'])
        dT = tmp_asm.temp['coolant_int'] - rr_data.inlet_temp
        # Calculate Q = mCdT in each channel
        Q = tmp_asm.sc_properties['heat_capacity'] * tmp_asm.sc_mfr * dT        
        
        print('dz (m): ' + str(dz))
        print('Power added (W): ' + str(ans))
        print('Mdot (kg/s): ' + str(tmp_asm.int_flow_rate))
        print('Cp (J/kgK): ' + str(tmp_asm.coolant.heat_capacity))
        print('Power result (W): ' + str(Q))
        assert np.allclose(ans, Q)
        
class TestEnthalpy():
    """
    Class to test the enthalpy calculation in the RoddedRegion
    """

    def test_calc_delta_h(self, simple_ctrl_rr_ent: dassh.RoddedRegion):
        """
        Test the _calc_delta_h method of RoddedRegion
        
        Parameters
        ----------
        simple_ctrl_rr_ent : dassh.RoddedRegion
            The RoddedRegion object to test
        """
        for mat in rr_data.enthalpy['delta_h'].keys():
            simple_ctrl_rr_ent.coolant.name = mat
            simple_ctrl_rr_ent.enthalpy_coeffs = simple_ctrl_rr_ent._read_enthalpy_coefficients()
            assert simple_ctrl_rr_ent._calc_delta_h(rr_data.enthalpy['T1'], rr_data.enthalpy['T2']) == \
                pytest.approx(rr_data.enthalpy['delta_h'][mat], abs=rr_data.enthalpy['tol'])

    def test_temp_from_enthalpy(self, simple_ctrl_rr_ent: dassh.RoddedRegion):
        """
        Test the _temp_from_enthalpy method of RoddedRegion
        
        Parameters
        ----------
        simple_ctrl_rr_ent : dassh.RoddedRegion
            The RoddedRegion object to test
        """
        simple_ctrl_rr_ent.temp['coolant_int'] = \
            np.array(rr_data.enthalpy['T1'] * np.ones(simple_ctrl_rr_ent.subchannel.n_sc['coolant']['total']))

        for mat in rr_data.enthalpy['delta_h'].keys():
            simple_ctrl_rr_ent.coolant.name = mat
            simple_ctrl_rr_ent.enthalpy_coeffs = simple_ctrl_rr_ent._read_enthalpy_coefficients()
            dh = rr_data.enthalpy['delta_h'][mat] * np.ones(simple_ctrl_rr_ent.subchannel.n_sc['coolant']['total'])
            assert simple_ctrl_rr_ent._temp_from_enthalpy(dh) == \
                pytest.approx(rr_data.enthalpy['T2'] * np.ones(simple_ctrl_rr_ent.subchannel.n_sc['coolant']['total']),
                              abs=rr_data.enthalpy['tol'])

    def test_temp_from_enthalpy_zero_deltah(self, simple_ctrl_rr_ent: dassh.RoddedRegion):
        """
        Test that the _temp_from_enthalpy method of RoddedRegion with zero deltaH
        returns zero deltaT
        
        Parameters
        ----------
        simple_ctrl_rr_ent : dassh.RoddedRegion
            The RoddedRegion object to test
        """
        dh = np.zeros(simple_ctrl_rr_ent.subchannel.n_sc['coolant']['total'])
        assert simple_ctrl_rr_ent._temp_from_enthalpy(dh) == \
            pytest.approx(simple_ctrl_rr_ent.temp['coolant_int'],
                          abs=rr_data.enthalpy['tol'])
            
    def test_coolant_int_temp_no_power(self, simple_ctrl_rr_ent: dassh.RoddedRegion):        
        """
        Test that the internal coolant temperature calculation
        with enthalpy and zero power returns no temperature change
        
        Parameters
        ----------
        simple_ctrl_rr_ent : dassh.RoddedRegion
            The RoddedRegion object to test
        """
        tmp_asm = simple_ctrl_rr_ent.clone()

        qpins = np.zeros(tmp_asm.n_pin)
        qcool = np.zeros(tmp_asm.subchannel.n_sc['coolant']['total'])
        power = {'pins': qpins, 'cool': qcool}
        dz, _ = dassh.region_rodded.calculate_min_dz(tmp_asm, rr_data.inlet_temp,
                                                      rr_data.outlet_temp)
        
        # Calculate new temperatures and deltaT
        tmp_asm._calc_coolant_int_temp(dz, power['pins'], power['cool'])
        assert np.allclose(tmp_asm.temp['coolant_int'], 
                           rr_data.inlet_temp*np.ones(tmp_asm.subchannel.n_sc['coolant']['total']),
                           atol=rr_data.enthalpy['tol'])

    def test_coolant_int_temp(self, simple_ctrl_rr_ent: dassh.RoddedRegion):
        """
        Test that the internal coolant temperature calculation
        with enthalpy satisfies the energy balance

        Parameters
        ----------
        simple_ctrl_rr_ent : dassh.RoddedRegion
            The RoddedRegion object to test
        """
        tmp_asm = simple_ctrl_rr_ent.clone()

        power = mock_AssemblyPower(simple_ctrl_rr_ent)
        dz, _ = dassh.region_rodded.calculate_min_dz(tmp_asm, rr_data.inlet_temp,
                                                      rr_data.outlet_temp)
        ans = dz * tmp_asm._calc_int_sc_power(power['pins'], power['cool'])
        # Calculate new temperatures
        tmp_asm._update_subchannels_properties(tmp_asm.temp['coolant_int'])
        tmp_asm._calc_coolant_int_temp(dz, power['pins'], power['cool'])
        dT = tmp_asm.temp['coolant_int'] - rr_data.inlet_temp
        # Calculate Q = mCdT in each channel
        Q = tmp_asm.sc_properties['heat_capacity'] * tmp_asm.sc_mfr * dT        
        
        print('dz (m): ' + str(dz))
        print('Power added (W): ' + str(ans))
        print('Mdot (kg/s): ' + str(tmp_asm.int_flow_rate))
        print('Cp (J/kgK): ' + str(tmp_asm.coolant.heat_capacity))
        print('Power result (W): ' + str(Q))
        assert np.allclose(ans, Q, atol=rr_data.enthalpy['tol'])
        
    def test_avg_coolant_int_temp(self, simple_ctrl_rr_ent: dassh.RoddedRegion):
        """
        Test that the average coolant temperature is calculated correctly
        
        Parameters
        ----------
        simple_ctrl_rr_ent : dassh.RoddedRegion
            The RoddedRegion object to test
        """
        tmp_asm = simple_ctrl_rr_ent.clone()

        power = {
            'pins': np.zeros(tmp_asm.n_pin),
            'cool': rr_data.enthalpy['linear_cool_power'] * np.ones(tmp_asm.subchannel.n_sc['coolant']['total'])
        }
        
        tmp_asm._update_subchannels_properties(tmp_asm.temp['coolant_int'])
        tmp_asm._calc_coolant_int_temp(rr_data.enthalpy['dz'], power['pins'], power['cool'])
        print(tmp_asm.sc_mfr, tmp_asm.sc_properties['heat_capacity'])
        assert tmp_asm.avg_coolant_int_temp == \
            pytest.approx(rr_data.inlet_temp + rr_data.enthalpy['dT'], 
                          abs=rr_data.enthalpy['tol'])
        
        
        