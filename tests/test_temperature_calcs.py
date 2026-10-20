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
date: 2022-12-06
author: matz
Test the DASSH Assembly object
"""
########################################################################
import os
import numpy as np
import copy
import pytest
import dassh

# Constants
INTERNAL_IND: list[int] = [0, 2, 19]
"""Indices of inter-assembly gap subchannels between the sides of two 
assemblies in a 3-assembly core."""
CORNER_IND: list[int] = [3, 11, 20]
"""Indices of inter-assembly gap subchannels between the corners of two
assemblies in a 3-assembly core."""
CENTRAL_IND: int = 1
"""Index of inter-assembly gap subchannel between three assemblies in a
3-assembly core."""
EXTERNAL_IND: list[list[int]] = [
    [4, 5, 6, 7, 8, 9, 10], 
    [12, 13, 14, 15, 16, 17, 18], 
    [24, 25, 26, 27, 21, 22, 23]
]
"""Indices of inter-assembly gap subchannels that are close to only one 
assembly in a 3-assembly core. Each sublist corresponds to one assembly."""
DELTAZ = 1e-3
"""Axial step size for calculating inter-assembly gap temperatures in tests."""
DELTA_T_DUCT = 10.0
"""Temperature difference for duct wall temperatures in tests."""
HIGH_MFR = 100.0
"""High mass flow rate for testing inter-assembly gap flow models in tests."""
ABSTOL = 1e-2
"""Absolute tolerance for comparing inter-assembly gap temperatures in tests."""


# Use "print_option" to print temperatures and parameters
# for use in Excel spreadsheet for verification
print_option = False


def test_int_coolant_verification(simple_asm):
    """Test that the method to calculate interior and bypass coolant
    temperatures performs as expected"""
    # Set up some stuff
    inlet_temp = 623.15
    gap_t = np.ones(simple_asm.rodded.subchannel.n_sc['duct']['total'])
    gap_t *= inlet_temp
    simple_asm.rodded._update_coolant_int_params(inlet_temp)
    # gap_htc = np.ones(simple_asm.rodded.subchannel.n_sc['duct']['total'])
    # gap_htc *= simple_asm.rodded.coolant_int_params['htc'][1:]
    gap_htc = simple_asm.rodded.coolant_int_params['htc'][1:]
    gap_htc = gap_htc[
        simple_asm.rodded.subchannel.type[
            simple_asm.rodded.subchannel.n_sc['coolant']['interior']:
            simple_asm.rodded.subchannel.n_sc['coolant']['total']] - 1]

    ans = np.array([
        6.2361894147E+02, 6.2357888738E+02, 6.2361746368E+02,
        6.2370276753E+02, 6.2374961923E+02, 6.2370448287E+02,
        6.2358481199E+02, 6.2412338768E+02, 6.2351685219E+02,
        6.2408479054E+02, 6.2356204267E+02, 6.2434522339E+02,
        6.2368527386E+02, 6.2469859277E+02, 6.2376607109E+02,
        6.2474451781E+02, 6.2371183773E+02, 6.2443032335E+02
    ])

    z = 1.29
    for i in range(200):
        # Calculate coolant and duct temperatures at the current level
        simple_asm.calculate(DELTAZ, gap_t, gap_htc, z=z)

        # Collect data to print for verification if test is not passed
        z_power = simple_asm.power.get_power(z)
        print_list = [z]
        print_list += list(simple_asm.temp_coolant)
        print_list += [simple_asm.active_region.coolant.heat_capacity,
                       simple_asm.active_region.coolant.thermal_conductivity,
                       simple_asm.active_region.coolant.density,
                       simple_asm.active_region.coolant_int_params['fs'][0],
                       simple_asm.active_region.coolant_int_params['fs'][1],
                       simple_asm.active_region.coolant_int_params['fs'][2],
                       simple_asm.active_region.coolant_int_params['htc'][1],
                       simple_asm.active_region.coolant_int_params['htc'][2],
                       simple_asm.active_region.coolant_int_params['eddy'],
                       simple_asm.active_region.coolant_int_params['swirl'][1]]
        print_list += list(z_power['pins'])
        print_list += list(z_power['cool'])
        print_list += list(simple_asm.temp_duct_surf[0, 0])
        if print_option:
            print(' '.join(['{:.10e}'.format(v) for v in print_list]))
        z += DELTAZ

    # print(simple_asm.temp_coolant - ans)
    assert np.allclose(simple_asm.temp_coolant, ans)


def test_pin_only_int_coolant_verification(testdir):
    """Test that the method to calculate interior coolant temperatures
    performs as expected in the simplest case: adiabatic duct wall,
    power delivered only to pins"""
    rpath = os.path.join(testdir,
                         'test_results',
                         'conservation-1',
                         'dassh_reactor.pkl')
    if os.path.exists(rpath):
        r = dassh.reactor.load(rpath)
    else:
        pytest.skip('Cannot load necessary reactor object')

    T_ans = copy.deepcopy(r.assemblies[0].rodded.temp['coolant_int'])
    dT_ans = T_ans - r.inlet_temp
    r.reset()
    asm = r.assemblies[0].clone(new_loc=(0, 0))
    asm.rodded._init_static_correlated_params(623.15)
    assert np.all(asm.rodded.temp['coolant_int'] == 623.15)
    assert np.all(asm.rodded.temp['duct_mw'] == 623.15)
    assert np.all(asm.rodded.temp['duct_surf'] == 623.15)
    assert asm._z == 0.0
    assert asm.power._step == 0.0
    # Set up some stuff
    inlet_temp = 623.15
    gap_t = np.ones(asm.rodded.subchannel.n_sc['duct']['total'])
    gap_t *= inlet_temp
    asm.rodded._update_coolant_int_params(inlet_temp)
    gap_htc = asm.rodded.coolant_int_params['htc'][1:]
    gap_htc = gap_htc[
        asm.rodded.subchannel.type[
            asm.rodded.subchannel.n_sc['coolant']['interior']:
            asm.rodded.subchannel.n_sc['coolant']['total']] - 1]

    ans = np.array([8.013890951216E+02, 8.013890951216E+02,
                    8.013890951216E+02, 8.013890951216E+02,
                    8.013890951216E+02, 8.013890951216E+02,
                    7.828783009315E+02, 7.909220491584E+02,
                    7.829210893224E+02, 7.828783009315E+02,
                    7.909220491584E+02, 7.829210893224E+02,
                    7.828783009315E+02, 7.909220491584E+02,
                    7.829210893224E+02, 7.828783009315E+02,
                    7.909220491584E+02, 7.829210893224E+02,
                    7.828783009315E+02, 7.909220491584E+02,
                    7.829210893224E+02, 7.828783009315E+02,
                    7.909220491584E+02, 7.829210893224E+02,
                    7.732208818580E+02, 7.733680511105E+02,
                    7.730334575518E+02, 7.732208818580E+02,
                    7.733680511105E+02, 7.730334575518E+02,
                    7.732208818580E+02, 7.733680511105E+02,
                    7.730334575518E+02, 7.732208818580E+02,
                    7.733680511105E+02, 7.730334575518E+02,
                    7.732208818580E+02, 7.733680511105E+02,
                    7.730334575518E+02, 7.732208818580E+02,
                    7.733680511105E+02, 7.730334575518E+02])

    for i in range(len(r.dz)):
        z = r.z[i + 1]
        dz = r.dz[i]
        # Calculate coolant and duct temperatures at the current level
        asm.calculate(dz, gap_t, gap_htc, adiabatic=True)

        # Collect data to print for verification if test is not passed
        z_power = asm.power.get_power(z - 0.5 * dz)
        z_power['pins'] *= asm.power._renorm[asm.power._kfint[i]]
        print_list = [z]
        print_list += list(asm.temp_coolant)
        print_list += [z_power['pins'][0]]
        print_list += list(asm.temp_duct_surf[0, 0])
        if print_option:
            print(' '.join(['{:.12e}'.format(v) for v in print_list]))

    # assert 0
    dT_ss = ans - r.inlet_temp
    dT_res = asm.rodded.temp['coolant_int'] - r.inlet_temp
    # print(dT_res - dT_ss)
    # print(dT_res - dT_ans)
    assert np.allclose(dT_res, dT_ans)
    assert np.allclose(dT_res, dT_ss)
    assert np.allclose(asm.rodded.temp['coolant_int'], ans)


def test_duct_verification(simple_asm):
    """Test that the method to calculate interior and bypass coolant
    temperatures performs as expected"""
    # Set up some stuff
    inlet_temp = 623.15
    gap_t = np.ones(simple_asm.rodded.subchannel.n_sc['duct']['total'])
    gap_t *= inlet_temp
    simple_asm.rodded._update_coolant_int_params(inlet_temp)
    gap_htc = simple_asm.rodded.coolant_int_params['htc'][1:]
    ans = {
        's_in': np.array([
            6.2331319134E+02, 6.2366394109E+02, 6.2328907297E+02, 
            6.2365345876E+02, 6.2330724534E+02, 6.2380963397E+02, 
            6.2335834747E+02, 6.2401040966E+02, 6.2338832871E+02, 
            6.2402299513E+02, 6.2336542713E+02, 6.2383288702E+02]),
        'mw': np.array([
            6.2327798834E+02, 6.2346662470E+02, 6.2326345231E+02, 
            6.2346117940E+02, 6.2327484122E+02, 6.2356840786E+02, 
            6.2332709336E+02, 6.2371667394E+02, 6.2334993380E+02, 
            6.2372321177E+02, 6.2333084051E+02, 6.2358048722E+02]),
        's_out': np.array([
            6.2316828467E+02, 6.2317719276E+02, 6.2316650079E+02, 
            6.2317678449E+02, 6.2316793643E+02, 6.2318663956E+02, 
            6.2317621504E+02, 6.2320081444E+02, 6.2317943306E+02, 
            6.2320130462E+02, 6.2317662969E+02, 6.2318754524E+02])
    }
    simple_asm._z = 1.29
    z = 1.29
    for i in range(100):
        htc1 = simple_asm.active_region.coolant_int_params['htc'][1]
        htc2 = simple_asm.active_region.coolant_int_params['htc'][2]
        start = simple_asm.active_region.subchannel.n_sc['coolant']['interior']
        coolant_temps = list(simple_asm.temp_coolant[start:])
        simple_asm.calculate(DELTAZ, gap_t, gap_htc, z=z)

        # Collect data to print for verification if test is not passed
        z_power = simple_asm.power.get_power(z - DELTAZ * 0.5)
        print_list = [z]
        print_list += list(simple_asm.temp_duct_surf[0, 0])
        print_list += list(simple_asm.temp_duct_mw[0])
        print_list += list(simple_asm.temp_duct_surf[0, 1])
        print_list += [htc1, htc2]
        print_list.append(simple_asm.active_region.duct.thermal_conductivity)
        print_list += list(z_power['duct'])
        print_list += coolant_temps
        if print_option:
            print(' '.join(['{:.10e}'.format(v) for v in print_list]))

        z += DELTAZ

    assert np.allclose(ans['s_in'], simple_asm.temp_duct_surf[0, 0])
    assert np.allclose(ans['mw'], simple_asm.temp_duct_mw[0])
    assert np.allclose(ans['s_out'], simple_asm.temp_duct_surf[0, 1])


def test_bypass_gap_verification(simple_ctrl_asm):
    """Test that the method to calculate interior and bypass coolant
    temperatures performs as expected"""
    # Set up some stuff
    inlet_temp = 623.15
    gap_t = np.ones(simple_ctrl_asm.rodded.subchannel.n_sc['duct']['total'])
    gap_t *= inlet_temp
    simple_ctrl_asm.rodded._update_coolant_int_params(inlet_temp)
    simple_ctrl_asm.rodded._update_coolant_byp_params([inlet_temp])
    gap_htc = simple_ctrl_asm.rodded.coolant_byp_params['htc'][0]
    if print_option:
        keys = ['d_bypass', 'Flow area (edge)', 'Flow area (corner)',
                'Flow area (total)', 'WP in (edge)', 'WP out (edge)',
                'WP in (corner)', 'WP out (corner)', 'L67']
        vals = [simple_ctrl_asm.rodded.d['bypass'][0],
                simple_ctrl_asm.rodded.bypass_params['area'][0, 0],
                simple_ctrl_asm.rodded.bypass_params['area'][0, 1],
                simple_ctrl_asm.rodded.bypass_params['total area'][0],
                simple_ctrl_asm.rodded.L[5][5][0],
                simple_ctrl_asm.rodded.L[5][5][0],
                2 * simple_ctrl_asm.rodded.d['wcorner'][0, 1],
                2 * simple_ctrl_asm.rodded.d['wcorner'][1, 1],
                simple_ctrl_asm.rodded.L[5][6][0]]
        for i in range(len(keys)):
            print(keys[i] + ': ' + '{:.15e}'.format(vals[i]))

    ans = np.array([6.2343766926209E+02, 6.2361524214799E+02,
                    6.2343766744661E+02, 6.2361524220506E+02,
                    6.2343766926209E+02, 6.2361524214799E+02,
                    6.2343766744661E+02, 6.2361524220506E+02,
                    6.2343766926209E+02, 6.2361524214799E+02,
                    6.2343766744661E+02, 6.2361524220506E+02])
    simple_ctrl_asm._z = 1.29
    z = 1.29
    for i in range(100):
        simple_ctrl_asm.calculate(DELTAZ, gap_t, gap_htc, z=z)

        # Print things to see what's going on in the bypass gap
        print_list = [z]
        print_list += list(simple_ctrl_asm.temp_bypass[0])
        print_list += \
            [simple_ctrl_asm.active_region.coolant.heat_capacity,
             simple_ctrl_asm.active_region.coolant.thermal_conductivity,
             simple_ctrl_asm.active_region.byp_flow_rate[0],
             simple_ctrl_asm.active_region.coolant_byp_params['htc'][0, 0],
             simple_ctrl_asm.active_region.coolant_byp_params['htc'][0, 1]]
        print_list += list(simple_ctrl_asm.temp_duct_surf[0, 1])
        print_list += list(simple_ctrl_asm.temp_duct_surf[1, 0])

        if print_option:
            print(' '.join(['{:.12e}'.format(v) for v in print_list]))
        z += DELTAZ

    assert np.allclose(ans, simple_ctrl_asm.temp_bypass[0])


def test_interasm_gap_flow_model_verification(three_asm_core):
    """Test that the method to calculate inter-assembly gap coolant
    temperatures (flowing gap model) performs as expected"""
    # Set up some stuff
    asm_list, core_obj = three_asm_core
    inlet_temp = 623.15
    n_zpts = 20
    r = np.random.RandomState(seed=42)  # set for reproducibility

    ans = np.array([6.4430257093E+02, 6.4133750612E+02,
                    6.4464137575E+02, 6.4204271522E+02,
                    6.4378541819E+02, 6.4345130150E+02,
                    6.4357382799E+02, 6.4400775157E+02,
                    6.4345297922E+02, 6.4337305191E+02,
                    6.4317890562E+02, 6.4074841996E+02,
                    6.3492735383E+02, 6.3384866583E+02,
                    6.3386094328E+02, 6.3326827778E+02,
                    6.3417287934E+02, 6.3349425161E+02,
                    6.3428529964E+02, 6.3624922416E+02,
                    6.3399980086E+02, 6.3267989605E+02,
                    6.3384325575E+02, 6.3344587959E+02,
                    6.3414989863E+02, 6.3341650255E+02,
                    6.3430409517E+02, 6.3349615942E+02])
    # 2021-04-29: ANSWER FOR OLD DASSH GAP MESHING
    # Spreadsheet for this answer still in file
    # ans = np.array([6.45037183080E+02, 6.41980972740E+02,
    #                 6.45358606160E+02, 6.42969608370E+02,
    #                 6.44963090990E+02, 6.39769145330E+02,
    #                 6.44748592010E+02, 6.40273536950E+02,
    #                 6.44595385000E+02, 6.39742980240E+02,
    #                 6.44317033130E+02, 6.41608817810E+02,
    #                 6.35587316080E+02, 6.31907150960E+02,
    #                 6.34474448330E+02, 6.31444749440E+02,
    #                 6.34788828880E+02, 6.31676407570E+02,
    #                 6.34928995790E+02, 6.36683571690E+02,
    #                 6.34538836150E+02, 6.33215033410E+02,
    #                 6.31916035520E+02, 6.34023142120E+02,
    #                 6.34780302460E+02, 6.31590910040E+02,
    #                 6.34916559090E+02, 6.31607081410E+02])
    # Create fake duct wall surface temperature array
    core_obj._update_coolant_gap_params(inlet_temp)
    duct_temps = []
    for asm in asm_list:
        tmp = np.zeros((n_zpts, asm.rodded.subchannel.n_sc['duct']['total']))
        tmp[0] = asm.duct_outer_surf_temp
        duct_temps.append(tmp)

    for zi in range(n_zpts):
        # Assemble information used to calculate gap temperatures
        core_obj._update_coolant_gap_params(core_obj.avg_coolant_gap_temp)
        print_list = [core_obj.gap_coolant.heat_capacity,
                      core_obj.gap_coolant.thermal_conductivity,
                      core_obj.gap_flow_rate,
                      core_obj.coolant_gap_params['htc'][0],
                      core_obj.coolant_gap_params['htc'][1],
                      core_obj.coolant_gap_params['htc'][5]]
        print_list += list(duct_temps[0][zi])
        print_list += list(duct_temps[1][zi])
        print_list += list(duct_temps[2][zi])

        # Calculate and print gap subchannel temperatures
        tduct = np.array([d[zi] for d in duct_temps])
        core_obj.calculate_gap_temperatures(DELTAZ, tduct)
        print_list = [DELTAZ * zi] + list(core_obj.coolant_gap_temp) + print_list
        if print_option:
            print(' '.join(['{:.10e}'.format(v) for v in print_list]))

        # Update duct temperatures
        if zi + 1 < n_zpts:
            for d in duct_temps:
                dT = r.rand(len(d[zi])) + 0.5  # range = [0.5, 1.5]
                d[zi + 1] = d[zi] + dT
            duct_temps[0][zi + 1] += 1.0

    assert np.allclose(ans, core_obj.coolant_gap_temp)


def test_interasm_gap_noflow_model_verification(three_asm_core):
    """Test no-flow (conduction) model for inter-assembly gap coolant"""
    asm_list, core_obj = three_asm_core
    core_obj.model = 'no_flow'
    core_obj.gap_flow_rate = 0.0
    core_obj.load(asm_list)

    # Set up some stuff
    inlet_temp = 623.15
    n_zpts = 20
    r = np.random.RandomState(seed=42)  # set for reproducibility
    ans = np.array([
        6.5150426488E+02, 6.4752164488E+02, 6.5138793687E+02,
        6.5338567527E+02, 6.6059116544E+02, 6.6205940071E+02,
        6.6076180758E+02, 6.6202876454E+02, 6.5974523677E+02,
        6.6106137785E+02, 6.5953737318E+02, 6.5148242142E+02,
        6.4385701672E+02, 6.4355821554E+02, 6.4283016740E+02,
        6.4219931914E+02, 6.4205760814E+02, 6.4148057698E+02,
        6.4369322289E+02, 6.4000978899E+02, 6.4087071644E+02,
        6.3993620392E+02, 6.4299347495E+02, 6.4133422169E+02,
        6.4319991462E+02, 6.4239624871E+02, 6.4249846862E+02,
        6.4345278609E+02])

    # Create fake duct wall surface temperature array
    core_obj._update_coolant_gap_params(inlet_temp)
    duct_temps = []
    for asm in asm_list:
        tmp = np.zeros((n_zpts, asm.rodded.subchannel.n_sc['duct']['total']))
        tmp[0] = asm.duct_outer_surf_temp
        duct_temps.append(tmp)

    for zi in range(n_zpts):
        # Assemble information used to calculate gap temperatures
        core_obj._update_coolant_gap_params(core_obj.avg_coolant_gap_temp)
        print_list = [core_obj.gap_coolant.thermal_conductivity]
        print_list += list(duct_temps[0][zi])
        print_list += list(duct_temps[1][zi])
        print_list += list(duct_temps[2][zi])

        # Calculate and print gap subchannel temperatures
        tduct = np.array([d[zi] for d in duct_temps])
        core_obj.calculate_gap_temperatures(DELTAZ, tduct)
        print_list = [DELTAZ * zi] + list(core_obj.coolant_gap_temp) + print_list
        if print_option:
            print(' '.join(['{:.10e}'.format(v) for v in print_list]))

        # Update duct temperatures
        if zi + 1 < n_zpts:
            for d in duct_temps:
                dT = r.rand(len(d[zi])) + 0.5  # [0.5, 1.5]
                d[zi + 1] = d[zi] + dT
            duct_temps[0][zi + 1] += 1.0

    assert np.allclose(ans, core_obj.coolant_gap_temp)


def test_interasm_gap_ductavg_model_verification(three_asm_core):
    """Test no-flow (duct-avg) model for inter-assembly gap coolant"""
    asm_list, core_obj = three_asm_core
    core_obj.model = 'duct_average'
    core_obj.ia_obj._model = 'duct_average'
    core_obj.gap_flow_rate = 0.0

    # Set up some stuff
    inlet_temp = 623.15
    n_zpts = 20
    r = np.random.RandomState(seed=42)  # set for reproducibility
    ans = np.array([651.511243165, 647.525353140, 651.392401735,
                    653.399834485, 660.609908130, 662.082117640,
                    660.764436810, 662.053100090, 659.745021560,
                    661.084043170, 659.557681360, 651.491705490,
                    643.869481620, 643.566222600, 642.834433430,
                    642.203403990, 642.061278200, 641.475761430,
                    643.682222870, 640.004054625, 640.870416540,
                    639.927400840, 643.014265190, 641.336761630,
                    643.189862100, 642.399881550, 642.500626920,
                    643.471961030])

    # Create fake duct wall surface temperature array
    core_obj._update_coolant_gap_params(inlet_temp)
    duct_temps = []
    for asm in asm_list:
        tmp = np.zeros((n_zpts, asm.rodded.subchannel.n_sc['duct']['total']))
        tmp[0] = asm.duct_outer_surf_temp
        duct_temps.append(tmp)

    for zi in range(n_zpts):
        # Assemble information used to calculate gap temperatures
        core_obj._update_coolant_gap_params(core_obj.avg_coolant_gap_temp)
        print_list = list(duct_temps[0][zi])
        print_list += list(duct_temps[1][zi])
        print_list += list(duct_temps[2][zi])

        # Calculate and print gap subchannel temperatures
        tduct = np.array([d[zi] for d in duct_temps])
        core_obj.calculate_gap_temperatures(DELTAZ, tduct)
        print_list = [DELTAZ * zi] + list(core_obj.coolant_gap_temp) + print_list
        if print_option:
            print(' '.join(['{:.10e}'.format(v) for v in print_list]))

        # Update duct temperatures
        if zi + 1 < n_zpts:
            for d in duct_temps:
                dT = r.rand(len(d[zi])) + 0.5  # [0.5, 1.5]
                d[zi + 1] = d[zi] + dT
            duct_temps[0][zi + 1] += 1.0

    assert np.allclose(ans, core_obj.coolant_gap_temp)


def print_bypass_gap_energy_cons_verification(simple_ctrl_asm_pins_cmat):
    """Test that method to calculate bypass coolant conserves energy;
    not actually a test, only run to print data"""
    # Notes:
    # - Power to pins only (no duct wall heating)
    # - Interior coolant temperature taken as given: will use to get
    #   inner duct temperatures and then bypass gap temperatures
    # - Will be calculating heat lost from interior coolant to duct
    #   and transferred from inner duct to bypass coolant
    # - Assuming adiabatic outer duct: no heat transfer there.

    # Set up some stuff
    asm = simple_ctrl_asm_pins_cmat
    inlet_temp = 623.15
    gap_t = np.ones(asm.rodded.subchannel.n_sc['duct']['total'])
    gap_t *= inlet_temp
    asm.rodded._update_coolant_int_params(inlet_temp)
    asm.rodded._update_coolant_byp_params([inlet_temp])
    gap_htc = asm.rodded.coolant_byp_params['htc'][0]

    # Print all the parameters we'll need
    asm_params = {
        'flow_rate': asm.flow_rate,
        'pin_pitch': asm.rodded.pin_pitch,
        'dwc00': asm.rodded.d['wcorner'][0, 0],
        'dwc01': asm.rodded.d['wcorner'][0, 1],
        'dwc10': asm.rodded.d['wcorner'][1, 0],
        'dwc11': asm.rodded.d['wcorner'][1, 1],
        'dthickness0': asm.rodded.duct_params['thickness'][0],
        'byp_thickness': asm.rodded.d['bypass'][0],
        'a_byp_edge': asm.rodded.bypass_params['area'][0, 0],
        'a_byp_corn': asm.rodded.bypass_params['area'][0, 1],
        'htc_int_edge': asm.rodded.coolant_int_params['htc'][1],
        'htc_int_corn': asm.rodded.coolant_int_params['htc'][2],
        'htc_byp_edge': asm.rodded.coolant_byp_params['htc'][0, 0],
        'htc_byp_corn': asm.rodded.coolant_byp_params['htc'][0, 1],
        'coolant_cp': asm.rodded.coolant.heat_capacity,
        'coolant_k': asm.rodded.coolant.thermal_conductivity,
        'duct_k': asm.rodded.duct.thermal_conductivity,
        'int_fr': asm.rodded.int_flow_rate,
        'byp_fr': asm.rodded.byp_flow_rate[0]
    }
    if print_option:
        for k in asm_params.keys():
            print(k, asm_params[k])
    asm._z = 1.29
    z = 1.29
    for i in range(20):
        asm.calculate(z, DELTAZ, gap_t, gap_htc, adiabatic=True, ebal=True)

        # Collect data to print for verification if test is not passed
        print_list = [z]
        # Interior edge/corner coolant channels
        start = asm.active_region.subchannel.n_sc['coolant']['interior']
        print_list += list(asm.temp_coolant[start:])
        # Duct wall inner surface, midwall, and outer surface temps
        print_list += list(asm.temp_duct_surf[0, 0])
        print_list += list(asm.temp_duct_mw[0])
        print_list += list(asm.temp_duct_surf[0, 1])
        # Coolant bypass temperatures
        print_list += list(asm.temp_bypass[0])
        # Ignore outer duct: adiabatic boundary so it should be
        # effectively the same temp as the bypass coolant.
        if print_option:
            print(' '.join(['{:.10e}'.format(v) for v in print_list]))

        z += DELTAZ
    if print_option:
        assert 0


def _setup_mixed_flow(three_asm_core: tuple[list[dassh.Assembly], 
                                            dassh.Core]) -> tuple[dassh.Core, 
                                                                  np.ndarray]:
    """
    Setup function for the `mixed_flow` model tests
    
    Parameters
    ----------
    three_asm_core : tuple[list[dassh.Assembly], dassh.Core]
        A tuple containing a list of three assemblies and a core object.
        The assemblies are instances of the `Assembly` class, and the core
        object is an instance of the `Core` class.
        
    Returns
    -------
    tuple[dassh.Core, np.ndarray]
        A tuple containing the core object and an array of duct wall
        temperatures for the three assemblies.
    """
    asm_list, core_obj = three_asm_core
    core_obj.model = 'mixed_flow'
    duct_temps = np.array([asm.duct_outer_surf_temp for asm in asm_list])
    return core_obj, duct_temps
    

def test_mixed_flow_zero_flux(three_asm_core: tuple[list[dassh.Assembly],
                                                    dassh.Core]) -> None:
    """
    Test that the `mixed_flow` model for inter-assembly gap coolant
    temperatures does not produce temperature increases in the gap
    when the duct wall temperatures are equal to the inlet coolant temperature
    
    Parameters
    ----------
    three_asm_core : tuple[list[dassh.Assembly], dassh.Core]
        A tuple containing a list of three assemblies and a core object.
        The assemblies are instances of the `Assembly` class, and the core
        object is an instance of the `Core` class.
    """
    core_obj, duct_temps = _setup_mixed_flow(three_asm_core)
    # Copy the initial coolant gap temperatures to compare after calculation
    ans = core_obj.coolant_gap_temp.copy()
    # Calculate gap temperatures with duct wall temperatures
    core_obj.calculate_gap_temperatures(DELTAZ, duct_temps)
    assert np.allclose(core_obj.coolant_gap_temp, ans)    
    

def test_mixed_flow_symmetric(three_asm_core: tuple[list[dassh.Assembly], 
                                                    dassh.Core]) -> None:
    """
    Test that the `mixed_flow` model for inter-assembly gap coolant
    temperatures produces symmetric temperature distributions in the gap
    when the duct wall temperatures are symmetric.
    
    Parameters
    ----------
    three_asm_core : tuple[list[dassh.Assembly], dassh.Core]
        A tuple containing a list of three assemblies and a core object.
        The assemblies are instances of the `Assembly` class, and the core
        object is an instance of the `Core` class.
    """
    core_obj, duct_temps = _setup_mixed_flow(three_asm_core)
    duct_temps += DELTA_T_DUCT # increase all duct wall temperatures by 10 K 
    core_obj.calculate_gap_temperatures(DELTAZ, duct_temps)
    # SCs between two assembly sides
    assert np.all(core_obj.coolant_gap_temp[INTERNAL_IND] == 
                  core_obj.coolant_gap_temp[INTERNAL_IND[0]])
    # SCs between two assembly corners
    assert np.all(core_obj.coolant_gap_temp[CORNER_IND] == 
                  core_obj.coolant_gap_temp[CORNER_IND[0]])
    # External SCs
    assert np.all(core_obj.coolant_gap_temp[EXTERNAL_IND[0]] == 
                  core_obj.coolant_gap_temp[EXTERNAL_IND[1]])
    assert np.all(core_obj.coolant_gap_temp[EXTERNAL_IND[2]] == 
                  core_obj.coolant_gap_temp[EXTERNAL_IND[1]])
    
    
def test_mixed_flow_asymmetric(three_asm_core: tuple[list[dassh.Assembly], 
                                                     dassh.Core]) -> None:
    """
    Test that the `mixed_flow` model for inter-assembly gap coolant
    temperatures produces larger temperature increase in the gap adjacent to 
    higher duct wall temperature. 
    Additionally, test energy exchange between higher temperature subchannel
    and its neighbors.
    
    Parameters
    ----------
    three_asm_core : tuple[list[dassh.Assembly], dassh.Core]
        A tuple containing a list of three assemblies and a core object.
        The assemblies are instances of the `Assembly` class, and the core
        object is an instance of the `Core` class.
        
    Notes
    -----
    This test is conducted in two steps:
    1. One side of the duct wall temperature of the first assembly is 
       increased by 10 K, while the other duct wall temperatures remain at 
       the inlet coolant temperature. When the `calculate_gap_temperatures` 
       method is called, only the temperature of the subchannel adjacent to the
       higher duct wall temperature changes. The other subchannels remain at 
       the inlet coolant temperature because inter-subchannel heat transfer 
       is based on the previous axial step.
       Test: The maximum temperature in the gap is in the subchannel adjacent 
       to the higher duct wall temperature.
    2. The `calculate_gap_temperatures` method is called again with the same
       duct wall temperatures. This time, the subchannel adjacent to the higher
       duct wall temperature exchanges heat with its neighbors. 
       Test: The maximum temperature in the gap is still in the subchannel 
       adjacent to the higher duct wall temperature. Temperatures of the 
       neighboring subchannels are higher than the inlet coolant temperature, 
       but lower than the subchannel adjacent to the higher duct wall 
       temperature.
    """
    core_obj, duct_temps = _setup_mixed_flow(three_asm_core)
    duct_temps[0][0] += DELTA_T_DUCT
    t_in = core_obj.coolant_gap_temp[0]
    core_obj.calculate_gap_temperatures(DELTAZ, duct_temps)
    # Maximum temperature should be in the gap adjacent to the higher duct wall 
    # temperature
    assert np.all(core_obj.coolant_gap_temp <= 
                  core_obj.coolant_gap_temp[INTERNAL_IND[0]])
    # Call the method again to allow heat exchange between subchannels
    core_obj.calculate_gap_temperatures(DELTAZ, duct_temps)
    # Maximum temperature should still be in the gap adjacent to the 
    # higher duct wall temperature
    assert np.all(core_obj.coolant_gap_temp <= 
                  core_obj.coolant_gap_temp[INTERNAL_IND[0]])
    # Check that the neighboring subchannels have higher temperatures than the
    # inlet coolant temperature, but lower than the subchannel adjacent to the
    # higher duct wall temperature
    assert np.all(core_obj.coolant_gap_temp[[CENTRAL_IND, CORNER_IND[1]]] < 
                  core_obj.coolant_gap_temp[INTERNAL_IND[0]])
    assert np.all(core_obj.coolant_gap_temp[[CENTRAL_IND, CORNER_IND[1]]] >
                  t_in)  


def test_mixed_flow_high_mfr(three_asm_core: tuple[list[dassh.Assembly], 
                                                   dassh.Core]) -> None:
    """Test that the `mixed_flow` model for inter-assembly gap coolant
    temperatures produces a temperature distribution that is close to the
    one predicted by the `flow` model when the mass flow rate in the gap is 
    high
    
    Parameters
    ----------
    three_asm_core : tuple[list[dassh.Assembly], dassh.Core]
        A tuple containing a list of three assemblies and a core object.
        The assemblies are instances of the `Assembly` class, and the core
        object is an instance of the `Core` class.
    """
    def run_model():
        """Internal function to run the `calculate_gap_temperatures` method 
        twice and return the resulting coolant gap temperatures"""
        core_obj.calculate_gap_temperatures(DELTAZ, duct_temps)
        core_obj.calculate_gap_temperatures(DELTAZ, duct_temps)
        return core_obj.coolant_gap_temp.copy()
    
    core_obj, duct_temps = _setup_mixed_flow(three_asm_core)
    core_obj.gap_flow_rate = HIGH_MFR
    t_in = core_obj.coolant_gap_temp.copy()
    # Add some random variation to the duct wall temperatures to simulate a 
    # non-uniform scenario
    duct_temps += np.random.rand(*duct_temps.shape) * DELTA_T_DUCT
    
    gap_temps_mixed_flow = run_model()
    # reset temperature to inlet temperature
    core_obj.coolant_gap_temp = t_in
    core_obj.model = 'flow'
    gap_temps_flow = run_model()
    assert np.allclose(gap_temps_mixed_flow, gap_temps_flow, atol=ABSTOL)
    

# def test_porous_media_method(simple_asm, conceptual_core):
#     """Test that the method to calculate interior and bypass coolant
#     temperatures performs as expected"""
#     # Set up some stuff
#     inlet_temp = 623.15
#     z = 0.0
#     # z_end = 1.281
#
#     from dassh.correlations import nusselt_db
#     conceptual_core.gap_coolant.update(inlet_temp)
#     Nu = nusselt_db.calculate_interasm_gap_sc_Nu(conceptual_core)
#     gap_htc = np.ones(2) * (conceptual_core.gap_coolant.thermal_conductivity
#                             * Nu / conceptual_core.gap_params['de'])
#     # print(gap_htc)
#     dz = dassh.axial_constraint.calculate_asm_min_dz(simple_asm,
#                                                          inlet_temp,
#                                                          inlet_temp + 150.0)
#     gap_temps = (np.ones(simple_asm.subchannel.n_sc['duct']['total'])
#                  * simple_asm.avg_coolant_int_temp)
#     simple_asm.update_coolant_int_params(inlet_temp)
#     simple_asm.duct.update(inlet_temp)
#     while z < 3.862:
#         simple_asm.calculate_temperatures(z, dz, gap_temps, gap_htc)
#         z_power = simple_asm.power.get_power(z)
#         print_list = [z,
#                       simple_asm.coolant.heat_capacity,
#                       simple_asm.coolant.thermal_conductivity,
#                       simple_asm.coolant.density,
#                       simple_asm.duct.thermal_conductivity,
#                       simple_asm.duct.heat_capacity,
#                       gap_htc[0],
#                       simple_asm.porous_media['area'] * dz,
#                       simple_asm.porous_media['R'],
#                       simple_asm.avg_coolant_int_temp_j,
#                       simple_asm.avg_duct_mw_temp_j[0]]
#         if z < 1.281 or z > 2.1233:
#             print_list.append(z_power['refl'])
#         else:
#             print_list.append(np.sum(z_power['pins']))
#             print_list.append(np.sum(z_power['duct']))
#             print_list.append(np.sum(z_power['cool']))
#         print(' '.join(['{:.10e}'.format(v) for v in print_list]))
#         z += dz
#     assert 0
#
#
# def test_calc_pm_temps(simple_asm):
#     # power = 4.0061173027e-04
#     dz = 1.1725749335e-02
#     z = 0.0
#     for i in range(30):
#         # z = 3.4004673071e-01
#         power = simple_asm.power.get_power(z)['refl']
#         porosity = 0.25
#         temp_gap = 623.15
#         htc_gap = [2.5e4]
#         dT = simple_asm.calculate_porous_media_temps(dz, power, porosity,
#                                                      temp_gap, htc_gap)
#         print(dT)
#         z += dz
#     assert 0
