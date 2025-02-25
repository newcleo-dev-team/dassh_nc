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
date: 2023-02-28
author: matz
Test the correlations
"""
########################################################################
import os
import pandas as pd
import numpy as np
import pytest
from tests.conftest import activate_rodded_region
import dassh
from dassh.correlations import (friction_ctd,
                                friction_cts,
                                friction_nov,
                                friction_reh,
                                friction_uctd)
from dassh.correlations import (flowsplit_mit,
                                flowsplit_nov,
                                flowsplit_ctd,
                                flowsplit_uctd)
from dassh.correlations import (mixing_ctd,
                                mixing_mit,
                                mixing_kc)
from dassh.correlations import grid_rehme, grid_cdd
from pytest import mat_data

def make_assembly(n_ring, pin_pitch, pin_diameter, clad_thickness,
                  wire_pitch, wire_diameter, duct_ftf, coolant_obj,
                  duct_obj, inlet_temp, inlet_flow_rate,
                  corr_friction='CTD', corr_flowsplit='CTD',
                  corr_mixing='CTD', corr_shapefactor=None,
                  se2geo=False):
    m = {'coolant': coolant_obj, 'duct': duct_obj}
    rr = dassh.RoddedRegion('fuel', n_ring, pin_pitch, pin_diameter,
                            wire_pitch, wire_diameter, clad_thickness,
                            duct_ftf, inlet_flow_rate,
                            m['coolant'], m['duct'],
                            htc_params_duct=None,
                            corr_friction=corr_friction,
                            corr_flowsplit=corr_flowsplit,
                            corr_mixing=corr_mixing,
                            corr_nusselt='DB',
                            corr_shapefactor=corr_shapefactor,
                            se2=se2geo)
    return activate_rodded_region(rr, inlet_temp)

########################################################################
# TEST WARNINGS
########################################################################


def test_correlation_warnings(caplog):
    """Test that attempting to use correlations for assemblies that
    are outside the acceptable ranges raises warnings"""
    # Some standard parameters for assembly instantiation (some will
    # be changed to assess that the proper warnings are raised)
    n_ring = 2
    clad_thickness = 0.5 / 1e3
    wire_diameter = 1.094 / 1e3  # mm -> m
    duct_ftf = [0.11154, 0.11757]  # m
    p2d = 1.6
    h2d = 3.0
    pin_diameter = ((duct_ftf[0] - 2 * wire_diameter)
                    / (np.sqrt(3) * (n_ring - 1) * p2d + 1))
    pin_pitch = pin_diameter * p2d
    wire_pitch = pin_diameter * h2d
    inlet_flow_rate = 30.0  # kg /s
    inlet_temp = 273.15 + 350.0  # K
    coolant_obj = dassh.Material('sodium')
    duct_obj = dassh.Material('ss316')
    make_assembly(n_ring, pin_pitch, pin_diameter, clad_thickness,
                  wire_pitch, wire_diameter, duct_ftf, coolant_obj,
                  duct_obj, inlet_temp, inlet_flow_rate,
                  corr_friction='CTD')

    assert all([param in caplog.text for param in
                ['pin pitch to diameter ratio',
                 'wire-pitch to pin-diameter ratio',
                 'number of rods in bundle']])


########################################################################
# FRICTION FACTOR TESTS
########################################################################


def test_nov_sample_problem():
    """Test the sample problem given in the Novendstern correlation paper;
    parameters are all calculated and shown in the paper, just using them
    to demonstrate that I get the same result with the implemented corr."""

    # Dummy class to mock DASSH Coolant, Subchannel, RegionRodded objects
    class Dummy(object):
        def __init__(self, **kwargs):
            for k in kwargs.keys():
                setattr(self, k, kwargs[k])

    # Dummy Coolant object
    coolant_properties = {
        'viscosity': 0.677 * 0.00041337887,  # lb/hrft --> kg/m-s
        'density': 53.5 * 16.0185}  # lb/ft3 --> kg/m3
    coolant = Dummy(**coolant_properties)

    # Dummy Subchannel object
    subchannel = Dummy(**{'n_sc': {'coolant': {'interior': 384,
                                               'edge': 48,
                                               'corner': 6,
                                               'total': 438}}})

    # Dummy Region object
    fftf = {
        'n_ring': 9,
        'n_pin': 217,
        'duct_ftf': [[4.335 * 2.54 / 100, 4.835 * 2.54 / 100]],
        'pin_diameter': 0.23 * 2.54 / 100,
        'pin_pitch': 0.2879 * 2.54 / 100,
        'wire_diameter': 0.056 * 2.54 / 100,
        'wire_pitch': 12 * 2.54 / 100,
        'coolant': coolant,
        'subchannel': subchannel,
        'params': {'area': np.array([0.0139 * 2.54 * 2.54 / 100 / 100,
                                     0.0278 * 2.54 * 2.54 / 100 / 100,
                                     0.0099 * 2.54 * 2.54 / 100 / 100]),
                   'de': np.array([0.124 * 2.54 / 100,
                                   0.151 * 2.54 / 100,
                                   0.114 * 2.54 / 100])},
        'bundle_params': {'area': 6.724 * 2.54 * 2.54 / 100 / 100,
                          'de': 0.128 * 2.54 / 100},
        'int_flow_rate': 183000 * 0.000125998  # lb/hr --> kg/s
    }
    asm = Dummy(**fftf)

    # Calculate the necessary coolant flow parameters: velocity, Re; then
    # assign to the dummy assembly
    v_tot = asm.int_flow_rate / asm.coolant.density / asm.bundle_params['area']
    Re = (asm.coolant.density
          * v_tot
          * asm.bundle_params['de']
          / asm.coolant.viscosity)
    asm.coolant_int_params = {'Re': Re, 'vel': v_tot}

    # Calculate friction factor, use to determine pressure drop / L
    ff = dassh.correlations.friction_nov.calculate_bundle_friction_factor(asm)
    dp = ff * asm.coolant.density * v_tot**2 / 2 / asm.bundle_params['de']
    ans = 4.64 * 6894.76 / 0.3048
    diff = ans - dp
    rel_diff = diff / ans
    assert rel_diff < 0.002


def test_ctd_intermittency_factor(thesis_asm_rr):
    """Test the calculation of intermittency factor used in the
    Cheng-Todreas friction and flow split correlations in the
    transition regime

    """
    print(dir(thesis_asm_rr))
    test_asm = thesis_asm_rr.clone()
    test_asm._update_coolant_int_params(300.15)
    # Intermittency factor should be greater than 1 if turbulent
    Re_bnds = friction_ctd.calculate_Re_bounds(test_asm)
    assert test_asm.coolant_int_params['Re'] > Re_bnds[1]  # chk trblnce
    x = friction_ctd.calc_intermittency_factor(test_asm, Re_bnds[0],
                                               Re_bnds[1])
    assert x > 1

    # Intermittency factor should be less than 1 if transition
    test_asm.coolant_int_params['Re'] = Re_bnds[1] - 2000.0
    assert test_asm.coolant_int_params['Re'] > Re_bnds[0]  # chk laminar
    x = friction_ctd.calc_intermittency_factor(test_asm, Re_bnds[0],
                                               Re_bnds[1])
    assert 0 < x < 1


def test_ctd_laminar_cfb(testdir):
    """Test the Cheng-Todreas Detailed (1986) correlations against
    data published in the 1986 paper for calculation of the bundle
    average friction factor constant for laminar flow.

    Notes
    -----
    The tolerance for this test is set rather high (6%) because the
    paper does not provide the exact set of parameters required to
    calculated the friction factor constant. The constant is very
    sensitive to the inner duct flat-to-flat distance, but this value
    is not given in the paper. Without knowing the exact value Cheng-
    Todreas used to calculate the results tabulated in their paper,
    this is the closest we can get.

    """
    df = pd.read_csv(os.path.join(testdir, 'test_data',
                                  'cfb_laminar.csv'), header=0)

    # Will only print if the test fails
    print('{:15s} {:>5s} {:>8s} {:>8s} {:>8s}'
          .format('Experiment', 'Year', 'Result', 'CTD', 'Rel err.'))
    abs_err = np.zeros(len(df))
    rel_err = np.zeros(len(df))
    for exp in df.index:
        # Pull assembly geometry information from table
        n_ring = df.loc[exp]['Rings']
        wire_pitch = df.loc[exp]['H']
        pin_diameter = df.loc[exp]['D']
        pin_pitch = pin_diameter * df.loc[exp]['P/D']  # m
        clad_thickness = 0.5 / 1e3  # m
        wire_diameter = df.loc[exp]['D_wire']
        duct_inner_ftf = df.loc[exp]['Duct FTF']
        duct_ftf = [duct_inner_ftf, duct_inner_ftf + 0.001]  # m
        inlet_flow_rate = 0.5  # kg /s
        inlet_temp = 273.15
        coolant_obj = dassh.Material('water', mat_data.water_temperature)
        duct_obj = dassh.Material('ss316')
        try:
            a = make_assembly(n_ring, pin_pitch, pin_diameter,
                              clad_thickness, wire_pitch,
                              wire_diameter, duct_ftf, coolant_obj,
                              duct_obj, inlet_temp, inlet_flow_rate)
        except:
            raise
            pytest.xfail("Failure in DASSH Assembly instantiation "
                         "(should raise another error elsewhere in "
                         "the tests)")
        res = (friction_ctd
               .calculate_bundle_friction_factor_const(a)['laminar'])
        abs_err[exp] = res - df.loc[exp]['CTD']
        rel_err[exp] = abs_err[exp] / df.loc[exp]['CTD']

        # Only print if assertion fails
        print('{:<2d} {:12s} {:5d} {:8.2f} {:8.2f} {:8.2f}'
              .format(exp, df.loc[exp]['Investigators'],
                      df.loc[exp]['Year'], res, df.loc[exp]['CTD'],
                      100 * rel_err[exp]))
    print(max(np.abs(abs_err)))
    print(max(np.abs(rel_err)))
    assert max(np.abs(abs_err)) < 0.35  # I made up this tolerance
    assert max(np.abs(rel_err)) < 0.005


def test_cts_laminar_cfb(testdir):
    """Test the Cheng-Todreas Simple (1986) correlations against
    data published in the 1986 paper for calculation of the bundle
    average friction factor constant for laminar flow.

    Notes
    -----
    Because the simple correlation only requires a few parameters,
    the tolerance for this test (1%) is much lower than that for
    the analogous Cheng-Todreas Detailed test.

    """
    df = pd.read_csv(os.path.join(testdir, 'test_data',
                                  'cfb_laminar.csv'), header=0)

    # Will only print if the test fails
    print('{:15s} {:>5s} {:>8s} {:>8s} {:>8s}'
          .format('Experiment', 'Year', 'Result', 'CTS', 'Rel err.'))
    abs_err = np.zeros(len(df))
    rel_err = np.zeros(len(df))
    for exp in df.index:
        # Pull assembly geometry information from table
        n_ring = df.loc[exp]['Rings']
        wire_pitch = df.loc[exp]['H']
        pin_diameter = df.loc[exp]['D']
        pin_pitch = pin_diameter * df.loc[exp]['P/D']  # m
        clad_thickness = 0.5 / 1e3  # m
        wire_diameter = df.loc[exp]['D_wire']
        duct_inner_ftf = df.loc[exp]['Duct FTF']  # + 0.0001
        duct_ftf = [duct_inner_ftf, duct_inner_ftf + 0.001]  # m
        inlet_flow_rate = 0.5  # kg /s
        inlet_temp = 273.15
        coolant_obj = dassh.Material('water', mat_data.water_temperature)
        duct_obj = dassh.Material('ss316')
        try:
            a = make_assembly(n_ring, pin_pitch, pin_diameter,
                              clad_thickness, wire_pitch,
                              wire_diameter, duct_ftf, coolant_obj,
                              duct_obj, inlet_temp, inlet_flow_rate)
        except:
            pytest.xfail("Failure in DASSH Assembly instantiation "
                         "(should raise another error elsewhere in "
                         "the tests)")
        res = (friction_cts
               .calculate_bundle_friction_factor_const(a)['laminar'])
        abs_err[exp] = res - df.loc[exp]['CTS']
        rel_err[exp] = abs_err[exp] / df.loc[exp]['CTS']

        # Only print if assertion fails
        print('{:<2d} {:12s} {:5d} {:8.2f} {:8.2f} {:8.2f}'
              .format(exp, df.loc[exp]['Investigators'],
                      df.loc[exp]['Year'], res, df.loc[exp]['CTS'],
                      100 * rel_err[exp]))
    print(max(np.abs(abs_err)))
    print(max(np.abs(rel_err)))
    assert max(np.abs(abs_err)) < 0.75
    assert max(np.abs(rel_err)) < 0.01


def test_ctd_turbulent_cfb(testdir):
    """Test the Cheng-Todreas Detailed (1986) correlations against
    data published in the 1986 paper for calculation of the bundle
    average friction factor constant for laminar flow.

    Notes
    -----
    I (M. Atz) could not find any tabulated data, so I visually
    assessed the data by hand from Figure 13 of the Cheng-Todreas
    paper (1986). The pin bundle characteristics for the Marten
    data came from the description in Table 6 of the same paper.
    The characteristics for the Rehme data come from Table 1 of
    Chen et al (2014).

    """
    df = pd.read_csv(os.path.join(testdir, 'test_data',
                                  'cfb_turbulent.csv'), header=0)

    # Will only print if the test fails
    print('{:15s} {:>5s} {:>8s} {:>8s} {:>8s}'
          .format('Experiment', 'Year', 'Result', 'CTD', 'Rel. err'))
    abs_err = np.zeros(len(df))
    rel_err = np.zeros(len(df))
    for exp in df.index:
        # Pull assembly geometry information from table
        n_ring = df.loc[exp]['Rings']
        wire_pitch = df.loc[exp]['H']
        pin_diameter = df.loc[exp]['D']
        pin_pitch = pin_diameter * df.loc[exp]['P/D']  # m
        clad_thickness = 0.5 / 1e3  # m
        wire_diameter = df.loc[exp]['D_wire']
        duct_inner_ftf = df.loc[exp]['Duct FTF'] + 1e-4

        # None of this stuff matters - it's not used in the calculation
        duct_ftf = [duct_inner_ftf, duct_inner_ftf + 0.001]  # m
        inlet_flow_rate = 30.0  # kg /s; enough to be turbulent
        # inlet_temp = 298.15
        inlet_temp = 273.15
        coolant_obj = dassh.Material('water', mat_data.water_temperature)
        duct_obj = dassh.Material('ss316')
        try:
            a = make_assembly(n_ring, pin_pitch, pin_diameter,
                              clad_thickness, wire_pitch,
                              wire_diameter, duct_ftf, coolant_obj,
                              duct_obj, inlet_temp, inlet_flow_rate)
        except:
            pytest.xfail("Failure in DASSH Assembly instantiation "
                         "(should raise another error elsewhere in "
                         "the tests)")
        res = (friction_ctd.calculate_bundle_friction_factor_const(a)
               ['turbulent'])
        abs_err[exp] = res - df.loc[exp]['CTD']
        rel_err[exp] = abs_err[exp] / df.loc[exp]['CTD']
        # Only print if assertion fails
        print('{:<2d} {:12s} {:5d} {:8.5f} {:8.3f} {:8.2f}'
              .format(exp, df.loc[exp]['Investigators'],
                      df.loc[exp]['Year'], res, df.loc[exp]['CTD'],
                      100 * rel_err[exp]))
    print(max(np.abs(abs_err)))
    print(max(np.abs(rel_err)))
    assert max(np.abs(abs_err)) < 0.025
    assert max(np.abs(rel_err)) < 0.10


def test_thesis_asm_sc_friction_constants(thesis_asm_rr):
    """Test DASSH correlation results against Cheng 1984 thesis result
    (WWCD program) for subchannel friction constants; see page B9"""
    # NOTE: The WWCD program has two bugs that result in incorrect
    # values for the corner subchannel. See the note in unit test
    # "test_ctd_fs_constants" for details.
    tmp = (friction_ctd
           .calculate_subchannel_friction_factor_const(thesis_asm_rr))
    ans = {'laminar': [79.78, 89.07, 114.97],  # 103.54
           'turbulent': [0.2273, 0.2416, 0.3526]}  # 0.2456
    for key in tmp.keys():
        print('ans', key, ans[key])
        print('result', key, tmp[key])
        for cfi in range(len(tmp[key])):
            err = (tmp[key][cfi] - ans[key][cfi]) / ans[key][cfi]
            assert abs(100 * err) < 1.0


def test_thesis_asm_friction_constants(thesis_asm_rr):
    """Test DASSH correlation results against Cheng 1984 thesis result
    (WWCD program) for bundle friction constants; see page B9"""
    # Note - this test uses the DASSH-calculated subchannel friction
    # factor constants, which are different than those generated by
    # the WWCD program. The WWCD program has a bug that results in
    # an incorrect corner subchannel friction factor. See the note
    # in unit test "test_ctd_fs_constants" for details.
    cfb_L = (friction_ctd
             .calculate_bundle_friction_factor_const(thesis_asm_rr)
             ['laminar'])
    error = (cfb_L - 81.11) / 81.11
    print(cfb_L)
    print(error)
    assert abs(100 * error) < 1.0

    cfb_T = (friction_ctd
             .calculate_bundle_friction_factor_const(thesis_asm_rr)
             ['turbulent'])
    error = (cfb_T - 0.2306) / 0.2306
    print(cfb_T)
    print(error)
    assert abs(100 * error) < 2.0


def test_thesis_asm_friction_constants_exact(thesis_asm_rr):
    """Test DASSH correlation results against Cheng 1984 thesis result
    (WWCD program) for ; see page B9"""
    # NOTE: The "Cf_sc" values are taken as those generated by the
    # WWCD code, but the code has bugs. See the note in unit test
    # "test_ctd_fs_constants" for details.
    cf_sc = {}
    cf_sc['laminar'] = [79.78, 89.07, 103.54]
    cf_sc['turbulent'] = [0.2273, 0.2416, 0.2456]
    cfb = friction_ctd._calc_cfb(thesis_asm_rr, cf_sc)
    error_l = (81.11 - cfb['laminar']) / 81.11
    assert abs(100 * error_l) < 0.01
    error_t = (0.2306 - cfb['turbulent']) / 0.2306
    assert abs(100 * error_t) < 0.01


def test_compare_ff_correlations_turbulent(textbook_active_rr):
    """Compare the friction factors obtained by different
    correlations for turbulent flow

    Notes
    -----
    This test compares the calculated friction factors to each
    other, not to experimental data. In the future, another
    test may utilize a dataset of experimental data to confirm
    that the correlations achieve their advertised uncertainties.

    """
    # tol = 10.0  # percent
    # Will only print if the test fails
    # print(textbook_asm.corr)
    # print(textbook_asm.corr_names)
    # print(textbook_asm.correlation_constants['fs'])
    print('P/D ', (textbook_active_rr.pin_pitch
                   / textbook_active_rr.pin_diameter))
    print('H/D ', (textbook_active_rr.wire_pitch
                   / textbook_active_rr.pin_diameter))
    print('Re  ', textbook_active_rr.coolant_int_params['Re'])
    print('{:<6s} {:>6s} {:>6s}'.format('Corr.', 'ff', '% Diff'))
    corr = [friction_ctd, friction_cts, friction_uctd,
            friction_nov, friction_reh]
    name = ['CTD', 'CTS', 'UCTD', 'NOV', 'REH']
    res = np.zeros(len(corr))
    abs_err = np.zeros(len(corr))
    rel_err = np.zeros(len(corr))
    for i in range(len(corr)):
        textbook_active_rr._setup_correlations(
            name[i], 'CTD', 'CTD', 'DB', None)
        textbook_active_rr._update_coolant_int_params(300.15)
        res[i] = corr[i].calculate_bundle_friction_factor(textbook_active_rr)
        abs_err[i] = (res[i] - res[0])
        rel_err[i] = abs_err[i] / res[0]
        print('{:<6s} {:6.5f} {:6.5f}'
              .format(name[i], res[i], 100 * rel_err[i]))
    print('Expect similar results from all correlations')
    print('Max abs. err relative to CTD: ', max(abs(abs_err)))
    print('Max rel. err relative to CTD: ', max(abs(rel_err)))
    assert max(np.abs(abs_err)) < 0.001
    assert max(np.abs(rel_err)) < 0.05


########################################################################
# FLOW SPLIT
########################################################################


def test_flow_split(textbook_active_rr):
    """Test the relative magnitude of the flow split values given
    by the four correlations"""
    fs = [flowsplit_mit, flowsplit_nov, flowsplit_ctd, flowsplit_uctd]
    res = np.zeros((len(fs), 3))
    for i in range(len(fs)):
        del textbook_active_rr.corr_constants['fs']
        textbook_active_rr.corr_constants['fs'] = \
            fs[i].calc_constants(textbook_active_rr)
        res[i, :] = fs[i].calculate_flow_split(textbook_active_rr)

    # In all cases, x1 < x2 and x2 >= x3
    assert all([x[0] < x[1] for x in res])
    assert all([x[1] >= x[2] for x in res])

    # Make sure all have similar total velocity magnitude
    vel = np.array([sum(res[i, :]) for i in range(len(res))])
    assert np.std(vel) < 0.10  # arbitrary but seems relatively small


def test_flowsplit_x2_ctd(testdir):
    """Test the Cheng-Todreas Detailed (1986) flow split
    correlations against data published in the 1986 paper

    Notes
    -----
    I (M. Atz) could not find any tabulated data, so I visually
    assessed the data by hand from Figure 14 of the Cheng-Todreas
    paper (1986). Some of the pin bundle characteristics were known
    from the friction factor tests.

    """
    df = pd.read_csv(os.path.join(testdir, 'test_data',
                                  'ctd_x2.csv'), header=0)
    df = df.dropna()
    flowrate = {'laminar': 0.5, 'turbulent': 30.0}  # arbitary
    abs_err = np.zeros(len(df))
    rel_err = np.zeros(len(df))
    # Will only print if the test fails
    print('{:15s} {:>5s} {:>8s} {:>8s} {:>8s}'
          .format('Experiment', 'Year', 'X2', 'CTD', 'Rel. err'))
    idx = 0
    for exp in df.index:
        # Pull assembly geometry information from table
        n_ring = int(df.loc[exp]['Rings'])
        wire_pitch = df.loc[exp]['H']
        pin_diameter = df.loc[exp]['D']
        pin_pitch = pin_diameter * df.loc[exp]['P/D']  # m
        clad_thickness = 0.5 / 1e3  # m
        wire_diameter = df.loc[exp]['D_wire']
        duct_inner_ftf = df.loc[exp]['Duct FTF'] + 1e-4
        # None of this stuff matters - it's not used in the calculation
        duct_ftf = [duct_inner_ftf, duct_inner_ftf + 0.001]  # m
        inlet_flow_rate = flowrate[df.loc[exp]['Regime']]
        inlet_temp = 273.35
        coolant_obj = dassh.Material('water', mat_data.water_temperature)
        duct_obj = dassh.Material('ss316')
        try:
            a = make_assembly(n_ring, pin_pitch, pin_diameter,
                              clad_thickness, wire_pitch,
                              wire_diameter, duct_ftf, coolant_obj,
                              duct_obj, inlet_temp, inlet_flow_rate)
        except:
            pytest.xfail("Failure in DASSH Assembly instantiation "
                         "(should raise another error elsewhere in "
                         "the tests)")
        a._update_coolant_int_params(inlet_temp)
        print(a.coolant_int_params['Re'], df.loc[exp]['Regime'])
        res = flowsplit_ctd.calculate_flow_split(a)
        abs_err[idx] = res[1] - df.loc[exp]['CTD']
        rel_err[idx] = abs_err[idx] / df.loc[exp]['CTD']
        # Only print if assertion fails
        print('{:<2d} {:12s} {:5d} {:8.5f} {:8.3f} {:8.2f}'
              .format(exp, df.loc[exp]['Investigators'],
                      df.loc[exp]['Year'], res[1], df.loc[exp]['CTD'],
                      100 * rel_err[idx]))
        idx += 1
    print('Max abs err: ', round(max(np.abs(abs_err)), 2))
    print('Max rel err: ', round(max(np.abs(rel_err)), 2))
    assert max(np.abs(abs_err)) < 0.02
    assert max(np.abs(rel_err)) < 0.02


def test_ctd_transition_flowsplit(thesis_asm_rr):
    """Test the Cheng-Todreas Detailed (1986) flow split
    correlations in the transition regime

    Notes
    -----
    The transition flowsplit values should fall within the range
    set by the laminar and turbulent flow split values. It should
    approach the laminar/turbulent flow split values as the Reynolds
    number approaches the laminar/turbulent regime.

    """
    test_asm = thesis_asm_rr.clone()
    flowsplit = {}

    # Re bounds depend only on geometry
    Re_bnds = friction_ctd.calculate_Re_bounds(test_asm)
    print(Re_bnds)

    # Turbulent - thesis_asm fixture comes in turbulent regime as-is
    test_asm._init_static_correlated_params(300.15)  # abitrary water temp
    test_asm._update_coolant_int_params(300.15)  # arbitrary water temp
    print('turbulent ', test_asm.coolant_int_params['Re'])
    assert test_asm.coolant_int_params['Re'] > Re_bnds[1]  # turbulent
    flowsplit['turbulent'] = test_asm.coolant_int_params['fs']

    # Laminar - need to adjust flow rate and update params
    test_asm.int_flow_rate = 0.1
    test_asm._init_static_correlated_params(300.15)
    test_asm._update_coolant_int_params(300.15)
    print('laminar ', test_asm.coolant_int_params['Re'])
    assert test_asm.coolant_int_params['Re'] < Re_bnds[0]  # laminar
    flowsplit['laminar'] = test_asm.coolant_int_params['fs']

    # Transition
    test_asm.int_flow_rate = 1.0
    test_asm._init_static_correlated_params(300.15)
    test_asm._update_coolant_int_params(300.15)
    print('transition ', test_asm.coolant_int_params['Re'])
    assert test_asm.coolant_int_params['Re'] > Re_bnds[0]
    assert test_asm.coolant_int_params['Re'] < Re_bnds[1]

    # Compare transition with turbulent, laminar
    for i in range(0, len(test_asm.coolant_int_params['fs'])):
        bnds = [flowsplit['laminar'][i], flowsplit['turbulent'][i]]
        assert min(bnds) < test_asm.coolant_int_params['fs'][i] < max(bnds)

    # Make flow more turbulent, flow split should approach turbulent
    fs1 = test_asm.coolant_int_params['fs']
    test_asm.int_flow_rate = 6.0
    test_asm._init_static_correlated_params(300.15)
    test_asm._update_coolant_int_params(300.15)
    print('transition 2', test_asm.coolant_int_params['Re'])
    assert test_asm.coolant_int_params['Re'] > Re_bnds[0]
    assert test_asm.coolant_int_params['Re'] < Re_bnds[1]

    for i in range(0, len(test_asm.coolant_int_params['fs'])):
        assert (abs(flowsplit['turbulent'][i] - fs1[i])
                > abs(flowsplit['turbulent'][i]
                      - test_asm.coolant_int_params['fs'][i]))


def test_ctd_fs_constants(thesis_asm_rr):
    """Test DASSH correlation results against Cheng 1984 thesis result
    (WWCD program); see page B9

    Note
    ----
    The corner subchannel results in the reference are incorrect.
    There are two bugs in the WWCD program. One of them affects only
    the edge subchannel; the other affects both the edge and corner
    subchannels. The two bugs for the edge subchannel cancel out.

    Bug 1: Edge subchannel friction factor
    Page B18, g(2) = ...
        The value "ar" equals "Ar2" * 2 and "Ar3" * 3. In the equation
        for g(3), "ar" is divided by 3, but it is not divided by 2 in
        the equation for g(2). Therefore, g(2) is 2x too big.

    Bug 2: Edge and corner subchannel wire sweeping constant coeffs
    Page B19, a2 and b2
        The values for a2 and b2 are 1/2 of the value that is reported
        in the thesis body and in the 1986 paper. Therefore, the wire
        sweeping constant calculated by WWCD is 2x too small.
    Because the corner subchannel friction factor is different, all
    subchannel flow split parameters will be slightly different.

    """
    # ans = {'laminar': np.array([0.860, 1.251, 0.552]),
    #        'turbulent': np.array([0.955, 1.080, 0.862])}
    ans = {'laminar': np.array([0.860, 1.251, 0.499]),
           'turbulent': np.array([0.955, 1.080, 0.712])}
    res = thesis_asm_rr.corr_constants['fs']['fs']
    for k in res.keys():
        diff = ans[k] - res[k]
        reldiff = diff / ans[k]
        maxdiff = np.max(np.abs(reldiff))
        if maxdiff >= 0.01:
            print(k)
            print('ans:', ans[k])
            print('res:', res[k])
            print('rel diff:', reldiff)
        assert maxdiff < 0.01


########################################################################
# MIXING
########################################################################


def test_eddy_diffusivity_constants(testdir):
    """Test the Cheng-Todreas Detailed (1986) eddy diffusivity
    constant (CmT) correlations against data published in the 1986
    paper

    Notes
    -----
    From Table 1 of the 1986 paper

    """
    df = pd.read_csv(os.path.join(testdir, 'test_data',
                                  'mp_turbulent.csv'), header=0)
    df = df.dropna(subset=['CmT'])
    abs_err = np.zeros(len(df))
    rel_err = np.zeros(len(df))
    # Will only print if the test fails
    print('{:15s} {:>5s} {:>8s} {:>8s} {:>8s}'
          .format('Experiment', 'Year', 'CmT', 'Data', 'Rel. err'))
    idx = 0
    for exp in df.index:
        # Pull assembly geometry information from table
        n_ring = int(df.loc[exp]['Rings'])
        wire_pitch = df.loc[exp]['H']
        pin_diameter = df.loc[exp]['D']
        pin_pitch = pin_diameter * df.loc[exp]['P/D']  # m
        clad_thickness = 0.5 / 1e3  # m
        wire_diameter = df.loc[exp]['D_wire']
        duct_inner_ftf = df.loc[exp]['Duct FTF']
        # None of this stuff matters - it's not used in the calculation
        duct_ftf = [duct_inner_ftf, duct_inner_ftf + 0.001]  # m
        inlet_flow_rate = 30.0  # enough to be turbulent
        inlet_temp = 273.15
        coolant_obj = dassh.Material('water', mat_data.water_temperature)
        duct_obj = dassh.Material('ss316')
        try:
            a = make_assembly(n_ring, pin_pitch, pin_diameter,
                              clad_thickness, wire_pitch,
                              wire_diameter, duct_ftf, coolant_obj,
                              duct_obj, inlet_temp, inlet_flow_rate)
        except:
            pytest.xfail("Failure in DASSH Assembly instantiation "
                         "(should raise another error elsewhere in "
                         "the tests)")

        res = (mixing_ctd
               .calculate_mixing_param_constants(a)[0]['turbulent'])
        abs_err[idx] = res - df.loc[exp]['CmT']
        rel_err[idx] = abs_err[idx] / df.loc[exp]['CmT']
        # Only print if assertion fails
        print('{:<2d} {:12s} {:5d} {:8.5f} {:8.3f} {:8.2f}'
              .format(exp, df.loc[exp]['Investigators'],
                      df.loc[exp]['Year'], res, df.loc[exp]['CmT'],
                      100 * rel_err[idx]))
        idx += 1
    # print(round(max(np.abs(abs_err)), 2))
    # print(round(max(np.abs(rel_err)), 2))
    var = sum(np.array([x**2 for x in abs_err])) / len(abs_err)
    assert np.sqrt(var) <= 0.20


def test_eddy_diffusivity(testdir):
    """Test the eddy diffusivity (epsilon) from the Cheng-Todreas
    (1986) correlations against data published in the 1986 paper

    Notes
    -----
    From Table 1 of the 1986 paper

    """
    df = pd.read_csv(os.path.join(testdir, 'test_data',
                                  'mp_turbulent.csv'), header=0)
    df = df.dropna(subset=['eps'])  # , 'Duct FTF'])
    # Drop the second Chiu experiment - the value for eddy diffusivity
    # must be wrong - see Figure 5.17 (page 241) of Cheng's MIT thesis
    df = df[(df['Investigators'] != 'Chiu') & (df['H/D'] != 8)]
    for mp in [mixing_ctd]:
        abs_err = np.zeros(len(df))
        rel_err = np.zeros(len(df))
        # Will only print if the test fails
        print('{:15s} {:>5s} {:11s} {:>8s} {:>8s} {:>8s}'
              .format('Experiment', 'Year', 'Model',
                      'Eps', 'Data', 'Rel. err'))
        idx = 0
        for exp in df.index:
            # Pull assembly geometry information from table
            n_ring = int(df.loc[exp]['Rings'])
            wire_pitch = df.loc[exp]['H']
            pin_diameter = df.loc[exp]['D']
            pin_pitch = pin_diameter * df.loc[exp]['P/D']  # m
            clad_thickness = 0.5 / 1e3  # m
            wire_diameter = df.loc[exp]['D_wire']
            duct_inner_ftf = df.loc[exp]['Duct FTF']
            # None of this stuff matters - it's not used in the calculation
            duct_ftf = [duct_inner_ftf, duct_inner_ftf + 0.001]  # m
            inlet_flow_rate = 100.0  # enough to be turbulent
            inlet_temp = 273.15
            coolant_obj = dassh.Material('water', mat_data.water_temperature)
            duct_obj = dassh.Material('ss316')
            try:
                a = make_assembly(n_ring, pin_pitch, pin_diameter,
                                  clad_thickness, wire_pitch,
                                  wire_diameter, duct_ftf, coolant_obj,
                                  duct_obj, inlet_temp, inlet_flow_rate)
            except:
                pytest.xfail("Failure in DASSH Assembly instantiation "
                             "(should raise another error elsewhere in "
                             "the tests)")
            # Force turbulence
            a.coolant_int_params['Re'] = 2e4

            # Calculate eddy diffusivity
            res = mp.calculate_mixing_params(a)[0]
            # Eddy diffusivity is scaled in module by L[0][0]; need to
            # UNDO because here we're comparing the dimensionless value
            res /= a.L[0][0]

            abs_err[idx] = res - df.loc[exp]['eps']
            rel_err[idx] = abs_err[idx] / df.loc[exp]['eps']
            # Only print if assertion fails
            print('{:<2d} {:12s} {:5d} {:12s} {:8.5f} {:8.3f} {:8.2f}'
                  .format(exp, df.loc[exp]['Investigators'],
                          df.loc[exp]['Year'],
                          mp.__name__.split('.')[-1], res,
                          df.loc[exp]['eps'], 100 * rel_err[idx]))
            idx += 1
        # print(round(max(np.abs(abs_err)), 2))
        # print(round(max(np.abs(rel_err)), 2))
        var = np.array([x**2 for x in abs_err])
        var = sum(var) / len(var)
        print(np.sqrt(var))
        assert np.sqrt(var) <= 0.01
        # demonstrated fit (Cheng 1986 fig 17) is +/- 25%, but some
        # exceed that or are not shown, so 35% is used here (yikes!)
        assert max(rel_err) < 0.35


def test_mit_eddy_diffusivity(testdir):
    """Test the eddy diffusivity (epsilon) from the MIT (1978)
    correlations against data published in the 1980 report
    comparing the ENERGY and SIMPLE correlations.

    Notes
    -----
    From Tables 1, 2 of the 1980 report
    Three assemblies: fuel, blanket, intermediate

    """
    # n_pin = np.array([217, 61, 37])
    n_ring = np.array([9, 5, 4])
    d_pin = np.array([0.230, 0.506, 0.501]) * 0.0254  # in -> m
    d_wire = np.array([0.056, 0.033, 0.075]) * 0.0254  # in -> m
    pitch = np.array([0.288, 0.542, 0.578]) * 0.0254  # in -> m
    clad_thickness = 0.5 / 1e3  # m
    lead = np.array([11.9, 4.00, 10.5]) * 0.0254  # in -> m
    ans = np.array([0.0288, 0.448, 0.100])
    abs_err = np.zeros(len(ans))
    rel_err = np.zeros(len(ans))
    for i in range(3):
        # ftf distance doesn't, matter, just need one that's big enough
        duct_inner_ftf = (np.sqrt(3) * (n_ring[i] - 1) * pitch[i]
                          + d_pin[i] + 2 * d_wire[i] + 0.002)
        duct_ftf = [duct_inner_ftf, duct_inner_ftf + 0.001]  # m
        inlet_flow_rate = 100.0  # enough to be turbulent
        inlet_temp = 273.15
        coolant_obj = dassh.Material('water', mat_data.water_temperature)
        duct_obj = dassh.Material('ss316')
        try:
            a = make_assembly(n_ring[i], pitch[i], d_pin[i],
                              clad_thickness, lead[i], d_wire[i],
                              duct_ftf, coolant_obj, duct_obj,
                              inlet_temp, inlet_flow_rate,
                              corr_mixing='MIT', se2geo=True)
        except:
            pytest.xfail("Failure in DASSH Assembly instantiation "
                         "(should raise another error elsewhere in "
                         "the tests)")
        # Force turbulence
        a.coolant_int_params['Re'] = 2e4
        # Calculate eddy diffusivity
        res = mixing_mit.calculate_mixing_params(a)[0]
        # print(res)
        # print(a.corr_names['mix'])
        # Eddy diffusivity is scaled in module by hydraulic diameter
        # of interior subchannel; need to UNDO because here we're
        # comparing the dimensionless value
        res /= a.params['de'][0]

        abs_err[i] = res - ans[i]
        rel_err[i] = abs_err[i] / ans[i]
        print(ans[i], res, abs_err[i], rel_err[i])
    var = np.array([x**2 for x in abs_err])
    var = sum(var) / len(var)
    print('std err: ', np.sqrt(var))
    assert np.sqrt(var) <= 0.02
    print('max rel err: ', max(abs(rel_err)))
    assert max(abs(rel_err)) < 0.065


def test_swirl_velocity_constants(testdir):
    """Test the swirl velocity constant (CsT) from the
    Cheng-Todreas (1986) correlations against data published
    in the 1986 paper

    Notes
    -----
    From Table 2 of the 1986 paper

    """
    df = pd.read_csv(os.path.join(testdir, 'test_data',
                                  'mp_turbulent.csv'), header=0)
    df = df.dropna(subset=['CsT'])
    abs_err = np.zeros(len(df))
    rel_err = np.zeros(len(df))
    # Will only print if the test fails
    print('{:15s} {:>5s} {:>8s} {:>8s} {:>8s}'
          .format('Experiment', 'Year', 'CsT', 'Data', 'Rel. err'))
    idx = 0
    for exp in df.index:
        # Pull assembly geometry information from table
        n_ring = int(df.loc[exp]['Rings'])
        wire_pitch = df.loc[exp]['H']
        pin_diameter = df.loc[exp]['D']
        pin_pitch = pin_diameter * df.loc[exp]['P/D']  # m
        clad_thickness = 0.5 / 1e3  # m
        wire_diameter = df.loc[exp]['D_wire']
        duct_inner_ftf = df.loc[exp]['Duct FTF']
        # None of this stuff matters - it's not used in the calculation
        duct_ftf = [duct_inner_ftf, duct_inner_ftf + 0.001]  # m
        inlet_flow_rate = 30.0  # enough to be turbulent
        inlet_temp = 273.15
        coolant_obj = dassh.Material('water', mat_data.water_temperature)
        duct_obj = dassh.Material('ss316')
        try:
            a = make_assembly(n_ring, pin_pitch, pin_diameter,
                              clad_thickness, wire_pitch,
                              wire_diameter, duct_ftf, coolant_obj,
                              duct_obj, inlet_temp, inlet_flow_rate)
        except:
            pytest.xfail("Failure in DASSH Assembly instantiation "
                         "(should raise another error elsewhere in "
                         "the tests)")
        a.coolant_int_params['Re'] = 2e4
        print()
        print(a.edge_pitch)
        res = mixing_ctd.calculate_mixing_param_constants(a)[1]
        res = res['turbulent']
        abs_err[idx] = res - df.loc[exp]['CsT']
        rel_err[idx] = abs_err[idx] / df.loc[exp]['CsT']
        # Only print if assertion fails
        print('{:<2d} {:12s} {:5d} {:8.5f} {:8.3f} {:8.2f}'
              .format(exp, df.loc[exp]['Investigators'],
                      df.loc[exp]['Year'], res,
                      df.loc[exp]['CsT'], 100 * rel_err[idx]))
        idx += 1
    # print(round(max(np.abs(abs_err)), 2))
    # print(round(max(np.abs(rel_err)), 2))
    var = np.array([x**2 for x in abs_err])
    var = sum(var) / len(abs_err)
    print('Standard error: ', np.sqrt(var))
    assert np.sqrt(var) <= 0.30
    assert all(x < 0.30 for x in rel_err)


def test_swirl_velocity(testdir):
    """Test the swirl velocity (C1L) from the MIT (1978) and
    Cheng-Todreas (1986) correlations against data published
    in the 1986 paper

    Notes
    -----
    From Table 2 of the 1986 paper

    """
    df = pd.read_csv(os.path.join(testdir, 'test_data',
                                  'mp_turbulent.csv'), header=0)
    df = df.dropna(subset=['C1L'])  # , 'Duct FTF'])
    corr_name = ['CTD', 'MIT']
    corr = [mixing_ctd, mixing_mit]
    for mpi in range(2):
        mp = corr[mpi]
        abs_err = np.zeros(len(df))
        rel_err = np.zeros(len(df))
        # Will only print if the test fails
        print('{:15s} {:>5s} {:11s} {:>8s} {:>8s} {:>8s}'
              .format('Experiment', 'Year', 'Model',
                      'CsT', 'Data', 'Rel. err'))
        idx = 0
        for exp in df.index:
            # Pull assembly geometry information from table
            n_ring = int(df.loc[exp]['Rings'])
            wire_pitch = df.loc[exp]['H']
            pin_diameter = df.loc[exp]['D']
            pin_pitch = pin_diameter * df.loc[exp]['P/D']  # m
            clad_thickness = 0.5 / 1e3  # m
            wire_diameter = df.loc[exp]['D_wire']
            duct_inner_ftf = df.loc[exp]['Duct FTF']
            # None of this matters - it's not used in the calculation
            duct_ftf = [duct_inner_ftf, duct_inner_ftf + 0.001]  # m
            inlet_flow_rate = 100.0  # enough to be turbulent
            inlet_temp = 273.15
            coolant_obj = dassh.Material('water', mat_data.water_temperature)
            duct_obj = dassh.Material('ss316')
            # try:
            #     a = make_assembly(n_ring, pin_pitch, pin_diameter,
            #                       clad_thickness, wire_pitch,
            #                       wire_diameter, duct_ftf, coolant_obj,
            #                       duct_obj, inlet_temp, inlet_flow_rate)
            # except ValueError:
            #     raise
            # except:
            #     pytest.xfail("Failure in DASSH Assembly instantiation "
            #                  "(should raise another error elsewhere in "
            #                  "the tests)")
            a = make_assembly(n_ring, pin_pitch, pin_diameter,
                              clad_thickness, wire_pitch,
                              wire_diameter, duct_ftf, coolant_obj,
                              duct_obj, inlet_temp, inlet_flow_rate,
                              corr_mixing=corr_name[mpi])
            # Force turbulence
            a.coolant_int_params['Re'] = 2e4
            res = mp.calculate_mixing_params(a)[1]
            abs_err[idx] = round(res, 2) - df.loc[exp]['C1L']
            rel_err[idx] = abs_err[idx] / df.loc[exp]['C1L']
            # Only print if assertion fails
            print('{:<2d} {:12s} {:5d} {:12s} {:8.5f} {:8.3f} {:8.2f}'
                  .format(exp, df.loc[exp]['Investigators'],
                          df.loc[exp]['Year'],
                          mp.__name__.split('.')[-1], res,
                          df.loc[exp]['C1L'], 100 * rel_err[idx]))

            idx += 1
        # print(round(max(np.abs(abs_err)), 2))
        # print(round(max(np.abs(rel_err)), 2))
        var = np.array([x**2 for x in abs_err])
        var = sum(var) / len(var)
        print('std err: ', np.sqrt(var))
        assert np.sqrt(var) <= 0.05


def test_ctd_sc_intermittency_factor(thesis_asm_rr):
    """Test the calculation of intermittency factor used in the
    Cheng-Todreas friction and flow split correlations in the
    transition regime for the individual coolant subchannels

    """
    test_asm = thesis_asm_rr.clone()
    # Intermittency factor should be greater than 1 if turbulent
    Re_bnds = friction_ctd.calculate_Re_bounds(test_asm)
    x = mixing_ctd.calc_sc_intermittency_factors(test_asm,
                                                 Re_bnds[0],
                                                 Re_bnds[1])
    assert all([x[i] > 1 for i in range(len(x))])

    # Intermittency factor should be less than 1 if transition
    test_asm.int_flow_rate = 5.0
    test_asm._init_static_correlated_params(300.15)  # abitrary water temp
    test_asm._update_coolant_int_params(300.15)  # arbitrary water temp
    # (check that transition flow is achieved)
    assert Re_bnds[0] < test_asm.coolant_int_params['Re'] < Re_bnds[1]
    x = mixing_ctd.calc_sc_intermittency_factors(test_asm,
                                                 Re_bnds[0],
                                                 Re_bnds[1])
    print(x)
    assert all([0 < x[i] < 1 for i in range(len(x))])


class MockRR(object):
    """Mock RoddedRegion class for testing Kim-Chung (2001) bare
    rod bundle turbulent mixing"""
    def __init__(self, pitch, diam):
        self.pin_pitch = pitch
        self.pin_diameter = diam
        A_int = 0.25 * np.sqrt(3) * pitch**2 - 0.125 * np.pi * diam**2
        Pw_int = np.pi * diam / 2
        De_int = 4 * A_int / Pw_int
        eta_int = np.sqrt(3) * pitch / 3
        self.params = {
            'area': [A_int, 0.0, 0.0],
            'wp': [Pw_int, 0.0, 0.0],
            'de': [De_int, 0.0, 0.0]
        }
        self.L = [[eta_int, 0.0, 0.0]]


def test_kc_bare_mixing_parameter_fig6_g2d():
    """Test the turbulent mixing parameter result from the Kim
    and Chung correlation for bare rod bundles vs g/D"""
    # Set up some examples
    g2d = [0.1, 0.2, 0.3]
    diameter = 0.01
    Pr = [0.001, 0.001, 0.01, 0.01]
    Re = [1e5, 3e5, 1e4, 1e5]

    # Answers from the graph
    ans = [[3.50, 4.77, 6.12],
           [1.94, 2.31, 2.72],
           [3.00, 4.01, 5.08],
           [1.28, 1.19, 1.22]]

    for i in range(len(Pr)):
        for j in range(len(g2d)):
            p2d = g2d[j] + 1
            pitch = p2d * diameter
            mock_rr = MockRR(pitch, diameter)
            C = mixing_kc.calc_constants(mock_rr, use_simple=True)
            Stg = mixing_kc._calculate_stg(Pr[i], Re[i], *C)
            Stg_mod = Stg * 100 / Re[i]**-0.1
            ans_ij = ans[i][j]
            diff_ij = abs(ans_ij - Stg_mod)
            rdiff_ij = diff_ij / ans_ij
            if rdiff_ij > 0.02:
                print('Result:   ', round(Stg_mod, 2))
                print('Answer:   ', ans_ij)
                print('Diff:     ', round(diff_ij, 4))
                print('Rel diff: ', round(rdiff_ij, 4))
                assert rdiff_ij <= 0.02


def test_kc_bare_mixing_parameter_fig5_g2d():
    """Test the turbulent mixing parameter result from the Kim
    and Chung correlation for bare rod bundles vs g/D"""
    # Set up some examples
    g2d = [0.1, 0.2, 0.3, 0.4]
    diameter = 0.01
    Pr = 1.0
    Re = 1e5

    # Answers from the graph
    ans = [1.03, 0.79, 0.69, 0.61]

    for j in range(len(g2d)):
        p2d = g2d[j] + 1
        pitch = p2d * diameter
        mock_rr = MockRR(pitch, diameter)
        C = mixing_kc.calc_constants(mock_rr, use_simple=True)
        Stg = mixing_kc._calculate_stg(Pr, Re, *C)
        Stg_mod = Stg * 100 / Re**-0.1
        ans_ij = ans[j]
        diff_ij = abs(ans_ij - Stg_mod)
        rdiff_ij = diff_ij / ans_ij
        if rdiff_ij > 0.02:
            print('g/D:      ', g2d[j])
            print('Result:   ', round(Stg_mod, 2))
            print('Answer:   ', ans_ij)
            print('Diff:     ', round(diff_ij, 4))
            print('Rel diff: ', round(rdiff_ij, 4))
            assert rdiff_ij <= 0.02


# def test_kc_bare_mixing_parameter_fig6_Pr(testdir):
#     """Test the turbulent mixing parameter result from the Kim
#     and Chung correlation for bare rod bundles vs Pr
#
#     NOTE: Fig 6 e-h do not make sense. There should be some overlap
#     between the data shown in these figures and that shown in the
#     preceding ones, but there is not. Cannot discern how the authors
#     generated these values.
#
#     """
#     # Set up some examples
#     p2d = [1.1]  # , 1.3, 1.3, 1.3]
#     diameter = 0.01
#     Re = [1e4, 1e5, 1e4, 1e5]
#     p2d = [1.1]
#     Re = [1e5]
#     Pr = [0.001, 0.01, 0.1, 1.0]
#
#     # Answers from the graph
#     ans = [[110, 11.0, 1.5, 0.41], [], [], []]

########################################################################
# CORRELATION APPLICABILITY
########################################################################
# if name[i] == 'ENG':  # outside the Engel range
#     with pytest.warns(UserWarning):
#         corr[i].calculate_bundle_friction_factor(textbook_asm)

########################################################################
# SHAPE FACTOR
########################################################################


def test_ct_shape_factor():
    """Test Cheng-Todreas shape factor against analytical results"""
    # Vary P/D, get different answers
    p2d = [1.1, 1.2, 1.3, 1.4]
    # Answers calculated by hand, verified against Fig 8 in Lodi 2016
    ans = [1.45, 1.28, 1.235, 1.22]
    # Some standard parameters for assembly instantiation (some will
    # be changed to assess that the proper warnings are raised)
    n_ring = 4
    clad_thickness = 0.5 / 1e3
    wire_diameter = 1.094 / 1e3  # mm -> m
    duct_ftf = [0.11154, 0.11757]  # m
    h2d = 3.0
    inlet_flow_rate = 30.0  # kg /s
    inlet_temp = 273.15 + 350.0  # K
    coolant_obj = dassh.Material('sodium')
    duct_obj = dassh.Material('ss316')
    for i in range(len(p2d)):
        pin_diameter = ((duct_ftf[0] - 2 * wire_diameter)
                        / (np.sqrt(3) * (n_ring - 1) * p2d[i] + 1))
        pin_diameter -= 1e-7  # fudge factor to ensure pins fit in duct
        pin_pitch = pin_diameter * p2d[i]
        wire_pitch = pin_diameter * h2d
        a = make_assembly(
            n_ring, pin_pitch, pin_diameter, clad_thickness, wire_pitch,
            wire_diameter, duct_ftf, coolant_obj, duct_obj, inlet_temp,
            inlet_flow_rate, corr_shapefactor='CT')
        diff = a._sf - ans[i]
        assert abs(diff) / ans[i] <= 0.01


########################################################################
# SPACER GRID PRESSURE LOSS
########################################################################


def test_rehme_spacergrid():
    """Test that Rehme spacer grid method returns expected loss coeff
    (generated based on visual inspection of figure in paper)"""
    Re = np.array([   5e3, 1.5e4,   5e4,  1.5e5])
    ans = np.array([11.32, 8.174, 6.851, 6.4185])
    res = grid_rehme.calc_loss_coeff(Re, 1.0)
    diff = res - ans
    assert np.allclose(diff, 0.0)


def test_cdd_spacergrid():
    """Test that CDD spacer grid method with default coefficients gives
    loss coeff similar to that from the simpler Rehme interpolation"""
    # Use the Rehme coefficient for Cv data as a function of bundle Re.
    # Skip the first entry because that was generated by linear interp
    # and isn't a relevant comparison between REH and CDD. (For
    # reference, the REH data starts at Re_bundle = ~4e3.
    Cv_REH = grid_rehme.Cv[1:]
    coeffs = grid_cdd._DEFAULT_COEFFS
    coeffs[5] = 100.0
    ans_CDD = np.zeros(Cv_REH.shape[0])
    for i in range(ans_CDD.shape[0]):
        ans_CDD[i] = grid_cdd.calc_loss_coeff(Cv_REH[i, 0], 1.0, coeffs)
    diff = ans_CDD - Cv_REH[:, 1]
    rdiff = diff / Cv_REH[:, 1]
    # The tolerance here is rather arbitrary - the max difference is
    # 0.163, so I set the tolerance to be 0.20 so it would pass. The
    # two relationships do generate similar curves for loss coeff vs.
    # Re, so I think this an acceptable check.
    assert np.max(np.abs(rdiff)) < 0.2
