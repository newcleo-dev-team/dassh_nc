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
date: 2023-03-02
author: matz
Main DASSH calculation procedures
"""
########################################################################
import os
import sys
import numpy as np
import argparse
import cProfile
import logging
import dassh
from dassh.run_dassh import _log_info


def main(args=None):
    """Perform temperature sweep in DASSH"""
    # Parse command line arguments to DASSH
    parser = argparse.ArgumentParser(description='Process DASSH cmd')
    parser.add_argument('inputfile',
                        metavar='inputfile',
                        help='The input file to run with DASSH')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Verbose; print summary with each axial step')
    parser.add_argument('--save_reactor',
                        action='store_true',
                        help='Save DASSH Reactor object after sweep')
    parser.add_argument('--profile',
                        action='store_true',
                        help='Profile the execution of DASSH')
    parser.add_argument('--no_power_calc',
                        action='store_false',
                        help='Skip VARPOW calculation if done previously')
    parser.add_argument('--plot_only_geom',
                        action='store_true',
                        help='Only plot the assembly lattice (pin + subchannel)'
                        ', and then exit')
    args = parser.parse_args(args)

    # Enable the profiler, if desired
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    # Initiate logger
    print(dassh._ascii._ascii_title)
    in_path = os.path.split(args.inputfile)[0]
    dassh_logger = dassh.logged_class.init_root_logger(in_path, 'dassh')
    dassh_logger.info(f"DASSH v{dassh.__version__} logger initialized")

    # Pre-processing
    # Read input file and set up DASSH input object
    dassh_logger.log(_log_info, f'Reading input: {args.inputfile}')
    dassh_input = dassh.DASSH_Input(args.inputfile)

    # CHECK FOR PYTHON VERSION WARNINGS/ERRORS
    # check_version(dassh_input, dassh_logger, args.save_reactor)

    # DASSH calculation without orificing optimization
    if dassh_input.data['Orificing'] is False:
        arg_dict = {
            'save_reactor': args.save_reactor,
            'verbose': args.verbose,
            'no_power_calc': args.no_power_calc,
            'plot_only_geom': args.plot_only_geom,
        }
        dassh.run_dassh(dassh_input, arg_dict)

    # Orificing optimization with DASSH
    else:
        orifice_obj = dassh.orificing.Orificing(dassh_input)
        orifice_obj.optimize()

    # Finish the calculation
    dassh_logger.log(_log_info, 'DASSH execution complete')
    # Print/dump profiler results
    if args.profile:
        pr.disable()
        pr.dump_stats('dassh_profile.out')

    # Shutdown logger by removing file handlers
    dassh.logged_class.shutdown_logger('dassh')


def check_version(dassh_inp, save_reactor):
    """Check for DASSH limitations depending on Python version;
    tentatively deprecated."""
    dassh_logger = logging.getLogger('dassh')
    version = '.'.join([str(sys.version_info.major),
                        str(sys.version_info.minor),
                        str(sys.version_info.micro)])
    if dassh_inp.data['Plot'] and sys.version_info < (3, 7):
        dassh_logger.log(
            30,
            'WARNING: DASSH plotting capability requires '
            f'Python 3.7+; detected {version}')
    if save_reactor and sys.version_info < (3, 7):
        dassh_logger.log(
            30,
            'WARNING: --save_reactor capability requires '
            f'Python 3.7+; detected {version}')
    if dassh_logger.data['Orificing'] and sys.version_info < (3, 7):
        dassh_logger.log(
            40,
            'ERROR: DASSH orificing optimization requires '
            f'Python 3.7+; detected {version}')
        sys.exit(1)
    else:
        pass


def plot():
    """Command-line interface to postprocess DASSH data to make
    matplotlib figures"""
    # Get input file from command line arguments
    parser = argparse.ArgumentParser(description='Process DASSH cmd')
    parser.add_argument('inputfile',
                        metavar='inputfile',
                        help='The input file to run with DASSH')
    args = parser.parse_args()

    # Initiate logger
    print(dassh._ascii._ascii_title)
    in_path = os.path.split(args.inputfile)[0]
    dassh_logger = dassh.logged_class.init_root_logger(in_path,
                                                       'dassh_plot')

    # Check whether Reactor object exists; if so, process with
    # DASSHPlot_Input and get remaining info from Reactor object
    rpath = os.path.join(os.path.abspath(in_path), 'dassh_reactor.pkl')
    if os.path.exists(rpath):
        dassh_logger.log(_log_info, f'Loading DASSH Reactor: {rpath}')
        r = dassh.reactor.load(rpath)
        dassh_logger.log(_log_info, f'Reading input: {args.inputfile}')
        inp = dassh.DASSHPlot_Input(args.inputfile, r)

    # Otherwise, build Reactor object from complete DASSH input
    else:
        dassh_logger.log(_log_info, f'Reading input: {args.inputfile}')
        inp = dassh.DASSH_Input(args.inputfile)
        dassh_logger.log(_log_info, 'Building DASSH Reactor from input')
        r = dassh.Reactor(inp, calc_power=False)

    # Generate figures
    dassh_logger.log(_log_info, 'Generating figures')
    dassh.plot.plot_all(inp, r)
    dassh_logger.log(_log_info, 'DASSH_PLOT execution complete')


def integrate_pin_power(args=None):
    """Set up DASSH Reactor object, integrate pin power, and write to CSV"""
    # Get input file from command line arguments
    parser = argparse.ArgumentParser(description='Process DASSH cmd')
    parser.add_argument('inputfile',
                        metavar='inputfile',
                        help='The input file to run with DASSH')
    parser.add_argument('--save_reactor',
                        action='store_true',
                        help='Save DASSH Reactor object after sweep')
    args = parser.parse_args(args)

    # Initiate logger
    print(dassh._ascii._ascii_title)
    in_path = os.path.split(args.inputfile)[0]
    dassh_logger = dassh.logged_class.init_root_logger(in_path, 'dassh_power')

    # Pre-processing
    # Read input file and set up DASSH input object
    dassh_logger.log(_log_info, f'Reading input: {args.inputfile}')
    dassh_input = dassh.DASSHPower_Input(args.inputfile)

    # Initialize the Reactor object
    reactor = dassh.Reactor(dassh_input, write_output=False)

    # Generate pin power distributions
    dassh_logger.log(_log_info, 'Generating pin power distributions...')
    asm_ids = []
    n_pins = []
    integrated_pin_powers = []
    for a in reactor.assemblies:
        if a.has_rodded:
            asm_ids.append(a.id)
            n_pins.append(a.rodded.n_pin)
            integrated_pin_powers.append(
                dassh.power._integrate_pin_power(a.power))

    # Save reactor if desired
    dassh_logger.log(_log_info, 'Saving data')
    if args.save_reactor:
        if sys.version_info < (3, 7):
            handlers = dassh_logger.handlers[:]
            for handler in handlers:
                handler.close()
                dassh_logger.removeHandler(handler)
        reactor.save()
        if sys.version_info < (3, 7):
            dassh_logger = dassh.logged_class.init_root_logger(
                os.path.split(dassh_logger._root_logfile_path)[0],
                'dassh', 'a+')

    # Write distributions to CSV
    arr_to_write = np.zeros((max(n_pins) + 1, len(asm_ids)))
    arr_to_write[0] = asm_ids
    for col in range(arr_to_write.shape[1]):
        arr_to_write[1:(n_pins[col] + 1), col] = integrated_pin_powers[col]
    outpath = os.path.join(dassh_input.path, 'total_pin_power.csv')
    np.savetxt(outpath, arr_to_write, delimiter=',')
    dassh_logger.log(_log_info, 'DASSH_POWER execution complete')


if __name__ == '__main__':
    main()
