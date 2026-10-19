import os
import sys
import logging
import dassh
_log_info = 20  # logging levels must be int


def run_dassh(dassh_input, rx_args):
    """Run DASSH without orificing optimization"""
    # For each timestep in the DASSH input, create the necessary DASSH
    # DASSH objects, run DASSH, and process the results
    dassh_logger = logging.getLogger('dassh')
    need_subdir = False
    if dassh_input.timepoints > 1:
        need_subdir = True
        if dassh_input.data['Setup']['parallel']:
            import multiprocessing as mp
            if dassh_input.data['Setup']['n_cpu'] is not None:
                n_procs = dassh_input.data['Setup']['n_cpu']
            else:
                n_procs = min((mp.cpu_count(), dassh_input.timepoints))
            pool = mp.Pool(processes=n_procs)
            workers = []

    for i in range(dassh_input.timepoints):
        working_dir = None
        if need_subdir:
            # Only log info about timestep if you have multiple
            dassh_logger.log(_log_info, f'Timestep {i + 1}')
            working_dir = os.path.join(
                dassh_input.path, f'timestep_{i + 1}')
        # Set up working dirs, run DASSH, write output, make plots
        if dassh_input.data['Setup']['parallel']:
            workers.append(
                pool.apply_async(
                    _run_dassh,
                    args=(dassh_input,
                          rx_args,
                          i,
                          working_dir, )
                )
            )
        else:
            _run_dassh(dassh_input, rx_args, i, working_dir)

    # Clean up from parallel execution, if applicable
    if dassh_input.data['Setup']['parallel']:
        for w in workers:
            w.get()
        pool.terminate()
        pool.close()
        pool.join()


def _run_dassh(dassh_inp, args, timestep, wdir, link=None):
    """Run DASSH for a single timestep

    Parameters
    ----------
    dassh_inp : DASSH_Input object
        Base DASSH input class
    args : dict
        Various args for instantiating DASSH objects
    timestep : int
        Timestep for which to run DASSH
    wdir : str
        Path to working directory for this timestep
    link : str (optional)
        Try to link VARPOW output files from another path
        Avoids repetitive calcs in orificing optimization
        (default = None; run VARPOW as usual)

    """
    dassh_logger = logging.getLogger('dassh')
    # Try to link VARPOW output from another source. If it doesn't
    # exist or work, just rerun VARPOW.
    if link is not None:
        files_linked = 0
        for f in ['varpow_MatPower.out',
                  'varpow_MonoExp.out',
                  'VARPOW.out']:
            src = os.path.join(link, f)
            dest = os.path.join(wdir, f)
            if os.path.exists(src):
                os.symlink(src, dest)
                files_linked += 1
            else:
                break
        # If all VARPOW files were linked, can skip VARPOW calculation
        if files_linked == 3:
            args['no_power_calc'] = False  # if linked, skip VARPOW
        else:
            args['no_power_calc'] = True

    # Initialize the Reactor object
    reactor = dassh.Reactor(dassh_inp,
                            calc_power=args['no_power_calc'],
                            path=wdir,
                            timestep=timestep,
                            write_output=True)
    # Perform the sweep
    dassh_logger.log(_log_info, 'Performing temperature sweep...')
    reactor.temperature_sweep(verbose=args['verbose'])
    reactor.postprocess()

    # Post-processing: write output, save reactor if desired
    dassh_logger.log(_log_info, 'Temperature sweep complete')
    if args['save_reactor'] or dassh_inp.data['Plot']:
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
    dassh_logger.log(_log_info, 'Output written')

    # Post-processing: generate figures, if desired
    if ('Plot' in dassh_inp.data.keys()
            and len(dassh_inp.data['Plot']) > 0):
        dassh_logger.log(_log_info, 'Generating figures')
        dassh.plot.plot_all(dassh_inp, reactor)
    return dassh_logger

