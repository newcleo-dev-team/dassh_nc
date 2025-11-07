########################################################################
"""
date: 2025-07-xx
author: fpepe
Methods for mixed convection axial regions; to be used within
Assembly objects
"""
########################################################################
import numpy as np
import logging
from dassh.region_rodded import RoddedRegion, calculate_ht_constants, \
    _setup_conduction_constants, _setup_convection_constants
from dassh.pin_model import PinModel
from typing import Tuple
from ._commons import GRAVITY_CONST, MIX_CON_VERBOSE_OUTPUT, MC_MAX_ITER


module_logger = logging.getLogger('dassh.region_mixed')


def make(inp, name, mat, fr, se2geo=False, update_tol=0.0, 
         mixed_convection_tol=1e-5):
    """Create RoddedRegion object within DASSH Assembly

    Parameters
    ----------
    inp : dict
        DASSH "Assembly" input dictionary
        dassh_input.data['Assembly'][...]
    name : str
        Name of this assembly (e.g. 'fuel')
    mat : dict
        Dictionary of DASSH material objects (coolant, duct)
    fr : float
        Coolant mass flow rate [kg/s]
    se2geo (optional) : boolean
        Indicate whether to use the (incorrect) geometry setup from
        SE2ANL. Note: this was created for developer comparison with
        SE2ANL and is not intended for general use (default=False)
    update_tol (optional) : float
        Tolerance in the change in coolant temp-dependent material
        properties that triggers correlated parameter recalculation
        (default=0.0; recalculate at every step)
    mixed_convection_tol (optional) : float
        Tolerance for the mixed convection region solver (default=1e-5)

    Returns
    -------
    DASSH MixedRegion object

    """
    rr = MixedRegion(name, inp['num_rings'], inp['pin_pitch'], 
                     inp['pin_diameter'], inp['wire_pitch'], 
                     inp['wire_diameter'], inp['clad_thickness'], 
                     inp['duct_ftf'], inp['verbose'], 
                     inp['approx_star_quantities'], fr, mat['coolant'],
                     mat['duct'], inp['htc_params_duct'], inp['corr_friction'],
                     inp['corr_flowsplit'], inp['corr_mixing'], 
                     inp['corr_nusselt'], inp['corr_shapefactor'],
                     inp['SpacerGrid'], inp['bypass_gap_flow_fraction'],
                     inp['bypass_gap_loss_coeff'], inp['wire_direction'], 
                     inp['shape_factor'], se2geo, update_tol,
                     mixed_convection_tol)

    # Add z lower/upper boundaries
    rr.z = [inp['AxialRegion']['rods']['z_lo'],
            inp['AxialRegion']['rods']['z_hi']]

    # Add fuel pin model, if requested
    if 'FuelModel' in inp.keys():
        if inp['FuelModel']['htc_params_clad'] is None:
            p2d = inp['pin_pitch'] / inp['pin_diameter']
            inp['FuelModel']['htc_params_clad'] = \
                [p2d**3.8 * 0.01**0.86 / 3.0,
                 0.86, 0.86, 4.0 + 0.16 * p2d**5]
        rr.pin_model = PinModel(inp['pin_diameter'],
                                inp['clad_thickness'],
                                mat['clad'],
                                fuel_params=inp['FuelModel'],
                                gap_mat=mat['gap'])
    elif 'PinModel' in inp.keys():
        if inp['PinModel']['htc_params_clad'] is None:
            p2d = inp['pin_pitch'] / inp['pin_diameter']
            inp['PinModel']['htc_params_clad'] = \
                [p2d**3.8 * 0.01**0.86 / 3.0,
                 0.86, 0.86, 4.0 + 0.16 * p2d**5]
        inp['PinModel']['pin_material'] = \
            [x.clone() for x in mat['pin']]
        rr.pin_model = PinModel(inp['pin_diameter'],
                                inp['clad_thickness'],
                                mat['clad'],
                                pin_params=inp['PinModel'],
                                gap_mat=mat['gap'])
    else:
        pass

    if hasattr(rr, 'pin_model'):
        # Only the last 6 columns are for data:
        # (local avg coolant temp, clad OD/MW/ID, fuel OD/CL);
        # The first 4 columns are for identifying stuff:
        # (id, z (remains blank), pin number)
        rr.pin_temps = np.zeros((rr.n_pin, 9))
        # Fill with pin numbers
        rr.pin_temps[:, 2] = np.arange(0, rr.n_pin, 1)
    return rr

class MixedRegion(RoddedRegion):
    """Class to represent a rodded region with mixed convection
    
    Parameters
    ----------
    name : str
        Name of the region
    n_ring : int
        Number of rings of pins in the assembly
    pin_pitch : float
        Fuel pin center-to-center pin_pitch distance (m)
    pin_diam : float
        Outer diameter of the fuel pin cladding (m)
    wire_pitch : float
        Wire wrap pitch (m)
    wire_diam : float
        Wire wrap diameter (m)
    clad_thickness : float
        Cladding thickness (m)
    duct_ftf : list
        List of tuples containing the inner and outer duct 
        flat-to-flat distances for each duct surrounding the bundle
    verbose : boolean
        Indicate whether to print verbose output during iterations
    approx_star_quantities : boolean
        Indicate whether to approximate star quantities
    flow_rate : float
        Coolant mass flow rate (kg/s)
    coolant_mat : DASSH Material
        Coolant Material object
    duct_mat : DASSH Material
        Duct Material object
    htc_params_duct : list
        List of heat transfer coefficient parameters for duct
    corr_friction : str {'NOV', 'REH', 'ENG', 'CTD', 'CTS', 'UCTD'}
        Correlation for bundle friction factor; "CTD" is recommended.
    corr_flowsplit : str {'NOV', 'MIT', 'CTD', 'UCTD'}
        Correlation for subchannel flow split; "CTD" is recommended
    corr_mixing : str {'MIT', 'CTD'}
        Correlation for subchannel mixing params; "CTD" is recommended
    corr_nusselt : str (optional) {'DB'}
        Correlation for Nu; "DB" (Dittus-Boelter) is recommended
    corr_shapefactor : str (optional) {'CT'}
        Correlation for conduction shape factor
    spacer_grid : dict (optional) {None}
        Input parameters for spacer grid pressure losses
    byp_ff : float (optional)
        Unused
    byp_k : float (optional)
        Unused
    wwdir : str {'clockwise', 'counterclockwise'}
        Wire wrap direction
    sf : float (optional)
        Shape factor multiplier on coolant material thermal conductivity
    se2 : bool
        Indicate whether to use DASSH or SE2 bundle geometry definitions
        (use only when comparing DASSH and SE2)
    param_update_tol (optional) : float
        Fractional change in material properties required to trigger
        correlation recalculation
    mixed_convection_tol (optional) : float
        Tolerance for the mixed convection region solver
    """
    def __init__(self, name, n_ring, pin_pitch, pin_diam, wire_pitch,
                 wire_diam, clad_thickness, duct_ftf, verbose, 
                 approx_star_quantities, flow_rate,
                 coolant_mat, duct_mat, htc_params_duct, corr_friction,
                 corr_flowsplit, corr_mixing, corr_nusselt,
                 corr_shapefactor, spacer_grid=None, byp_ff=None,
                 byp_k=None, wwdir='clockwise', sf=1.0, se2=False,
                 param_update_tol=0.0, mixed_convection_tol=1e-5):
        """Instantiate MixedRegion object"""
        # Instantiate RoddedRegion object
        RoddedRegion.__init__(self, name, n_ring, pin_pitch, pin_diam,
                              wire_pitch, wire_diam, clad_thickness,
                              duct_ftf, flow_rate, True, coolant_mat, 
                              duct_mat, htc_params_duct, corr_friction,
                              corr_flowsplit, corr_mixing, corr_nusselt,
                              corr_shapefactor, spacer_grid, byp_ff,
                              byp_k, wwdir, sf, se2, param_update_tol,
                              rad_isotropic=False)
                 
        # Variables of interest are:
        # - SC velocities and their variations over dz
        # - SC densities and their variations over dz
        # - Bundle pressure drop and their variations over dz
        
        self._pressure_drop = 0.0    # This overrides the attribute in RoddedRegion
        # New attributes for the densities and velocities for the time being. 
        # In principle we already have self.sc_properties['density'] and 
        # self.coolant_int_params['sc_vel'] so this is not very smart... 
        # We should decide wether to use these or the old ones.
        self._delta_P = 0
        self._delta_v = 0*np.ones(self.subchannel.n_sc['coolant']['total'])
        self._delta_rho = np.ones(self.subchannel.n_sc['coolant']['total'])
        # Flag to indicate whether to track iteration convergence or not 
        self._verbose = verbose
        # Tolerance for mixed convection solver and star quantities calculation
        self._mixed_convection_tol = mixed_convection_tol
        self._approx_star_quantities = approx_star_quantities
        # Initialize enthalpy array
        self._enthalpy = \
            self.coolant.enthalpy_from_temp(self.coolant.temperature) * \
                np.ones(self.subchannel.n_sc['coolant']['total'])


    ####################################################################
    # TEMPERATURE AND PRESSURE DROP CALCULATION
    ####################################################################
    def calculate(self, dz, z, q, t_gap, h_gap, adiab=False, ebal=False) \
        -> None:
        """Calculate new coolant and duct temperatures and pressure
        drop across axial step

        Parameters
        ----------
        dz : float
            Axial step size (m)
        z : float
            Axial position (m)
        q : dict
            Power (W/m) generated in pins, duct, and coolant
        t_gap : numpy.ndarray
            Interassembly gap temperatures around the assembly at the
            j+1 axial level (array length = n_sc['duct']['total'])
        h_gap : float
            Heat transfer coefficient for convection between the gap
            coolant and outer duct wall based on core-average inter-
            assembly gap coolant temperature
        adiab : boolean
            Indicate whether outer duct has adiabatic BC
        ebal : boolean
            Indicate whether to track energy balance
        """
        # Duct temperatures: calculate with new coolant properties
        self._calc_duct_temp(q['duct'], t_gap, h_gap, adiab) 
        # Solve the system of equations for the SC
        # velocity and density variations and bundle pressure drop
        self._solve_system(dz, z, q['pins'], q['cool'], ebal)
        # Update coolant temperatures from enthalpy
        self.temp['coolant_int'] = \
            self.coolant.convert_properties(enthalpy=self._enthalpy) 
        # Update coolant properties
        self._update_coolant_int_params(self.avg_coolant_int_temp)
        # Bypass coolant temperatures
        if self.n_bypass > 0:
            if self.byp_flow_rate > 0:
                self.temp['coolant_byp'] += \
                    self._calc_coolant_byp_temp(dz, ebal)
            else:
                self.temp['coolant_byp'] += \
                    self._calc_coolant_byp_temp_stagnant(dz, ebal)
            # Update bypass coolant properties 
            self._update_coolant_byp_params(self.avg_coolant_byp_temp)
        
        
    ####################################################################
    # COOLANT TEMPERATURE  AND PRESSURE CALCULATION METHODS
    ####################################################################
    def _solve_system(self, dz: float, z: float, q_pins: np.ndarray, 
                      q_cool: np.ndarray, ebal: bool) -> None:
        """
        Method to solve the system
        
        Parameters
        ----------
        dz : float
            Axial step size (m)
        z : float
            Axial position (m)
        q_pins : np.ndarray
            Power (W/m) generated in pins
        q_cool : np.ndarray
            Power (W/m) generated in coolant
        ebal : bool
            Indicate whether to track energy balance
        """
        # Use previous step deltas as initial guesses
        delta_v0 = self._delta_v.copy()
        delta_rho0 = self._delta_rho.copy()
        delta_P0 = self._delta_P
        # Calculate power added to coolant
        qq = self._calc_int_sc_power(q_pins, q_cool)
        # Build known vector
        bb = self._build_vector(qq, dz, z)
        # Calculate initial RR using guess `delta_rho0`
        RR = self._calc_RR(delta_rho0)  
        # Verbose output header
        if self._verbose:
            self.log('info', MIX_CON_VERBOSE_OUTPUT[0])
            self.log('info', MIX_CON_VERBOSE_OUTPUT[1])
        # Iterate to solve the non-linear system
        iter = 0
        err_vect = np.ones(3) # Max. errors on delta_rho and delta_v, and
                              # error on delta_P
        while np.any(err_vect) > self._mixed_convection_tol \
            and iter < MC_MAX_ITER:
            # Build matrix
            AA = self._build_matrix(dz, delta_v0, delta_rho0, RR)
            # Solve system
            xx = np.linalg.solve(AA, bb)
            # Extract deltas from solution vector
            delta_rho = xx[0:2*self.subchannel.n_sc['coolant']['total']:2]
            delta_v = xx[1:2*self.subchannel.n_sc['coolant']['total']:2]
            delta_P = xx[-1]
            # Calculate errors
            err_vect[0] = np.max(np.abs(delta_rho - delta_rho0))
            err_vect[1] = np.max(np.abs(delta_v - delta_v0))
            err_vect[2] = np.max(np.abs(delta_P - delta_P0))
            # Verbose output iteration info
            if self._verbose:
                self.log('info', f'{iter+1}       {err_vect[0]:.6e}' + \
                    f'       {err_vect[1]:.6e}       {err_vect[2]:.6e}')
            # Update guesses for next iteration
            delta_v0 =    delta_v.copy()
            delta_rho0 =  delta_rho.copy()
            delta_P0 =    delta_P
            # Recalculate RR
            RR = self._calc_RR(delta_rho)
            # Update iteration counter
            iter += 1
        # Update deltas with converged values
        self._delta_v = delta_v.copy()
        self._delta_rho = delta_rho.copy()
        self._delta_P = delta_P
        # Update state variables
        self._sc_vel += delta_v
        self._density += delta_rho
        self._pressure_drop -= delta_P
        self._delta_h = RR * self._delta_rho
        self._enthalpy += self._delta_h
        # Update energy balance if requested
        # Calculated as:
        # Q_in [from z to z+dz] - (m*delta_h)_(z+dz) + (m*delta_h)_(z) = err
        if ebal:                          
            enthalpy_old = self._enthalpy - self._delta_h
            mfr_old = (self._density - self._delta_rho) * \
                (self._sc_vel - self._delta_v) * \
                self.params['area'][self.subchannel.type[
                    :self.subchannel.n_sc['coolant']['total']]]
            mcpdT_i = self.sc_mfr * self._enthalpy - mfr_old * enthalpy_old
            self.update_ebal(dz*np.sum(qq), 0, mcpdT_i)
            
            # Error introduced in the energy balance by h_star approximation
            # delta_m = self.sc_mfr - mfr_old
            # error_introduced = self._hstar * delta_m
            

    def _build_vector(self, qq: np.ndarray, dz: float, z: float) -> np.ndarray:
        """
        Build the known vector for the system of equations
        
        Parameters
        ----------
        qq : np.ndarray
            Power (W/m) added to the coolant 
        dz : float
            Axial step size (m)
        z : float
            Axial position (m)

        Returns
        -------
        bb : np.ndarray
            Known vector for the system of equations
        """
        # Number of coolant subchannels
        nn = self.subchannel.n_sc['coolant']['total']
        # Calculate MEX, EEX and GG terms
        EEX, MEX = self._calc_EEX_MEX(dz)
        GG = - self._density * (GRAVITY_CONST * dz + dz * 
                                self.coolant_int_params['ff_i'] * 
                                self._sc_vel**2 / 2 / 
                                self.params['de'][self.subchannel.type[:nn]])
        # Build energy terms of the known vector
        energy_b = qq * dz / self.params['area'][self.subchannel.type[:nn]] \
            + EEX
        # Wall convection term
        energy_b[self.ht['conv']['ind']] += self._wall_convection()
        # Build momentum terms of the known vector
        momentum_b = GG + MEX 
        if 'grid' in self.corr_constants.keys():
            momentum_b += self.params['area'][self.subchannel.type[:nn]] * \
                          self.calculate_spacergrid_pressure_drop(z, dz)
        # Assemble known vector
        bb = np.zeros(2*nn + 1)
        bb[1:2*nn:2] = energy_b
        bb[0:2*nn:2] = momentum_b
        return bb


    def _wall_convection(self) -> np.ndarray:
        """
        Calculate convection between edge/corner subchannels and duct wall
        
        Returns
        -------
        dT_conv_over_R : np.ndarray
            Temperature difference over the resistance between coolant
            and duct wall (K)
        """
        # CONVECTION BETWEEN EDGE/CORNER SUBCHANNELS AND DUCT WALL
        # Heat transfer coefficient
        htc_coeff = self.coolant_int_params['sc_htc'][self.ht['conv']['ind']]

        # Low flow case: use SE2ANL model
        if self._conv_approx:
            # Resistance between coolant and duct MW
            # self.duct.update(self.avg_duct_mw_temp[0])
            self._update_duct(self.avg_duct_mw_temp[0])
            # R1 = 1 / h; R2 = dw / 2 / k (half wall thickness over k)
            # R units: m2K / W; heat transfer area included in const
            R1 = 1 / htc_coeff  # R1 = 1 / h
            R2 = 0.5 * self.d['wall'][0] / self.duct.thermal_conductivity
            dT_conv_over_R = \
                ((self.temp['duct_mw'][0, self.ht['conv']['adj']]
                  - self.temp['coolant_int'][self.ht['conv']['ind']])
                 / (R1 + R2))
        else:
            dT_conv_over_R = htc_coeff * \
                (self.temp['duct_surf'][0, 0, self.ht['conv']['adj']]
                 - self.temp['coolant_int'][self.ht['conv']['ind']])

        return self.ht['conv']['const'] * dT_conv_over_R


    def _calc_EEX_MEX(self, dz: float) -> np.ndarray:
        """
        Calculate the inter-channel energy and momentum exchange terms 
        
        Parameters
        ----------
        dz : float
            Axial step size (m)

        Returns
        -------
        EEX : np.ndarray
            Energy exchange term between adjacent subchannels
        MEX : np.ndarray
            Momentum exchange term between adjacent subchannels
        """
        # Instantiate exchange arrays
        ene_exchange = np.zeros((self.subchannel.n_sc['coolant']['total'], 3))
        mom_exchange = np.zeros((self.subchannel.n_sc['coolant']['total'], 3))
        # Iterate over subchannels and adjacent subchannels
        for i in range(self.subchannel.n_sc['coolant']['total']):
            # Loop over adjacent subchannels
            for k in range(3):
                j = self.ht['cond']['adj'][i][k]
                # If `i` is a corner subchannel and `j` is also a corner 
                # subchannel (`k==2`), skip the calculation
                if i in self.ht['conv']['ind'][self.ht['conv']['type'] == 2] \
                    and k == 2:
                    continue
                # Calculate mass flow averaged properties between SCs
                rho_ij = self._calc_mass_flow_average_property(
                    'density', i, j)
                cp_ij = self._calc_mass_flow_average_property(
                    'heat_capacity', i, j)
                k_ij = self._calc_mass_flow_average_property(
                    'thermal_conductivity', i, j)
                # Eddy diffusivity coefficient is given as adimensional 
                # coefficient. It has to be multiplied by velocity to get 
                # the correct units (m/s)
                WW_ij = self.coolant_int_params['eddy'] *  rho_ij 
                # Calculate energy and momentum exchange terms
                # Eddy diffusivity + conduction for energy exchange
                ene_exchange[i][k] = \
                    (WW_ij + self._sf * k_ij / cp_ij) * \
                        (self.ht['cond']['const'][i][k] 
                            * (self._enthalpy[j] - self._enthalpy[i]))
                # Eddy diffusivity for momentum exchange
                mom_exchange[i][k] = \
                    WW_ij * (self.ht['cond']['const'][i][k]
                                * (self._sc_vel[j] - self._sc_vel[i]))
        # Sum over adjacent subchannels
        EEX = (ene_exchange[:, 0] + ene_exchange[:, 1] + ene_exchange[:, 2])
        MEX = (mom_exchange[:, 0] + mom_exchange[:, 1] + mom_exchange[:, 2])
        
        # Swirl mixing term
        swirl_consts = self.d['pin-wall'] * self.coolant_int_params['swirl']
        swirl_consts = swirl_consts[self.ht['conv']['type']]
        # Calculate swirl energy and momentum exchange terms
        swirl_energy = swirl_consts * \
                    (self._density[self.subchannel.sc_adj[
                        self.ht['conv']['ind'], self._adj_sw]] 
                     * self._enthalpy[self.subchannel.sc_adj[
                         self.ht['conv']['ind'], self._adj_sw]]
                     - self._density[self.ht['conv']['ind']]
                     * self._enthalpy[self.ht['conv']['ind']])       
        swirl_momentum = swirl_consts * \
                    (self._density[self.subchannel.sc_adj[
                        self.ht['conv']['ind'], self._adj_sw]] 
                     * self._sc_vel[self.subchannel.sc_adj[
                         self.ht['conv']['ind'], self._adj_sw]]
                     - self._density[self.ht['conv']['ind']]
                     * self._sc_vel[self.ht['conv']['ind']])
        # Add swirl terms to total exchange terms, and multiply by dz/area            
        EEX[self.ht['conv']['ind']] += swirl_energy
        EEX *= dz / self.params['area'][self.subchannel.type[
            :self.subchannel.n_sc['coolant']['total']]]
        
        MEX[self.ht['conv']['ind']] += swirl_momentum
        MEX *= dz/self.params['area'][self.subchannel.type[
            :self.subchannel.n_sc['coolant']['total']]]
        return EEX, MEX


    def _build_matrix(self, dz: float, delta_v: np.ndarray,
                      delta_rho: np.ndarray, RR: np.ndarray) -> np.ndarray:
        """
        Build the matrix for the system of equations

        Parameters
        ----------
        dz : float
            Axial step size (m)
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
        delta_rho : np.ndarray
            Variation of the SC densities (kg/m^3)
        RR : np.ndarray
            Enthalpy variation coefficient (J/kg/K)

        Returns
        -------
        AA : np.ndarray
            Coefficient matrix for the system of equations
        """
        hstar, vstar = self._calc_h_v_star(delta_v, delta_rho, RR)
        self.hstar = hstar  # Store hstar for energy balance error calc
        self.vstar = vstar  # Store vstar for momentum balance error calc
        nn = self.subchannel.n_sc['coolant']['total']
        # Calculate coefficients for the matrix
        EE, FF = self._calc_momentum_coefficients(nn, dz, delta_v, vstar)
        SS, TT = self._calc_energy_coefficients(delta_v, delta_rho, hstar, RR)
        C_rho, C_v = self._calc_continuity_coefficients(nn, delta_v)
        # Build matrix
        AA = np.zeros((2*nn + 1, 2*nn + 1))
            
        diag = np.zeros(2*nn + 1)
        sup_diag = np.zeros(2*nn)
        sub_diag = np.zeros(2*nn)
    
        diag[0:2*nn:2] = EE
        diag[1:2*nn:2] = TT
        sup_diag[0:2*nn:2] = FF
        sub_diag[0:2*nn:2] = SS
            
        AA += np.diag(diag) + np.diag(sup_diag, k=1) + np.diag(sub_diag, k=-1)
        AA[0:-2:2, -1] = 1
        AA[-1, 0:2*nn:2] = C_rho
        AA[-1, 1:2*nn:2] = C_v
        return AA


    def _calc_h_v_star(self, delta_v: np.ndarray, delta_rho: np.ndarray, 
                      RR: np.ndarray) -> Tuple[np.ndarray]:
        """
        Calculate hstar and vstar
        
        Parameters
        ----------
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
        delta_rho : np.ndarray
            Variation of the SC densities (kg/m^3)
        RR : np.ndarray
            Enthalpy variation coefficient (J/kg/K)
            
        Returns
        -------
        hstar : np.ndarray
            Effective enthalpy transported from adjacent subchannels (J/kg)
        vstar : np.ndarray
            Effective velocity transported from adjacent subchannels (m/s)
            
        Notes
        -----
        Two options are available:
        1) Approximate hstar and vstar as the midpoint values of enthalpy
           and velocity (i.e., at z + dz/2) `h_mid` and `v_mid`
        2) Calculate hstar and vstar as the mass-flow-weighted average
           of enthalpy and velocity from adjacent subchannels
        """
        h_mid = self._enthalpy + RR * delta_rho / 2
        v_mid = self._sc_vel + delta_v / 2
        # OPTION 1: Approximate hstar and vstar as midpoints
        if self._approx_star_quantities:
            return h_mid, v_mid
        # OPTION 2: Calculate hstar and vstar as mass-flow-weighted averages
        # Initialize arrays
        nn = self.subchannel.n_sc['coolant']['total']
        numerator_h = np.zeros((nn, 3))
        numerator_v = np.zeros((nn, 3))
        denominator = np.zeros((nn, 3))
        xij = np.zeros((nn, 3))
        # Calculate delta_m for each subchannel
        vrho_1 = self._density*self._sc_vel 
        vrho_2 = (self._density + delta_rho) * \
            (self._sc_vel + delta_v) 
        delta_m = (vrho_2 - vrho_1) * \
            self.params['area'][self.subchannel.type[:nn]]
        # Iterate over subchannels and adjacent subchannels
        for i in range(nn):
            for k in range(3):
                j = self.ht['cond']['adj'][i][k]
                if i in self.ht['conv']['ind'][self.ht['conv']['type'] == 2] \
                    and k == 2:
                    continue
                # Calculate delta_m difference between adjacent subchannel
                xij[i][k] = delta_m[i] - delta_m[j] + 1e-15
                # Calculate numerators and denominators
                numerator_h[i][k] = np.abs(xij[i][k]) * \
                    (h_mid[i] + h_mid[j]) - xij[i][k] * (h_mid[i] - h_mid[j])
                numerator_v[i][k] = np.abs(xij[i][k]) * \
                    (v_mid[i] + v_mid[j]) - xij[i][k] * (v_mid[i] - v_mid[j])
                denominator[i][k] = np.abs(xij[i][k])
        # Calculate hstar and vstar
        sum_den = 2 * (denominator[:, 0] + denominator[:, 1] + 
                       denominator[:, 2])
        hstar = (numerator_h[:, 0] + numerator_h[:, 1] + numerator_h[:, 2]) \
            / sum_den
        vstar = (numerator_v[:, 0] + numerator_v[:, 1] + numerator_v[:, 2]) \
            / sum_den
        return hstar, vstar


    def _calc_momentum_coefficients(self, nn: int, dz: float, 
                                    delta_v: np.ndarray, vstar: np.ndarray) \
                                        -> Tuple[np.ndarray]:
        """
        Calculate Ei and Fi coefficients for the momentum equation
        
        Parameters
        ----------
        nn : int
            Number of coolant subchannels
        dz : float
            Axial step size (m)
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
        vstar : np.ndarray
            Effective velocity transported from adjacent subchannels (m/s)
            
        Returns
        -------
        EE : np.ndarray
            Coefficients for the momentum equation
        FF : np.ndarray
            Coefficients for the momentum equation
        """
        EE = (self._sc_vel + delta_v) * (self._sc_vel + delta_v - vstar) + \
            GRAVITY_CONST * dz / 2 + self.coolant_int_params['ff_i'] * dz \
                / 16 / self.params['de'][self.subchannel.type[:nn]] * \
                    (2 * self._sc_vel + delta_v)**2 
        FF = self._density * ((2 + self.coolant_int_params['ff_i'] * dz / 2 / 
                               self.params['de'][self.subchannel.type[:nn]]) 
                              * self._sc_vel + 
                              (1 + self.coolant_int_params['ff_i'] * dz / 8 /
                               self.params['de'][self.subchannel.type[:nn]]) 
                              * delta_v - vstar)
        return EE, FF


    def _calc_energy_coefficients(self, delta_v: np.ndarray, 
                                  delta_rho: np.ndarray, hstar: np.ndarray, 
                                  RR: np.ndarray) -> Tuple[np.ndarray]:
        """
        Calculate coefficients for the energy equation.
        
        Parameters
        ----------
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
        delta_rho : np.ndarray
            Variation of the SC densities (kg/m^3)
        hstar : np.ndarray
            Average enthalpy between adjacent subchannels (J/kg)
        RR : np.ndarray
            Enthalpy variation coefficient (J/kg/K)
            
        Returns
        -------
        SS : np.ndarray
            Coefficients for the energy equation
        TT : np.ndarray
            Coefficients for the energy equation
        """
        SS = (self._sc_vel + delta_v) * \
                (self._enthalpy - hstar + RR * (self._density + delta_rho))
        TT = self._density * (self._enthalpy - hstar)
        return SS, TT


    def _calc_continuity_coefficients(self, nn: int, delta_v: np.ndarray) \
        -> Tuple[np.ndarray]:
        """
        Calculate coefficients for the continuity equation.
        
        Parameters
        ----------
        nn : int
            Number of coolant subchannels
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
            
        Returns
        -------
        Tuple[np.ndarray]
            C_rho : np.ndarray
                Coefficients for the continuity equation
            C_v : np.ndarray
                Coefficients for the continuity equation
        """
        AA = self.params['area'][self.subchannel.type[:nn]]
        return AA * (self._sc_vel + delta_v), AA * self._density
    
    
    def _calc_RR(self, drho: np.ndarray) -> np.ndarray:
        """
        Calculate the RR coefficient.
        
        Parameters
        ----------
        drho : np.ndarray
            Variation of the SC densities (kg/m^3)
            
        Returns
        -------
        RR : np.ndarray
            Enthalpy variation coefficient (J/kg/K)
        """
        return (self.coolant.convert_properties(density=self._density+drho) -
                self.coolant.convert_properties(density=self._density)) / drho


    def _init_static_correlated_params(self, t):
        """Calculate bundle friction factor and flowsplit parameters
        at the bundle-average temperature

        Parameters
        ----------
        t : float
            Bundle axial average temperature ((T_in + T_out) / 2)

        Returns
        -------
        None

        Notes
        -----
        This method is called within the '_setup_asm' method inside
        the Reactor object instantiation. It is very similar to
        '_update_coolant_int_params', found below.

        """
        # Update coolant material properties
        self._update_coolant(t)
        # Coolant axial velocity, bundle Reynolds number
        mfr_over_area = self.int_flow_rate / self.bundle_params['area']
        self.coolant_int_params['vel'] = mfr_over_area / self.coolant.density
        self.coolant_int_params['Re'] = mfr_over_area * \
            self.bundle_params['de'] / self.coolant.viscosity
        # Spacer grid, if present
        if 'grid' in self.corr_constants.keys():
            try:
                self.coolant_int_params['grid_loss_coeff'] = \
                    self.corr_constants['grid']['loss_coeff']
            except (KeyError, TypeError):
                self.coolant_int_params['grid_loss_coeff'] = \
                    self.corr['grid'](
                        self.coolant_int_params['Re'],
                        self.corr_constants['grid']['solidity'],
                        self.corr_constants['grid']['corr_coeff'])
        # Flow split parameters
        if self.corr['fs'] is not None:
            if 'grid' in self.corr.keys():
                self.coolant_int_params['fs'] = \
                    self.corr['fs'](self, grid=True)
            else:
                self.coolant_int_params['fs'] = \
                    self.corr['fs'](self)  
        self.coolant_int_params['Re_all_sc'] = \
            self.coolant_int_params['Re'] * self.coolant_int_params['fs'][
                self.subchannel.type[
                    :self.subchannel.n_sc['coolant']['total']]] / \
                        self.bundle_params['de'] * self.params['de'][
                            self.subchannel.type[
                                :self.subchannel.n_sc['coolant']['total']]]
        # Friction factor
        if self.corr['ff'] is not None:
            self.coolant_int_params['ff'] = self.corr['ff'](self)
        if self.corr['ff_i'] is not None:
            self.coolant_int_params['ff_i'] = self.corr['ff_i'](self)
        
        self._sc_vel = self.coolant_int_params['vel'] * \
            self.coolant_int_params['fs'][self.subchannel.type[
                :self.subchannel.n_sc['coolant']['total']]]
                
        self._density = self.coolant.density * np.ones(
            self.subchannel.n_sc['coolant']['total'])     
    
    
    def _setup_ht_constants(self):
        """Setup heat transfer constants in numpy arrays"""
        const = calculate_ht_constants(self, mixed=True)
        # self.ht_consts = const
        self.ht = {}
        self.ht['old'] = const
        self.ht['cond'] = _setup_conduction_constants(self, const)
        self.ht['conv'] = _setup_convection_constants(self, const)
        
    
    @property
    def pressure_drop(self):
        return self._pressure_drop
    
        
    @property
    def sc_mfr(self):
        """Return mass flow rate in each subchannel"""
        mfr = self._density * self._sc_vel * self.params['area'][
            self.subchannel.type[:self.subchannel.n_sc['coolant']['total']]]
        return mfr    
