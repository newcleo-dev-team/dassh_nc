########################################################################
"""
date: 2025-xx-xx
author: fpepe
Methods for mixed convection axial regions; to be used within Assembly objects
"""
########################################################################
import numpy as np
from dassh.region_rodded import RoddedRegion, calculate_ht_constants, \
    setup_conduction_constants, setup_convection_constants, \
        specify_region_details
from dassh._commons import GRAVITY_CONST, MIX_CON_VERBOSE_OUTPUT, \
    MC_MAX_ITER, MIXED_CONV_PROP_TO_UPDATE
import sys
import scipy.sparse as sp


def make(inp, name, mat, fr, se2geo=False, update_tol=0.0, 
         mixed_convection_rel_tol=1e-3):
    """Create MixeddRegion object within DASSH Assembly

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
    mixed_convection_rel_tol (optional) : float
        Relative tolerance for the mixed convection region solver (default=1e-3)

    Returns
    -------
    DASSH MixedRegion object

    """
    rr = MixedRegion(name, inp['num_rings'], inp['pin_pitch'], 
                     inp['pin_diameter'], inp['wire_pitch'], 
                     inp['wire_diameter'], inp['clad_thickness'], 
                     inp['duct_ftf'], inp['verbose'], 
                     inp['accurate_star_quantities'], fr, mat['coolant'],
                     mat['duct'], inp['htc_params_duct'], inp['corr_friction'],
                     inp['corr_flowsplit'], inp['corr_mixing'], 
                     inp['corr_nusselt'], inp['corr_shapefactor'],
                     inp['SpacerGrid'], inp['bypass_gap_flow_fraction'],
                     inp['bypass_gap_loss_coeff'], inp['wire_direction'], 
                     inp['shape_factor'], se2geo, update_tol,
                     mixed_convection_rel_tol)
    return specify_region_details(rr, inp, mat)


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
    accurate_star_quantities : boolean
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
    mixed_convection_rel_tol (optional) : float
        Tolerance for the mixed convection region solver
    """
    _enthalpy: np.ndarray
    _delta_P: float
    _delta_v: np.ndarray
    _delta_rho: np.ndarray
    _hstar: np.ndarray
    _vstar: np.ndarray

    def __init__(self, name, n_ring, pin_pitch, pin_diam, wire_pitch,
                 wire_diam, clad_thickness, duct_ftf, verbose, 
                 accurate_star_quantities, flow_rate,
                 coolant_mat, duct_mat, htc_params_duct, corr_friction,
                 corr_flowsplit, corr_mixing, corr_nusselt,
                 corr_shapefactor, spacer_grid=None, byp_ff=None,
                 byp_k=None, wwdir='clockwise', sf=1.0, se2=False,
                 param_update_tol=0.0, mixed_convection_rel_tol=1e-3):
        # Instantiate RoddedRegion object
        super(MixedRegion, self).__init__(name, n_ring, pin_pitch, pin_diam,
                                          wire_pitch, wire_diam, 
                                          clad_thickness, duct_ftf, flow_rate, 
                                          True, coolant_mat, duct_mat,
                                          htc_params_duct, corr_friction, 
                                          corr_flowsplit, corr_mixing, 
                                          corr_nusselt, corr_shapefactor, 
                                          spacer_grid, byp_ff, byp_k, wwdir, 
                                          sf, se2, param_update_tol, 
                                          rad_isotropic=False)

        self._pressure_drop = 0.0 # This overrides the attribute in RoddedRegion
        self._delta_P = 1.0 # Guess on pressure drop
        self._delta_v = 0.1 * \
            np.ones(self.subchannel.n_sc['coolant']['total']) # Guess on velocity variation
        self._delta_rho = np.ones(self.subchannel.n_sc['coolant']['total']) # Guess on density variation
        # Flag to indicate whether to track iteration convergence or not 
        self._verbose = verbose
        # Tolerance for mixed convection solver and star quantities calculation
        self._mixed_convection_rel_tol = mixed_convection_rel_tol
        self._accurate_star_quantities = accurate_star_quantities
        # Initialize star quantities
        self._hstar = np.zeros_like(self._delta_v)
        self._vstar = np.zeros_like(self._delta_v)
        self.sc_properties['density'] = self.coolant.density * \
            np.ones(self.subchannel.n_sc['coolant']['total']) 
        # Initialize enthalpy array
        self._enthalpy = self.coolant.convert_properties(
            density=self.sc_properties['density'])


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
            Axial position of the cell outlet (m)
        q : dict
            Power (W/m) generated in pins, duct, and coolant
        t_gap : numpy.ndarray
            Interassembly gap temperatures around the assembly 
            (array length = n_sc['duct']['total'])
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
        self._update_coolant_int_params(self.avg_coolant_int_temp, 
                                        sc_vel=self._sc_vel)
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
    # COOLANT TEMPERATURE AND PRESSURE CALCULATION METHODS
    ####################################################################
    def _solve_system(self, dz: float, z: float, q_pins: np.ndarray, 
                      q_cool: np.ndarray, ebal: bool) -> None:
        """
        System builder and solver in terms of delta_v, delta_rho, and delta_P
        
        Parameters
        ----------
        dz : float
            Axial step size (m)
        z : float
            Axial position of the cell outlet (m)
        q_pins : np.ndarray
            Power (W/m) generated in pins
        q_cool : np.ndarray
            Power (W/m) generated in coolant
        ebal : bool
            Indicate whether to track energy balance
        """
        # Number of coolant subchannels
        nn = self.subchannel.n_sc['coolant']['total']
        # Use previous step deltas as initial guesses
        delta_rho0, delta_v0, delta_P0 = \
            self._copy_solution(self._delta_rho, self._delta_v, self._delta_P)
        # Calculate power added to coolant
        qq = self._calc_int_sc_power(q_pins, q_cool)
        # Build known vector
        bb = self._build_vector(qq, dz, z, nn)
        # Calculate initial RR using guess `delta_rho0`
        RR = self._calc_RR(delta_rho0)  
        # Verbose output header
        if self._verbose:
            for msg in MIX_CON_VERBOSE_OUTPUT:
                self.log('info', msg)
        # Iterate to solve the system
        iter = 0
        err_vect = np.ones(3) # Max. errors on delta_rho and delta_v, and
                              # error on delta_P
        while np.any(err_vect > self._mixed_convection_rel_tol) \
            and iter < MC_MAX_ITER:
            # Build matrix
            AA = self._build_matrix(dz, delta_v0, delta_rho0, RR, nn)
            #AA = sp.csr_matrix(self._build_matrix(dz, delta_v0, delta_rho0, RR))
            # Solve system
            xx = np.linalg.solve(AA, bb)
            #xx = sp.linalg.spsolve(AA, bb)
            # Extract deltas from solution vector
            delta_rho = xx[0:2*nn:2]
            delta_v = xx[1:2*nn:2]
            delta_P = xx[-1]
            # Calculate errors
            new_solution = np.dstack((delta_rho, delta_v))
            old_solution = np.dstack((delta_rho0, delta_v0))
            err_vect[:2] = np.max(np.abs((new_solution - old_solution) 
                                         / old_solution), axis=1)
            err_vect[2] = np.abs((delta_P - delta_P0) / delta_P0)
            # Verbose output iteration info
            if self._verbose:
                self.log('info', f'{iter+1}       {err_vect[0]:.6e}' + \
                    f'       {err_vect[1]:.6e}       {err_vect[2]:.6e}')
            # Update guesses for next iteration
            delta_rho0, delta_v0, delta_P0 = \
                self._copy_solution(delta_rho, delta_v, delta_P)
            # Recalculate RR
            RR = self._calc_RR(delta_rho)
            # Update iteration counter
            iter += 1
        # Update deltas with converged values
        self._delta_rho, self._delta_v, self._delta_P = \
            self._copy_solution(delta_rho, delta_v, delta_P)
        # Store old mass flow rate and enthalpy if energy balance is tracked
        if ebal: 
            old_mfr = self.sc_mfr.copy()
            mdh_old = old_mfr * self._enthalpy.copy()
        # Update velocity, density, and pressure drop adding converged deltas
        self._sc_vel += delta_v
        self.sc_properties['density'] += delta_rho
        self._pressure_drop -= delta_P
        # Update enthalpy converting density
        self._enthalpy = self.coolant.convert_properties(
            density=self.sc_properties['density'])
        # Update energy balance if requested
        # Calculated as:
        # Q_in [from z to z+dz] - (m*delta_h)_(z+dz) + (m*delta_h)_(z) = err
        if ebal:
            mcpdT_i = self.sc_mfr * self._enthalpy - mdh_old
            # Error introduced in the energy balance by h_star approximation
            delta_m = self.sc_mfr - old_mfr
            star_error = self._hstar * delta_m
            self.update_ebal(dz*np.sum(qq), self._qw, mcpdT_i, star_error)
            
            
    def _copy_solution(self, drho: np.ndarray, dv: np.ndarray, 
                       dP: float) -> tuple[np.ndarray, float]:
        """
        Copy solution deltas
        
        Parameters
        ----------
        drho : np.ndarray
            Density variation (kg/m^3)
        dv : np.ndarray
            Velocity variation (m/s)
        dP : float
            Pressure drop (Pa)
            
        Returns
        -------
        tuple[np.ndarray, float]
            Copied density variation, velocity variation, and pressure drop
        """
        return drho.copy(), dv.copy(), dP
    
    
    def _build_vector(self, qq: np.ndarray, dz: float, z: float, 
                      nn: int) -> np.ndarray:
        """
        Build the vector of known terms
        
        Parameters
        ----------
        qq : np.ndarray
            Power (W/m) added to the coolant 
        dz : float
            Axial step size (m)
        z : float
            Axial position of the cell outlet (m)
        nn : int
            Number of coolant subchannels

        Returns
        -------
        bb : np.ndarray
            Array of known terms
        """
        # Calculate MEX, EEX and GG terms
        EEX, MEX = self._calc_EEX_MEX(dz, nn)
        GG = - self.sc_properties['density'] * dz * \
            (GRAVITY_CONST + self.coolant_int_params['ff_i'] * 
             self._sc_vel**2 / 2 / 
             self.params['de'][self.subchannel.type[:nn]])
        # Build energy terms of the known vector
        energy_b = qq * dz / self.params['area'][self.subchannel.type[:nn]] \
            + EEX
        # Wall convection term
        self._qw = self._wall_convection()
        energy_b[self.ht['conv']['ind']] += self._qw / self.params['area'][
            self.subchannel.type[:nn][self.ht['conv']['ind']]]
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
        np.ndarray
            Convection term for edge/corner subchannels
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


    def _calc_EEX_MEX(self, dz: float, nn: int) -> np.ndarray:
        """
        Calculate the inter-channel energy and momentum exchange terms 
        
        Parameters
        ----------
        dz : float
            Axial step size (m)
        nn : int
            Number of coolant subchannels

        Returns
        -------
        EEX : np.ndarray
            Energy exchange term between adjacent subchannels
        MEX : np.ndarray
            Momentum exchange term between adjacent subchannels
        """
        # Instantiate exchange arrays
        EEX = np.zeros(nn)
        MEX = np.zeros(nn)
        # Iterate over subchannels and adjacent subchannels
        for i in range(nn):
            # Loop over adjacent subchannels
            ene_exchange = 0.0
            mom_exchange = 0.0
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
                WW_ij = self.coolant_int_params['eddy'] * rho_ij 
                # Eddy diffusivity + conduction for energy exchange 
                # summed up over adj SCs
                ene_exchange += (WW_ij + self._sf * k_ij / cp_ij) * \
                    (self.ht['cond']['const'][i][k] 
                     * (self._enthalpy[j] - self._enthalpy[i]))
                # Eddy diffusivity for momentum exchange summed up over adj SCs
                mom_exchange += WW_ij * (self.ht['cond']['const'][i][k] 
                                         * (self._sc_vel[j] - self._sc_vel[i]))
            EEX[i] = ene_exchange
            MEX[i] = mom_exchange
        # Swirl mixing term constants
        swirl_consts = self.d['pin-wall'] * \
            self.coolant_int_params['swirl'][self.ht['conv']['type']]
        # Add swirl terms to total exchange terms, and multiply by dz/area   
        self._finalize(EEX, swirl_consts, nn, dz)
        self._finalize(MEX, swirl_consts, nn, dz, is_mom=True)
        return EEX, MEX


    def _finalize(self, EX: np.ndarray, swirl_consts: np.ndarray, 
                  nn: int, dz: float, is_mom: bool = False):
        """
        Finalize either energy or momentum exchange term by adding swirl term
        and multiplying by dz/area
        
        Parameters
        ----------
        EX : np.ndarray
            Energy or momentum exchange term between adjacent subchannels
        swirl_consts : np.ndarray
            Swirl exchange constants for edge/corner subchannels
        nn : int
            Number of coolant subchannels
        dz : float 
            Axial step size (m)
        is_mom : bool
            Indicate whether to calculate momentum (True) or energy (False)
        """
        EX[self.ht['conv']['ind']] += self._calc_swirl_term(swirl_consts, 
                                                            is_mom)
        EX *= dz / self.params['area'][self.subchannel.type[:nn]]

    
    def _calc_swirl_term(self, swirl_consts: np.ndarray, 
                         is_mom: bool = False) -> np.ndarray:
        """
        Calculate swirl exchange term for either energy or momentum equation
        
        Parameters
        ----------
        swirl_consts : np.ndarray
            Swirl exchange constants for edge/corner subchannels
        is_mom : bool
            Indicate whether to calculate momentum (True) or energy (False)
            
        Returns
        -------
        np.ndarray
            Swirl exchange term for energy or momentum equation
        """
        adj_ind = self.subchannel.sc_adj[self.ht['conv']['ind'], self._adj_sw]
        var = self._sc_vel if is_mom else self._enthalpy
        return swirl_consts * \
            (self.sc_properties['density'][adj_ind] * var[adj_ind] 
             - self.sc_properties['density'][self.ht['conv']['ind']]
             * var[self.ht['conv']['ind']])


    def _build_matrix(self, dz: float, delta_v: np.ndarray,
                      delta_rho: np.ndarray, RR: np.ndarray, 
                      nn: int) -> np.ndarray:
        """
        Build the matrix for the system to solve

        Parameters
        ----------
        dz : float
            Axial step size (m)
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
        delta_rho : np.ndarray
            Variation of the SC densities (kg/m^3)
        RR : np.ndarray
            Enthalpy variation coefficient (J*m^3/kg^2)
        nn : int
            Number of coolant subchannels

        Returns
        -------
        AA : np.ndarray
            Coefficient matrix for the system to solve
        """
        self._calc_h_v_star(delta_v, delta_rho, RR, nn)
        # Calculate coefficients for the matrix
        EE, FF = self._calc_momentum_coefficients(nn, dz, delta_v)
        SS, TT = self._calc_energy_coefficients(delta_v, delta_rho, RR)
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
        AA[0:-2:2,-1] = 1
        AA[-1,0:2*nn:2] = C_rho
        AA[-1,1:2*nn:2] = C_v
        return AA


    def _calc_h_v_star(self, delta_v: np.ndarray, delta_rho: np.ndarray, 
                       RR: np.ndarray, nn: int) -> None:
        """
        Update hstar and vstar
        
        Parameters
        ----------
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
        delta_rho : np.ndarray
            Variation of the SC densities (kg/m^3)
        RR : np.ndarray
            Enthalpy variation coefficient (J*m^3/kg^2)
        nn : int
            Number of coolant subchannels
            
        Notes
        -----
        Two options are available:
        1) Approximate hstar and vstar as the midpoint value of enthalpy
           and velocity (i.e., at z + dz/2) `h_mid` and `v_mid`
        2) Calculate hstar and vstar as per "Cheng, S.K., 1984. Constitutive 
           Correlations for wire-wrapped subchannel analysis under forced and
           mixed convection conditions (Ph.D. thesis). MIT."
        """
        h_mid = self._enthalpy + RR * delta_rho / 2
        v_mid = self._sc_vel + delta_v / 2
        # OPTION 1: Approximate hstar and vstar as midpoints
        if not self._accurate_star_quantities:
            self._hstar = h_mid
            self._vstar = v_mid
            return
        # OPTION 2: Calculate hstar and vstar as per Cheng 
        numerator_h = np.zeros(nn)
        numerator_v = np.zeros(nn)
        sum_den = np.zeros(nn)
        # Calculate delta_m for each subchannel
        vrho_1 = self.sc_properties['density'] * self._sc_vel 
        vrho_2 = (self.sc_properties['density'] + delta_rho) * \
            (self._sc_vel + delta_v) 
        delta_m = (vrho_2 - vrho_1) * \
            self.params['area'][self.subchannel.type[:nn]]
        # Iterate over subchannels and adjacent subchannels
        for i in range(nn):
            denominator = 0.0
            num_h = 0.0
            num_v = 0.0
            for k in range(3):
                j = self.ht['cond']['adj'][i][k]
                if i in self.ht['conv']['ind'][self.ht['conv']['type'] == 2] \
                    and k == 2:
                    continue
                # Calculate delta_m difference between adjacent subchannel
                xij = delta_m[i] - delta_m[j]
                # Calculate numerators and denominators
                num_h += self._calc_star_quantity_numerator(
                    h_mid[i], h_mid[j], xij)
                num_v += self._calc_star_quantity_numerator(
                    v_mid[i], v_mid[j], xij)
                denominator += np.abs(xij)
            numerator_h[i] = num_h
            numerator_v[i] = num_v
            sum_den[i] = denominator      
        # Calculate hstar and vstar
        sum_den = 2 * sum_den + sys.float_info.epsilon 
        self._hstar = numerator_h / sum_den
        self._vstar = numerator_v / sum_den
        
        
    def _calc_star_quantity_numerator(self, var_mid_i: float, var_mid_j: float,
                                      xij: float) -> float:
        """
        Calculate the numerator for hstar and vstar calculation
        
        Parameters
        ----------
        var_mid_i : float
            Midpoint value of the `var_mid_i` for subchannel i
            `var_mid_j` can be enthalpy or velocity
        var_mid_j : float
            Midpoint value of the `var_mid_j` for subchannel j
            `var_mid_j` can be enthalpy or velocity
        xij : float
            Difference in mass flow rate between subchannels i and j
        
        Returns
        -------
        float
            Numerator for hstar and vstar calculation
        """
        return np.abs(xij) * (var_mid_i + var_mid_j) \
            - xij * (var_mid_i - var_mid_j)
        
        
    def _calc_momentum_coefficients(self, nn: int, dz: float, 
                                    delta_v: np.ndarray) -> tuple[np.ndarray]:
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
            
        Returns
        -------
        Tuple[np.ndarray]
            Container of the two following np.ndarrays:
            
            - EE coefficients 
            - FF coefficients
        """
        EE = (self._sc_vel + delta_v) * \
            (self._sc_vel + delta_v - self._vstar) + GRAVITY_CONST * dz / 2 + \
                self.coolant_int_params['ff_i'] * dz / 16 / \
                    self.params['de'][self.subchannel.type[:nn]] * \
                        (2 * self._sc_vel + delta_v)**2 
        FF = self.sc_properties['density'] * (
            (2 + self.coolant_int_params['ff_i'] * dz / 2 / 
             self.params['de'][self.subchannel.type[:nn]]) * self._sc_vel + 
            (1 + self.coolant_int_params['ff_i'] * dz / 8 /
             self.params['de'][self.subchannel.type[:nn]]) * delta_v 
            - self._vstar)
        return EE, FF


    def _calc_energy_coefficients(self, delta_v: np.ndarray, 
                                  delta_rho: np.ndarray, 
                                  RR: np.ndarray) -> tuple[np.ndarray]:
        """
        Calculate coefficients for the energy equation.
        
        Parameters
        ----------
        delta_v : np.ndarray
            Variation of the SC velocities (m/s)
        delta_rho : np.ndarray
            Variation of the SC densities (kg/m^3)
        RR : np.ndarray
            Enthalpy variation coefficient (J*m^3/kg^2)
            
        Returns
        -------
        Tuple[np.ndarray]
            Container of the two following np.ndarrays:
            
            - SS coefficients 
            - TT coefficients
        """
        SS = (self._sc_vel + delta_v) * \
            (self._enthalpy - self._hstar + 
             RR * (self.sc_properties['density'] + delta_rho))
        TT = self.sc_properties['density'] * (self._enthalpy - self._hstar)
        return SS, TT


    def _calc_continuity_coefficients(self, nn: int, delta_v: np.ndarray) \
        -> tuple[np.ndarray]:
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
            Container of the two following np.ndarrays:
            
            - C_rho coefficients 
            - C_v coefficients
        """
        areas = self.params['area'][self.subchannel.type[:nn]]
        return areas * (self._sc_vel + delta_v), \
            areas * self.sc_properties['density']
    
    
    def _calc_RR(self, drho: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of enthalpy w.r.t. density at constant 
        pressure, that is the RR coefficient
        
        Parameters
        ----------
        drho : np.ndarray
            Variation of the SC densities (kg/m^3)
            
        Returns
        -------
        RR : np.ndarray
            Enthalpy variation coefficient (J*m^3/kg^2)
            RR = dh / drho = [h(rho + drho) - h(rho)] / drho
        """
        return (self.coolant.convert_properties(
            density=self.sc_properties['density']+drho) 
                - self._enthalpy) / drho
        

    def _init_static_correlated_params(self, t: float) -> None:
        """Calculate bundle friction factor and flowsplit parameters

        Parameters
        ----------
        t : float
            Inlet coolant temperature (K)
        """
        self._init_params_partial(t)  
        # Reynolds number in each subchannel          
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
        else:
            self.coolant_int_params['ff_i'] = self.coolant_int_params['ff']
        
        #Initialize subchannel velocities and densities
        self._sc_vel: np.ndarray = self.coolant_int_params['vel'] * \
            self.coolant_int_params['fs'][self.subchannel.type[
                :self.subchannel.n_sc['coolant']['total']]]     


    def _update_subchannels_properties(self, temp: np.ndarray) -> None:
        """
        Update subchannel properties based on temperature
        
        Parameters
        ----------
        temp : np.ndarray
            Array of temperatures
        """
        for i in range(len(temp)):  
            self.coolant.update(temp[i])
            for prop in MIXED_CONV_PROP_TO_UPDATE:
                self.sc_properties[prop][i] = getattr(self.coolant, prop)
                
                
    def _setup_ht_constants(self):
        """Setup heat transfer constants in numpy arrays"""
        const = calculate_ht_constants(self, mixed=True)
        self.ht = {}
        self.ht['old'] = const
        self.ht['cond'] = setup_conduction_constants(self, const)
        self.ht['conv'] = setup_convection_constants(self, const)
        
    
    @property
    def pressure_drop(self):
        return self._pressure_drop
    
        
    @property
    def sc_mfr(self):
        """Return mass flow rate in each subchannel"""
        mfr = self.sc_properties['density'] * self._sc_vel * \
            self.params['area'][self.subchannel.type[
                :self.subchannel.n_sc['coolant']['total']]]
        return mfr    
