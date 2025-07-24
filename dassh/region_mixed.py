########################################################################
"""
date: 2025-07-09
author: fpepe
Methods for mixed convection axial regions; to be used within
Assembly objects
"""
########################################################################
import os
import csv
import numpy as np
# import warnings
import logging
from dassh.region_rodded import RoddedRegion, calculate_ht_constants, _setup_conduction_constants, _setup_convection_constants
from dassh.pin_model import PinModel
from lbh15 import Lead, Bismuth, LBE

# Coefficients for the density of lead, bismuth and LBE
# Density correlation is always in the form:
# rho(T) = a - b*T 
DENSITY_COEFF = {
    'lead': [11441, 1.2795],
    'bismuth': [10725, 1.22],
    'lbe': [11065, 1.293]
}
SODIUM_DENS_COEFF =  [275.32/2503.7, 511.58/(2503.7)**0.5, 1005.9]

_ROOT = os.path.dirname(os.path.abspath(__file__))

_gg = 9.81  # Gravity acceleration [m/s^2]
# Surface of pins in contact with each type of subchannel
q_p2sc = np.array([0.166666666666667, 0.25, 0.166666666666667])

module_logger = logging.getLogger('dassh.region_mixed')


def make(inp, name, mat, fr, se2geo=False, update_tol=0.0, gravity=False, rad_isotropic=True):
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
    gravity : bool (optional)
        Include gravity head loss in pressure drop (default=False)

    Returns
    -------
    DASSH RoddedRegion object

    """
    rr = MixedRegion(name,
                      inp['num_rings'],
                      inp['pin_pitch'],
                      inp['pin_diameter'],
                      inp['wire_pitch'],
                      inp['wire_diameter'],
                      inp['clad_thickness'],
                      inp['duct_ftf'],
                      inp['mixed_convection'],
                      fr,
                      mat['coolant'],
                      mat['duct'],
                      inp['htc_params_duct'],
                      inp['corr_friction'],
                      inp['corr_flowsplit'],
                      inp['corr_mixing'],
                      inp['corr_nusselt'],
                      inp['corr_shapefactor'],
                      inp['SpacerGrid'],
                      inp['bypass_gap_flow_fraction'],
                      inp['bypass_gap_loss_coeff'],
                      inp['wire_direction'],
                      inp['shape_factor'],
                      se2geo,
                      update_tol,
                      gravity, 
                      rad_isotropic)

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
    def __init__(self, name, n_ring, pin_pitch, pin_diam, wire_pitch,
                 wire_diam, clad_thickness, duct_ftf, flow_rate,
                 coolant_mat, duct_mat, htc_params_duct, corr_friction,
                 corr_flowsplit, corr_mixing, corr_nusselt,
                 corr_shapefactor, spacer_grid=None, byp_ff=None,
                 byp_k=None, wwdir='clockwise', sf=1.0, se2=False,
                 param_update_tol=0.0, gravity=False, rad_isotropic=True,
                 solve_enthalpy=False):
        """Instantiate MixedRegion object"""
        
        # Instantiate RoddedRegion object
        RoddedRegion.__init__(self, name, n_ring, pin_pitch, pin_diam,
                              wire_pitch, wire_diam, clad_thickness,
                              duct_ftf, flow_rate, coolant_mat, duct_mat,
                              htc_params_duct, corr_friction,
                              corr_flowsplit, corr_mixing, corr_nusselt,
                              corr_shapefactor, spacer_grid, byp_ff,
                              byp_k, wwdir, sf, se2, param_update_tol,
                              gravity, rad_isotropic, solve_enthalpy)
                 

        # Variables of interest are:
        # - SC velocities and their variations over dz
        # - SC densities and their variations over dz
        # - Bundle pressure drop and their variations over dz
        
        self._pressure_drop = 0.0    # This overrides the attribute in RoddedRegion
        self._density = np.zeros(self.subchannel.n_sc['coolant']['total']) 
        self._sc_vel = np.zeros(self.subchannel.n_sc['coolant']['total'])
        self._enthalpy = np.zeros(self.subchannel.n_sc['coolant']['total']) 
        # New attributes for the densities and velocities for the time being. 
        # In principle we already have self.sc_properties['density'] and 
        # self.coolant_int_params['sc_vel'] so this is not very smart... 
        # We should decide wether to use these or the old ones.
        self._delta_P = 1e3
        self._delta_v = 1*np.ones(self.subchannel.n_sc['coolant']['total'])
        self._delta_rho = 100*np.ones(self.subchannel.n_sc['coolant']['total'])
        #self._delta_h = 150*np.ones(self.subchannel.n_sc['coolant']['total'])
       #self._delta_v = np.zeros(self.subchannel.n_sc['coolant']['total'])
       #self._delta_rho = np.zeros(self.subchannel.n_sc['coolant']['total'])
       #self._delta_h = np.zeros(self.subchannel.n_sc['coolant']['total'])
        
    def activate(self, previous_reg, t_gap, h_gap, adiabatic):
        """Activate region by averaging coolant temperatures from
        previous region and calculating new steady state duct temps

        Parameters
        ----------
        previous_reg : DASSH Region
            Previous region to get average coolant temperatures to
            assign to new region
        t_gap : numpy.ndarray
            Gap temperatures on the new region duct mesh
        h_gap : numpy.ndarray
            Gap coolant HTC on the new region duct mesh
        adiabatic : boolean
            Indicate whether outer duct wall is adiabatic

        Notes
        -----

        """
        # Base method assigns coolant and duct MW temperatures
        # - Coolant temperature(s) set to previous region average
        # - Outer duct MW temperatures set to average outer duct MW
        #   temp in previous region; set MW temp of other ducts in
        #   new region to average coolant temperature
        self._activate_base(previous_reg)

        # Make new duct temperature calculation based on new coolant
        # temperatures and input gap temperatures / HTC. Assume zero
        # power: previous region was not a pin bundle and therefore
        # did not have power generated in the duct.
        p_duct = np.zeros(self.temp['duct_mw'].size)
        self._update_coolant_int_params(self.avg_coolant_int_temp,
                                        use_mat_tracker=False)
        
        if self.n_bypass > 0:
            self._update_coolant_byp_params(self.avg_coolant_byp_temp)
        self._calc_duct_temp(p_duct, t_gap, h_gap, adiabatic)

    ####################################################################
    # TEMPERATURE CALCULATION
    ####################################################################


    def calculate(self, dz, q, t_gap, h_gap, adiab=False, ebal=False):
        """Calculate new coolant and duct temperatures and pressure
        drop across axial step

        Parameters
        ----------
        dz : float
            Axial step size (m)
        q : dict
            Power (W/m) generated in pins, duct, and coolant
        t_gap : numpy.ndarray
            Interassembly gap temperatures around the assembly at the
            j+1 axial level (array length = n_sc['duct']['total'])
        h_gap : float
            Heat transfer coefficient for convection between the gap
            coolant and outer duct wall based on core-average inter-
            assembly gap coolant temperature
        adiabatic : boolean
            Indicate whether outer duct has adiabatic BC
        ebal : boolean
            Indicate whether to track energy balance

        Returns
        -------
        None

        """
        # Duct temperatures: calculate with new coolant properties
        self._calc_duct_temp(q['duct'], t_gap, h_gap, adiab) # This is the same as in RoddedRegion (fttb)

        q_pins, q_cool = q['pins'], q['cool']
        # Solve the system of equations for the SC
        # velocities and densities and bundle pressure drop.
        
        RR = self._calc_RR(self._delta_rho)
        self._solve_system(dz, q_pins, q_cool, RR, ebal)

        self._enthalpy += self._delta_h
        self.temp['coolant_int'] = self._temp_from_enthalpy() 
        
        # Update coolant properties for the duct wall calculation
        self._update_coolant_int_params(self.avg_coolant_int_temp)
        # Bypass coolant temperatures
        if self.n_bypass > 0:
            if self.byp_flow_rate > 0:
                self.temp['coolant_byp'] += \
                    self._calc_coolant_byp_temp(dz, ebal)
            else:
                self.temp['coolant_byp'] += \
                    self._calc_coolant_byp_temp_stagnant(dz, ebal)
            # Update bypass coolant properties for the duct wall calc
            self._update_coolant_byp_params(self.avg_coolant_byp_temp)

    def calculate_pressure_drop(self, z, dz):
        #### This is here to override the method in RoddedRegion
        #### It is not necessary because we calculate the pressure drop
        #### in the _solve_system method. We keep it fttb because it is called
        #### in the assembly object.
        return self._pressure_drop
        
    ####################################################################
    # COOLANT TEMPERATURE CALCULATION METHODS
    ####################################################################

    def _solve_system(self, dz, q_pins, q_cool, RR, ebal):
        """
        Method to solve the system. 
        """
        delta_v0 = self._delta_v.copy()
        delta_rho0 = self._delta_rho.copy()

       
        qq = self._calc_int_sc_power(q_pins, q_cool)
        
        bb = self._build_vector(qq, dz)
        RR = self._calc_RR(delta_rho0)  

        print('###########################')
        res = np.array([1, 1, 1])  # Initialize residuals
        while np.any(res > 1e-11):

            AA = self._build_matrix(dz, delta_v0, delta_rho0, RR)
            xx = np.linalg.solve(AA, bb)
            
            delta_rho = xx[0:2*self.subchannel.n_sc['coolant']['total']:2]
            delta_v = xx[1:2*self.subchannel.n_sc['coolant']['total']:2]
            delta_P = xx[-1]
            
            residuals = np.abs(AA @ xx - bb)
            res_rho = np.sum(residuals[0:2*self.subchannel.n_sc['coolant']['total']:2])
            res_v = np.sum(residuals[1:2*self.subchannel.n_sc['coolant']['total']:2])
            res_P = np.sum(residuals[-1])
            res = np.array([res_rho, res_v, res_P])                       
            print('Residuals:', res)
            delta_v0 = 0.7*delta_v + 0.3*delta_v0
            delta_rho0 = 0.7*delta_rho + 0.3 * delta_rho0

            RR = self._calc_RR(delta_rho)
            
        self._delta_v = delta_v
        self._delta_rho = delta_rho
        self._delta_P = delta_P
        self._sc_vel += delta_v
        self._density += delta_rho  
        self._pressure_drop -= delta_P
        self._delta_h = RR*delta_rho
       # print(self._pressure_drop, 'Pa')
        if ebal:
            mcpdT_i = self.params['area'][self.subchannel.type[:self.subchannel.n_sc['coolant']['total']]] * \
                self._density * self._sc_vel *self._delta_h
            self.update_ebal(dz*np.sum(qq), 0, mcpdT_i)

    def _build_vector(self, qq, dz):
        nn = self.subchannel.n_sc['coolant']['total']
        
        MEX = self._calc_MEX(dz)
        EEX = self._calc_EEX(dz)
        GG = -self._density * \
             (_gg*dz + dz*self.coolant_int_params['ff']*self._sc_vel**2/2 \
             /self.params['de'][self.subchannel.type[:nn]])
        
        energy_b = qq*dz/self.params['area'][self.subchannel.type[:nn]] + EEX    
        momentum_b = GG + MEX 
      #  print(np.sum(GG), ',')
        bb = np.zeros(2*nn + 1)
        bb[1:2*nn:2] = energy_b
        bb[0:2*nn:2] = momentum_b
        #bb[-1] = 0
        return bb

    def _build_matrix(self, dz, delta_v, delta_rho, RR):
        """
        Build the matrix and known vector.
        """
        hstar = self._enthalpy + RR*self._delta_rho/2 
        vstar = self._sc_vel + delta_v/2
        #hstar = self._calc_hstar()
        #vstar = self._calc_vstar()
        
        nn = self.subchannel.n_sc['coolant']['total']
        
        EE, FF = self._calc_momentum_coefficients(nn, dz, delta_v, vstar)
        SS, TT = self._calc_energy_coefficients(delta_v, delta_rho, hstar, RR)
        C_rho, C_v = self._calc_continuity_coefficients(nn, delta_v)
        
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
            
        AA[-1, 0:2*nn:2] = C_rho  # 2*n, non 2*n+1 perché l'ultimo è 0 (corrsponde al deltaP)
        AA[-1, 1:2*nn:2] = C_v 
        
        return AA


    def _calc_vstar(self):
        numeratore = np.zeros((self.subchannel.n_sc['coolant']['total'], 3))
        denominatore = np.zeros((self.subchannel.n_sc['coolant']['total'], 3))
        for i in range(self.subchannel.n_sc['coolant']['total']):
            for k in range(3):
                j = self.ht['cond']['adj'][i][k]
                if j == 0 and k == 2:
                    continue
                else:
                    xij = (self.sc_mfr[i] - self.sc_mfr[j]) 
                    numeratore[i][k] = np.abs(xij)*(self._sc_vel[j]+ self._sc_vel[i]) \
                                        - xij * (- self._sc_vel[j]+ self._sc_vel[i])
                        
        vstar = (numeratore[:, 0] + \
            numeratore[:, 1] + numeratore[:, 2]) / 2 / (denominatore[:, 0] + \
            denominatore[:, 1] + denominatore[:, 2])
        return np.abs(vstar)
    
    def _calc_hstar(self):
        numeratore = np.zeros((self.subchannel.n_sc['coolant']['total'], 3))
        denominatore = np.zeros((self.subchannel.n_sc['coolant']['total'], 3))
        for i in range(self.subchannel.n_sc['coolant']['total']):
            for k in range(3):
                j = self.ht['cond']['adj'][i][k]
                if j == 0 and k == 2:
                    continue
                else:
                    xij = (self.sc_mfr[i] - self.sc_mfr[j]) 
                    numeratore[i][k] = np.abs(xij)*(self._enthalpy[j]+ self._enthalpy[i]) \
                                        - xij * (- self._enthalpy[j]+ self._enthalpy[i])
                    denominatore[i][k] = np.abs(xij)

        hstar = (numeratore[:, 0] + \
            numeratore[:, 1] + numeratore[:, 2]) / 2 / (denominatore[:, 0] + \
            denominatore[:, 1] + denominatore[:, 2])
        return np.abs(hstar)
    
        

    def _calc_momentum_coefficients(self, nn, dz, delta_v, vstar):      
        """
        Calculate coefficients for the system of equations.
        """
        EE = (self._sc_vel + delta_v) * \
            (self._sc_vel + delta_v - vstar) + \
                _gg*dz/2 + \
            self.coolant_int_params['ff']*dz/16/self.params['de'][self.subchannel.type[:nn]] * \
                (2*self._sc_vel + delta_v)**2 
                
        FF = self._density * \
            ((2 + self.coolant_int_params['ff']*dz \
            /2/self.params['de'][self.subchannel.type[:nn]])*self._sc_vel + \
            (1 + self.coolant_int_params['ff']*dz/8/
            self.params['de'][self.subchannel.type[:nn]])* delta_v - vstar) 

        return EE, FF
    
    def _calc_energy_coefficients(self, delta_v, delta_rho, hstar, RR):
        """
        Calculate coefficients for the system of equations.
        """
        
        SS = (self._sc_vel + delta_v) * \
                (-hstar + self._enthalpy + RR*(self._density + delta_rho))
        
        TT = self._density*(-hstar + self._enthalpy)
        return SS, TT

    def _calc_continuity_coefficients(self, nn, delta_v):
        C_rho = self.params['area'][self.subchannel.type[:nn]] \
                                *(self._sc_vel + delta_v)
        C_v = self.params['area'][self.subchannel.type[:nn]]* \
                self._density
        return C_rho, C_v
    
    def _calc_EEX(self, dz):
        ene_exchange = np.zeros((self.subchannel.n_sc['coolant']['total'], 3))
        for i in range(self.subchannel.n_sc['coolant']['total']):
            for k in range(3):
                j = self.ht['cond']['adj'][i][k]
                if j == 0 and k == 2:
                    continue
                else:
                    rho_ij = self._calc_mass_flow_average_property('density', i, j)
                    cp_ij = self._calc_mass_flow_average_property('heat_capacity', i, j)
                    k_ij = self._calc_mass_flow_average_property('thermal_conductivity', i, j)
                    WW_ij = self.coolant_int_params['eddy'] * rho_ij + self._sf * k_ij / cp_ij
                    ene_exchange[i][k] = WW_ij  * (self.ht['cond']['const'][i][k]
                    * (self._enthalpy[j]
                    - self._enthalpy[i]))
                        
        EEX = (ene_exchange[:, 0] + \
            ene_exchange[:, 1] + ene_exchange[:, 2])
        
        swirl_consts = self.d['pin-wall'] * self.coolant_int_params['swirl']  
        swirl_consts = swirl_consts[self.ht['conv']['type']]    
        swirl_exchange = swirl_consts* \
                    (self._density[self.subchannel.sc_adj[self.ht['conv']['ind'], self._adj_sw]] 
                     * self._enthalpy[self.subchannel.sc_adj[self.ht['conv']['ind'], self._adj_sw]]
                     - self._density[self.ht['conv']['ind']]
                     * self._enthalpy[self.ht['conv']['ind']])
        EEX[self.ht['conv']['ind']] += swirl_exchange
        EEX *= dz/self.params['area'][self.subchannel.type[:self.subchannel.n_sc['coolant']['total']]] 
        return EEX
    
    def _calc_MEX(self, dz):
        mom_exchange = np.zeros((self.subchannel.n_sc['coolant']['total'], 3))
        for i in range(self.subchannel.n_sc['coolant']['total']):
            for k in range(3):
                j = self.ht['cond']['adj'][i][k]
                if j == 0 and k == 2:
                    continue
                else:
                    rho_ij = self._calc_mass_flow_average_property('density', i, j)
                    WW_ij = self.coolant_int_params['eddy'] * rho_ij
                    mom_exchange[i][k] = WW_ij  * (self.ht['cond']['const'][i][k]
                                        * (self._sc_vel[j]
                                        - self._sc_vel[i]))
                        
        MEX = (mom_exchange[:, 0] + \
            mom_exchange[:, 1] + mom_exchange[:, 2])

        swirl_consts = self.d['pin-wall'] * self.coolant_int_params['swirl']  
        swirl_consts = swirl_consts[self.ht['conv']['type']]  
        swirl_exchange = swirl_consts* \
                    (self._density[self.subchannel.sc_adj[self.ht['conv']['ind'], self._adj_sw]] 
                     * self._sc_vel[self.subchannel.sc_adj[self.ht['conv']['ind'], self._adj_sw]]
                     - self._density[self.ht['conv']['ind']]
                     * self._sc_vel[self.ht['conv']['ind']])
        MEX[self.ht['conv']['ind']] += swirl_exchange
        MEX *= dz/self.params['area'][self.subchannel.type[:self.subchannel.n_sc['coolant']['total']]]
        return MEX
    
    def _calc_RR(self, drho):
        """
        Calculate the RR coefficient.
        """
        T1 = self._T_from_rho(self._density)
        T2 = self._T_from_rho(self._density + drho)
        deltah = self._calc_delta_h(T1, T2)

        RR = deltah/drho
        return RR
        
    def _temp_from_enthalpy(self) -> np.ndarray:
        """
        Convert enthalpy difference to temperature difference
        
        Parameters
        ----------
        dh : float
            Enthalpy difference (J/kg)
        
        Returns
        -------
        float
            Temperature difference (K)
        
        """  
        dh = self._delta_h
        tref = self.temp['coolant_int'].copy()
        TT = np.zeros(len(dh))
        for i in range(len(dh)):
            toll = 1e-2
            err = 1
            iter = 1
            while (err >= toll) and (iter < 10):
                deltah = self._calc_delta_h(self.temp['coolant_int'][i], tref[i])
                self.coolant.update(tref[i])
                TT[i] = tref[i] + (dh[i] - deltah)/self.coolant.heat_capacity
                err = np.abs((TT[i]-tref[i]))
                tref[i] = TT[i] 
                iter += 1
        return TT 
    
        
    
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
        t_inlet = self.coolant.temperature
        self._update_coolant(t)
        # Coolant axial velocity, bundle Reynolds number
        mfr_over_area = self.int_flow_rate / self.bundle_params['area']
        self.coolant_int_params['vel'] = mfr_over_area / self.coolant.density
        self.coolant_int_params['Re'] = \
            mfr_over_area * self.bundle_params['de'] / self.coolant.viscosity
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

        self._sc_vel = self.coolant_int_params['vel'] * self.coolant_int_params['fs']
        self._sc_vel = self._sc_vel[self.subchannel.type[:self.subchannel.n_sc['coolant']['total']]]
        
        # Friction factor
        if self.corr['ff'] is not None:
            self.coolant_int_params['ff'] = self.corr['ff'](self)

        # Reset inlet temperature
        self.coolant.temperature = t_inlet
        self._density = self.coolant.density * np.ones(
            self.subchannel.n_sc['coolant']['total'])
        
    def _calculate_mixing_params(self) -> None:
        """
        Calculate mixing parameters for the coolant subchannels
        """
        if self.corr['mix'] is not None:
            interior_idx = self.subchannel.n_sc['coolant']['interior']
            central_sc_vel = np.sum(self.sc_mfr[:interior_idx] * 
                                    self._sc_vel[:interior_idx]) / \
                                    np.sum(self.sc_mfr[:interior_idx])

            mix = self.corr['mix'](self)
            self.coolant_int_params['eddy'] = mix[0] * central_sc_vel
                
            peripheral_sc_vel = np.sum(self.sc_mfr[self.ht['conv']['ind']] * 
                                    self._sc_vel[[self.ht['conv']['ind']]]) / \
                                    np.sum(self.sc_mfr[self.ht['conv']['ind']])
                                    
            swirl_vel = mix[1] * peripheral_sc_vel
            
            self.coolant_int_params['swirl'][1] = swirl_vel
            self.coolant_int_params['swirl'][2] = swirl_vel
                
    @property
    def pressure_drop(self):
        return self._pressure_drop
    
    def _setup_ht_constants(self):
        """Setup heat transfer constants in numpy arrays"""
        const = calculate_ht_constants(self, mixed=True)
        # self.ht_consts = const
        self.ht = {}
        self.ht['old'] = const
        self.ht['cond'] = _setup_conduction_constants(self, const)
        self.ht['conv'] = _setup_convection_constants(self, const)
        
    def _T_from_rho(self, rho):
        if self.coolant.name == 'sodium':
            aa, bb, cc = SODIUM_DENS_COEFF
            return ((-bb+np.sqrt(bb**2 - 4*aa*(rho-cc)))/(2*aa))**2
        
        elif self.coolant.name in DENSITY_COEFF.keys():
            a, b = DENSITY_COEFF[self.coolant.name]
            return (a - rho) / b
        
        else:
            path = os.path.join(_ROOT, 'data', self.coolant.name + '.csv')

            with open(path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader) 

            idx = header.index('density')
            y = np.genfromtxt(path, delimiter=',', skip_header=1)[:,0] # temperatures always in the first column
            x = np.genfromtxt(path, delimiter=',', skip_header=1)[:,idx] 
        
        
            y = y[~np.isnan(x)][::-1]
            x = x[~np.isnan(x)][::-1]
            print('x:', x)
            print('y:', y)
            return np.interp(rho, x, y)
        
        
    ############################################################################
    # Following methods and variables are here for the time being, to be removed
    # when the branch on enthalpy will be merged !!!
    
    
    def _calc_delta_h(self, T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """
        Calculate the enthalpy difference between two temperatures
        using the enthalpy coefficients for the coolant
        
        Parameters
        ----------
        T1 : numpy.ndarray
            Initial temperature (K)
        T2 : numpy.ndarray
            Final temperature (K)

        Returns
        -------
        numpy.ndarray
            Enthalpy difference (J/kg)
        """
        ENTHALPY_COEFF = {
        'lead': [176.2, -2.4615e-2, 5.147e-6, 1.524e6],
        'bismuth': [118.2, 2.967e-3, 0.0, -7.183e6],
        'lbe': [164.8, -1.97e-2, 4.167e-6, 4.56e5],
        'sodium': [1.6582e3, -4.2395e-1, 1.4847e-4, 2.9926e6],
        'nak': [971.3376, -0.18465, 1.1443e-4, 0.0],
        }
        a, b, c, d = ENTHALPY_COEFF[self.coolant.name]
        return (a * (T2 - T1) \
                + b * (T2**2 - T1**2)
                + c * (T2**3 - T1**3)
                + d * (T2**(-1) - T1**(-1)))
        
    @property
    def sc_mfr(self):
        """Return mass flow rate in each subchannel"""
        mfr = self._density * self._sc_vel * \
            self.params['area'][self.subchannel.type[:self.subchannel.n_sc['coolant']['total']]]

        return mfr