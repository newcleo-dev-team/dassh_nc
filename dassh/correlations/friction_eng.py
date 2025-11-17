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
date: 2020-03-19
author: matz
Engel friction factor correlation (1979)
"""
########################################################################
import numpy as np


# Application ranges of friction factor correlations
applicability = {}
applicability['P/D'] = np.array([1.067, 1.082])
applicability['H/D'] = np.array([7.7, 8.3])
applicability['Nr'] = np.array([19, 61])
applicability['regime'] = ['turbulent', 'transition', 'laminar']
applicability['Re'] = np.array([50, 1e5])
applicability['bare rod'] = False


########################################################################
# BUNDLE FRICTION FACTOR
########################################################################


def calculate_bundle_friction_factor(asm_obj, Re: float = None) -> float:
    """Calculate the bundle-average friction factor. Optionally is used to
    calculate friction factor at subchannel Reynolds numbers

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the assembly geometric details and bundle Re
    Re : float, optional
        Bundle Reynolds number to use in case of subchannel `ff` calculation

    Returns
    -------
    float
        Bundle-average friction factor based on assembly geometry
        and flow regime. Optionally calculated at given subchannel `Re`
    """
    if not Re:
        Re = asm_obj.coolant_int_params['Re']
        
    if Re < 400.0:
        return 110 / Re
    elif Re > 5000.0:
        return 0.55 / Re**0.25

    x = (Re - 400) / 4600
    return (110 * np.sqrt(1 - x) / Re 
            + 0.55 * np.sqrt(x) / Re**0.25)


def calculate_subchannel_friction_factor(asm_obj) -> np.ndarray:
    """
    Calculate the subchannel friction factors using the Engel correlation 
    
    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the assembly geometric details and subchannel Reynolds numbers
    
    Returns
    -------
    np.ndarray
        Subchannel friction factors at given flow conditions
    """
    f_sc = np.zeros_like(asm_obj.coolant_int_params['Re_all_sc'])
    for i, Re_sc in enumerate(asm_obj.coolant_int_params['Re_all_sc']):
        f_sc[i] = calculate_bundle_friction_factor(asm_obj, Re=Re_sc)
    return f_sc
