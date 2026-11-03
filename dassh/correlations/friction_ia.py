########################################################################
"""
date: 2026-08-xx
author: fpepe
Friction factor correlation for inter-assembly gap flow
"""
########################################################################
import numpy as np


def calculate_subchannel_friction_factor(Re: np.ndarray) -> np.ndarray:
    """Calculate the subchannel friction factors 
    
    Parameters
    ----------
    Re : np.ndarray
        Subchannel Reynolds numbers
    
    Returns
    -------
    np.ndarray
        Subchannel friction factors
        
    Note
    ----
    Blasius correlation function
    """
    return 0.316/Re**0.25