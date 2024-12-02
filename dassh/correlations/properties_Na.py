import numpy as np

class Na_from_corr(object):
    """
    Correlation object for sodium properties
    
    Parameters
    ----------
    
    """
    def __init__(self, prop):
        self.prop = prop
    
    def __call__(self, temperature):
        return getattr(self, self.prop)(temperature)
    
    def density(self, T):
        return 219 + 275.32*(1-T/2503.7) + 511.58*(1-T/2503.7)**0.5 
    def thermal_conductivity(self, T):
        return 124.67 - 0.11381*T + 5.5226e-5*T**2 - 1.1842e-8*T**3
    def viscosity(self, T):
        return np.exp(-6.4406 - 0.3958 * np.log(T) + 556.835/T)
    def specific_heat(self, T):
        c_s = 1.6582 - 8.4790e-4*T + 4.4541e-7*T**2 - 2992.6*T**-2
        gamma = (12633.7*T**-2 - 0.4672*T**-1) * np.exp(11.9463 - 12633.7*T**-1 - 0.4672*np.log(T))
        alpha_s = -1/self.density(T) * (-0.1022* (1- 0.0003994*T)**0.5 -0.1100)
        beta_s = 1.7171e-4*((1+ (T-371)/(2503.7-371))/3.2682)/(1-(T-371)/(2503.7-371))
        beta = (beta_s*c_s + (T/self.density(T))*alpha_s*(alpha_s + beta_s*gamma))/(c_s + (T/self.density(T))*gamma*(alpha_s + beta_s*gamma))
        alpha = alpha_s + beta_s*gamma
        cp = c_s + (T*alpha*gamma/self.density(T))
        return cp