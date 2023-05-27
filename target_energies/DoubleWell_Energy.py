import jax
import jax.numpy as jnp
import sys,math
import numpy as np
from flow.densities import Generalized_Gaussian

class DoubleWell_Energy():
    def __init__(self,cfg):
        self.cfg = cfg
        T=4
        self.dT = T/cfg.N
        self.configuration_shape = (cfg.N,)

    def f(self,x):
        pot = self.V(x)
        kin = (self.cfg.m / 2) * (x-jnp.roll(x,1))**2/self.dT**2
        return self.dT*(pot+kin).sum()
        
    
    def V(self,x):
        V =  (self.cfg.lam / 4) * x**4 - (self.cfg.m / 2) * x**2
        return V

    def roots(self):
        roots = np.sort(np.real(np.roots(np.array([self.cfg.lam,0,-self.cfg.m,0]))))[0::2]
        return roots
