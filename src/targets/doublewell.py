import jax
import jax.numpy as jnp
import sys, math
import numpy as np
from flow.densities import Generalized_Gaussian


class doubleWellEnergy:
    def __init__(self, params):
        self.params = params
        T = 4
        self.dT = T / params.N
        self.configuration_shape = (params.N,)

    def f(self, x):
        pot = self.V(x)
        kin = (self.params.m / 2) * (x - jnp.roll(x, 1)) ** 2 / self.dT**2
        return self.dT * (pot + kin).sum()

    def V(self, x):
        V = (self.params.lam / 4) * x**4 - (self.params.m / 2) * x**2
        return V

    def roots(self):
        roots = np.real(np.roots(np.array([self.params.lam, 0, -self.params.m, 0])))
        return np.sort(roots)[0::2]
