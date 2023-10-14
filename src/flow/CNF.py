import jax
import jax.numpy as jnp
import hydra
from flow.models_MLP import models_MLP
from flow.ode import odeint_rk4
from functools import partial
from flow.densities import Density_ABC


class CNF(Density_ABC):
    def __init__(self, cfg, target, base):
        self.cfg = cfg
        self.target = target
        self.base = base

        self.models = hydra.utils.instantiate(cfg.model_architecture, target=target)

        print(self.models)
        self.params = {**self.models.params}

    def ODESolve(self, VF, x0, params, T0=0, T1=1):
        def VF_xt(x, t, params):  # experimental odesolve wants the arguments reversed
            return VF(t, x, params)

        sol = jax.vmap(
            lambda x: odeint_rk4(
                VF_xt,
                x,
                args=params,
                step_size=1 / self.cfg.integration_steps,
                start_time=T0,
                end_time=T1,
            )
        )(x0)
        return sol

    @partial(jax.jit, static_argnames=("self", "T1", "N"))
    def Sample(self, params, key, N, T1=1.0):
        z, logq0 = self.base.Sample(params, key, N)
        return self.ODESolve(self.models.VF_and_div, (z, logq0), params["VF"], T1=T1)

    def LogLikelihood(self, params, x):
        def VF(T, x, params):
            VF, div = self.models.VF_and_div(1 - T, x, params)
            return -VF, div

        z, logq = self.ODESolve(VF, (x, jnp.zeros((len(x),))), params["VF"])
        logbase = self.base.LogLikelihood(params, z)
        return logq + logbase
