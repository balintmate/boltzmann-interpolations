import sys
import jax
import jax.numpy as jnp


class models_ABC:
    def __init__(self, cfg, target):
        self.cfg = cfg
        self.target = target

    def VF_and_div(self, T, x, params):
        return self.VF.VF_and_div(T, x, params)

    def VF_at_xT(self, T, x, params):
        return self.VF(T, x, params)

    def C_at_T(self, T, params):
        return self.C(T, params)

    def f_NN_at_xT(self, T, x, params):
        return self.fNN(T, x, params)

    def dT_f(self, T, x, params):
        dT_0 = jax.grad(lambda T: self.f_at_xT(T, x, params))(jnp.array(T))
        return dT_0

    def dx_f(self, T, x, params):
        return jax.grad(lambda x: self.f_at_xT(T, x, params))(x)

    def continuity_error_at_xT(self, T, x, params):
        Vt, mdiv_Vt = self.VF_and_div(T, (x, 0), params["VF"])
        div_Vt = -1 * mdiv_Vt
        C = self.C_at_T(T, params["C"])
        dt_St = self.dT_f(T, x, params["f"])
        nabla_St = self.dx_f(T, x, params["f"])
        # print(nabla_St.shape,Vt.shape,dt_St.shape,div_Vt.shape,C.shape)
        # sys.stdout.flush()
        continuity_error = dt_St - div_Vt + (nabla_St * Vt).sum() + C
        return continuity_error

    def f0(self, x):
        return ((x / self.cfg.base.sigma) ** self.cfg.base.P).sum()

    def f_at_xT(self, T, x, params):
        if self.cfg.f_interpolation == "linear_trainable":
            S = T * self.target.f(x) + (1 - T) * self.f0(x)
            S = S + T * (1 - T) * self.f_NN_at_xT(T, x, params)
            return S
        elif self.cfg.f_interpolation == "trig_trainable":
            S = jnp.cos(jnp.pi / 2 * T) * self.target.f(x) + jnp.sin(
                jnp.pi / 2 * T
            ) * self.f0(x)
            S = S + jnp.sin(jnp.pi * T) * self.f_NN_at_xT(T, x, params)
            return S
        elif self.cfg.f_interpolation == "linear":
            S = T * self.target.f(x) + (1 - T) * self.f0(x)
            return S
        elif self.cfg.f_interpolation == "diffusion":
            assert self.cfg.targetE == "Gaussians2D"
            return self.target.diffusionfT(x, T, self.cfg.base_sigma)
