import jax
import jax.numpy as jnp


class continuityLoss:
    def __init__(self, batch_size, integration_steps, continuity_L_function, target):
        self.integration_steps = integration_steps
        self.batch_size = batch_size
        self.target = target
        self.continuity_L_function = continuity_L_function

    def __call__(self, flow, params, key):
        z, logq0 = flow.base.Sample(params, key, self.batch_size)

        def step_func(f, cur_t, dt, cur_y):
            """Take one step of RK4."""
            k1 = f(cur_t, cur_y)
            k2 = f(cur_t + dt / 2, cur_y + dt * k1 / 2)
            k3 = f(cur_t + dt / 2, cur_y + dt * k2 / 2)
            k4 = f(cur_t + dt, cur_y + dt * k3)
            return cur_y + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

        def body_fun(T, z):
            z, L = z
            T = T / self.integration_steps
            error = jax.vmap(
                lambda x: flow.models.continuity_error_at_xT(T, x, params)
            )(z)
            L += (jnp.abs(error) + error**2).mean()
            f = lambda t, y: flow.models.VF_at_xT(t, y, params["VF"])
            z = jax.vmap(lambda z: step_func(f, T, 1 / self.integration_steps, z))(z)
            z = jax.lax.stop_gradient(z)
            return (z, L)

        x, L = jax.lax.fori_loop(0, self.integration_steps + 1, body_fun, (z, 0))

        return L / (self.integration_steps + 1)

    def grad(self, flow, params, key):
        z, logq0 = flow.base.Sample(params, key, self.batch_size)
        # compute grad at T=0
        params_grad = jax.tree_map(lambda x: x * 0, params)

        def step_func(f, cur_t, dt, cur_y):
            """Take one step of RK4."""
            k1 = f(cur_t, cur_y)
            k2 = f(cur_t + dt / 2, cur_y + dt * k1 / 2)
            k3 = f(cur_t + dt / 2, cur_y + dt * k2 / 2)
            k4 = f(cur_t + dt, cur_y + dt * k3)
            return cur_y + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

        def body_fun(T, z):
            z, params_grad = z
            T = T / self.integration_steps

            def cont_loss(params):
                error = jax.vmap(
                    lambda x: flow.models.continuity_error_at_xT(T, x, params)
                )(z)
                if self.continuity_L_function == "L1":
                    return (jnp.abbs(error)).mean()
                elif self.continuity_L_function == "L2":
                    return (error**2).mean()
                elif self.continuity_L_function == "L1+L2":
                    return (jnp.abs(error) + error**2).mean()

            params_grad_cur_t = jax.grad(lambda params: cont_loss(params))(params)
            params_grad = jax.tree_map(
                lambda x, y: x + y / (self.integration_steps + 1),
                params_grad,
                params_grad_cur_t,
            )

            f = lambda t, y: flow.models.VF_at_xT(t, y, params["VF"])
            z = jax.vmap(lambda z: step_func(f, T, 1 / self.integration_steps, z))(z)
            z = jax.lax.stop_gradient(z)
            return (z, params_grad)

        x, params_grad = jax.lax.fori_loop(
            0, self.integration_steps + 1, body_fun, (z, params_grad)
        )

        return params_grad
