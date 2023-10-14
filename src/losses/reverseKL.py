import jax


class reverseKL:
    def __init__(self, batch_size, target):
        self.batch_size = batch_size
        self.target = target

    def __call__(self, flow, params, key):
        x, logq = flow.Sample(params, key, self.batch_size)
        S = jax.vmap(lambda x: self.target.f(x))(x)
        rKL = logq + S
        return rKL.mean()

    def grad(self, flow, params, key):
        return jax.grad(lambda p: self.__call__(flow, p, key))(params)
