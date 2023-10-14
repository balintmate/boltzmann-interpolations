import jax
import jax.numpy as jnp


class Density_ABC:
    def __init__(self):
        raise NotImplementedError

    def Sample(self, key):
        raise NotImplementedError

    def LogLikelihood(self, x):
        raise NotImplementedError


class Generalized_Gaussian(Density_ABC):
    def __init__(self, cfg, configuration_shape):
        self.P = cfg.P
        self.sigma = cfg.sigma
        self.configuration_shape = configuration_shape

    def Sample(self, params, key, N):
        z = self.sigma * jax.random.generalized_normal(
            key, self.P, shape=(N,) + self.configuration_shape
        )
        logq = jax.scipy.stats.gennorm.logpdf(z / self.sigma, self.P) - jnp.log(
            self.sigma
        )
        logq = jnp.reshape(logq, (len(logq), -1)).sum(-1)
        return z, logq

    def LogLikelihood(self, params, x):
        logq = jax.scipy.stats.gennorm.logpdf(x / self.sigma, self.P) - jnp.log(
            self.sigma
        )
        logq = jnp.reshape(logq, (len(logq), -1)).sum(-1)
        return logq
