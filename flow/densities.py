import jax
import jax.numpy as jnp

class Density_ABC():
    def __init__(self):
        raise NotImplementedError
    
    def Sample(self,key):
        raise NotImplementedError

    def LogLikelihood(self,x):
        raise NotImplementedError


class Generalized_Gaussian(Density_ABC):
    def __init__(self,cfg,configuration_shape):
        self.baseP = cfg.baseP
        self.base_sigma = cfg.base_sigma
        self.configuration_shape = configuration_shape

    def Sample(self,params,key,N):
        z = self.base_sigma*jax.random.generalized_normal(key, self.baseP, shape=(N,)+self.configuration_shape)
        logq = jax.scipy.stats.gennorm.logpdf(z/self.base_sigma, self.baseP)-jnp.log(self.base_sigma)
        logq = jnp.reshape(logq,(len(logq),-1)).sum(-1)
        return z,logq

    def LogLikelihood(self,params,x):
        logq = jax.scipy.stats.gennorm.logpdf(x/self.base_sigma, self.baseP)-jnp.log(self.base_sigma)
        logq = jnp.reshape(logq,(len(logq),-1)).sum(-1)
        return logq
    
