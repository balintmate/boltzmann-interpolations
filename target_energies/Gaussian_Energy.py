from curses import init_pair
import jax
import jax.numpy as jnp
import sys,math
 
class Gaussian_Energy():
    def __init__(self,cfg):
        self.cfg =cfg
        self.means = jnp.array([mode[0] for mode in cfg.modes])
        self.sigmas = jnp.array([mode[1] for mode in cfg.modes])
        self.weights = jnp.array([mode[2] for mode in cfg.modes])
        self.weights/=self.weights.sum()
        self.configuration_shape = (2,)

    def f(self,x):
        x = jnp.reshape(x,(1,2))
        Si = .5*((x-self.means)**2).sum(-1)/(self.sigmas**2)
        likelihood = self.weights*jnp.exp(-Si)/(2*math.pi*self.sigmas)
        return  -jnp.log(likelihood.sum())

    def diffusionfT(self,x,T,base_sigma):
        T = T + 1e-3
        x = jnp.reshape(x,(1,2))
        means = self.means*jnp.sqrt(T)
        sigmas =  jnp.sqrt(T * self.sigmas**2 +(1-T)*base_sigma**2)
        Si = .5*((x-means)**2).sum(-1)/(sigmas**2)
        likelihood = self.weights*jnp.exp(-Si)/(2*math.pi*sigmas)
        return  -jnp.log(likelihood.sum())


    def sample(self,key,N):
        z = jax.random.normal(key,shape=(N,2))
        z=jax.random.permutation(key, z)
        indices = jax.random.categorical(key,jnp.log(self.weights),shape=(N,))
        return jnp.expand_dims(self.sigmas[indices],-1)*z +self.means[indices]