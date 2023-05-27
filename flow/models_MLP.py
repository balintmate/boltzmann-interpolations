import jax
import jax.numpy as jnp
import haiku as hk
from flow.models_ABC import models_ABC
import sys


class models_MLP(models_ABC):
    def __init__(self,cfg,target):
        super().__init__(cfg,target)
        key = jax.random.PRNGKey(cfg.RNGkey)
        self.VF = V_MLP(cfg.MLP,target,key)
        self.params = {'VF':self.VF.params}
        if self.cfg.loss == 'continuity':
            key1,key2 = jax.random.split(key,2)
            self.fNN = f_MLP(cfg.MLP,target,key1)
            self.C = C_MLP(cfg.MLP,target,key2)
            self.params = {**self.params, 'C':self.C.params,'f':self.fNN.params}

#################################### models for f(x,T), V(x,T) and C(T) ###########################################
class V_MLP():
    def __init__(self,cfg,target,key):
        self.num_models = cfg.num_models
        def V_Net(T,x):
            T_x = jnp.concatenate((x,jnp.expand_dims(T,-1)),-1)
            mlp = hk.nets.MLP(output_sizes=cfg.num_hidden_layers*[cfg.MLP_size]+[x.shape[-1]],
                              activation=jax.nn.swish,
                              activate_final= False)
            return mlp(T_x)

        self.Net = hk.transform(V_Net)
        def init_return_params(key):
            x = jax.random.normal(key,target.configuration_shape)
            T = 0
            return self.Net.init(key, T, x)

        self.params =  jax.vmap(init_return_params)(jax.random.split(key,cfg.num_models))

    def __call__(self,T,x,params):
        VFs = jax.vmap(lambda P: self.Net.apply(P,None,T,x))(params)
        time_centers = jnp.linspace(0,1,self.num_models)
        time_weights = jnp.exp(-(time_centers-T)**2*(self.num_models**2))
        return jnp.einsum('tv,t->v',VFs,time_weights)
    

    def VF_and_div(self,T, x,params):
        x, _ = x
        fn = lambda x: self.__call__(T,x,params)
        dx_dt = fn (x)
        jac = jax.jacfwd(fn)(x)
        div = jnp.trace(jac)
        return (dx_dt, -div)

class f_MLP():
    def __init__(self,cfg,target,key):
        self.num_models = cfg.num_models
        def f_Net(T, x):
            T_x = jnp.concatenate((x,jnp.expand_dims(T,-1)),-1)
            mlp = hk.nets.MLP(output_sizes=cfg.num_hidden_layers*[cfg.MLP_size]+[1],
                              activation=jax.nn.swish,
                              activate_final= False)
            return mlp(T_x)
        
        self.Net = hk.transform(f_Net)
        def init_return_params(key):
            x = jax.random.normal(key,target.configuration_shape)
            T = 0
            return self.Net.init(key, T, x)
        self.params = jax.vmap(init_return_params)(jax.random.split(key,cfg.num_models))


    def __call__(self,T,x,params):
        f_all = jax.vmap(lambda P: self.Net.apply(P,None,T,x))(params)
        time_centers = jnp.linspace(0,1,self.num_models)
        time_weights = jnp.exp(-(time_centers-T)**2*(self.num_models**2))
        f =  jnp.einsum('tn,t->',f_all,time_weights)
        return f
        

class C_MLP():
    def __init__(self,cfg,target,key):

        def C_Net(T):
            mlp = hk.nets.MLP(output_sizes=cfg.num_hidden_layers*[cfg.MLP_size]+[1],
                              activation=jax.nn.swish,
                              activate_final= False)
            return mlp(T)

        self.Net = hk.transform(C_Net)
        self.params = self.Net.init(key, jax.random.normal(key, (1,)))

    def __call__(self,T,params):
        return self.Net.apply(params,None,jnp.array([T]))