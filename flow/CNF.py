import jax
import jax.numpy as jnp
from flow.models_MLP import models_MLP
from flow.ode import odeint_rk4
from experiments.ESS_MCMC import calc_ESS
from functools import partial
from flow.densities import Density_ABC

class CNF(Density_ABC):
    def __init__(self,cfg,target, base):
        self.cfg = cfg
        self.target = target
        self.base = base
        
        self.models = models_MLP(cfg,target)

        self.params= {**self.models.params}


    def ODESolve(self,VF,x0,params,T0=0,T1=1):
        def VF_xt(x,t,params): # experimental odesolve wants the arguments reversed
            return VF(t,x,params)
        sol = jax.vmap(lambda x: odeint_rk4(VF_xt,x,
                    args=params,
                    step_size=1/self.cfg.integration_steps,
                    start_time=T0,
                    end_time=T1))(x0)
        return sol

    @partial(jax.jit, static_argnames=('self','T1','N'))
    def Sample(self,params,key,N,T1=1.):
        z,logq0 = self.base.Sample(params,key,N)
        return self.ODESolve(self.models.VF_and_div,(z,logq0),params['VF'],T1=T1)

    def ReverseKL(self,params,key):
        x, logq = self.Sample(params,key,self.cfg.batch_size)
        S = jax.vmap(lambda x: self.target.f(x))(x)
        rKL = logq + S 
        return rKL.mean()

    def LogLikelihood(self,params,x):
        def VF(T,x,params):
            VF,div = self.models.VF_and_div(1-T,x,params)
            return -VF,div
        z, logq = self.ODESolve(VF,(x,jnp.zeros((len(x),))),params['VF'])
        logbase = self.base.LogLikelihood(params,z)
        return logq+ logbase

#################################### Continuity Loss ###########################################    
    def ContLoss_grad(self,params,key):
        z,logq0 = self.base.Sample(params,key,self.cfg.batch_size)
        # compute grad at T=0
        params_grad = jax.tree_map(lambda x:x*0,params)
        
        def step_func(f,cur_t, dt, cur_y):
            """Take one step of RK4."""
            k1 = f(cur_t, cur_y)
            k2 = f(cur_t + dt / 2, cur_y + dt * k1 / 2)
            k3 = f(cur_t + dt / 2, cur_y + dt * k2 / 2)
            k4 = f(cur_t + dt, cur_y + dt * k3)
            return cur_y + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

        def body_fun(T, z):
            z,params_grad = z
            T=T/self.cfg.integration_steps
            def cont_loss(params):
                error = jax.vmap(lambda x: self.models.continuity_error_at_xT(T,x,params))(z)
                if self.cfg.continuity_L_function == 'L1':
                    return (jnp.abbs(error)).mean()
                elif self.cfg.continuity_L_function == 'L2':
                    return (error**2).mean()
                elif self.cfg.continuity_L_function == 'L1+L2':
                    return (jnp.abs(error)+error**2).mean()
                
            params_grad_cur_t = jax.grad(lambda params:cont_loss(params))(params)
            params_grad= jax.tree_map(lambda x,y:x+y/(self.cfg.integration_steps+1),params_grad,params_grad_cur_t)

            f = lambda t,y : self.models.VF_at_xT(t,y,params['VF'])
            z = jax.vmap(lambda z: step_func(f,T,1/self.cfg.integration_steps,z))(z)
            z = jax.lax.stop_gradient(z)
            return (z,params_grad)
        
        x, params_grad = jax.lax.fori_loop(0, self.cfg.integration_steps+1, body_fun, (z,params_grad))
            
        return params_grad


    @partial(jax.jit, static_argnames=('self'))
    def ContLoss(self,params,key):
        z,logq0 = self.base.Sample(params,key,self.cfg.batch_size)
        def step_func(f,cur_t, dt, cur_y):
            """Take one step of RK4."""
            k1 = f(cur_t, cur_y)
            k2 = f(cur_t + dt / 2, cur_y + dt * k1 / 2)
            k3 = f(cur_t + dt / 2, cur_y + dt * k2 / 2)
            k4 = f(cur_t + dt, cur_y + dt * k3)
            return cur_y+(k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

        def body_fun(T, z):
            z, L = z
            T=T/self.cfg.integration_steps
            error = jax.vmap(lambda x: self.models.continuity_error_at_xT(T,x,params))(z)
            L+=(jnp.abs(error)+error**2).mean()
            f = lambda t,y : self.models.VF_at_xT(t,y,params['VF'])
            z = jax.vmap(lambda z: step_func(f,T,1/self.cfg.integration_steps,z))(z)
            z = jax.lax.stop_gradient(z)
            return (z,L)
        
        x, L = jax.lax.fori_loop(0, self.cfg.integration_steps+1, body_fun, (z,0))
            
            
        return L/(self.cfg.integration_steps+1)
    
   