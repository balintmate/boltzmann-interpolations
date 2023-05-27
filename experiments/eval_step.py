import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import sys

plt.rcParams["mathtext.fontset"] = 'cm'

import wandb
from experiments.ESS_MCMC import calc_ESS

#metrics

def BuildLog(cfg,CNF,step,key):
    logdict, samples = metrics(cfg,CNF,CNF.params,key)
    if step % (5*cfg.eval_steps) == 0:
        if cfg.target == 'DoubleWell':
            logdict = potential_histogram(cfg,CNF.target,samples,logdict)
        if samples.shape[1:]== (2,):
            logdict = Samples_Trajectory(cfg,CNF,samples,logdict,key)
    ESS, logZ, H = logdict['eval/ESS'], -logdict['eval/reverseKL'], logdict['eval/Hausdorff(means,samples)']
    print(f'Training step: {step}/{cfg.train_steps} , ESS: {ESS:.3f}, log Z:{logZ:.3f}, Hausdorrf:{H:.3f}')
    sys.stdout.flush()
    return logdict


def metrics(cfg,CNF,params,key):
    keys = jax.random.split(key, num=cfg.eval_size//cfg.batch_size)
    samples, logq = jax.vmap(lambda key: CNF.Sample(params,key,N=cfg.batch_size))(keys)
    samples = jnp.reshape(samples,(-1,)+(samples.shape[2:]))
    logq = jnp.reshape(logq,(-1,))

    logp = -jax.vmap(lambda x: CNF.target.f(x))(samples)
    KLloss  = jax.jit(CNF.ReverseKL)(params,key)
    logdict = { 'eval/ESS': jax.jit(calc_ESS)(logp,logq).item(),
                #'eval/MCMC':calc_MCMC(logp, logq, key),
                'eval/logq':logq.mean().item(), 
                'eval/logp':logp.mean().item(),
                'eval/reverseKL':KLloss.item()}
    if cfg.loss == 'continuity':
        cont_loss= CNF.ContLoss(params,key)
        logdict['eval/continuity loss'] = cont_loss.item()

    if cfg.target == 'Gaussians2D':
        D_modes = jnp.expand_dims(CNF.target.means,1)
        D = jax.vmap(lambda x:jnp.sqrt(((x-samples)**2).sum(-1)))(D_modes)
        logdict['eval/Hausdorff(means,samples)'] = D.min(1).max().item()
    if cfg.target == 'DoubleWell':
        roots = CNF.target.roots()
        D = jax.jit(jax.vmap(lambda x:jnp.sqrt(((x-samples)**2).sum(-1))))(roots)
        logdict['eval/Hausdorff(means,samples)'] = D.min(1).max().item()
        

    if cfg.target == 'Gaussians2D':
        real_samples = CNF.target.sample(key,cfg.eval_size)
        logp = -jax.vmap(lambda x: CNF.target.f(x))(real_samples)
        logq = CNF.LogLikelihood(params,real_samples)
        logdict['eval/forward KL'] = (logp-logq).mean().item()
        #logdict['eval/forward MCMC'] = calc_MCMC(logq,logp,key)
        logdict['eval/forward ESS'] = calc_ESS(logq,logp).item()
        
    return logdict, samples

#plots
def potential_histogram(cfg,target,samples,logdict):
    xlim = cfg.DoubleWell.plot_xlim
    x = jnp.linspace(-xlim,xlim,1000)
    V = target.V(x)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(211)
    ax.plot(x,V)
    ax.set_ylabel(r'$V(\phi)$', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(212)
    ax.plot(x,jnp.exp(-V))
    ax.set_xlabel(r'$\phi$', fontsize=16)
    #ax.set_ylabel('exp(-V)', fontsize=16)
    ax.set_ylabel(r'$e^{-V(\phi)}$', fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(bottom=0)
    ax = ax.twinx()
    ax.hist(samples.reshape(-1),200,color='orange',alpha=.3);
    
    ax.set_xlim(-xlim,xlim)
    ax.set_yticks([])
    # ax.axvline(x = target.roots()[0],c='r')
    # ax.axvline(x = target.roots()[1],c='r')
    logdict["Potential"] = wandb.Image(fig)
    plt.close()
    return logdict

def Target2D(cfg,CNF):
    res = 80
    PS = cfg.Gaussians2D.plot_size
    x, y = jnp.meshgrid(jnp.linspace(-PS, PS, res), jnp.linspace(-PS, PS, res))
    grid = jnp.stack((x,y),axis=-1).reshape((-1,2))
    plt.figure(figsize=(5,5))
    target_density = jnp.exp(-jax.vmap(lambda x: CNF.target.f(x))(grid)).reshape(res, res)
    plt.contourf(x, y, target_density, cmap='Blues')
    plt.xlim(-PS,PS)
    plt.ylim(-PS,PS)
    plt.xticks([])
    plt.yticks([])
    plt.close()
    return wandb.Image(plt)
    
def Samples_Trajectory(cfg,CNF,samples,logdict,key):
    res = 80
    PS = cfg.Gaussians2D.plot_size
    x, y = jnp.meshgrid(jnp.linspace(-PS, PS, res), jnp.linspace(-PS, PS, res))
    grid = jnp.stack((x,y),axis=-1).reshape((-1,2))
    

    num_steps = cfg.Gaussians2D.num_plot_steps+1
    fig = plt.figure(figsize=(5 * num_steps,5 + 5 * (cfg.loss == 'continuity')))
    gs = fig.add_gridspec(1 + (cfg.loss == 'continuity'), num_steps, wspace=0,hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    
    if cfg.loss == 'continuity':
        axs[0,0].set_ylabel(r'target', fontsize=20)
        axs[1,0].set_ylabel(r'samples', fontsize=20)
    for i in range(num_steps):
        T = i/(num_steps-1)
        
        if cfg.loss == 'ReverseKL':
            target_density = jnp.exp(-jax.vmap(lambda x: CNF.target.f(x))(grid).reshape(res, res))
    
            axs[i].contourf(x, y, target_density, cmap='Blues')
            keys = jax.random.split(key, num=cfg.eval_size//cfg.batch_size)
            samples, _ = jax.vmap(lambda key: CNF.Sample(CNF.params,key,N=cfg.batch_size,T1=max(1e-3,T)))(keys)
            samples = jnp.reshape(samples,(-1,)+(samples.shape[2:]))

            x2, y2 = jnp.meshgrid(jnp.linspace(-PS, PS, res//5), jnp.linspace(-PS, PS, res//5))
            grid2 = jnp.stack((x2,y2),axis=-1).reshape((-1,2))
            VF = jax.vmap(lambda x: CNF.models.VF_at_xT(T,x,CNF.params['VF']))(grid2).reshape(res//5, res//5,2)
            axs[i].quiver(x2,y2,VF[:,:,0],VF[:,:,1],alpha=.5)


            axs[i].scatter(samples[:1024,0],samples[:1024,1],s = 5, c='red',alpha = .6)
            axs[i].set_title(fr"$t={i/(num_steps-1):.2f}$",fontsize=40)
            axs[i].set_xlim(-PS,PS)
            axs[i].set_ylim(-PS,PS)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            
            
        if cfg.loss == 'continuity':
            target_density = jax.vmap(lambda x: jnp.exp(-CNF.models.f_at_xT(T,x,CNF.params['f'])))(grid).reshape(res, res)
            axs[0,i].contourf(x, y, target_density, cmap='Blues')
            axs[1,i].contourf(x, y, target_density, cmap='Blues')
            
            x2, y2 = jnp.meshgrid(jnp.linspace(-PS, PS, res//5), jnp.linspace(-PS, PS, res//5))
            grid2 = jnp.stack((x2,y2),axis=-1).reshape((-1,2))
            VF = jax.vmap(lambda x: CNF.models.VF_at_xT(T,x,CNF.params['VF']))(grid2).reshape(res//5, res//5,2)
            axs[1,i].quiver(x2,y2,VF[:,:,0],VF[:,:,1], alpha=.5)

            keys = jax.random.split(key, num=cfg.eval_size//cfg.batch_size)
            samples, _ = jax.vmap(lambda key: CNF.Sample(CNF.params,key,N=cfg.batch_size,T1=max(1e-3,T)))(keys)
            samples = jnp.reshape(samples,(-1,)+(samples.shape[2:]))

            axs[0,i].set_title(fr"$t={i/(num_steps-1):.2f}$",fontsize=40)
            axs[0,i].set_xlim(-PS,PS)
            axs[0,i].set_ylim(-PS,PS)
            axs[0,i].set_xticks([])
            axs[0,i].set_yticks([])
            axs[1,i].scatter(samples[:1024,0],samples[:1024,1],s = 5, c='red',alpha = .6)
            axs[1,i].set_xlim(-PS,PS)
            axs[1,i].set_ylim(-PS,PS)
            axs[1,i].set_xticks([])
            axs[1,i].set_yticks([])
    logdict['Samples Trajectory'] = wandb.Image(plt)
    plt.close()
    return logdict