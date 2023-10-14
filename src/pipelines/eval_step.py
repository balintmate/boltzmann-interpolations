import jax.numpy as jnp
import jax
import sys
import hydra
from losses.continuity import continuityLoss


import wandb
from pipelines.utils import calc_ESS
from pipelines.eval_plots import potential_histogram, samples_trajectory


def BuildLog(cfg, flow, step, key):
    logdict, samples = metrics(cfg, flow, flow.params, key)
    if step % (5 * cfg.eval_steps) == 0:
        if cfg.targetE["_target_"] == "targets.doublewell.doubleWellEnergy":
            logdict = potential_histogram(cfg, flow.target, samples, logdict)
        if samples.shape[1:] == (2,):
            logdict = samples_trajectory(cfg, flow, samples, logdict, key)
    ESS, logZ, H = (
        logdict["eval/ESS"],
        -logdict["eval/reverseKL"],
        logdict["eval/Hausdorff(means,samples)"],
    )
    print(
        f"Training step: {step}/{cfg.train_steps} , ESS: {ESS:.3f}, log Z:{logZ:.3f}, Hausdorrf:{H:.3f}"
    )
    sys.stdout.flush()
    return logdict


def metrics(cfg, flow, params, key):
    keys = jax.random.split(key, num=cfg.eval_size // cfg.batch_size)
    samples, logq = jax.vmap(lambda key: flow.Sample(params, key, N=cfg.batch_size))(
        keys
    )
    samples = jnp.reshape(samples, (-1,) + (samples.shape[2:]))
    logq = jnp.reshape(logq, (-1,))
    logp = -jax.vmap(lambda x: flow.target.f(x))(samples)

    logdict = {
        "eval/ESS": jax.jit(calc_ESS)(logp, logq).item(),
        "eval/logq": logq.mean().item(),
        "eval/logp": logp.mean().item(),
        "eval/reverseKL": (logq - logp).mean().item(),
    }
    if cfg.loss["_target_"] == "losses.continuity.continuityLoss":
        loss_obj = hydra.utils.instantiate(cfg.loss, target=flow.target)
        cont_loss = loss_obj(flow, params, key)
        logdict["eval/continuity loss"] = cont_loss.item()
        del loss_obj

    if cfg.targetE["_target_"] == "targets.gaussian.gaussianEnergy":
        D_modes = jnp.expand_dims(flow.target.means, 1)
        D = jax.vmap(lambda x: jnp.sqrt(((x - samples) ** 2).sum(-1)))(D_modes)
        logdict["eval/Hausdorff(means,samples)"] = D.min(1).max().item()

        real_samples = flow.target.sample(key, cfg.eval_size)
        logp = -jax.vmap(lambda x: flow.target.f(x))(real_samples)
        logq = flow.LogLikelihood(params, real_samples)
        logdict["eval/forward KL"] = (logp - logq).mean().item()
        # logdict['eval/forward MCMC'] = calc_MCMC(logq,logp,key)
        logdict["eval/forward ESS"] = calc_ESS(logq, logp).item()

    if cfg.targetE["_target_"] == "targets.doublewell.doubleWellEnergy":
        roots = flow.target.roots()
        D = jax.jit(jax.vmap(lambda x: jnp.sqrt(((x - samples) ** 2).sum(-1))))(roots)
        logdict["eval/Hausdorff(means,samples)"] = D.min(1).max().item()

    return logdict, samples
