import jax
import random
import jax.numpy as jnp

def calc_ESS(logp,logq):
    # https://github.com/mathisgerdes/continuous-flow-lft/blob/dbb66b11442452e263bc5c1f2f758829d24391b6/jaxlft/util.py
    """Compute the ESS given log likelihoods.
    The two likelihood arrays must be evaluated for the same set of samples.
    The samples are assumed to be sampled from ``p``, such that ``logp``
    is are the corresponding log-likelihoods.
    Args:
        logp: The log likelihood of p (up to a constant shift).
        logq: The log likelihood of q (up to a constant shift).
    Returns:
        The effective sample size per sample (between 0 and 1).
    """
    logw = logp - logq
    log_ess = 2*jax.nn.logsumexp(logw, axis=0) - jax.nn.logsumexp(2*logw, axis=0)
    ess_per_sample = jnp.exp(log_ess) / len(logw)
    return ess_per_sample.mean()

def calc_MCMC(logp,logq, key):
    logp = jnp.array(logp,dtype=jnp.float64)
    logq = jnp.array(logq,dtype=jnp.float64)
    # compute MCMC
    accepted_hist = []
    accepted_hist.append(1)
    last_logq = logq[0]
    last_logp = logp[0]
    for i in range(logp.shape[0]):
        new_logq = logq[i]
        new_logp = logp[i]
        accept_prob = min(1, jnp.exp(last_logq-last_logp+new_logp-new_logq).item())
        if jax.random.uniform(key, shape=(1,)).item() < accept_prob:
            accepted_hist.append(1)
            last_logq = new_logq
            last_logp = new_logp
        else:
            accepted_hist.append(0)
    MCMC = sum(accepted_hist)/len(accepted_hist)
    return MCMC