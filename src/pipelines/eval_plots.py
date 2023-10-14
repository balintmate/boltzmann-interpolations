import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import sys
import hydra
from losses.continuity import continuityLoss
import wandb

plt.rcParams["mathtext.fontset"] = "cm"


# plots
def potential_histogram(cfg, target, samples, logdict):
    xlim = cfg.DoubleWellEnergy.params.plot_xlim
    x = jnp.linspace(-xlim, xlim, 1000)
    V = target.V(x)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(211)
    ax.plot(x, V)
    ax.set_ylabel(r"$V(\phi)$", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(212)
    ax.plot(x, jnp.exp(-V))
    ax.set_xlabel(r"$\phi$", fontsize=16)
    # ax.set_ylabel('exp(-V)', fontsize=16)
    ax.set_ylabel(r"$e^{-V(\phi)}$", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(bottom=0)
    ax = ax.twinx()
    ax.hist(samples.reshape(-1), 200, color="C1", alpha=0.3)

    ax.set_xlim(-xlim, xlim)
    ax.set_yticks([])
    # ax.axvline(x = target.roots()[0],c='r')
    # ax.axvline(x = target.roots()[1],c='r')
    logdict["Potential"] = wandb.Image(fig)
    plt.close()
    return logdict


def Target2D(cfg, flow):
    res = 80
    PS = cfg.GaussianEnergy.params.plot_size
    x, y = jnp.meshgrid(jnp.linspace(-PS, PS, res), jnp.linspace(-PS, PS, res))
    grid = jnp.stack((x, y), axis=-1).reshape((-1, 2))
    plt.figure(figsize=(5, 5))
    target_density = jnp.exp(-jax.vmap(lambda x: flow.target.f(x))(grid)).reshape(
        res, res
    )
    plt.contourf(x, y, target_density, cmap="Blues")
    plt.xlim(-PS, PS)
    plt.ylim(-PS, PS)
    plt.xticks([])
    plt.yticks([])
    plt.close()
    return wandb.Image(plt)


def samples_trajectory(cfg, flow, samples, logdict, key):
    res = 80
    PS = cfg.GaussianEnergy.params.plot_size
    x, y = jnp.meshgrid(jnp.linspace(-PS, PS, res), jnp.linspace(-PS, PS, res))
    grid = jnp.stack((x, y), axis=-1).reshape((-1, 2))

    num_steps = cfg.GaussianEnergy.params.num_plot_steps + 1
    fig = plt.figure(
        figsize=(
            5 * num_steps,
            5 + 5 * (cfg.loss["_target_"] == "losses.continuity.continuityLoss"),
        )
    )
    gs = fig.add_gridspec(
        1 + (cfg.loss["_target_"] == "losses.continuity.continuityLoss"),
        num_steps,
        wspace=0,
        hspace=0,
    )
    axs = gs.subplots(sharex=True, sharey=True)

    if cfg.loss["_target_"] == "losses.continuity.continuityLoss":
        axs[0, 0].set_ylabel(r"target", fontsize=20)
        axs[1, 0].set_ylabel(r"samples", fontsize=20)
    for i in range(num_steps):
        T = i / (num_steps - 1)

        if cfg.loss["_target_"] == "losses.reverseKL.reverseKL":
            target_density = jnp.exp(
                -jax.vmap(lambda x: flow.target.f(x))(grid).reshape(res, res)
            )

            axs[i].contourf(x, y, target_density, cmap="Blues")
            keys = jax.random.split(key, num=cfg.eval_size // cfg.batch_size)
            samples, _ = jax.vmap(
                lambda key: flow.Sample(
                    flow.params, key, N=cfg.batch_size, T1=max(1e-3, T)
                )
            )(keys)
            samples = jnp.reshape(samples, (-1,) + (samples.shape[2:]))

            x2, y2 = jnp.meshgrid(
                jnp.linspace(-PS, PS, res // 5), jnp.linspace(-PS, PS, res // 5)
            )
            grid2 = jnp.stack((x2, y2), axis=-1).reshape((-1, 2))
            VF = jax.vmap(lambda x: flow.models.VF_at_xT(T, x, flow.params["VF"]))(
                grid2
            ).reshape(res // 5, res // 5, 2)
            axs[i].quiver(x2, y2, VF[:, :, 0], VF[:, :, 1], alpha=0.5)

            axs[i].scatter(samples[:1024, 0], samples[:1024, 1], s=5, c="C1", alpha=0.6)
            axs[i].set_title(rf"$t={i/(num_steps-1):.2f}$", fontsize=40)
            axs[i].set_xlim(-PS, PS)
            axs[i].set_ylim(-PS, PS)
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        if cfg.loss["_target_"] == "losses.continuity.continuityLoss":
            target_density = jax.vmap(
                lambda x: jnp.exp(-flow.models.f_at_xT(T, x, flow.params["f"]))
            )(grid).reshape(res, res)
            axs[0, i].contourf(x, y, target_density, cmap="Blues")
            axs[1, i].contourf(x, y, target_density, cmap="Blues")

            x2, y2 = jnp.meshgrid(
                jnp.linspace(-PS, PS, res // 5), jnp.linspace(-PS, PS, res // 5)
            )
            grid2 = jnp.stack((x2, y2), axis=-1).reshape((-1, 2))
            VF = jax.vmap(lambda x: flow.models.VF_at_xT(T, x, flow.params["VF"]))(
                grid2
            ).reshape(res // 5, res // 5, 2)
            axs[1, i].quiver(x2, y2, VF[:, :, 0], VF[:, :, 1], alpha=0.5)

            keys = jax.random.split(key, num=cfg.eval_size // cfg.batch_size)
            samples, _ = jax.vmap(
                lambda key: flow.Sample(
                    flow.params, key, N=cfg.batch_size, T1=max(1e-3, T)
                )
            )(keys)
            samples = jnp.reshape(samples, (-1,) + (samples.shape[2:]))

            axs[0, i].set_title(rf"$t={i/(num_steps-1):.2f}$", fontsize=40)
            axs[0, i].set_xlim(-PS, PS)
            axs[0, i].set_ylim(-PS, PS)
            axs[0, i].set_xticks([])
            axs[0, i].set_yticks([])
            axs[1, i].scatter(
                samples[:1024, 0], samples[:1024, 1], s=5, c="C1", alpha=0.6
            )
            axs[1, i].set_xlim(-PS, PS)
            axs[1, i].set_ylim(-PS, PS)
            axs[1, i].set_xticks([])
            axs[1, i].set_yticks([])
    logdict["Samples Trajectory"] = wandb.Image(plt)
    plt.close()
    return logdict
