import jax
import wandb
import hydra
import omegaconf


from targets.gaussian import gaussianEnergy
from targets.doublewell import doubleWellEnergy
from flow.densities import Generalized_Gaussian
from pipelines.train import train


@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg):
    #### jax flags ###
    for cfg_name, cfg_value in cfg.jax_config.items():
        jax.config.update(cfg_name, cfg_value)
    ### wandb setup ###
    try:
        wandb_key = open("./wandb.key", "r").read()
        wandb.login(key=wandb_key)
        run = wandb.init(project=cfg.wandb_project_name)
    except:
        print("Weights and biases key not found or not valid. Will be logging locally.")
        run = wandb.init(project=cfg.wandb_project_name, mode="offline")
    wandb.run.log_code("..")
    wandb.config.update(
        omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    #### Energy function and flow ####
    target = hydra.utils.instantiate(cfg.targetE)

    base = Generalized_Gaussian(cfg.base, target.configuration_shape)
    train(target, base, cfg)

    run.finish()


if __name__ == "__main__":
    main()
