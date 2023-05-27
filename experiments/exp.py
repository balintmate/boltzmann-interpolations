import jax 
import wandb
import hydra
import omegaconf


from target_energies.Gaussian_Energy import Gaussian_Energy
from target_energies.DoubleWell_Energy import DoubleWell_Energy
from flow.densities import Generalized_Gaussian
from experiments.train import train



@hydra.main(version_base=None, config_path='.', config_name="config.yaml")
def main(cfg):

    #### jax flags ###
    if cfg.float64: jax.config.update("jax_enable_x64", True)
    if cfg.kill_when_broadcast: jax.config.update('jax_numpy_rank_promotion','raise')
    if cfg.kill_when_nans: jax.config.update("jax_debug_nans", True)

    #### Logging ####
    try:
        wandb_key = open('./wandb.key', 'r').read()
        wandb.login(key=wandb_key)
        run = wandb.init(project="Learning Interpolations",tags=[cfg.target])
    except:
        run = wandb.init(project="Learning Interpolations", mode='offline',tags=[cfg.target])
    wandb.run.log_code("..")
    wandb.config.update(omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True))
    
    #### Energy function and flow ####
    if cfg.target == 'Gaussians2D': 
        target =  Gaussian_Energy(cfg.Gaussians2D)
    elif cfg.target == 'DoubleWell': 
        target =  DoubleWell_Energy(cfg.DoubleWell)

    
    base =  Generalized_Gaussian(cfg,target.configuration_shape)
    train(target,base,cfg)
   
    
    run.finish()

if __name__ == "__main__":
    main()
