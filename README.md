# Boltzmann Interpolations

This repository contains the implementation of the paper
> [Learning Interpolations between Boltzmann Densities](https://openreview.net/forum?id=TH6YrEcbth) by Bálint Máté and François Fleuret.

## Environment
Given an existing installation of ```virtualenv```, the environment necessary to executing the experiments in this repository can be set up by ```env/_setup/create_env.sh```.
## Experiments in the paper

Experiments can be launched by commands like
```
source env/bin/activate
export PYTHONPATH=${PWD}:${PYTHONPATH}
cd experiments
python3 exp.py target=Gaussians2D loss=continuity f_interpolation=linear_trainable
```
See ```experiments/config.yaml``` for the tuneable parameters of the experiments.

Alternatively, the scripts ```Gaussian_experiments.py``` and ```DoubleWell_experiments.py``` contain the configurations used in the paper and can be used to launch our experiments in detached tmux terminals (assuming that ```tmux``` is already installed). To start them,  just uncomment the lines corresponding to the configurations you are interested in. By default everything is commented out to avoid starting a bunch of runs at once.


## Logging
All the plots and metrics are also logged to the ```wandb```directory by default. If you create a file at ```experiments/wandb.key``` containing your weights and biases key, then all the logs will be pushed to your wandb account.

## Citation
If you find our paper or this repository useful, consider citing us at

```
@misc{máté2023learning,
      title={Learning Interpolations between Boltzmann Densities}, 
      author={Bálint Máté and François Fleuret},
      year={2023},
      eprint={2301.07388},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```