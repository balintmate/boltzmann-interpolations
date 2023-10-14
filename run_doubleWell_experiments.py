import os

command = [
    "tmux new-session -d",
    "./run_single.sh",
    "base.P=4",
    "base.sigma=1",
    "MLP.params.features=128",
    "MLP.params.num_hidden_layers=3",
    "MLP.params.num_models=4",
    "MLP.params.f_interpolation=linear_trainable",
    "jax_config.jax_enable_x64=False",
    "targetE=\${DoubleWellEnergy}",
    "learning_rate=3e-3",
]

command = " ".join(command)


for i in [0]:
    seed = f"RNGkey={i}"
    for m in [6]:
        target = f"DoubleWellEnergy.params.N=16 DoubleWellEnergy.params.m={m}"
        for loss in ["loss=\${KLLoss}", "loss=\${ContLoss}"]:
            os.system(f"{command} {target} {seed} train_steps=10000 " + loss)

# for i in [0]:
#     seed = f"RNGkey={i}"
#     for target in [
#         "DoubleWellEnergy.params.N=4 train_steps=10000",
#         "DoubleWellEnergy.params.N=8 train_steps=10000",
#         "DoubleWellEnergy.params.N=16 train_steps=10000"
#         "DoubleWellEnergy.params.N=32 train_steps=100000",
#         "DoubleWellEnergy.params.N=64 train_steps=250000",
#     ]:
#         target += " DoubleWellEnergy.params.m=1.50"
#         os.system(f"{command} {target} {seed} train_steps=10000 " + "loss=\${KLLoss}")
#         os.system(f"{command} {target} {seed} train_steps=10000 " + "loss=\${ContLoss}")
