import os

command = [
    "tmux new-session -d",
    "./run_single.sh",
    "base.P=2",
    "MLP.params.features=64",
    "MLP.params.num_hidden_layers=2",
    "MLP.params.num_models=4",
    "jax_config.jax_enable_x64=True",
    "train_steps=10000",
    "targetE=\${GaussianEnergy}",
    "learning_rate=3e-3",
]

command = " ".join(command)

for i in range(1):
    seed = f"RNGkey={i}"

    for target in [
        # "GaussianEnergy.params.modes=[[[8,8],1,1],[[-8,-8],1,1],[[8,-8],1,1],[[-8,8],1,1]] base.sigma=1.4142",  # N(0,1)
        "GaussianEnergy.params.modes=[[[4,4],1,1],[[-8,-8],1,2]] base.sigma=2.8284"
    ]:
        for loss in [
            "loss=\${KLLoss}",
            # "loss=\${ContLoss} MLP.params.f_interpolation=linear",
            "loss=\${ContLoss} MLP.params.f_interpolation=linear_trainable",
        ]:
            os.system(f"{command} {target} {seed} " + loss)
