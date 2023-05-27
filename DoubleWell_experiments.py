import os
import sys
script_path = f'tmux new-session -d  {sys.path[0]}/start_the_run.sh target=DoubleWell baseP=4 base_sigma=1 float64=False'
arch = 'MLP.MLP_size=128 MLP.num_hidden_layers=3 MLP.num_models=4 f_interpolation=linear_trainable'
train_params = "learning_rate=3e-3"

for i in [0,1,2]:
    seed = f"RNGkey={i}"
    for m in [6,4.5,3,1.5]:
        target = f'DoubleWell.N=16 DoubleWell.m={m} train_steps=10000'
        #os.system(f"{script_path} loss=ReverseKL  {target} {arch} {train_params} {seed}")
        # os.system(f"{script_path} loss=continuity {target} {arch} {train_params} {seed}")


# for i in [0]:
#     seed = f"RNGkey={i}"
#     for target in [
#         'DoubleWell.N=4 DoubleWell.m=1.50 train_steps=10000',
#         'DoubleWell.N=8 DoubleWell.m=1.50 train_steps=10000',
#         'DoubleWell.N=16 DoubleWell.m=1.50 train_steps=10000'
#         'DoubleWell.N=32 DoubleWell.m=1.50 train_steps=100000',
#         'DoubleWell.N=64 DoubleWell.m=1.50 train_steps=250000',
#     ]:

#         os.system(f"{script_path} loss=ReverseKL {target} {arch} {train_params} {seed}")
#         os.system(f"{script_path} loss=continuity  {target} {arch} {train_params} {seed}")






