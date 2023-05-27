import os
import sys
script_path = f'tmux new-session -d  {sys.path[0]}/start_the_run.sh target=Gaussians2D baseP=2 MLP.MLP_size=64 MLP.num_hidden_layers=2 MLP.num_models=4 float64=True'

train_params = "train_steps=10000"
LR = 'learning_rate=3e-3'
for i in range(5):
    seed = f"RNGkey={i}"

    target = "Gaussians2D.modes=[[[8,8],1,1],[[-8,-8],1,1],[[8,-8],1,1],[[-8,8],1,1]] base_sigma=1.4142" # N(0,1)
    # os.system(f"{script_path} loss=ReverseKL {LR} {target} {train_params} {seed}")
    # os.system(f"{script_path} loss=continuity f_interpolation=linear {LR} {target} {train_params} {seed}")
    # os.system(f"{script_path} loss=continuity f_interpolation=linear_trainable {LR} {target} {train_params} {seed}")
    
    # # # #########################

    target = "Gaussians2D.modes=[[[4,4],1,1],[[-8,-8],1,2]] base_sigma=2.8284" # N(0,2)
    # os.system(f"{script_path} loss=ReverseKL {LR} {target} {train_params} {seed}")
    # os.system(f"{script_path} loss=continuity f_interpolation=linear {LR} {target} {train_params} {seed}")
    # os.system(f"{script_path} loss=continuity f_interpolation=linear_trainable {LR} {target} {train_params} {seed}")
    # if i==0:
    #      os.system(f"{script_path} loss=continuity f_interpolation=diffusion {LR} {target} {train_params} {seed}")