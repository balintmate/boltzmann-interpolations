import jax
import jax.numpy as jnp
import optax
from flow.CNF import CNF 
import wandb
import sys
from eval_step import BuildLog
 
def train(target,base,cfg):
    flow = CNF(cfg,target,base = base)
    
    #### Optimizer ####
    lr_schedule = lambda i: jnp.cos(jnp.pi/2 * i/cfg.train_steps)
    optimizer = optax.chain(
        optax.adam(cfg.learning_rate),
        optax.scale_by_schedule(lr_schedule)
        
    )
    opt_state = optimizer.init(flow.params)

    #### Loss function ####
    if cfg.loss == 'ReverseKL':
        grad_fn = lambda params,key : jax.grad(lambda params:flow.ReverseKL(params,key))(params)
    elif cfg.loss == 'continuity':
        if cfg.contloss == 'integrate':
            grad_fn = lambda params,key : flow.ContLoss_grad(params,key)
        elif cfg.contloss == 'sample':
            grad_fn = lambda params,key : flow.ContLoss_grad_sample(params,key)

    #### Training and eval step ####
    @jax.jit
    def update_step(params,opt_state,key):
        grad = grad_fn(params,key)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params,opt_state


    def eval_step(model,step,key):
        logdict = BuildLog(cfg,model,step,key)
        logdict['other/LR'] = cfg.learning_rate*lr_schedule(opt_state[1].count.item())
        logdict['other/training_progress'] = (step+1)/cfg.train_steps
        wandb.log(logdict)
        del logdict
        return

    #### Counting params ####
    def print_param_count(params,str=''):
        print(f'{str}:{sum(x.size for x in jax.tree_util.tree_leaves(params))//1000}K')
    print_param_count(flow.params,'total_params')
    for key in flow.params.keys():
        print_param_count(flow.params[key],' - '+key)
    sys.stdout.flush()

    #### Training Loop ####
    key = jax.random.PRNGKey(cfg.RNGkey)
    params = flow.params
    for step in range(cfg.train_steps+1):
        key, _ = jax.random.split(key)
        params, opt_state = update_step(params,opt_state,key)
        if step % cfg.eval_steps == 0:
            key, _ = jax.random.split(key)
            flow.params = params
            eval_step(flow,step,key)
    

    flow.params = params
    return flow
    ### save params
    # if cfg.save_model:
    #     model_path = f'{wandb.run.dir}/model_params'
    #     with open(model_path, 'wb') as file:
    #         pickle.dump(params, file)
    #         wandb.save(model_path, policy="now")