import jax
import jax.numpy as jnp
import flax.linen as nn
from flow.models_ABC import models_ABC
from typing import Sequence


class models_MLP(models_ABC):
    def __init__(self, params, target):
        super().__init__(params, target)
        key = jax.random.PRNGKey(params.RNGkey)
        hidden_features = params.num_hidden_layers * [params.features]
        N = target.configuration_shape[0]
        self.VF = V_MLP(
            key=key,
            num_models=params.num_models,
            features=[N + 1] + hidden_features + [N],
        )
        self.params = {"VF": self.VF.params}
        if params.loss == "losses.continuity.continuityLoss":
            key1, key2 = jax.random.split(key, 2)
            self.fNN = f_MLP(
                key=key1,
                num_models=params.num_models,
                features=[N + 1] + hidden_features + [1],
            )
            self.C = C_MLP(key=key2, features=[1] + hidden_features + [1])
            self.params = {**self.params, "C": self.C.params, "f": self.fNN.params}


#################################### models for f(x,T), V(x,T) and C(T) ###########################################
class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, f in enumerate(self.features):
            x = nn.Dense(f)(x)
            if i != len(self.features) - 1:
                x = nn.swish(x)
        return x


class V_MLP:
    def __init__(self, features, num_models, key):
        x = jax.random.uniform(key, (features[0],))
        self.mlp = MLP(features=features[1:])
        self.params = jax.vmap(lambda key: self.mlp.init(key, x))(
            jax.random.split(key, num_models)
        )
        self.num_models = num_models

    def __call__(self, T, x, params):
        T_x = jnp.concatenate((x, jnp.expand_dims(T, -1)), -1)
        VFs = jax.vmap(lambda P: self.mlp.apply(P, T_x))(params)
        time_centers = jnp.linspace(0, 1, self.num_models)
        time_weights = jnp.exp(-((time_centers - T) ** 2) * (self.num_models**2))
        return jnp.einsum("tv,t->v", VFs, time_weights)

    def VF_and_div(self, T, x, params):
        x, _ = x
        fn = lambda x: self.__call__(T, x, params)
        dx_dt = fn(x)
        jac = jax.jacfwd(fn)(x)
        div = jnp.trace(jac)
        return (dx_dt, -div)


class f_MLP:
    def __init__(self, features, num_models, key):
        x = jax.random.uniform(key, (features[0],))
        self.mlp = MLP(features=features[1:])
        self.params = jax.vmap(lambda key: self.mlp.init(key, x))(
            jax.random.split(key, num_models)
        )
        self.num_models = num_models

    def __call__(self, T, x, params):
        T_x = jnp.concatenate((x, jnp.expand_dims(T, -1)), -1)
        fs = jax.vmap(lambda P: self.mlp.apply(P, T_x))(params)
        time_centers = jnp.linspace(0, 1, self.num_models)
        time_weights = jnp.exp(-((time_centers - T) ** 2) * (self.num_models**2))
        return jnp.einsum("to,t->", fs, time_weights)


class C_MLP:
    def __init__(self, features, key):
        T = jax.random.uniform(key, (features[0],))
        self.mlp = MLP(features=features[1:])
        self.params = self.mlp.init(key, T)

    def __call__(self, T, params):
        return self.mlp.apply(params, jnp.expand_dims(T, -1))
