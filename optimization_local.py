import importlib
import aqsc
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jrd
import optax
import optax.tree
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
from shared import *
from jax import grad
i_test = 1
# eq_fin = jnp.load(f'pass_2/best_x.npy', allow_pickle=True).item()
fitness = jnp.load(f'./fitness.npy')
population = jnp.load(f'./population.npy')
x_fin = population[jnp.argmin(fitness)]
eq_test = solve_order_6(x_fin, padded=True)

fun = lambda x: objective(x, w_aspect, w_anisotropy, w_iota, w_p, 10, full_mode=False, padded=True)

obj_wrapped = jit(fun)
jac = jit(grad(fun))
# Crashes kernel

# Define optimizer
lr = 1e-1
opt = optax.scale_by_lbfgs()
val_and_grad = jit(jax.value_and_grad(fun))
update = jit(opt.update)

# Define objective
niter = 1000

# Initialize optimization
w_opt = x_fin
state = opt.init(w_opt)

# Run optimization
for i in range(niter):
    v, g = val_and_grad(w)
    if i%10 == 0:
        print(f'Iteration: {i}, Value:{v:.2e}')
    u, state = update(g, state, w)
    w = w - lr * u

print(f'Final value: {fun(w):.2e}')
jnp.save('local_result', w)