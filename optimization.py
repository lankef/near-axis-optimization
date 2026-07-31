#!/usr/bin/env python
# coding: utf-8

# # NAE optimization
# This file sees how far we can push pyAQSC's aspect ratio

# In[1]:
import os
import aqsc
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
from pathlib import Path
from shared import *
from jax import flatten_util, jit, jacfwd
from functools import partial
from jax.lax import while_loop


def _objective_vmap(x, a, b, c, d):
    return jax.vmap(
        lambda x_, a_, b_, c_, d_: objective(x_, a_, b_, c_, d_, 10),
        in_axes=(0, None, None, None, None),
        out_axes=0,
    )(x, a, b, c, d)

def _objective_vmap_full(x, a, b, c, d):
    return jax.vmap(
        lambda x_, a_, b_, c_, d_: objective(x_, a_, b_, c_, d_, 10, full_mode=True),
        in_axes=(0, None, None, None, None),
        out_axes=0,
    )(x, a, b, c, d)

objective_vmap = jax.jit(_objective_vmap)
objective_vmap_full = jax.jit(_objective_vmap_full)

import jax
from evosax.algorithms import CMA_ES
import time
import numpy as np

path = ''
if os.path.exists("best_x.npy"):
    x_flat_init = jnp.array(np.load("best_x.npy"))
    path = 'pass_2/'
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path+'global_steps').mkdir(parents=True, exist_ok=True)
    Path(path+'population_eq').mkdir(parents=True, exist_ok=True)
    print('*** restarting ***')
    restart_stats = objective(x_flat_init, w_aspect, w_anisotropy, w_iota, w_p, 10, full_mode=True)
    print(restart_stats)

if os.path.exists("pass_2/best_x.npy"):
    x_flat_init = jnp.array(np.load("pass_2/best_x.npy"))
    path = 'pass_3/'
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path+'global_steps').mkdir(parents=True, exist_ok=True)
    Path(path+'population_eq').mkdir(parents=True, exist_ok=True)
    print('*** restarting ***')
    restart_stats = objective(x_flat_init, w_aspect, w_anisotropy, w_iota, w_p, 10, full_mode=True)
    print(restart_stats)
    
if os.path.exists("pass_3/best_x.npy"):
    x_flat_init = jnp.array(np.load("pass_3/best_x.npy"))
    path = 'pass_4/'
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path+'global_steps').mkdir(parents=True, exist_ok=True)
    Path(path+'population_eq').mkdir(parents=True, exist_ok=True)
    print('*** restarting ***')
    restart_stats = objective(x_flat_init, w_aspect, w_anisotropy, w_iota, w_p, 10, full_mode=True)
    print(restart_stats)

# Instantiate the search strategy
es = CMA_ES(population_size=population_size, solution=x_flat_init)
# The original initial std is 1.0. Increase it to hopefully find a better sln
params = es.default_params.replace(std_init=std_init)

# Initialize state
key = jax.random.key(0)
state = es.init(key, x_flat_init, params)
key, key_ask, key_eval = jax.random.split(key, 3)
# Generate a set of candidate solutions to evaluate
population, state = es.ask(key_ask, state, params)

fitness1 = objective_vmap(population, w_aspect, w_anisotropy, w_iota, w_p)
num_generations = 2000
fitness_list = []
i_list = []
time_tot = 0
ask = jit(es.ask)
last_avg = np.inf

# Ask-Eval-Tell loop
time_init = time.time()
for i in range(num_generations):
    key, key_ask, key_eval = jax.random.split(key, 3)

    # Generate a set of candidate solutions to evaluate
    population, state = ask(key_ask, state, params)

    # Actually fits in memory
    time1 = time.time()
    fitness = objective_vmap(population, w_aspect, w_anisotropy, w_iota, w_p)
    time2 = time.time()
    
    # Update the evolution strategy
    state, metrics = es.tell(key, population, fitness, state, params)
    
    if i%10==0:
        data_dict = objective_vmap_full(population, w_aspect, w_anisotropy, w_iota, w_p)
        fitness_list.append(fitness.copy())
        i_list.append(i)
        fit_min = jnp.nanmin(fitness)
        fit_avg = jnp.nanmean(fitness)
        print(
            'i', i,
            'time', time2-time1, 
            'eps_crit', jnp.nanmin(data_dict['eps_crit']), jnp.nanmax(data_dict['eps_crit']), 
            'eps_conv', jnp.nanmin(data_dict['eps_conv']), jnp.nanmax(data_dict['eps_conv'])
        )
        print('min', fit_min, 'mean', fit_avg)
        print('#nan', np.sum(np.isnan(fitness)))
        jnp.save(path + 'global_steps/step'+str(i), {
            'population': population,
            'fitness': fitness,
            'data_dict': data_dict,
        })
        if fit_min < 2:
            print('Good enough')
            break
        
        if jnp.abs(fit_avg - last_avg) < 1:
            print('Converged')
            break
        last_avg = fit_avg
    
time_fin = time.time()
time_tot = time_tot + (time_fin - time_init)
print('total time', time_fin - time_init)
# Get best solution


# In[95]:


argsort_fitness = np.argsort(fitness)
sorted_fitness = fitness[argsort_fitness]
sorted_population = population[argsort_fitness]


# In[71]:


jnp.save(path + 'population.npy', population)
jnp.save(path + 'fitness.npy', fitness)
jnp.save(path + 'fitness_list.npy', fitness_list)
jnp.save(path + 'time_tot.npy', time_tot)

best_i = np.nanargmin(fitness)
best_x = population[best_i]
jnp.save(path + 'best_x.npy', best_x)


# In[96]:


plt.plot(fitness[argsort_fitness])
plt.yscale('log')


for i in range(len(sorted_fitness)):
    if sorted_fitness[i]:
        eq_fin = solve_order_6(jnp.array(sorted_population[i]), padded=False)
        fin_anisotropy = rms_anisotropy(eq_fin)
        fin_aspect = aspect_conv(eq_fin)
        fin_eps_conv = eps_conv(eq_fin)
        fin_iota = iota_axis(eq_fin)
        p20_avg = jnp.real(jnp.average(eq_fin.p_perp[2][0].content))
        p00_avg = jnp.real(jnp.average(eq_fin.p_perp[0][0].content))
        p_edge_eff = (p00_avg + p20_avg*eps**2)
        p_axis_eff = p00_avg
        print(
            'Δ:', f'{fin_anisotropy:>12.4e}', '    '
            'R/r:', f'{fin_aspect:>12.4e}', '    '
            'ι0:', f'{fin_iota:>12.4e}', '    '
            'p_axis-p_edge (estimate):', f'{p_axis_eff - p_edge_eff:>12.4e}', '    '
        )
        pres_profile = eq_fin.p_perp.eval_eps(jnp.linspace(0, fin_eps_conv, 30), 0, 0)
        plt.plot(pres_profile/jnp.max(pres_profile))
        if fin_anisotropy and fin_aspect and fin_iota:
            jnp.save(path + 'population_eq/eq'+str(i), eq_fin)

