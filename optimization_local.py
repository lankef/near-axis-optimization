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
import optimistix as optx
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
from pathlib import Path
from shared import *
from jax import flatten_util, jit, jacfwd
from functools import partial
from jax.lax import while_loop
from shared import * 

x_flat_init = jnp.array(np.load("best_x.npy"))

# LBFGS (via optimistix) runs its whole iteration loop as a single compiled
# lax.while_loop, so there's no Python-level hook to report progress from.
# jax.debug.callback runs on the host at actual runtime (once per real
# objective evaluation, not just once at trace time), so it's used here to
# print progress periodically. Note this counts *objective evaluations*,
# not accepted LBFGS steps -- LBFGS's backtracking line search calls the
# objective more than once per step, so this undercounts true iterations
# somewhat, but tracks progress closely enough to tell whether it's stuck.
_progress_state = {'n': 0}
_report_every = 20

def _report_progress(aspect, loss):
    n = _progress_state['n'] + 1
    _progress_state['n'] = n
    if n % _report_every == 0:
        print(
            f'[eval {n}] aspect={float(aspect):.4f} loss={float(loss):.4e}',
            flush=True,
        )

def obj_fin_full(x, args):
    out = objective(x, w_aspect, w_anisotropy, w_iota, w_p, 10, full_mode=True)
    loss = jnp.clip(out['loss'], a_min=-1e10, a_max=1e10)
    jax.debug.callback(_report_progress, out['aspect'], loss, ordered=True)
    return loss

obj_fin = jit(obj_fin_full)

solver = optx.LBFGS(
    rtol=1e-5,
    atol=1e-5,
    history_length=20,
)

sol = optx.minimise(
    obj_fin,
    solver,
    x_flat_init,
    has_aux=False,
    max_steps=1000,
    throw=False,
)

print('Final objective', obj_fin(sol.value, None))
eq_fin = solve_order_6(sol.value, padded=False)
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
jnp.save('local/eq', eq_fin)
jnp.save('local/x', sol.value)
jnp.save('local/sol', sol)