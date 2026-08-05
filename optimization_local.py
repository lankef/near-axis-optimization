#!/usr/bin/env python
# coding: utf-8

# # NAE optimization
# This file sees how far we can push pyAQSC's aspect ratio.
# Outer loop is scipy L-BFGS-B so each objective/jacobian compile stays
# separate (unlike optimistix.minimise, which JITs the whole iteration loop).

import os
os.environ['XLA_FLAGS'] = "--xla_gpu_autotune_level=2"
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, jacfwd
from scipy.optimize import minimize
from pathlib import Path
from shared import *

m_opt = 10
maxiter = 1000
x_flat_init = jnp.array(np.load("best_x.npy"))
opt_args = (w_aspect, w_anisotropy, w_iota, w_p, m_opt)

jac = jit(jacfwd(objective))

def jac_safe(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m):
    out = jac(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m)
    return np.asarray(jnp.nan_to_num(out, nan=0., posinf=0., neginf=0.))

def fun_np(x_flat, *args):
    return float(np.asarray(obj_wrapped(x_flat, *args)))

_progress_state = {'n': 0}
_report_every = 2

def _callback(xk):
    n = _progress_state['n']
    _progress_state['n'] = n + 1
    if n % _report_every == 0:
        out = objective(jnp.asarray(xk), *opt_args, full_mode=True)
        print(
            f'[iter {n}] aspect={float(out["aspect"]):.4f} '
            f'loss={float(out["loss"]):.4e}',
            flush=True,
        )

Path('local').mkdir(exist_ok=True)

print('Devices:', jax.devices(), flush=True)
print('Compiling / evaluating initial objective...', flush=True)
t0 = time.time()
f0 = fun_np(x_flat_init, *opt_args)
print(f'Init objective {f0:.6e}  ({time.time() - t0:.1f}s)', flush=True)
print('Init objective', objective(x_flat_init, *opt_args, full_mode=False))

print('Compiling / evaluating initial jacobian (jacfwd)...', flush=True)
t0 = time.time()
g0 = jac_safe(x_flat_init, *opt_args)
print(
    f'Init |grad| {float(np.linalg.norm(g0)):.6e}  ({time.time() - t0:.1f}s)',
    flush=True,
)

t0 = time.time()
sol = minimize(
    fun=fun_np,
    x0=np.asarray(x_flat_init),
    jac=jac_safe,
    args=opt_args,
    method='L-BFGS-B',
    callback=_callback,
    options={
        'maxiter': maxiter,
        'ftol': 1e-5,
        'gtol': 1e-5,
    },
)
print(f'scipy L-BFGS-B done in {time.time() - t0:.1f}s: {sol.message}', flush=True)

x_fin = jnp.asarray(sol.x)
print('Final objective', fun_np(x_fin, *opt_args))
eq_fin = solve_order_6(x_fin, m=m_opt, padded=False)
fin_anisotropy = rms_anisotropy(eq_fin)
fin_aspect = aspect_conv(eq_fin)
fin_eps_conv = eps_conv(eq_fin)
fin_iota = iota_axis(eq_fin)
psi_crit, _, _ = eq_fin.get_psi_crit()
eps = jnp.minimum(fin_eps_conv, jnp.sqrt(psi_crit))
p20_avg = jnp.real(jnp.average(eq_fin.p_perp[2][0].content))
p00_avg = jnp.real(jnp.average(eq_fin.p_perp[0][0].content))
p_edge_eff = (p00_avg + p20_avg * eps**2)
p_axis_eff = p00_avg
print(
    'Δ:', f'{fin_anisotropy:>12.4e}', '    '
    'R/r:', f'{fin_aspect:>12.4e}', '    '
    'ι0:', f'{fin_iota:>12.4e}', '    '
    'p_axis-p_edge (estimate):', f'{p_axis_eff - p_edge_eff:>12.4e}', '    '
)
jnp.save('local/eq', eq_fin)
jnp.save('local/x', x_fin)
np.save('local/sol.npy', sol.x)
