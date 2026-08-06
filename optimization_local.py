#!/usr/bin/env python
# coding: utf-8

# # NAE optimization
# This file sees how far we can push pyAQSC's aspect ratio.
# Outer loop is jax.scipy.optimize.minimize (BFGS), so the whole iteration
# loop -- line search included -- is traced into a single lax.while_loop and
# compiled once. That means one big up-front compile instead of the separate
# per-call objective/jacobian compiles of the scipy version, and no host
# round-trip per iteration.

import os
os.environ['XLA_FLAGS'] = "--xla_gpu_autotune_level=2"
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, jacfwd
from jax.scipy.optimize import minimize
from functools import partial
from pathlib import Path
from shared import *

m_opt = 10
maxiter = 1000
x_flat_init = jnp.array(np.load("best_x.npy"))
opt_args = (w_aspect, w_anisotropy, w_iota, w_p, m_opt)

jac = jit(jacfwd(objective))

# There is no callback hook in jax.scipy.optimize.minimize, so progress is
# reported from inside the traced loop via jax.debug.callback. This fires once
# per objective evaluation (line-search trials included) rather than once per
# accepted BFGS iteration.
_progress_state = {'n': 0}
_report_every = 2


def _report(aspect, loss_val):
    n = _progress_state['n']
    _progress_state['n'] = n + 1
    if n % _report_every == 0:
        print(
            f'[eval {n}] aspect={float(aspect):.4f} '
            f'loss={float(loss_val):.4e}',
            flush=True,
        )


def _loss_impl(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m):
    out = objective(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m,
                    full_mode=True)
    jax.debug.callback(_report, out['aspect'], out['loss'])
    return jnp.clip(out['loss'], a_min=-1e10, a_max=1e10)


# jax.scipy's BFGS takes its gradient from jax.value_and_grad, i.e. reverse
# mode, but the order-6 solve is far cheaper in memory under forward mode.
# Wrapping the objective in a custom_vjp whose backward pass calls jacfwd keeps
# the gradient path identical to the scipy version, and gives us somewhere to
# zero out the nans that the line search would otherwise choke on. The weights
# and m are nondiff_argnums: they are Python scalars held fixed for the run.
@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
def loss(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m):
    return _loss_impl(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m)


def _loss_fwd(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m):
    val = _loss_impl(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m)
    return val, x_flat


def _loss_bwd(w_aspect, w_anisotropy, w_iota, w_p, m, x_flat, g):
    out = jac(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m)
    return (g * jnp.nan_to_num(out, nan=0., posinf=0., neginf=0.),)


loss.defvjp(_loss_fwd, _loss_bwd)


def jac_safe(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m):
    out = jac(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m)
    return np.asarray(jnp.nan_to_num(out, nan=0., posinf=0., neginf=0.))


def fun_np(x_flat, *args):
    return float(np.asarray(obj_wrapped(x_flat, *args)))


_STATUS_MSG = {
    0: 'converged (|grad| < gtol)',
    1: 'maximum number of iterations reached',
    -1: 'undefined',
}

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

# Note on options: jax.scipy's BFGS only accepts maxiter, norm, gtol and
# line_search_maxiter. There is no ftol (the scipy run's 'ftol': 1e-5 has no
# equivalent and is dropped), and no bounds -- this is plain BFGS carrying a
# dense inverse Hessian rather than L-BFGS-B. The scipy run passed no bounds
# either, so the feasible set is unchanged.
print('Tracing / compiling the full BFGS loop...', flush=True)
t0 = time.time()
sol = minimize(
    fun=loss,
    x0=x_flat_init,
    args=opt_args,
    method='BFGS',
    options={
        'maxiter': maxiter,
        'gtol': 1e-5,
    },
)
status = int(sol.status)
message = _STATUS_MSG.get(status, f'line search failed (status {status})')
print(
    f'jax BFGS done in {time.time() - t0:.1f}s '
    f'({int(sol.nit)} iters, {int(sol.nfev)} fevals): {message}',
    flush=True,
)

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
np.save('local/sol.npy', np.asarray(sol.x))
