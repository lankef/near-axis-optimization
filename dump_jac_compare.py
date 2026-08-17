#!/usr/bin/env python
"""
Compare make_jaxpr cost of fun vs grad(fun), isolating get_psi_crit / stop_gradient.

Runs on CPU by default so it does not fight a notebook GPU compile.

  JAX_PLATFORMS=cpu python dump_jac_compare.py --order 4
  JAX_PLATFORMS=cpu python dump_jac_compare.py --order 6 --skip-psicrit-grad
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import grad

import dump_fun_hlo as D
import shared as S


def _count_eqns(jaxpr, counter=None):
    if counter is None:
        counter = collections.Counter()
    for eqn in jaxpr.eqns:
        counter[eqn.primitive.name] += 1
        for param in eqn.params.values():
            if hasattr(param, "jaxpr"):
                _count_eqns(param.jaxpr, counter)
            elif hasattr(param, "eqns"):
                _count_eqns(param, counter)
    return counter


def make_objective(order: int, mode: str, padded: bool = True):
    """
    mode:
      iota          — scalar iota only (MHD path, no volume / psi_crit)
      nopsicrit     — notebook-like loss using eps_conv only
      psicrit       — notebook objective (eps = min(eps_conv, eps_crit))
      psicrit_sg    — same but stop_gradient(psi_crit)
    """
    m = 10
    w_aspect, w_anisotropy, w_iota, w_p = S.w_aspect, S.w_anisotropy, S.w_iota, S.w_p

    def fun(x_flat):
        in_dict = S.unravel_x(x_flat)
        eq = D.solve_to_order(x_flat, m=m, order=order, padded=padded)
        if mode == "iota":
            return jnp.real(eq.iota.eval(psi=0, chi=0, phi=0))

        anisotropy = S.rms_anisotropy(eq)
        eps_conv_val = S.eps_conv(eq)
        if mode == "nopsicrit":
            eps = eps_conv_val
        else:
            psi_crit, _, _ = eq.get_psi_crit(n_grid_chi=50, n_grid_phi_skip=5)
            if mode == "psicrit_sg":
                psi_crit = jax.lax.stop_gradient(psi_crit)
            eps = jnp.minimum(eps_conv_val, jnp.sqrt(psi_crit))

        aspect = eq.aspect_ratio_eps(eps)
        iota_a = jnp.real(eq.iota.eval(psi=0, chi=0, phi=0))
        p20_avg = jnp.real(jnp.average(eq.p_perp[2][0].content))
        p00_avg = jnp.real(jnp.average(eq.p_perp[0][0].content))
        p_edge_eff = p00_avg + p20_avg * eps**2
        p_axis_eff = p00_avg
        term1 = w_aspect * (jnp.maximum(aspect - S.target_aspect, 0) / S.target_aspect) ** 2
        term2 = (
            w_anisotropy
            * (jnp.maximum(anisotropy - S.target_anisotropy, 0) / S.target_anisotropy) ** 2
        )
        term3 = w_iota * (
            jnp.maximum(jnp.abs(S.target_iota) - jnp.abs(iota_a), 0) / S.target_iota
        ) ** 2
        term4 = w_p * (jnp.maximum(p_edge_eff - p_axis_eff, 0) / p_axis_eff) ** 2
        return term1 + term2 + term3 + term4

    return fun


def probe(name: str, f, x):
    t0 = time.perf_counter()
    jp = jax.make_jaxpr(f)(x)
    dt = time.perf_counter() - t0
    ctr = _count_eqns(jp)
    n = sum(ctr.values())
    interesting = {
        k: ctr[k]
        for k in (
            "custom_linear_solve",
            "while",
            "scan",
            "remat2",
            "fft",
            "lu",
            "triangular_solve",
            "custom_vjp_call_jaxpr",
            "custom_jvp_call_jaxpr",
            "pjit",
        )
        if ctr[k]
    }
    print(
        f"{name}: make_jaxpr {dt:.1f}s, eqns={n}, top={interesting}",
        flush=True,
    )
    return n, dt, ctr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--order", type=int, default=4, choices=(2, 4, 6))
    p.add_argument(
        "--skip-psicrit-grad",
        action="store_true",
        help="Skip grad through live get_psi_crit (very expensive at high order)",
    )
    args = p.parse_args()
    x = S.x_flat_init
    print(
        f"device={jax.devices()} x_dim={x.shape} order={args.order} "
        f"static_freq={S.static_freq}",
        flush=True,
    )

    modes = ["iota"]
    if args.order >= 4:
        modes += ["nopsicrit", "psicrit_sg"]
        if not args.skip_psicrit_grad:
            modes.append("psicrit")

    for mode in modes:
        fun = make_objective(args.order, mode)
        probe(f"o{args.order} {mode} fun", fun, x)
        probe(f"o{args.order} {mode} grad", grad(fun), x)


if __name__ == "__main__":
    main()
