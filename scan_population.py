#!/usr/bin/env python
"""Scan population.npy for the lowest-aspect member with positive p_grad_max_eff."""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from shared import eps_conv, solve_order_6, unravel_x


def metrics(x_flat):
    in_dict = unravel_x(x_flat)
    eq = solve_order_6(x_flat, m=10, padded=True)
    psi_crit, _, _ = eq.get_psi_crit()
    eps = jnp.minimum(eps_conv(eq), jnp.sqrt(psi_crit))
    aspect = eq.aspect_ratio_eps(eps)
    # effective beta
    p_eff_axis = jnp.real(eq.p_perp.eval_eps(eps=0, chi=0, phi=0))
    p_eff_edge = jnp.real(eq.p_perp.eval_eps(eps=eps, chi=0, phi=0)) 
    eps_profile = jnp.linspace(0, eps, 10)
    p_eff_prof = jnp.real(eq.p_perp.eval_eps(eps=eps_profile, chi=jnp.zeros(10), phi=jnp.zeros(10))) 
    p_grad_max_eff = jnp.max(jnp.gradient(p_eff_prof) * 10) # eps_max * partial p / partial eps
    return aspect, p_grad_max_eff


metrics_jit = jax.jit(metrics)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate aspect and p_grad_max_eff for every member of population.npy "
            "and save the x with the lowest aspect among those with p_grad_max_eff > 0."
        )
    )
    parser.add_argument(
        "--population",
        default="./population.npy",
        help="Path to population.npy (default: ./population.npy)",
    )
    parser.add_argument(
        "--out-x",
        default="best_x_min_aspect.npy",
        help="Output path for the winning x (default: best_x_min_aspect.npy)",
    )
    parser.add_argument(
        "--out-metrics",
        default="best_x_min_aspect_metrics.npy",
        help=(
            "Output path for the winning metrics dict "
            "(default: best_x_min_aspect_metrics.npy)"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    population = jnp.load(args.population)
    n = int(population.shape[0])
    print(f"Loaded {args.population}: {n} members, dim {population.shape[1]}")

    print("Warming up JIT (first evaluation compiles) ...")
    t0 = time.time()
    aspect0, beta0 = metrics_jit(population[0])
    aspect0 = float(aspect0)
    beta0 = float(beta0)
    print(
        f"  i {0:4d}  aspect {aspect0:.8e}  p_grad_max_eff {beta0:.8e}  "
        f"({time.time() - t0:.2f}s, includes compile)"
    )

    best_i = None
    best_aspect = np.inf
    best_beta = None
    if np.isfinite(aspect0) and np.isfinite(beta0) and beta0 > 0:
        best_i = 0
        best_aspect = aspect0
        best_beta = beta0

    for i in range(1, n):
        t1 = time.time()
        aspect, p_grad_max_eff = metrics_jit(population[i])
        aspect = float(aspect)
        p_grad_max_eff = float(p_grad_max_eff)
        print(
            f"  i {i:4d}  aspect {aspect:.8e}  p_grad_max_eff {p_grad_max_eff:.8e}  "
            f"({time.time() - t1:.2f}s)"
        )
        if (
            np.isfinite(aspect)
            and np.isfinite(p_grad_max_eff)
            and p_grad_max_eff > 0
            and aspect < best_aspect
        ):
            best_i = i
            best_aspect = aspect
            best_beta = p_grad_max_eff

    if best_i is None:
        print("No member with finite aspect and positive p_grad_max_eff. Nothing saved.")
        return

    best_x = np.asarray(population[best_i])
    jnp.save(args.out_x, best_x)
    jnp.save(
        args.out_metrics,
        {
            "index": best_i,
            "aspect": best_aspect,
            "p_grad_max_eff": best_beta,
        },
    )
    print(
        f"Best: i={best_i}  aspect={best_aspect:.8e}  p_grad_max_eff={best_beta:.8e}\n"
        f"Saved {args.out_x} and {args.out_metrics}"
    )


if __name__ == "__main__":
    main()
