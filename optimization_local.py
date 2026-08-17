import time
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.optimize import minimize

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)

from shared import *

# ---------------------------------------------------------------------------
# Load the best point from the global stage and set up the objective.
# ---------------------------------------------------------------------------
fitness = jnp.load("./fitness.npy")
population = jnp.load("./population.npy")
x_fin = population[jnp.argmin(fitness)]

m_opt = 10
# padded=False on purpose: with padded=True, `ChiPhiEpsFunc.append` in the
# runtime pyAQSC clone falls into `elif jnp.array(item).ndim!=0` for
# ChiPhiFuncPadded inputs (its isinstance check only covers ChiPhiFunc),
# raising TypeError from `iterate_2`. Non-padded backend takes the same
# path and works without a library-side fix.
fun = lambda x: objective(
    x, w_aspect, w_anisotropy, w_iota, w_p, m_opt,
    full_mode=False, padded=False,
)

# Primal-only JIT. Order-6 reverse-mode AD (`jax.grad(fun)`) unrolls a
# ~1e6 equation graph that XLA either OOMs or takes >30min to compile,
# so we avoid `grad(fun)` entirely and use vmapped central differences
# instead. `vmap` batches the perturbed forward passes into a single
# compiled kernel, so 2N primal evaluations reuse one XLA module.
obj_jit = jax.jit(fun)


# ---------------------------------------------------------------------------
# vmap'd central-difference gradient
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("h",))
def fd_grad(x_flat, h=1e-6):
    """Central-difference gradient of `fun` at `x_flat`, batched via vmap.

    Builds a stack of 2N perturbed points x +/- h*e_i and evaluates `fun`
    on all of them in a single vmap'd kernel. Compilation cost is one
    primal JIT, and runtime is ~2N * (single primal cost / batching speedup).
    Much cheaper than compiling reverse-mode AD through the order-6 graph.
    """
    n = x_flat.shape[0]
    eye = jnp.eye(n, dtype=x_flat.dtype)
    x_plus = x_flat[None, :] + h * eye
    x_minus = x_flat[None, :] - h * eye
    f_plus = jax.vmap(fun)(x_plus)
    f_minus = jax.vmap(fun)(x_minus)
    return (f_plus - f_minus) / (2.0 * h)


# scipy expects plain numpy arrays / python floats.
def fun_np(x_np):
    return float(obj_jit(jnp.asarray(x_np)))


def jac_np(x_np):
    g = fd_grad(jnp.asarray(x_np))
    # Guard against occasional non-finite entries from the underlying
    # equilibrium solve (e.g. near singular configurations).
    g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(g, dtype=np.float64)


# ---------------------------------------------------------------------------
# Warm up: compile obj_jit and fd_grad once so the reported iteration
# times reflect steady-state cost.
# ---------------------------------------------------------------------------
print("Warming up primal JIT ...")
t0 = time.time()
v0 = fun_np(x_fin)
print(f"  fun(x_fin) = {v0:.6e}  ({time.time() - t0:.2f}s)")

print("Warming up vmap'd finite-difference gradient ...")
t0 = time.time()
g0 = jac_np(np.asarray(x_fin))
print(
    f"  ||grad|| = {np.linalg.norm(g0):.3e}  "
    f"(compile+first eval {time.time() - t0:.2f}s)"
)

t0 = time.time()
_ = jac_np(np.asarray(x_fin))
print(f"  Second gradient eval: {time.time() - t0:.3f}s")


# ---------------------------------------------------------------------------
# Optimization with scipy L-BFGS-B, using the vmap'd FD gradient.
# ---------------------------------------------------------------------------
class Callback:
    def __init__(self):
        self.n = 0
        self.t0 = time.time()

    def __call__(self, xk):
        val = fun_np(xk)
        elapsed = time.time() - self.t0
        print(
            f"iter {self.n:4d} | f = {val:.6e} | "
            f"elapsed {elapsed:7.1f}s"
        )
        self.n += 1


callback = Callback()
niter = 1000

print("\nStarting L-BFGS-B ...")
sol = minimize(
    fun_np,
    np.asarray(x_fin, dtype=np.float64),
    method="L-BFGS-B",
    jac=jac_np,
    callback=callback,
    options=dict(maxiter=niter, disp=True, gtol=1e-8, ftol=1e-10),
)

x_opt = jnp.asarray(sol.x)
print(f"\nFinal objective: {fun_np(sol.x):.6e}")
print(f"scipy message : {sol.message}")

jnp.save("local_result", x_opt)
