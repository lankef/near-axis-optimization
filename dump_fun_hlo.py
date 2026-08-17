#!/usr/bin/env python
"""
Minimum reproduction of jitted `fun` from near_axis_optimization2.ipynb,
dumping jaxpr / StableHLO / XLA HLO in timed stages for compile-bottleneck analysis.

Usage (from this directory, GPU free):
  python dump_fun_hlo.py --order 2 --no-psi-crit --skip-compile   # fast probe
  python dump_fun_hlo.py --order 4                                 # reduced MHD
  python dump_fun_hlo.py --full                                    # notebook path (slow)
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# XLA dump + fresh cache MUST be set before importing JAX / shared.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "tmp" / "fun_hlo"
XLA_DUMP_DIR = HERE / "tmp" / "fun_xla_dump"
JAX_CACHE_DIR = HERE / "tmp" / "fun_jax_cache"

OUT_DIR.mkdir(parents=True, exist_ok=True)
XLA_DUMP_DIR.mkdir(parents=True, exist_ok=True)
JAX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Append (do not clobber user XLA_FLAGS) but ensure dump path is present.
_existing = os.environ.get("XLA_FLAGS", "")
_dump_flag = f"--xla_dump_to={XLA_DUMP_DIR} --xla_dump_hlo_as_text"
if "--xla_dump_to=" not in _existing:
    os.environ["XLA_FLAGS"] = (_existing + " " + _dump_flag).strip()

# Prefer a process-local cache so a prior hit does not skip the XLA dump.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(JAX_CACHE_DIR))

os.chdir(HERE)
sys.path.insert(0, str(HERE))

import jax  # noqa: E402

# Override shared.py's cache settings: keep dumps reproducible / not skipped.
jax.config.update("jax_compilation_cache_dir", str(JAX_CACHE_DIR))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import jax.numpy as jnp  # noqa: E402
from jax import jit  # noqa: E402

# Import shared after jax config so its config.update does not redirect dumps
# to a stale cache that skips XLA emission. shared.py still sets cache dirs;
# re-apply ours after import.
import shared as S  # noqa: E402

jax.config.update("jax_compilation_cache_dir", str(JAX_CACHE_DIR))

import aqsc  # noqa: E402


def _timed(label: str):
    class _T:
        def __enter__(self):
            print(f"[timer] {label} ...", flush=True)
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            self.dt = time.perf_counter() - self.t0
            print(f"[timer] {label}: {self.dt:.2f}s", flush=True)

    return _T()


def solve_to_order(x_flat, m=10, order=6, padded=True):
    """
    Unjitted solve path matching shared.solve_order_6, with optional early stop.
    order=2: leading_orders only
    order=4: + one iterate_2 (orders 3,4)
    order=6: + second iterate_2 (orders 5,6) — notebook path
    """
    in_dict = S.unravel_x(x_flat)

    Rc_raw = jnp.clip(in_dict["Rc_raw"], -S.RZ_clip, S.RZ_clip)
    Zs_raw = jnp.clip(in_dict["Zs_raw"], -S.RZ_clip, S.RZ_clip)
    pc_raw = in_dict["pc_raw"]

    chiphifunc_type = aqsc.ChiPhiFuncPadded if padded else aqsc.ChiPhiFunc

    weight_pc = S.spec_weight(len(pc_raw))
    weight_RZ = S.spec_weight(len(Rc_raw))
    pc_raw = pc_raw * weight_pc
    Rc_raw = Rc_raw * weight_RZ
    Zs_raw = Zs_raw * weight_RZ

    m_list = jnp.arange(len(in_dict["Rc_raw"]))
    pc_raw = jnp.where(m_list <= m, pc_raw, 0)
    Rc_raw = jnp.where(m_list <= m, Rc_raw, 0)
    Zs_raw = jnp.where(m_list <= m, Zs_raw, 0)

    Rc = list(Rc_raw)
    Zs = list(Zs_raw)
    Rc[0] = S.R0_fixed
    Rs = [0] * len(Rc)
    Zc = [0] * len(Zs)

    p_mode_num = jnp.arange(len(pc_raw)) * S.nfp
    p_Phi0 = S.phis_2pi / S.nfp
    p_cos_arr = jnp.cos(p_mode_num[:, None] * p_Phi0[None, :])
    p0 = jnp.sum(pc_raw[:, None] * p_cos_arr, axis=0)

    equilibrium = aqsc.leading_orders(
        nfp=S.nfp,
        Rc=Rc,
        Rs=Rs,
        Zc=Zc,
        Zs=Zs,
        p0=p0,
        Delta_0_avg=in_dict["Delta_0_avg"],
        B_alpha_1=in_dict["B_alpha_1"],
        B0=in_dict["B0"],
        B11c=in_dict["B11c"],
        B22c=in_dict["B22c"],
        B20=in_dict["B20"],
        B22s=in_dict["B22s"],
        B_theta_20_avg=in_dict["B_theta_20_avg"],
        len_phi=S.len_phi,
        len_phi_axis=500,
        static_max_freq=(S.static_freq, S.static_freq),
        traced_max_freq=(S.traced_freq, S.traced_freq),
        tol_riccati=1e-8,
        max_iter_riccati=50,
        n_shooting_riccati=1000,
        padded=padded,
    )
    if order <= 2:
        return equilibrium

    equilibrium = aqsc.iterate_2(
        equilibrium,
        B_denom_nm1=chiphifunc_type(
            jnp.array(in_dict["B_denom_3"])[:, None],
            equilibrium.nfp,
            trig_mode=True,
        ),
        B_denom_n=chiphifunc_type(
            jnp.array(in_dict["B_denom_4"])[:, None],
            equilibrium.nfp,
            trig_mode=True,
        ),
        B_alpha_nb2=in_dict["B_alpha_nb2_34"],
        static_max_freq=(S.static_freq, S.static_freq),
        traced_max_freq=(S.traced_freq, S.traced_freq),
    )
    if order <= 4:
        return equilibrium

    equilibrium = aqsc.iterate_2(
        equilibrium,
        B_denom_nm1=chiphifunc_type(
            jnp.array(in_dict["B_denom_5"])[:, None],
            equilibrium.nfp,
            trig_mode=True,
        ),
        B_denom_n=chiphifunc_type(
            jnp.array(in_dict["B_denom_6"])[:, None],
            equilibrium.nfp,
            trig_mode=True,
        ),
        B_alpha_nb2=in_dict["B_alpha_nb2_56"],
        static_max_freq=(S.static_freq, S.static_freq),
        traced_max_freq=(S.traced_freq, S.traced_freq),
    )
    return equilibrium


def make_fun(order: int, use_psi_crit: bool, padded: bool = True):
    """
    Single outer jit only — no nested jit(solve_order_6).
    Uses unjitted helpers from shared (eps_conv / rms_anisotropy paths still
    call @jit helpers; those are nested. For a clean dump of the MHD path we
    inline the no-psi-crit scalar when requested.
    """
    m = 10
    w_aspect, w_anisotropy, w_iota, w_p = S.w_aspect, S.w_anisotropy, S.w_iota, S.w_p

    def fun(x_flat):
        in_dict = S.unravel_x(x_flat)
        eq = solve_to_order(x_flat, m=m, order=order, padded=padded)

        if not use_psi_crit:
            # Minimal scalar that depends on the solve. Avoid eps_conv /
            # divergence_rate here: those index last-3 orders and fail when
            # order<=2 (B_psi series too short). iota is always available.
            return jnp.real(eq.iota.eval(psi=0, chi=0, phi=0))

        # Match notebook objective (padded=True), without going through
        # shared.objective which currently defaults padded=False and may
        # call the nested-jitted solve_order_6.
        # Full objective needs enough orders for divergence_rate (eps_conv).
        anisotropy = S.rms_anisotropy(eq)
        psi_crit, _, _ = eq.get_psi_crit()
        eps_crit_val = jnp.sqrt(psi_crit)
        eps_conv_val = S.eps_conv(eq)
        eps = jnp.minimum(eps_conv_val, eps_crit_val)
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
        term3 = w_iota * (jnp.maximum(jnp.abs(S.target_iota) - jnp.abs(iota_a), 0) / S.target_iota) ** 2
        term4 = w_p * (jnp.maximum(p_edge_eff - p_axis_eff, 0) / p_axis_eff) ** 2
        return term1 + term2 + term3 + term4

    return fun


def _count_eqns(jaxpr, counter=None):
    if counter is None:
        counter = collections.Counter()
    for eqn in jaxpr.eqns:
        counter[eqn.primitive.name] += 1
        for param in eqn.params.values():
            if hasattr(param, "jaxpr"):
                _count_eqns(param.jaxpr, counter)
            elif hasattr(param, "eqns"):  # ClosedJaxpr vs Jaxpr
                _count_eqns(param, counter)
    return counter


def _write(path: Path, text: str):
    path.write_text(text)
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.2f} MB)", flush=True)


def _summarize_primitives(counter: collections.Counter, path: Path):
    interesting = [
        "add",
        "mul",
        "sub",
        "div",
        "fft",
        "ifft",
        "dot_general",
        "gather",
        "scatter",
        "broadcast_in_dim",
        "concatenate",
        "slice",
        "dynamic_slice",
        "dynamic_update_slice",
        "pad",
        "reduce_sum",
        "reduce_min",
        "reduce_max",
        "while",
        "scan",
        "cond",
        "custom_call",
        "lu",
        "triangular_solve",
        "cholesky",
        "qr",
        "svd",
        "eig",
        "pjit",
        "jit",
        "xla_call",
    ]
    lines = [f"total eqns (recursive): {sum(counter.values())}", ""]
    lines.append("=== interesting primitives ===")
    for name in interesting:
        if counter[name]:
            lines.append(f"{name:30s} {counter[name]:8d}")
    lines.append("")
    lines.append("=== top 40 by count ===")
    for name, n in counter.most_common(40):
        lines.append(f"{name:30s} {n:8d}")
    _write(path, "\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--order", type=int, choices=(2, 4, 6), default=2,
                   help="Max NAE order to solve (default 2 = leading_orders only)")
    p.add_argument("--full", action="store_true",
                   help="Notebook path: order=6 with get_psi_crit")
    p.add_argument("--no-psi-crit", action="store_true",
                   help="Return eps_conv(eq) only (skip get_psi_crit / aspect / etc.)")
    p.add_argument("--skip-compile", action="store_true",
                   help="Stop after jaxpr + StableHLO lower (no GPU XLA optimize)")
    p.add_argument("--padded", action=argparse.BooleanOptionalAction, default=True,
                   help="Use ChiPhiFuncPadded backend (default True)")
    args = p.parse_args()

    order = 6 if args.full else args.order
    # --full implies psi-crit; --no-psi-crit always wins.
    if args.no_psi_crit:
        use_psi_crit = False
    elif args.full:
        use_psi_crit = True
    else:
        # Default: include psi_crit / full-ish objective only when order>=4,
        # so --order 2 stays a cheap leading_orders probe.
        use_psi_crit = order >= 4

    tag = f"order{order}_{'psicrit' if use_psi_crit else 'nopsicrit'}_{'padded' if args.padded else 'ragged'}"
    print(f"device={jax.devices()}", flush=True)
    print(f"tag={tag}", flush=True)
    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS')}", flush=True)
    print(f"static_freq={S.static_freq} traced_freq={S.traced_freq} len_phi={S.len_phi}", flush=True)

    x = S.x_flat_init
    fun = make_fun(order=order, use_psi_crit=use_psi_crit, padded=args.padded)
    fun_jit = jit(fun)

    # 1) Trace
    with _timed("make_jaxpr"):
        closed = jax.make_jaxpr(fun)(x)
    jaxpr_path = OUT_DIR / f"{tag}.jaxpr.txt"
    with _timed("write jaxpr"):
        _write(jaxpr_path, str(closed))
    counter = _count_eqns(closed.jaxpr)
    _summarize_primitives(counter, OUT_DIR / f"{tag}.primitives.txt")

    # 2) Lower
    with _timed("lower"):
        lowered = fun_jit.lower(x)
    with _timed("write lowered as_text"):
        try:
            _write(OUT_DIR / f"{tag}.lowered.txt", lowered.as_text())
        except Exception as e:
            _write(OUT_DIR / f"{tag}.lowered.txt", f"as_text failed: {e!r}\n")
    with _timed("write stablehlo"):
        try:
            ir = lowered.compiler_ir("stablehlo")
            _write(OUT_DIR / f"{tag}.stablehlo.mlir", str(ir))
        except Exception as e:
            _write(OUT_DIR / f"{tag}.stablehlo.mlir", f"compiler_ir failed: {e!r}\n")

    if args.skip_compile:
        print("skipping compile (--skip-compile)", flush=True)
        print(f"artifacts in {OUT_DIR}", flush=True)
        return

    # 3) Compile (triggers XLA dump)
    with _timed("compile"):
        compiled = lowered.compile()
    try:
        cost = compiled.cost_analysis()
        _write(OUT_DIR / f"{tag}.cost_analysis.txt", str(cost))
    except Exception as e:
        _write(OUT_DIR / f"{tag}.cost_analysis.txt", f"cost_analysis failed: {e!r}\n")
    try:
        mem = compiled.memory_analysis()
        _write(OUT_DIR / f"{tag}.memory_analysis.txt", str(mem))
    except Exception as e:
        _write(OUT_DIR / f"{tag}.memory_analysis.txt", f"memory_analysis failed: {e!r}\n")

    # Smoke-run once so any remaining lazy compile happens under the dump flags.
    with _timed("first eval"):
        y = compiled(x)
        y.block_until_ready()
        print(f"  fun(x) = {y}", flush=True)

    print(f"artifacts in {OUT_DIR}", flush=True)
    print(f"XLA dump in {XLA_DUMP_DIR}", flush=True)
    dump_files = sorted(XLA_DUMP_DIR.glob("*"))
    print(f"  {len(dump_files)} dump files", flush=True)
    for f in dump_files[:20]:
        print(f"  - {f.name} ({f.stat().st_size / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
