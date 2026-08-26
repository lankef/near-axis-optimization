import importlib
import aqsc
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "./tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

from jax import flatten_util, jit
from jax.lax import while_loop

# Constants and resolution
# static_max_freq sets the FFT tensor size in iterate_looped (len_tensor =
# min(len_phi, 2*static_freq)). HLO dumps of fun showed cusolver LU on
# c128[400,400] at order 4 with static_freq=100; matching traced_freq cuts
# that to 2*40=80 modes and shrinks the tensorsolve markedly.
static_freq = 100 # low-pass / FFT-tensor freq (was 100; see dump_fun_hlo.py)
traced_freq = 40 # low-pass filter freq
len_phi = 100 # number of phi sample points
phis_2pi = jnp.linspace(0,2*jnp.pi, len_phi, endpoint=False)
nfp = 2
R0_fixed = 1.0 # fixed major radius
RZ_clip = 0.3 # fixing min and max R and Z coeffs to prevent crazy axes

# Targets
target_iota = 0.1
target_anisotropy = 1e-2
target_aspect = 25.
target_beta = 0.02
population_size = 200
w_aspect, w_anisotropy, w_iota, w_p = 6., 0.5, 1., 1.
std_init = 0.5


def spec_weight(n):
    return jnp.e**(-jnp.arange(n))

Rc = jnp.array([1.,   0.1,   0., 0., 0., 0., 0., 0.])
Zs = jnp.array([0.,   0.1,   0., 0., 0., 0., 0., 0.])
pc = jnp.array([0.05, 0.005, 0., 0., 0., 0., 0., 0.])

# Spectral weighting to make optimization converge better
Rc_raw = Rc/spec_weight(len(Rc))
Zs_raw = Zs/spec_weight(len(Zs))
pc_raw = pc/spec_weight(len(pc))

# Initial cond and unraveling
init_dict = {
    'Rc_raw': Rc_raw,
    'Zs_raw': Zs_raw,
    # 'Rc': qa_Rc,
    # 'Zs': qa_Zs,
    'pc_raw': pc_raw,
    'Delta_0_avg': 0.,
    'B_theta_20_avg': 0.,
    'B_alpha_1': 0.1,
    'B0': 1.0,
    'B11c': -1.8,
    'B22c': 0.,
    'B20': 0., 
    'B22s': 0.,
    'B_alpha_nb2_34': 0.,
    'B_alpha_nb2_56': 0.,
    'B_denom_3': [0., 0.], # 0., 0.], # B33s, B31s, B31c, B33c
    'B_denom_4': [0.,], # 0., 0., 0., 0.], # B33s, B31s, B31c, B33c
    'B_denom_5': [0., 0.,], # 0., 0., 0., 0.], # B33s, B31s, B31c, B33c
    'B_denom_6': [0.,], # 0., 0., 0., 0., 0., 0.], # B33s, B31s, B31c, B33c
}

x_flat_init, unravel_x = flatten_util.ravel_pytree(init_dict)

# Not jitted here on purpose: callers wrap objective / fun in a single outer
# jit. Nesting jit(solve_order_6) inside jit(fun) duplicates XLA modules.
def solve_order_6(x_flat, m=10, padded=True):
    in_dict = unravel_x(x_flat)
    
    Rc_raw = in_dict['Rc_raw']
    Zs_raw = in_dict['Zs_raw']
    pc_raw = in_dict['pc_raw']
    
    # Clipping the fourier coefficients
    # to prevent crazy axis
    Rc_raw = jnp.clip(Rc_raw, -RZ_clip, RZ_clip)
    Zs_raw = jnp.clip(Zs_raw, -RZ_clip, RZ_clip)

    if padded:
        chiphifunc_type = aqsc.ChiPhiFuncPadded
    else:
        chiphifunc_type = aqsc.ChiPhiFunc
    
    # Weighting Fourier coeffs of pressure and axis shape
    weight_pc = spec_weight(len(pc_raw))
    weight_RZ = spec_weight(len(Rc_raw))
    
    pc_raw = pc_raw * weight_pc
    Rc_raw = Rc_raw * weight_RZ
    Zs_raw = Zs_raw * weight_RZ
    
    m_list = jnp.arange(len(in_dict['Rc_raw']))
    pc_raw = jnp.where(m_list<=m, pc_raw, 0)
    Rc_raw = jnp.where(m_list<=m, Rc_raw, 0)
    Zs_raw = jnp.where(m_list<=m, Zs_raw, 0)
    
    pc = pc_raw
    Rc = list(Rc_raw)
    Zs = list(Zs_raw)
    
    # Fixing major radius
    Rc[0] = R0_fixed
    
    # Stellsym axis
    Rs = [0] * len(Rc)
    Zc = [0] * len(Zs)

    Delta_0_avg = in_dict['Delta_0_avg']
    B_theta_20_avg = in_dict['B_theta_20_avg']
    B_alpha_1 = in_dict['B_alpha_1']
    B0 = in_dict['B0']
    B11c = in_dict['B11c']
    B22c = in_dict['B22c']
    B20 = in_dict['B20']
    B22s = in_dict['B22s']
    B_alpha_nb2_34 = in_dict['B_alpha_nb2_34']
    B_alpha_nb2_56 = in_dict['B_alpha_nb2_56']
    B_denom_3 = in_dict['B_denom_3']
    B_denom_4 = in_dict['B_denom_4']
    B_denom_5 = in_dict['B_denom_5']
    B_denom_6 = in_dict['B_denom_6']
    # Reconstructing the p array from cos coeffs
    p_mode_num = jnp.arange(len(pc))*nfp
    p_Phi0 = phis_2pi/nfp
    p_phi_times_mode = p_mode_num[:, None]*p_Phi0[None, :]
    p_cos_arr = jnp.cos(p_phi_times_mode)
    p0 = jnp.sum(pc[:, None]*p_cos_arr, axis=0)
    equilibrium = aqsc.leading_orders(
        nfp=nfp,
        Rc=Rc,
        Rs=Rs,
        Zc=Zc,
        Zs=Zs,
        p0=p0,
        Delta_0_avg=Delta_0_avg,
        B_alpha_1=B_alpha_1,
        B0=B0,
        B11c=B11c,
        B22c=B22c,
        B20=B20,
        B22s=B22s,
        B_theta_20_avg=B_theta_20_avg,
        len_phi=len_phi,
        len_phi_axis=500,
        static_max_freq=(static_freq, static_freq),
        traced_max_freq=(traced_freq, traced_freq),
        tol_riccati=1e-8,
        max_iter_riccati=50,
        n_shooting_riccati=1000,
        padded=padded
    )
    equilibrium = aqsc.iterate_2(
        equilibrium,
        B_denom_nm1 = chiphifunc_type(
            jnp.array(B_denom_3)[:, None],
            equilibrium.nfp,
            trig_mode=True
        ), # B3
        B_denom_n = chiphifunc_type(
            jnp.array(B_denom_4)[:, None],
            equilibrium.nfp,
            trig_mode=True
        ), # B4
        B_alpha_nb2 = B_alpha_nb2_34,
        static_max_freq=(static_freq, static_freq),
        traced_max_freq=(traced_freq, traced_freq),
    )
    equilibrium = aqsc.iterate_2(
        equilibrium,
        B_denom_nm1 = chiphifunc_type(
            jnp.array(B_denom_5)[:, None],
            equilibrium.nfp,
            trig_mode=True
        ), # B5
        B_denom_n = chiphifunc_type(
            jnp.array(B_denom_6)[:, None],
            equilibrium.nfp,
            trig_mode=True
        ), # B6
        B_alpha_nb2 = B_alpha_nb2_56,
        static_max_freq=(static_freq, static_freq),
        traced_max_freq=(traced_freq, traced_freq),
    )
    return equilibrium


def divergence_rate(eq):
    amp_B_psi_coef_cp = eq.unknown['B_psi_coef_cp'].get_l2_order_by_order()
    amp_B_theta_coef_cp = eq.unknown['B_theta_coef_cp'].get_l2_order_by_order()
    amp_Delta_coef_cp = eq.unknown['Delta_coef_cp'].get_l2_order_by_order()
    amp_X_coef_cp = eq.unknown['X_coef_cp'].get_l2_order_by_order()
    amp_Y_coef_cp = eq.unknown['Y_coef_cp'].get_l2_order_by_order()
    amp_Z_coef_cp = eq.unknown['Z_coef_cp'].get_l2_order_by_order()
    amp_p_perp_coef_cp = eq.unknown['p_perp_coef_cp'].get_l2_order_by_order()
    rate_X_coef_cp = amp_X_coef_cp[-2:] / amp_X_coef_cp[-3:-1]
    rate_Y_coef_cp = amp_Y_coef_cp[-2:] / amp_Y_coef_cp[-3:-1]
    rate_Z_coef_cp = amp_Z_coef_cp[-2:] / amp_Z_coef_cp[-3:-1]
    rate_B_psi_coef_cp = amp_B_psi_coef_cp[-2:] / amp_B_psi_coef_cp[-3:-1]
    rate_B_theta_coef_cp = amp_B_theta_coef_cp[-2:] / amp_B_theta_coef_cp[-3:-1]
    rate_Delta_coef_cp = amp_Delta_coef_cp[-2:] / amp_Delta_coef_cp[-3:-1]
    rate_p_perp_coef_cp = amp_p_perp_coef_cp[-2:] / amp_p_perp_coef_cp[-3:-1]
    all_rates = jnp.array([
        rate_X_coef_cp,
        rate_Y_coef_cp,
        rate_Z_coef_cp,
        rate_B_psi_coef_cp,
        rate_B_theta_coef_cp,
        rate_Delta_coef_cp,
        rate_p_perp_coef_cp,
    ])
    return jnp.max(all_rates)

def eps_conv(eq):
    return 1 / divergence_rate(eq)

# Helpers below are intentionally not @jit: the notebook wraps objective /
# grad(objective) in a single outer jit. Nested jit fragments the AD graph
# (dump_jac_compare.py saw O(1e5) pjit nodes inside grad jaxprs).
def p_rms_vol(eq, eps, eps2, n_max=float('inf')):
    vol = eq.volume_eps(eps=eps, n_max=n_max)
    vol2 = eq.volume_eps(eps=eps2, n_max=n_max)
    p2 = eq.p_perp * eq.p_perp
    p2_int = eq.volume_integral(p2, n_max=n_max).eval_eps(eps, 0., 0.)
    p2_int2 = eq.volume_integral(p2, n_max=n_max).eval_eps(eps2, 0., 0.)
    p2_int_shell = p2_int - p2_int2
    vol_shell = vol - vol2
    p_rms = jnp.sqrt(jnp.real(
        p2_int_shell/vol_shell
    ))
    return p_rms

def vol_conv(eq):
    eps = eps_conv(eq)
    return eq.volume_eps(eps)

def aspect_conv(eq):
    eps = eps_conv(eq)
    return eq.aspect_ratio_eps(eps)

def aspect_crit(eq):
    psi, _, _ = eq.get_psi_crit()
    return eq.aspect_ratio_eps(jnp.sqrt(psi))

def rms_anisotropy(eq):
    eps = eps_conv(eq)
    int_anisotropy = jnp.real(
        eq.volume_integral(eq.Delta*eq.Delta).eval_eps(eps=eps, chi=0, phi=0)
    )
    return jnp.sqrt(int_anisotropy/vol_conv(eq))

def rms_p(eq):
    eps = eps_conv(eq)
    int_p_perp = jnp.real(
        eq.volume_integral(eq.p_perp*eq.p_perp).eval_eps(eps=eps, chi=0, phi=0)
    )
    return jnp.sqrt(int_p_perp/vol_conv(eq))

def iota_axis(eq):
    return jnp.real(eq.iota.eval(psi=0, chi=0, phi=0))

def objective(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m, full_mode=False, padded=True):
    in_dict = unravel_x(x_flat)
    eq = solve_order_6(x_flat, m, padded=padded)
    anisotropy = rms_anisotropy(eq)

    psi_crit, _, _ = eq.get_psi_crit()
    eps_crit_val = jnp.sqrt(psi_crit)
    eps_conv_val = eps_conv(eq)
    eps = jnp.minimum(eps_conv_val, eps_crit_val)
    aspect = eq.aspect_ratio_eps(eps)

    # Iota
    iota_a = jnp.real(eq.iota.eval(psi=0, chi=0, phi=0))

    # Effective field strength at the edge
    B_denom_20 = jnp.real(in_dict['B20'])
    B_denom_0 = jnp.real(in_dict['B0'])
    B_denom_edge_eff = (B_denom_0 + B_denom_20*eps**2)
    # Effective pressure at the edge
    p20_avg = jnp.real(jnp.average(eq.p_perp[2][0].content))
    p10_avg = jnp.real(jnp.average(eq.p_perp[1][0].content))
    p00_avg = jnp.real(jnp.average(eq.p_perp[0][0].content))
    # effective beta
    p_eff_axis = jnp.real(eq.p_perp.eval_eps(eps=0, chi=0, phi=0))
    p_eff_edge = jnp.real(eq.p_perp.eval_eps(eps=eps, chi=0, phi=0)) 
    beta_eff = (p_eff_axis - p_eff_edge) * B_denom_0
    
    # Terms
    term1 = w_aspect * (
        jnp.maximum(aspect - target_aspect, 0) / target_aspect
    )**2
    term2 = w_anisotropy * (
        jnp.maximum(anisotropy - target_anisotropy, 0) / target_anisotropy
    )**2
    term3 = w_iota * (
        jnp.maximum(jnp.abs(target_iota) - jnp.abs(iota_a), 0) / target_iota
    )**2
    term4 = w_p * (
        jnp.maximum(- beta_eff, 0) * 10000
    )**2
    # term4 = w_p * (
    #     jnp.maximum(p20_avg * eps**2, 0) / p00_avg
    # )**2
    # term5 = w_p * (
    #     jnp.maximum(p10_avg * eps, 0) / p00_avg
    # )**2
    out = term1 + term2 + term3 + term4 #  + term5
    if full_mode:
        return {
            'eps_crit': eps_crit_val,
            'eps_conv': eps_conv_val,
            'aspect': aspect,
            'anisotropy': anisotropy,
            'p_eff_axis': p_eff_axis,
            'p_eff_edge': p_eff_edge,
            'beta_eff': beta_eff,
            'iota_a': iota_a,
            'loss': out,
        }
            
    return out

def obj_wrapped(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m):
    out = objective(x_flat, w_aspect, w_anisotropy, w_iota, w_p, m)
    return jnp.clip(out, a_min=-1e10, a_max=1e10)