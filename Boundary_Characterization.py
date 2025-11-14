# -*- coding: utf-8 -*-
"""
PPP Monte Carlo validation for §1.5 optimal strategy boundary (Δ per Eq. (3b))
- Heatmap and slices use  Δ(θ, ρ) = α(ρ) - θ b(ρ)
  with α(ρ) = 1 - (1+ρ)^(-1/2) and b(ρ) = κ(ρ) ρ/(1+ρ).
- MC estimators are scaled by 2*sqrt(λ_C) to match the dimensionless form:
    alpha_bar = (2*sqrt(λ_C)) * a_bar
    b_bar_dimless = (2*sqrt(λ_C)) * b_bar
  where a_bar = E[D_C - D], b_bar = E[1_{nearest=PSC} * Y] (in distance units),
        λ_C = 1/(1+ρ).
- θ*(ρ) = a/b is invariant to this scaling; boundary curves and errors are unchanged.

Dependencies: numpy, scipy, matplotlib, mpmath, pandas
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import mpmath as mp

# ---------- Aesthetics ----------
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = True
mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["legend.frameon"] = False

# ---------- Experiment parameters (paper settings) ----------
L = 20.0
K = 200
M = 10000
SEED = 20251022
RHO_GRID = np.array([0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0], dtype=float)

# Plotting theta range: fixed [0, 0.8]
THETA_MIN, THETA_MAX, THETA_N = 0.0, 0.8, 61
THETA_GRID = np.linspace(THETA_MIN, THETA_MAX, THETA_N)

PLOT_FIGURES = True
SAVE_FIGURES = True
OUTPUT_DIR = "./outputs_ppp_boundary_dimless"

rng = np.random.default_rng(SEED)

# ---------- Torus KDTree (3x3 copies) ----------
def build_periodic_tree(points, L):
    n = points.shape[0]
    if n == 0:
        raise RuntimeError("Empty point set.")
    offsets = np.array([[dx, dy] for dx in (-L, 0.0, L) for dy in (-L, 0.0, L)], dtype=float)
    tiled = (points[None, :, :] + offsets[:, None, :]).reshape(-1, 2)
    tree = cKDTree(tiled)
    return tree, n

def query_periodic_tree(tree, base_size, query_pts):
    dist, idx = tree.query(query_pts, k=1)
    base_idx = idx % base_size
    return dist, base_idx

# ---------- PPP sampling ----------
def sample_ppp_points(lam, A, L, rng):
    N = rng.poisson(lam * A)
    pts = rng.uniform(0.0, L, size=(N, 2)) if N > 0 else np.zeros((0, 2), dtype=float)
    return pts

# ---------- One instance: compute a_k, b_k (distance units) ----------
def simulate_one_instance(rho, L, M, rng):
    A = L * L
    lam_C = 1.0 / (1.0 + rho)
    lam_P = rho  / (1.0 + rho)

    # ensure at least one facility of each type
    while True:
        CSC = sample_ppp_points(lam_C, A, L, rng)
        PSC = sample_ppp_points(lam_P, A, L, rng)
        if CSC.shape[0] > 0 and PSC.shape[0] > 0:
            break

    tree_csc, nC = build_periodic_tree(CSC, L)
    dist_psc_to_csc, _ = query_periodic_tree(tree_csc, nC, PSC)

    PTS = rng.uniform(0.0, L, size=(M, 2))
    D_C, _ = query_periodic_tree(tree_csc, nC, PTS)

    tree_psc, nP = build_periodic_tree(PSC, L)
    D_P, psc_idx_base = query_periodic_tree(tree_psc, nP, PTS)

    D = np.minimum(D_C, D_P)
    nearest_is_psc = (D_P < D_C)  # ties negligible

    Y = np.zeros(M, dtype=float)
    if np.any(nearest_is_psc):
        Y[nearest_is_psc] = dist_psc_to_csc[psc_idx_base[nearest_is_psc]]

    # distance units
    a_k = float(np.mean(D_C - D))
    b_k = float(np.mean(nearest_is_psc.astype(float) * Y))

    sanity = dict(
        N_C=int(CSC.shape[0]),
        N_P=int(PSC.shape[0]),
        p_near_psc_hat=float(np.mean(nearest_is_psc.astype(float))),
        D_mean=float(np.mean(D)),
    )
    return a_k, b_k, sanity

# ---------- κ(ρ) with analytic tail (high precision) ----------
mp.mp.dps = 80
pi = mp.pi

def _clamp_unit(x):
    return min(1.0, max(-1.0, x))

def h_dimless(x):
    x = mp.mpf(x)
    if x < 2:
        a = _clamp_unit(float(x/2))
        b = _clamp_unit(float(1 - x**2/2))
        term = (x**2 * mp.acos(a) + mp.acos(b) - (x/2) * mp.sqrt(max(0, 4 - x**2)))
        return x**2 - (1/pi) * term
    else:
        return x**2 - 1

def kappa_rho(rho):
    rho = mp.mpf(rho)
    g = lambda t: ((1+rho) + h_dimless(t))**(-mp.mpf('1.5'))
    I0_2 = mp.quad(g, [0, 0.9, 1.8, 1.95, 1.99, 2.0])
    # analytic tail over (2, ∞)
    tail = (1/rho) * (1 - 2/mp.sqrt(rho + 4))
    return float((1+rho) * (I0_2 + tail))

# ---------- Theory (Δ per Eq. 3b) ----------
def alpha_theory(rho):
    return 1.0 - 1.0/np.sqrt(1.0 + rho)

def b_theory(rho):
    return (rho / (1.0 + rho)) * kappa_rho(rho)

def theta_star_theory(rho):
    return alpha_theory(rho) / b_theory(rho)

# ---------- Analytic threshold curves from κ ∈ (1, 1.193) ----------
def theta_U_curve(r):  # κ = 1  → MS guaranteed (θ above this)
    r = np.asarray(r, dtype=float)
    return alpha_theory(r) * (1.0 + r) / r

def theta_L_curve(r):  # κ = 1.193 → DS guaranteed (θ below this)
    return theta_U_curve(r) / 1.193

# ---------- Run across ρ ----------
def run_experiment(rho_grid, K, M, L, rng):
    rows, sanity_rows = [], []
    for rho in rho_grid:
        a_list, b_list, S = [], [], []
        for _ in range(K):
            a_k, b_k, sdict = simulate_one_instance(rho, L, M, rng)
            a_list.append(a_k); b_list.append(b_k); S.append(sdict)

        a_arr, b_arr = np.array(a_list), np.array(b_list)
        a_bar, b_bar = float(a_arr.mean()), float(b_arr.mean())

        if K > 1:
            s_a2 = float(a_arr.var(ddof=1))
            s_b2 = float(b_arr.var(ddof=1))
            s_ab = float(np.cov(a_arr, b_arr, ddof=1)[0, 1])
        else:
            s_a2 = s_b2 = s_ab = 0.0

        theta_star_mc = (a_bar / b_bar) if b_bar > 0 else np.nan
        if K > 1 and b_bar > 0:
            var_theta = (s_a2 / (K * b_bar**2)
                         + (a_bar**2) / (K * b_bar**4) * s_b2
                         - (2 * a_bar) / (K * b_bar**3) * s_ab)
            var_theta = max(0.0, var_theta)
            se_theta = float(np.sqrt(var_theta))
            ci_lo = theta_star_mc - 1.96 * se_theta
            ci_hi = theta_star_mc + 1.96 * se_theta
        else:
            se_theta = ci_lo = ci_hi = np.nan

        theta_star_th = theta_star_theory(rho)

        rows.append(dict(
            rho=rho, a_bar=a_bar, b_bar=b_bar,
            theta_star_mc=theta_star_mc,
            theta_star_mc_se=se_theta,
            theta_star_mc_lo=ci_lo, theta_star_mc_hi=ci_hi,
            theta_star_th=theta_star_th
        ))

        s_df = pd.DataFrame(S)
        sanity_rows.append(dict(
            rho=rho,
            N_C_mean=float(s_df["N_C"].mean()),
            N_P_mean=float(s_df["N_P"].mean()),
            D_mean=float(s_df["D_mean"].mean()),
            p_near_psc_hat=float(s_df["p_near_psc_hat"].mean()),
            p_near_psc_theory=rho / (1.0 + rho),
            p_near_psc_err=float(s_df["p_near_psc_hat"].mean() - rho / (1.0 + rho)),
        ))

        print(f"[rho={rho:>5}] ā={a_bar:.6f}, b̄={b_bar:.6f}, "
              f"θ*MC={theta_star_mc:.4f} [{ci_lo:.4f},{ci_hi:.4f}] | θ*th={theta_star_th:.4f}")

    return pd.DataFrame(rows), pd.DataFrame(sanity_rows)

# ---------- Figures (Δ for heatmap & slices) ----------
def make_heatmap_and_boundaries(df_main, theta_grid, rho_grid, outdir):
    A = df_main.set_index("rho")
    R, T = rho_grid, theta_grid
    X, Y = np.meshgrid(R, T)

    # Build Δ_MC(θ,ρ) column by column
    Delta = np.zeros((len(T), len(R)))
    for j, rho in enumerate(R):
        a_bar = A.loc[rho, "a_bar"]
        b_bar = A.loc[rho, "b_bar"]
        lam_c = 1.0 / (1.0 + rho)
        scale = 2.0 * np.sqrt(lam_c)           # = 2 / sqrt(1+rho)
        alpha_bar = scale * a_bar               #  α̂
        b_bar_dimless = scale * b_bar           #  b̂
        Delta[:, j] = alpha_bar - T * b_bar_dimless

    # Symmetric color range about 0 using 98th percentile of |Δ|
    vmax = float(np.percentile(np.abs(Delta), 98))
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(8.4, 5.4))
    ax = fig.add_subplot(111)
    cf = ax.contourf(X, Y, Delta, levels=256, cmap="coolwarm", norm=norm)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(r" $\Delta(\theta,\rho)$")

    # Δ=0 isoline
    ax.contour(X, Y, Delta, levels=[0.0], colors="k", linewidths=1.6, zorder=3)

    # Theory & MC boundaries (θ* unaffected by scaling)
    theta_th = A["theta_star_th"].values
    ax.plot(R, theta_th, lw=2.2, color="#1f77b4", label=r"$\theta_{\mathrm{th}}^*(\rho)$", zorder=4)

    theta_mc = A["theta_star_mc"].values
    lo = A["theta_star_mc_lo"].values
    hi = A["theta_star_mc_hi"].values
    ax.plot(R, theta_mc, marker='o', linestyle='none', markersize=5, color="#ff7f0e",
            label=r"$\theta_{\mathrm{sim}}^*(\rho)$", zorder=4)
    for x, l, h in zip(R, lo, hi):
        if np.isfinite(l) and np.isfinite(h):
            ax.plot([x, x], [l, h], lw=1.2, color="#ff7f0e", zorder=4)

    # ===== NEW: analytic threshold curves (two dashed lines) =====
    rho_curve = np.logspace(np.log10(R.min()), np.log10(R.max()), 600)
    theta_U = theta_U_curve(rho_curve)   # κ=1 → MS guaranteed if θ > theta_U
    theta_L = theta_L_curve(rho_curve)   # κ=1.193 → DS guaranteed if θ < theta_L
    # 仅绘制落入图窗的部分
    mU = (theta_U >= THETA_MIN) & (theta_U <= THETA_MAX)
    mL = (theta_L >= THETA_MIN) & (theta_L <= THETA_MAX)
    ax.plot(rho_curve[mL], theta_L[mL], ls='--', lw=1.3, color='#7f7f7f', alpha=0.95,
            label=r"$\theta_{\mathrm{L}}(\rho)$ (DS guaranteed)")
    ax.plot(rho_curve[mU], theta_U[mU], ls='--', lw=1.3, color='#ff7f0e', alpha=0.95,
            label=r"$\theta_{\mathrm{U}}(\rho)$ (MS guaranteed)")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho = \lambda_P / \lambda_C$ (log scale)")
    ax.set_ylabel(r"$\theta$")
    ax.set_ylim(T.min(), T.max())

    # Legend to bottom-right with translucent white background
    leg = ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.96, 0.06),
        borderaxespad=0.6,
        frameon=True
    )
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_facecolor("white")

    # Region labels（按你之前的约定位置）
    ax.text(0.15, 0.86, "MS better\n($\\Delta<0$)", transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=4), zorder=6)
    ax.text(0.15, 0.10, "DS better\n($\\Delta>0$)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=4), zorder=6)

    ax.set_title("expected distance difference $\\Delta(\\theta,\\rho)$ with boundaries")
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(os.path.join(outdir, "heatmap_delta_mc_dimless.png"), bbox_inches="tight")
        fig.savefig(os.path.join(outdir, "heatmap_delta_mc_dimless.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Same figure with linear-ρ inset for [0,2]
    fig = plt.figure(figsize=(8.4, 5.4))
    ax2 = fig.add_subplot(111)
    cf2 = ax2.contourf(X, Y, Delta, levels=256, cmap="coolwarm", norm=norm)
    cbar2 = fig.colorbar(cf2, ax=ax2, pad=0.02)
    cbar2.set_label(r" $\Delta(\theta,\rho)$")
    ax2.contour(X, Y, Delta, levels=[0.0], colors="k", linewidths=1.6, zorder=3)
    ax2.plot(R, theta_th, lw=2.2, color="#1f77b4")
    ax2.plot(R, theta_mc, marker='o', linestyle='none', markersize=5, color="#ff7f0e")
    for x, l, h in zip(R, lo, hi):
        if np.isfinite(l) and np.isfinite(h):
            ax2.plot([x, x], [l, h], lw=1.2, color="#ff7f0e")

    # ===== NEW: 两条解析阈值曲线也叠加到主图上 =====
    mU2 = (theta_U >= THETA_MIN) & (theta_U <= THETA_MAX)
    mL2 = (theta_L >= THETA_MIN) & (theta_L <= THETA_MAX)
    ax2.plot(rho_curve[mL2], theta_L[mL2], ls='--', lw=1.3, color='#7f7f7f', alpha=0.95)
    ax2.plot(rho_curve[mU2], theta_U[mU2], ls='--', lw=1.3, color='#ff7f0e', alpha=0.95)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\rho = \lambda_P / \lambda_C$ (log scale)")
    ax2.set_ylabel(r"$\theta$")
    ax2.set_ylim(T.min(), T.max())

    axins = inset_axes(ax2, width="40%", height="55%", loc="upper left",
                       bbox_to_anchor=(0.06, 0.98, 0.9, 0.9),
                       bbox_transform=ax2.transAxes, borderpad=0.8)
    mask = (R <= 2.0)
    axins.contourf(X[:, mask], Y[:, mask], Delta[:, mask], levels=256, cmap="coolwarm", norm=norm)
    axins.contour(X[:, mask], Y[:, mask], Delta[:, mask], levels=[0.0], colors="k", linewidths=1.2)
    axins.plot(R[mask], theta_th[mask], lw=1.6, color="#1f77b4")
    axins.plot(R[mask], theta_mc[mask], marker='o', linestyle='none', markersize=3, color="#ff7f0e")
    for x, l, h in zip(R[mask], lo[mask], hi[mask]):
        if np.isfinite(l) and np.isfinite(h):
            axins.plot([x, x], [l, h], lw=0.9, color="#ff7f0e")

    # ===== NEW: 解析阈值也画进 inset（注意用相同 ρ 连续网格取子集）=====
    maskU_in = (rho_curve <= 2.0) & mU
    maskL_in = (rho_curve <= 2.0) & mL
    axins.plot(rho_curve[maskL_in], theta_L[maskL_in], ls='--', lw=1.0, color='#7f7f7f', alpha=0.95)
    axins.plot(rho_curve[maskU_in], theta_U[maskU_in], ls='--', lw=1.0, color='#ff7f0e', alpha=0.95)

    axins.set_xlim(0.0, 2.0)
    axins.set_xlabel(r"$\rho$ (linear)", fontsize=9)
    axins.set_ylabel(r"$\theta$", fontsize=9)
    axins.tick_params(labelsize=8)

    # Legend to bottom-right for this figure too
    leg2 = ax2.legend(
        loc="lower right",
        bbox_to_anchor=(0.96, 0.06),
        borderaxespad=0.6,
        frameon=True
    )
    leg2.get_frame().set_alpha(0.85)
    leg2.get_frame().set_facecolor("white")

    ax2.set_title("Heatmap (Δ) + linear-ρ inset")
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(os.path.join(outdir, "heatmap_delta_mc_with_inset_dimless.png"), bbox_inches="tight")
        fig.savefig(os.path.join(outdir, "heatmap_delta_mc_with_inset_dimless.pdf"), bbox_inches="tight")
    plt.close(fig)

def make_slices(df_main, theta_grid, rho_slices, outdir):
    A = df_main.set_index("rho")
    T = theta_grid
    for rho in rho_slices:
        a_bar = A.loc[rho, "a_bar"]
        b_bar = A.loc[rho, "b_bar"]
        lam_c = 1.0 / (1.0 + rho)
        scale = 2.0 * np.sqrt(lam_c)
        alpha_bar = scale * a_bar
        b_bar_dimless = scale * b_bar

        # MC & theory in dimensionless form
        delta_mc = alpha_bar - T * b_bar_dimless
        alpha_th = alpha_theory(rho)
        b_th = b_theory(rho)
        delta_th = alpha_th - T * b_th

        fig = plt.figure(figsize=(6.2, 4.2))
        ax = fig.add_subplot(111)
        ax.plot(T, delta_mc, label="MC", lw=2.2, color="#ff7f0e")
        ax.plot(T, delta_th, label="Theory", lw=2.0, color="#1f77b4")
        ax.axhline(0.0, lw=1.0, color="k")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r" $\Delta(\theta,\rho)$")
        ax.set_title(fr"$\Delta$–$\theta$ slice at $\rho={rho}$")
        ax.legend(loc="best")
        fig.tight_layout()
        if SAVE_FIGURES:
            fig.savefig(os.path.join(outdir, f"slice_delta_theta_rho{rho}_dimless.png"), bbox_inches="tight")
            fig.savefig(os.path.join(outdir, f"slice_delta_theta_rho{rho}_dimless.pdf"), bbox_inches="tight")
        plt.close(fig)

def make_boundary_alignment(df_main, outdir):
    R = df_main["rho"].values
    th = df_main["theta_star_th"].values
    mc = df_main["theta_star_mc"].values
    lo = df_main["theta_star_mc_lo"].values
    hi = df_main["theta_star_mc_hi"].values

    fig = plt.figure(figsize=(6.8, 4.3))
    ax = fig.add_subplot(111)
    ax.plot(R, th, label=r"Theory $\theta^*(\rho)$", lw=2.2, color="#1f77b4")
    ax.plot(R, mc, marker='o', linestyle='none', label=r"MC $\theta^*(\rho)$", color="#ff7f0e")
    for x, l, h in zip(R, lo, hi):
        if np.isfinite(l) and np.isfinite(h):
            ax.plot([x, x], [l, h], lw=1.2, color="#ff7f0e")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho = \lambda_P / \lambda_C$ (log scale)")
    ax.set_ylabel(r"$\theta^*(\rho)$")
    ax.set_title("Boundary alignment: theory vs MC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(os.path.join(outdir, "theta_star_vs_theory.png"), bbox_inches="tight")
        fig.savefig(os.path.join(outdir, "theta_star_vs_theory.pdf"), bbox_inches="tight")
    plt.close(fig)

# ---------- Error analysis & plots ----------
def analyze_results(df_main, outdir, latex=False):
    df = df_main.copy()
    df["e_rho"] = df["theta_star_mc"] - df["theta_star_th"]
    df["z_rho"] = df["e_rho"] / df["theta_star_mc_se"]
    df.loc[~np.isfinite(df["z_rho"]), "z_rho"] = np.nan
    df["within_95_CI"] = df["theta_star_th"].between(df["theta_star_mc_lo"], df["theta_star_mc_hi"])

    eval_cols = [
        "rho", "theta_star_mc", "theta_star_mc_lo", "theta_star_mc_hi",
        "theta_star_th", "theta_star_mc_se", "e_rho", "z_rho", "within_95_CI"
    ]
    df_eval = df[eval_cols].sort_values("rho")
    df_eval.to_csv(os.path.join(outdir, "ppp_boundary_eval.csv"), index=False)

    # Summary metrics
    e = df_eval["e_rho"].values
    se = df_eval["theta_star_mc_se"].values
    max_abs_err = float(np.nanmax(np.abs(e)))
    mae = float(np.nanmean(np.abs(e)))
    rmse = float(np.sqrt(np.nanmean(e**2)))
    coverage = float(np.mean(df_eval["within_95_CI"].values))
    print("\n[Boundary error summary]")
    print(f"Max |e(ρ)|  = {max_abs_err:.4f}")
    print(f"MAE         = {mae:.4f}")
    print(f"RMSE        = {rmse:.4f}")
    print(f"95% CI coverage (theory in MC CI) = {coverage:.3f}")

    # Plots
    R = df_eval["rho"].values
    eR = df_eval["e_rho"].values
    SE = df_eval["theta_star_mc_se"].values

    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)
    ax.fill_between(R, -2*SE, 2*SE, color="#dddddd", alpha=0.8, label="±2 SE band")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.plot(R, eR, marker="o", color="#2ca02c", lw=1.8, label=r"$e(\rho)=\theta^*_{\rm MC}-\theta^*_{\rm th}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho$ (log scale)")
    ax.set_ylabel(r"$e(\rho)$")
    ax.set_title("Boundary deviation by $\\rho$ ")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "error_vs_rho.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(outdir, "error_vs_rho.pdf"), bbox_inches="tight")
    plt.close(fig)

    zR = df_eval["z_rho"].values
    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axhline(1.96, color="r", lw=1.0, ls="--")
    ax.axhline(-1.96, color="r", lw=1.0, ls="--", label="±1.96")
    ax.plot(R, zR, marker="o", color="#1f77b4", lw=1.8, label=r"$z(\rho)=e/SE$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho$ (log scale)")
    ax.set_ylabel(r"$z(\rho)$")
    ax.set_title("Standardized boundary deviation by $\\rho$")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "zscore_vs_rho.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(outdir, "zscore_vs_rho.pdf"), bbox_inches="tight")
    plt.close(fig)

# ---------- Main ----------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_main, df_sanity = run_experiment(RHO_GRID, K=K, M=M, L=L, rng=rng)

    # Save data
    df_main.to_csv(os.path.join(OUTPUT_DIR, "ppp_boundary_main.csv"), index=False)
    df_sanity.to_csv(os.path.join(OUTPUT_DIR, "ppp_boundary_sanity.csv"), index=False)

    # Console sanity snapshot
    print("\n[Sanity checks]")
    cols = ["rho", "N_C_mean", "N_P_mean", "D_mean",
            "p_near_psc_hat", "p_near_psc_theory", "p_near_psc_err"]
    print(df_sanity[cols])

    if PLOT_FIGURES:
        make_heatmap_and_boundaries(df_main, THETA_GRID, RHO_GRID, OUTPUT_DIR)
        make_slices(df_main, THETA_GRID, rho_slices=[0.2, 0.5, 1.0, 2.0, 5.0], outdir=OUTPUT_DIR)
        make_boundary_alignment(df_main, OUTPUT_DIR)

    # Error analysis & plots (θ* only; unaffected by scaling)
    analyze_results(df_main, OUTPUT_DIR, latex=False)

    print(f"\nDone. Results saved to: {OUTPUT_DIR}")
