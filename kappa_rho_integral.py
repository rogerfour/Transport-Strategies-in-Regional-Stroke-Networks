# kappa_rho_integral.py
# 计算并绘制 κ(ρ)，并输出代表性取值的结果
# ρ = λ_p / λ_c
# κ(ρ) = π^{3/2}(1+ρ) ∫_{0}^{∞} [π(1+ρ) + h(u)]^{-3/2} du
#      = ∫_{0}^{2} π^{3/2}(1+ρ) / [π(1+ρ) + h(u)]^{3/2} du
#        + ∫_{2}^{∞} (1+ρ) / (u^2 + ρ)^{3/2} du

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi

# -----------------------------
# 可调参数
# -----------------------------
RHO_MIN = 0.01       # ρ 的起点（必须 > 0；取 0.02 避免极限处数值不稳）
RHO_MAX = 100.0       # ρ 的最大值（可按需调整）
NUM_PTS = 400        # 曲线采样点数
ABS_TOL = 1e-10      # 自适应辛普森积分的绝对误差容限

# 代表性 ρ（对应旧记号中的 1.1, 1.5, 2, 3, 5, 10, 15, 20, 30 的新定义为减 1）
REP_RHOS = [0.05, 0.1, 0.2, 0.5, 1, 2,3, 5, 10, 20, 50]

PLOT_PATH = "kappa_vs_rho.png"
CSV_PATH = "kappa_values_rho.csv"

# -----------------------------
# 函数定义
# -----------------------------

def h_piece(u: float) -> float:
    """
    面积函数 h(u)，对应 |B(x,s) \ B(0,r)| 的无量纲化分段形式：
      u in [0,2): 透镜区
      u in [2,∞): 整圆减去禁入圆
    标量实现，便于自适应积分。
    """
    if u < 2.0:
        # 数值保护：arccos 入参限制在 [-1, 1]
        a1 = max(min(u / 2.0, 1.0), -1.0)
        a2 = max(min(1.0 - (u*u)/2.0, 1.0), -1.0)
        term = (u*u)*np.arccos(a1) + np.arccos(a2) - 0.5*u*np.sqrt(max(0.0, 4.0 - u*u))
        return pi*u*u - term
    else:
        return pi*(u*u - 1.0)

def integrand_0_2(u: float, rho: float) -> float:
    """
    κ(ρ) 在 [0,2] 的被积函数：
      integrand = π^{3/2} · (1+ρ) / [π(1+ρ) + h(u)]^{3/2}
    """
    denom = pi*(1.0 + rho) + h_piece(u)
    return (pi**1.5)*(1.0 + rho) / (denom**1.5)

def _simpson(f, a, b, fa, fb, fm):
    return (b - a) * (fa + 4.0*fm + fb) / 6.0

def adaptive_simpson(f, a, b, tol=1e-10, maxdepth=20):
    """
    自适应辛普森积分，对标量函数 f 在 [a,b] 上积分。
    """
    fa = f(a)
    fb = f(b)
    m = 0.5*(a + b)
    fm = f(m)
    S = _simpson(f, a, b, fa, fb, fm)

    def recurse(a, b, fa, fb, fm, S, depth):
        m = 0.5*(a + b)
        lm = 0.5*(a + m); rm = 0.5*(m + b)
        flm = f(lm); frm = f(rm)
        S_left = _simpson(f, a, m, fa, fm, flm)
        S_right = _simpson(f, m, b, fm, fb, frm)
        if depth <= 0 or abs(S_left + S_right - S) < 15.0*tol:
            return S_left + S_right + (S_left + S_right - S)/15.0
        return recurse(a, m, fa, fm, flm, S_left, depth-1) + \
               recurse(m, b, fm, fb, frm, S_right, depth-1)

    return recurse(a, b, fa, fb, fm, S, maxdepth)

def tail_integral_rho(rho: float) -> float:
    """
    对于 u∈[2,∞)，被积函数化简为 (1+ρ)/(u^2 + ρ)^{3/2}。
    其不定积分为 (1+ρ) * u / (ρ * sqrt(u^2 + ρ))，ρ>0。
    解析尾积分：
      ∫_{2}^{∞} (1+ρ)/(u^2 + ρ)^{3/2} du
        = (1+ρ) * [ 1/ρ - 2/(ρ*sqrt(4+ρ)) ].
    """
    if rho <= 0:
        raise ValueError("rho must be > 0.")
    # 为避免小 ρ 下的消差，可改为： (1+ρ)/ρ * (1 - 2/√(4+ρ))
    return (1.0 + rho) * (1.0/rho - 2.0/(rho * sqrt(4.0 + rho)))

def kappa_rho(rho: float, abs_tol=1e-10) -> float:
    """
    κ(ρ) = ∫_{0}^{2} π^{3/2}(1+ρ) / [π(1+ρ)+h(u)]^{3/2} du
           + ∫_{2}^{∞} (1+ρ)/(u^2+ρ)^{3/2} du
    """
    if rho <= 0:
        raise ValueError("rho must be > 0.")
    f = lambda u: integrand_0_2(u, rho)
    I1 = adaptive_simpson(f, 0.0, 2.0, tol=abs_tol, maxdepth=22)
    I2 = tail_integral_rho(rho)
    return I1 + I2

# -----------------------------
# 主流程
# -----------------------------
if __name__ == "__main__":
    # 计算整条曲线
    rhos = np.linspace(RHO_MIN, RHO_MAX, NUM_PTS)
    kappas = np.array([kappa_rho(r, ABS_TOL) for r in rhos])

    # 绘图
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(rhos, kappas, label=r'$\kappa(\rho)$ (integral)')
    plt.xlabel(r'$\rho = \lambda_p/\lambda_c$')
    plt.ylabel(r'$\kappa(\rho)$')
    plt.grid(True, linewidth=0.3, alpha=0.6)
    plt.legend()
    plt.title(r'Kappa vs $\rho$ from the integral definition')
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    print(f"[Saved] curve figure -> {PLOT_PATH}")

    # 保存曲线数据为 CSV
    np.savetxt(CSV_PATH, np.column_stack([rhos, kappas]),
               delimiter=",", header="rho,kappa(rho)", comments="")
    print(f"[Saved] curve data -> {CSV_PATH}")

    # 输出代表性 ρ 的结果
    print("\nRepresentative values:")
    header = f"{'rho':>8}  {'kappa(rho)':>12}"
    print(header)
    print("-"*len(header))
    for rho in REP_RHOS:
        val = kappa_rho(float(rho), ABS_TOL)
        print(f"{rho:8.3f}  {val:12.6f}")
