"""
Experiment 5: Future Evolution of American Option Values
========================================================
Demonstrates that the reflected BSDE framework provides full pathwise future
option value evolution Y_t, not only Y_0.

Substudies:
  A. SABSR model: S0=100, K=110, kappa_rho=2, varying beta ∈ {0.5, 0.7, 1.0}
  B. HSCLV model: S0=100, K=105, mu_rho=-0.2, varying beta ∈ {0.5, 0.7, 1.0}

For each (model, beta), we:
  1. Simulate forward paths
  2. Solve the reflected BSDE with store_path=True to get full Y_t trajectories
  3. Compute mean Y_t trajectory over all paths
  4. Extract Y at t=T/2 for cross-sectional distribution

Paper parameters (SABSR):
  nu0=0.2, sigma_nu=0.08, rho0=-0.4, mu_rho=-0.4, sigma_rho=0.5, r=0
  M=100000, N_T=50 (demo; paper uses N_T=400 equivalent for kappa_rho=2)

Paper parameters (HSCLV):
  nu0=0.04, kappa_nu=4, mu_nu=0.035, sigma_nu=0.15, r=0.03, mu_s=0.03
  rho0=-0.75, kappa_rho=2, sigma_rho=0.1, rho_rhonu=0
  M=200000, N_T=30

Note: Beta values verified from paper figure; using beta ∈ {0.5, 0.7, 1.0}
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forward_simulation import simulate_sabsr_paths
from src.hslv_simulation import simulate_hsclv_paths
from src.rbsde_solver import (
    build_sabsr_features,
    build_hsclv_features,
    solve_rbsde_3factor,
)
from src.reporting import (
    ensure_results_dir,
    plot_future_evolution,
    plot_midtime_distribution,
    write_results_md,
)

# ---------------------------------------------------------------------------
# SABSR substudy parameters
# ---------------------------------------------------------------------------
SABSR_S0 = 100.0
SABSR_K = 110.0
SABSR_T = 1.0
SABSR_NU0 = 0.2
SABSR_SIGMA_NU = 0.08
SABSR_RHO0 = -0.4
SABSR_MU_RHO = -0.4
SABSR_SIGMA_RHO = 0.5
SABSR_KAPPA_RHO = 2.0
SABSR_R = 0.0
SABSR_M = 50_000   # reduced from 100000 for memory/speed
SABSR_N_T = 50     # demo; scale with paper's 400 for production
SABSR_BETAS = [0.5, 0.7, 1.0]

# ---------------------------------------------------------------------------
# HSCLV substudy parameters
# ---------------------------------------------------------------------------
HSCLV_S0 = 100.0
HSCLV_K = 105.0
HSCLV_T = 1.0
HSCLV_NU0 = 0.04
HSCLV_KAPPA_NU = 4.0
HSCLV_MU_NU = 0.035
HSCLV_SIGMA_NU = 0.15
HSCLV_MU_S = 0.03
HSCLV_R = 0.03
HSCLV_RHO0 = -0.75
HSCLV_KAPPA_RHO = 2.0
HSCLV_MU_RHO = -0.2
HSCLV_SIGMA_RHO = 0.1
HSCLV_M = 50_000   # reduced from 200000
HSCLV_N_T = 30     # paper specification
HSCLV_BETAS = [0.5, 0.7, 1.0]

N_SEEDS_EVO = 1  # illustrative run (paper uses single illustrative seed for figures)


def run_sabsr_evolution(beta: float, seed: int = 0) -> dict:
    """
    Run SABSR RBSDE experiment with store_path=True for one (beta, seed).

    Returns
    -------
    dict with:
      'times'    : (N_T+1,) time grid
      'mean_traj': (N_T+1,) mean Y_t across paths
      'Y_mid'    : (M,) Y values at t ≈ T/2
      'price'    : float Y_0
      'runtime'  : float
    """
    dt = SABSR_T / SABSR_N_T
    times = np.linspace(0, SABSR_T, SABSR_N_T + 1)
    mid_idx = SABSR_N_T // 2  # index nearest to t=T/2

    t0 = time.time()

    rho_paths, nu_paths, S_paths, dW_rho, dW_nu, dW_s = simulate_sabsr_paths(
        S0=SABSR_S0, nu0=SABSR_NU0, rho0=SABSR_RHO0,
        T=SABSR_T, N_T=SABSR_N_T, M=SABSR_M,
        beta=beta, sigma_nu=SABSR_SIGMA_NU, sigma_rho=SABSR_SIGMA_RHO,
        mu_rho=SABSR_MU_RHO, kappa_rho=SABSR_KAPPA_RHO,
        rho_rhonu=0.0, seed=seed,
    )

    result = solve_rbsde_3factor(
        rho_paths=rho_paths,
        nu_paths=nu_paths,
        S_paths=S_paths,
        dW_rho=dW_rho,
        dW_nu=dW_nu,
        dW_s=dW_s,
        dt=dt,
        K=SABSR_K,
        feature_builder=build_sabsr_features,
        driver_fn=None,
        n_picard=10,
        train_frac=0.75,
        r=SABSR_R,
        store_path=True,
    )

    Y_path = result["Y_path"]  # (M, N_T+1)
    # Verify reflection constraint holds: Y_i >= g(S_i) for all i, all paths
    for i in range(SABSR_N_T + 1):
        g_i = np.maximum(SABSR_K - S_paths[:, i], 0.0)
        violations = (Y_path[:, i] < g_i - 1e-8).sum()
        if violations > 0:
            print(f"    WARNING: {violations} reflection violations at step {i}")

    return {
        "times": times,
        "mean_traj": Y_path.mean(axis=0),
        "Y_mid": Y_path[:, mid_idx],
        "price": result["price"],
        "runtime": time.time() - t0,
    }


def run_hsclv_evolution(beta: float, seed: int = 0) -> dict:
    """
    Run HSCLV RBSDE experiment with store_path=True for one (beta, seed).

    Returns
    -------
    dict with: times, mean_traj, Y_mid, price, runtime
    """
    dt = HSCLV_T / HSCLV_N_T
    times = np.linspace(0, HSCLV_T, HSCLV_N_T + 1)
    mid_idx = HSCLV_N_T // 2

    t0 = time.time()

    rho_paths, nu_paths, S_paths, dW_rho, dW_nu, dW_s = simulate_hsclv_paths(
        S0=HSCLV_S0, nu0=HSCLV_NU0, rho0=HSCLV_RHO0,
        T=HSCLV_T, N_T=HSCLV_N_T, M=HSCLV_M,
        beta=beta, kappa_nu=HSCLV_KAPPA_NU, mu_nu=HSCLV_MU_NU,
        sigma_nu=HSCLV_SIGMA_NU, mu_s=HSCLV_MU_S,
        kappa_rho=HSCLV_KAPPA_RHO, mu_rho=HSCLV_MU_RHO,
        sigma_rho=HSCLV_SIGMA_RHO, rho_rhonu=0.0, seed=seed,
    )

    result = solve_rbsde_3factor(
        rho_paths=rho_paths,
        nu_paths=nu_paths,
        S_paths=S_paths,
        dW_rho=dW_rho,
        dW_nu=dW_nu,
        dW_s=dW_s,
        dt=dt,
        K=HSCLV_K,
        feature_builder=build_hsclv_features,
        driver_fn=None,
        n_picard=10,
        train_frac=0.75,
        r=HSCLV_R,
        store_path=True,
    )

    Y_path = result["Y_path"]

    return {
        "times": times,
        "mean_traj": Y_path.mean(axis=0),
        "Y_mid": Y_path[:, mid_idx],
        "price": result["price"],
        "runtime": time.time() - t0,
    }


def main() -> None:
    """Run Experiment 5 and produce evolution plots."""
    ensure_results_dir()
    print("=" * 65)
    print("Experiment 5: Future Evolution of American Option Values")
    print("=" * 65)

    md_sections = []

    # -----------------------------------------------------------------------
    # SABSR substudy
    # -----------------------------------------------------------------------
    print("\n[Substudy A: SABSR Evolution]")
    print(f"  S0={SABSR_S0}, K={SABSR_K}, T={SABSR_T}")
    print(f"  betas={SABSR_BETAS}, M={SABSR_M}, N_T={SABSR_N_T}")

    sabsr_mean_trajectories = {}
    sabsr_mid_distributions = {}
    sabsr_prices = {}

    for beta in SABSR_BETAS:
        print(f"  beta={beta}...")
        res = run_sabsr_evolution(beta=beta, seed=42)
        label = f"beta={beta}"
        sabsr_mean_trajectories[label] = res["mean_traj"]
        sabsr_mid_distributions[label] = res["Y_mid"]
        sabsr_prices[label] = res["price"]
        print(f"    Y_0={res['price']:.4f}, runtime={res['runtime']:.1f}s")

    # Plot mean trajectories
    times_sabsr = np.linspace(0, SABSR_T, SABSR_N_T + 1)
    plot_future_evolution(
        times=times_sabsr,
        mean_trajectories=sabsr_mean_trajectories,
        title=f"SABSR: Mean Option Value Evolution (S0={SABSR_S0}, K={SABSR_K})",
        filename="exp5_sabsr_evolution.png",
        y_label="Mean American Put Value",
    )
    plot_midtime_distribution(
        distributions=sabsr_mid_distributions,
        title=f"SABSR: Option Value Distribution at t=T/2 (S0={SABSR_S0}, K={SABSR_K})",
        filename="exp5_sabsr_mid_distribution.png",
        t_mid=SABSR_T / 2,
    )

    md_sections.append({
        "title": f"Exp 5A: SABSR Evolution (S0={SABSR_S0}, K={SABSR_K})",
        "content": "\n".join(f"beta={b}: Y_0={sabsr_prices[f'beta={b}']:.4f}" for b in SABSR_BETAS),
    })

    # -----------------------------------------------------------------------
    # HSCLV substudy
    # -----------------------------------------------------------------------
    print("\n[Substudy B: HSCLV Evolution]")
    print(f"  S0={HSCLV_S0}, K={HSCLV_K}, T={HSCLV_T}")
    print(f"  betas={HSCLV_BETAS}, M={HSCLV_M}, N_T={HSCLV_N_T}")

    hsclv_mean_trajectories = {}
    hsclv_mid_distributions = {}
    hsclv_prices = {}

    for beta in HSCLV_BETAS:
        print(f"  beta={beta}...")
        res = run_hsclv_evolution(beta=beta, seed=42)
        label = f"beta={beta}"
        hsclv_mean_trajectories[label] = res["mean_traj"]
        hsclv_mid_distributions[label] = res["Y_mid"]
        hsclv_prices[label] = res["price"]
        print(f"    Y_0={res['price']:.4f}, runtime={res['runtime']:.1f}s")

    times_hsclv = np.linspace(0, HSCLV_T, HSCLV_N_T + 1)
    plot_future_evolution(
        times=times_hsclv,
        mean_trajectories=hsclv_mean_trajectories,
        title=f"HSCLV: Mean Option Value Evolution (S0={HSCLV_S0}, K={HSCLV_K})",
        filename="exp5_hsclv_evolution.png",
        y_label="Mean American Put Value",
    )
    plot_midtime_distribution(
        distributions=hsclv_mid_distributions,
        title=f"HSCLV: Option Value Distribution at t=T/2 (S0={HSCLV_S0}, K={HSCLV_K})",
        filename="exp5_hsclv_mid_distribution.png",
        t_mid=HSCLV_T / 2,
    )

    md_sections.append({
        "title": f"Exp 5B: HSCLV Evolution (S0={HSCLV_S0}, K={HSCLV_K})",
        "content": "\n".join(f"beta={b}: Y_0={hsclv_prices[f'beta={b}']:.4f}" for b in HSCLV_BETAS),
    })

    write_results_md(sections=md_sections, filename="RESULTS_exp5.md")

    print("\n" + "=" * 65)
    print("Experiment 5 complete. Results saved to results/")
    print("\nKey findings:")
    print("  SABSR Y_0 by beta:", {k: f"{v:.4f}" for k, v in sabsr_prices.items()})
    print("  HSCLV Y_0 by beta:", {k: f"{v:.4f}" for k, v in hsclv_prices.items()})


if __name__ == "__main__":
    main()
