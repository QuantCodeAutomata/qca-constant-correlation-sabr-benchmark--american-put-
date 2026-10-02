"""
Experiment 4: Stochastic-Correlation HSCLV Analysis
====================================================
American put pricing under the 3-factor HSCLV model (Heston SLV with
stochastic correlation) and comparison with DPDB confidence intervals.

Model (rho_rhonu = 0):
  d rho_t = kappa_rho*(mu_rho - rho_t)*(1-rho_t^2)*dt + sigma_rho*(1-rho_t^2)*dW^rho
  d nu_t  = kappa_nu*(mu_nu-nu_t)*dt + sigma_nu*sqrt(nu_t)*dW^nu   [CIR]
  d S_t   = mu_s*S_t*dt + sqrt(nu_t)*S_t^beta*(rho_t*dW^nu + sqrt(1-rho_t^2)*dW^s)

Paper parameters:
  S0=10, T=1, beta=0.7, nu0=0.04, kappa_nu=4, mu_nu=0.035
  sigma_nu=0.15, r=0.03, mu_s=0.03, rho0=-0.75
  kappa_rho=2.0, sigma_rho=0.1, rho_rhonu=0
  M=200000, N_T=30
  mu_rho ∈ {-0.2, 0, 0.75}
  Strike grid K ∈ {7, 8, 9, 10, 11, 12, 13}
  10 seeds

Expected outcomes:
  - Prices within DPDB 95% CIs
  - mu_rho=-0.2: ~0.2784 at K=10
  - mu_rho=0:    ~0.2786 at K=10
  - mu_rho=0.75: ~0.2755 at K=10
  - Deep ITM puts show weak sensitivity to mu_rho
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hslv_simulation import simulate_hsclv_paths
from src.rbsde_solver import (
    build_hsclv_features,
    solve_rbsde_3factor,
    extract_hsclv_delta,
)
from src.benchmarks import get_hsclv_dpdb, HSLV_CTMC_PRICES, HSLV_CTMC_STRIKES
from src.reporting import (
    ensure_results_dir,
    save_price_table,
    plot_price_comparison,
    plot_delta_profile,
    plot_mu_rho_comparison,
    write_results_md,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0 = 10.0
T = 1.0
BETA = 0.7
NU0 = 0.04
KAPPA_NU = 4.0
MU_NU = 0.035
SIGMA_NU = 0.15
MU_S = 0.03
R = 0.03
RHO0 = -0.75
KAPPA_RHO = 2.0
SIGMA_RHO = 0.1
RHO_RHONU = 0.0
M = 200_000
N_T = 30  # paper specification

STRIKES = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
MU_RHO_VALUES = [-0.2, 0.0, 0.75]
N_SEEDS = 3  # demo; paper uses 10


def run_hsclv_experiment(mu_rho: float, n_seeds: int = N_SEEDS) -> dict:
    """
    Run the HSCLV RBSDE experiment for given mu_rho.

    Parameters
    ----------
    mu_rho  : Long-run correlation level
    n_seeds : Number of independent seeds

    Returns
    -------
    dict with mean_prices, std_prices, mean_deltas, std_deltas, mean_runtime
    """
    dt = T / N_T
    n_strikes = len(STRIKES)
    prices_all = np.zeros((n_seeds, n_strikes))
    deltas_all = np.zeros((n_seeds, n_strikes))
    runtimes = []

    for seed_idx in range(n_seeds):
        seed = seed_idx * 400 + int((mu_rho + 1.0) * 100)
        print(f"    Seed {seed_idx + 1}/{n_seeds} (mu_rho={mu_rho})...")

        t_seed = time.time()

        rho_paths, nu_paths, S_paths, dW_rho, dW_nu, dW_s = simulate_hsclv_paths(
            S0=S0, nu0=NU0, rho0=RHO0, T=T, N_T=N_T, M=M,
            beta=BETA, kappa_nu=KAPPA_NU, mu_nu=MU_NU, sigma_nu=SIGMA_NU,
            mu_s=MU_S, kappa_rho=KAPPA_RHO, mu_rho=mu_rho,
            sigma_rho=SIGMA_RHO, rho_rhonu=RHO_RHONU, seed=seed,
        )

        for k_idx, K in enumerate(STRIKES):
            result = solve_rbsde_3factor(
                rho_paths=rho_paths,
                nu_paths=nu_paths,
                S_paths=S_paths,
                dW_rho=dW_rho,
                dW_nu=dW_nu,
                dW_s=dW_s,
                dt=dt,
                K=K,
                feature_builder=build_hsclv_features,
                driver_fn=None,
                n_picard=10,
                train_frac=0.75,
                r=R,
                store_path=False,
            )
            prices_all[seed_idx, k_idx] = result["price"]
            deltas_all[seed_idx, k_idx] = extract_hsclv_delta(
                Z0=result["Z0"],
                nu0=NU0,
                S0=S0,
                beta=BETA,
                rho0=RHO0,
            )

        runtimes.append(time.time() - t_seed)
        print(f"      Seed runtime: {runtimes[-1]:.1f}s")

    return {
        "mean_prices": prices_all.mean(axis=0),
        "std_prices": prices_all.std(axis=0),
        "mean_deltas": deltas_all.mean(axis=0),
        "std_deltas": deltas_all.std(axis=0),
        "mean_runtime": np.mean(runtimes),
    }


def check_ci_containment(
    mean_prices: np.ndarray,
    dpdb_lower: np.ndarray,
    dpdb_upper: np.ndarray,
) -> np.ndarray:
    """Return boolean array: True if price falls within DPDB CI."""
    return (mean_prices >= dpdb_lower) & (mean_prices <= dpdb_upper)


def main() -> None:
    """Run full Experiment 4 and save results."""
    ensure_results_dir()
    print("=" * 65)
    print("Experiment 4: Stochastic-Correlation HSCLV Analysis")
    print(f"  M={M}, N_T={N_T}, Seeds={N_SEEDS}")
    print(f"  mu_rho ∈ {MU_RHO_VALUES}, kappa_rho={KAPPA_RHO}")
    print("=" * 65)

    all_results = {}
    md_sections = []

    for mu_rho in MU_RHO_VALUES:
        print(f"\n[mu_rho = {mu_rho}]")
        res = run_hsclv_experiment(mu_rho=mu_rho)
        all_results[mu_rho] = res

        dpdb_df = get_hsclv_dpdb(mu_rho)
        dpdb_lower = dpdb_df["dpdb_lower"].values
        dpdb_upper = dpdb_df["dpdb_upper"].values
        dpdb_center = dpdb_df["dpdb_center"].values

        in_ci = check_ci_containment(res["mean_prices"], dpdb_lower, dpdb_upper)
        n_in_ci = in_ci.sum()
        print(f"  CI containment: {n_in_ci}/{len(STRIKES)} strikes within DPDB 95% CI")

        # Check K=10 specifically
        k10_idx = np.where(STRIKES == 10.0)[0][0]
        print(f"  K=10: price={res['mean_prices'][k10_idx]:.4f}")

        save_price_table(
            strikes=STRIKES,
            rbsde_mean=res["mean_prices"],
            rbsde_std=res["std_prices"],
            benchmark=dpdb_center,
            benchmark_name="dpdb_center",
            filename=f"exp4_hsclv_mu{mu_rho:.2f}.csv".replace("-", "m"),
            n_t_label=str(N_T),
            extra_cols={
                "dpdb_lower": dpdb_lower,
                "dpdb_upper": dpdb_upper,
                "in_ci": in_ci.astype(int),
            },
        )
        plot_price_comparison(
            strikes=STRIKES,
            rbsde_mean=res["mean_prices"],
            rbsde_std=res["std_prices"],
            benchmark=dpdb_center,
            benchmark_name=f"DPDB center (mu_rho={mu_rho})",
            title=f"HSCLV American Put: RBSDE vs DPDB (mu_rho={mu_rho})",
            filename=f"exp4_hsclv_prices_mu{mu_rho:.2f}.png".replace("-", "m"),
            ci_lower=dpdb_lower,
            ci_upper=dpdb_upper,
        )
        plot_delta_profile(
            strikes=STRIKES,
            deltas_mean=res["mean_deltas"],
            deltas_std=res["std_deltas"],
            title=f"HSCLV Spot Delta (mu_rho={mu_rho})",
            filename=f"exp4_hsclv_delta_mu{mu_rho:.2f}.png".replace("-", "m"),
        )

        df_out = pd.DataFrame({
            "strike": STRIKES,
            "rbsde_mean": res["mean_prices"],
            "rbsde_std": res["std_prices"],
            "dpdb_lower": dpdb_lower,
            "dpdb_upper": dpdb_upper,
            "in_ci": in_ci.astype(int),
        })
        md_sections.append({
            "title": f"Exp 4: HSCLV mu_rho={mu_rho} (N_T={N_T}, M={M}, Seeds={N_SEEDS})",
            "content": df_out,
        })

    # Effect of mu_rho: compare HSCLV vs HSLV (constant-correlation benchmark)
    hslv_ctmc = HSLV_CTMC_PRICES

    print("\n[Effect of mu_rho on prices vs constant-correlation HSLV CTMC]")
    for mu_rho, res in all_results.items():
        diff = res["mean_prices"] - hslv_ctmc
        k10_idx = np.where(STRIKES == 10.0)[0][0]
        print(f"  mu_rho={mu_rho}: diff at K=10 = {diff[k10_idx]:.5f}")

    # Deep ITM check: stochastic correlation should have weak effect
    print("\n[Deep ITM sensitivity to mu_rho]")
    deep_idx = [np.where(STRIKES == k)[0][0] for k in [11.0, 12.0, 13.0]]
    for idx, k in zip(deep_idx, [11.0, 12.0, 13.0]):
        prices = [all_results[mu]["mean_prices"][idx] for mu in MU_RHO_VALUES]
        spread = max(prices) - min(prices)
        intrinsic = k - S0
        print(
            f"  K={k:.0f}: prices={[f'{p:.4f}' for p in prices]}, "
            f"spread={spread:.4f}, intrinsic={intrinsic:.1f}"
        )

    # Remap keys for reporting helper (expects "mean")
    _mu_report = {mu: {"mean": r["mean_prices"]} for mu, r in all_results.items()}

    # mu_rho comparison plot
    plot_mu_rho_comparison(
        strikes=STRIKES,
        hsclv_results=_mu_report,
        hslv_mean=hslv_ctmc,
        filename="exp4_hsclv_mu_rho_comparison.png",
    )

    md_sections.append({
        "title": "Exp 4: mu_rho Effect on HSCLV-HSLV Price Differences",
        "content": "\n".join(
            f"mu_rho={mu}: HSCLV K=10 price = {all_results[mu]['mean_prices'][k10_idx]:.5f}"
            for mu in MU_RHO_VALUES
        ),
    })

    write_results_md(sections=md_sections, filename="RESULTS_exp4.md")
    print("\nExperiment 4 complete. Results saved to results/")


if __name__ == "__main__":
    main()
