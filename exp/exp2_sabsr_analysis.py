"""
Experiment 2: Stochastic-Correlation SABSR Analysis
====================================================
American put pricing under the 3-factor SABSR model and comparison with
DPDB benchmark confidence intervals (Yang & Li 2024a).

Model (rho_rhonu = 0):
  d rho_t = kappa_rho*(mu_rho - rho_t)*(1 - rho_t^2)*dt + sigma_rho*(1-rho_t^2)*dW^rho
  d nu_t  = sigma_nu * nu_t * dW^nu_t
  d S_t   = nu_t * S_t^beta * (rho_t*dW^nu + sqrt(1-rho_t^2)*dW^s)

Paper parameters:
  S0=1.1, T=1, beta=0.7, nu0=0.2, sigma_nu=0.08, r=0
  rho0=-0.4, mu_rho=-0.4, sigma_rho=0.5, rho_rhonu=0
  M=100000, N_T=400 (demo: N_T=100)
  kappa_rho ∈ {0.1, 2, 20}
  Strike grid K ∈ {1.00, 1.02, ..., 1.20}
  10 seeds

Comparison:
  - SBSR prices vs DPDB 95% CIs
  - SABSR prices vs constant-correlation SABR (from exp1)
  - Effect of kappa_rho on prices and deltas
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forward_simulation import simulate_sabsr_paths
from src.rbsde_solver import (
    build_sabsr_features,
    solve_rbsde_3factor,
    extract_sabsr_delta,
)
from src.benchmarks import get_sabsr_dpdb, SABR_CTMC_PRICES, SABR_CTMC_STRIKES
from src.reporting import (
    ensure_results_dir,
    save_price_table,
    plot_price_comparison,
    plot_delta_profile,
    plot_kappa_comparison,
    write_results_md,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0 = 1.1
T = 1.0
BETA = 0.7
NU0 = 0.2
SIGMA_NU = 0.08
RHO0 = -0.4
MU_RHO = -0.4
SIGMA_RHO = 0.5
RHO_RHONU = 0.0
R = 0.0
M = 100_000
N_T = 100      # demo; paper uses 400
N_SEEDS = 3    # demo; paper uses 10

STRIKES = np.array([1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20])
KAPPA_RHO_VALUES = [0.1, 2.0, 20.0]


def run_sabsr_experiment(kappa_rho: float, n_seeds: int = N_SEEDS) -> dict:
    """
    Run the SABSR RBSDE experiment for given kappa_rho.

    Parameters
    ----------
    kappa_rho : Mean-reversion speed for rho
    n_seeds   : Number of independent seeds

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
        seed = seed_idx * 200 + int(kappa_rho * 10)
        print(f"    Seed {seed_idx + 1}/{n_seeds} (kappa_rho={kappa_rho})...")

        t_seed = time.time()

        # Forward paths — shared across all strikes
        rho_paths, nu_paths, S_paths, dW_rho, dW_nu, dW_s = simulate_sabsr_paths(
            S0=S0, nu0=NU0, rho0=RHO0, T=T, N_T=N_T, M=M,
            beta=BETA, sigma_nu=SIGMA_NU, sigma_rho=SIGMA_RHO,
            mu_rho=MU_RHO, kappa_rho=kappa_rho, rho_rhonu=RHO_RHONU,
            seed=seed,
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
                feature_builder=build_sabsr_features,
                driver_fn=None,
                n_picard=10,
                train_frac=0.75,
                r=R,
                store_path=False,
            )
            prices_all[seed_idx, k_idx] = result["price"]
            deltas_all[seed_idx, k_idx] = extract_sabsr_delta(
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
    """Run full Experiment 2 and save results."""
    ensure_results_dir()
    print("=" * 65)
    print("Experiment 2: Stochastic-Correlation SABSR Analysis")
    print(f"  M={M}, N_T={N_T}, Seeds={N_SEEDS}")
    print(f"  kappa_rho ∈ {KAPPA_RHO_VALUES}")
    print("=" * 65)

    all_results = {}
    md_sections = []

    for kappa_rho in KAPPA_RHO_VALUES:
        print(f"\n[kappa_rho = {kappa_rho}]")
        res = run_sabsr_experiment(kappa_rho=kappa_rho)
        all_results[kappa_rho] = res

        # Load DPDB benchmark
        dpdb_df = get_sabsr_dpdb(kappa_rho)
        dpdb_lower = dpdb_df["dpdb_lower"].values
        dpdb_upper = dpdb_df["dpdb_upper"].values
        dpdb_center = dpdb_df["dpdb_center"].values

        # Check CI containment
        in_ci = check_ci_containment(res["mean_prices"], dpdb_lower, dpdb_upper)
        n_in_ci = in_ci.sum()
        print(f"  CI containment: {n_in_ci}/{len(STRIKES)} strikes within DPDB 95% CI")

        # Save table
        save_price_table(
            strikes=STRIKES,
            rbsde_mean=res["mean_prices"],
            rbsde_std=res["std_prices"],
            benchmark=dpdb_center,
            benchmark_name="dpdb_center",
            filename=f"exp2_sabsr_kappa{kappa_rho:.1f}.csv",
            n_t_label=str(N_T),
            extra_cols={
                "dpdb_lower": dpdb_lower,
                "dpdb_upper": dpdb_upper,
                "in_ci": in_ci.astype(int),
            },
        )

        # Plot
        plot_price_comparison(
            strikes=STRIKES,
            rbsde_mean=res["mean_prices"],
            rbsde_std=res["std_prices"],
            benchmark=dpdb_center,
            benchmark_name=f"DPDB center (kappa_rho={kappa_rho})",
            title=f"SABSR American Put: RBSDE vs DPDB (kappa_rho={kappa_rho})",
            filename=f"exp2_sabsr_kappa{kappa_rho:.1f}.png",
            ci_lower=dpdb_lower,
            ci_upper=dpdb_upper,
        )

        # Delta profile
        plot_delta_profile(
            strikes=STRIKES,
            deltas_mean=res["mean_deltas"],
            deltas_std=res["std_deltas"],
            title=f"SABSR Spot Delta (kappa_rho={kappa_rho})",
            filename=f"exp2_sabsr_delta_kappa{kappa_rho:.1f}.png",
        )

        # Summary
        df_out = pd.DataFrame({
            "strike": STRIKES,
            "rbsde_mean": res["mean_prices"],
            "rbsde_std": res["std_prices"],
            "dpdb_lower": dpdb_lower,
            "dpdb_upper": dpdb_upper,
            "in_ci": in_ci.astype(int),
        })
        md_sections.append({
            "title": f"Exp 2: SABSR kappa_rho={kappa_rho} (N_T={N_T}, M={M}, Seeds={N_SEEDS})",
            "content": df_out,
        })

    # Compare across kappa_rho: effect on prices relative to SABR benchmark
    print("\n[Effect of kappa_rho on prices relative to CTMC SABR]")
    sabr_ctmc = SABR_CTMC_PRICES  # constant-correlation SABR reference

    # Price differences: SABSR vs SABR
    print("\n  Price difference SABSR - SABR_CTMC by kappa_rho:")
    for kappa_rho, res in all_results.items():
        diff = res["mean_prices"] - sabr_ctmc
        print(f"  kappa_rho={kappa_rho}: mean_diff={diff.mean():.5f}, max_abs_diff={np.abs(diff).max():.5f}")

    # Verify kappa_rho=20 is closest to SABR (faster reversion → approach constant-rho)
    diffs = {k: np.abs(r["mean_prices"] - sabr_ctmc).mean() for k, r in all_results.items()}
    closest = min(diffs, key=diffs.get)
    print(f"\n  Closest kappa_rho to constant-correlation SABR: {closest}")
    if closest == 20.0:
        print("  ✓ Confirmed: higher kappa_rho approaches constant-correlation SABR")

    # Remap keys for reporting helper (expects "mean")
    _kappa_report = {k: {"mean": r["mean_prices"]} for k, r in all_results.items()}

    # kappa comparison plot using SABR CTMC as reference
    plot_kappa_comparison(
        strikes=STRIKES,
        sabsr_results=_kappa_report,
        sabr_mean=sabr_ctmc,
        filename="exp2_sabsr_kappa_comparison.png",
    )

    md_sections.append({
        "title": "Exp 2: kappa_rho Effect on SABSR-SABR Price Differences",
        "content": "\n".join(
            f"kappa_rho={k}: mean |SABSR−SABR| = {diffs[k]:.5f}"
            for k in sorted(diffs)
        ),
    })

    write_results_md(sections=md_sections, filename="RESULTS_exp2.md")
    print("\nExperiment 2 complete. Results saved to results/")


if __name__ == "__main__":
    main()
