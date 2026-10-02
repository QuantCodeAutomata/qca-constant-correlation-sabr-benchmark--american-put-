"""
Experiment 1: Constant-Correlation SABR Benchmark
==================================================
American put pricing via XGBoost-based reflected BSDE and comparison with
CTMC benchmark values (Cui et al. 2018).

Model:
  d nu_t = sigma_nu * nu_t * dW^nu_t
  d S_t  = nu_t * S_t^beta * dW^S_t
  Corr(dW^nu, dW^S) = rho = -0.4

Paper parameters:
  S0=1.1, T=1, beta=0.7, nu0=0.2, sigma_nu=0.08, rho=-0.4, r=0
  M=100000, N_T ∈ {100, 200, 400} (paper); demo uses reduced N_T
  Strike grid K ∈ {1.00, 1.02, ..., 1.20}
  10 independent seeds

Notes on implementation:
  - Log(S) feature transformation (Gobet & Lemor 2005)
  - Positivity fix for nu: max(nu, 1e-10) after each Euler step
  - Driver f=0 since r=0 and lambda_rho=lambda_nu=0
  - Delta extracted as Z_2 / (nu_0 * S_0^beta * sqrt(1-rho^2))
  - Paper runtimes: ~23s (N_T=100), ~25s (N_T=200), ~126s (N_T=400)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forward_simulation import simulate_sabr_paths
from src.rbsde_solver import (
    build_sabr_features,
    solve_rbsde_2factor,
    extract_sabr_delta,
)
from src.benchmarks import get_sabr_ctmc, SABR_CTMC_PRICES, SABR_CTMC_STRIKES
from src.reporting import (
    ensure_results_dir,
    save_price_table,
    save_convergence_table,
    plot_price_comparison,
    plot_convergence,
    plot_delta_profile,
    write_results_md,
)

# ---------------------------------------------------------------------------
# Model parameters (paper specification)
# ---------------------------------------------------------------------------
S0 = 1.1
T = 1.0
BETA = 0.7
NU0 = 0.2
SIGMA_NU = 0.08
RHO = -0.4
R = 0.0
M = 100_000  # paths per seed

# Strike grid (paper)
STRIKES = np.array([1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20])

# N_T values to test (use smaller for demo runs; paper uses {100, 200, 400})
NT_VALUES = [10, 50, 100]  # reduced for practical runtime; full: [100, 200, 400]

# Number of independent seeds
N_SEEDS = 3  # paper uses 10; reduced for runtime


def run_sabr_experiment(
    N_T: int, n_seeds: int = N_SEEDS
) -> dict:
    """
    Run the SABR RBSDE experiment for given N_T and n_seeds.

    Parameters
    ----------
    N_T     : Number of time steps
    n_seeds : Number of independent seeds

    Returns
    -------
    dict with:
      'mean_prices'  : (n_strikes,) mean prices across seeds
      'std_prices'   : (n_strikes,) std-dev of prices
      'mean_deltas'  : (n_strikes,) mean deltas
      'std_deltas'   : (n_strikes,) std-dev of deltas
      'mean_runtime' : float — average runtime per seed (seconds)
    """
    dt = T / N_T
    n_strikes = len(STRIKES)
    prices_all = np.zeros((n_seeds, n_strikes))
    deltas_all = np.zeros((n_seeds, n_strikes))
    runtimes = []

    for seed_idx in range(n_seeds):
        seed = seed_idx * 100 + N_T  # deterministic seed
        print(f"    Seed {seed_idx + 1}/{n_seeds} (N_T={N_T})...")

        t_seed = time.time()

        # Simulate forward paths — shared across all strikes
        S_paths, nu_paths, dB1, dB2 = simulate_sabr_paths(
            S0=S0, nu0=NU0, T=T, N_T=N_T, M=M,
            beta=BETA, sigma_nu=SIGMA_NU, rho=RHO, seed=seed,
        )

        for k_idx, K in enumerate(STRIKES):
            result = solve_rbsde_2factor(
                S_paths=S_paths,
                nu_paths=nu_paths,
                dB1=dB1,
                dB2=dB2,
                dt=dt,
                K=K,
                feature_builder=build_sabr_features,
                driver_fn=None,  # f=0 for r=0, lambda=0
                n_picard=10,
                train_frac=0.75,
                r=R,
                store_path=False,
            )
            prices_all[seed_idx, k_idx] = result["price"]
            deltas_all[seed_idx, k_idx] = extract_sabr_delta(
                Z0=result["Z0"],
                nu0=NU0,
                S0=S0,
                beta=BETA,
                rho=RHO,
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


def main() -> None:
    """Run full Experiment 1 and save results."""
    ensure_results_dir()
    print("=" * 65)
    print("Experiment 1: Constant-Correlation SABR American Put Benchmark")
    print(f"  M={M}, N_T∈{NT_VALUES}, Seeds={N_SEEDS}")
    print(f"  S0={S0}, beta={BETA}, nu0={NU0}, sigma_nu={SIGMA_NU}, rho={RHO}")
    print("=" * 65)

    ctmc = get_sabr_ctmc()
    ctmc_prices = ctmc["ctmc_price"].values

    results_by_nt = {}

    for N_T in NT_VALUES:
        print(f"\n[N_T = {N_T}]")
        res = run_sabr_experiment(N_T=N_T)
        results_by_nt[N_T] = res

        # Save per-N_T table
        save_price_table(
            strikes=STRIKES,
            rbsde_mean=res["mean_prices"],
            rbsde_std=res["std_prices"],
            benchmark=ctmc_prices,
            benchmark_name="ctmc",
            filename=f"exp1_sabr_prices_NT{N_T}.csv",
            n_t_label=str(N_T),
        )
        # Plot price comparison for each N_T
        plot_price_comparison(
            strikes=STRIKES,
            rbsde_mean=res["mean_prices"],
            rbsde_std=res["std_prices"],
            benchmark=ctmc_prices,
            benchmark_name="CTMC (Cui et al. 2018)",
            title=f"SABR American Put: RBSDE vs CTMC (N_T={N_T})",
            filename=f"exp1_sabr_prices_NT{N_T}.png",
        )

    # Remap keys for reporting helpers (expect "mean"/"std")
    _nt_report = {nt: {"mean": r["mean_prices"], "std": r["std_prices"]}
                  for nt, r in results_by_nt.items()}

    # Convergence table
    save_convergence_table(
        strikes=STRIKES,
        results_by_nt=_nt_report,
        benchmark=ctmc_prices,
        benchmark_name="ctmc",
        filename="exp1_sabr_convergence.csv",
    )
    plot_convergence(
        strikes=STRIKES,
        results_by_nt=_nt_report,
        benchmark=ctmc_prices,
        benchmark_name="CTMC",
        title="SABR American Put: Convergence in N_T",
        filename="exp1_sabr_convergence.png",
    )

    # Delta profile for finest grid
    best_nt = max(NT_VALUES)
    plot_delta_profile(
        strikes=STRIKES,
        deltas_mean=results_by_nt[best_nt]["mean_deltas"],
        deltas_std=results_by_nt[best_nt]["std_deltas"],
        title=f"SABR American Put: Spot Delta (N_T={best_nt})",
        filename="exp1_sabr_deltas.png",
    )

    # Print summary
    print("\n" + "=" * 65)
    print(f"Summary (N_T={best_nt}):")
    best = results_by_nt[best_nt]
    for k, p, std, delta, ctmc_p in zip(
        STRIKES, best["mean_prices"], best["std_prices"],
        best["mean_deltas"], ctmc_prices
    ):
        print(
            f"  K={k:.2f}  price={p:.5f}±{std:.5f}  ctmc={ctmc_p:.4f}"
            f"  err={abs(p-ctmc_p):.5f}  delta={delta:.4f}"
        )
    print(f"\n  Avg runtime (N_T={best_nt}): {best['mean_runtime']:.1f}s")

    # RESULTS.md entries
    best_df = pd.DataFrame({
        "strike": STRIKES,
        "rbsde_mean": best["mean_prices"],
        "rbsde_std": best["std_prices"],
        "ctmc": ctmc_prices,
        "abs_error": np.abs(best["mean_prices"] - ctmc_prices),
        "delta_mean": best["mean_deltas"],
    })
    write_results_md(
        sections=[
            {
                "title": f"Exp 1: SABR Benchmark (N_T={best_nt}, M={M}, Seeds={N_SEEDS})",
                "content": best_df,
            },
            {
                "title": "Exp 1: Runtimes",
                "content": "\n".join(
                    f"N_T={nt}: {results_by_nt[nt]['mean_runtime']:.1f}s"
                    for nt in NT_VALUES
                ),
            },
        ],
        filename="RESULTS_exp1.md",
    )

    print("\nExperiment 1 complete. Results saved to results/")


if __name__ == "__main__":
    main()
