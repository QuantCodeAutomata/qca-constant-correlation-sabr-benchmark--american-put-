"""
Experiment 3: Constant-Correlation HSLV Benchmark
==================================================
American put pricing under HSLV (Heston-type stochastic local volatility)
with constant correlation, compared against CTMC benchmarks (Cui et al. 2018).

Model:
  d nu_t = kappa_nu*(mu_nu - nu_t)*dt + sigma_nu*sqrt(nu_t)*dW^nu_t
  d S_t  = mu_s*S_t*dt + sqrt(nu_t)*S_t^beta*dW^S_t
  Corr(dW^nu, dW^S) = rho = -0.75

Paper parameters:
  S0=10, T=1, beta=0.7, nu0=0.04, kappa_nu=4, mu_nu=0.035
  sigma_nu=0.15, r=0.03, mu_s=0.03, rho=-0.75
  M=200000, N_T ∈ {10, 20, 30} (paper values)
  Strike grid K ∈ {7, 8, 9, 10, 11, 12, 13}
  10 seeds

CTMC benchmarks (Cui et al. 2018):
  K=7: 0.0002, K=8: 0.0044, K=9: 0.0473, K=10: 0.2813
  K=11: 0.9999, K=12: 1.9999, K=13: 2.9999

Notes:
  - CIR full-truncation for nu_t: use max(nu_t, 0) in drift and diffusion
  - Risk-neutral drift mu_s = r = 0.03 included for S
  - Driver f = -r*Y for Feynman-Kac discounting (lambda=0)
  - Delta: Z_2 / (sqrt(nu_0) * S_0^beta * sqrt(1-rho^2))
  - Deep ITM puts (K=11,12,13) should approach intrinsic values 1, 2, 3
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hslv_simulation import simulate_hslv_paths
from src.rbsde_solver import (
    build_hslv_features,
    solve_rbsde_2factor,
    extract_hslv_delta,
)
from src.benchmarks import get_hslv_ctmc, HSLV_CTMC_PRICES, HSLV_CTMC_STRIKES
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
S0 = 10.0
T = 1.0
BETA = 0.7
NU0 = 0.04
KAPPA_NU = 4.0
MU_NU = 0.035
SIGMA_NU = 0.15
MU_S = 0.03
R = 0.03
RHO = -0.75
M = 200_000  # paper specification

# Paper values for N_T (already small enough for practical runtime)
NT_VALUES = [10, 20, 30]

STRIKES = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
INTRINSIC = np.maximum(STRIKES - S0, 0.0)  # intrinsic value for ATM/ITM

N_SEEDS = 3  # demo; paper uses 10


def run_hslv_experiment(N_T: int, n_seeds: int = N_SEEDS) -> dict:
    """
    Run the HSLV RBSDE experiment for a given N_T.

    Parameters
    ----------
    N_T     : Number of time steps
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
        seed = seed_idx * 300 + N_T
        print(f"    Seed {seed_idx + 1}/{n_seeds} (N_T={N_T})...")

        t_seed = time.time()

        # Forward paths — shared across all strikes
        S_paths, nu_paths, dB1, dB2 = simulate_hslv_paths(
            S0=S0, nu0=NU0, T=T, N_T=N_T, M=M,
            beta=BETA, kappa_nu=KAPPA_NU, mu_nu=MU_NU,
            sigma_nu=SIGMA_NU, mu_s=MU_S, rho=RHO, seed=seed,
        )

        for k_idx, K in enumerate(STRIKES):
            result = solve_rbsde_2factor(
                S_paths=S_paths,
                nu_paths=nu_paths,
                dB1=dB1,
                dB2=dB2,
                dt=dt,
                K=K,
                feature_builder=build_hslv_features,
                driver_fn=None,   # default Feynman-Kac: f = -r*Y
                n_picard=10,
                train_frac=0.75,
                r=R,              # r=0.03 for discounting
                store_path=False,
            )
            prices_all[seed_idx, k_idx] = result["price"]
            deltas_all[seed_idx, k_idx] = extract_hslv_delta(
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
    """Run full Experiment 3 and save results."""
    ensure_results_dir()
    print("=" * 65)
    print("Experiment 3: Constant-Correlation HSLV American Put Benchmark")
    print(f"  M={M}, N_T∈{NT_VALUES}, Seeds={N_SEEDS}")
    print(f"  S0={S0}, beta={BETA}, nu0={NU0}, kappa_nu={KAPPA_NU}, rho={RHO}")
    print("=" * 65)

    ctmc = get_hslv_ctmc()
    ctmc_prices = ctmc["ctmc_price"].values

    results_by_nt = {}

    for N_T in NT_VALUES:
        print(f"\n[N_T = {N_T}]")
        res = run_hslv_experiment(N_T=N_T)
        results_by_nt[N_T] = res

        save_price_table(
            strikes=STRIKES,
            rbsde_mean=res["mean_prices"],
            rbsde_std=res["std_prices"],
            benchmark=ctmc_prices,
            benchmark_name="ctmc",
            filename=f"exp3_hslv_prices_NT{N_T}.csv",
            n_t_label=str(N_T),
        )
        plot_price_comparison(
            strikes=STRIKES,
            rbsde_mean=res["mean_prices"],
            rbsde_std=res["std_prices"],
            benchmark=ctmc_prices,
            benchmark_name="CTMC (Cui et al. 2018)",
            title=f"HSLV American Put: RBSDE vs CTMC (N_T={N_T})",
            filename=f"exp3_hslv_prices_NT{N_T}.png",
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
        filename="exp3_hslv_convergence.csv",
    )
    plot_convergence(
        strikes=STRIKES,
        results_by_nt=_nt_report,
        benchmark=ctmc_prices,
        benchmark_name="CTMC",
        title="HSLV American Put: Convergence in N_T",
        filename="exp3_hslv_convergence.png",
    )

    # Delta profile for finest grid
    best_nt = max(NT_VALUES)
    best = results_by_nt[best_nt]
    plot_delta_profile(
        strikes=STRIKES,
        deltas_mean=best["mean_deltas"],
        deltas_std=best["std_deltas"],
        title=f"HSLV American Put: Spot Delta (N_T={best_nt})",
        filename="exp3_hslv_deltas.png",
    )

    # Check near-intrinsic for deep ITM puts
    print("\n[Near-intrinsic value check for deep ITM puts]")
    deep_itm_strikes = np.array([11.0, 12.0, 13.0])
    intrinsic_vals = np.array([1.0, 2.0, 3.0])
    deep_idx = [np.where(STRIKES == k)[0][0] for k in deep_itm_strikes]
    for idx, k, intr in zip(deep_idx, deep_itm_strikes, intrinsic_vals):
        price = best["mean_prices"][idx]
        print(f"  K={k:.0f}: price={price:.4f}, intrinsic={intr:.4f}, diff={abs(price-intr):.4f}")

    # Check K=10 near CTMC
    k10_idx = np.where(STRIKES == 10.0)[0][0]
    print(f"\n  K=10: RBSDE={best['mean_prices'][k10_idx]:.4f}, CTMC=0.2813 (paper target ~0.2811)")

    # Summary
    print("\n" + "=" * 65)
    print(f"Summary (N_T={best_nt}):")
    for i, (k, p, std, ctmc_p) in enumerate(
        zip(STRIKES, best["mean_prices"], best["std_prices"], ctmc_prices)
    ):
        print(
            f"  K={k:.0f}  price={p:.5f}±{std:.5f}  ctmc={ctmc_p:.4f}"
            f"  err={abs(p-ctmc_p):.5f}"
        )
    print(f"\n  Avg runtime (N_T={best_nt}): {best['mean_runtime']:.1f}s")
    print("  (Paper target: ~6s N_T=10, ~11s N_T=20, ~16s N_T=30)")

    # RESULTS.md
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
                "title": f"Exp 3: HSLV Benchmark (N_T={best_nt}, M={M}, Seeds={N_SEEDS})",
                "content": best_df,
            },
            {
                "title": "Exp 3: Runtimes",
                "content": "\n".join(
                    f"N_T={nt}: {results_by_nt[nt]['mean_runtime']:.1f}s"
                    for nt in NT_VALUES
                ),
            },
        ],
        filename="RESULTS_exp3.md",
    )

    print("\nExperiment 3 complete. Results saved to results/")


if __name__ == "__main__":
    main()
