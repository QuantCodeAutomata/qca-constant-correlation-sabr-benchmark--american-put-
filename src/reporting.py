"""
Reporting utilities: table generation, CSV export, and matplotlib visualisations.

All figures are saved to the results/ directory using a non-interactive Agg backend.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"


def ensure_results_dir() -> Path:
    """Create results/ directory if it does not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_price_table(
    strikes: np.ndarray,
    rbsde_mean: np.ndarray,
    rbsde_std: np.ndarray,
    benchmark: np.ndarray,
    benchmark_name: str,
    filename: str,
    n_t_label: str = "",
    extra_cols: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a price comparison table and save to CSV.

    Columns: strike, rbsde_mean, rbsde_std, <benchmark_name>, abs_error, rel_error_pct
    Prints a formatted summary to stdout.

    Parameters
    ----------
    strikes       : Strike array
    rbsde_mean    : Mean RBSDE prices across seeds
    rbsde_std     : Std-dev of RBSDE prices across seeds
    benchmark     : Reference benchmark prices
    benchmark_name: Name for benchmark column
    filename      : CSV filename (saved to results/)
    n_t_label     : Optional N_T label for display
    extra_cols    : Optional extra columns dict {name: array}

    Returns
    -------
    DataFrame written to results/<filename>
    """
    ensure_results_dir()
    df = pd.DataFrame(
        {
            "strike": strikes,
            "rbsde_mean": rbsde_mean,
            "rbsde_std": rbsde_std,
            benchmark_name: benchmark,
            "abs_error": np.abs(rbsde_mean - benchmark),
            "rel_error_pct": 100.0 * np.abs(rbsde_mean - benchmark) / np.maximum(benchmark, 1e-10),
        }
    )
    if extra_cols:
        for col_name, col_data in extra_cols.items():
            df[col_name] = col_data

    path = RESULTS_DIR / filename
    df.to_csv(path, index=False, float_format="%.6f")

    header = f"  Price Comparison — {benchmark_name}{' (N_T=' + n_t_label + ')' if n_t_label else ''}"
    print(header)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print()

    return df


def save_convergence_table(
    strikes: np.ndarray,
    results_by_nt: dict,
    benchmark: np.ndarray,
    benchmark_name: str,
    filename: str,
) -> pd.DataFrame:
    """
    Save a convergence table showing RBSDE prices for each N_T value.

    Parameters
    ----------
    strikes       : Strike array
    results_by_nt : {N_T: {'mean': array, 'std': array}}
    benchmark     : Reference prices
    benchmark_name: Column name for benchmark
    filename      : CSV filename

    Returns
    -------
    DataFrame saved to results/<filename>
    """
    ensure_results_dir()
    df = pd.DataFrame({"strike": strikes, benchmark_name: benchmark})
    for nt, res in sorted(results_by_nt.items()):
        df[f"rbsde_N{nt}"] = res["mean"]
        df[f"std_N{nt}"] = res["std"]

    path = RESULTS_DIR / filename
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  Convergence table saved: {path}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print()
    return df


def plot_price_comparison(
    strikes: np.ndarray,
    rbsde_mean: np.ndarray,
    rbsde_std: np.ndarray,
    benchmark: np.ndarray,
    benchmark_name: str,
    title: str,
    filename: str,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
) -> None:
    """
    Plot RBSDE prices vs benchmark with ±1 std-dev error bars.

    Optionally adds a shaded DPDB 95% confidence-interval band.

    Parameters
    ----------
    strikes       : Strike values for x-axis
    rbsde_mean    : Mean RBSDE prices
    rbsde_std     : Std-dev of RBSDE prices
    benchmark     : Reference prices
    benchmark_name: Legend label for benchmark
    title         : Figure title
    filename      : Output filename (saved to results/)
    ci_lower      : Optional lower 95% CI boundary
    ci_upper      : Optional upper 95% CI boundary
    """
    ensure_results_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(strikes, ci_lower, ci_upper, alpha=0.25, color="tab:green", label="95% CI")

    ax.errorbar(
        strikes,
        rbsde_mean,
        yerr=rbsde_std,
        fmt="o-",
        color="tab:blue",
        capsize=4,
        label="RBSDE/XGBoost (mean ± std)",
    )
    ax.plot(strikes, benchmark, "s--", color="tab:red", label=benchmark_name)

    ax.set_xlabel("Strike K")
    ax.set_ylabel("American Put Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_convergence(
    strikes: np.ndarray,
    results_by_nt: dict,
    benchmark: np.ndarray,
    benchmark_name: str,
    title: str,
    filename: str,
) -> None:
    """
    Plot convergence of RBSDE prices toward benchmark as N_T increases.

    Creates a figure with one panel showing prices by N_T, and
    another showing absolute errors vs benchmark.

    Parameters
    ----------
    strikes       : Strike values
    results_by_nt : {N_T: {'mean': array, 'std': array}}
    benchmark     : Reference prices
    benchmark_name: Benchmark label
    title         : Figure title
    filename      : Output file
    """
    ensure_results_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(results_by_nt)))
    for (nt, res), col in zip(sorted(results_by_nt.items()), colors):
        ax1.plot(strikes, res["mean"], "o-", color=col, label=f"N_T={nt}")
        ax2.plot(
            strikes,
            np.abs(res["mean"] - benchmark),
            "o-",
            color=col,
            label=f"N_T={nt}",
        )

    ax1.plot(strikes, benchmark, "s--", color="tab:red", lw=2, label=benchmark_name)
    ax1.set_xlabel("Strike K")
    ax1.set_ylabel("American Put Price")
    ax1.set_title(f"{title} — Prices")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("|RBSDE − Benchmark|")
    ax2.set_title(f"{title} — Absolute Errors")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Convergence plot saved: {path}")


def plot_delta_profile(
    strikes: np.ndarray,
    deltas_mean: np.ndarray,
    deltas_std: np.ndarray,
    title: str,
    filename: str,
) -> None:
    """
    Plot estimated spot deltas across the strike grid.

    Parameters
    ----------
    strikes    : Strike values
    deltas_mean: Mean deltas across seeds
    deltas_std : Std-dev of deltas across seeds
    title      : Figure title
    filename   : Output file
    """
    ensure_results_dir()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        strikes,
        deltas_mean,
        yerr=deltas_std,
        fmt="D-",
        color="tab:purple",
        capsize=4,
        label="RBSDE Delta (mean ± std)",
    )
    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Spot Delta Δ_S")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Delta plot saved: {path}")


def plot_future_evolution(
    times: np.ndarray,
    mean_trajectories: dict,
    title: str,
    filename: str,
    y_label: str = "Mean Option Value",
) -> None:
    """
    Plot mean future option value trajectories over time for different configurations.

    Parameters
    ----------
    times             : (N_T+1,) time grid [0, T]
    mean_trajectories : {label: (N_T+1,) mean Y_t array}
    title             : Figure title
    filename          : Output file
    y_label           : Y-axis label
    """
    ensure_results_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, traj in mean_trajectories.items():
        ax.plot(times, traj, label=label)
    ax.set_xlabel("Time t")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Future-evolution plot saved: {path}")


def plot_midtime_distribution(
    distributions: dict,
    title: str,
    filename: str,
    t_mid: float = 0.5,
) -> None:
    """
    Plot cross-sectional distributions of option value at t ≈ T/2.

    Uses histograms (normalised density) overlaid for each label.

    Parameters
    ----------
    distributions : {label: (M,) sample of Y_{t=T/2}}
    title         : Figure title
    filename      : Output file
    t_mid         : Mid-time label (for axis annotation)
    """
    ensure_results_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, samples in distributions.items():
        ax.hist(
            samples[samples > 0],  # exclude zero-payoff paths for readability
            bins=60,
            density=True,
            alpha=0.5,
            label=label,
        )
    ax.set_xlabel(f"Option Value at t = {t_mid}")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Distribution plot saved: {path}")


def plot_kappa_comparison(
    strikes: np.ndarray,
    sabsr_results: dict,
    sabr_mean: np.ndarray,
    filename: str,
    title: str = "SABSR vs SABR: Price Differences by kappa_rho",
) -> None:
    """
    Plot price differences between SABSR and constant-correlation SABR
    for each kappa_rho value.

    Parameters
    ----------
    strikes      : Strike array
    sabsr_results: {kappa_rho: {'mean': array}} for SABSR
    sabr_mean    : Mean prices for constant-correlation SABR benchmark
    filename     : Output file
    title        : Figure title
    """
    ensure_results_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    for kappa_rho, res in sabsr_results.items():
        diff = res["mean"] - sabr_mean
        ax.plot(strikes, diff, "o-", label=f"κ_ρ={kappa_rho}")
    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Price Difference (SABSR − SABR)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  kappa comparison plot saved: {path}")


def plot_mu_rho_comparison(
    strikes: np.ndarray,
    hsclv_results: dict,
    hslv_mean: np.ndarray,
    filename: str,
    title: str = "HSCLV vs HSLV: Price Differences by mu_rho",
) -> None:
    """
    Plot price differences between HSCLV and constant-correlation HSLV
    for each mu_rho value.

    Parameters
    ----------
    strikes      : Strike array
    hsclv_results: {mu_rho: {'mean': array}} for HSCLV
    hslv_mean    : Mean prices for constant-correlation HSLV
    filename     : Output file
    title        : Figure title
    """
    ensure_results_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    for mu_rho, res in hsclv_results.items():
        diff = res["mean"] - hslv_mean
        ax.plot(strikes, diff, "o-", label=f"μ_ρ={mu_rho}")
    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Price Difference (HSCLV − HSLV)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  mu_rho comparison plot saved: {path}")


def write_results_md(
    sections: list,
    filename: str = "RESULTS.md",
) -> None:
    """
    Write a RESULTS.md file summarising experiment outputs.

    Parameters
    ----------
    sections : List of dicts with keys 'title' (str) and 'content' (str or DataFrame)
    filename : Output filename in results/
    """
    ensure_results_dir()
    path = RESULTS_DIR / filename
    lines = ["# Experiment Results\n\n"]
    for sec in sections:
        lines.append(f"## {sec['title']}\n\n")
        content = sec["content"]
        if isinstance(content, pd.DataFrame):
            lines.append(content.to_markdown(index=False, floatfmt=".5f"))
            lines.append("\n\n")
        else:
            lines.append(str(content))
            lines.append("\n\n")
    path.write_text("".join(lines))
    print(f"  RESULTS.md written: {path}")
