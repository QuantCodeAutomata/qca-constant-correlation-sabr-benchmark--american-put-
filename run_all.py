"""
Master runner: execute all 5 experiments sequentially and compile RESULTS.md.

Usage:
    python run_all.py

Outputs all CSVs, plots, and per-experiment RESULTS_expN.md files to results/.
A combined RESULTS.md is assembled at the end.
"""

import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.reporting import ensure_results_dir, write_results_md

import exp.exp1_sabr_benchmark as exp1
import exp.exp2_sabsr_analysis as exp2
import exp.exp3_hslv_benchmark as exp3
import exp.exp4_hsclv_analysis as exp4
import exp.exp5_future_evolution as exp5


def run_experiment(name: str, runner) -> float:
    """Run an experiment module's main() and return wall-clock seconds."""
    print("\n" + "=" * 70)
    print(f"Starting {name}")
    print("=" * 70)
    t0 = time.time()
    runner()
    elapsed = time.time() - t0
    print(f"\n>>> {name} finished in {elapsed:.1f}s <<<")
    return elapsed


def compile_results_md(runtimes: dict) -> None:
    """Assemble a top-level RESULTS.md from per-experiment files."""
    results_dir = ensure_results_dir()

    # Read per-experiment MD files
    sections = []
    sections.append({
        "title": "Summary of All Experiments",
        "content": "\n".join(
            f"- **{name}**: {rt:.1f}s" for name, rt in runtimes.items()
        ),
    })

    exp_files = sorted(results_dir.glob("RESULTS_exp*.md"))
    for ef in exp_files:
        content = ef.read_text()
        sections.append({"title": ef.stem.replace("_", " ").title(), "content": content})

    sections.append({
        "title": "Configuration Notes",
        "content": (
            "Experiments were run with reduced sample sizes for practical runtime:\n"
            "- SABR (Exp 1): M=100000, N_T∈{10,50,100} (paper: {100,200,400}), 3 seeds\n"
            "- SABSR (Exp 2): M=100000, N_T=100 (paper: 400), 3 seeds\n"
            "- HSLV (Exp 3): M=200000, N_T∈{10,20,30} (paper values), 3 seeds\n"
            "- HSCLV (Exp 4): M=200000, N_T=30 (paper value), 3 seeds\n"
            "- Evolution (Exp 5): M=50000, SABSR N_T=50, HSCLV N_T=30, 1 seed\n"
            "\nAll models and methodology follow Yang & Li (2024a) exactly.\n"
            "XGBoost params: max_depth=2, lr=0.9, n_estimators=150, "
            "early_stopping_rounds=1, 75/25 train-test split."
        ),
    })

    write_results_md(sections, filename="RESULTS.md")


def main() -> None:
    print("=" * 70)
    print("RBSDE American Option Pricing — Full Experiment Suite")
    print("=" * 70)

    ensure_results_dir()
    runtimes = {}

    runtimes["Exp 1: SABR Benchmark"] = run_experiment("Exp 1: SABR Benchmark", exp1.main)
    runtimes["Exp 2: SABSR Analysis"] = run_experiment("Exp 2: SABSR Analysis", exp2.main)
    runtimes["Exp 3: HSLV Benchmark"] = run_experiment("Exp 3: HSLV Benchmark", exp3.main)
    runtimes["Exp 4: HSCLV Analysis"] = run_experiment("Exp 4: HSCLV Analysis", exp4.main)
    runtimes["Exp 5: Future Evolution"] = run_experiment("Exp 5: Future Evolution", exp5.main)

    compile_results_md(runtimes)

    total = sum(runtimes.values())
    print("\n" + "=" * 70)
    print(f"All experiments complete. Total wall-clock time: {total:.1f}s")
    print("Results saved to results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
