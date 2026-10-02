# American Put Pricing via XGBoost-Based Reflected BSDE

This repository implements and validates the paper's XGBoost-based reflected BSDE (RBSDE) framework for American put option pricing under stochastic local volatility (SLV) and stochastic-correlation models.

## Overview

Five experiments are reproduced:

| Experiment | Model | Description |
|-----------|-------|-------------|
| exp_1 | Constant-correlation SABR | American put benchmark vs CTMC, convergence in N_T |
| exp_2 | Stochastic-correlation SABSR | Effect of kappa_rho, comparison with DPDB CIs |
| exp_3 | Constant-correlation HSLV | American put benchmark vs CTMC, convergence in N_T |
| exp_4 | Stochastic-correlation HSCLV | Effect of mu_rho, comparison with DPDB CIs |
| exp_5 | Future evolution | Pathwise Y_t trajectories under SABSR and HSCLV |

## Repository Structure

```
repo/
├── data/                # Placeholder (experiments are simulation-based)
├── src/
│   ├── forward_simulation.py   # SABR and SABSR path generation
│   ├── hslv_simulation.py      # HSLV and HSCLV path generation
│   ├── rbsde_solver.py         # XGBoost-based RBSDE backward induction
│   ├── benchmarks.py           # CTMC and DPDB benchmark values
│   └── reporting.py            # Tables and plots
├── exp/
│   ├── exp1_sabr_benchmark.py
│   ├── exp2_sabsr_analysis.py
│   ├── exp3_hslv_benchmark.py
│   ├── exp4_hsclv_analysis.py
│   └── exp5_future_evolution.py
├── tests/
│   ├── test_forward_simulation.py
│   ├── test_rbsde_solver.py
│   ├── test_benchmarks.py
│   └── test_experiments.py
├── results/
│   └── RESULTS.md
└── README.md
```

## Setup

```bash
pip install numpy pandas xgboost scipy matplotlib seaborn scikit-learn
```

## Running Experiments

```bash
# Run individual experiments
python exp/exp1_sabr_benchmark.py
python exp/exp2_sabsr_analysis.py
python exp/exp3_hslv_benchmark.py
python exp/exp4_hsclv_analysis.py
python exp/exp5_future_evolution.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Model Descriptions

### SABR (Constant Correlation)
- dν_t = σ_ν ν_t dW^ν_t
- dS_t = ν_t S_t^β dW^S_t
- Corr(dW^ν, dW^S) = ρ = -0.4

### SABSR (Stochastic Correlation)
- dρ_t = κ_ρ(μ_ρ - ρ_t)(1 - ρ_t²) dt + σ_ρ(1 - ρ_t²) dW^ρ_t
- dν_t = σ_ν ν_t dW^ν_t
- dS_t = ν_t S_t^β (ρ_t dW^ν_t + √(1-ρ_t²) dW^S_t)

### HSLV (Constant Correlation, CIR variance)
- dν_t = κ_ν(μ_ν - ν_t) dt + σ_ν √ν_t dW^ν_t
- dS_t = μ_s S_t dt + √ν_t S_t^β dW^S_t
- Corr(dW^ν, dW^S) = ρ = -0.75

### HSCLV (Stochastic Correlation, CIR variance)
- Same ρ_t dynamics as SABSR
- dν_t = κ_ν(μ_ν - ν_t) dt + σ_ν √ν_t dW^ν_t
- dS_t = μ_s S_t dt + √ν_t S_t^β (ρ_t dW^ν_t + √(1-ρ_t²) dW^S_t)

## RBSDE Solver

The reflected BSDE backward recursion at each time step i:
1. Z_i = (1/Δt) E_i[Y_{i+1} ΔW_{i+1}] via XGBoost regression
2. Ẽ[Y_{i+1}|X_i] via XGBoost regression
3. Ỹ_i = E[Y_{i+1}] + Δt · f(t_i, X_i, Ỹ_i, Z_i) (10 Picard iterations)
4. Y_i = max(Ỹ_i, g(S_i))  [reflection / American exercise]

XGBoost hyperparameters: max_depth=2, learning_rate=0.9, n_estimators=150,
early_stopping_rounds=1, 3:1 train:test split.

## References

- Yang, L. and Li, L. (2024a). American option pricing via reflected BSDE with XGBoost.
- Cui, Z. et al. (2018). CTMC benchmark for American options.
- Gobet, E. and Lemor, J.-P. (2005). Regression-based Monte Carlo for BSDE.
