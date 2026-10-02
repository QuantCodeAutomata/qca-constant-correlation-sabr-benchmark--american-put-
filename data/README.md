# Data Directory

All experiments in this repository are **simulation-based**. No market data files are required.

Simulated Monte Carlo paths are generated at runtime by:
- `src/forward_simulation.py` — SABR and SABSR dynamics
- `src/hslv_simulation.py`   — HSLV and HSCLV dynamics

Benchmark reference values (CTMC and DPDB) are hard-coded in `src/benchmarks.py`
from Cui et al. (2018) and Yang & Li (2024a).
