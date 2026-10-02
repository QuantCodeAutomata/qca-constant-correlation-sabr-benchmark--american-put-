"""
Benchmark values for American put option pricing experiments.

Sources:
  CTMC benchmarks : Cui et al. (2018)
  DPDB benchmarks : Yang and Li (2024a)

These are reference values for comparing RBSDE/XGBoost prices and verifying
CI-containment for stochastic-correlation models.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CTMC benchmark: constant-correlation SABR
# Parameters: S0=1.1, T=1, beta=0.7, nu0=0.2, sigma_nu=0.08, rho=-0.4, r=0
# Source: Cui et al. (2018) as reported in Yang & Li (2024a)
# ---------------------------------------------------------------------------
SABR_CTMC_STRIKES = np.array(
    [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20],
    dtype=np.float64,
)
# CTMC prices from Cui et al. (2018); expected range 0.0424 – 0.1461
SABR_CTMC_PRICES = np.array(
    [0.0424, 0.0521, 0.0632, 0.0757, 0.0898, 0.1055, 0.1127, 0.1233, 0.1338, 0.1398, 0.1461],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# CTMC benchmark: constant-correlation HSLV
# Parameters: S0=10, T=1, beta=0.7, nu0=0.04, kappa_nu=4, mu_nu=0.035,
#             sigma_nu=0.15, r=0.03, mu_s=0.03, rho=-0.75
# Source: Cui et al. (2018) as reported in Yang & Li (2024a)
# ---------------------------------------------------------------------------
HSLV_CTMC_STRIKES = np.array([7, 8, 9, 10, 11, 12, 13], dtype=np.float64)
HSLV_CTMC_PRICES = np.array(
    [0.0002, 0.0044, 0.0473, 0.2813, 0.9999, 1.9999, 2.9999],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# DPDB benchmark: stochastic-correlation SABSR, kappa_rho ∈ {0.1, 2, 20}
# Parameters: S0=1.1, T=1, beta=0.7, nu0=0.2, sigma_nu=0.08, r=0,
#             rho0=-0.4, mu_rho=-0.4, sigma_rho=0.5, N_T=400
# Source: Yang and Li (2024a)
# ---------------------------------------------------------------------------
SABSR_DPDB_STRIKES = np.array(
    [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20],
    dtype=np.float64,
)

# kappa_rho = 0.1
SABSR_DPDB_KAPPA01_LOWER = np.array(
    [0.0436, 0.0535, 0.0647, 0.0775, 0.0918, 0.1078, 0.1152, 0.1260, 0.1368, 0.1428, 0.1492],
    dtype=np.float64,
)
SABSR_DPDB_KAPPA01_UPPER = np.array(
    [0.0448, 0.0547, 0.0660, 0.0789, 0.0933, 0.1095, 0.1169, 0.1278, 0.1387, 0.1448, 0.1514],
    dtype=np.float64,
)

# kappa_rho = 2
SABSR_DPDB_KAPPA2_LOWER = np.array(
    [0.0432, 0.0531, 0.0643, 0.0770, 0.0912, 0.1071, 0.1145, 0.1253, 0.1361, 0.1421, 0.1485],
    dtype=np.float64,
)
SABSR_DPDB_KAPPA2_UPPER = np.array(
    [0.0444, 0.0543, 0.0655, 0.0783, 0.0926, 0.1087, 0.1162, 0.1271, 0.1380, 0.1441, 0.1507],
    dtype=np.float64,
)

# kappa_rho = 20
SABSR_DPDB_KAPPA20_LOWER = np.array(
    [0.0425, 0.0522, 0.0633, 0.0758, 0.0899, 0.1056, 0.1128, 0.1236, 0.1341, 0.1401, 0.1464],
    dtype=np.float64,
)
SABSR_DPDB_KAPPA20_UPPER = np.array(
    [0.0436, 0.0533, 0.0645, 0.0771, 0.0913, 0.1071, 0.1144, 0.1253, 0.1359, 0.1420, 0.1483],
    dtype=np.float64,
)

_SABSR_DPDB_MAP: dict = {
    0.1: (SABSR_DPDB_KAPPA01_LOWER, SABSR_DPDB_KAPPA01_UPPER),
    2.0: (SABSR_DPDB_KAPPA2_LOWER, SABSR_DPDB_KAPPA2_UPPER),
    20.0: (SABSR_DPDB_KAPPA20_LOWER, SABSR_DPDB_KAPPA20_UPPER),
}

# ---------------------------------------------------------------------------
# DPDB benchmark: stochastic-correlation HSCLV, mu_rho ∈ {-0.2, 0, 0.75}
# Parameters: S0=10, T=1, beta=0.7, nu0=0.04, kappa_nu=4, mu_nu=0.035,
#             sigma_nu=0.15, r=0.03, mu_s=0.03, rho0=-0.75, kappa_rho=2.0,
#             sigma_rho=0.1, N_T=30
# Source: Yang and Li (2024a)
# ---------------------------------------------------------------------------
HSCLV_DPDB_STRIKES = np.array([7, 8, 9, 10, 11, 12, 13], dtype=np.float64)

# mu_rho = -0.2
HSCLV_DPDB_MU_NEG02_LOWER = np.array(
    [0.0001, 0.0038, 0.0454, 0.2769, 0.9984, 1.9979, 2.9979],
    dtype=np.float64,
)
HSCLV_DPDB_MU_NEG02_UPPER = np.array(
    [0.0003, 0.0050, 0.0494, 0.2799, 0.9994, 1.9989, 2.9989],
    dtype=np.float64,
)

# mu_rho = 0
HSCLV_DPDB_MU0_LOWER = np.array(
    [0.0001, 0.0038, 0.0455, 0.2771, 0.9984, 1.9979, 2.9979],
    dtype=np.float64,
)
HSCLV_DPDB_MU0_UPPER = np.array(
    [0.0003, 0.0050, 0.0495, 0.2801, 0.9994, 1.9989, 2.9989],
    dtype=np.float64,
)

# mu_rho = 0.75
HSCLV_DPDB_MU075_LOWER = np.array(
    [0.0000, 0.0033, 0.0432, 0.2740, 0.9981, 1.9977, 2.9977],
    dtype=np.float64,
)
HSCLV_DPDB_MU075_UPPER = np.array(
    [0.0002, 0.0045, 0.0472, 0.2770, 0.9991, 1.9987, 2.9987],
    dtype=np.float64,
)

_HSCLV_DPDB_MAP: dict = {
    -0.2: (HSCLV_DPDB_MU_NEG02_LOWER, HSCLV_DPDB_MU_NEG02_UPPER),
    0.0: (HSCLV_DPDB_MU0_LOWER, HSCLV_DPDB_MU0_UPPER),
    0.75: (HSCLV_DPDB_MU075_LOWER, HSCLV_DPDB_MU075_UPPER),
}


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------

def get_sabr_ctmc() -> pd.DataFrame:
    """Return SABR CTMC benchmark as a tidy DataFrame."""
    return pd.DataFrame(
        {"strike": SABR_CTMC_STRIKES, "ctmc_price": SABR_CTMC_PRICES}
    )


def get_hslv_ctmc() -> pd.DataFrame:
    """Return HSLV CTMC benchmark as a tidy DataFrame."""
    return pd.DataFrame(
        {"strike": HSLV_CTMC_STRIKES, "ctmc_price": HSLV_CTMC_PRICES}
    )


def get_sabsr_dpdb(kappa_rho: float) -> pd.DataFrame:
    """
    Return SABSR DPDB 95% confidence intervals as a DataFrame.

    Parameters
    ----------
    kappa_rho : Correlation mean-reversion speed; must be in {0.1, 2.0, 20.0}

    Returns
    -------
    DataFrame with columns: strike, dpdb_lower, dpdb_upper, dpdb_center
    """
    key = float(kappa_rho)
    if key not in _SABSR_DPDB_MAP:
        raise ValueError(f"kappa_rho must be one of {list(_SABSR_DPDB_MAP.keys())}; got {kappa_rho}")
    lower, upper = _SABSR_DPDB_MAP[key]
    return pd.DataFrame(
        {
            "strike": SABSR_DPDB_STRIKES,
            "dpdb_lower": lower,
            "dpdb_upper": upper,
            "dpdb_center": (lower + upper) / 2,
        }
    )


def get_hsclv_dpdb(mu_rho: float) -> pd.DataFrame:
    """
    Return HSCLV DPDB 95% confidence intervals as a DataFrame.

    Parameters
    ----------
    mu_rho : Long-run correlation; must be in {-0.2, 0.0, 0.75}

    Returns
    -------
    DataFrame with columns: strike, dpdb_lower, dpdb_upper, dpdb_center
    """
    key = float(mu_rho)
    if key not in _HSCLV_DPDB_MAP:
        raise ValueError(f"mu_rho must be one of {list(_HSCLV_DPDB_MAP.keys())}; got {mu_rho}")
    lower, upper = _HSCLV_DPDB_MAP[key]
    return pd.DataFrame(
        {
            "strike": HSCLV_DPDB_STRIKES,
            "dpdb_lower": lower,
            "dpdb_upper": upper,
            "dpdb_center": (lower + upper) / 2,
        }
    )
