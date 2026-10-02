"""
Forward path simulation for SABR and SABSR models.

Model dynamics:
  SABR (constant correlation):
    d nu_t = sigma_nu * nu_t * dW^nu_t
    d S_t  = nu_t * S_t^beta * dW^S_t
    Corr(dW^nu, dW^S) = rho

  SABSR (stochastic correlation, 3-factor):
    d rho_t = kappa_rho*(mu_rho - rho_t)*(1 - rho_t^2)*dt
              + sigma_rho*(1 - rho_t^2)*dW^rho_t
    d nu_t  = sigma_nu * nu_t * dW^nu_t          [when rho_rhonu=0]
    d S_t   = nu_t * S_t^beta * (rho_t*dW^nu_t + sqrt(1-rho_t^2)*dW^s_t)

References:
  Gobet & Lemor (2005) — log(S) feature transformation
  Yang & Li (2024a) — RBSDE pricing paper
"""

import numpy as np


def simulate_sabr_paths(
    S0: float,
    nu0: float,
    T: float,
    N_T: int,
    M: int,
    beta: float,
    sigma_nu: float,
    rho: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate constant-correlation SABR paths via Euler-Maruyama.

    Model:
        d nu_t = sigma_nu * nu_t * dW^nu_t
        d S_t  = nu_t * S_t^beta * dW^S_t
        Corr(dW^nu, dW^S) = rho

    Brownian decomposition using two independent BMs (B1, B2):
        dW^nu = dB1
        dW^S  = rho * dB1 + sqrt(1 - rho^2) * dB2

    Positivity fix: nu_t = max(nu_t, 1e-10) after each Euler step.
    Feature log(S_t) is used in RBSDE regression following Gobet & Lemor (2005).

    Parameters
    ----------
    S0       : Initial spot price
    nu0      : Initial vol-of-vol factor
    T        : Maturity in years
    N_T      : Number of equidistant time steps
    M        : Number of Monte Carlo paths
    beta     : CEV elasticity parameter (0 <= beta <= 1)
    sigma_nu : Vol-of-vol (diffusion coefficient for nu)
    rho      : Constant instantaneous correlation in (-1, 1)
    seed     : NumPy random seed for reproducibility

    Returns
    -------
    S_paths  : (M, N_T+1) float64 array of spot prices
    nu_paths : (M, N_T+1) float64 array of vol factors
    dB1      : (M, N_T)   float64 independent BM increments ~ N(0, dt)
    dB2      : (M, N_T)   float64 independent BM increments ~ N(0, dt)
    """
    rng = np.random.default_rng(seed)
    dt = T / N_T
    sqrt_dt = np.sqrt(dt)
    sqrt_1_rho2 = np.sqrt(1.0 - rho ** 2)

    # Pre-generate all standard normal increments
    dZ1 = rng.standard_normal((M, N_T)).astype(np.float64)  # for dW^nu
    dZ2 = rng.standard_normal((M, N_T)).astype(np.float64)  # independent

    # BM increments scaled by sqrt(dt): ~ N(0, dt)
    dB1 = dZ1 * sqrt_dt
    dB2 = dZ2 * sqrt_dt

    # Correlated BM increments for S
    dW_S = rho * dB1 + sqrt_1_rho2 * dB2  # ~ N(0, dt)

    # Initialise path arrays
    S_paths = np.empty((M, N_T + 1), dtype=np.float64)
    nu_paths = np.empty((M, N_T + 1), dtype=np.float64)
    S_paths[:, 0] = S0
    nu_paths[:, 0] = nu0

    # Euler-Maruyama forward simulation
    for i in range(N_T):
        nu_i = nu_paths[:, i]
        S_i = S_paths[:, i]

        # --- nu update (geometric Brownian motion) ---
        nu_next = nu_i + sigma_nu * nu_i * dB1[:, i]
        # Positivity fix (floor at small positive value)
        nu_paths[:, i + 1] = np.maximum(nu_next, 1e-10)

        # --- S update (CEV diffusion, no drift for r=0) ---
        S_next = S_i + nu_i * (S_i ** beta) * dW_S[:, i]
        # Floor spot at small positive value to avoid log(0)
        S_paths[:, i + 1] = np.maximum(S_next, 1e-10)

    return S_paths, nu_paths, dB1, dB2


def simulate_sabsr_paths(
    S0: float,
    nu0: float,
    rho0: float,
    T: float,
    N_T: int,
    M: int,
    beta: float,
    sigma_nu: float,
    sigma_rho: float,
    mu_rho: float,
    kappa_rho: float,
    rho_rhonu: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate 3-factor stochastic-correlation SABSR paths via Euler-Maruyama.

    Model (with rho_rhonu = 0, as in the paper):
        d rho_t = kappa_rho*(mu_rho - rho_t)*(1 - rho_t^2)*dt
                  + sigma_rho*(1 - rho_t^2)*dW^rho_t
        d nu_t  = sigma_nu * nu_t * dW^nu_t       [independent of W^rho]
        d S_t   = nu_t * S_t^beta * (rho_t*dW^nu_t + sqrt(1 - rho_t^2)*dW^s_t)

    Three independent Brownian motions: W^rho, W^nu, W^s.
    Numerical safeguards:
        - rho_t clipped to (-1 + 1e-6, 1 - 1e-6) after each step
        - nu_t floored at 1e-10 after each step

    Parameters
    ----------
    S0, nu0, rho0 : Initial spot, vol factor, and correlation
    T             : Maturity in years
    N_T           : Number of equidistant time steps
    M             : Number of Monte Carlo paths
    beta          : CEV elasticity
    sigma_nu      : Vol-of-vol for nu
    sigma_rho     : Diffusion coefficient for rho process
    mu_rho        : Long-run (mean) correlation level
    kappa_rho     : Mean-reversion speed for rho
    rho_rhonu     : Correlation between W^rho and W^nu (set to 0 in paper)
    seed          : Random seed

    Returns
    -------
    rho_paths : (M, N_T+1) float64 — correlation paths
    nu_paths  : (M, N_T+1) float64 — vol factor paths
    S_paths   : (M, N_T+1) float64 — spot price paths
    dW_rho    : (M, N_T)   float64 — BM increments for rho, ~ N(0, dt)
    dW_nu     : (M, N_T)   float64 — BM increments for nu, ~ N(0, dt)
    dW_s      : (M, N_T)   float64 — BM increments for spot, ~ N(0, dt)
    """
    rng = np.random.default_rng(seed)
    dt = T / N_T
    sqrt_dt = np.sqrt(dt)

    RHO_CLIP = 1.0 - 1e-6

    # Three independent standard normal increments
    dZ1 = rng.standard_normal((M, N_T)).astype(np.float64)  # -> dW^rho
    dZ2 = rng.standard_normal((M, N_T)).astype(np.float64)  # -> dW^nu
    dZ3 = rng.standard_normal((M, N_T)).astype(np.float64)  # -> dW^s

    # BM increments ~ N(0, dt)
    dW_rho = dZ1 * sqrt_dt
    dW_nu = dZ2 * sqrt_dt
    dW_s = dZ3 * sqrt_dt

    # Initialise paths
    rho_paths = np.empty((M, N_T + 1), dtype=np.float64)
    nu_paths = np.empty((M, N_T + 1), dtype=np.float64)
    S_paths = np.empty((M, N_T + 1), dtype=np.float64)
    rho_paths[:, 0] = rho0
    nu_paths[:, 0] = nu0
    S_paths[:, 0] = S0

    for i in range(N_T):
        rho_i = rho_paths[:, i]
        nu_i = nu_paths[:, i]
        S_i = S_paths[:, i]
        one_minus_rho2 = 1.0 - rho_i ** 2
        sqrt_1_rho2 = np.sqrt(np.maximum(one_minus_rho2, 0.0))

        # --- rho update ---
        drho = (
            kappa_rho * (mu_rho - rho_i) * one_minus_rho2 * dt
            + sigma_rho * one_minus_rho2 * dW_rho[:, i]
        )
        rho_next = np.clip(rho_i + drho, -RHO_CLIP, RHO_CLIP)
        rho_paths[:, i + 1] = rho_next

        # --- nu update (geometric BM, rho_rhonu=0 => independent of W^rho) ---
        nu_next = nu_i + sigma_nu * nu_i * dW_nu[:, i]
        nu_paths[:, i + 1] = np.maximum(nu_next, 1e-10)

        # --- S update ---
        # dS = nu * S^beta * (rho * dW^nu + sqrt(1-rho^2) * dW^s)
        dS = nu_i * (S_i ** beta) * (
            rho_i * dW_nu[:, i] + sqrt_1_rho2 * dW_s[:, i]
        )
        S_paths[:, i + 1] = np.maximum(S_i + dS, 1e-10)

    return rho_paths, nu_paths, S_paths, dW_rho, dW_nu, dW_s
