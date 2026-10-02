"""
Forward path simulation for HSLV and HSCLV models.

Model dynamics:
  HSLV (constant correlation, CIR variance):
    d nu_t = kappa_nu*(mu_nu - nu_t)*dt + sigma_nu*sqrt(nu_t)*dW^nu_t
    d S_t  = mu_s*S_t*dt + sqrt(nu_t)*S_t^beta*dW^S_t
    Corr(dW^nu, dW^S) = rho

  HSCLV (stochastic correlation, 3-factor, CIR variance):
    d rho_t = kappa_rho*(mu_rho - rho_t)*(1 - rho_t^2)*dt
              + sigma_rho*(1 - rho_t^2)*dW^rho_t
    d nu_t  = kappa_nu*(mu_nu - nu_t)*dt + sigma_nu*sqrt(nu_t)*dW^nu_t  [rho_rhonu=0]
    d S_t   = mu_s*S_t*dt + sqrt(nu_t)*S_t^beta*(rho_t*dW^nu_t + sqrt(1-rho_t^2)*dW^s_t)

Numerical safeguards:
  - CIR full truncation: replace nu with max(nu, 0) before using it in diffusion/drift
  - rho clipped to (-1+1e-6, 1-1e-6) after each Euler step

References:
  Yang & Li (2024a) — RBSDE pricing with XGBoost
  Cui et al. (2018) — CTMC benchmark for HSLV
"""

import numpy as np


def simulate_hslv_paths(
    S0: float,
    nu0: float,
    T: float,
    N_T: int,
    M: int,
    beta: float,
    kappa_nu: float,
    mu_nu: float,
    sigma_nu: float,
    mu_s: float,
    rho: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate constant-correlation HSLV paths via Euler-Maruyama.

    Model:
        d nu_t = kappa_nu*(mu_nu - nu_t)*dt + sigma_nu*sqrt(nu_t)*dW^nu_t
        d S_t  = mu_s*S_t*dt + sqrt(nu_t)*S_t^beta*dW^S_t
        Corr(dW^nu, dW^S) = rho

    CIR Full-Truncation Scheme (Higham & Mao):
        nu_eff = max(nu_t, 0)   (used in both drift and diffusion)
        nu_{t+dt} = nu_t + kappa_nu*(mu_nu - nu_eff)*dt
                          + sigma_nu*sqrt(nu_eff)*dW^nu
    This ensures no negative square roots while preserving the drift direction.

    Brownian decomposition (two independent BMs):
        dW^nu = dB1
        dW^S  = rho*dB1 + sqrt(1 - rho^2)*dB2

    Parameters
    ----------
    S0       : Initial spot price
    nu0      : Initial variance (nu, not vol)
    T        : Maturity in years
    N_T      : Number of equidistant time steps
    M        : Number of Monte Carlo paths
    beta     : CEV elasticity parameter
    kappa_nu : CIR mean-reversion speed
    mu_nu    : CIR long-run mean (variance)
    sigma_nu : CIR vol-of-vol coefficient
    mu_s     : Risk-neutral drift of spot (= r)
    rho      : Constant instantaneous correlation
    seed     : Random seed

    Returns
    -------
    S_paths  : (M, N_T+1) float64 spot price paths
    nu_paths : (M, N_T+1) float64 variance paths
    dB1      : (M, N_T)   float64 independent BM increments ~ N(0, dt)
    dB2      : (M, N_T)   float64 independent BM increments ~ N(0, dt)
    """
    rng = np.random.default_rng(seed)
    dt = T / N_T
    sqrt_dt = np.sqrt(dt)
    sqrt_1_rho2 = np.sqrt(max(1.0 - rho ** 2, 0.0))

    # Pre-generate all increments
    dZ1 = rng.standard_normal((M, N_T)).astype(np.float64)
    dZ2 = rng.standard_normal((M, N_T)).astype(np.float64)

    dB1 = dZ1 * sqrt_dt   # BM increments for nu
    dB2 = dZ2 * sqrt_dt   # independent BM increments

    # Correlated BM for S
    dW_S = rho * dB1 + sqrt_1_rho2 * dB2

    # Initialise arrays
    S_paths = np.empty((M, N_T + 1), dtype=np.float64)
    nu_paths = np.empty((M, N_T + 1), dtype=np.float64)
    S_paths[:, 0] = S0
    nu_paths[:, 0] = nu0

    for i in range(N_T):
        nu_i = nu_paths[:, i]
        S_i = S_paths[:, i]

        # --- CIR full-truncation for nu ---
        nu_eff = np.maximum(nu_i, 0.0)
        sqrt_nu_eff = np.sqrt(nu_eff)

        nu_next = (
            nu_i
            + kappa_nu * (mu_nu - nu_eff) * dt
            + sigma_nu * sqrt_nu_eff * dB1[:, i]
        )
        nu_paths[:, i + 1] = nu_next  # may be negative; truncated at use

        # --- S update with risk-neutral drift ---
        dS = (
            mu_s * S_i * dt
            + sqrt_nu_eff * (S_i ** beta) * dW_S[:, i]
        )
        S_paths[:, i + 1] = np.maximum(S_i + dS, 1e-10)

    return S_paths, nu_paths, dB1, dB2


def simulate_hsclv_paths(
    S0: float,
    nu0: float,
    rho0: float,
    T: float,
    N_T: int,
    M: int,
    beta: float,
    kappa_nu: float,
    mu_nu: float,
    sigma_nu: float,
    mu_s: float,
    kappa_rho: float,
    mu_rho: float,
    sigma_rho: float,
    rho_rhonu: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate 3-factor stochastic-correlation HSCLV paths via Euler-Maruyama.

    Model (with rho_rhonu = 0, as in the paper):
        d rho_t = kappa_rho*(mu_rho - rho_t)*(1 - rho_t^2)*dt
                  + sigma_rho*(1 - rho_t^2)*dW^rho_t
        d nu_t  = kappa_nu*(mu_nu - nu_t)*dt + sigma_nu*sqrt(nu_t)*dW^nu_t
        d S_t   = mu_s*S_t*dt
                  + sqrt(nu_t)*S_t^beta*(rho_t*dW^nu_t + sqrt(1 - rho_t^2)*dW^s_t)

    CIR full-truncation for nu_t; clip rho_t to (-1 + 1e-6, 1 - 1e-6).
    Three independent BMs: W^rho, W^nu, W^s.

    Note: variance diffusion uses sigma_nu*sqrt(nu_t) (not sigma_nu*nu_t),
    consistent with the CIR specification and paper Corollary substitutions.

    Parameters
    ----------
    S0, nu0, rho0 : Initial spot, variance, and correlation
    T             : Maturity
    N_T           : Number of time steps
    M             : Number of paths
    beta          : CEV elasticity
    kappa_nu      : CIR mean-reversion speed
    mu_nu         : CIR long-run variance
    sigma_nu      : CIR vol-of-vol
    mu_s          : Spot drift (= r for risk-neutral)
    kappa_rho     : Correlation mean-reversion speed
    mu_rho        : Long-run correlation level
    sigma_rho     : Correlation diffusion coefficient
    rho_rhonu     : Cross-correlation between W^rho and W^nu (0 in paper)
    seed          : Random seed

    Returns
    -------
    rho_paths : (M, N_T+1) float64
    nu_paths  : (M, N_T+1) float64
    S_paths   : (M, N_T+1) float64
    dW_rho    : (M, N_T)   float64 BM increments for rho ~ N(0, dt)
    dW_nu     : (M, N_T)   float64 BM increments for nu  ~ N(0, dt)
    dW_s      : (M, N_T)   float64 BM increments for S   ~ N(0, dt)
    """
    rng = np.random.default_rng(seed)
    dt = T / N_T
    sqrt_dt = np.sqrt(dt)
    RHO_CLIP = 1.0 - 1e-6

    # Three independent standard normals
    dZ1 = rng.standard_normal((M, N_T)).astype(np.float64)  # -> dW^rho
    dZ2 = rng.standard_normal((M, N_T)).astype(np.float64)  # -> dW^nu
    dZ3 = rng.standard_normal((M, N_T)).astype(np.float64)  # -> dW^s

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

        # CIR full-truncation for nu
        nu_eff = np.maximum(nu_i, 0.0)
        sqrt_nu_eff = np.sqrt(nu_eff)

        # --- rho update ---
        drho = (
            kappa_rho * (mu_rho - rho_i) * one_minus_rho2 * dt
            + sigma_rho * one_minus_rho2 * dW_rho[:, i]
        )
        rho_paths[:, i + 1] = np.clip(rho_i + drho, -RHO_CLIP, RHO_CLIP)

        # --- nu update (CIR full-truncation) ---
        nu_next = (
            nu_i
            + kappa_nu * (mu_nu - nu_eff) * dt
            + sigma_nu * sqrt_nu_eff * dW_nu[:, i]
        )
        nu_paths[:, i + 1] = nu_next  # stored raw; truncated at use

        # --- S update ---
        dS = (
            mu_s * S_i * dt
            + sqrt_nu_eff * (S_i ** beta) * (
                rho_i * dW_nu[:, i] + sqrt_1_rho2 * dW_s[:, i]
            )
        )
        S_paths[:, i + 1] = np.maximum(S_i + dS, 1e-10)

    return rho_paths, nu_paths, S_paths, dW_rho, dW_nu, dW_s
