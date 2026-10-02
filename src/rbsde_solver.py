"""
XGBoost-based Reflected BSDE (RBSDE) solver for American put option pricing.

Methodology (Yang & Li 2024a):
  Backward induction on equidistant grid. At each time step i (from N_T-1 to 0):
    1. Build feature matrix X_i from state variables
    2. Split M paths into 75% train / 25% validation
    3. Fit XGBoost for continuation expectation:
         E_hat[Y_{i+1} | X_i]
    4. Fit XGBoost for each Z-component:
         Z_k = (1/dt) * E_hat[Y_{i+1} * dW_{i+1,k} | X_i]
    5. Picard iterations (n=10) for implicit continuation:
         Y_tilde = E_cont + dt * f(t_i, X_i, Y_tilde, Z)
    6. American exercise (reflection):
         Y_i = max(Y_tilde, g(S_i))

XGBoost hyperparameters (paper specification):
  max_depth=2, learning_rate=0.9, n_estimators=150, early_stopping_rounds=1

Delta extraction uses the Z-component corresponding to the spot-orthogonal BM.
"""

import time
from typing import Callable, Optional

import numpy as np
import xgboost as xgb

# Default XGBoost parameters from the paper
XGB_PARAMS: dict = dict(
    objective="reg:squarederror",
    max_depth=2,
    learning_rate=0.9,
    n_estimators=150,
    early_stopping_rounds=1,
    verbosity=0,
    random_state=42,
)


def fit_xgb_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
) -> xgb.XGBRegressor:
    """
    Fit a single XGBoost regressor with early stopping on validation set.

    Parameters
    ----------
    X_train, y_train : Training features and targets
    X_val,   y_val   : Validation features and targets
    params           : XGBoost hyperparameter dict

    Returns
    -------
    Fitted XGBRegressor
    """
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_sabr_features(S: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """
    Feature matrix for constant-correlation SABR.
    Uses log(S) transformation following Gobet & Lemor (2005).

    Returns (M, 2) array with columns [log(S), nu].
    """
    return np.column_stack([np.log(np.maximum(S, 1e-10)), nu])


def build_sabsr_features(
    rho: np.ndarray, nu: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """
    Feature matrix for 3-factor SABSR.
    Returns (M, 3) array with columns [rho, nu, log(S)].
    """
    return np.column_stack([rho, nu, np.log(np.maximum(S, 1e-10))])


def build_hslv_features(nu: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Feature matrix for constant-correlation HSLV.
    Returns (M, 2) array with columns [nu, log(S)].
    """
    return np.column_stack([nu, np.log(np.maximum(S, 1e-10))])


def build_hsclv_features(
    rho: np.ndarray, nu: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """
    Feature matrix for 3-factor HSCLV.
    Returns (M, 3) array with columns [rho, nu, log(S)].
    """
    return np.column_stack([rho, nu, np.log(np.maximum(S, 1e-10))])


# ---------------------------------------------------------------------------
# Payoff
# ---------------------------------------------------------------------------

def american_put_payoff(S: np.ndarray, K: float) -> np.ndarray:
    """Return put payoff max(K - S, 0) element-wise."""
    return np.maximum(K - S, 0.0)


# ---------------------------------------------------------------------------
# 2-factor RBSDE solver
# ---------------------------------------------------------------------------

def solve_rbsde_2factor(
    S_paths: np.ndarray,
    nu_paths: np.ndarray,
    dB1: np.ndarray,
    dB2: np.ndarray,
    dt: float,
    K: float,
    feature_builder: Callable,
    driver_fn: Optional[Callable] = None,
    n_picard: int = 10,
    train_frac: float = 0.75,
    xgb_params: Optional[dict] = None,
    store_path: bool = False,
    r: float = 0.0,
) -> dict:
    """
    Solve the reflected BSDE for a 2-factor model (SABR or HSLV).

    At each time step i from N_T-1 down to 0:
      1. Construct feature matrix X_i from state at time i
      2. Split into train (75%) and validation (25%) sets
      3. Fit XGBoost for E[Y_{i+1} | X_i] (continuation)
      4. Fit XGBoost for Z_k = E[Y_{i+1}*dB_k/dt | X_i], k in {1,2}
      5. 10 Picard iterations for implicit Y_tilde
      6. Reflect: Y_i = max(Y_tilde, payoff(S_i))

    BM increment convention: dB1[:, i] and dB2[:, i] are the increments
    from t_i to t_{i+1}, each ~ N(0, dt). Z-targets are Y_next * dBk / dt.

    When driver_fn is None:
      f(t, X, Y, Z) = -r * Y  (Feynman-Kac discounting)
      For r=0 (SABR), all Picard steps yield Y_tilde = E_cont (trivially).
      For r>0 (HSLV), Picard converges to E_cont / (1 + r*dt).

    Parameters
    ----------
    S_paths      : (M, N_T+1) spot price paths
    nu_paths     : (M, N_T+1) vol-factor or variance paths
    dB1          : (M, N_T)   first independent BM increments ~ N(0, dt)
    dB2          : (M, N_T)   second independent BM increments ~ N(0, dt)
    dt           : Time step size T/N_T
    K            : American put strike
    feature_builder : Callable(S, nu) -> (M, n_feat) feature matrix
    driver_fn    : Optional BSDE driver f(t, X, Y, Z); if None uses -r*Y
    n_picard     : Number of Picard iterations (10 per paper)
    train_frac   : Training fraction (0.75 = 3:1 split)
    xgb_params   : XGBoost hyperparameters (uses XGB_PARAMS if None)
    store_path   : If True, store full Y matrix (M, N_T+1); memory-intensive
    r            : Risk-free rate for default discounting driver

    Returns
    -------
    dict with:
      'price'   : float — time-0 price E[Y_0]
      'Z0'      : (2,) ndarray — time-0 Z-vector (mean across paths)
      'Y_path'  : (M, N_T+1) ndarray or None
      'runtime' : float — elapsed wall-clock seconds
    """
    params = xgb_params if xgb_params is not None else XGB_PARAMS.copy()
    M, N_T1 = S_paths.shape
    N_T = N_T1 - 1
    n_train = int(M * train_frac)

    t0 = time.time()

    # Terminal condition
    Y = american_put_payoff(S_paths[:, N_T], K)

    Y_all = None
    if store_path:
        Y_all = np.zeros((M, N_T + 1), dtype=np.float64)
        Y_all[:, N_T] = Y.copy()

    Z0_stored = np.zeros(2, dtype=np.float64)

    for i in range(N_T - 1, -1, -1):
        # Build feature matrix at time i
        X_i = feature_builder(S_paths[:, i], nu_paths[:, i])

        # Train / validation split (consistent partitioning)
        X_train, X_val = X_i[:n_train], X_i[n_train:]
        Y_next = Y  # current Y corresponds to time i+1 (backward order)
        y_train, y_val = Y_next[:n_train], Y_next[n_train:]

        # Fit continuation expectation regressor
        cont_model = fit_xgb_regressor(X_train, y_train, X_val, y_val, params)
        E_cont = cont_model.predict(X_i)

        # Fit Z-component regressors: Z_k = E[Y_{i+1} * dBk / dt | X_i]
        Z_list = []
        for dBk in (dB1, dB2):
            target = Y_next * dBk[:, i] / dt
            t_train, t_val = target[:n_train], target[n_train:]
            z_model = fit_xgb_regressor(X_train, t_train, X_val, t_val, params)
            Z_list.append(z_model.predict(X_i))
        Z_arr = np.column_stack(Z_list)  # (M, 2)

        # Picard iterations for implicit continuation Y_tilde
        Y_tilde = E_cont.copy()
        for _ in range(n_picard):
            if driver_fn is not None:
                f_val = driver_fn(i * dt, X_i, Y_tilde, Z_arr)
            else:
                # Default: Feynman-Kac discounting, f = -r*Y
                f_val = -r * Y_tilde
            Y_tilde = E_cont + dt * f_val

        # American exercise reflection
        g_i = american_put_payoff(S_paths[:, i], K)
        Y = np.maximum(Y_tilde, g_i)

        if store_path:
            Y_all[:, i] = Y.copy()

        if i == 0:
            Z0_stored = Z_arr.mean(axis=0)

    return {
        "price": float(Y.mean()),
        "Z0": Z0_stored,
        "Y_path": Y_all,
        "runtime": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 3-factor RBSDE solver
# ---------------------------------------------------------------------------

def solve_rbsde_3factor(
    rho_paths: np.ndarray,
    nu_paths: np.ndarray,
    S_paths: np.ndarray,
    dW_rho: np.ndarray,
    dW_nu: np.ndarray,
    dW_s: np.ndarray,
    dt: float,
    K: float,
    feature_builder: Callable,
    driver_fn: Optional[Callable] = None,
    n_picard: int = 10,
    train_frac: float = 0.75,
    xgb_params: Optional[dict] = None,
    store_path: bool = False,
    r: float = 0.0,
) -> dict:
    """
    Solve the reflected BSDE for a 3-factor model (SABSR or HSCLV).

    Same structure as solve_rbsde_2factor but with 3 Brownian motions,
    requiring 4 XGBoost regressors per time step (1 continuation + 3 Z).

    BM ordering: [W^rho, W^nu, W^s] where W^s is the spot-orthogonal BM.
    Delta is extracted from Z_s (third component).

    Parameters
    ----------
    rho_paths : (M, N_T+1) correlation paths
    nu_paths  : (M, N_T+1) vol-factor or variance paths
    S_paths   : (M, N_T+1) spot price paths
    dW_rho    : (M, N_T)   BM increments for rho ~ N(0, dt)
    dW_nu     : (M, N_T)   BM increments for nu  ~ N(0, dt)
    dW_s      : (M, N_T)   BM increments for S   ~ N(0, dt)
    dt        : Time step T/N_T
    K         : Strike
    feature_builder : Callable(rho, nu, S) -> (M, n_feat) feature matrix
    driver_fn : Optional BSDE driver f(t, X, Y, Z)
    n_picard  : Picard iterations (10)
    train_frac: Training fraction (0.75)
    xgb_params: XGBoost params
    store_path: Store full Y path?
    r         : Risk-free rate

    Returns
    -------
    dict with:
      'price'   : float
      'Z0'      : (3,) ndarray  [Z_rho, Z_nu, Z_s]
      'Y_path'  : (M, N_T+1) or None
      'runtime' : float
    """
    params = xgb_params if xgb_params is not None else XGB_PARAMS.copy()
    M, N_T1 = S_paths.shape
    N_T = N_T1 - 1
    n_train = int(M * train_frac)

    t0 = time.time()

    Y = american_put_payoff(S_paths[:, N_T], K)

    Y_all = None
    if store_path:
        Y_all = np.zeros((M, N_T + 1), dtype=np.float64)
        Y_all[:, N_T] = Y.copy()

    Z0_stored = np.zeros(3, dtype=np.float64)

    for i in range(N_T - 1, -1, -1):
        X_i = feature_builder(rho_paths[:, i], nu_paths[:, i], S_paths[:, i])

        X_train, X_val = X_i[:n_train], X_i[n_train:]
        Y_next = Y
        y_train, y_val = Y_next[:n_train], Y_next[n_train:]

        # Continuation expectation
        cont_model = fit_xgb_regressor(X_train, y_train, X_val, y_val, params)
        E_cont = cont_model.predict(X_i)

        # Three Z-component regressors
        Z_list = []
        for dWk in (dW_rho, dW_nu, dW_s):
            target = Y_next * dWk[:, i] / dt
            t_train, t_val = target[:n_train], target[n_train:]
            z_model = fit_xgb_regressor(X_train, t_train, X_val, t_val, params)
            Z_list.append(z_model.predict(X_i))
        Z_arr = np.column_stack(Z_list)  # (M, 3)

        # Picard iterations
        Y_tilde = E_cont.copy()
        for _ in range(n_picard):
            if driver_fn is not None:
                f_val = driver_fn(i * dt, X_i, Y_tilde, Z_arr)
            else:
                f_val = -r * Y_tilde
            Y_tilde = E_cont + dt * f_val

        # Reflection
        g_i = american_put_payoff(S_paths[:, i], K)
        Y = np.maximum(Y_tilde, g_i)

        if store_path:
            Y_all[:, i] = Y.copy()

        if i == 0:
            Z0_stored = Z_arr.mean(axis=0)

    return {
        "price": float(Y.mean()),
        "Z0": Z0_stored,
        "Y_path": Y_all,
        "runtime": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Delta extraction functions
# ---------------------------------------------------------------------------

def extract_sabr_delta(
    Z0: np.ndarray,
    nu0: float,
    S0: float,
    beta: float,
    rho: float,
) -> float:
    """
    Extract spot delta for constant-correlation SABR.

    BM decomposition: dW^S = rho*dB1 + sqrt(1-rho^2)*dB2
    The spot-loading coefficient for dB2 (orthogonal to vol BM) is:
        alpha(nu)*beta(S)*sqrt(1-rho^2) = nu_0 * S_0^beta * sqrt(1-rho^2)

    Delta_S = Z_2 / (nu_0 * S_0^beta * sqrt(1-rho^2))

    Parameters
    ----------
    Z0   : (2,) Z-vector [Z_B1, Z_B2]
    nu0  : Initial vol factor
    S0   : Initial spot
    beta : Elasticity
    rho  : Constant correlation
    """
    Z2 = Z0[1]  # component for dB2 (orthogonal to vol)
    denom = nu0 * (S0 ** beta) * np.sqrt(max(1.0 - rho ** 2, 0.0))
    return float(Z2 / denom) if denom != 0 else 0.0


def extract_sabsr_delta(
    Z0: np.ndarray,
    nu0: float,
    S0: float,
    beta: float,
    rho0: float,
) -> float:
    """
    Extract spot delta for 3-factor SABSR.

    For SABSR (rho_rhonu=0):
        dS = nu*S^beta*(rho_t*dW^nu + sqrt(1-rho_t^2)*dW^s)
    The spot-orthogonal BM is W^s (third component of Z).

    Delta_S = Z_s / (nu_0 * S_0^beta * sqrt(1-rho_0^2))

    Parameters
    ----------
    Z0   : (3,) Z-vector [Z_rho, Z_nu, Z_s]
    nu0  : Initial vol factor
    S0   : Initial spot
    beta : Elasticity
    rho0 : Initial correlation
    """
    Z_s = Z0[2]
    denom = nu0 * (S0 ** beta) * np.sqrt(max(1.0 - rho0 ** 2, 0.0))
    return float(Z_s / denom) if denom != 0 else 0.0


def extract_hslv_delta(
    Z0: np.ndarray,
    nu0: float,
    S0: float,
    beta: float,
    rho: float,
) -> float:
    """
    Extract spot delta for constant-correlation HSLV.

    BM decomposition: dW^S = rho*dB1 + sqrt(1-rho^2)*dB2
    The orthogonal loading is sqrt(nu_0) * S_0^beta * sqrt(1-rho^2).

    Delta_S = Z_2 / (sqrt(nu_0) * S_0^beta * sqrt(1-rho^2))

    Parameters
    ----------
    Z0   : (2,) Z-vector [Z_B1, Z_B2]
    nu0  : Initial variance
    S0   : Initial spot
    beta : Elasticity
    rho  : Constant correlation
    """
    Z2 = Z0[1]
    denom = np.sqrt(max(nu0, 0.0)) * (S0 ** beta) * np.sqrt(max(1.0 - rho ** 2, 0.0))
    return float(Z2 / denom) if denom != 0 else 0.0


def extract_hsclv_delta(
    Z0: np.ndarray,
    nu0: float,
    S0: float,
    beta: float,
    rho0: float,
) -> float:
    """
    Extract spot delta for 3-factor HSCLV.

    Delta_S = Z_s / (sqrt(nu_0) * S_0^beta * sqrt(1-rho_0^2))

    Parameters
    ----------
    Z0   : (3,) Z-vector [Z_rho, Z_nu, Z_s]
    nu0  : Initial variance
    S0   : Initial spot
    beta : Elasticity
    rho0 : Initial correlation
    """
    Z_s = Z0[2]
    denom = np.sqrt(max(nu0, 0.0)) * (S0 ** beta) * np.sqrt(max(1.0 - rho0 ** 2, 0.0))
    return float(Z_s / denom) if denom != 0 else 0.0
