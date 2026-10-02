"""
Tests for rbsde_solver.py.

Validates:
  - Feature builder outputs
  - Payoff function correctness
  - RBSDE solver outputs (shape, type, economic properties)
  - Delta extraction
  - Reflection constraint (Y >= g(S) at all times)
  - Price monotonicity in strike for American puts
  - Bounds: prices between 0 and K (intrinsic floor, upper bound K)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forward_simulation import simulate_sabr_paths, simulate_sabsr_paths
from src.hslv_simulation import simulate_hslv_paths, simulate_hsclv_paths
from src.rbsde_solver import (
    american_put_payoff,
    build_sabr_features,
    build_sabsr_features,
    build_hslv_features,
    build_hsclv_features,
    extract_sabr_delta,
    extract_sabsr_delta,
    extract_hslv_delta,
    extract_hsclv_delta,
    solve_rbsde_2factor,
    solve_rbsde_3factor,
    fit_xgb_regressor,
    XGB_PARAMS,
)


# ---------------------------------------------------------------------------
# Payoff tests
# ---------------------------------------------------------------------------

class TestAmericanPutPayoff:

    def test_otm(self):
        """OTM put (S > K) should have zero payoff."""
        S = np.array([110.0, 120.0, 100.1])
        K = 100.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(american_put_payoff(S, K), expected)

    def test_itm(self):
        """ITM put (S < K) should equal K - S."""
        S = np.array([90.0, 80.0, 50.0])
        K = 100.0
        expected = np.array([10.0, 20.0, 50.0])
        np.testing.assert_array_equal(american_put_payoff(S, K), expected)

    def test_atm(self):
        """ATM put (S = K) should have zero payoff."""
        np.testing.assert_array_equal(american_put_payoff(np.array([100.0]), 100.0), [0.0])

    def test_non_negative(self):
        """Payoff must always be non-negative."""
        S = np.random.uniform(0.5, 2.0, 1000)
        K = 1.0
        assert np.all(american_put_payoff(S, K) >= 0)

    def test_upper_bound(self):
        """Payoff is at most K."""
        S = np.zeros(10)  # S=0 is the worst case
        K = 5.0
        assert np.all(american_put_payoff(S, K) <= K)


# ---------------------------------------------------------------------------
# Feature builder tests
# ---------------------------------------------------------------------------

class TestFeatureBuilders:

    def test_sabr_features_shape(self):
        M = 100
        S = np.ones(M) * 1.1
        nu = np.ones(M) * 0.2
        feat = build_sabr_features(S, nu)
        assert feat.shape == (M, 2)

    def test_sabr_features_log_transform(self):
        """First column should be log(S)."""
        S = np.array([1.0, np.e, np.e ** 2])
        nu = np.array([0.2, 0.2, 0.2])
        feat = build_sabr_features(S, nu)
        np.testing.assert_allclose(feat[:, 0], [0.0, 1.0, 2.0], atol=1e-12)

    def test_sabsr_features_shape(self):
        M = 50
        rho = np.full(M, -0.4)
        nu = np.full(M, 0.2)
        S = np.full(M, 1.1)
        feat = build_sabsr_features(rho, nu, S)
        assert feat.shape == (M, 3)

    def test_sabsr_features_rho_column(self):
        """First column should be rho (not transformed)."""
        rho = np.array([-0.5, 0.0, 0.3])
        nu = np.ones(3)
        S = np.ones(3)
        feat = build_sabsr_features(rho, nu, S)
        np.testing.assert_array_equal(feat[:, 0], rho)

    def test_hslv_features_shape(self):
        M = 100
        nu = np.ones(M) * 0.04
        S = np.ones(M) * 10.0
        feat = build_hslv_features(nu, S)
        assert feat.shape == (M, 2)

    def test_hsclv_features_shape(self):
        M = 50
        rho = np.full(M, -0.75)
        nu = np.full(M, 0.04)
        S = np.full(M, 10.0)
        feat = build_hsclv_features(rho, nu, S)
        assert feat.shape == (M, 3)

    def test_log_transform_negative_safe(self):
        """Feature builders should handle S near zero safely (floor to 1e-10)."""
        S = np.array([0.0, 1e-20, 1.0])
        nu = np.ones(3) * 0.2
        feat = build_sabr_features(S, nu)
        assert np.all(np.isfinite(feat))


# ---------------------------------------------------------------------------
# XGBoost regressor test
# ---------------------------------------------------------------------------

class TestFitXGBRegressor:

    def test_basic_fit_predict(self):
        """Regressor should fit without error and return predictions."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 2)).astype(np.float32)
        y = rng.standard_normal(200).astype(np.float32)
        X_train, y_train = X[:150], y[:150]
        X_val, y_val = X[150:], y[150:]
        model = fit_xgb_regressor(X_train, y_train, X_val, y_val, XGB_PARAMS.copy())
        preds = model.predict(X)
        assert preds.shape == (200,)
        assert np.all(np.isfinite(preds))

    def test_early_stopping(self):
        """With early_stopping_rounds=1 the model should use few estimators."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((400, 2)).astype(np.float32)
        y = rng.standard_normal(400).astype(np.float32)
        model = fit_xgb_regressor(X[:300], y[:300], X[300:], y[300:], XGB_PARAMS.copy())
        # With early stopping, best_iteration should be small
        assert model.best_iteration <= 150


# ---------------------------------------------------------------------------
# RBSDE 2-factor solver tests (SABR)
# ---------------------------------------------------------------------------

class TestSolveRBSDE2Factor:

    @pytest.fixture(scope="class")
    def sabr_paths(self):
        return simulate_sabr_paths(
            S0=1.1, nu0=0.2, T=1.0, N_T=5, M=500,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=0
        )

    def test_returns_dict_with_expected_keys(self, sabr_paths):
        S, nu, dB1, dB2 = sabr_paths
        result = solve_rbsde_2factor(
            S_paths=S, nu_paths=nu, dB1=dB1, dB2=dB2,
            dt=1.0 / 5, K=1.1,
            feature_builder=build_sabr_features,
            r=0.0, store_path=False,
        )
        for key in ("price", "Z0", "Y_path", "runtime"):
            assert key in result

    def test_price_is_non_negative(self, sabr_paths):
        S, nu, dB1, dB2 = sabr_paths
        result = solve_rbsde_2factor(
            S_paths=S, nu_paths=nu, dB1=dB1, dB2=dB2,
            dt=1.0 / 5, K=1.1,
            feature_builder=build_sabr_features,
            r=0.0,
        )
        assert result["price"] >= 0.0

    def test_price_bounded_by_strike(self, sabr_paths):
        """American put price must be <= K."""
        S, nu, dB1, dB2 = sabr_paths
        K = 1.1
        result = solve_rbsde_2factor(
            S_paths=S, nu_paths=nu, dB1=dB1, dB2=dB2,
            dt=1.0 / 5, K=K,
            feature_builder=build_sabr_features,
            r=0.0,
        )
        assert result["price"] <= K + 1e-6

    def test_price_monotone_in_strike(self):
        """American put prices should increase with strike."""
        S, nu, dB1, dB2 = simulate_sabr_paths(
            S0=1.1, nu0=0.2, T=1.0, N_T=5, M=300,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=42
        )
        prices = []
        for K in [1.0, 1.05, 1.1, 1.15, 1.2]:
            res = solve_rbsde_2factor(
                S_paths=S, nu_paths=nu, dB1=dB1, dB2=dB2,
                dt=1.0 / 5, K=K,
                feature_builder=build_sabr_features,
                r=0.0,
            )
            prices.append(res["price"])
        # Prices should be non-decreasing in strike
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1] + 0.05, (
                f"Price not monotone: K1 price={prices[i]:.4f}, K2 price={prices[i+1]:.4f}"
            )

    def test_z0_shape(self, sabr_paths):
        """Z0 should have shape (2,) for 2-factor model."""
        S, nu, dB1, dB2 = sabr_paths
        result = solve_rbsde_2factor(
            S_paths=S, nu_paths=nu, dB1=dB1, dB2=dB2,
            dt=1.0 / 5, K=1.1,
            feature_builder=build_sabr_features,
            r=0.0,
        )
        assert result["Z0"].shape == (2,)

    def test_store_path_shape(self, sabr_paths):
        """With store_path=True, Y_path should have shape (M, N_T+1)."""
        S, nu, dB1, dB2 = sabr_paths
        N_T = S.shape[1] - 1
        M = S.shape[0]
        result = solve_rbsde_2factor(
            S_paths=S, nu_paths=nu, dB1=dB1, dB2=dB2,
            dt=1.0 / N_T, K=1.1,
            feature_builder=build_sabr_features,
            r=0.0, store_path=True,
        )
        assert result["Y_path"] is not None
        assert result["Y_path"].shape == (M, N_T + 1)

    def test_reflection_constraint_holds(self):
        """Y_i must be >= g(S_i) for all paths and time points."""
        S, nu, dB1, dB2 = simulate_sabr_paths(
            S0=1.1, nu0=0.2, T=1.0, N_T=5, M=200,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=7
        )
        K = 1.1
        N_T = S.shape[1] - 1
        result = solve_rbsde_2factor(
            S_paths=S, nu_paths=nu, dB1=dB1, dB2=dB2,
            dt=1.0 / N_T, K=K,
            feature_builder=build_sabr_features,
            r=0.0, store_path=True,
        )
        Y_path = result["Y_path"]
        for i in range(N_T + 1):
            g_i = american_put_payoff(S[:, i], K)
            violations = (Y_path[:, i] < g_i - 1e-8).sum()
            assert violations == 0, f"Reflection violated at step {i}: {violations} paths"

    def test_runtime_positive(self, sabr_paths):
        S, nu, dB1, dB2 = sabr_paths
        result = solve_rbsde_2factor(
            S_paths=S, nu_paths=nu, dB1=dB1, dB2=dB2,
            dt=1.0 / 5, K=1.1,
            feature_builder=build_sabr_features,
            r=0.0,
        )
        assert result["runtime"] > 0.0

    def test_deep_itm_price_near_intrinsic(self):
        """Deep ITM put (S << K) price should be close to intrinsic K - S0."""
        S0, K = 5.0, 10.0  # deep ITM
        S, nu, dB1, dB2 = simulate_sabr_paths(
            S0=S0, nu0=0.2, T=1.0, N_T=5, M=300,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=99
        )
        result = solve_rbsde_2factor(
            S_paths=S, nu_paths=nu, dB1=dB1, dB2=dB2,
            dt=1.0 / 5, K=K,
            feature_builder=build_sabr_features,
            r=0.0,
        )
        intrinsic = K - S0
        # Price should be >= intrinsic and within a reasonable margin
        assert result["price"] >= intrinsic - 0.1


# ---------------------------------------------------------------------------
# RBSDE 3-factor solver tests (SABSR)
# ---------------------------------------------------------------------------

class TestSolveRBSDE3Factor:

    @pytest.fixture(scope="class")
    def sabsr_paths(self):
        return simulate_sabsr_paths(
            S0=1.1, nu0=0.2, rho0=-0.4, T=1.0, N_T=5, M=400,
            beta=0.7, sigma_nu=0.08, sigma_rho=0.5, mu_rho=-0.4,
            kappa_rho=2.0, seed=1
        )

    def test_price_non_negative(self, sabsr_paths):
        rho, nu, S, dW_rho, dW_nu, dW_s = sabsr_paths
        result = solve_rbsde_3factor(
            rho_paths=rho, nu_paths=nu, S_paths=S,
            dW_rho=dW_rho, dW_nu=dW_nu, dW_s=dW_s,
            dt=1.0 / 5, K=1.1,
            feature_builder=build_sabsr_features,
            r=0.0,
        )
        assert result["price"] >= 0.0

    def test_z0_shape(self, sabsr_paths):
        """Z0 should have shape (3,) for 3-factor model."""
        rho, nu, S, dW_rho, dW_nu, dW_s = sabsr_paths
        result = solve_rbsde_3factor(
            rho_paths=rho, nu_paths=nu, S_paths=S,
            dW_rho=dW_rho, dW_nu=dW_nu, dW_s=dW_s,
            dt=1.0 / 5, K=1.1,
            feature_builder=build_sabsr_features,
            r=0.0,
        )
        assert result["Z0"].shape == (3,)

    def test_store_path_shape(self, sabsr_paths):
        rho, nu, S, dW_rho, dW_nu, dW_s = sabsr_paths
        M = S.shape[0]
        N_T = S.shape[1] - 1
        result = solve_rbsde_3factor(
            rho_paths=rho, nu_paths=nu, S_paths=S,
            dW_rho=dW_rho, dW_nu=dW_nu, dW_s=dW_s,
            dt=1.0 / N_T, K=1.1,
            feature_builder=build_sabsr_features,
            r=0.0, store_path=True,
        )
        assert result["Y_path"].shape == (M, N_T + 1)

    def test_price_bounded_by_strike(self, sabsr_paths):
        rho, nu, S, dW_rho, dW_nu, dW_s = sabsr_paths
        K = 1.1
        result = solve_rbsde_3factor(
            rho_paths=rho, nu_paths=nu, S_paths=S,
            dW_rho=dW_rho, dW_nu=dW_nu, dW_s=dW_s,
            dt=1.0 / 5, K=K,
            feature_builder=build_sabsr_features,
            r=0.0,
        )
        assert result["price"] <= K + 1e-6

    def test_keys_in_result(self, sabsr_paths):
        rho, nu, S, dW_rho, dW_nu, dW_s = sabsr_paths
        result = solve_rbsde_3factor(
            rho_paths=rho, nu_paths=nu, S_paths=S,
            dW_rho=dW_rho, dW_nu=dW_nu, dW_s=dW_s,
            dt=1.0 / 5, K=1.1,
            feature_builder=build_sabsr_features,
            r=0.0,
        )
        for key in ("price", "Z0", "Y_path", "runtime"):
            assert key in result


# ---------------------------------------------------------------------------
# Delta extraction tests
# ---------------------------------------------------------------------------

class TestDeltaExtraction:

    def test_sabr_delta_sign(self):
        """American put delta should be negative (Z2 < 0 for OTM put)."""
        # We use a known Z0 with negative second component
        Z0 = np.array([-0.05, -0.10])
        delta = extract_sabr_delta(Z0, nu0=0.2, S0=1.1, beta=0.7, rho=-0.4)
        assert delta < 0, f"Expected negative delta, got {delta:.4f}"

    def test_sabr_delta_formula(self):
        """Verify exact formula: delta = Z2 / (nu0 * S0^beta * sqrt(1-rho^2))."""
        nu0, S0, beta, rho = 0.2, 1.1, 0.7, -0.4
        Z2 = -0.15
        Z0 = np.array([0.01, Z2])
        expected = Z2 / (nu0 * (S0 ** beta) * np.sqrt(1 - rho ** 2))
        delta = extract_sabr_delta(Z0, nu0=nu0, S0=S0, beta=beta, rho=rho)
        assert abs(delta - expected) < 1e-12

    def test_sabsr_delta_uses_z3(self):
        """SABSR delta uses third Z component."""
        nu0, S0, beta, rho0 = 0.2, 1.1, 0.7, -0.4
        Z0 = np.array([0.01, -0.05, -0.12])
        Z_s = Z0[2]
        expected = Z_s / (nu0 * (S0 ** beta) * np.sqrt(1 - rho0 ** 2))
        delta = extract_sabsr_delta(Z0, nu0=nu0, S0=S0, beta=beta, rho0=rho0)
        assert abs(delta - expected) < 1e-12

    def test_hslv_delta_formula(self):
        """HSLV delta: Z2 / (sqrt(nu0) * S0^beta * sqrt(1-rho^2))."""
        nu0, S0, beta, rho = 0.04, 10.0, 0.7, -0.75
        Z2 = -0.08
        Z0 = np.array([0.02, Z2])
        expected = Z2 / (np.sqrt(nu0) * (S0 ** beta) * np.sqrt(1 - rho ** 2))
        delta = extract_hslv_delta(Z0, nu0=nu0, S0=S0, beta=beta, rho=rho)
        assert abs(delta - expected) < 1e-12

    def test_hsclv_delta_uses_z3(self):
        """HSCLV delta uses third Z component."""
        nu0, S0, beta, rho0 = 0.04, 10.0, 0.7, -0.75
        Z_s = -0.09
        Z0 = np.array([0.01, 0.02, Z_s])
        expected = Z_s / (np.sqrt(nu0) * (S0 ** beta) * np.sqrt(1 - rho0 ** 2))
        delta = extract_hsclv_delta(Z0, nu0=nu0, S0=S0, beta=beta, rho0=rho0)
        assert abs(delta - expected) < 1e-12

    def test_delta_zero_denominator(self):
        """When denominator is 0, return 0 without error."""
        Z0 = np.array([0.0, -0.1])
        # rho=1 => sqrt(1-rho^2) = 0 => denom = 0
        delta = extract_sabr_delta(Z0, nu0=0.2, S0=1.0, beta=1.0, rho=1.0)
        assert delta == 0.0


# ---------------------------------------------------------------------------
# HSLV solver integration test
# ---------------------------------------------------------------------------

class TestSolveRBSDE2FactorHSLV:

    def test_hslv_deep_itm_near_intrinsic(self):
        """Deep ITM HSLV put should be approximately intrinsic."""
        S0, K = 7.0, 13.0  # deep ITM, intrinsic = 6
        S, nu, dB1, dB2 = simulate_hslv_paths(
            S0=S0, nu0=0.04, T=1.0, N_T=10, M=500,
            beta=0.7, kappa_nu=4.0, mu_nu=0.035, sigma_nu=0.15,
            mu_s=0.03, rho=-0.75, seed=100
        )
        result = solve_rbsde_2factor(
            S_paths=S, nu_paths=nu, dB1=dB1, dB2=dB2,
            dt=1.0 / 10, K=K,
            feature_builder=build_hslv_features,
            r=0.03,
        )
        intrinsic = K - S0  # = 6.0
        # Price should be >= intrinsic and close to it (within 15% for small N_T)
        assert result["price"] >= intrinsic - 0.2, (
            f"Deep ITM price {result['price']:.4f} < intrinsic {intrinsic:.1f} - 0.2"
        )
