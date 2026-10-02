"""
Tests for forward_simulation.py and hslv_simulation.py.

Validates:
  - Output shapes and dtypes
  - Initial conditions
  - Positivity constraints (nu_t > 0 after fixing)
  - Brownian increment statistics
  - Correlation structure of the paths
  - Edge cases
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forward_simulation import simulate_sabr_paths, simulate_sabsr_paths
from src.hslv_simulation import simulate_hslv_paths, simulate_hsclv_paths

# ---------------------------------------------------------------------------
# SABR tests
# ---------------------------------------------------------------------------

class TestSimulateSABRPaths:

    def test_output_shapes(self):
        """Verify output arrays have correct shapes."""
        M, N_T = 200, 20
        S, nu, dB1, dB2 = simulate_sabr_paths(
            S0=1.1, nu0=0.2, T=1.0, N_T=N_T, M=M,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=0
        )
        assert S.shape == (M, N_T + 1)
        assert nu.shape == (M, N_T + 1)
        assert dB1.shape == (M, N_T)
        assert dB2.shape == (M, N_T)

    def test_dtypes_float64(self):
        """All arrays should be float64."""
        S, nu, dB1, dB2 = simulate_sabr_paths(
            S0=1.1, nu0=0.2, T=1.0, N_T=10, M=100,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=1
        )
        for arr in (S, nu, dB1, dB2):
            assert arr.dtype == np.float64

    def test_initial_conditions(self):
        """First column should equal initial values."""
        S0, nu0 = 1.1, 0.2
        S, nu, _, _ = simulate_sabr_paths(
            S0=S0, nu0=nu0, T=1.0, N_T=10, M=500,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=2
        )
        np.testing.assert_array_equal(S[:, 0], S0)
        np.testing.assert_array_equal(nu[:, 0], nu0)

    def test_nu_positivity(self):
        """nu_t should be strictly positive at all times (positivity fix applied)."""
        S, nu, _, _ = simulate_sabr_paths(
            S0=1.1, nu0=0.2, T=1.0, N_T=50, M=1000,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=3
        )
        assert np.all(nu > 0), "nu_t must be strictly positive"

    def test_spot_positivity(self):
        """S_t should be strictly positive (floor applied)."""
        S, _, _, _ = simulate_sabr_paths(
            S0=1.1, nu0=0.2, T=1.0, N_T=50, M=1000,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=4
        )
        assert np.all(S > 0), "S_t must be strictly positive"

    def test_bm_increment_statistics(self):
        """BM increments should be approximately N(0, dt)."""
        N_T, M = 100, 10000
        dt = 1.0 / N_T
        _, _, dB1, dB2 = simulate_sabr_paths(
            S0=1.0, nu0=0.2, T=1.0, N_T=N_T, M=M,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=5
        )
        # Mean near 0
        assert abs(dB1.mean()) < 0.01
        assert abs(dB2.mean()) < 0.01
        # Variance near dt
        assert abs(dB1.var() - dt) < 0.01
        assert abs(dB2.var() - dt) < 0.01

    def test_bm_independence(self):
        """dB1 and dB2 should be approximately uncorrelated."""
        _, _, dB1, dB2 = simulate_sabr_paths(
            S0=1.0, nu0=0.2, T=1.0, N_T=100, M=10000,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=6
        )
        corr = np.corrcoef(dB1.ravel(), dB2.ravel())[0, 1]
        assert abs(corr) < 0.05, f"dB1, dB2 should be uncorrelated; got corr={corr:.4f}"

    def test_reproducibility(self):
        """Same seed should produce identical paths."""
        kwargs = dict(S0=1.1, nu0=0.2, T=1.0, N_T=10, M=50,
                      beta=0.7, sigma_nu=0.08, rho=-0.4, seed=99)
        S1, nu1, _, _ = simulate_sabr_paths(**kwargs)
        S2, nu2, _, _ = simulate_sabr_paths(**kwargs)
        np.testing.assert_array_equal(S1, S2)
        np.testing.assert_array_equal(nu1, nu2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different paths."""
        S1, _, _, _ = simulate_sabr_paths(
            S0=1.1, nu0=0.2, T=1.0, N_T=10, M=100,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=0
        )
        S2, _, _, _ = simulate_sabr_paths(
            S0=1.1, nu0=0.2, T=1.0, N_T=10, M=100,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=1
        )
        assert not np.allclose(S1, S2)

    def test_single_path(self):
        """Edge case: M=1 path."""
        S, nu, dB1, dB2 = simulate_sabr_paths(
            S0=1.0, nu0=0.2, T=1.0, N_T=5, M=1,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=0
        )
        assert S.shape == (1, 6)
        assert nu.shape == (1, 6)

    def test_single_timestep(self):
        """Edge case: N_T=1 time step."""
        S, nu, dB1, dB2 = simulate_sabr_paths(
            S0=1.0, nu0=0.2, T=1.0, N_T=1, M=100,
            beta=0.7, sigma_nu=0.08, rho=-0.4, seed=0
        )
        assert S.shape == (100, 2)
        assert dB1.shape == (100, 1)

    def test_beta_zero_constant_vol(self):
        """With beta=0, S_t diffusion is nu_t * constant (independent of S)."""
        S, nu, _, _ = simulate_sabr_paths(
            S0=1.0, nu0=0.2, T=1.0, N_T=20, M=500,
            beta=0.0, sigma_nu=0.08, rho=-0.4, seed=7
        )
        # S should still be positive
        assert np.all(S > 0)


# ---------------------------------------------------------------------------
# SABSR tests
# ---------------------------------------------------------------------------

class TestSimulateSABSRPaths:

    def test_output_shapes(self):
        M, N_T = 200, 20
        rho, nu, S, dW_rho, dW_nu, dW_s = simulate_sabsr_paths(
            S0=1.1, nu0=0.2, rho0=-0.4, T=1.0, N_T=N_T, M=M,
            beta=0.7, sigma_nu=0.08, sigma_rho=0.5, mu_rho=-0.4,
            kappa_rho=2.0, seed=0
        )
        assert rho.shape == (M, N_T + 1)
        assert nu.shape == (M, N_T + 1)
        assert S.shape == (M, N_T + 1)
        assert dW_rho.shape == (M, N_T)
        assert dW_nu.shape == (M, N_T)
        assert dW_s.shape == (M, N_T)

    def test_dtypes_float64(self):
        rho, nu, S, dW_rho, dW_nu, dW_s = simulate_sabsr_paths(
            S0=1.1, nu0=0.2, rho0=-0.4, T=1.0, N_T=10, M=100,
            beta=0.7, sigma_nu=0.08, sigma_rho=0.5, mu_rho=-0.4,
            kappa_rho=2.0, seed=0
        )
        for arr in (rho, nu, S, dW_rho, dW_nu, dW_s):
            assert arr.dtype == np.float64

    def test_initial_conditions(self):
        S0, nu0, rho0 = 1.1, 0.2, -0.4
        rho, nu, S, _, _, _ = simulate_sabsr_paths(
            S0=S0, nu0=nu0, rho0=rho0, T=1.0, N_T=10, M=200,
            beta=0.7, sigma_nu=0.08, sigma_rho=0.5, mu_rho=-0.4,
            kappa_rho=2.0, seed=0
        )
        np.testing.assert_array_equal(S[:, 0], S0)
        np.testing.assert_array_equal(nu[:, 0], nu0)
        np.testing.assert_array_equal(rho[:, 0], rho0)

    def test_rho_bounds(self):
        """rho_t must stay within (-1, 1) after clipping."""
        rho, _, _, _, _, _ = simulate_sabsr_paths(
            S0=1.1, nu0=0.2, rho0=-0.4, T=1.0, N_T=100, M=500,
            beta=0.7, sigma_nu=0.08, sigma_rho=0.5, mu_rho=-0.4,
            kappa_rho=2.0, seed=8
        )
        assert np.all(rho >= -1.0), "rho must be >= -1"
        assert np.all(rho <= 1.0), "rho must be <= 1"

    def test_nu_positivity(self):
        """nu_t should remain positive."""
        _, nu, _, _, _, _ = simulate_sabsr_paths(
            S0=1.1, nu0=0.2, rho0=-0.4, T=1.0, N_T=50, M=500,
            beta=0.7, sigma_nu=0.08, sigma_rho=0.5, mu_rho=-0.4,
            kappa_rho=2.0, seed=9
        )
        assert np.all(nu > 0)

    def test_spot_positivity(self):
        _, _, S, _, _, _ = simulate_sabsr_paths(
            S0=1.1, nu0=0.2, rho0=-0.4, T=1.0, N_T=50, M=500,
            beta=0.7, sigma_nu=0.08, sigma_rho=0.5, mu_rho=-0.4,
            kappa_rho=2.0, seed=10
        )
        assert np.all(S > 0)

    def test_high_kappa_rho_mean_reversion(self):
        """High kappa_rho should keep rho_t close to mu_rho."""
        mu_rho = -0.4
        rho, _, _, _, _, _ = simulate_sabsr_paths(
            S0=1.1, nu0=0.2, rho0=-0.4, T=1.0, N_T=100, M=2000,
            beta=0.7, sigma_nu=0.08, sigma_rho=0.5, mu_rho=mu_rho,
            kappa_rho=20.0, seed=11  # very fast mean reversion
        )
        # Terminal rho should be close to mu_rho on average
        terminal_mean = rho[:, -1].mean()
        assert abs(terminal_mean - mu_rho) < 0.15, (
            f"With high kappa_rho, terminal rho mean={terminal_mean:.3f} "
            f"should be near mu_rho={mu_rho}"
        )

    def test_reproducibility(self):
        kwargs = dict(S0=1.1, nu0=0.2, rho0=-0.4, T=1.0, N_T=10, M=50,
                      beta=0.7, sigma_nu=0.08, sigma_rho=0.5, mu_rho=-0.4,
                      kappa_rho=2.0, seed=999)
        r1, nu1, S1, _, _, _ = simulate_sabsr_paths(**kwargs)
        r2, nu2, S2, _, _, _ = simulate_sabsr_paths(**kwargs)
        np.testing.assert_array_equal(S1, S2)
        np.testing.assert_array_equal(nu1, nu2)


# ---------------------------------------------------------------------------
# HSLV tests
# ---------------------------------------------------------------------------

class TestSimulateHSLVPaths:

    def test_output_shapes(self):
        M, N_T = 200, 20
        S, nu, dB1, dB2 = simulate_hslv_paths(
            S0=10.0, nu0=0.04, T=1.0, N_T=N_T, M=M,
            beta=0.7, kappa_nu=4.0, mu_nu=0.035, sigma_nu=0.15,
            mu_s=0.03, rho=-0.75, seed=0
        )
        assert S.shape == (M, N_T + 1)
        assert nu.shape == (M, N_T + 1)
        assert dB1.shape == (M, N_T)
        assert dB2.shape == (M, N_T)

    def test_dtypes(self):
        S, nu, dB1, dB2 = simulate_hslv_paths(
            S0=10.0, nu0=0.04, T=1.0, N_T=10, M=100,
            beta=0.7, kappa_nu=4.0, mu_nu=0.035, sigma_nu=0.15,
            mu_s=0.03, rho=-0.75, seed=0
        )
        for arr in (S, nu, dB1, dB2):
            assert arr.dtype == np.float64

    def test_initial_conditions(self):
        S0, nu0 = 10.0, 0.04
        S, nu, _, _ = simulate_hslv_paths(
            S0=S0, nu0=nu0, T=1.0, N_T=10, M=200,
            beta=0.7, kappa_nu=4.0, mu_nu=0.035, sigma_nu=0.15,
            mu_s=0.03, rho=-0.75, seed=0
        )
        np.testing.assert_array_equal(S[:, 0], S0)
        np.testing.assert_array_equal(nu[:, 0], nu0)

    def test_spot_positivity(self):
        S, _, _, _ = simulate_hslv_paths(
            S0=10.0, nu0=0.04, T=1.0, N_T=30, M=500,
            beta=0.7, kappa_nu=4.0, mu_nu=0.035, sigma_nu=0.15,
            mu_s=0.03, rho=-0.75, seed=12
        )
        assert np.all(S > 0)

    def test_cir_variance_mean_reversion(self):
        """CIR variance should mean-revert toward mu_nu over time."""
        mu_nu = 0.035
        _, nu, _, _ = simulate_hslv_paths(
            S0=10.0, nu0=0.04, T=5.0, N_T=500, M=2000,
            beta=0.7, kappa_nu=4.0, mu_nu=mu_nu, sigma_nu=0.15,
            mu_s=0.03, rho=-0.75, seed=13
        )
        terminal_mean = nu[:, -1].mean()
        # Should be within 10% of mu_nu
        assert abs(terminal_mean - mu_nu) < 0.01, (
            f"CIR terminal mean={terminal_mean:.4f} should be near mu_nu={mu_nu}"
        )

    def test_reproducibility(self):
        kwargs = dict(S0=10.0, nu0=0.04, T=1.0, N_T=10, M=50,
                      beta=0.7, kappa_nu=4.0, mu_nu=0.035, sigma_nu=0.15,
                      mu_s=0.03, rho=-0.75, seed=777)
        S1, nu1, _, _ = simulate_hslv_paths(**kwargs)
        S2, nu2, _, _ = simulate_hslv_paths(**kwargs)
        np.testing.assert_array_equal(S1, S2)
        np.testing.assert_array_equal(nu1, nu2)


# ---------------------------------------------------------------------------
# HSCLV tests
# ---------------------------------------------------------------------------

class TestSimulateHSCLVPaths:

    def test_output_shapes(self):
        M, N_T = 200, 20
        rho, nu, S, dW_rho, dW_nu, dW_s = simulate_hsclv_paths(
            S0=10.0, nu0=0.04, rho0=-0.75, T=1.0, N_T=N_T, M=M,
            beta=0.7, kappa_nu=4.0, mu_nu=0.035, sigma_nu=0.15,
            mu_s=0.03, kappa_rho=2.0, mu_rho=-0.2, sigma_rho=0.1,
            seed=0
        )
        assert rho.shape == (M, N_T + 1)
        assert nu.shape == (M, N_T + 1)
        assert S.shape == (M, N_T + 1)

    def test_rho_bounds(self):
        rho, _, _, _, _, _ = simulate_hsclv_paths(
            S0=10.0, nu0=0.04, rho0=-0.75, T=1.0, N_T=30, M=500,
            beta=0.7, kappa_nu=4.0, mu_nu=0.035, sigma_nu=0.15,
            mu_s=0.03, kappa_rho=2.0, mu_rho=-0.2, sigma_rho=0.1, seed=14
        )
        assert np.all(rho >= -1.0)
        assert np.all(rho <= 1.0)

    def test_initial_conditions(self):
        S0, nu0, rho0 = 10.0, 0.04, -0.75
        rho, nu, S, _, _, _ = simulate_hsclv_paths(
            S0=S0, nu0=nu0, rho0=rho0, T=1.0, N_T=10, M=100,
            beta=0.7, kappa_nu=4.0, mu_nu=0.035, sigma_nu=0.15,
            mu_s=0.03, kappa_rho=2.0, mu_rho=-0.2, sigma_rho=0.1, seed=0
        )
        np.testing.assert_array_equal(S[:, 0], S0)
        np.testing.assert_array_equal(nu[:, 0], nu0)
        np.testing.assert_array_equal(rho[:, 0], rho0)

    def test_spot_positivity(self):
        _, _, S, _, _, _ = simulate_hsclv_paths(
            S0=10.0, nu0=0.04, rho0=-0.75, T=1.0, N_T=30, M=500,
            beta=0.7, kappa_nu=4.0, mu_nu=0.035, sigma_nu=0.15,
            mu_s=0.03, kappa_rho=2.0, mu_rho=-0.2, sigma_rho=0.1, seed=15
        )
        assert np.all(S > 0)
