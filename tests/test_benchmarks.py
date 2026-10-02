"""
Tests for benchmarks.py.

Validates:
  - Benchmark data shapes and consistency
  - CTMC price ranges (economic plausibility)
  - DPDB CI ordering (lower <= upper)
  - CI containment checks
  - Error handling for invalid keys
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks import (
    SABR_CTMC_STRIKES,
    SABR_CTMC_PRICES,
    HSLV_CTMC_STRIKES,
    HSLV_CTMC_PRICES,
    SABSR_DPDB_STRIKES,
    HSCLV_DPDB_STRIKES,
    get_sabr_ctmc,
    get_hslv_ctmc,
    get_sabsr_dpdb,
    get_hsclv_dpdb,
)


# ---------------------------------------------------------------------------
# SABR CTMC
# ---------------------------------------------------------------------------

class TestSABRCTMC:

    def test_arrays_same_length(self):
        assert len(SABR_CTMC_STRIKES) == len(SABR_CTMC_PRICES)

    def test_strikes_sorted_ascending(self):
        assert np.all(np.diff(SABR_CTMC_STRIKES) > 0)

    def test_prices_non_negative(self):
        assert np.all(SABR_CTMC_PRICES >= 0.0)

    def test_prices_monotone_in_strike(self):
        """American put prices should increase with strike."""
        assert np.all(np.diff(SABR_CTMC_PRICES) >= 0), \
            "SABR CTMC prices should be non-decreasing in strike"

    def test_price_range(self):
        """Expected range ~0.0424 to ~0.1461."""
        assert SABR_CTMC_PRICES.min() > 0.03
        assert SABR_CTMC_PRICES.max() < 0.20

    def test_get_sabr_ctmc_returns_dataframe(self):
        df = get_sabr_ctmc()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"strike", "ctmc_price"}

    def test_get_sabr_ctmc_shape(self):
        df = get_sabr_ctmc()
        assert len(df) == len(SABR_CTMC_STRIKES)

    def test_strikes_include_expected_values(self):
        """Strikes should include 1.00 and 1.20."""
        assert 1.00 in SABR_CTMC_STRIKES
        assert 1.20 in SABR_CTMC_STRIKES

    def test_prices_bounded_by_strike_k100(self):
        """At K=1.00, price must be <= 1.00."""
        idx = np.where(SABR_CTMC_STRIKES == 1.00)[0][0]
        assert SABR_CTMC_PRICES[idx] <= 1.00

    def test_eleven_strike_levels(self):
        """Should have exactly 11 strike levels (1.00 to 1.20 step 0.02)."""
        assert len(SABR_CTMC_STRIKES) == 11


# ---------------------------------------------------------------------------
# HSLV CTMC
# ---------------------------------------------------------------------------

class TestHSLVCTMC:

    def test_arrays_same_length(self):
        assert len(HSLV_CTMC_STRIKES) == len(HSLV_CTMC_PRICES)

    def test_strikes_sorted_ascending(self):
        assert np.all(np.diff(HSLV_CTMC_STRIKES) > 0)

    def test_prices_non_negative(self):
        assert np.all(HSLV_CTMC_PRICES >= 0.0)

    def test_prices_monotone_in_strike(self):
        assert np.all(np.diff(HSLV_CTMC_PRICES) >= 0)

    def test_deep_itm_near_intrinsic(self):
        """Deep ITM strikes (11,12,13 with S0=10) should be near intrinsic."""
        for K, intrinsic in [(11.0, 1.0), (12.0, 2.0), (13.0, 3.0)]:
            idx = np.where(HSLV_CTMC_STRIKES == K)[0][0]
            price = HSLV_CTMC_PRICES[idx]
            assert abs(price - intrinsic) < 0.01, (
                f"K={K}: CTMC price={price:.4f}, expected near {intrinsic:.1f}"
            )

    def test_seven_strike_levels(self):
        assert len(HSLV_CTMC_STRIKES) == 7

    def test_get_hslv_ctmc_dataframe(self):
        df = get_hslv_ctmc()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 7

    def test_otm_price_small(self):
        """OTM strike K=7 (S0=10) should have very small price."""
        idx = np.where(HSLV_CTMC_STRIKES == 7.0)[0][0]
        assert HSLV_CTMC_PRICES[idx] < 0.01


# ---------------------------------------------------------------------------
# SABSR DPDB
# ---------------------------------------------------------------------------

class TestSABSRDPDB:

    @pytest.mark.parametrize("kappa_rho", [0.1, 2.0, 20.0])
    def test_ci_ordering(self, kappa_rho):
        """Lower bound must be <= upper bound for all strikes."""
        df = get_sabsr_dpdb(kappa_rho)
        assert np.all(df["dpdb_lower"].values <= df["dpdb_upper"].values)

    @pytest.mark.parametrize("kappa_rho", [0.1, 2.0, 20.0])
    def test_center_is_midpoint(self, kappa_rho):
        df = get_sabsr_dpdb(kappa_rho)
        expected_center = (df["dpdb_lower"] + df["dpdb_upper"]) / 2
        np.testing.assert_allclose(df["dpdb_center"].values, expected_center.values, atol=1e-12)

    @pytest.mark.parametrize("kappa_rho", [0.1, 2.0, 20.0])
    def test_prices_non_negative(self, kappa_rho):
        df = get_sabsr_dpdb(kappa_rho)
        assert np.all(df["dpdb_lower"].values >= 0.0)

    def test_invalid_kappa_raises(self):
        with pytest.raises(ValueError):
            get_sabsr_dpdb(kappa_rho=5.0)

    @pytest.mark.parametrize("kappa_rho", [0.1, 2.0, 20.0])
    def test_correct_n_strikes(self, kappa_rho):
        df = get_sabsr_dpdb(kappa_rho)
        assert len(df) == 11

    def test_high_kappa_ci_closer_to_sabr_ctmc(self):
        """
        For kappa_rho=20 (fast mean reversion), DPDB center values should be
        closer to SABR CTMC than for kappa_rho=0.1.
        """
        df_low = get_sabsr_dpdb(0.1)
        df_high = get_sabsr_dpdb(20.0)
        ctmc = SABR_CTMC_PRICES
        err_low = np.abs(df_low["dpdb_center"].values - ctmc).mean()
        err_high = np.abs(df_high["dpdb_center"].values - ctmc).mean()
        assert err_high <= err_low, (
            f"kappa_rho=20 should be closer to CTMC: err_high={err_high:.5f}, err_low={err_low:.5f}"
        )

    @pytest.mark.parametrize("kappa_rho", [0.1, 2.0, 20.0])
    def test_ci_width_positive(self, kappa_rho):
        df = get_sabsr_dpdb(kappa_rho)
        widths = df["dpdb_upper"] - df["dpdb_lower"]
        assert np.all(widths.values > 0)


# ---------------------------------------------------------------------------
# HSCLV DPDB
# ---------------------------------------------------------------------------

class TestHSCLVDPDB:

    @pytest.mark.parametrize("mu_rho", [-0.2, 0.0, 0.75])
    def test_ci_ordering(self, mu_rho):
        df = get_hsclv_dpdb(mu_rho)
        assert np.all(df["dpdb_lower"].values <= df["dpdb_upper"].values)

    @pytest.mark.parametrize("mu_rho", [-0.2, 0.0, 0.75])
    def test_correct_n_strikes(self, mu_rho):
        df = get_hsclv_dpdb(mu_rho)
        assert len(df) == 7

    def test_invalid_mu_raises(self):
        with pytest.raises(ValueError):
            get_hsclv_dpdb(mu_rho=0.5)

    @pytest.mark.parametrize("mu_rho", [-0.2, 0.0, 0.75])
    def test_deep_itm_ci_near_intrinsic(self, mu_rho):
        """K=13 (S0=10) deep ITM: CI should contain values near 3."""
        df = get_hsclv_dpdb(mu_rho)
        idx = np.where(df["strike"].values == 13.0)[0][0]
        lower = df["dpdb_lower"].iloc[idx]
        upper = df["dpdb_upper"].iloc[idx]
        assert lower >= 2.9, f"mu_rho={mu_rho}: lower={lower:.4f} < 2.9"
        assert upper <= 3.01, f"mu_rho={mu_rho}: upper={upper:.4f} > 3.01"

    @pytest.mark.parametrize("mu_rho", [-0.2, 0.0, 0.75])
    def test_center_is_midpoint(self, mu_rho):
        df = get_hsclv_dpdb(mu_rho)
        expected = (df["dpdb_lower"] + df["dpdb_upper"]) / 2
        np.testing.assert_allclose(df["dpdb_center"].values, expected.values, atol=1e-12)

    def test_mu075_atm_lower_price(self):
        """mu_rho=0.75 should produce lower ATM price than mu_rho=-0.2 (expected)."""
        df_neg = get_hsclv_dpdb(-0.2)
        df_pos = get_hsclv_dpdb(0.75)
        idx = np.where(df_neg["strike"].values == 10.0)[0][0]
        center_neg = df_neg["dpdb_center"].iloc[idx]
        center_pos = df_pos["dpdb_center"].iloc[idx]
        # mu_rho=0.75 should give lower price at K=10 (positive corr → lower put price)
        assert center_pos <= center_neg + 0.005, (
            f"mu_rho=0.75 K=10 center={center_pos:.4f} not <= mu_rho=-0.2 center={center_neg:.4f}"
        )
