"""Overfitting-control statistics: PSR/DSR, CSCV PBO, CPCV path assembly.

References:
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
- Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest Overfitting"
- Lopez de Prado (2018), AFML Ch. 7 (purging/embargo) and Ch. 12 (CPCV)
"""

from __future__ import annotations

import math
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.validation.cv.cpcv import CombinatorialPurgedCV, CPCVConfig, CPCVPathResult, CPCVResult
from src.validation.cv.pbo import (
    PBOConfig,
    compute_pbo,
    directional_strategy_returns,
    pbo_gate,
)
from src.validation.deflated_sharpe import (
    EULER_MASCHERONI,
    DSRComputeConfig,
    compute_deflated_sharpe,
    compute_deflated_sharpe_from_returns,
    compute_dsr_from_optuna_study,
    dsr_gate,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    return_moments,
    sharpe_ratio_per_period,
)

# =============================================================================
# PSR / DSR
# =============================================================================


class TestPSR:
    def test_normal_returns_hand_computed(self) -> None:
        # SR=0.1, SR*=0, T=101, g3=0, g4=3:
        # z = 0.1 * sqrt(100) / sqrt(1 + (2/4) * 0.01) = 1 / sqrt(1.005)
        expected = stats.norm.cdf(1.0 / math.sqrt(1.005))
        assert probabilistic_sharpe_ratio(0.1, 0.0, 101) == pytest.approx(expected, abs=1e-12)

    def test_non_normal_hand_computed(self) -> None:
        # SR=0.08, SR*=0.02, T=501, g3=-1, g4=6 (non-excess)
        # denom = 1 - (-1)(0.08) + (5/4)(0.0064) = 1.088
        z = (0.08 - 0.02) * math.sqrt(500) / math.sqrt(1.088)
        got = probabilistic_sharpe_ratio(0.08, 0.02, 501, skewness=-1.0, kurtosis=6.0)
        assert got == pytest.approx(stats.norm.cdf(z), abs=1e-12)

    def test_uses_non_excess_kurtosis(self) -> None:
        # kurtosis=3 (normal) must equal the default; passing excess (0) must differ
        base = probabilistic_sharpe_ratio(0.2, 0.0, 50)
        assert probabilistic_sharpe_ratio(0.2, 0.0, 50, kurtosis=3.0) == base
        assert probabilistic_sharpe_ratio(0.2, 0.0, 50, kurtosis=0.0) != base

    def test_at_benchmark_is_half(self) -> None:
        assert probabilistic_sharpe_ratio(0.05, 0.05, 1000, -2.0, 9.0) == pytest.approx(0.5)

    def test_more_observations_more_confidence(self) -> None:
        low = probabilistic_sharpe_ratio(0.05, 0.0, 100)
        high = probabilistic_sharpe_ratio(0.05, 0.0, 2000)
        assert high > low

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            probabilistic_sharpe_ratio(0.1, 0.0, 1)
        with pytest.raises(ValueError):
            probabilistic_sharpe_ratio(float("nan"), 0.0, 100)
        with pytest.raises(ValueError):
            # impossible moments: kurtosis < 1 + skew^2 can make the variance term negative
            probabilistic_sharpe_ratio(2.0, 0.0, 100, skewness=5.0, kurtosis=1.0)


class TestExpectedMaxSharpe:
    def test_formula(self) -> None:
        n, v = 37, 0.004
        z1 = stats.norm.ppf(1 - 1 / n)
        z2 = stats.norm.ppf(1 - 1 / (n * math.e))
        expected = math.sqrt(v) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
        assert expected_max_sharpe(n, v) == pytest.approx(expected, rel=1e-12)

    def test_single_trial_or_no_dispersion(self) -> None:
        assert expected_max_sharpe(1, 0.01) == 0.0
        assert expected_max_sharpe(10, 0.0) == 0.0

    def test_increases_with_trials(self) -> None:
        assert expected_max_sharpe(1000, 0.01) > expected_max_sharpe(10, 0.01)


class TestDSR:
    def test_bailey_lopez_de_prado_numerical_example(self) -> None:
        """Paper example: N=100, V[SR]=0.5 (annual), SR=2.5 (annual), T=1250, g3=-3, g4=10.

        The paper reports SR0 = 0.1132 (per period) and DSR = 0.9004.
        """
        periods = 250
        sr0 = expected_max_sharpe(100, 0.5 / periods)
        assert sr0 == pytest.approx(0.1132, abs=1e-4)

        # trial_sharpes are only used for their variance: two values with
        # sample variance exactly 0.5 / 250
        half_spread = math.sqrt(0.5 / periods / 2.0)
        trials = np.array([-half_spread, half_spread])
        result = compute_deflated_sharpe(
            sharpe_ratio=2.5 / math.sqrt(periods),
            trial_sharpes=trials,
            n_observations=1250,
            n_trials=100,
            skewness=-3.0,
            kurtosis=10.0,
        )
        assert result.variance_trial_sharpes == pytest.approx(0.5 / periods)
        assert result.expected_max_sharpe == pytest.approx(sr0)
        assert result.dsr == pytest.approx(0.9004, abs=5e-4)
        assert not result.should_deploy  # 0.90 < 0.95 gate

    def test_dsr_is_psr_at_sr0(self) -> None:
        trials = np.array([0.01, 0.03, -0.02, 0.05, 0.0])
        result = compute_deflated_sharpe(0.05, trials, n_observations=800, skewness=0.3)
        expected = probabilistic_sharpe_ratio(
            0.05, expected_max_sharpe(5, float(np.var(trials, ddof=1))), 800, 0.3, 3.0
        )
        assert result.dsr == pytest.approx(expected)
        assert result.psr == pytest.approx(probabilistic_sharpe_ratio(0.05, 0.0, 800, 0.3, 3.0))
        assert result.dsr <= result.psr

    def test_from_returns_uses_return_moments(self) -> None:
        rng = np.random.default_rng(3)
        returns = rng.standard_t(df=5, size=600) * 0.01 + 0.001
        trials = rng.normal(0, 0.03, size=20)
        skew, kurt = return_moments(returns)
        result = compute_deflated_sharpe_from_returns(returns, trials)
        manual = compute_deflated_sharpe(
            sharpe_ratio_per_period(returns),
            trials,
            n_observations=600,
            skewness=skew,
            kurtosis=kurt,
        )
        assert result.dsr == pytest.approx(manual.dsr)
        assert result.kurtosis == pytest.approx(stats.kurtosis(returns, fisher=False, bias=False))
        assert result.n_observations == 600

    def test_more_trials_deflate_more(self) -> None:
        trials = np.linspace(-0.05, 0.05, 11)
        few = compute_deflated_sharpe(0.06, trials, n_observations=1000, n_trials=2)
        many = compute_deflated_sharpe(0.06, trials, n_observations=1000, n_trials=1000)
        assert many.dsr < few.dsr

    def test_non_finite_trials_ignored(self) -> None:
        trials = np.array([0.01, np.nan, -np.inf, 0.02, 0.03])
        result = compute_deflated_sharpe(0.03, trials, n_observations=500)
        assert result.n_trials == 3

    def test_noise_strategies_best_trial_not_deployable(self) -> None:
        """Monte-Carlo: best of N pure-noise strategies should rarely pass DSR >= 0.95."""
        rng = np.random.default_rng(1)
        dsrs = []
        for _ in range(25):
            matrix = rng.normal(0.0, 1.0, size=(1000, 50))
            sharpes = np.array([sharpe_ratio_per_period(matrix[:, j]) for j in range(50)])
            best = int(np.argmax(sharpes))
            dsrs.append(compute_deflated_sharpe_from_returns(matrix[:, best], sharpes).dsr)
        dsrs_arr = np.array(dsrs)
        assert dsrs_arr.max() < 0.95
        assert dsrs_arr.mean() < 0.7

    def test_strong_drift_strategy_dsr_near_one(self) -> None:
        rng = np.random.default_rng(2)
        matrix = rng.normal(0.0, 1.0, size=(1000, 50))
        matrix[:, 7] += 0.2
        sharpes = np.array([sharpe_ratio_per_period(matrix[:, j]) for j in range(50)])
        result = compute_deflated_sharpe_from_returns(matrix[:, 7], sharpes)
        assert result.dsr > 0.99
        assert result.should_deploy
        assert dsr_gate(result)[0]

    def test_gate_threshold_configurable(self) -> None:
        trials = np.array([-0.01, 0.0, 0.01])
        result = compute_deflated_sharpe(
            0.04,
            trials,
            n_observations=900,
            config=DSRComputeConfig(deployment_threshold=0.5, strict_threshold=0.9999),
        )
        assert result.should_deploy
        assert dsr_gate(result)[0]
        assert not dsr_gate(result, strict=True)[0]
        with pytest.raises(ValueError):
            DSRComputeConfig(deployment_threshold=1.5)

    def test_optuna_study_deannualizes_trial_values(self) -> None:
        optuna = pytest.importorskip("optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        annualized = [0.5, 1.0, -0.3, 2.0, float("-inf")]
        for v in annualized:
            study.add_trial(optuna.trial.create_trial(value=v, params={}, distributions={}))

        result = compute_dsr_from_optuna_study(study, n_observations=2000, periods_per_year=252)
        per_period = np.array(annualized[:4]) / math.sqrt(252)
        manual = compute_deflated_sharpe(per_period.max(), per_period, n_observations=2000)
        assert result.sharpe_ratio == pytest.approx(2.0 / math.sqrt(252))
        assert result.n_trials == 4  # -inf (failed) trial ignored
        assert result.dsr == pytest.approx(manual.dsr)

        with pytest.raises(ValueError):
            compute_dsr_from_optuna_study(study, n_observations=100, metric_name="f1_weighted")


# =============================================================================
# PBO (CSCV)
# =============================================================================


class TestPBO:
    def test_hand_computed_fully_overfit(self) -> None:
        """S=2, N=2: each strategy wins exactly the block the other loses."""
        block_a = np.array([[0.02, -0.01], [0.03, -0.02], [0.01, -0.03]])
        block_b = np.array([[-0.02, 0.01], [-0.01, 0.03], [-0.03, 0.02]])
        matrix = np.vstack([block_a, block_b])
        result = compute_pbo(matrix, PBOConfig(n_partitions=2))
        # Both combinations: IS winner ranks last OOS -> w = 1/(N+1) = 1/3
        assert result.n_combinations == 2
        np.testing.assert_allclose(result.logit_distribution, [math.log(0.5)] * 2)
        assert result.pbo == 1.0
        assert result.should_block

    def test_logit_count_is_c_s_half(self) -> None:
        rng = np.random.default_rng(0)
        result = compute_pbo(rng.normal(0, 0.01, (400, 5)), PBOConfig(n_partitions=8))
        assert result.n_combinations == comb(8, 4)
        assert len(result.logit_distribution) == comb(8, 4)
        assert np.all(np.isfinite(result.logit_distribution))

    def test_noise_strategies_pbo_around_half(self) -> None:
        rng = np.random.default_rng(42)
        pbos = [
            compute_pbo(rng.normal(0, 0.01, (800, 20)), PBOConfig(n_partitions=8)).pbo
            for _ in range(20)
        ]
        assert 0.35 <= float(np.mean(pbos)) <= 0.65

    def test_dominant_strategy_pbo_zero(self) -> None:
        rng = np.random.default_rng(7)
        matrix = rng.normal(0, 0.01, (1600, 10))
        matrix[:, 4] += 0.01  # dominates in every block
        result = compute_pbo(matrix, PBOConfig(n_partitions=16))
        assert result.pbo == 0.0
        assert result.best_is_strategy_idx == 4
        assert result.prob_oos_loss == 0.0
        assert pbo_gate(result)[0]

    def test_truncates_oldest_rows_to_multiple_of_s(self) -> None:
        rng = np.random.default_rng(1)
        result = compute_pbo(rng.normal(0, 1, (103, 3)), PBOConfig(n_partitions=4))
        assert result.n_observations == 100

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            compute_pbo(np.zeros((100, 1)))
        with pytest.raises(ValueError):
            bad = np.zeros((100, 3))
            bad[5, 1] = np.nan
            compute_pbo(bad, PBOConfig(n_partitions=4))
        with pytest.raises(ValueError):
            compute_pbo(np.zeros((10, 3)), PBOConfig(n_partitions=8))
        with pytest.raises(ValueError):
            PBOConfig(n_partitions=7)


class TestDirectionalStrategyReturns:
    def test_hand_computed_with_costs_and_group_reset(self) -> None:
        preds = np.array([1, 1, -1, 0, -1, -1])
        fwd = np.array([0.01, -0.02, 0.03, 0.05, 0.02, np.nan])
        groups = np.array(["A", "A", "A", "A", "B", "B"])
        got = directional_strategy_returns(preds, fwd, cost_per_turnover=0.001, groups=groups)
        # turnover: |1-0|, 0, |-1-1|, |0+1|, |-1-0| (new group), 0
        expected = np.array([0.01 - 0.001, -0.02, -0.03 - 0.002, 0.0 - 0.001, -0.02 - 0.001, 0.0])
        np.testing.assert_allclose(got, expected)


# =============================================================================
# CPCV
# =============================================================================


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({"f": np.arange(n, dtype=float)})


class TestCPCVPaths:
    @pytest.mark.parametrize(("n_groups", "k"), [(6, 2), (5, 3), (8, 2), (4, 1), (10, 4)])
    def test_number_of_paths_phi(self, n_groups: int, k: int) -> None:
        cpcv = CombinatorialPurgedCV(CPCVConfig(n_groups=n_groups, n_test_groups=k, purge_bars=0))
        phi = k * comb(n_groups, k) // n_groups
        assert cpcv.n_paths == phi == comb(n_groups - 1, k - 1)
        assert cpcv.get_n_splits() == comb(n_groups, k)
        assert sum(1 for _ in cpcv.split(_frame(10 * n_groups))) == comb(n_groups, k)

    @pytest.mark.parametrize(("n_groups", "k"), [(6, 2), (5, 3), (7, 3)])
    def test_assignment_uses_every_split_group_cell_once(self, n_groups: int, k: int) -> None:
        cpcv = CombinatorialPurgedCV(CPCVConfig(n_groups=n_groups, n_test_groups=k, purge_bars=0))
        assignments = cpcv.get_path_assignments()
        assert assignments.shape == (cpcv.n_paths, n_groups)
        combos = list(combinations(range(n_groups), k))
        used = set()
        for p in range(cpcv.n_paths):
            for g in range(n_groups):
                split_id = int(assignments[p, g])
                assert g in combos[split_id]  # the split really tested group g
                used.add((split_id, g))
        # every (split, test group) cell appears in exactly one path
        assert used == {(s, g) for s, c in enumerate(combos) for g in c}
        assert len(used) == cpcv.n_paths * n_groups

    def test_each_path_covers_every_group_exactly_once(self) -> None:
        n = 97  # uneven group sizes
        cpcv = CombinatorialPurgedCV(CPCVConfig(n_groups=6, n_test_groups=2, purge_bars=3))
        split_values = {}
        for _train, test_idx, split_id in cpcv.split(_frame(n)):
            split_values[split_id] = test_idx.astype(float)
        paths = cpcv.assemble_paths(split_values, n)
        assert paths.shape == (cpcv.n_paths, n)
        for p in range(cpcv.n_paths):
            np.testing.assert_array_equal(paths[p], np.arange(n))

    def test_paths_mix_predictions_from_different_splits(self) -> None:
        n = 60
        cpcv = CombinatorialPurgedCV(CPCVConfig(n_groups=6, n_test_groups=2, purge_bars=0))
        split_values = {}
        for _train, test_idx, split_id in cpcv.split(_frame(n)):
            split_values[split_id] = np.full(len(test_idx), float(split_id))
        paths = cpcv.assemble_paths(split_values, n)
        assignments = cpcv.get_path_assignments()
        bounds = cpcv.group_boundaries(n)
        for p in range(cpcv.n_paths):
            for g, (s, e) in enumerate(bounds):
                assert np.all(paths[p, s:e] == assignments[p, g])

    def test_oos_matrix_has_one_column_per_path(self) -> None:
        results = [
            CPCVPathResult(path_id=p, split_ids=(0,), n_samples=50, returns=np.zeros(50))
            for p in range(5)
        ]
        matrix = CPCVResult(config=CPCVConfig(), path_results=results).get_oos_matrix()
        assert matrix.shape == (50, 5)


class TestCPCVPurge:
    def test_purge_and_embargo_bars_removed_around_test_groups(self) -> None:
        n, purge, embargo = 120, 4, 3
        cpcv = CombinatorialPurgedCV(
            CPCVConfig(n_groups=6, n_test_groups=2, purge_bars=purge, embargo_bars=embargo)
        )
        bounds = cpcv.group_boundaries(n)
        for train_idx, test_idx, split_id in cpcv.split(_frame(n)):
            expected = np.ones(n, dtype=bool)
            for g in cpcv.test_combinations[split_id]:
                s, e = bounds[g]
                expected[max(0, s - purge) : min(n, e + purge + embargo)] = False
            np.testing.assert_array_equal(train_idx, np.flatnonzero(expected))
            assert np.intersect1d(train_idx, test_idx).size == 0
            # no training row within purge_bars before / purge+embargo after any test group
            for g in cpcv.test_combinations[split_id]:
                s, e = bounds[g]
                assert not np.any((train_idx >= s - purge) & (train_idx < e + purge + embargo))

    def test_zero_purge_keeps_all_non_test_rows(self) -> None:
        n = 60
        cpcv = CombinatorialPurgedCV(CPCVConfig(n_groups=6, n_test_groups=2, purge_bars=0))
        for train_idx, test_idx, _ in cpcv.split(_frame(n)):
            assert len(train_idx) + len(test_idx) == n

    def test_label_end_times_purge_long_labels(self) -> None:
        n = 60
        index = pd.date_range("2024-01-01", periods=n, freq="5min")
        X = pd.DataFrame({"f": np.arange(n, dtype=float)}, index=index)
        label_end = pd.Series(index + pd.Timedelta(minutes=5), index=index)
        # row 20 (group 2) has a label that resolves inside the last group (rows 50-59)
        label_end.iloc[20] = index[55]
        cpcv = CombinatorialPurgedCV(CPCVConfig(n_groups=6, n_test_groups=2, purge_bars=1))
        split_35 = cpcv.test_combinations.index((3, 5))

        bar_only = {s: tr for tr, _te, s in cpcv.split(X)}
        label_aware = {s: tr for tr, _te, s in cpcv.split(X, label_end_times=label_end)}
        # purge_bars alone keeps row 20; its real label overlaps test group 5
        assert 20 in bar_only[split_35]
        assert 20 not in label_aware[split_35]
        for split_id, test_groups in enumerate(cpcv.test_combinations):
            if 5 in test_groups:
                assert 20 not in label_aware[split_id]
            # label-aware purging only ever removes more rows
            assert set(label_aware[split_id]) <= set(bar_only[split_id])

    def test_split_raises_when_purge_leaves_no_training(self) -> None:
        cpcv = CombinatorialPurgedCV(CPCVConfig(n_groups=3, n_test_groups=2, purge_bars=50))
        with pytest.raises(ValueError):
            list(cpcv.split(_frame(30)))


# =============================================================================
# Evaluator / CLI helpers
# =============================================================================


class _SignModel:
    """Predicts sign(feature * direction); fit is a no-op."""

    def __init__(self, direction: float) -> None:
        self.direction = direction

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        return None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.sign(X["signal"].to_numpy() * self.direction)


def test_cpcv_pbo_evaluator_ranks_real_signal(tmp_path) -> None:
    from src.validation.evaluation.cpcv_pbo_evaluator import CPCVPBOEvaluator

    rng = np.random.default_rng(5)
    n = 640
    signal = rng.normal(size=n)
    fwd = 0.002 * np.sign(signal) + rng.normal(0, 0.001, size=n)
    X = pd.DataFrame({"signal": signal})
    y = pd.Series(np.sign(fwd))

    evaluator = CPCVPBOEvaluator(
        {"purge_bars": 2, "pbo_partitions": 8, "output_dir": str(tmp_path)}
    )
    out = evaluator.run(
        X,
        y,
        {"good": _SignModel(1.0), "bad": _SignModel(-1.0), "flat": _SignModel(0.0)},
        forward_returns=fwd,
    )
    assert out["n_paths"] == 5
    assert out["pbo_result"]["pbo"] == 0.0
    assert (
        out["cpcv_results"]["good"]["mean_sharpe"] > 0 > out["cpcv_results"]["bad"]["mean_sharpe"]
    )
    assert len(out["cpcv_results"]["good"]["paths"]) == 5


def test_cli_forward_returns_and_costs() -> None:
    from src.cli.commands.evaluate import _forward_returns_and_costs
    from src.config.symbol import SymbolConfig
    from src.data.pipeline.config.barriers_config import get_total_trade_cost

    df = pd.DataFrame(
        {
            "symbol": ["MES", "MES", "MES", "MGC", "MGC"],
            "close": [100.0, 101.0, 99.99, 2000.0, 2010.0],
        }
    )
    fwd, cost, groups = _forward_returns_and_costs(df, "symbol", include_costs=True)
    np.testing.assert_allclose(fwd[:2], [0.01, 99.99 / 101.0 - 1.0])
    assert np.isnan(fwd[2])  # no next bar within MES
    assert fwd[3] == pytest.approx(0.005)
    assert np.isnan(fwd[4])

    mes_side = get_total_trade_cost("MES") / 2 * SymbolConfig.from_symbol("MES").tick_size
    mgc_side = get_total_trade_cost("MGC") / 2 * SymbolConfig.from_symbol("MGC").tick_size
    np.testing.assert_allclose(cost[:3], mes_side / df["close"].to_numpy()[:3])
    np.testing.assert_allclose(cost[3:], mgc_side / df["close"].to_numpy()[3:])
    assert groups is not None and list(groups) == list(df["symbol"])

    _, no_cost, _ = _forward_returns_and_costs(df, "symbol", include_costs=False)
    assert np.all(no_cost == 0.0)
