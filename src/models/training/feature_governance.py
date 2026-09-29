"""Opt-in feature-governance diagnostics for a training run.

Runs AFTER feature selection and only reads its outcome: it can never change
which features are selected. Given the train-split frame and the live MDA
ranking it answers, per candidate feature:

- Is it STABLE? How often the REAL selection (ranking, budget, filters,
  decorrelation, per-model cut -- replayed by ``select_fn``) keeps it across
  random contiguous blocks of the train rows (stability selection with block
  subsamples, ``BootstrapFeatureStability``), per model and overall.
- Is it ROBUST to the label definition? Its rank when the triple-barrier widths
  are scaled, measured against a x1.0 control relabel (``LabelPerturbationTester``).
- Where is it in its lifecycle across runs? A ``FeatureRegistry`` persisted next
  to the run directories is updated from this run's selection and verdicts.

Every ranking is the selection's own purged-CV out-of-sample MDA
(``importance_fn``), so label-span purging and embargo apply inside each block
and each label variant, and only train-split rows are ever touched.

The result is a JSON report (``<output_dir>/feature_governance/h{h}.json``) whose
``flags`` section lists the selected features a human should look at.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.label_spans import INVALID_LABEL, NO_LABEL_END, frame_label_ends, label_end_positions
from src.optimization.feature_selection.bootstrap_stability import (
    BootstrapFeatureStability,
    StabilitySummary,
)
from src.optimization.feature_selection.label_perturbation import (
    LabelPerturbationTester,
    PerturbationSummary,
)
from src.optimization.feature_selection.registry import (
    FeatureRegistry,
    RunUpdate,
    context_fingerprint,
)
from src.optimization.feature_selection.robustness_scoring import RobustnessScorer

if TYPE_CHECKING:
    from src.core import PipelineConfig

logger = logging.getLogger(__name__)

GOVERNANCE_DIR = "feature_governance"

# importance_fn(frame, features, labels, label_ends) -> importance per feature (or None)
ImportanceFn = Callable[
    [pd.DataFrame, list[str], np.ndarray, "np.ndarray | None"], "pd.Series | None"
]


# select_fn(block_frame, candidates, importance) -> features each model would keep
SelectFn = Callable[[pd.DataFrame, list[str], pd.Series], dict[str, list[str]]]


def _scale_tag(scale: float) -> str:
    return f"barriers_x{scale:g}"


def experiment_context(config: PipelineConfig, label_col: str) -> dict[str, Any]:
    """The setup a registry's lifecycles are only comparable within.

    A different symbol, bar timeframe, ranking horizon, MTF feature set or model
    set changes which features exist and get selected; that is a new experiment,
    not evidence that a feature decayed.
    """
    return {
        "symbol": config.symbol.upper(),
        "bar_timeframe": config.bar_timeframe,
        "ranking_label": label_col,
        "mtf_timeframes": sorted(config.mtf_timeframes) if config.compute_mtf_features else [],
        "models": sorted(config.models),
    }


def default_registry_path(config: PipelineConfig, context: dict[str, Any]) -> Path:
    """Registry of one experiment context, next to the run directories."""
    name = f"feature_registry_{config.symbol.upper()}_{context_fingerprint(context)}.json"
    return Path(config.output_dir).parent / name


class FeatureGovernance:
    """Compute and persist the governance report for one training run.

    Args:
        config: The run's PipelineConfig (``governance`` holds the switches,
            ``label_barriers`` the resolved barrier params per horizon).
        importance_fn: Purged-CV out-of-sample MDA importance for a frame.
        select_fn: Replays the live post-ranking selection on a frame.
    """

    def __init__(
        self, config: PipelineConfig, importance_fn: ImportanceFn, select_fn: SelectFn
    ) -> None:
        from src.config.data import FeatureGovernanceConfig

        self.config = config
        self.settings = FeatureGovernanceConfig.from_dict(dict(config.governance))
        self._importance_fn = importance_fn
        self._select_fn = select_fn

    @property
    def enabled(self) -> bool:
        return self.settings.report

    def run(
        self,
        df_train: pd.DataFrame,
        *,
        label_col: str,
        candidates: list[str],
        raw_importance: pd.Series | None,
        selected_by_model: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Build the report (and update the registry); returns the report dict.

        Args:
            df_train: Train-split rows (a prefix of the labeled frame).
            label_col: The label the selection ranked on (``label_h{h}``).
            candidates: Every feature that entered the selection.
            raw_importance: The selection's MDA importance (None = it fell back
                to variance ranking, so there is no importance to compare).
            selected_by_model: Features each model ended up with.
        """
        cfg = self.settings
        labels = df_train[label_col].to_numpy()
        label_ends = frame_label_ends(df_train, label_col)
        selected = sorted({f for feats in selected_by_model.values() for f in feats})

        stability, stability_status = self._stability(
            df_train, candidates, labels, label_ends, raw_importance
        )
        perturbation, perturbation_status = self._label_perturbation(
            df_train, label_col, candidates, raw_importance
        )

        stable_by_feature = {r.feature_name: r.is_stable for r in stability.results}
        robust_by_feature = (
            {r.feature_name: r.is_robust for r in perturbation.results} if perturbation else {}
        )
        freq_by_feature = {r.feature_name: r.selection_frequency for r in stability.results}
        model_freq = {r.feature_name: r.group_frequency for r in stability.results}
        model_stable = {r.feature_name: r.group_stable for r in stability.results}
        shift_by_feature = (
            {r.feature_name: r.max_rank_change for r in perturbation.results}
            if perturbation
            else {}
        )
        scorer = RobustnessScorer(stability_weight=0.5, predictive_weight=0.5, regime_weight=0.0)
        composite = scorer.score_features(
            feature_names=candidates,
            fold_counts=freq_by_feature or None,
            n_folds=1,
            mda_importance=raw_importance,
        ).set_index("feature")["composite_score"]

        verdicts: dict[str, bool] = {}
        for f in candidates:
            measured = [d[f] for d in (stable_by_feature, robust_by_feature) if f in d]
            if measured:
                verdicts[f] = all(measured)

        features = {
            f: {
                "mda_importance": (
                    float(raw_importance[f])
                    if raw_importance is not None and f in raw_importance.index
                    else None
                ),
                "selected": f in selected,
                "selection_frequency": freq_by_feature.get(f),
                "model_frequency": model_freq.get(f),
                "stable": stable_by_feature.get(f),
                "label_rank_shift": shift_by_feature.get(f),
                "label_robust": robust_by_feature.get(f),
                "composite_score": float(composite.get(f, 0.0)),
            }
            for f in candidates
        }
        report: dict[str, Any] = {
            "run_id": Path(self.config.output_dir).name,
            "symbol": self.config.symbol,
            "ranking_label": label_col,
            "ranking_method": "MDA" if raw_importance is not None else "variance",
            "train_rows": len(df_train),
            "n_candidates": len(candidates),
            "settings": cfg.to_dict(),
            "selection": {"models": selected_by_model, "union": selected},
            "stability": {
                "status": stability_status,
                "threshold": cfg.stability_threshold,
                "window_fraction": cfg.window_fraction,
                "n_blocks_drawn": stability.n_blocks_drawn,
                "n_blocks_used": stability.n_blocks_used,
            },
            "label_perturbation": {
                "status": perturbation_status,
                "variants": perturbation.variants_used if perturbation else [],
                "rank_correlation": perturbation.rank_correlation if perturbation else {},
                "control_rank_correlation": (
                    perturbation.control_rank_correlation if perturbation else None
                ),
            },
            "flags": {
                "selected_unstable": sorted(
                    f for f in selected if stable_by_feature.get(f) is False
                ),
                "selected_unstable_by_model": {
                    m: sorted(f for f in feats if model_stable.get(f, {}).get(m) is False)
                    for m, feats in selected_by_model.items()
                },
                "selected_label_fragile": sorted(
                    f for f in selected if robust_by_feature.get(f) is False
                ),
            },
            "features": features,
        }

        if cfg.registry and selected:
            report["registry"] = self._update_registry(
                label_col,
                candidates,
                raw_importance,
                freq_by_feature,
                composite,
                selected,
                verdicts,
            )
        else:
            report["registry"] = {"status": "disabled"}

        self._write(report, label_col)
        return report

    # ------------------------------------------------------------------ parts

    def _stability(
        self,
        df_train: pd.DataFrame,
        candidates: list[str],
        labels: np.ndarray,
        label_ends: np.ndarray | None,
        raw_importance: pd.Series | None,
    ) -> tuple[StabilitySummary, str]:
        cfg = self.settings
        empty = StabilitySummary([], 0, 0)
        if not cfg.bootstrap_stability:
            return empty, "disabled"
        if raw_importance is None:
            # The live selection ranked by variance; a replay would test another procedure
            return empty, "skipped: MDA ranking unavailable"
        tester = BootstrapFeatureStability(
            n_bootstrap=cfg.n_bootstrap,
            stability_threshold=cfg.stability_threshold,
            window_fraction=cfg.window_fraction,
            random_state=self.config.random_state,
        )

        def replay_block(start: int, stop: int) -> dict[str, list[str]] | None:
            block = df_train.iloc[start:stop]
            block_ends = None
            if label_ends is not None:
                # Same bars, block coordinates (ends past the block still purge its tail)
                block_ends = np.where(
                    label_ends[start:stop] >= 0, label_ends[start:stop] - start, NO_LABEL_END
                )
            importance = self._importance_fn(block, candidates, labels[start:stop], block_ends)
            if importance is None:
                return None
            # The live post-ranking path, on this block
            return self._select_fn(block, candidates, importance)

        summary = tester.evaluate(candidates, len(df_train), replay_block)
        return summary, "ok" if summary.results else "skipped: no block could be ranked"

    def _label_perturbation(
        self,
        df_train: pd.DataFrame,
        label_col: str,
        candidates: list[str],
        raw_importance: pd.Series | None,
    ) -> tuple[PerturbationSummary | None, str]:
        cfg = self.settings
        if not cfg.label_perturbation:
            return None, "disabled"
        if raw_importance is None:
            return None, "skipped: MDA ranking unavailable"
        horizon = label_col.removeprefix("label_h")
        barriers = self.config.label_barriers.get(horizon)
        if not barriers:
            return None, "skipped: barrier parameters for the ranking horizon unavailable"
        if not {"high", "low", "close"} <= set(df_train.columns):
            return None, "skipped: OHLC columns unavailable to relabel"

        def rank_at(scale: float) -> pd.Series | None:
            try:
                labels, ends = self._relabel(df_train, barriers, scale)
                return self._importance_fn(df_train, candidates, labels, ends)
            except Exception as exc:
                logger.warning(f"  Label perturbation x{scale:g} skipped: {exc}")
                return None

        # x1.0 control: relabeled exactly like the variants, so relabeling artifacts
        # (truncated tail, recalibrated costs) are not blamed on the perturbation
        control = rank_at(1.0)
        variants = {_scale_tag(scale): rank_at(scale) for scale in cfg.barrier_scales}
        summary = LabelPerturbationTester().evaluate(raw_importance, variants, control=control)
        if not summary.variants_used:
            return None, "skipped: no label variant could be ranked"
        return summary, "ok"

    def _relabel(
        self, df_train: pd.DataFrame, barriers: dict[str, float], scale: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Triple-barrier labels of the train rows with both barriers scaled."""
        from src.data.labeling import TripleBarrierConfig, TripleBarrierLabeler

        max_bars = int(barriers["max_bars"])
        labeler = TripleBarrierLabeler(
            TripleBarrierConfig(
                horizon=max_bars,
                upper_mult=barriers["k_up"] * scale,
                lower_mult=barriers["k_down"] * scale,
                atr_period=self.config.atr_period,
                atr_column=None,
                symbol=self.config.symbol.upper(),
                # The frame IS the train split: calibrate costs on all of it
                cost_calibration_fraction=1.0,
            )
        )
        result = labeler.compute_labels(df_train, horizon=max_bars)
        labels = np.asarray(result.labels)
        ends = label_end_positions(labels, result.metadata["bars_to_hit"])
        if self.config.n_classes == 2:
            # Same remap as the factory: a barrier hit either way = 1, time-out = 0
            labels = np.where(labels == INVALID_LABEL, INVALID_LABEL, (labels != 0).astype(int))
        return labels, ends

    def _update_registry(
        self,
        label_col: str,
        candidates: list[str],
        raw_importance: pd.Series | None,
        freq_by_feature: dict[str, float],
        composite: pd.Series,
        selected: list[str],
        verdicts: dict[str, bool],
    ) -> dict[str, Any]:
        cfg = self.settings
        context = experiment_context(self.config, label_col)
        path = (
            Path(cfg.registry_path)
            if cfg.registry_path
            else default_registry_path(self.config, context)
        )
        scores = {
            f: {
                "mda_score": (
                    float(raw_importance[f])
                    if raw_importance is not None and f in raw_importance.index
                    else 0.0
                ),
                "stability_score": float(freq_by_feature.get(f, 0.0)),
                "composite_score": float(composite.get(f, 0.0)),
            }
            for f in candidates
        }
        # Locked load-modify-save: concurrent runs cannot lose each other's updates
        with FeatureRegistry.transaction(path) as registry:
            update: RunUpdate = registry.record_run(
                Path(self.config.output_dir).name,
                scores=scores,
                selected=selected,
                stable=verdicts,
                max_degraded_runs=cfg.max_degraded_runs,
                context=context,
            )
            states = Counter(record.state for record in registry.all_features())
            n_features = len(registry)
        return {
            "status": "skipped" if update.skipped else "ok",
            "skipped_reason": update.skipped,
            "path": str(path),
            "context": context,
            "n_features": n_features,
            "n_new": update.n_new,
            "states": dict(states),
            "transitions": update.transitions,
            "retired_but_selected": update.retired_but_selected,
        }

    def _write(self, report: dict[str, Any], label_col: str) -> Path:
        out_dir = Path(self.config.output_dir) / GOVERNANCE_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{label_col.removeprefix('label_')}.json"
        path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        logger.info(f"  Feature governance report: {path}")
        return path


__all__ = [
    "GOVERNANCE_DIR",
    "FeatureGovernance",
    "default_registry_path",
    "experiment_context",
]
