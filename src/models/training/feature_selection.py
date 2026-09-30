"""
Feature selection pipeline mixin for UnifiedTrainingOrchestrator.

Extracted from unified_orchestrator.py (M2 refactor).
Contains: MDA ranking, correlation dedup, low-variance filter, per-model selection,
pre-training validation (contract, leakage, lookahead).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.exceptions import PreTrainingValidationError
from src.core.label_spans import INVALID_LABEL, LabelSpans, frame_label_ends
from src.optimization.feature_selection.filtering import (
    filter_low_variance,
    select_decorrelated_by_rank,
)
from src.optimization.feature_selection.ranking import (
    IMPORTANCE_NOISE_FLOOR,
    rank_by_importance,
)
from src.optimization.feature_selection.timeframe_budget import apply_timeframe_budget
from src.validation.cv import PurgedKFold, PurgedKFoldConfig

if TYPE_CHECKING:
    from src.core import PipelineConfig

logger = logging.getLogger(__name__)

# MDA ranking is stable at this many rows; larger datasets are strided down to it.
MDA_MAX_ROWS = 50_000


def _temporal_stride_subsample(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, int]:
    """Subsample to at most ``max_rows`` by taking every ``stride``-th row in order.

    Preserves temporal order (monotonic index) so purge/embargo stay meaningful.

    Returns:
        (subsampled frame, stride); stride == 1 means the frame is unchanged.
    """
    n = len(df)
    if n <= max_rows:
        return df, 1
    stride = -(-n // max_rows)  # ceil: guarantees len(result) <= max_rows
    return df.iloc[::stride], stride


class FeatureSelectionMixin:
    """Mixin providing feature selection pipeline methods for the orchestrator."""

    config: PipelineConfig
    _per_model_features: dict[str, list[str]]
    _all_feature_names: list[str]

    def _compute_mda_ranking(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
    ) -> pd.Series | None:
        """Compute MDA (permutation importance) ranking for all features.

        Uses a lightweight RandomForest with 3-fold PurgedKFold to rank features
        by predictive power. Rows with an invalid label (-99) are dropped, and
        the folds purge every training row whose label span overlaps the test
        block (label-end column of the ranking label). Returns pd.Series sorted
        descending, or None on failure.

        ``df`` rows are positions of the labeled frame (the train-split prefix),
        the coordinate system of its label-end columns.

        The ranking label is the first configured horizon's; the resulting
        per-model feature set is shared by every horizon of that model (one
        selection on train-split rows only, so no horizon's evaluation data
        is involved).
        """
        try:
            label_col = self._ranking_label_column(df)
            if label_col is None:
                logger.warning("MDA ranking: no label column found, falling back to variance")
                return None
            return self._mda_importance(
                df,
                feature_names,
                df[label_col].to_numpy(),
                frame_label_ends(df, label_col),
            )
        except Exception as e:
            logger.warning(f"MDA ranking failed: {e}, falling back to variance")
            return None

    def _ranking_label_column(self, df: pd.DataFrame) -> str | None:
        """Label column the selection ranks on: the first configured horizon's."""
        for h in self.config.horizons:
            candidate = f"label_h{h}"
            if candidate in df.columns:
                return candidate
        return None

    def _mda_importance(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        labels: np.ndarray,
        label_ends: np.ndarray | None,
    ) -> pd.Series | None:
        """Mean out-of-sample MDA importance of ``feature_names`` on ``df``'s rows.

        Args:
            df: Frame holding the features (rows are bar positions).
            feature_names: Features to rank.
            labels: Label per row of ``df`` (-99 = invalid).
            label_ends: Bar position each label resolves at, in ``df``'s row
                coordinates (None = no label-span purging beyond ``purge_bars``).

        Returns:
            Importance per feature sorted descending, or None when the frame
            cannot be ranked (too few rows, one class, CV failure).
        """
        from src.optimization.feature_selection.walk_forward import (
            WalkForwardFeatureSelector,
        )

        try:
            # Complete rows with a real label (-99 = no barrier outcome, not a class)
            usable = (
                df[list(feature_names)].notna().all(axis=1).to_numpy()
                & ~pd.isna(labels)
                & (labels != INVALID_LABEL)
            )
            positions = np.flatnonzero(usable)
            clean_df = df.iloc[positions][list(feature_names)]
            clean_y = pd.Series(labels[positions], index=clean_df.index)

            if len(clean_df) < 200:
                logger.warning(
                    f"MDA ranking: too few clean rows ({len(clean_df)} < 200), "
                    "falling back to variance"
                )
                return None

            # Subsample for large datasets — MDA ranking is stable at 50K rows.
            # Strided (every-Nth) sampling keeps rows in temporal order; a shuffling
            # subsample would make PurgedKFold's positional purge/embargo meaningless.
            clean_df, stride = _temporal_stride_subsample(clean_df, MDA_MAX_ROWS)
            clean_y = clean_y.iloc[::stride]
            positions = positions[::stride]
            if stride > 1:
                logger.info(
                    f"  MDA subsampling: strided every {stride}th row → {len(clean_df):,} rows"
                )
            # Label spans in bar positions: exact purging after row filtering
            # and striding (None when there are no label ends)
            spans = LabelSpans.from_rows(positions, label_ends) if label_ends is not None else None

            X = clean_df
            y = clean_y

            if y.nunique() < 2:
                logger.warning("MDA ranking: labels have < 2 classes, falling back to variance")
                return None

            # Purge/embargo are in original bars; after striding one row spans
            # `stride` bars, so scale down (ceil: never under-purge).
            purge_bars = -(-self.config.purge_bars // stride)
            embargo_bars = -(-self.config.embargo_bars // stride)

            # Cap embargo to at most 15% of data so MDA works on small datasets
            n_samples = len(X)
            max_embargo = int(n_samples * 0.15)
            mda_embargo = min(embargo_bars, max_embargo)
            mda_cv_config = PurgedKFoldConfig(
                n_splits=3,
                purge_bars=purge_bars,
                embargo_bars=mda_embargo,
            )
            mda_cv = PurgedKFold(mda_cv_config)
            cv_splits = list(mda_cv.split(X, y, label_spans=spans))

            selector = WalkForwardFeatureSelector(
                n_features_to_select=len(feature_names),
                selection_method="mda",
                n_estimators=50,
                mda_n_repeats=5,
                min_feature_frequency=0.01,
                random_state=self.config.random_state,
                use_clustered_importance=True,
                # Never force-merge uncorrelated features: the correlation-distance
                # cut alone decides clusters (joint permutation costs the same as
                # plain MDA at one group per feature, so no cap is needed).
                max_clusters=len(feature_names),
            )
            result = selector.select_features_walkforward(X, y, cv_splits)

            all_importances: dict[str, list[float]] = {f: [] for f in feature_names}
            for fold_info in result.importance_history:
                fold_imp = fold_info.get("importance", {})
                for feat in feature_names:
                    if feat in fold_imp:
                        all_importances[feat].append(fold_imp[feat])

            mean_importance = rank_by_importance(
                pd.Series(
                    {f: np.mean(scores) if scores else 0.0 for f, scores in all_importances.items()}
                ),
                noise_floor=IMPORTANCE_NOISE_FLOOR,
            )

            logger.info(
                f"  MDA ranking complete: {len(mean_importance)} features ranked "
                f"across {len(cv_splits)} folds"
            )
            for feat, score in mean_importance.head(5).items():
                logger.info(f"    {feat}: {score:.4f}")

            return mean_importance

        except Exception as e:
            logger.warning(f"MDA ranking failed: {e}, falling back to variance")
            return None

    def _pre_training_validation(
        self,
        df: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> None:
        """Validate data before training. Raises PreTrainingValidationError on failure.

        Runs: (1) leakage detection, (2) lookahead audit.

        NOTE: Feature selection and contract validation are NOT run here.
        Feature selection runs on train-only data via
        ``_run_feature_selection_on_train_data()``, and contract validation
        runs after that via ``_post_selection_contract_validation()`` — both
        called from the orchestrator's ``train()`` method.
        """
        errors: list[str] = []
        warnings: list[str] = []

        logger.info("\n" + "-" * 40)
        logger.info("PRE-TRAINING VALIDATION")
        logger.info("-" * 40)

        # Determine feature columns
        if feature_names is None:
            exclude_patterns = ["label", "sample_weight", "datetime", "date", "time"]
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            feature_names = [
                col
                for col in df.columns
                if col not in ohlcv_cols and not any(pat in col.lower() for pat in exclude_patterns)
            ]

        # 1. Leakage detection
        if self.config.check_leakage:
            self._validate_leakage(df, feature_names, errors, warnings)
        else:
            logger.info("  [1/2] Leakage detection: SKIPPED (check_leakage=False)")

        # 2. Lookahead audit — mandatory (consistent with data pipeline Phase 14C)
        logger.info("  [2/2] Lookahead audit: running (mandatory)")
        self._validate_lookahead(errors, warnings)

        if warnings:
            logger.warning("\n  Validation warnings:")
            for w in warnings:
                logger.warning(f"    - {w}")

        if errors:
            error_summary = "\n".join([f"  - {e}" for e in errors])
            raise PreTrainingValidationError(
                f"Pre-training validation failed:\n{error_summary}\n\n"
                f"To bypass validation, set in PipelineConfig:\n"
                f"  - check_leakage=False (disable leakage detection)\n"
                f"  Note: lookahead audit is mandatory and cannot be disabled."
            )

        logger.info("\n  Pre-training validation: PASSED")
        logger.info("-" * 40 + "\n")

    def _post_selection_contract_validation(self, df: pd.DataFrame) -> None:
        """Validate model contracts AFTER feature selection has populated _per_model_features.

        Raises PreTrainingValidationError if any model's selected feature count
        violates its contract bounds.
        """
        if not self.config.strict_validation:
            logger.info("  Contract validation: SKIPPED (strict_validation=False)")
            return

        # Determine feature columns for fallback
        exclude_patterns = ["label", "sample_weight", "datetime", "date", "time"]
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        feature_names = [
            col
            for col in df.columns
            if col not in ohlcv_cols and not any(pat in col.lower() for pat in exclude_patterns)
        ]

        errors: list[str] = []
        warnings: list[str] = []
        self._validate_contracts(df, feature_names, errors, warnings)

        if warnings:
            for w in warnings:
                logger.warning(f"  Contract warning: {w}")

        if errors:
            error_summary = "\n".join([f"  - {e}" for e in errors])
            hint = (
                ""
                if self.config.optimize_features
                else "Feature selection is off (data.features.selection_enabled=False); "
                "turn it on to cut each model to its contract's feature budget, or "
            )
            raise PreTrainingValidationError(
                f"Contract validation failed after feature selection:\n{error_summary}\n\n"
                f"{hint}set strict_validation=False in PipelineConfig to bypass."
            )

    def _run_feature_selection_on_train_data(
        self,
        df: pd.DataFrame,
    ) -> None:
        """Run feature selection pipeline on TRAINING data only.

        Computes the train split boundary using ``self.config.train_ratio``
        (matching ``UnifiedDataPreparation._split_data``) and passes only
        the training portion to ``_run_feature_selection_pipeline``.  This
        prevents correlation / variance statistics from leaking test-set
        information into feature selection decisions.

        Does nothing when ``config.optimize_features`` is off
        (``data.features.selection_enabled=False``): every model then gets
        every feature column.
        """
        # Determine feature columns (same logic as _pre_training_validation)
        exclude_patterns = ["label", "sample_weight", "datetime", "date", "time"]
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        feature_names = [
            col
            for col in df.columns
            if col not in ohlcv_cols and not any(pat in col.lower() for pat in exclude_patterns)
        ]

        if not feature_names:
            return

        if not self.config.optimize_features:
            logger.info(
                "  Feature selection disabled (data.features.selection_enabled=False): "
                f"every model trains on all {len(feature_names)} features"
            )
            if self.config.governance.get("report"):
                logger.warning("  Feature governance report skipped: feature selection is off")
            return

        # Compute train split boundary (mirrors UnifiedDataPreparation._split_data)
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        df_train = df.iloc[:train_end]

        logger.info(
            f"  Feature selection on train-only data: "
            f"{train_end:,}/{n:,} rows ({self.config.train_ratio:.0%})"
        )

        self._run_feature_selection_pipeline(df_train, feature_names)

    def _run_feature_selection_pipeline(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
    ) -> None:
        """Run MDA-first feature selection: MDA ranking -> correlation dedup -> low-variance."""
        self._all_feature_names = list(feature_names)
        candidate_features = list(feature_names)
        original_count = len(feature_names)

        # Step 1: MDA importance ranking (target-aware, falls back to variance)
        mda_importance = self._compute_mda_ranking(df, feature_names)
        if mda_importance is not None:
            ranking = mda_importance
            ranking_method = "MDA"
        else:
            feature_df_full = df[feature_names].dropna()
            ranking = (
                rank_by_importance(feature_df_full.var()) if len(feature_df_full) > 0 else None
            )
            ranking_method = "variance"

        logger.info(f"  Feature ranking method: {ranking_method}")

        # Steps 1b-4: timeframe budget, low-variance, decorrelation, per-model head
        feature_names, ranking, per_model = self._select_from_ranking(
            df, feature_names, ranking, ranking_method
        )
        self._per_model_features.update(per_model)
        logger.info(f"  Filter result: {original_count} -> {len(feature_names)} features")

        # Step 5: Robustness scoring (diagnostic — logs top features, doesn't change selection)
        if ranking is not None:
            try:
                from src.optimization.feature_selection.robustness_scoring import (
                    RobustnessScorer,
                )

                scorer = RobustnessScorer()
                scores_df = scorer.score_features(
                    feature_names=feature_names,
                    mda_importance=ranking,
                )
                if len(scores_df) > 0:
                    top5 = scores_df.head(5)
                    logger.info("  Robustness scores (top 5):")
                    for _, row in top5.iterrows():
                        logger.info(
                            f"    {row['feature']}: "
                            f"composite={row['composite_score']:.4f} "
                            f"(pred={row['predictive_power']:.3f})"
                        )
            except Exception as e:
                logger.debug(f"  Robustness scoring skipped: {e}")

        # Step 6: Opt-in governance report (diagnostic only — selection is final here)
        self._run_feature_governance(df, candidate_features, mda_importance)

    def _select_from_ranking(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        ranking: pd.Series | None,
        ranking_method: str,
        *,
        verbose: bool = True,
    ) -> tuple[list[str], pd.Series | None, dict[str, list[str]]]:
        """Everything after the ranking: timeframe budget, filters, per-model head.

        Pure with respect to the orchestrator (touches no attribute), so the live
        selection and the governance stability replay run the SAME code.

        Args:
            df: Frame the filters (variance, correlation) read; TRAIN rows.
            feature_names: Candidate features the ranking covers.
            ranking: Importance per feature (MDA, or variance fallback); None = unranked.
            ranking_method: Label for logs ("MDA" / "variance").
            verbose: Log the intermediate counts (off for replays inside blocks).

        Returns:
            (features surviving the filters, ranking re-sliced to them, features per model)
        """
        from src.core.contracts import get_model_contract

        info = logger.info if verbose else logger.debug
        warn = logger.warning if verbose else logger.debug
        # Step 1b: Timeframe budget (data.features.mtf_max_per_timeframe, None = off)
        budget = self.config.mtf_max_per_timeframe
        mtf_timeframes = self.config.mtf_timeframes if self.config.compute_mtf_features else []
        if budget is not None and ranking is not None and mtf_timeframes:
            budgeted = apply_timeframe_budget(
                ranking, feature_names, list(mtf_timeframes), max_per_timeframe=budget
            )
            info(
                f"  Timeframe budget: {len(feature_names)} -> {len(budgeted)} features "
                f"(max {budget} per timeframe of {list(mtf_timeframes)})"
            )
            feature_names = budgeted
            ranking = ranking.loc[ranking.index.isin(feature_names)]

        # Step 2: Low-variance cleanup, then greedy decorrelation in rank order
        contracts = [get_model_contract(m) for m in self.config.models]
        max_model_features = int(max(c.max_features for c in contracts))
        min_model_features = int(max(c.min_features for c in contracts))
        ranked = (
            [f for f in ranking.index if f in set(feature_names)]
            if ranking is not None
            else list(feature_names)
        )
        kept_after_var, removed_low_var = filter_low_variance(
            df[ranked], ranked, variance_threshold=0.01
        )
        if len(kept_after_var) < min_model_features:
            warn(
                f"  Low-variance filter would leave {len(kept_after_var)} features "
                f"(< {min_model_features}); skipped"
            )
            kept_after_var = ranked
        selected = select_decorrelated_by_rank(
            df,
            kept_after_var,
            n_target=max_model_features,
            n_min=min_model_features,
            correlation_threshold=0.85,
        )
        info(
            f"  Filter: low-variance removed {len(removed_low_var)}, decorrelation kept "
            f"{len(selected)}/{len(kept_after_var)} (target {max_model_features}, "
            f"min {min_model_features})"
        )
        feature_names = selected

        # Re-slice ranking to surviving features
        if ranking is not None:
            ranking = ranking.loc[ranking.index.isin(feature_names)]

        # Per-model feature subset selection
        per_model: dict[str, list[str]] = {}
        for model_name in self.config.models:
            model_contract = get_model_contract(model_name)
            max_feat = model_contract.max_features
            if len(feature_names) > max_feat and ranking is not None:
                model_features = ranking.head(int(max_feat)).index.tolist()
                info(
                    f"    {model_name}: selected top {len(model_features)}"
                    f"/{len(feature_names)} features by {ranking_method}"
                    f" (max={max_feat})"
                )
            else:
                model_features = list(feature_names)
            per_model[model_name] = model_features

        return feature_names, ranking, per_model

    def _run_feature_governance(
        self,
        df: pd.DataFrame,
        candidates: list[str],
        raw_importance: pd.Series | None,
    ) -> None:
        """Write the feature-governance report when ``config.governance`` enables it.

        Reads the finished selection (``_per_model_features``) and the TRAIN
        frame; changes nothing about either. A failure here is logged and
        never interrupts training.
        """
        settings = getattr(self.config, "governance", None)
        if not settings or not settings.get("report"):
            return
        from src.models.training.feature_governance import FeatureGovernance

        try:
            label_col = self._ranking_label_column(df)
            if label_col is None:
                logger.warning("  Feature governance skipped: no label column found")
                return
            FeatureGovernance(self.config, self._mda_importance, self._replay_selection).run(
                df,
                label_col=label_col,
                candidates=candidates,
                raw_importance=raw_importance,
                selected_by_model=self._per_model_features,
            )
        except Exception as e:
            logger.warning(f"  Feature governance skipped: {e}")

    def _replay_selection(
        self,
        block_df: pd.DataFrame,
        candidates: list[str],
        importance: pd.Series,
    ) -> dict[str, list[str]]:
        """Per-model selection the live pipeline WOULD make from ``importance`` on ``block_df``."""
        _, _, per_model = self._select_from_ranking(
            block_df, list(candidates), importance, "MDA", verbose=False
        )
        return per_model

    def _validate_contracts(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate data contracts for each model."""
        logger.info("  [1/3] Validating data contracts...")
        # Import OUTSIDE the try: the except clause below references
        # DataContractViolation, so a failed import inside the try would turn
        # the graceful-degradation path into a NameError.
        from src.core.contracts import DataContractViolation, get_model_contract

        try:
            if feature_names and len(feature_names) > 0:
                for model_name in self.config.models:
                    model_contract = get_model_contract(model_name)
                    model_feat = self._per_model_features.get(model_name, feature_names)
                    n_feat = len(model_feat)
                    issues = []
                    if n_feat < model_contract.min_features:
                        issues.append(
                            f"Too few features: model needs >= {model_contract.min_features}, "
                            f"data has {n_feat}"
                        )
                    if n_feat > model_contract.max_features:
                        issues.append(
                            f"Too many features: model max is {model_contract.max_features}, "
                            f"data has {n_feat}"
                        )
                    if issues:
                        errors.append(f"Contract violation for {model_name}: {'; '.join(issues)}")
                logger.info(
                    f"    Per-model features: {len(feature_names)} total, "
                    f"{len(self.config.models)} models validated"
                )
            else:
                warnings.append("No feature columns identified for contract validation")
        except DataContractViolation as e:
            errors.append(f"Data contract violation: {e}")
        except Exception as e:
            warnings.append(f"Contract validation skipped: {e}")

    def _validate_leakage(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Run correlation-based leakage check."""
        logger.info("  [2/3] Running leakage detection...")
        try:
            from src.validation.leakage_detection import check_feature_label_correlation

            label_col = None
            for h in self.config.horizons:
                candidate = f"label_h{h}"
                if candidate in df.columns:
                    label_col = candidate
                    break

            if label_col is not None and feature_names and len(feature_names) > 0:
                # Rows with the invalid-label sentinel (-99: warmup, end of data, and
                # every non-event bar under event sampling) carry no label; keeping
                # them would correlate features with "is a valid row" instead
                valid_rows = df[label_col].to_numpy() != INVALID_LABEL
                X_features = df.loc[valid_rows, feature_names]
                y_labels = df.loc[valid_rows, label_col].values
                report = check_feature_label_correlation(
                    features=X_features,
                    labels=y_labels,
                    feature_names=feature_names,
                    correlation_threshold=self.config.validation_correlation_threshold,
                )
                if report.n_suspicious > 0:
                    suspicious_names = report.get_suspicious_feature_names()[:5]
                    error_msg = (
                        f"LEAKAGE DETECTED: {report.n_suspicious} features have "
                        f"suspicious correlation (threshold={report.correlation_threshold}). "
                        f"Top suspicious: {', '.join(suspicious_names)}"
                    )
                    errors.append(error_msg)
                    logger.warning(f"    {error_msg}")
                else:
                    logger.info(
                        f"    Leakage check passed: "
                        f"{report.n_features} features analyzed, 0 suspicious"
                    )
            else:
                warnings.append("Leakage detection skipped: no label column or features found")
        except Exception as e:
            warnings.append(f"Leakage detection failed: {e}")

    def _validate_lookahead(
        self,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Run lookahead audit on resample config."""
        logger.info("  [3/3] Running lookahead audit...")
        try:
            from src.validation.lookahead_audit import validate_resample_config

            is_valid, issues = validate_resample_config(
                closed="left",
                label="left",
            )
            if not is_valid:
                errors.append(f"Lookahead risk: {'; '.join(issues)}")
            elif issues:
                if self.config.validation_fail_on_warning:
                    errors.extend(issues)
                else:
                    warnings.extend(issues)
            if is_valid:
                logger.info("    Lookahead audit passed: resample config validated")
        except Exception as e:
            warnings.append(f"Lookahead audit failed: {e}")

    def _analyze_ensemble_diversity(
        self,
        aligned_oof: Any | None,
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """Analyze diversity of base model predictions using DiversityAnalyzer (Phase 16E)."""
        if aligned_oof is None:
            logger.debug("No aligned OOF, skipping diversity analysis")
            return {}

        try:
            from src.models.ensemble.diversity import DiversityAnalyzer

            logger.info("\n--- Ensemble Diversity Analysis (Phase 16E) ---")

            base_predictions: dict[str, np.ndarray] = {}
            base_probabilities: dict[str, np.ndarray] = {}

            # Only the ensemble's own members (one horizon): other horizons'
            # OOF predict different labels
            members = set(getattr(aligned_oof, "model_names", None) or self._oof_predictions)
            for model_key, oof in self._oof_predictions.items():
                if model_key not in members:
                    continue
                if oof is not None and hasattr(oof, "predictions"):
                    # Extract class predictions and probabilities as numpy arrays
                    class_preds = oof.get_class_predictions()
                    probs = oof.get_probabilities()

                    # For sequence models with original_indices, first filter
                    # to valid rows, then apply the common_indices mask
                    if (
                        hasattr(aligned_oof, "common_indices")
                        and hasattr(oof, "original_indices")
                        and oof.original_indices is not None
                    ):
                        # Filter to valid predictions only (remove NaN rows)
                        valid_preds = class_preds[oof.original_indices]
                        valid_probs = probs[oof.original_indices]
                        # Now mask to common indices
                        mask = np.isin(oof.original_indices, aligned_oof.common_indices)
                        aligned_preds = valid_preds[mask]
                        aligned_probs = valid_probs[mask]
                    else:
                        # Tabular models: positional indexing
                        mask = np.isin(np.arange(len(class_preds)), aligned_oof.common_indices)
                        aligned_preds = class_preds[mask]
                        aligned_probs = probs[mask]

                    if len(aligned_preds) == len(aligned_oof.common_indices):
                        # Handle any remaining NaN from sequence boundaries
                        safe_preds = np.where(np.isnan(aligned_preds), 0, aligned_preds)
                        base_predictions[model_key] = safe_preds.astype(int)

                    if len(aligned_probs) == len(aligned_oof.common_indices):
                        base_probabilities[model_key] = aligned_probs

            if len(base_predictions) < 2:
                logger.warning("Need at least 2 models for diversity analysis")
                return {}

            y_true = None
            for h in self.config.horizons:
                label_col = f"label_h{h}"
                if label_col in df.columns:
                    y_true = df[label_col].values[aligned_oof.common_indices]
                    break

            analyzer = DiversityAnalyzer(
                min_diversity_threshold=0.3,
                correlation_threshold=0.8,
                n_classes=self.config.n_classes,
            )
            metrics = analyzer.analyze(
                base_predictions=base_predictions,
                base_probabilities=base_probabilities if base_probabilities else None,
                y_true=y_true,
            )

            logger.info(f"  Diversity Score: {metrics.diversity_score:.4f}")
            logger.info(f"  Q-statistic: {metrics.q_statistic:.4f}")
            logger.info(f"  Pairwise Correlation: {metrics.pairwise_correlation:.4f}")
            logger.info(f"  Disagreement Rate: {metrics.disagreement:.4f}")
            logger.info(f"  Double Fault Rate: {metrics.double_fault:.4f}")
            logger.info(f"  Entropy: {metrics.entropy:.4f}")

            if metrics.recommendations:
                logger.info("\n  Diversity Recommendations:")
                for rec in metrics.recommendations[:3]:
                    logger.info(f"    - {rec}")

            high_corr_pairs = [
                (m1, m2, corr)
                for (m1, m2), corr in metrics.model_pair_correlations.items()
                if abs(corr) > 0.8
            ]
            if high_corr_pairs:
                logger.info("\n  High Correlation Pairs (>0.8):")
                for m1, m2, corr in sorted(high_corr_pairs, key=lambda x: -abs(x[2]))[:5]:
                    logger.info(f"    {m1} <-> {m2}: {corr:.4f}")

            return {
                "diversity_score": metrics.diversity_score,
                "diversity_q_statistic": metrics.q_statistic,
                "diversity_correlation": metrics.pairwise_correlation,
                "diversity_disagreement": metrics.disagreement,
                "diversity_double_fault": metrics.double_fault,
                "diversity_entropy": metrics.entropy,
                "diversity_kl_divergence": metrics.kl_divergence,
            }

        except ImportError as e:
            logger.warning(f"Diversity analysis not available: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Diversity analysis failed: {e}")
            return {}

    def _generate_financial_reports(self, df: pd.DataFrame) -> None:
        """Generate financial reports with visualizations for trained models."""
        try:
            from src.models.evaluation.financial_report import (
                FinancialReportConfig,
                generate_financial_report,
            )
        except ImportError:
            logger.warning("Financial report generation not available - missing dependencies")
            return

        logger.info("\n" + "=" * 60)
        logger.info("GENERATING FINANCIAL REPORTS")
        logger.info("=" * 60)

        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        if "close" not in df.columns:
            logger.warning("Cannot generate financial report - 'close' column not found")
            return

        prices = df["close"].values
        timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        if timestamps is None and "datetime" in df.columns:
            timestamps = pd.to_datetime(df["datetime"])

        config = FinancialReportConfig(
            initial_equity=100000.0,
            commission_per_trade=2.50,
            slippage_ticks=1.0,
            tick_value=1.25 if self.config.symbol == "MES" else 0.10,
        )

        for model_key, result in self._model_results.items():
            oof = self._oof_predictions.get(model_key)
            if oof is None:
                logger.info(f"Skipping {model_key} - no OOF predictions available")
                continue

            # Use class predictions (1D numpy array), not the raw DataFrame
            predictions = oof.get_class_predictions()

            # For sequence models, filter to valid (non-NaN) predictions only
            valid_mask = ~np.isnan(predictions)
            if not valid_mask.all():
                predictions = predictions[valid_mask].astype(int)
            else:
                predictions = predictions.astype(int)

            y_true = None
            label_col = f"label_h{result.horizon}"
            if label_col in df.columns:
                y_labels = df[label_col].values[: len(oof.get_class_predictions())]
                # Apply same valid_mask to labels and prices
                y_true = (
                    y_labels[valid_mask] if not valid_mask.all() else y_labels[: len(predictions)]
                )

            if y_true is None:
                logger.warning(f"Skipping {model_key} - no true labels available")
                continue

            # Get matching prices slice
            prices_slice = prices[: len(oof.get_class_predictions())]
            if not valid_mask.all():
                prices_slice = prices_slice[valid_mask]
            else:
                prices_slice = prices_slice[: len(predictions)]

            ts_slice = None
            if timestamps is not None:
                ts_slice = timestamps[: len(oof.get_class_predictions())]
                if not valid_mask.all():
                    ts_slice = ts_slice[valid_mask]
                else:
                    ts_slice = ts_slice[: len(predictions)]

            if y_true is None:
                logger.warning(f"Skipping {model_key} - no true labels available")
                continue

            try:
                model_report_dir = reports_dir / model_key
                generate_financial_report(
                    model_name=result.model_name,
                    horizon=result.horizon,
                    predictions=predictions,
                    y_true=y_true,
                    prices=prices_slice,
                    timestamps=ts_slice,
                    output_dir=model_report_dir,
                    run_id=self.run_id,
                    config=config,
                )
                logger.info(f"Generated financial report for {model_key}")
            except Exception as e:
                logger.error(f"Failed to generate report for {model_key}: {e}")
