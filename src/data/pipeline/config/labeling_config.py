"""
Label balance constraints used by the GA optimization stage (PipelineRunner Stage 5).
"""

# =============================================================================
# LABEL BALANCE CONSTRAINTS (GA optimization)
# =============================================================================
# These thresholds prevent extreme class imbalance in optimized labels.
# The neutral class represents "hold" / timeout positions - important for:
# 1. Avoiding overtrading (transaction costs)
# 2. Filtering low-confidence signals
# 3. Maintaining realistic signal rates
LABEL_BALANCE_CONSTRAINTS = {
    "min_long_pct": 0.05,  # Minimum long share of total samples
    "min_short_pct": 0.05,  # Minimum short share of total samples
    "min_neutral_pct": 0.10,  # MINIMUM neutral share (HARD constraint)
    "target_neutral_low": 0.20,  # Target neutral range lower bound
    "target_neutral_high": 0.30,  # Target neutral range upper bound
    "max_neutral_pct": 0.40,  # Maximum neutral (too few signals)
    "min_short_signal_ratio": 0.10,  # Minimum short share among signals
    "max_short_signal_ratio": 0.90,  # Maximum short share among signals
    "min_any_class_pct": 0.10,  # Minimum for ANY class (fail-safe)
}
