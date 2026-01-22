"""
Validation functions for feature sets.

Contains functions to resolve, validate, and get feature set columns.
"""

import logging

from .core import FEATURE_SET_ALIASES, FeatureSetDefinition
from .definitions import FEATURE_SET_DEFINITIONS


def get_feature_set_definitions() -> dict[str, FeatureSetDefinition]:
    """Return a copy of feature set definitions."""
    return FEATURE_SET_DEFINITIONS.copy()


def resolve_feature_set_name(name: str) -> str:
    """Resolve a feature set name or alias to a canonical name."""
    if not name:
        raise ValueError("feature_set must be a non-empty string")
    normalized = name.strip().lower()
    canonical = FEATURE_SET_ALIASES.get(normalized, normalized)
    if canonical not in FEATURE_SET_DEFINITIONS and canonical != "all":
        valid = sorted(list(FEATURE_SET_DEFINITIONS.keys()) + ["all"])
        raise ValueError(f"Unknown feature_set '{name}'. Valid options: {valid}")
    return canonical


def resolve_feature_set_names(name: str) -> list[str]:
    """
    Resolve a feature set selection into a list of canonical names.

    Supports 'all' or comma-separated values.
    """
    if not name:
        raise ValueError("feature_set must be a non-empty string")
    selections = [part.strip() for part in name.split(",") if part.strip()]
    resolved: list[str] = []
    for selection in selections:
        canonical = resolve_feature_set_name(selection)
        if canonical == "all":
            return sorted(FEATURE_SET_DEFINITIONS.keys())
        resolved.append(canonical)
    # De-duplicate while preserving order
    unique: list[str] = []
    for item in resolved:
        if item not in unique:
            unique.append(item)
    return unique


def validate_feature_set_config(feature_set: str) -> list[str]:
    """Validate feature set selection."""
    errors: list[str] = []
    try:
        resolve_feature_set_names(feature_set)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def validate_feature_set_coverage(
    df_columns: list[str],
    feature_set: FeatureSetDefinition,
    *,
    raise_on_empty: bool = False,
) -> dict:
    """
    Validate that a feature set matches expected columns in a DataFrame.

    This function checks:
    1. That include_prefixes match at least one column
    2. That explicit_columns (if specified) are all present
    3. That the resolved feature set is not empty

    Parameters
    ----------
    df_columns : list[str]
        List of column names from the DataFrame
    feature_set : FeatureSetDefinition
        The feature set definition to validate
    raise_on_empty : bool, default False
        If True, raise ValueError when feature set resolves to zero columns.
        If False, return the validation result dict with warnings.

    Returns
    -------
    dict
        Validation result with keys:
        - valid: bool - True if no critical issues found
        - matched_columns: list[str] - Columns that matched the feature set
        - matched_count: int - Number of matched columns
        - unmatched_prefixes: list[str] - Prefixes that matched no columns
        - missing_explicit: list[str] - Required explicit columns not found
        - warnings: list[str] - Warning messages

    Raises
    ------
    ValueError
        If raise_on_empty=True and the feature set resolves to zero columns

    Examples
    --------
    >>> df_columns = ['return_1', 'return_5', 'rsi_14', 'volume_ratio']
    >>> feature_set = FEATURE_SET_DEFINITIONS['boosting_optimal']
    >>> result = validate_feature_set_coverage(df_columns, feature_set)
    >>> print(result['matched_count'])
    4
    """
    logger = logging.getLogger(__name__)

    result = {
        "valid": True,
        "matched_columns": [],
        "matched_count": 0,
        "unmatched_prefixes": [],
        "missing_explicit": [],
        "warnings": [],
    }

    df_columns_set = set(df_columns)

    # Find columns matching include_prefixes
    matched_by_prefix = set()
    for prefix in feature_set.include_prefixes:
        prefix_matches = [c for c in df_columns if c.startswith(prefix)]
        if prefix_matches:
            matched_by_prefix.update(prefix_matches)
        else:
            result["unmatched_prefixes"].append(prefix)
            result["warnings"].append(
                f"Feature set '{feature_set.name}': prefix '{prefix}' matched no columns"
            )

    # Add explicitly included columns
    matched_explicit = set()
    for col in feature_set.include_columns:
        if col in df_columns_set:
            matched_explicit.add(col)
        else:
            # include_columns are optional, just log if missing
            logger.debug(f"Feature set '{feature_set.name}': include_column '{col}' not found")

    # Check required explicit_columns (these MUST be present)
    for col in feature_set.explicit_columns:
        if col not in df_columns_set:
            result["missing_explicit"].append(col)
            result["warnings"].append(
                f"Feature set '{feature_set.name}': required column '{col}' not found"
            )

    # Combine all matched columns
    all_matched = matched_by_prefix | matched_explicit

    # Apply exclusions
    for prefix in feature_set.exclude_prefixes:
        all_matched = {c for c in all_matched if not c.startswith(prefix)}
    for col in feature_set.exclude_columns:
        all_matched.discard(col)

    result["matched_columns"] = sorted(all_matched)
    result["matched_count"] = len(all_matched)

    # Check for empty result
    if len(all_matched) == 0:
        msg = (
            f"Feature set '{feature_set.name}' resolved to zero columns. "
            f"Checked prefixes: {feature_set.include_prefixes}, "
            f"include_columns: {feature_set.include_columns}"
        )
        result["valid"] = False
        result["warnings"].append(msg)
        logger.warning(msg)

        if raise_on_empty:
            raise ValueError(msg)

    # Mark as invalid if any explicit columns are missing
    if result["missing_explicit"]:
        result["valid"] = False

    # Log warnings
    for warning in result["warnings"]:
        logger.warning(warning)

    return result


def get_feature_set_columns(
    df_columns: list[str],
    feature_set_name: str,
) -> list[str]:
    """
    Get the list of columns that match a feature set definition.

    Convenience function that resolves a feature set name and returns
    the columns from df_columns that match it.

    Parameters
    ----------
    df_columns : list[str]
        List of column names from the DataFrame
    feature_set_name : str
        Name or alias of the feature set (e.g., 'boosting_optimal', 'lstm')

    Returns
    -------
    list[str]
        Sorted list of columns that match the feature set

    Raises
    ------
    ValueError
        If feature_set_name is not recognized
    """
    canonical = resolve_feature_set_name(feature_set_name)
    feature_set = FEATURE_SET_DEFINITIONS[canonical]
    result = validate_feature_set_coverage(df_columns, feature_set, raise_on_empty=False)
    return result["matched_columns"]
