"""Tests for the @validated decorator."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportArgumentType=false

from typing import Literal

from loguru import logger
import numpy as np
import pandas as pd
import pytest

from pdval import (
  Datetime,
  Finite,
  Ge,
  HasColumn,
  HasColumns,
  Index,
  MaybeEmpty,
  MonoUp,
  Nullable,
  Positive,
  Validated,
  validated,
)


class TestValidatedDecorator:
  """Tests for @validated decorator basic functionality."""

  def test_function_with_validation(self):
    """Test @validated decorator validates arguments."""

    @validated
    def process(data: Validated[pd.Series, Finite]):
      return data.sum()

    valid_data = pd.Series([1.0, 2.0, 3.0])
    result = process(valid_data)
    assert result == 6.0

  def test_function_rejects_invalid_data(self):
    """Test @validated decorator rejects invalid data."""

    @validated
    def process(data: Validated[pd.Series, Finite]):
      return data.sum()

    invalid_data = pd.Series([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="must be finite"):
      process(invalid_data)

  def test_validation_can_be_disabled(self):
    """Test validation can be disabled with skip_validation=True."""

    @validated
    def process(data: Validated[pd.Series, Finite]):
      return data.sum()

    invalid_data = pd.Series([1.0, np.inf, 3.0])
    # Should not raise when validation is disabled
    result = process(invalid_data, skip_validation=True)
    assert np.isinf(result)

  def test_multiple_validators(self):
    """Test multiple validators in chain."""

    @validated
    def process(data: Validated[pd.Series, Finite, Positive, Nullable]):
      return data.sum()

    # Valid data
    valid_data = pd.Series([1.0, 2.0, 3.0])
    result = process(valid_data)
    assert result == 6.0

    # Fails Finite check
    with pytest.raises(ValueError, match="must be finite"):
      process(pd.Series([1.0, np.nan, 3.0]))

    # Fails Positive check
    with pytest.raises(ValueError, match="must be positive"):
      process(pd.Series([1.0, 0.0, 3.0]))

  def test_dataframe_validation(self):
    """Test DataFrame validation."""

    @validated
    def process(
      data: Validated[pd.DataFrame, HasColumns[Literal["a", "b"]], Finite],
    ):
      return data["a"] + data["b"]

    valid_data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = process(valid_data)
    assert result.tolist() == [4, 6]

    # Missing column
    with pytest.raises(ValueError, match="Missing columns"):
      process(pd.DataFrame({"a": [1, 2]}))

  def test_preserves_function_metadata(self):
    """Test decorator preserves function name and docstring."""

    @validated
    def my_function(data: Validated[pd.Series, Finite]):
      """My docstring."""
      return data.sum()

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring."

  def test_works_with_methods(self):
    """Test @validated works with class methods."""

    class Processor:
      @validated
      def process(self, data: Validated[pd.Series, Finite]):
        return data.sum()

    processor = Processor()
    result = processor.process(pd.Series([1.0, 2.0, 3.0]))  # pyright: ignore[reportAttributeAccessIssue]
    assert result == 6.0

  def test_optional_validated_argument(self):
    """Test Optional[Validated[...]] annotation."""

    @validated
    def process(data: Validated[pd.Series, Finite] | None = None):
      if data is None:
        return 0
      return data.sum()

    # None is allowed
    result = process(None)
    assert result == 0

    # Valid data works
    result = process(pd.Series([1.0, 2.0, 3.0]))
    assert result == 6.0

    # Invalid data still raises
    with pytest.raises(ValueError, match="must be finite"):
      process(pd.Series([1.0, np.inf, 3.0]))

  def test_optional_validated_argument_with_nan(self):
    """Test Optional[Validated[..., Nullable]] allows NaNs."""

    @validated
    def process(data: Validated[pd.Series, Nullable] | None = None):
      if data is None:
        return 0
      return data.sum()

    # NaNs allowed due to Nullable
    result = process(pd.Series([1.0, np.nan, 3.0]))
    # sum() skips NaNs, so result is 4.0
    assert result == 4.0

  def test_multiple_arguments(self):
    """Test validation with multiple arguments."""

    @validated
    def combine(
      data1: Validated[pd.Series, Finite, Nullable],
      data2: Validated[pd.Series, Finite, Nullable],
    ):
      return data1 + data2

    valid1 = pd.Series([1.0, 2.0])
    valid2 = pd.Series([3.0, 4.0])
    result = combine(valid1, valid2)
    assert result.tolist() == [4.0, 6.0]

    # First argument invalid
    with pytest.raises(ValueError, match="must be finite"):
      combine(pd.Series([np.inf, 2.0]), valid2)

    # Second argument invalid
    with pytest.raises(ValueError, match="must be finite"):
      combine(valid1, pd.Series([3.0, np.nan]))

  def test_non_validated_arguments_ignored(self):
    """Test non-validated arguments are not validated."""

    @validated
    def process(
      data: Validated[pd.Series, Finite],
      multiplier: float,
    ):
      return data * multiplier

    valid_data = pd.Series([1.0, 2.0, 3.0])
    result = process(valid_data, multiplier=2.0)
    assert result.tolist() == [2.0, 4.0, 6.0]


class TestComplexValidations:
  """Tests for complex validation scenarios."""

  def test_ohlc_validation(self):
    """Test validation of OHLC data."""

    @validated
    def calculate_true_range(
      data: Validated[
        pd.DataFrame,
        HasColumns[Literal["high", "low", "close"]],
        Ge[Literal["high", "low"]],
      ],
    ):
      hl = data["high"] - data["low"]
      hc = abs(data["high"] - data["close"].shift(1))
      lc = abs(data["low"] - data["close"].shift(1))
      return pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # Valid OHLC
    valid_data = pd.DataFrame({
      "high": [102, 105, 104],
      "low": [100, 103, 101],
      "close": [101, 104, 102],
    })
    result = calculate_true_range(valid_data)
    assert len(result) == 3

    # High < Low should fail
    invalid_data = pd.DataFrame({
      "high": [100, 105, 104],
      "low": [102, 103, 101],
      "close": [101, 104, 102],
    })
    with pytest.raises(ValueError, match="high must be >= low"):
      calculate_true_range(invalid_data)

  def test_time_series_validation(self):
    """Test time series specific validation."""

    @validated
    def resample_data(
      data: Validated[pd.Series, Index[Datetime, MonoUp], Finite],
      freq: str = "1D",
    ):
      return data.resample(freq).mean()

    # Valid time series
    dates = pd.date_range("2024-01-01", periods=10, freq="h")
    valid_data = pd.Series(range(10), index=dates)
    result = resample_data(valid_data)
    assert isinstance(result.index, pd.DatetimeIndex)

    # Non-datetime index
    invalid_data = pd.Series(range(10))
    with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
      resample_data(invalid_data)

    # Non-monotonic datetime index
    dates_shuffled = [
      pd.Timestamp("2024-01-01"),
      pd.Timestamp("2024-01-03"),
      pd.Timestamp("2024-01-02"),
    ]
    non_monotonic = pd.Series([1, 2, 3], index=dates_shuffled)
    with pytest.raises(ValueError, match="must be monotonically increasing"):
      resample_data(non_monotonic)

  def test_percentage_returns_validation(self):
    """Test validation for percentage returns calculation."""

    @validated
    @validated
    def calculate_returns(prices: Validated[pd.Series, Finite, Positive, Nullable]):
      return prices.pct_change()

    # Valid prices
    valid_prices = pd.Series([100.0, 102.0, 101.0, 103.0])
    result = calculate_returns(valid_prices)
    assert len(result) == 4

    # Zero price fails Positive check
    with pytest.raises(ValueError, match="must be positive"):
      calculate_returns(pd.Series([100.0, 0.0, 101.0]))

    # NaN price fails Finite check
    with pytest.raises(ValueError, match="must be finite"):
      calculate_returns(pd.Series([100.0, np.nan, 101.0]))


class TestEdgeCases:
  """Tests for edge cases and error conditions."""

  def test_empty_series_fails_by_default(self):
    """Test validation fails with empty Series by default."""

    @validated
    def process(data: Validated[pd.Series, Finite]):
      return len(data)

    empty_data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Data must not be empty"):
      process(empty_data)

  def test_empty_series_allowed_with_marker(self):
    """Test validation allows empty Series with MaybeEmpty."""

    @validated
    def process(data: Validated[pd.Series, MaybeEmpty]):
      return len(data)

    empty_data = pd.Series([], dtype=float)
    result = process(empty_data)
    assert result == 0

  def test_validator_class_vs_instance(self):
    """Test both Validator class and instance work."""

    @validated
    def with_class(data: Validated[pd.Series, Finite]):
      return data.sum()

    @validated
    def with_instance(data: Validated[pd.Series, Finite()]):
      return data.sum()

    valid_data = pd.Series([1.0, 2.0, 3.0])
    invalid_data = pd.Series([1.0, np.inf, 3.0])

    # Both should work with valid data
    assert with_class(valid_data) == 6.0
    assert with_instance(valid_data) == 6.0

    # Both should reject invalid data
    with pytest.raises(ValueError):
      with_class(invalid_data)
    with pytest.raises(ValueError):
      with_instance(invalid_data)

  def test_function_without_validate_param(self):
    """Test function without validate parameter defaults to True."""

    @validated
    def process(data: Validated[pd.Series, Finite]):
      return data.sum()

    valid_data = pd.Series([1.0, 2.0, 3.0])
    result = process(valid_data)
    assert result == 6.0

    # Should still validate and reject invalid data
    invalid_data = pd.Series([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="must be finite"):
      process(invalid_data)

  def test_kwargs_arguments(self):
    """Test validation works with keyword arguments."""

    @validated
    def process(data: Validated[pd.Series, Finite]):
      return data.sum()

    valid_data = pd.Series([1.0, 2.0, 3.0])

    # Positional
    result = process(valid_data)
    assert result == 6.0

    # Keyword
    result = process(data=valid_data)
    assert result == 6.0

    # Mixed
    result = process(valid_data, skip_validation=False)
    assert result == 6.0

  def test_default_argument_values(self):
    """Test validation with default argument values."""

    @validated
    def process(
      data: Validated[pd.Series, Finite] | None = None,
    ):
      if data is None:
        data = pd.Series([1.0, 2.0])
      return data.sum()

    # No arguments (uses default)
    result = process()
    assert result == 3.0

    # Override default
    result = process(pd.Series([5.0, 6.0]))
    assert result == 11.0


def test_validated_decorator_defaults():
  @validated
  def process(data: Validated[pd.Series, Finite]):
    return data

  # Default: Validation ON
  with pytest.raises(ValueError):
    process(pd.Series([float("inf")]))

  # Explicit Skip: Validation OFF
  process(pd.Series([float("inf")]), skip_validation=True)  # pyright: ignore[reportCallIssue]


def test_validated_decorator_skip_default():
  @validated(skip_validation_by_default=True)
  def process(data: Validated[pd.Series, Finite]):
    return data

  # Default: Validation OFF
  process(pd.Series([float("inf")]))

  # Explicit Enable: Validation ON
  with pytest.raises(ValueError):
    process(pd.Series([float("inf")]), skip_validation=False)  # pyright: ignore[reportCallIssue]


def test_validated_decorator_no_args_call():
  # This is technically valid python: @validated()
  @validated()
  def process(data: Validated[pd.Series, Finite]):
    return data

  # Default: Validation ON
  with pytest.raises(ValueError):
    process(pd.Series([float("inf")]))


def test_validated_decorator_explicit_false_default():
  @validated(skip_validation_by_default=False)
  def process(data: Validated[pd.Series, Finite]):
    return data

  # Default: Validation ON
  with pytest.raises(ValueError):
    process(pd.Series([float("inf")]))


def test_warn_only():
  """Test warn_only functionality and runtime overrides."""

  # Case 1: Default False, Override True
  @validated(warn_only_by_default=False)
  def process_strict(data: Validated[pd.Series, Finite]):
    return data.sum()

  invalid_data = pd.Series([1.0, float("inf")])

  # Should raise by default
  with pytest.raises(ValueError):
    process_strict(invalid_data)

  # Should return None when overridden
  logs = []
  handler_id = logger.add(logs.append, format="{message}")
  try:
    result = process_strict(invalid_data, warn_only=True)  # pyright: ignore[reportCallIssue]
    assert result is None
    assert "Validation failed" in "".join(str(log) for log in logs)
  finally:
    logger.remove(handler_id)

  # Case 2: Default True, Override False
  @validated(warn_only_by_default=True)
  def process_warn(data: Validated[pd.Series, Finite]):
    return data.sum()

  # Should return None by default
  assert process_warn(invalid_data) is None

  # Should raise when overridden
  with pytest.raises(ValueError):
    process_warn(invalid_data, warn_only=False)  # pyright: ignore[reportCallIssue]


class TestDefaultStrictness:
  """Tests for default NonNaN and NonEmpty behavior."""

  def test_default_non_nan(self):
    """Test that Validated implies NonNaN by default."""

    @validated
    def process(data: Validated[pd.Series, None]):
      return data.sum()

    # Valid data
    assert process(pd.Series([1, 2, 3])) == 6

    # NaN data fails
    with pytest.raises(ValueError, match="must not contain NaN"):
      process(pd.Series([1, np.nan, 3]))

  def test_default_non_empty(self):
    """Test that Validated implies NonEmpty by default."""

    @validated
    def process(data: Validated[pd.Series, None]):
      return len(data)

    # Valid data
    assert process(pd.Series([1])) == 1

    # Empty data fails
    with pytest.raises(ValueError, match="Data must not be empty"):
      process(pd.Series([], dtype=float))

  def test_nullable_opt_out(self):
    """Test Nullable opts out of NonNaN."""

    @validated
    def process(data: Validated[pd.Series, Nullable]):
      return data.sum()

    # NaN data allowed
    # sum() skips NaNs, so result is 4.0
    assert process(pd.Series([1, np.nan, 3])) == 4.0

  def test_maybe_empty_opt_out(self):
    """Test MaybeEmpty opts out of NonEmpty."""

    @validated
    def process(data: Validated[pd.Series, MaybeEmpty]):
      return len(data)

    # Empty data allowed
    assert process(pd.Series([], dtype=float)) == 0

  def test_has_column_defaults(self):
    """Test HasColumn implies NonNaN and NonEmpty by default."""

    @validated
    def process(data: Validated[pd.DataFrame, HasColumn["a"]]):  # noqa: F821
      return data["a"].sum()

    # Valid
    assert process(pd.DataFrame({"a": [1, 2]})) == 3

    # NaN fails
    with pytest.raises(ValueError, match="must not contain NaN"):
      process(pd.DataFrame({"a": [1, np.nan]}))

    # Empty fails (HasColumn checks column data)
    with pytest.raises(ValueError, match="Data must not be empty"):
      process(pd.DataFrame({"a": []}, dtype=float))

  def test_has_column_opt_out(self):
    """Test HasColumn opt-out."""

    @validated
    def process(
      data: Validated[
        pd.DataFrame, HasColumn["a", Nullable, MaybeEmpty]  # noqa: F821
      ],
    ):
      return len(data)

    # Empty allowed
    process(pd.DataFrame({"a": []}, dtype=float))

  def test_mixed_column_validation(self):
    """Test mixed strict and nullable columns."""

    @validated
    def process(
      data: Validated[
        pd.DataFrame,
        HasColumn["col1"],  # noqa: F821
        HasColumn["col2", Nullable],  # noqa: F821
      ],
    ):
      return len(data)

    # 1. Valid case: col1 strict, col2 has NaNs, col3 has NaNs
    df_valid = pd.DataFrame({
      "col1": [1, 2, 3],
      "col2": [1, np.nan, 3],
      "col3": [np.nan, np.nan, np.nan],  # Unspecified column with NaNs
    })
    assert process(df_valid) == 3

    # 2. Fail case: col1 has NaNs
    df_fail_col1 = pd.DataFrame({"col1": [1, np.nan, 3], "col2": [1, 2, 3]})
    with pytest.raises(ValueError, match="Data must not contain NaN values"):
      process(df_fail_col1)

    # 3. Fail case: col2 is empty (only Nullable was opted out)
    # Note: HasColumn extracts the column as Series.
    # If the DF is not empty but the column is empty?
    # A column in a non-empty DF cannot be empty (length matches index).
    # So we test with an empty DataFrame that has these columns.
    # But wait, if DF is empty, then col1 is empty.
    # col1 is Strict -> NonEmpty -> Fails.
    df_empty = pd.DataFrame({"col1": [], "col2": []}, dtype=float)
    with pytest.raises(ValueError, match="Data must not be empty"):
      process(df_empty)
