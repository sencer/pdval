"""Tests for the @validated decorator."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from pdval import (
  Datetime,
  Finite,
  Ge,
  HasColumns,
  Index,
  MonoUp,
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
    with pytest.raises(SchemaError, match=None):
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
    def process(data: Validated[pd.Series, Finite, Positive]):
      return data.sum()

    # Valid data
    valid_data = pd.Series([1.0, 2.0, 3.0])
    result = process(valid_data)
    assert result == 6.0

    # Fails Finite check
    with pytest.raises(SchemaError, match=None):
      process(pd.Series([1.0, np.nan, 3.0]))

    # Fails Positive check
    with pytest.raises(SchemaError, match=None):
      process(pd.Series([1.0, 0.0, 3.0]))

  def test_dataframe_validation(self):
    """Test DataFrame validation."""

    @validated
    def process(
      data: Validated[pd.DataFrame, HasColumns["a", "b"], Finite],  # noqa: F821
      validate: bool = True,
    ):
      return data["a"] + data["b"]

    valid_data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = process(valid_data)
    assert result.tolist() == [4, 6]

    # Missing column
    with pytest.raises(SchemaError, match=None):
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
    valid_data = pd.Series([1.0, 2.0, 3.0])
    result = processor.process(valid_data)
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
    with pytest.raises(SchemaError, match=None):
      process(pd.Series([1.0, np.inf, 3.0]))

  def test_multiple_arguments(self):
    """Test validation with multiple arguments."""

    @validated
    def combine(
      data1: Validated[pd.Series, Finite],
      data2: Validated[pd.Series, Finite],
    ):
      return data1 + data2

    valid1 = pd.Series([1.0, 2.0])
    valid2 = pd.Series([3.0, 4.0])
    result = combine(valid1, valid2)
    assert result.tolist() == [4.0, 6.0]

    # First argument invalid
    with pytest.raises(SchemaError, match=None):
      combine(pd.Series([np.inf, 2.0]), valid2)

    # Second argument invalid
    with pytest.raises(SchemaError, match=None):
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
        pd.DataFrame, HasColumns["high", "low", "close"], Ge["high", "low"]  # noqa: F821
      ],
    ):
      hl = data["high"] - data["low"]
      hc = abs(data["high"] - data["close"].shift(1))
      lc = abs(data["low"] - data["close"].shift(1))
      return pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # Valid OHLC
    valid_data = pd.DataFrame(
      {
        "high": [102, 105, 104],
        "low": [100, 103, 101],
        "close": [101, 104, 102],
      }
    )
    result = calculate_true_range(valid_data)
    assert len(result) == 3

    # High < Low should fail
    invalid_data = pd.DataFrame(
      {
        "high": [100, 105, 104],
        "low": [102, 103, 101],
        "close": [101, 104, 102],
      }
    )
    with pytest.raises(SchemaError, match=None):
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
    with pytest.raises(SchemaError, match=None):
      resample_data(invalid_data)

    # Non-monotonic datetime index
    dates_shuffled = [
      pd.Timestamp("2024-01-01"),
      pd.Timestamp("2024-01-03"),
      pd.Timestamp("2024-01-02"),
    ]
    non_monotonic = pd.Series([1, 2, 3], index=dates_shuffled)
    with pytest.raises(SchemaError, match=None):
      resample_data(non_monotonic)

  def test_percentage_returns_validation(self):
    """Test validation for percentage returns calculation."""

    @validated
    def calculate_returns(prices: Validated[pd.Series, Finite, Positive]):
      return prices.pct_change()

    # Valid prices
    valid_prices = pd.Series([100.0, 102.0, 101.0, 103.0])
    result = calculate_returns(valid_prices)
    assert len(result) == 4

    # Zero price fails Positive check
    with pytest.raises(SchemaError, match=None):
      calculate_returns(pd.Series([100.0, 0.0, 101.0]))

    # NaN price fails Finite check
    with pytest.raises(SchemaError, match=None):
      calculate_returns(pd.Series([100.0, np.nan, 101.0]))


class TestEdgeCases:
  """Tests for edge cases and error conditions."""

  def test_empty_series(self):
    """Test validation with empty Series."""

    @validated
    def process(data: Validated[pd.Series, Finite]):
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
    with pytest.raises(SchemaError):
      with_class(invalid_data)
    with pytest.raises(SchemaError):
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
    with pytest.raises(SchemaError, match=None):
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
      data: Validated[pd.Series, Finite] = pd.Series([1.0, 2.0]),
    ):
      return data.sum()

    # No arguments (uses default)
    result = process()
    assert result == 3.0

    # Override default
    result = process(pd.Series([5.0, 6.0]))
    assert result == 11.0
