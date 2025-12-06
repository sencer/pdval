"""Tests for individual validator classes."""

import numpy as np
import pandas as pd
import pytest

from pdval import (
  DateTimeIndexed,
  Finite,
  Ge,
  HasColumns,
  MonotonicIndex,
  NonNaN,
  NonNegative,
  Positive,
)


class TestFinite:
  """Tests for Finite validator."""

  def test_valid_series(self):
    """Test Finite validator with valid Series."""
    data = pd.Series([1.0, 2.0, 3.0])
    validator = Finite()
    result = validator.validate(data)
    assert result.equals(data)

  def test_valid_dataframe(self):
    """Test Finite validator with valid DataFrame."""
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    validator = Finite()
    result = validator.validate(data)
    assert result.equals(data)

  def test_series_with_inf(self):
    """Test Finite validator rejects Inf."""
    data = pd.Series([1.0, np.inf, 3.0])
    validator = Finite()
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(data)

  def test_series_with_nan(self):
    """Test Finite validator rejects NaN."""
    data = pd.Series([1.0, np.nan, 3.0])
    validator = Finite()
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(data)

  def test_dataframe_with_inf(self):
    """Test Finite validator rejects DataFrame with Inf."""
    data = pd.DataFrame({"a": [1.0, np.inf], "b": [3.0, 4.0]})
    validator = Finite()
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(data)

  def test_empty_series(self):
    """Test Finite validator with empty Series."""
    data = pd.Series([], dtype=float)
    validator = Finite()
    result = validator.validate(data)
    assert len(result) == 0

  def test_non_pandas_type(self):
    """Test Finite validator with non-pandas type."""
    validator = Finite()
    result = validator.validate(42)
    assert result == 42


class TestNonNaN:
  """Tests for NonNaN validator."""

  def test_valid_series(self):
    """Test NonNaN validator with valid Series."""
    data = pd.Series([1.0, 2.0, 3.0])
    validator = NonNaN()
    result = validator.validate(data)
    assert result.equals(data)

  def test_series_with_nan(self):
    """Test NonNaN validator rejects NaN."""
    data = pd.Series([1.0, np.nan, 3.0])
    validator = NonNaN()
    with pytest.raises(ValueError, match="must not contain NaN"):
      validator.validate(data)

  def test_series_with_inf_allowed(self):
    """Test NonNaN validator allows Inf."""
    data = pd.Series([1.0, np.inf, 3.0])
    validator = NonNaN()
    result = validator.validate(data)
    assert result.equals(data)

  def test_empty_series(self):
    """Test NonNaN validator with empty Series."""
    data = pd.Series([], dtype=float)
    validator = NonNaN()
    result = validator.validate(data)
    assert len(result) == 0


class TestNonNegative:
  """Tests for NonNegative validator."""

  def test_valid_series(self):
    """Test NonNegative validator with valid Series."""
    data = pd.Series([0.0, 1.0, 2.0])
    validator = NonNegative()
    result = validator.validate(data)
    assert result.equals(data)

  def test_series_with_negative(self):
    """Test NonNegative validator rejects negative values."""
    data = pd.Series([1.0, -1.0, 3.0])
    validator = NonNegative()
    with pytest.raises(ValueError, match="must be non-negative"):
      validator.validate(data)

  def test_zero_allowed(self):
    """Test NonNegative validator allows zero."""
    data = pd.Series([0.0, 0.0, 0.0])
    validator = NonNegative()
    result = validator.validate(data)
    assert result.equals(data)

  def test_dataframe(self):
    """Test NonNegative validator with DataFrame."""
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    validator = NonNegative()
    result = validator.validate(data)
    assert result.equals(data)


class TestPositive:
  """Tests for Positive validator."""

  def test_valid_series(self):
    """Test Positive validator with valid Series."""
    data = pd.Series([1.0, 2.0, 3.0])
    validator = Positive()
    result = validator.validate(data)
    assert result.equals(data)

  def test_series_with_zero(self):
    """Test Positive validator rejects zero."""
    data = pd.Series([1.0, 0.0, 3.0])
    validator = Positive()
    with pytest.raises(ValueError, match="must be positive"):
      validator.validate(data)

  def test_series_with_negative(self):
    """Test Positive validator rejects negative values."""
    data = pd.Series([1.0, -1.0, 3.0])
    validator = Positive()
    with pytest.raises(ValueError, match="must be positive"):
      validator.validate(data)

  def test_all_positive(self):
    """Test Positive validator with all positive values."""
    data = pd.Series([0.1, 100.0, 0.001])
    validator = Positive()
    result = validator.validate(data)
    assert result.equals(data)


class TestDateTimeIndexed:
  """Tests for DateTimeIndexed validator."""

  def test_valid_datetime_index(self):
    """Test DateTimeIndexed validator with valid DatetimeIndex."""
    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.Series([1, 2, 3], index=dates)
    validator = DateTimeIndexed()
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_int_index(self):
    """Test DateTimeIndexed validator rejects integer index."""
    data = pd.Series([1, 2, 3])
    validator = DateTimeIndexed()
    with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
      validator.validate(data)

  def test_dataframe_with_datetime_index(self):
    """Test DateTimeIndexed validator with DataFrame."""
    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.DataFrame({"a": [1, 2, 3]}, index=dates)
    validator = DateTimeIndexed()
    result = validator.validate(data)
    assert result.equals(data)

  def test_non_pandas_type(self):
    """Test DateTimeIndexed validator with non-pandas type."""
    validator = DateTimeIndexed()
    result = validator.validate([1, 2, 3])
    assert result == [1, 2, 3]


class TestMonotonicIndex:
  """Tests for MonotonicIndex validator."""

  def test_valid_monotonic_increasing(self):
    """Test MonotonicIndex validator with monotonic index."""
    data = pd.Series([1, 2, 3], index=[0, 1, 2])
    validator = MonotonicIndex()
    result = validator.validate(data)
    assert result.equals(data)

  def test_non_monotonic_index(self):
    """Test MonotonicIndex validator rejects non-monotonic index."""
    data = pd.Series([1, 2, 3], index=[0, 2, 1])
    validator = MonotonicIndex()
    with pytest.raises(ValueError, match="Index must be monotonic increasing"):
      validator.validate(data)

  def test_datetime_monotonic(self):
    """Test MonotonicIndex validator with datetime index."""
    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.Series([1, 2, 3], index=dates)
    validator = MonotonicIndex()
    result = validator.validate(data)
    assert result.equals(data)

  def test_datetime_non_monotonic(self):
    """Test MonotonicIndex validator rejects non-monotonic datetime."""
    dates = [
      pd.Timestamp("2024-01-01"),
      pd.Timestamp("2024-01-03"),
      pd.Timestamp("2024-01-02"),
    ]
    data = pd.Series([1, 2, 3], index=dates)
    validator = MonotonicIndex()
    with pytest.raises(ValueError, match="Index must be monotonic increasing"):
      validator.validate(data)


class TestHasColumns:
  """Tests for HasColumns validator."""

  def test_valid_single_column(self):
    """Test HasColumns validator with single column."""
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validator = HasColumns["a"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_valid_multiple_columns(self):
    """Test HasColumns validator with multiple columns."""
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    validator = HasColumns["a", "b"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_missing_single_column(self):
    """Test HasColumns validator with missing column."""
    data = pd.DataFrame({"a": [1, 2]})
    validator = HasColumns["b"]
    with pytest.raises(ValueError, match="Missing columns: \\['b'\\]"):
      validator.validate(data)

  def test_missing_multiple_columns(self):
    """Test HasColumns validator with missing columns."""
    data = pd.DataFrame({"a": [1, 2]})
    validator = HasColumns["b", "c"]
    with pytest.raises(ValueError, match="Missing columns:"):
      validator.validate(data)

  def test_non_dataframe(self):
    """Test HasColumns validator with non-DataFrame."""
    data = pd.Series([1, 2, 3])
    validator = HasColumns["a"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_class_getitem_single(self):
    """Test HasColumns __class_getitem__ with single column."""
    validator = HasColumns["col1"]
    assert validator.columns == ["col1"]

  def test_class_getitem_multiple(self):
    """Test HasColumns __class_getitem__ with multiple columns."""
    validator = HasColumns["col1", "col2", "col3"]
    assert validator.columns == ["col1", "col2", "col3"]


class TestGe:
  """Tests for Ge (column comparison) validator."""

  def test_valid_comparison(self):
    """Test Ge validator with valid column comparison."""
    data = pd.DataFrame({"high": [10, 20, 30], "low": [5, 10, 15]})
    validator = Ge["high", "low"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_equal_values_allowed(self):
    """Test Ge validator allows equal values."""
    data = pd.DataFrame({"high": [10, 10, 10], "low": [10, 10, 10]})
    validator = Ge["high", "low"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_comparison(self):
    """Test Ge validator rejects invalid comparison."""
    data = pd.DataFrame({"high": [10, 5, 30], "low": [5, 10, 15]})
    validator = Ge["high", "low"]
    with pytest.raises(ValueError, match="high must be >= low"):
      validator.validate(data)

  def test_missing_columns(self):
    """Test Ge validator with missing columns."""
    data = pd.DataFrame({"high": [10, 20]})
    validator = Ge["high", "low"]
    # Should not raise if column is missing
    result = validator.validate(data)
    assert result.equals(data)

  def test_class_getitem(self):
    """Test Ge __class_getitem__."""
    validator = Ge["col1", "col2"]
    assert validator.col1 == "col1"
    assert validator.col2 == "col2"

  def test_non_dataframe(self):
    """Test Ge validator with non-DataFrame."""
    data = pd.Series([1, 2, 3])
    validator = Ge["a", "b"]
    result = validator.validate(data)
    assert result.equals(data)
