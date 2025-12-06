"""Tests for individual validator classes."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportArgumentType=false

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from pdval import (
  Datetime,
  Finite,
  Ge,
  Gt,
  HasColumn,
  HasColumns,
  Index,
  IsDtype,
  Le,
  Lt,
  MonoDown,
  MonoUp,
  NonEmpty,
  NonNaN,
  NonNegative,
  NoTimeGaps,
  Positive,
  UniqueIndex,
  Validated,
  validated,
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


class TestIndexDatetime:
  """Tests for Index[Datetime] validator."""

  def test_valid_datetime_index(self):
    """Test Index[Datetime] validator with valid DatetimeIndex."""
    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.Series([1, 2, 3], index=dates)
    validator = Index[Datetime]
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_int_index(self):
    """Test Index[Datetime] validator rejects integer index."""
    data = pd.Series([1, 2, 3])
    validator = Index[Datetime]
    with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
      validator.validate(data)

  def test_dataframe_with_datetime_index(self):
    """Test Index[Datetime] validator with DataFrame."""
    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.DataFrame({"a": [1, 2, 3]}, index=dates)
    validator = Index[Datetime]
    result = validator.validate(data)
    assert result.equals(data)

  def test_non_pandas_type(self):
    """Test Index[Datetime] validator with non-pandas type."""
    validator = Index[Datetime]
    result = validator.validate([1, 2, 3])
    assert result == [1, 2, 3]


class TestIndexMonoUp:
  """Tests for Index[MonoUp] validator."""

  def test_valid_monotonic_increasing(self):
    """Test Index[MonoUp] validator with monotonic index."""
    data = pd.Series([1, 2, 3], index=[0, 1, 2])
    validator = Index[MonoUp]
    result = validator.validate(data)
    assert result.equals(data)

  def test_non_monotonic_index(self):
    """Test Index[MonoUp] validator rejects non-monotonic index."""
    data = pd.Series([1, 2, 3], index=[0, 2, 1])
    validator = Index[MonoUp]
    with pytest.raises(ValueError, match="Index must be monotonically increasing"):
      validator.validate(data)

  def test_datetime_monotonic(self):
    """Test Index[MonoUp] validator with datetime index."""
    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.Series([1, 2, 3], index=dates)
    validator = Index[MonoUp]
    result = validator.validate(data)
    assert result.equals(data)

  def test_datetime_non_monotonic(self):
    """Test Index[MonoUp] validator rejects non-monotonic datetime."""
    dates = [
      pd.Timestamp("2024-01-01"),
      pd.Timestamp("2024-01-03"),
      pd.Timestamp("2024-01-02"),
    ]
    data = pd.Series([1, 2, 3], index=dates)
    validator = Index[MonoUp]
    with pytest.raises(ValueError, match="Index must be monotonically increasing"):
      validator.validate(data)


class TestHasColumns:
  """Tests for HasColumns validator."""

  def test_valid_single_column(self):
    """Test HasColumns validator with single column."""
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validator = HasColumns[Literal["a"]]
    result = validator.validate(data)
    assert result.equals(data)

  def test_valid_multiple_columns(self):
    """Test HasColumns validator with multiple columns."""
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    validator = HasColumns[Literal["a", "b"]]
    result = validator.validate(data)
    assert result.equals(data)

  def test_missing_single_column(self):
    """Test HasColumns validator with missing column."""
    data = pd.DataFrame({"a": [1, 2]})
    validator = HasColumns[Literal["b"]]
    with pytest.raises(ValueError, match="Missing columns: \\['b'\\]"):
      validator.validate(data)

  def test_missing_multiple_columns(self):
    """Test HasColumns validator with missing columns."""
    data = pd.DataFrame({"a": [1, 2]})
    validator = HasColumns[Literal["b", "c"]]
    with pytest.raises(ValueError, match="Missing columns:"):
      validator.validate(data)

  def test_non_dataframe(self):
    """Test HasColumns validator with non-DataFrame."""
    data = pd.Series([1, 2, 3])
    validator = HasColumns[Literal["a"]]
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
      validator.validate(data)

  def test_class_getitem_single(self):
    """Test HasColumns __class_getitem__ with single column."""
    validator = HasColumns[Literal["col1"]]
    assert validator.columns == ["col1"]

  def test_class_getitem_multiple(self):
    """Test HasColumns __class_getitem__ with multiple columns."""
    validator = HasColumns[Literal["col1", "col2", "col3"]]
    assert validator.columns == ["col1", "col2", "col3"]


class TestGe:
  """Tests for Ge (column comparison) validator."""

  def test_valid_comparison(self):
    """Test Ge validator with valid column comparison."""
    data = pd.DataFrame({"high": [10, 20, 30], "low": [5, 10, 15]})
    validator = Ge[Literal["high", "low"]]
    result = validator.validate(data)
    assert result.equals(data)

  def test_equal_values_allowed(self):
    """Test Ge validator allows equal values."""
    data = pd.DataFrame({"high": [10, 10, 10], "low": [10, 10, 10]})
    validator = Ge[Literal["high", "low"]]
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_comparison(self):
    """Test Ge validator rejects invalid comparison."""
    data = pd.DataFrame({"high": [10, 5, 30], "low": [5, 10, 15]})
    validator = Ge[Literal["high", "low"]]
    with pytest.raises(ValueError, match="high must be >= low"):
      validator.validate(data)

  def test_missing_columns(self):
    """Test Ge validator with missing columns."""
    data = pd.DataFrame({"high": [10, 20]})
    validator = Ge[Literal["high", "low"]]
    # Should not raise if column is missing
    result = validator.validate(data)
    assert result.equals(data)

  def test_class_getitem(self):
    """Test Ge __class_getitem__."""
    validator = Ge[Literal["col1", "col2"]]
    assert validator.targets == ("col1", "col2")

  def test_non_dataframe(self):
    """Test Ge validator with non-DataFrame."""
    data = pd.Series([1, 2, 3])
    validator = Ge[Literal["a", "b"]]
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
      validator.validate(data)


class TestLe:
  """Tests for Le (<=) validator."""

  def test_valid_comparison(self):
    """Test Le validator with valid column comparison."""
    data = pd.DataFrame({"low": [5, 10, 15], "high": [10, 20, 30]})
    validator = Le["low", "high"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_equal_values_allowed(self):
    """Test Le validator allows equal values."""
    data = pd.DataFrame({"low": [10, 10, 10], "high": [10, 10, 10]})
    validator = Le["low", "high"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_comparison(self):
    """Test Le validator rejects invalid comparison."""
    data = pd.DataFrame({"low": [10, 25, 15], "high": [10, 20, 30]})
    validator = Le["low", "high"]
    with pytest.raises(ValueError, match="low must be <= high"):
      validator.validate(data)

  def test_class_getitem(self):
    """Test Le __class_getitem__."""
    validator = Le["col1", "col2"]
    assert validator.targets == ("col1", "col2")


class TestGt:
  """Tests for Gt (>) validator."""

  def test_valid_comparison(self):
    """Test Gt validator with valid column comparison."""
    data = pd.DataFrame({"high": [20, 30, 40], "low": [10, 20, 30]})
    validator = Gt["high", "low"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_equal_values_rejected(self):
    """Test Gt validator rejects equal values."""
    data = pd.DataFrame({"high": [10, 10, 10], "low": [10, 10, 10]})
    validator = Gt["high", "low"]
    with pytest.raises(ValueError, match="high must be > low"):
      validator.validate(data)

  def test_invalid_comparison(self):
    """Test Gt validator rejects invalid comparison."""
    data = pd.DataFrame({"high": [10, 15, 30], "low": [10, 20, 15]})
    validator = Gt["high", "low"]
    with pytest.raises(ValueError, match="high must be > low"):
      validator.validate(data)

  def test_class_getitem(self):
    """Test Gt __class_getitem__."""
    validator = Gt["col1", "col2"]
    assert validator.targets == ("col1", "col2")


class TestLt:
  """Tests for Lt (<) validator."""

  def test_valid_comparison(self):
    """Test Lt validator with valid column comparison."""
    data = pd.DataFrame({"low": [10, 20, 30], "high": [20, 30, 40]})
    validator = Lt["low", "high"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_equal_values_rejected(self):
    """Test Lt validator rejects equal values."""
    data = pd.DataFrame({"low": [10, 10, 10], "high": [10, 10, 10]})
    validator = Lt["low", "high"]
    with pytest.raises(ValueError, match="low must be < high"):
      validator.validate(data)

  def test_invalid_comparison(self):
    """Test Lt validator rejects invalid comparison."""
    data = pd.DataFrame({"low": [15, 20, 30], "high": [10, 30, 25]})
    validator = Lt["low", "high"]
    with pytest.raises(ValueError, match="low must be < high"):
      validator.validate(data)

  def test_class_getitem(self):
    """Test Lt __class_getitem__."""
    validator = Lt["col1", "col2"]
    assert validator.targets == ("col1", "col2")


class TestMonoUp:
  """Tests for MonoUp (monotonically increasing) validator."""

  def test_valid_series_increasing(self):
    """Test MonoUp validator with valid increasing Series."""
    data = pd.Series([1, 2, 3, 4, 5])
    validator = MonoUp()
    result = validator.validate(data)
    assert result.equals(data)

  def test_valid_series_equal(self):
    """Test MonoUp validator allows equal consecutive values."""
    data = pd.Series([1, 2, 2, 3, 3, 4])
    validator = MonoUp()
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_series_decreasing(self):
    """Test MonoUp validator rejects decreasing values."""
    data = pd.Series([1, 2, 3, 2, 5])
    validator = MonoUp()
    with pytest.raises(ValueError, match="must be monotonically increasing"):
      validator.validate(data)

  def test_valid_dataframe(self):
    """Test MonoUp validator with DataFrame."""
    data = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    validator = MonoUp()
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_dataframe(self):
    """Test MonoUp validator rejects DataFrame with non-monotonic column."""
    data = pd.DataFrame({"a": [1, 2, 3], "b": [10, 5, 30]})
    validator = MonoUp()
    with pytest.raises(ValueError, match="Column 'b' values must be monotonically"):
      validator.validate(data)


class TestMonoDown:
  """Tests for MonoDown (monotonically decreasing) validator."""

  def test_valid_series_decreasing(self):
    """Test MonoDown validator with valid decreasing Series."""
    data = pd.Series([5, 4, 3, 2, 1])
    validator = MonoDown()
    result = validator.validate(data)
    assert result.equals(data)

  def test_valid_series_equal(self):
    """Test MonoDown validator allows equal consecutive values."""
    data = pd.Series([5, 4, 4, 3, 3, 2])
    validator = MonoDown()
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_series_increasing(self):
    """Test MonoDown validator rejects increasing values."""
    data = pd.Series([5, 4, 3, 4, 1])
    validator = MonoDown()
    with pytest.raises(ValueError, match="must be monotonically decreasing"):
      validator.validate(data)

  def test_valid_dataframe(self):
    """Test MonoDown validator with DataFrame."""
    data = pd.DataFrame({"a": [3, 2, 1], "b": [30, 20, 10]})
    validator = MonoDown()
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_dataframe(self):
    """Test MonoDown validator rejects DataFrame with non-monotonic column."""
    data = pd.DataFrame({"a": [3, 2, 1], "b": [30, 35, 10]})
    validator = MonoDown()
    with pytest.raises(ValueError, match="Column 'b' values must be monotonically"):
      validator.validate(data)


class TestHasColumn:
  """Tests for HasColumn wrapper validator."""

  def test_single_validator(self):
    """Test HasColumn with single validator."""
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    validator = HasColumn[Literal["a"], Finite]
    result = validator.validate(data)
    assert result.equals(data)

  def test_single_validator_fails(self):
    """Test HasColumn validator fails when column violates constraint."""
    data = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, 6.0]})
    validator = HasColumn[Literal["a"], Finite]
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(data)

  def test_multiple_validators(self):
    """Test HasColumn with multiple validators."""
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    validator = HasColumn[Literal["a"], Finite, Positive]
    result = validator.validate(data)
    assert result.equals(data)

  def test_multiple_validators_fails(self):
    """Test HasColumn with multiple validators where one fails."""
    data = pd.DataFrame({"a": [1.0, 0.0, 3.0], "b": [4.0, 5.0, 6.0]})
    validator = HasColumn[Literal["a"], Finite, Positive]
    with pytest.raises(ValueError, match="must be positive"):
      validator.validate(data)

  def test_missing_column(self):
    """Test HasColumn with missing column."""
    data = pd.DataFrame({"b": [1.0, 2.0, 3.0]})
    validator = HasColumn[Literal["a"], Finite]
    with pytest.raises(ValueError, match="Column 'a' not found"):
      validator.validate(data)

  def test_monotonic_validator(self):
    """Test HasColumn with MonoUp validator."""
    data = pd.DataFrame({"a": [1, 2, 3], "b": [10, 5, 3]})
    validator = HasColumn[Literal["a"], MonoUp]
    result = validator.validate(data)
    assert result.equals(data)

    # Column b is not monotonic up
    validator_b = HasColumn[Literal["b"], MonoUp]
    with pytest.raises(ValueError, match="must be monotonically increasing"):
      validator_b.validate(data)

  def test_class_getitem(self):
    """Test HasColumn __class_getitem__."""
    validator = HasColumn[Literal["col"], Finite, Positive]
    assert validator.column == "col"
    assert len(validator.validators) == 2

  def test_class_getitem_no_validators(self):
    """Test HasColumn __class_getitem__ with no validators."""
    # HasColumn[Literal["col"]] just checks column existence
    validator = HasColumn[Literal["col"]]
    assert validator.column == "col"
    assert len(validator.validators) == 0

  def test_column_presence_only(self):
    """Test HasColumn just checks column presence when no validators."""
    data = pd.DataFrame({"a": [1.0, np.inf, -5.0], "b": [4.0, 5.0, 6.0]})

    # Should pass - column exists (even with invalid values)
    validator = HasColumn[Literal["a"]]
    result = validator.validate(data)
    assert result.equals(data)

    # Should fail - column doesn't exist
    validator_missing = HasColumn[Literal["missing"]]
    with pytest.raises(ValueError, match="Column 'missing' not found"):
      validator_missing.validate(data)


class TestHasColumnWithDecorator:
  """Tests for HasColumn used with @validated decorator."""

  def test_single_column_validation(self):
    """Test @validated with HasColumn for single column."""

    @validated
    def process(
      data: Validated[pd.DataFrame, HasColumn[Literal["price"], Finite, Positive]],
    ):
      return data["price"].sum()

    valid_data = pd.DataFrame({"price": [100.0, 200.0, 150.0], "volume": [10, 20, 15]})
    result = process(valid_data)
    assert result == 450.0

    # Fails Finite check
    invalid_data = pd.DataFrame({
      "price": [100.0, np.inf, 150.0],
      "volume": [10, 20, 15],
    })
    with pytest.raises(ValueError, match="must be finite"):
      process(invalid_data)

    # Fails Positive check
    invalid_data = pd.DataFrame({"price": [100.0, 0.0, 150.0], "volume": [10, 20, 15]})
    with pytest.raises(ValueError, match="must be positive"):
      process(invalid_data)

  def test_multiple_column_validation(self):
    """Test @validated with multiple HasColumn validators."""

    @validated
    def process(
      data: Validated[
        pd.DataFrame,
        HasColumn[Literal["price"], Finite, Positive],
        HasColumn[Literal["volume"], Positive],
      ],
    ):
      return (data["price"] * data["volume"]).sum()

    valid_data = pd.DataFrame({"price": [100.0, 200.0, 150.0], "volume": [10, 20, 15]})
    result = process(valid_data)
    assert result == 100 * 10 + 200 * 20 + 150 * 15

    # Fails volume Positive check
    invalid_data = pd.DataFrame({
      "price": [100.0, 200.0, 150.0],
      "volume": [10, -5, 15],
    })
    with pytest.raises(ValueError, match="must be positive"):
      process(invalid_data)

  def test_oncolumn_with_monotonic(self):
    """Test HasColumn with MonoUp validator."""

    @validated
    def process(
      data: Validated[pd.DataFrame, HasColumn[Literal["timestamp"], MonoUp]],
    ):
      return len(data)

    valid_data = pd.DataFrame({
      "timestamp": [1, 2, 3, 4, 5],
      "value": [10, 20, 30, 40, 50],
    })
    result = process(valid_data)
    assert result == 5

    # Fails MonoUp check
    invalid_data = pd.DataFrame({
      "timestamp": [1, 2, 5, 4, 3],
      "value": [10, 20, 30, 40, 50],
    })
    with pytest.raises(ValueError, match="must be monotonically increasing"):
      process(invalid_data)


class TestCombinedValidators:
  """Tests combining new validators with existing ones."""

  def test_comparison_validators_combined(self):
    """Test using multiple comparison validators."""

    @validated
    def process(
      data: Validated[
        pd.DataFrame, Gt[Literal["high", "low"]], Le[Literal["low", "close"]]
      ],
    ):
      return len(data)

    valid_data = pd.DataFrame({
      "high": [105, 110, 108],
      "low": [100, 105, 103],
      "close": [102, 107, 105],
    })
    result = process(valid_data)
    assert result == 3

    # Fails Gt check (high not > low)
    invalid_data = pd.DataFrame({
      "high": [100, 110, 108],
      "low": [100, 105, 103],
      "close": [102, 107, 105],
    })
    with pytest.raises(ValueError, match="high must be > low"):
      process(invalid_data)

  def test_oncolumn_with_nonnan(self):
    """Test HasColumn with NonNaN validator."""

    @validated
    def process(
      data: Validated[pd.DataFrame, HasColumn[Literal["price"], NonNaN, Positive]],
    ):
      return data["price"].mean()

    valid_data = pd.DataFrame({"price": [100.0, 200.0, 150.0]})
    result = process(valid_data)
    assert result == 150.0

    invalid_data = pd.DataFrame({"price": [100.0, np.nan, 150.0]})
    with pytest.raises(ValueError, match="must not contain NaN"):
      process(invalid_data)


class TestUniqueIndex:
  def test_unique_index_series(self):
    validator = UniqueIndex()

    # Valid
    data = pd.Series([1, 2], index=[1, 2])
    assert validator.validate(data) is data

    # Invalid
    data = pd.Series([1, 2], index=[1, 1])
    with pytest.raises(ValueError, match="Index must be unique"):
      validator.validate(data)

  def test_unique_index_dataframe(self):
    validator = UniqueIndex()

    # Valid
    data = pd.DataFrame({"a": [1, 2]}, index=pd.Index([1, 2]))
    assert validator.validate(data) is data

    # Invalid
    data = pd.DataFrame({"a": [1, 2]}, index=pd.Index([1, 1]))
    with pytest.raises(ValueError, match="Index must be unique"):
      validator.validate(data)


class TestNoTimeGaps:
  def test_no_time_gaps(self):
    validator = NoTimeGaps(freq="1D")

    # Valid
    dates = pd.date_range("2024-01-01", periods=3, freq="1D")
    data = pd.Series([1, 2, 3], index=dates)
    assert validator.validate(data) is data

    # Invalid (missing day)
    dates = pd.to_datetime(["2024-01-01", "2024-01-03"])
    data = pd.Series([1, 2], index=dates)
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)

  def test_empty_index(self):
    validator = NoTimeGaps(freq="1D")
    data = pd.Series([], index=pd.DatetimeIndex([]))
    assert validator.validate(data) is data


class TestIsDtype:
  def test_is_dtype_series(self):
    validator = IsDtype(float)

    # Valid
    data = pd.Series([1.0, 2.0])
    assert validator.validate(data) is data

    # Invalid
    data = pd.Series([1, 2])
    with pytest.raises(ValueError, match="Data must be of type"):
      validator.validate(data)

  def test_is_dtype_dataframe(self):
    validator = IsDtype(float)

    # Valid
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert validator.validate(data) is data

    # Invalid
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [1, 2]})
    with pytest.raises(ValueError, match="Columns with wrong dtype"):
      validator.validate(data)

  def test_numpy_dtype(self):
    validator = IsDtype(np.float64)
    data = pd.Series([1.0, 2.0], dtype=np.float64)
    assert validator.validate(data) is data


def test_integration():
  @validated
  def process(data: Validated[pd.Series, UniqueIndex, IsDtype(float)]):
    return data

  # Valid
  data = pd.Series([1.0, 2.0], index=[1, 2])
  process(data)

  # Invalid Index
  data = pd.Series([1.0, 2.0], index=[1, 1])
  with pytest.raises(ValueError, match="Index must be unique"):
    process(data)

  # Invalid Dtype
  data = pd.Series([1, 2], index=[1, 2])
  with pytest.raises(ValueError, match="Data must be of type"):
    process(data)


def test_validated_decorator_skip_validation():
  @validated
  def process(data: Validated[pd.Series, Finite]):
    return data

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


class TestNonEmpty:
  """Tests for NonEmpty validator."""

  def test_valid_series(self):
    """Test NonEmpty validator with valid Series."""
    data = pd.Series([1.0, 2.0, 3.0])
    validator = NonEmpty()
    result = validator.validate(data)
    assert result.equals(data)

  def test_empty_series(self):
    """Test NonEmpty validator rejects empty Series."""
    data = pd.Series([], dtype=float)
    validator = NonEmpty()
    with pytest.raises(ValueError, match="Data must not be empty"):
      validator.validate(data)

  def test_valid_dataframe(self):
    """Test NonEmpty validator with valid DataFrame."""
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    validator = NonEmpty()
    result = validator.validate(data)
    assert result.equals(data)

  def test_empty_dataframe(self):
    """Test NonEmpty validator rejects empty DataFrame."""
    data = pd.DataFrame({"a": [], "b": []})
    validator = NonEmpty()
    with pytest.raises(ValueError, match="Data must not be empty"):
      validator.validate(data)

  def test_valid_index(self):
    """Test NonEmpty validator with valid Index."""
    data = pd.Index([1, 2, 3])
    validator = NonEmpty()
    result = validator.validate(data)
    assert result.equals(data)

  def test_empty_index(self):
    """Test NonEmpty validator rejects empty Index."""
    data = pd.Index([], dtype=int)
    validator = NonEmpty()
    with pytest.raises(ValueError, match="Data must not be empty"):
      validator.validate(data)

  def test_non_pandas_type(self):
    """Test NonEmpty validator with non-pandas type."""
    validator = NonEmpty()
    result = validator.validate(42)
    assert result == 42
