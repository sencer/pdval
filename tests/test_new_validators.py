"""Tests for new validators: Le, Gt, Lt, MonoUp, MonoDown, HasColumn."""

import numpy as np
import pandas as pd
import pytest

from pdval import (
  Finite,
  Gt,
  HasColumn,
  Le,
  Lt,
  MonoDown,
  MonoUp,
  NonNaN,
  Positive,
  Validated,
  validated,
)


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
    assert validator.col1 == "col1"
    assert validator.col2 == "col2"


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
    assert validator.col1 == "col1"
    assert validator.col2 == "col2"


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
    assert validator.col1 == "col1"
    assert validator.col2 == "col2"


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
    validator = HasColumn["a", Finite]
    result = validator.validate(data)
    assert result.equals(data)

  def test_single_validator_fails(self):
    """Test HasColumn validator fails when column violates constraint."""
    data = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, 6.0]})
    validator = HasColumn["a", Finite]
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(data)

  def test_multiple_validators(self):
    """Test HasColumn with multiple validators."""
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    validator = HasColumn["a", Finite, Positive]
    result = validator.validate(data)
    assert result.equals(data)

  def test_multiple_validators_fails(self):
    """Test HasColumn with multiple validators where one fails."""
    data = pd.DataFrame({"a": [1.0, 0.0, 3.0], "b": [4.0, 5.0, 6.0]})
    validator = HasColumn["a", Finite, Positive]
    with pytest.raises(ValueError, match="must be positive"):
      validator.validate(data)

  def test_missing_column(self):
    """Test HasColumn with missing column."""
    data = pd.DataFrame({"b": [1.0, 2.0, 3.0]})
    validator = HasColumn["a", Finite]
    with pytest.raises(ValueError, match="Column 'a' not found"):
      validator.validate(data)

  def test_monotonic_validator(self):
    """Test HasColumn with MonoUp validator."""
    data = pd.DataFrame({"a": [1, 2, 3], "b": [10, 5, 3]})
    validator = HasColumn["a", MonoUp]
    result = validator.validate(data)
    assert result.equals(data)

    # Column b is not monotonic up
    validator_b = HasColumn["b", MonoUp]
    with pytest.raises(ValueError, match="must be monotonically increasing"):
      validator_b.validate(data)

  def test_class_getitem(self):
    """Test HasColumn __class_getitem__."""
    validator = HasColumn["col", Finite, Positive]
    assert validator.column == "col"
    assert len(validator.validators) == 2

  def test_class_getitem_no_validators(self):
    """Test HasColumn __class_getitem__ with no validators."""
    # HasColumn["col"] just checks column existence
    validator = HasColumn["col"]
    assert validator.column == "col"
    assert len(validator.validators) == 0

  def test_column_presence_only(self):
    """Test HasColumn just checks column presence when no validators."""
    data = pd.DataFrame({"a": [1.0, np.inf, -5.0], "b": [4.0, 5.0, 6.0]})

    # Should pass - column exists (even with invalid values)
    validator = HasColumn["a"]
    result = validator.validate(data)
    assert result.equals(data)

    # Should fail - column doesn't exist
    validator_missing = HasColumn["missing"]
    with pytest.raises(ValueError, match="Column 'missing' not found"):
      validator_missing.validate(data)


class TestHasColumnWithDecorator:
  """Tests for HasColumn used with @validated decorator."""

  def test_single_column_validation(self):
    """Test @validated with HasColumn for single column."""

    @validated
    def process(
      data: Validated[pd.DataFrame, HasColumn["price", Finite, Positive]],  # noqa: F821
      validate: bool = True,
    ):
      return data["price"].sum()

    valid_data = pd.DataFrame({"price": [100.0, 200.0, 150.0], "volume": [10, 20, 15]})
    result = process(valid_data)
    assert result == 450.0

    # Fails Finite check
    invalid_data = pd.DataFrame(
      {
        "price": [100.0, np.inf, 150.0],
        "volume": [10, 20, 15],
      }
    )
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
        HasColumn["price", Finite, Positive],  # noqa: F821
        HasColumn["volume", Positive],  # noqa: F821
      ],
      validate: bool = True,
    ):
      return (data["price"] * data["volume"]).sum()

    valid_data = pd.DataFrame({"price": [100.0, 200.0, 150.0], "volume": [10, 20, 15]})
    result = process(valid_data)
    assert result == 100 * 10 + 200 * 20 + 150 * 15

    # Fails volume Positive check
    invalid_data = pd.DataFrame(
      {
        "price": [100.0, 200.0, 150.0],
        "volume": [10, -5, 15],
      }
    )
    with pytest.raises(ValueError, match="must be positive"):
      process(invalid_data)

  def test_oncolumn_with_monotonic(self):
    """Test HasColumn with MonoUp validator."""

    @validated
    def process(
      data: Validated[pd.DataFrame, HasColumn["timestamp", MonoUp]],  # noqa: F821
      validate: bool = True,
    ):
      return len(data)

    valid_data = pd.DataFrame(
      {
        "timestamp": [1, 2, 3, 4, 5],
        "value": [10, 20, 30, 40, 50],
      }
    )
    result = process(valid_data)
    assert result == 5

    # Fails MonoUp check
    invalid_data = pd.DataFrame(
      {
        "timestamp": [1, 2, 5, 4, 3],
        "value": [10, 20, 30, 40, 50],
      }
    )
    with pytest.raises(ValueError, match="must be monotonically increasing"):
      process(invalid_data)


class TestCombinedValidators:
  """Tests combining new validators with existing ones."""

  def test_comparison_validators_combined(self):
    """Test using multiple comparison validators."""

    @validated
    def process(
      data: Validated[pd.DataFrame, Gt["high", "low"], Le["low", "close"]],  # noqa: F821
      validate: bool = True,
    ):
      return len(data)

    valid_data = pd.DataFrame(
      {
        "high": [105, 110, 108],
        "low": [100, 105, 103],
        "close": [102, 107, 105],
      }
    )
    result = process(valid_data)
    assert result == 3

    # Fails Gt check (high not > low)
    invalid_data = pd.DataFrame(
      {
        "high": [100, 110, 108],
        "low": [100, 105, 103],
        "close": [102, 107, 105],
      }
    )
    with pytest.raises(ValueError, match="high must be > low"):
      process(invalid_data)

  def test_oncolumn_with_nonnan(self):
    """Test HasColumn with NonNaN validator."""

    @validated
    def process(
      data: Validated[pd.DataFrame, HasColumn["price", NonNaN, Positive]],  # noqa: F821
      validate: bool = True,
    ):
      return data["price"].mean()

    valid_data = pd.DataFrame({"price": [100.0, 200.0, 150.0]})
    result = process(valid_data)
    assert result == 150.0

    invalid_data = pd.DataFrame({"price": [100.0, np.nan, 150.0]})
    with pytest.raises(ValueError, match="must not contain NaN"):
      process(invalid_data)
