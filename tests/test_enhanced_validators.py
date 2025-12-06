"""Tests for enhanced validator features."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportArgumentType=false

from typing import TypeVar

import pandas as pd
import pytest

from pdval import (
  Finite,
  Ge,
  HasColumn,
  HasColumns,
  Lt,
  Positive,
  Validated,
  validated,
)


class TestUnaryComparisons:
  """Tests for unary comparison validators (Ge[100], etc.)."""

  def test_ge_unary_series(self):
    """Test Ge[100] on Series."""
    data = pd.Series([100, 101, 200])
    validator = Ge[100]
    result = validator.validate(data)
    assert result.equals(data)

    invalid_data = pd.Series([99, 100, 200])
    with pytest.raises(ValueError, match="Data must be >= 100"):
      validator.validate(invalid_data)

  def test_ge_unary_dataframe(self):
    """Test Ge[100] on DataFrame."""
    data = pd.DataFrame({"a": [100, 200], "b": [150, 300]})
    validator = Ge[100]
    result = validator.validate(data)
    assert result.equals(data)

    invalid_data = pd.DataFrame({"a": [99, 200], "b": [150, 300]})
    with pytest.raises(ValueError, match="Data must be >= 100"):
      validator.validate(invalid_data)

  def test_lt_unary_series(self):
    """Test Lt[0] on Series."""
    data = pd.Series([-1, -2, -3])
    validator = Lt[0]
    result = validator.validate(data)
    assert result.equals(data)

    invalid_data = pd.Series([-1, 0, -3])
    with pytest.raises(ValueError, match="Data must be < 0"):
      validator.validate(invalid_data)


class TestChainedComparisons:
  """Tests for chained column comparisons (Ge["a", "b", "c"])."""

  def test_ge_chained(self):
    """Test Ge["high", "close", "low"]."""
    data = pd.DataFrame({
      "high": [100, 200, 300],
      "close": [90, 190, 290],
      "low": [80, 180, 280],
    })
    validator = Ge["high", "close", "low"]
    result = validator.validate(data)
    assert result.equals(data)

    # Fail high >= close
    invalid_data = pd.DataFrame({
      "high": [80, 200, 300],
      "close": [90, 190, 290],
      "low": [80, 180, 280],
    })
    with pytest.raises(ValueError, match="high must be >= close"):
      validator.validate(invalid_data)

    # Fail close >= low
    invalid_data_2 = pd.DataFrame({
      "high": [100, 200, 300],
      "close": [90, 190, 290],
      "low": [95, 180, 280],
    })
    with pytest.raises(ValueError, match="close must be >= low"):
      validator.validate(invalid_data_2)

  def test_lt_chained(self):
    """Test Lt["low", "close", "high"]."""
    # low < close < high
    data = pd.DataFrame({
      "low": [80, 180, 280],
      "close": [90, 190, 290],
      "high": [100, 200, 300],
    })
    validator = Lt["low", "close", "high"]
    result = validator.validate(data)
    assert result.equals(data)


class TestHasColumnsEnhanced:
  """Tests for HasColumns with validators."""

  def test_has_columns_with_validators(self):
    """Test HasColumns["a", "b", Positive]."""
    data = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [-1, -2, -3]})
    validator = HasColumns["a", "b", Positive]
    result = validator.validate(data)
    assert result.equals(data)

    # Fail Positive on 'a'
    invalid_data = pd.DataFrame({"a": [0, 2, 3], "b": [10, 20, 30]})
    with pytest.raises(ValueError, match="must be positive"):
      validator.validate(invalid_data)

  def test_has_columns_multiple_validators(self):
    """Test HasColumns["a", Finite, Positive]."""
    data = pd.DataFrame({"a": [1.0, 2.0]})
    validator = HasColumns["a", Finite, Positive]
    result = validator.validate(data)
    assert result.equals(data)


class TestHasColumnTemplating:
  """Tests for HasColumn templating."""

  def test_templating(self):
    """Test CustomVal[T] = HasColumn[T, ...]."""
    T = TypeVar("T")
    custom_val = HasColumn[T, Positive]

    # Instantiate for specific column
    validator_a = custom_val["a"]

    assert validator_a.column == "a"
    assert len(validator_a.validators) == 1
    v = validator_a.validators[0]
    # HasColumn stores validators as passed (can be class or instance)
    assert v is Positive or isinstance(v, Positive)

    data = pd.DataFrame({"a": [1, 2, 3]})
    result = validator_a.validate(data)
    assert result.equals(data)

    invalid_data = pd.DataFrame({"a": [-1, 2, 3]})
    with pytest.raises(ValueError, match="must be positive"):
      validator_a.validate(invalid_data)

  def test_templating_decorator(self):
    """Test templating usage in @validated."""
    T = TypeVar("T")
    positive_col = HasColumn[T, Positive]

    @validated
    def process(data: Validated[pd.DataFrame, positive_col["price"]]):  # noqa: F821
      return data["price"].sum()

    valid_data = pd.DataFrame({"price": [10, 20]})
    assert process(valid_data) == 30

    invalid_data = pd.DataFrame({"price": [-10, 20]})
    with pytest.raises(ValueError, match="must be positive"):
      process(invalid_data)
