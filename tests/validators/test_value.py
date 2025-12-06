"""Tests for value validators: Finite, NonEmpty, NonNaN, NonNegative, Positive, OneOf."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from pdval import Finite, Index, NonEmpty, NonNaN, NonNegative, OneOf, Positive


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
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.inf]})
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


class TestOneOf:
  """Tests for OneOf validator."""

  def test_valid_series_with_strings(self):
    """Test OneOf validator with valid string Series."""
    data = pd.Series(["a", "b", "a", "c"])
    validator = OneOf["a", "b", "c"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_series_with_strings(self):
    """Test OneOf validator rejects invalid values."""
    data = pd.Series(["a", "b", "d"])
    validator = OneOf["a", "b", "c"]
    with pytest.raises(ValueError, match="Values must be one of"):
      validator.validate(data)

  def test_literal_syntax(self):
    """Test OneOf with Literal["a", "b", "c"] syntax."""
    data = pd.Series(["a", "b", "c"])
    validator = OneOf[Literal["a", "b", "c"]]
    result = validator.validate(data)
    assert result.equals(data)

  def test_mixed_literal_syntax(self):
    """Test OneOf with OneOf[Literal["a"], Literal["b"], Literal["c"]] syntax."""
    data = pd.Series(["a", "b", "c"])
    validator = OneOf[Literal["a"], Literal["b"], Literal["c"]]
    result = validator.validate(data)
    assert result.equals(data)

  def test_numeric_values(self):
    """Test OneOf with numeric values."""
    data = pd.Series([1, 2, 1, 3])
    validator = OneOf[1, 2, 3]
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_numeric_values(self):
    """Test OneOf rejects invalid numeric values."""
    data = pd.Series([1, 2, 4])
    validator = OneOf[1, 2, 3]
    with pytest.raises(ValueError, match="Values must be one of"):
      validator.validate(data)

  def test_with_nan_values(self):
    """Test OneOf ignores NaN values in Series."""
    data = pd.Series(["a", "b", np.nan, "c"])
    validator = OneOf["a", "b", "c"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_with_index(self):
    """Test OneOf with pd.Index."""
    data = pd.Index(["x", "y", "z"])
    validator = OneOf["x", "y", "z"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_index(self):
    """Test OneOf rejects invalid Index values."""
    data = pd.Index(["x", "y", "w"])
    validator = OneOf["x", "y", "z"]
    with pytest.raises(ValueError, match="Values must be one of"):
      validator.validate(data)

  def test_with_index_wrapper(self):
    """Test OneOf with Index[] wrapper for DataFrame index."""
    df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
    validator = Index[OneOf["x", "y", "z"]]
    result = validator.validate(df)
    assert result.equals(df)

  def test_invalid_with_index_wrapper(self):
    """Test OneOf with Index[] wrapper rejects invalid index."""
    df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "w"])
    validator = Index[OneOf["x", "y", "z"]]
    with pytest.raises(ValueError, match="Values must be one of"):
      validator.validate(df)

  def test_single_value(self):
    """Test OneOf with single allowed value."""
    data = pd.Series(["only"])
    validator = OneOf["only"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_constructor_syntax(self):
    """Test OneOf with constructor syntax."""
    data = pd.Series(["a", "b"])
    validator = OneOf("a", "b", "c")
    result = validator.validate(data)
    assert result.equals(data)
