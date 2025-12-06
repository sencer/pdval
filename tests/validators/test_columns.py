"""Tests for column validators: IsDtype, HasColumns, HasColumn."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from pdval import (
  Finite,
  HasColumn,
  HasColumns,
  IsDtype,
  MaybeEmpty,
  MonoUp,
  Nullable,
  Positive,
)


class TestIsDtype:
  """Tests for IsDtype validator."""

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

  def test_dataframe_all_columns_match(self):
    """Test IsDtype with DataFrame where all columns match."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validator = IsDtype[np.dtype("int64")]
    result = validator.validate(df)
    assert result.equals(df)

  def test_dataframe_column_mismatch(self):
    """Test IsDtype with DataFrame where one column doesn't match."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    validator = IsDtype[np.dtype("int64")]
    with pytest.raises(ValueError, match="Columns with wrong dtype"):
      validator.validate(df)


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

  def test_hascolumns_applies_validators(self):
    """Test HasColumns applies validators to columns."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validator = HasColumns[Literal["a", "b"], Positive]
    result = validator.validate(df)
    assert result.equals(df)

  def test_hascolumns_validator_fails(self):
    """Test HasColumns fails when column validator fails."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 5, 6]})
    validator = HasColumns[Literal["a", "b"], Positive]
    with pytest.raises(ValueError, match="Data must be positive"):
      validator.validate(df)

  def test_hascolumns_with_nullable(self):
    """Test HasColumns with Nullable marker."""
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
    validator = HasColumns[Literal["a", "b"], Nullable]
    result = validator.validate(df)
    assert result.equals(df)

  def test_hascolumns_with_maybeempty(self):
    """Test HasColumns with MaybeEmpty marker."""
    df = pd.DataFrame({"a": [], "b": []})
    validator = HasColumns[Literal["a", "b"], MaybeEmpty, Nullable]
    result = validator.validate(df)
    assert result.equals(df)

  def test_hascolumns_single_column_string(self):
    """Test HasColumns with single column as string."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    validator = HasColumns["a"]
    result = validator.validate(df)
    assert result.equals(df)

  def test_hascolumns_with_literal_in_tuple(self):
    """Test HasColumns parsing with Literal in tuple."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    validator = HasColumns[Literal["x"], Literal["y"]]
    result = validator.validate(df)
    assert result.equals(df)


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

  def test_hascolumn_with_nullable(self):
    """Test HasColumn with Nullable marker."""
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
    validator = HasColumn[Literal["a"], Nullable]
    result = validator.validate(df)
    assert result.equals(df)

  def test_hascolumn_with_maybeempty(self):
    """Test HasColumn with MaybeEmpty marker."""
    df = pd.DataFrame({"a": [], "b": []})
    validator = HasColumn[Literal["a"], MaybeEmpty, Nullable]
    result = validator.validate(df)
    assert result.equals(df)

  def test_hascolumn_typevar_usage(self):
    """Test HasColumn with direct validator usage."""
    df = pd.DataFrame({"col": [1, 2, 3]})
    validator = HasColumn("col", Positive)
    result = validator.validate(df)
    assert result.equals(df)

  def test_hascolumn_getitem_syntax(self):
    """Test HasColumn with __getitem__ syntax."""
    df = pd.DataFrame({"test": [1, 2, 3]})
    validator = HasColumn("test", Positive)
    result = validator.validate(df)
    assert result.equals(df)

  def test_hascolumn_literal_syntax(self):
    """Test HasColumn with Literal syntax."""
    df = pd.DataFrame({"mycolumn": [1, 2, 3]})
    validator = HasColumn[Literal["mycolumn"], Positive, Finite]
    result = validator.validate(df)
    assert result.equals(df)
