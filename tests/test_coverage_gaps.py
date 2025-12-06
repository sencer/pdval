"""Tests to improve coverage for uncovered code paths."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

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
  MaybeEmpty,
  MonoDown,
  MonoUp,
  NoTimeGaps,
  Nullable,
  Positive,
  UniqueIndex,
  Validated,
  validated,
)


class TestIsDtypeDataFrame:
  """Test IsDtype validator with DataFrames."""

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


class TestComparisonValidatorsEdgeCases:
  """Test edge cases for comparison validators."""

  def test_ge_with_numeric_column_names_fails(self):
    """Test Ge validator with non-string column names raises TypeError."""
    data = pd.DataFrame({"high": [10, 20, 30], "low": [5, 10, 15]})
    # Create validator with numeric targets to trigger TypeError
    validator = Ge(1, 2)  # Non-string column names
    with pytest.raises(
      TypeError, match="Column comparison requires string column names"
    ):
      validator.validate(data)

  def test_le_with_numeric_column_names_fails(self):
    """Test Le validator with non-string column names raises TypeError."""
    data = pd.DataFrame({"high": [10, 20, 30], "low": [5, 10, 15]})
    validator = Le(1, 2)
    with pytest.raises(
      TypeError, match="Column comparison requires string column names"
    ):
      validator.validate(data)

  def test_gt_with_numeric_column_names_fails(self):
    """Test Gt validator with non-string column names raises TypeError."""
    data = pd.DataFrame({"high": [10, 20, 30], "low": [5, 10, 15]})
    validator = Gt(1, 2)
    with pytest.raises(
      TypeError, match="Column comparison requires string column names"
    ):
      validator.validate(data)

  def test_lt_with_numeric_column_names_fails(self):
    """Test Lt validator with non-string column names raises TypeError."""
    data = pd.DataFrame({"high": [10, 20, 30], "low": [5, 10, 15]})
    validator = Lt(1, 2)
    with pytest.raises(
      TypeError, match="Column comparison requires string column names"
    ):
      validator.validate(data)


class TestNoTimeGapsEdgeCases:
  """Test NoTimeGaps validator edge cases."""

  def test_notimegaps_with_series(self):
    """Test NoTimeGaps with Series."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    data = pd.Series([1, 2, 3, 4, 5], index=index)
    validator = NoTimeGaps["D"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_notimegaps_with_gaps_in_series(self):
    """Test NoTimeGaps detects gaps in Series."""
    index = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-04"])
    data = pd.Series([1, 2, 3], index=index)
    validator = NoTimeGaps["D"]
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)

  def test_notimegaps_with_non_datetime_index(self):
    """Test NoTimeGaps requires DatetimeIndex."""
    data = pd.Series([1, 2, 3], index=[0, 1, 2])
    validator = NoTimeGaps["D"]
    with pytest.raises(ValueError, match="NoTimeGaps requires a DatetimeIndex"):
      validator.validate(data)

  def test_notimegaps_empty_series(self):
    """Test NoTimeGaps with empty Series."""
    data = pd.Series([], dtype="float64", index=pd.DatetimeIndex([]))
    validator = NoTimeGaps["D"]
    result = validator.validate(data)
    assert len(result) == 0

  def test_notimegaps_with_dataframe(self):
    """Test NoTimeGaps with DataFrame."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=index)
    validator = NoTimeGaps["D"]
    result = validator.validate(df)
    assert result.equals(df)


class TestHasColumnsWithValidators:
  """Test HasColumns validator with applied validators."""

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


class TestIndexValidator:
  """Test Index validator."""

  def test_index_with_datetime(self):
    """Test Index validator with Datetime."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=index)
    validator = Index[Datetime]
    result = validator.validate(df)
    assert result.equals(df)

  def test_index_with_multiple_validators(self):
    """Test Index validator with multiple validators."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=index)
    validator = Index[Datetime, MonoUp]
    result = validator.validate(df)
    assert result.equals(df)

  def test_index_with_uniqueindex(self):
    """Test Index validator with UniqueIndex."""
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
    validator = Index[UniqueIndex]
    result = validator.validate(df)
    assert result.equals(df)


class TestHasColumnValidation:
  """Test HasColumn validator with validation."""

  def test_hascolumn_applies_validators(self):
    """Test HasColumn applies validators to column."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validator = HasColumn[Literal["a"], Positive]
    result = validator.validate(df)
    assert result.equals(df)

  def test_hascolumn_validator_fails(self):
    """Test HasColumn fails when validator fails."""
    df = pd.DataFrame({"a": [0, 2, 3], "b": [4, 5, 6]})
    validator = HasColumn[Literal["a"], Positive]
    with pytest.raises(ValueError, match="Data must be positive"):
      validator.validate(df)

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

  def test_hascolumn_with_multiple_validators(self):
    """Test HasColumn with multiple validators."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validator = HasColumn[Literal["a"], Positive, Finite]
    result = validator.validate(df)
    assert result.equals(df)


class TestMonoDownDataFrame:
  """Test MonoDown validator with DataFrame."""

  def test_monodown_dataframe_valid(self):
    """Test MonoDown with valid DataFrame."""
    df = pd.DataFrame({"a": [5, 4, 3], "b": [10, 8, 6]})
    validator = MonoDown()
    result = validator.validate(df)
    assert result.equals(df)

  def test_monodown_dataframe_invalid(self):
    """Test MonoDown with invalid DataFrame."""
    df = pd.DataFrame({"a": [5, 4, 6], "b": [10, 8, 6]})
    validator = MonoDown()
    with pytest.raises(
      ValueError, match="Column 'a' values must be monotonically decreasing"
    ):
      validator.validate(df)


class TestValidatedDecoratorPandasTypes:
  """Test validated decorator with pandas types."""

  def test_validated_with_dataframe_type(self):
    """Test validated decorator recognizes DataFrame type."""

    @validated
    def process(data: Validated[pd.DataFrame, Finite]) -> float:
      return data.sum().sum()

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = process(df)
    assert result == 21

  def test_validated_with_series_type(self):
    """Test validated decorator recognizes Series type."""

    @validated
    def process(data: Validated[pd.Series, Finite]) -> float:
      return data.sum()

    series = pd.Series([1, 2, 3, 4, 5])
    result = process(series)
    assert result == 15


class TestUniqueIndexOnIndex:
  """Test UniqueIndex validator on Index objects."""

  def test_uniqueindex_on_index_object(self):
    """Test UniqueIndex validator on Index object."""
    index = pd.Index([1, 2, 3, 4, 5])
    validator = UniqueIndex()
    result = validator.validate(index)
    assert result.equals(index)

  def test_uniqueindex_on_non_unique_index(self):
    """Test UniqueIndex fails on non-unique Index object."""
    index = pd.Index([1, 2, 2, 4, 5])
    validator = UniqueIndex()
    with pytest.raises(ValueError, match="Index must be unique"):
      validator.validate(index)


class TestComparisonValidatorsMoreEdgeCases:
  """Test more edge cases for Le, Gt, Lt validators."""

  def test_le_unary_with_series(self):
    """Test Le validator with unary comparison on Series."""
    data = pd.Series([1, 2, 3])
    validator = Le[5]
    result = validator.validate(data)
    assert result.equals(data)

  def test_le_unary_fails(self):
    """Test Le validator fails with unary comparison."""
    data = pd.Series([1, 2, 6])
    validator = Le[5]
    with pytest.raises(ValueError, match="Data must be <= 5"):
      validator.validate(data)

  def test_gt_unary_with_series(self):
    """Test Gt validator with unary comparison on Series."""
    data = pd.Series([2, 3, 4])
    validator = Gt[1]
    result = validator.validate(data)
    assert result.equals(data)

  def test_gt_unary_fails(self):
    """Test Gt validator fails with unary comparison."""
    data = pd.Series([1, 2, 3])
    validator = Gt[1]
    with pytest.raises(ValueError, match="Data must be > 1"):
      validator.validate(data)

  def test_lt_unary_with_series(self):
    """Test Lt validator with unary comparison on Series."""
    data = pd.Series([1, 2, 3])
    validator = Lt[5]
    result = validator.validate(data)
    assert result.equals(data)

  def test_lt_unary_fails(self):
    """Test Lt validator fails with unary comparison."""
    data = pd.Series([1, 2, 5])
    validator = Lt[5]
    with pytest.raises(ValueError, match="Data must be < 5"):
      validator.validate(data)


class TestIndexValidatorEdgeCases:
  """Test Index validator edge cases."""

  def test_index_with_non_dataframe(self):
    """Test Index validator with Index object (no-op)."""
    index = pd.Index([1, 2, 3, 4, 5])
    validator = Index[UniqueIndex]
    result = validator.validate(index)
    assert result.equals(index)

  def test_index_with_validator_instance(self):
    """Test Index validator with validator instance."""
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
    validator = Index(UniqueIndex())
    result = validator.validate(df)
    assert result.equals(df)


class TestNoTimeGapsLengthMismatch:
  """Test NoTimeGaps detects length mismatches."""

  def test_notimegaps_wrong_number_of_entries(self):
    """Test NoTimeGaps when index has wrong number of entries."""
    # Create index with only 3 days but should have 5
    index = pd.DatetimeIndex(["2023-01-01", "2023-01-03", "2023-01-05"])
    data = pd.Series([1, 2, 3], index=index)
    validator = NoTimeGaps["D"]
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)

  def test_notimegaps_correct_length_wrong_dates(self):
    """Test NoTimeGaps when we have correct length but wrong timestamps.

    This is the edge case where length check passes but difference check fails.
    This happens when we have duplicate timestamps!
    """
    # Create an index with duplicates:
    # Index: [2023-01-01, 2023-01-01, 2023-01-03] (3 entries)
    # Min: 2023-01-01, Max: 2023-01-03
    # Expected range: [2023-01-01, 2023-01-02, 2023-01-03] (3 entries)
    # Length check passes: len(3) == len(3)
    # But difference check fails because 2023-01-02 is missing from index
    index = pd.DatetimeIndex([
      "2023-01-01",
      "2023-01-01",  # Duplicate!
      "2023-01-03",
    ])
    data = pd.Series([1, 2, 3], index=index)
    validator = NoTimeGaps["D"]
    # This should fail at the difference check (line 176)
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)

  def test_notimegaps_with_index_object_directly(self):
    """Test NoTimeGaps with pd.Index object directly (not Series/DataFrame)."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    validator = NoTimeGaps["D"]
    result = validator.validate(index)
    assert result.equals(index)

  def test_notimegaps_with_index_object_with_gaps(self):
    """Test NoTimeGaps detects gaps in pd.Index object."""
    index = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-04"])
    validator = NoTimeGaps["D"]
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(index)


class TestHasColumnsParsingEdgeCases:
  """Test HasColumns parsing edge cases."""

  def test_hascolumns_single_column_string(self):
    """Test HasColumns with single column as string."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    validator = HasColumns["a"]
    result = validator.validate(df)
    assert result.equals(df)

  def test_hascolumns_with_literal_in_tuple(self):
    """Test HasColumns parsing with Literal in tuple."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    # Using the validator through annotation
    validator = HasColumns[Literal["x"], Literal["y"]]
    result = validator.validate(df)
    assert result.equals(df)


class TestHasColumnEdgeCases:
  """Test HasColumn edge cases."""

  def test_hascolumn_typevar_usage(self):
    """Test HasColumn with direct validator usage."""
    df = pd.DataFrame({"col": [1, 2, 3]})

    # Use string directly
    validator = HasColumn("col", Positive)
    result = validator.validate(df)
    assert result.equals(df)

  def test_hascolumn_getitem_syntax(self):
    """Test HasColumn with __getitem__ syntax."""
    df = pd.DataFrame({"test": [1, 2, 3]})
    # Create a validator with column and use getitem to change column
    validator = HasColumn("test", Positive)
    result = validator.validate(df)
    assert result.equals(df)

  def test_hascolumn_literal_syntax(self):
    """Test HasColumn with Literal syntax."""
    df = pd.DataFrame({"mycolumn": [1, 2, 3]})
    validator = HasColumn[Literal["mycolumn"], Positive, Finite]
    result = validator.validate(df)
    assert result.equals(df)
