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
  MaxDiff,
  MaxGap,
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

  def test_notimegaps_with_datetime_series_values(self):
    """Test NoTimeGaps with datetime Series values (new behavior)."""
    timestamps = pd.date_range("2023-01-01", periods=5, freq="D")
    data = pd.Series(timestamps)  # datetime values, not index
    validator = NoTimeGaps["D"]
    result = validator.validate(data)
    assert result.equals(data)

  def test_notimegaps_with_series_index_via_wrapper(self):
    """Test NoTimeGaps on Series index via Index wrapper."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    data = pd.Series([1, 2, 3, 4, 5], index=index)
    validator = Index[NoTimeGaps["D"]]
    result = validator.validate(data)
    assert result.equals(data)

  def test_notimegaps_detects_gaps_in_datetime_values(self):
    """Test NoTimeGaps detects gaps in datetime Series values."""
    timestamps = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-04"])
    data = pd.Series(timestamps)
    validator = NoTimeGaps["D"]
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)

  def test_notimegaps_detects_gaps_via_index_wrapper(self):
    """Test NoTimeGaps detects gaps via Index wrapper."""
    index = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-04"])
    data = pd.Series([1, 2, 3], index=index)
    validator = Index[NoTimeGaps["D"]]
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)

  def test_notimegaps_with_non_datetime_series(self):
    """Test NoTimeGaps requires datetime data."""
    data = pd.Series([1, 2, 3])  # numeric, not datetime
    validator = NoTimeGaps["D"]
    with pytest.raises(ValueError, match="NoTimeGaps requires datetime data"):
      validator.validate(data)

  def test_notimegaps_empty_datetime_series(self):
    """Test NoTimeGaps with empty datetime Series."""
    data = pd.Series([], dtype="datetime64[ns]")
    validator = NoTimeGaps["D"]
    result = validator.validate(data)
    assert len(result) == 0

  def test_notimegaps_with_dataframe_index_via_wrapper(self):
    """Test NoTimeGaps with DataFrame index via Index wrapper."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=index)
    validator = Index[NoTimeGaps["D"]]
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
  """Test NoTimeGaps detects various gap scenarios."""

  def test_notimegaps_wrong_number_of_entries(self):
    """Test NoTimeGaps when datetime values have wrong number of entries."""
    # Create datetime Series with gaps (missing days)
    timestamps = pd.DatetimeIndex(["2023-01-01", "2023-01-03", "2023-01-05"])
    data = pd.Series(timestamps)
    validator = NoTimeGaps["D"]
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)

  def test_notimegaps_correct_length_wrong_dates(self):
    """Test NoTimeGaps when we have correct length but wrong timestamps.

    This is the edge case where length check passes but difference check fails.
    This happens when we have duplicate timestamps!
    """
    # Create datetime Series with duplicates
    timestamps = pd.DatetimeIndex([
      "2023-01-01",
      "2023-01-01",  # Duplicate!
      "2023-01-03",
    ])
    data = pd.Series(timestamps)
    validator = NoTimeGaps["D"]
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)

  def test_notimegaps_with_datetimeindex_directly(self):
    """Test NoTimeGaps with DatetimeIndex object directly."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    validator = NoTimeGaps["D"]
    result = validator.validate(index)
    assert result.equals(index)

  def test_notimegaps_with_datetimeindex_with_gaps(self):
    """Test NoTimeGaps detects gaps in DatetimeIndex object."""
    index = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-04"])
    validator = NoTimeGaps["D"]
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(index)

  def test_notimegaps_index_validation_via_wrapper(self):
    """Test NoTimeGaps on Series index via Index wrapper."""
    index = pd.DatetimeIndex(["2023-01-01", "2023-01-03", "2023-01-05"])
    data = pd.Series([1, 2, 3], index=index)
    validator = Index[NoTimeGaps["D"]]
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)


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


class TestMaxGap:
  """Test MaxGap validator."""

  def test_maxgap_with_valid_datetime_series(self):
    """Test MaxGap passes when gaps are within limit."""
    timestamps = pd.date_range("2023-01-01", periods=5, freq="1min")
    data = pd.Series(timestamps)
    validator = MaxGap["2min"]  # Allow up to 2 minute gaps
    result = validator.validate(data)
    assert result.equals(data)

  def test_maxgap_with_gaps_within_tolerance(self):
    """Test MaxGap passes with gaps within tolerance."""
    # 1-minute data with one 2-minute gap (missing row)
    timestamps = pd.DatetimeIndex([
      "2023-01-01 09:00",
      "2023-01-01 09:01",
      "2023-01-01 09:02",
      "2023-01-01 09:04",  # Skipped 09:03
      "2023-01-01 09:05",
    ])
    data = pd.Series(timestamps)
    validator = MaxGap["2min"]  # 2-minute gaps allowed
    result = validator.validate(data)
    assert result.equals(data)

  def test_maxgap_fails_when_gap_exceeds_limit(self):
    """Test MaxGap fails when gap exceeds allowed limit."""
    timestamps = pd.DatetimeIndex([
      "2023-01-01 09:00",
      "2023-01-01 09:01",
      "2023-01-01 09:05",  # 4-minute gap!
    ])
    data = pd.Series(timestamps)
    validator = MaxGap["2min"]
    with pytest.raises(ValueError, match="Time gap exceeds maximum"):
      validator.validate(data)

  def test_maxgap_with_datetimeindex(self):
    """Test MaxGap with DatetimeIndex directly."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    validator = MaxGap["2D"]
    result = validator.validate(index)
    assert result.equals(index)

  def test_maxgap_with_index_wrapper(self):
    """Test MaxGap on DataFrame index via Index wrapper."""
    index = pd.DatetimeIndex([
      "2023-01-01",
      "2023-01-02",
      "2023-01-04",  # 2-day gap
    ])
    df = pd.DataFrame({"a": [1, 2, 3]}, index=index)
    validator = Index[MaxGap["3D"]]  # Allow up to 3-day gaps
    result = validator.validate(df)
    assert result.equals(df)

  def test_maxgap_with_non_datetime(self):
    """Test MaxGap requires datetime data."""
    data = pd.Series([1, 2, 3])
    validator = MaxGap["1D"]
    with pytest.raises(ValueError, match="MaxGap requires datetime data"):
      validator.validate(data)

  def test_maxgap_empty_series(self):
    """Test MaxGap with empty datetime Series."""
    data = pd.Series([], dtype="datetime64[ns]")
    validator = MaxGap["1D"]
    result = validator.validate(data)
    assert len(result) == 0

  def test_maxgap_single_value(self):
    """Test MaxGap with single value passes."""
    data = pd.Series([pd.Timestamp("2023-01-01")])
    validator = MaxGap["1D"]
    result = validator.validate(data)
    assert len(result) == 1


class TestMaxDiff:
  """Test MaxDiff validator for numeric gap validation."""

  def test_maxdiff_with_valid_series(self):
    """Test MaxDiff passes when differences are within limit."""
    data = pd.Series([10, 12, 14, 15, 17])
    validator = MaxDiff[5]
    result = validator.validate(data)
    assert result.equals(data)

  def test_maxdiff_fails_when_diff_exceeds_limit(self):
    """Test MaxDiff fails when difference exceeds limit."""
    data = pd.Series([10, 12, 20])  # 8-point jump!
    validator = MaxDiff[5]
    with pytest.raises(ValueError, match="Difference exceeds maximum"):
      validator.validate(data)

  def test_maxdiff_with_negative_changes(self):
    """Test MaxDiff handles negative changes (uses abs diff)."""
    data = pd.Series([20, 18, 15, 14])  # All diffs <= 3
    validator = MaxDiff[3]
    result = validator.validate(data)
    assert result.equals(data)

  def test_maxdiff_fails_negative_large_jump(self):
    """Test MaxDiff fails on large negative jump."""
    data = pd.Series([20, 10])  # -10 jump exceeds 5
    validator = MaxDiff[5]
    with pytest.raises(ValueError, match="Difference exceeds maximum"):
      validator.validate(data)

  def test_maxdiff_with_float_limit(self):
    """Test MaxDiff with float limit."""
    data = pd.Series([1.0, 1.2, 1.5, 1.6])
    validator = MaxDiff[0.5]
    result = validator.validate(data)
    assert result.equals(data)

  def test_maxdiff_with_hascolumn(self):
    """Test MaxDiff with HasColumn wrapper."""
    df = pd.DataFrame({"price": [100, 102, 105, 104]})
    validator = HasColumn[Literal["price"], MaxDiff[5], Nullable]
    result = validator.validate(df)
    assert result.equals(df)

  def test_maxdiff_requires_series(self):
    """Test MaxDiff requires a pandas Series."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    validator = MaxDiff[5]
    with pytest.raises(TypeError, match="MaxDiff requires a pandas Series"):
      validator.validate(df)  # type: ignore[arg-type]

  def test_maxdiff_requires_numeric(self):
    """Test MaxDiff requires numeric data."""
    data = pd.Series(["a", "b", "c"])
    validator = MaxDiff[5]
    with pytest.raises(ValueError, match="MaxDiff requires numeric data"):
      validator.validate(data)

  def test_maxdiff_empty_series(self):
    """Test MaxDiff with empty Series."""
    data = pd.Series([], dtype="float64")
    validator = MaxDiff[5]
    result = validator.validate(data)
    assert len(result) == 0

  def test_maxdiff_single_value(self):
    """Test MaxDiff with single value passes."""
    data = pd.Series([42])
    validator = MaxDiff[5]
    result = validator.validate(data)
    assert len(result) == 1
