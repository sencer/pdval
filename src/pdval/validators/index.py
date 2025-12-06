"""Index and monotonicity validators."""

from __future__ import annotations

from typing import Any, override

import pandas as pd

from pdval.base import Validator


class Datetime(Validator[pd.Series | pd.Index]):
  """Validator for datetime index or values.

  Use with Index[Datetime] to apply to Series/DataFrame index.
  Can be applied directly to pd.Index or pd.DatetimeIndex.
  """

  @override
  def validate(self, data: pd.Series | pd.Index) -> pd.Series | pd.Index:
    if isinstance(data, pd.Index) and not isinstance(data, pd.DatetimeIndex):
      raise ValueError("Index must be DatetimeIndex")
    if isinstance(data, pd.Series) and not isinstance(data.index, pd.DatetimeIndex):
      raise ValueError("Index must be DatetimeIndex")
    return data


class Unique(Validator[pd.Series | pd.Index]):
  """Validator for unique values.

  Use with Index[Unique] to apply to Series/DataFrame index.
  Can be applied directly to pd.Index or pd.Series values.
  """

  @override
  def validate(self, data: pd.Series | pd.Index) -> pd.Series | pd.Index:
    if isinstance(data, pd.Index) and not data.is_unique:
      raise ValueError("Values must be unique")
    if isinstance(data, pd.Series) and not data.is_unique:
      raise ValueError("Values must be unique")
    return data


class MonoUp(Validator[pd.Series | pd.Index]):
  """Validator for monotonically increasing values.

  Use with Index[MonoUp] to apply to Series/DataFrame index.
  Can be applied directly to pd.Index or pd.Series values.
  """

  @override
  def validate(self, data: pd.Series | pd.Index) -> pd.Series | pd.Index:
    if isinstance(data, pd.Index) and not data.is_monotonic_increasing:
      raise ValueError("Values must be monotonically increasing")
    if isinstance(data, pd.Series) and not data.is_monotonic_increasing:
      raise ValueError("Values must be monotonically increasing")
    return data


class MonoDown(Validator[pd.Series | pd.Index]):
  """Validator for monotonically decreasing values.

  Use with Index[MonoDown] to apply to Series/DataFrame index.
  Can be applied directly to pd.Index or pd.Series values.
  """

  @override
  def validate(self, data: pd.Series | pd.Index) -> pd.Series | pd.Index:
    if isinstance(data, pd.Index) and not data.is_monotonic_decreasing:
      raise ValueError("Values must be monotonically decreasing")
    if isinstance(data, pd.Series) and not data.is_monotonic_decreasing:
      raise ValueError("Values must be monotonically decreasing")
    return data


class Index(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for index properties.

  Can be used to apply validators to the index:
  - Index[Datetime] - Check index is DatetimeIndex
  - Index[MonoUp] - Check index is monotonically increasing
  - Index[Datetime, MonoUp] - Check both
  """

  def __init__(
    self,
    *validators: Validator[Any] | type[Validator[Any]],  # type: ignore[misc]
  ) -> None:
    super().__init__()
    self.validators = validators

  def __class_getitem__(cls, items: object) -> Index:
    # Handle single validator
    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)  # type: ignore[arg-type]

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, (pd.Series, pd.DataFrame)):
      index = data.index

      # Apply each validator to the index
      for validator_item in self.validators:
        # Handle both validator classes and instances
        if isinstance(validator_item, type) and issubclass(validator_item, Validator):
          validator = validator_item()
        elif isinstance(validator_item, Validator):
          validator = validator_item
        else:
          continue

        # Validate the index
        index = validator.validate(index)

    return data
