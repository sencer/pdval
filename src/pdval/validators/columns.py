"""Column validators for DataFrames."""

from __future__ import annotations

import typing
from typing import Any, Literal, get_args, get_origin, override

import numpy as np
import pandas as pd

from pdval.base import Validator, ValidatorMarker
from pdval.validators.markers import MaybeEmpty, Nullable
from pdval.validators.value import NonEmpty, NonNaN


def _instantiate_validator(
  item: object,
) -> Validator[Any] | ValidatorMarker | None:  # type: ignore[misc]
  """Helper to instantiate a validator from a type or instance."""
  if isinstance(item, (Validator, ValidatorMarker)):
    return item
  if isinstance(item, type) and issubclass(item, (Validator, ValidatorMarker)):
    return item()
  return None


class IsDtype(Validator[pd.Series | pd.DataFrame]):
  """Validator for specific dtype."""

  def __init__(self, dtype: str | type | np.dtype) -> None:
    super().__init__()
    self.dtype = np.dtype(dtype)

  def __class_getitem__(cls, item: str | type | np.dtype) -> IsDtype:
    return cls(item)

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, pd.Series):
      if data.dtype != self.dtype:
        raise ValueError(f"Data must be of type {self.dtype}, got {data.dtype}")
    elif isinstance(data, pd.DataFrame):
      # Vectorized dtype check: compare all column dtypes at once
      mismatches = data.dtypes != self.dtype
      if mismatches.any():
        bad_cols = mismatches[mismatches].index.tolist()
        bad_dtypes = data.dtypes[mismatches].tolist()
        msg = f"Columns with wrong dtype (expected {self.dtype}): {dict(zip(bad_cols, bad_dtypes, strict=True))}"
        raise ValueError(msg)
    return data


class HasColumns(Validator[pd.DataFrame]):
  """Validator for presence of specific columns in DataFrame.

  Can also apply validators to the specified columns:
  HasColumns["a", "b", Finite, Positive]
  """

  def __init__(
    self,
    columns: list[str],
    validators: tuple[Validator[Any] | ValidatorMarker, ...] = (),  # type: ignore[misc]
  ) -> None:
    super().__init__()
    self.columns = columns
    self.validators = validators

  def __class_getitem__(
    cls,
    items: object,
  ) -> HasColumns:
    if get_origin(items) is Literal:
      items = get_args(items)

    if not isinstance(items, tuple):
      items = (items,)

    # Parse columns and validators
    columns: list[str] = []
    validators: list[Validator[Any] | ValidatorMarker] = []  # type: ignore[misc]

    for item in items:  # type: ignore[union-attr]
      # Handle Literal inside tuple
      if get_origin(item) is Literal:  # type: ignore[arg-type]
        args = get_args(item)  # type: ignore[arg-type]
        columns.extend([arg for arg in args if isinstance(arg, str)])
        continue

      if isinstance(item, str):
        columns.append(item)
      else:
        v = _instantiate_validator(item)  # type: ignore[arg-type]
        if v:
          validators.append(v)

    return cls(columns, tuple(validators))

  @override
  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("HasColumns validator requires a pandas DataFrame")

    missing = [col for col in self.columns if col not in data.columns]
    if missing:
      raise ValueError(f"Missing columns: {missing}")

    # Determine validators to apply
    final_validators: list[Validator[Any]] = []  # type: ignore[misc]
    is_nullable = False
    maybe_empty = False

    # Check for markers in self.validators
    if self.validators:
      for v in self.validators:
        if isinstance(v, Nullable):
          is_nullable = True
        elif isinstance(v, MaybeEmpty):
          maybe_empty = True
        elif isinstance(v, Validator):
          final_validators.append(v)

    # Add defaults if not opted out
    if not is_nullable:
      final_validators.insert(0, NonNaN())
    if not maybe_empty:
      final_validators.insert(0, NonEmpty())

    if final_validators:
      for col in self.columns:
        column_data = data[col]  # type: ignore[union-attr]
        for v in final_validators:
          column_data = v.validate(column_data)  # type: ignore[assignment]

    return data


class HasColumn(Validator[pd.DataFrame]):
  """Wrapper to apply validators to specific DataFrame columns.

  Supports templating:
  T = TypeVar("T")
  CustomVal = HasColumn[T, Positive]
  CustomVal["my_col"]  # Creates HasColumn("my_col", Positive)
  """

  def __init__(
    self,
    column: str | typing.TypeVar,
    *validators: Validator[Any]  # type: ignore[misc]
    | ValidatorMarker
    | type[Validator[Any]]  # type: ignore[misc]
    | type[ValidatorMarker],
  ) -> None:
    super().__init__()
    self.column = column
    self.validators = validators

  def __getitem__(self, item: str) -> HasColumn:
    """Support for templating: CustomVal["col"]."""
    return HasColumn(item, *self.validators)

  def __class_getitem__(
    cls,
    items: object,
  ) -> HasColumn:
    if get_origin(items) is Literal:
      args = get_args(items)
      if len(args) == 1:
        items = args[0]

    # Handle single column name
    if isinstance(items, (str, typing.TypeVar)):
      return cls(items)

    # Handle tuple: (column, validators...)
    # Check if first item is Literal
    if isinstance(items, tuple) and len(items) > 0:
      first = items[0]
      if get_origin(first) is Literal:
        args = get_args(first)
        if args:
          items = (args[0], *items[1:])

    column = items[0]  # type: ignore[index]
    validators = items[1:] if len(items) > 1 else ()  # type: ignore[index]
    return cls(column, *validators)  # type: ignore[arg-type]

  @override
  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
      if self.column not in data.columns:
        raise ValueError(f"Column '{self.column}' not found in DataFrame")

      # Extract the column as a Series
      column_data = data[self.column]

      # Determine validators to apply
      final_validators: list[Validator[Any]] = []  # type: ignore[misc]
      is_nullable = False
      maybe_empty = False

      # Check for markers in self.validators
      for validator_item in self.validators:
        v = _instantiate_validator(validator_item)
        if v:
          if isinstance(v, Nullable):
            is_nullable = True
          elif isinstance(v, MaybeEmpty):
            maybe_empty = True
          elif isinstance(v, Validator):
            final_validators.append(v)

      # Add defaults if not opted out
      if not is_nullable:
        final_validators.insert(0, NonNaN())
      if not maybe_empty:
        final_validators.insert(0, NonEmpty())

      # Apply each validator
      for validator in final_validators:
        column_data = validator.validate(column_data)

    return data
