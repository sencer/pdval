"""Value validators for Series and DataFrame data."""

from __future__ import annotations

from typing import Any, Literal, get_args, get_origin, override

import numpy as np
import pandas as pd

from pdval.base import Validator


class Finite(Validator[pd.Series | pd.DataFrame]):
  """Validator for finite values (no Inf, no NaN).

  Checks for both NaN and infinite values using pandas methods
  for compatibility with all numeric dtypes.
  """

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, (pd.Series, pd.DataFrame)):
      # Check for NaN (axis=None returns scalar bool, pandas-stubs typing issue)
      if data.isna().any(axis=None):  # pyright: ignore[reportArgumentType,reportGeneralTypeIssues]
        raise ValueError("Data must be finite (contains NaN)")
      # Check for infinite values (only for numeric data)
      numeric_data = (
        data.select_dtypes(include=[np.number])
        if isinstance(data, pd.DataFrame)
        else data
      )
      if len(numeric_data) > 0 and np.any(np.isinf(numeric_data.values)):  # type: ignore[arg-type]
        raise ValueError("Data must be finite (contains Inf)")
    return data


class NonEmpty(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-empty data."""

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) and data.empty:
      raise ValueError("Data must not be empty")
    return data


class NonNaN(Validator[pd.Series | pd.DataFrame]):
  """Validator for non-NaN values.

  Uses pd.isna() for compatibility with all dtypes including object columns.
  """

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    # axis=None returns scalar bool, pandas-stubs typing issue
    if isinstance(data, (pd.Series, pd.DataFrame)) and data.isna().any(axis=None):  # pyright: ignore[reportArgumentType,reportGeneralTypeIssues]
      raise ValueError("Data must not contain NaN values")
    return data


class NonNegative(Validator[pd.Series | pd.DataFrame]):
  """Validator for non-negative values (>= 0)."""

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(data.values < 0):  # type: ignore[operator]
      raise ValueError("Data must be non-negative")
    return data


class Positive(Validator[pd.Series | pd.DataFrame]):
  """Validator for positive values (> 0)."""

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(data.values <= 0):  # type: ignore[operator]
      raise ValueError("Data must be positive")
    return data


class OneOf(Validator[pd.Series | pd.Index]):
  """Validator for categorical values - ensures all values are in allowed set.

  Supports multiple syntax forms:
  - OneOf[Literal["a", "b", "c"]]
  - OneOf["a", "b", "c"]
  - OneOf[Literal["a"], Literal["b"], Literal["c"]]

  Can be used with Index[] wrapper for index validation:
  - Index[OneOf["x", "y", "z"]]

  Can be used with HasColumn for column-specific validation:
  - HasColumn["category", OneOf["a", "b", "c"]]
  """

  def __init__(self, *allowed: Any) -> None:  # noqa: ANN401  # pyright: ignore[reportExplicitAny]
    super().__init__()
    self.allowed = set(allowed)

  def __class_getitem__(cls, items: object) -> OneOf:
    # Handle single Literal with multiple values: Literal["a", "b", "c"]
    if get_origin(items) is Literal:
      items = get_args(items)
    # Handle tuple of items (could be strings or Literals)
    elif isinstance(items, tuple):
      # Flatten any Literals in the tuple: (Literal["a"], Literal["b"]) -> ("a", "b")
      flattened: list[Any] = []  # pyright: ignore[reportExplicitAny]
      for item in items:
        if get_origin(item) is Literal:
          flattened.extend(get_args(item))
        else:
          flattened.append(item)
      items = tuple(flattened)

    if not isinstance(items, tuple):
      items = (items,)

    return cls(*items)

  @override
  def validate(self, data: pd.Series | pd.Index) -> pd.Series | pd.Index:
    if isinstance(data, pd.Index):
      invalid = set(data) - self.allowed
      if invalid:
        raise ValueError(
          f"Values must be one of {self.allowed}, got invalid: {invalid}"
        )
    elif isinstance(data, pd.Series):
      invalid = set(data.dropna().unique()) - self.allowed
      if invalid:
        raise ValueError(
          f"Values must be one of {self.allowed}, got invalid: {invalid}"
        )
    return data
