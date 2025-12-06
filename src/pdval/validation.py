"""Pandas validation using Annotated types and decorators.

This module provides a decorator-based validation system for pandas DataFrames
and Series using Python's Annotated types.
"""

from __future__ import annotations

# pyright: reportOperatorIssue=false, reportUnknownMemberType=false, reportUnknownVariableType=false
import functools
import inspect
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Annotated, Any, ParamSpec, TypeVar, get_args, get_origin

import numpy as np
import pandas as pd

# Validated alias for Annotated
Validated = Annotated

# Validator Classes


class Validator(ABC):
  """Base class for validators."""

  @abstractmethod
  def validate(self, data: Any) -> Any:  # noqa: ANN401
    """Validate the data and return it (potentially modified)."""


class Finite(Validator):
  """Validator for finite values (no Inf, no NaN)."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, (pd.Series, pd.DataFrame)) and not np.all(
      np.isfinite(data.values)
    ):
      raise ValueError("Data must be finite (no Inf, no NaN)")
    return data


class NonNaN(Validator):
  """Validator for non-NaN values."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(np.isnan(data.values)):
      raise ValueError("Data must not contain NaN values")
    return data


class NonNegative(Validator):
  """Validator for non-negative values (>= 0)."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(data.values < 0):
      raise ValueError("Data must be non-negative")
    return data


class Positive(Validator):
  """Validator for positive values (> 0)."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(data.values <= 0):
      raise ValueError("Data must be positive")
    return data


class Datetime(Validator):
  """Validator for datetime index or values."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, pd.Index) and not isinstance(data, pd.DatetimeIndex):
      raise ValueError("Index must be DatetimeIndex")
    if isinstance(data, (pd.Series, pd.DataFrame)) and not isinstance(
      data.index, pd.DatetimeIndex
    ):
      raise ValueError("Index must be DatetimeIndex")
    return data


class HasColumns(Validator):
  """Validator for presence of specific columns in DataFrame."""

  def __init__(self, columns: list[str]) -> None:
    self.columns = columns

  def __class_getitem__(cls, items: str | tuple[str, ...]) -> HasColumns:
    if isinstance(items, str):
      items = (items,)
    return cls(list(items))

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if not isinstance(data, pd.DataFrame):
      # We expect a DataFrame for column validation
      # If it's not a DataFrame, we can't check columns, so we should probably raise
      # unless we want to allow duck typing? But the validator is explicitly HasColumns.
      # Let's be strict as per plan.
      raise TypeError("HasColumns validator requires a pandas DataFrame")

    missing = [col for col in self.columns if col not in data.columns]
    if missing:
      raise ValueError(f"Missing columns: {missing}")
    return data


class Ge(Validator):
  """Validator that Col1 >= Col2."""

  def __init__(self, col1: str, col2: str) -> None:
    self.col1 = col1
    self.col2 = col2

  def __class_getitem__(cls, items: tuple[str, str]) -> Ge:
    return cls(items[0], items[1])

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Ge validator requires a pandas DataFrame")

    if (
      self.col1 in data.columns
      and self.col2 in data.columns
      and np.any(data[self.col1].values < data[self.col2].values)
    ):
      raise ValueError(f"{self.col1} must be >= {self.col2}")
    return data


class Le(Validator):
  """Validator that Col1 <= Col2."""

  def __init__(self, col1: str, col2: str) -> None:
    self.col1 = col1
    self.col2 = col2

  def __class_getitem__(cls, items: tuple[str, str]) -> Le:
    return cls(items[0], items[1])

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Le validator requires a pandas DataFrame")

    if (
      self.col1 in data.columns
      and self.col2 in data.columns
      and np.any(data[self.col1].values > data[self.col2].values)
    ):
      raise ValueError(f"{self.col1} must be <= {self.col2}")
    return data


class Gt(Validator):
  """Validator that Col1 > Col2."""

  def __init__(self, col1: str, col2: str) -> None:
    self.col1 = col1
    self.col2 = col2

  def __class_getitem__(cls, items: tuple[str, str]) -> Gt:
    return cls(items[0], items[1])

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Gt validator requires a pandas DataFrame")

    if (
      self.col1 in data.columns
      and self.col2 in data.columns
      and np.any(data[self.col1].values <= data[self.col2].values)
    ):
      raise ValueError(f"{self.col1} must be > {self.col2}")
    return data


class Lt(Validator):
  """Validator that Col1 < Col2."""

  def __init__(self, col1: str, col2: str) -> None:
    self.col1 = col1
    self.col2 = col2

  def __class_getitem__(cls, items: tuple[str, str]) -> Lt:
    return cls(items[0], items[1])

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Lt validator requires a pandas DataFrame")

    if (
      self.col1 in data.columns
      and self.col2 in data.columns
      and np.any(data[self.col1].values >= data[self.col2].values)
    ):
      raise ValueError(f"{self.col1} must be < {self.col2}")
    return data


class MonoUp(Validator):
  """Validator for monotonically increasing values or index."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, pd.Index) and not data.is_monotonic_increasing:
      raise ValueError("Index must be monotonically increasing")
    if isinstance(data, pd.Series) and not data.is_monotonic_increasing:
      raise ValueError("Values must be monotonically increasing")
    if isinstance(data, pd.DataFrame):
      for col in data.columns:
        if not data[col].is_monotonic_increasing:
          raise ValueError(f"Column '{col}' values must be monotonically increasing")
    return data


class MonoDown(Validator):
  """Validator for monotonically decreasing values or index."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, pd.Index) and not data.is_monotonic_decreasing:
      raise ValueError("Index must be monotonically decreasing")
    if isinstance(data, pd.Series) and not data.is_monotonic_decreasing:
      raise ValueError("Values must be monotonically decreasing")
    if isinstance(data, pd.DataFrame):
      for col in data.columns:
        if not data[col].is_monotonic_decreasing:
          raise ValueError(f"Column '{col}' values must be monotonically decreasing")
    return data


class Index(Validator):
  """Validator for index properties.

  Can be used to apply validators to the index:
  - Index[Datetime] - Check index is DatetimeIndex
  - Index[MonoUp] - Check index is monotonically increasing
  - Index[Datetime, MonoUp] - Check both
  """

  def __init__(self, *validators: Validator | type[Validator]) -> None:
    self.validators = validators

  def __class_getitem__(
    cls, items: type[Validator] | tuple[type[Validator], ...]
  ) -> Index:
    # Handle single validator
    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)

  def validate(self, data: Any) -> Any:  # noqa: ANN401
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


class HasColumn(Validator):
  """Wrapper to apply validators to specific DataFrame columns."""

  def __init__(self, column: str, *validators: Validator | type[Validator]) -> None:
    self.column = column
    self.validators = validators

  def __class_getitem__(cls, items: str | tuple[str, ...]) -> HasColumn:
    # Handle single column name
    if isinstance(items, str):
      return cls(items)

    # Handle tuple: (column, validators...)
    column = items[0]
    validators = items[1:] if len(items) > 1 else ()
    return cls(column, *validators)  # type: ignore[arg-type]

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, pd.DataFrame):
      if self.column not in data.columns:
        raise ValueError(f"Column '{self.column}' not found in DataFrame")

      # Extract the column as a Series
      column_data = data[self.column]

      # Apply each validator
      for validator_item in self.validators:
        # Handle both validator classes and instances
        if isinstance(validator_item, type) and issubclass(validator_item, Validator):
          validator = validator_item()
        elif isinstance(validator_item, Validator):
          validator = validator_item
        else:
          continue

        # Validate the column
        column_data = validator.validate(column_data)

    return data


P = ParamSpec("P")
R = TypeVar("R")


def validated(func: Callable[P, R]) -> Callable[P, R]:  # noqa: UP047
  """Decorator to validate function arguments based on Annotated types.

  The decorator automatically adds a `skip_validation` parameter to the function.
  When `skip_validation=False` (default), validation is performed. When
  `skip_validation=True`, validation is skipped for maximum performance.

  Args:
    func: The function to decorate.

  Returns:
    The decorated function with automatic validation support.

  Example:
    >>> from pdval import validated, Validated, Finite
    >>> import pandas as pd
    >>>
    >>> @validated
    ... def process(data: Validated[pd.Series, Finite]):
    ...     return data.sum()
    >>>
    >>> # Validation enabled by default
    >>> result = process(valid_data)
    >>>
    >>> # Skip validation for performance
    >>> result = process(valid_data, skip_validation=True)
  """
  # 1. Compute metadata ONCE at decoration time
  sig = inspect.signature(func)

  # Resolve type hints to handle string annotations
  try:
    type_hints = typing.get_type_hints(func, include_extras=True)
  except Exception:
    # Fallback if resolution fails (e.g. during development/testing)
    type_hints = {}

  # Pre-compute a mapping of arg_name -> list[Validator]
  arg_validators: dict[str, list[Validator]] = {}

  for name, param in sig.parameters.items():
    if name == "self":
      continue

    annotation = type_hints.get(name, param.annotation)
    origin = get_origin(annotation)

    # Handle Optional[Annotated[...]] / Union[Annotated[...], None]
    if origin is typing.Union:
      args = get_args(annotation)
      for arg in args:
        if get_origin(arg) is Annotated:
          annotation = arg
          origin = Annotated
          break

    if origin is Annotated:
      metadata = get_args(annotation)
      validators = []
      # Iterate over metadata (skipping the first one which is the type)
      for item in metadata[1:]:
        if isinstance(item, type) and issubclass(item, Validator):
          validators.append(item())
        elif isinstance(item, Validator) or (
          hasattr(item, "validate") and callable(item.validate)
        ):
          validators.append(item)

      if validators:
        arg_validators[name] = validators

  @functools.wraps(func)
  def wrapper(*args: P.args, skip_validation: bool = False, **kwargs: P.kwargs) -> R:
    if skip_validation:
      return func(*args, **kwargs)

    # 2. Bind args
    # Note: sig.bind is still somewhat expensive, but necessary to map args to names
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    # 3. Validate using pre-computed metadata
    for name, value in bound.arguments.items():
      # Skip if value is None (unless we want to validate None,
      # but usually we don't for Optional)
      if value is None:
        continue

      if name in arg_validators:
        for validator in arg_validators[name]:
          try:
            value = validator.validate(value)
          except Exception as e:
            # Re-raise validation errors
            raise e

        # Update the bound argument with the validated value
        # (in case validator modified it)
        bound.arguments[name] = value

    return func(*bound.args, **bound.kwargs)

  return wrapper  # type: ignore[return-value]
