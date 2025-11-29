"""Pandas validation using Annotated types and decorators.

This module provides a decorator-based validation system for pandas DataFrames
and Series using Python's Annotated types.
"""

from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Callable
from typing import Annotated, Any, ParamSpec, TypeVar, get_args, get_origin

import numpy as np
import pandas as pd

# Validated alias for Annotated
Validated = Annotated

# Validator Classes


class Validator:
  """Base class for validators."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    raise NotImplementedError


class Finite(Validator):
  """Validator for finite values (no Inf, no NaN)."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if (
      isinstance(data, (pd.Series, pd.DataFrame))
      and len(data) > 0
      and (np.any(np.isinf(data)) or np.any(np.isnan(data)))
    ):
      raise ValueError("Data must be finite (no Inf, no NaN)")
    return data


class NonNaN(Validator):
  """Validator for non-NaN values."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if (
      isinstance(data, (pd.Series, pd.DataFrame))
      and len(data) > 0
      and np.any(np.isnan(data))
    ):
      raise ValueError("Data must not contain NaN values")
    return data


class NonNegative(Validator):
  """Validator for non-negative values (>= 0)."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if (
      isinstance(data, (pd.Series, pd.DataFrame)) and len(data) > 0 and np.any(data < 0)
    ):
      raise ValueError("Data must be non-negative")
    return data


class Positive(Validator):
  """Validator for positive values (> 0)."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if (
      isinstance(data, (pd.Series, pd.DataFrame))
      and len(data) > 0
      and np.any(data <= 0)
    ):
      raise ValueError("Data must be positive")
    return data


class DateTimeIndexed(Validator):
  """Validator for DateTimeIndex."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, (pd.Series, pd.DataFrame)) and not isinstance(
      data.index, pd.DatetimeIndex
    ):
      raise ValueError("Index must be DatetimeIndex")
    return data


class MonotonicIndex(Validator):
  """Validator for monotonic increasing index."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if (
      isinstance(data, (pd.Series, pd.DataFrame))
      and not data.index.is_monotonic_increasing
    ):
      raise ValueError("Index must be monotonic increasing")
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
    if isinstance(data, pd.DataFrame):
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
    if (
      isinstance(data, pd.DataFrame)
      and self.col1 in data.columns
      and self.col2 in data.columns
      and np.any(data[self.col1] < data[self.col2])
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
    if (
      isinstance(data, pd.DataFrame)
      and self.col1 in data.columns
      and self.col2 in data.columns
      and np.any(data[self.col1] > data[self.col2])
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
    if (
      isinstance(data, pd.DataFrame)
      and self.col1 in data.columns
      and self.col2 in data.columns
      and np.any(data[self.col1] <= data[self.col2])
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
    if (
      isinstance(data, pd.DataFrame)
      and self.col1 in data.columns
      and self.col2 in data.columns
      and np.any(data[self.col1] >= data[self.col2])
    ):
      raise ValueError(f"{self.col1} must be < {self.col2}")
    return data


class MonoUp(Validator):
  """Validator for monotonically increasing values."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, pd.Series) and not data.is_monotonic_increasing:
      raise ValueError("Values must be monotonically increasing")
    if isinstance(data, pd.DataFrame):
      for col in data.columns:
        if not data[col].is_monotonic_increasing:
          raise ValueError(f"Column '{col}' values must be monotonically increasing")
    return data


class MonoDown(Validator):
  """Validator for monotonically decreasing values."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, pd.Series) and not data.is_monotonic_decreasing:
      raise ValueError("Values must be monotonically decreasing")
    if isinstance(data, pd.DataFrame):
      for col in data.columns:
        if not data[col].is_monotonic_decreasing:
          raise ValueError(f"Column '{col}' values must be monotonically decreasing")
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
    return cls(column, *validators)

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


def _validate_single_argument(
  name: str,
  value: Any,  # noqa: ANN401
  annotation: Any,  # noqa: ANN401
  bound_args: inspect.BoundArguments,
) -> None:
  """Validate a single argument against its annotation."""
  origin = get_origin(annotation)

  # Handle Optional[Annotated[...]] / Union[Annotated[...], None]
  if origin is typing.Union:
    args = get_args(annotation)
    # If value is None and None is allowed, skip validation
    if value is None and type(None) in args:
      return

    # Find the Annotated type within the Union
    for arg in args:
      if get_origin(arg) is Annotated:
        annotation = arg
        origin = Annotated
        break

  # Check if annotation is Annotated[Type, Schema]
  if origin is not Annotated:
    return

  metadata = get_args(annotation)

  # Iterate over all metadata items (skipping the first one which is the type)
  for item in metadata[1:]:
    validator = None

    # Check if item is a Validator class
    if isinstance(item, type) and issubclass(item, Validator):
      validator = item()
    # Check if item is a Validator instance
    elif isinstance(item, Validator) or (
      hasattr(item, "validate") and callable(item.validate)
    ):
      validator = item

    if validator:
      try:
        # Validate and update value for next validator
        value = validator.validate(value)
      except Exception as e:
        # Re-raise validation errors
        raise e

  # Update the bound argument with the final validated value
  bound_args.arguments[name] = value


def _validate_arguments(
  func: Callable[..., Any], bound_args: inspect.BoundArguments
) -> None:
  """Validate arguments based on Annotated types."""
  # Resolve type hints to handle string annotations (from __future__ import annotations)
  try:
    type_hints = typing.get_type_hints(func, include_extras=True)
  except Exception:
    # Fallback if resolution fails (e.g. during development/testing)
    return

  sig = inspect.signature(func)

  # Iterate over arguments and validate if they have a Schema annotation
  for name, value in bound_args.arguments.items():
    if name == "self":  # Skip self for methods
      continue

    annotation = type_hints.get(name, sig.parameters[name].annotation)
    _validate_single_argument(name, value, annotation, bound_args)


def validated(func: Callable[P, R]) -> Callable[P, R]:  # noqa: UP047
  """Decorator to validate function arguments based on Annotated types.

  If the function has a `validate` argument and it is True (default),
  this decorator inspects the type hints of the arguments. If an argument
  is annotated with `Annotated[Type, Validator]`, it calls `Validator.validate()`
  on that argument.

  Args:
    func: The function to decorate.

  Returns:
    The decorated function.

  Example:
    >>> from pdval import validated, Validated, Finite
    >>> import pandas as pd
    >>>
    >>> @validated
    ... def process(data: Validated[pd.Series, Finite], validate: bool = True):
    ...     return data.sum()
  """

  @functools.wraps(func)
  def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
    # Check if validation is enabled
    # We look for 'validate' in kwargs or default values
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    should_validate = bound_args.arguments.get("validate", True)

    if should_validate:
      _validate_arguments(func, bound_args)

    return func(*bound_args.args, **bound_args.kwargs)

  return wrapper
