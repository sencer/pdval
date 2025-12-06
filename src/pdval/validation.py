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
from typing import (
  Annotated,
  Any,
  Literal,
  ParamSpec,
  get_args,
  get_origin,
  overload,
)

import numpy as np
import pandas as pd
from loguru import logger

# Validated alias for Annotated
Validated = Annotated


# Validator Classes


class Validator[T](ABC):
  """Base class for validators."""

  @abstractmethod
  def validate(self, data: T) -> T:
    """Validate the data and return it (potentially modified)."""


class Finite(Validator[pd.Series | pd.DataFrame]):
  """Validator for finite values (no Inf, no NaN)."""

  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, (pd.Series, pd.DataFrame)) and not np.all(
      np.isfinite(data.values)
    ):
      raise ValueError("Data must be finite (no Inf, no NaN)")
    return data


class NonEmpty(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-empty data."""

  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) and data.empty:
      raise ValueError("Data must not be empty")
    return data


class NonNaN(Validator[pd.Series | pd.DataFrame]):
  """Validator for non-NaN values."""

  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(np.isnan(data.values)):
      raise ValueError("Data must not contain NaN values")
    return data


class NonNegative(Validator[pd.Series | pd.DataFrame]):
  """Validator for non-negative values (>= 0)."""

  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(data.values < 0):
      raise ValueError("Data must be non-negative")
    return data


class Positive(Validator[pd.Series | pd.DataFrame]):
  """Validator for positive values (> 0)."""

  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(data.values <= 0):
      raise ValueError("Data must be positive")
    return data


class Datetime(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for datetime index or values."""

  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, pd.Index) and not isinstance(data, pd.DatetimeIndex):
      raise ValueError("Index must be DatetimeIndex")
    if isinstance(data, (pd.Series, pd.DataFrame)) and not isinstance(
      data.index, pd.DatetimeIndex
    ):
      raise ValueError("Index must be DatetimeIndex")
    return data


class UniqueIndex(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for unique index."""

  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, pd.Index) and not data.is_unique:
      raise ValueError("Index must be unique")
    if isinstance(data, (pd.Series, pd.DataFrame)) and not data.index.is_unique:
      raise ValueError("Index must be unique")
    return data


class NoTimeGaps(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for no time gaps in DatetimeIndex."""

  def __init__(self, freq: str) -> None:
    self.freq = freq

  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    index = None
    if isinstance(data, pd.Index):
      index = data
    elif isinstance(data, (pd.Series, pd.DataFrame)):
      index = data.index

    if index is not None:
      if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("NoTimeGaps requires a DatetimeIndex")

      if index.empty:
        return data

      expected_range = pd.date_range(start=index.min(), end=index.max(), freq=self.freq)
      if len(index) != len(expected_range):
        raise ValueError(f"Time gaps detected with frequency '{self.freq}'")

      # Check if all expected timestamps are present
      # Using difference is faster than checking equality of sets or lists
      if not expected_range.difference(index).empty:
        raise ValueError(f"Time gaps detected with frequency '{self.freq}'")

    return data


class IsDtype(Validator[pd.Series | pd.DataFrame]):
  """Validator for specific dtype."""

  def __init__(self, dtype: str | type | np.dtype) -> None:
    self.dtype = np.dtype(dtype)

  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, pd.Series):
      if data.dtype != self.dtype:
        raise ValueError(f"Data must be of type {self.dtype}, got {data.dtype}")
    elif isinstance(data, pd.DataFrame):
      # Check all columns? Or just that the dataframe contains homogenous data?
      # Usually IsDtype on a DataFrame implies all columns match.
      for col in data.columns:
        if data[col].dtype != self.dtype:
          msg = f"Column '{col}' must be of type {self.dtype}, got {data[col].dtype}"
          raise ValueError(msg)
    return data


class HasColumns(Validator[pd.DataFrame]):
  """Validator for presence of specific columns in DataFrame."""

  def __init__(self, columns: list[str]) -> None:
    self.columns = columns

  def __class_getitem__(cls, items: str | tuple[str, ...]) -> HasColumns:
    if get_origin(items) is Literal:
      items = get_args(items)

    if isinstance(items, str):
      items = (items,)
    return cls(list(items))

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
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


class Ge(Validator[pd.DataFrame]):
  """Validator that Col1 >= Col2."""

  def __init__(self, col1: str, col2: str) -> None:
    self.col1 = col1
    self.col2 = col2

  def __class_getitem__(cls, items: tuple[str, str]) -> Ge:
    if get_origin(items) is Literal:
      items = get_args(items)  # type: ignore[assignment]
    return cls(items[0], items[1])

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Ge validator requires a pandas DataFrame")

    if (
      self.col1 in data.columns
      and self.col2 in data.columns
      and np.any(data[self.col1].values < data[self.col2].values)
    ):
      raise ValueError(f"{self.col1} must be >= {self.col2}")
    return data


class Le(Validator[pd.DataFrame]):
  """Validator that Col1 <= Col2."""

  def __init__(self, col1: str, col2: str) -> None:
    self.col1 = col1
    self.col2 = col2

  def __class_getitem__(cls, items: tuple[str, str]) -> Le:
    if get_origin(items) is Literal:
      items = get_args(items)  # type: ignore[assignment]
    return cls(items[0], items[1])

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Le validator requires a pandas DataFrame")

    if (
      self.col1 in data.columns
      and self.col2 in data.columns
      and np.any(data[self.col1].values > data[self.col2].values)
    ):
      raise ValueError(f"{self.col1} must be <= {self.col2}")
    return data


class Gt(Validator[pd.DataFrame]):
  """Validator that Col1 > Col2."""

  def __init__(self, col1: str, col2: str) -> None:
    self.col1 = col1
    self.col2 = col2

  def __class_getitem__(cls, items: tuple[str, str]) -> Gt:
    if get_origin(items) is Literal:
      items = get_args(items)  # type: ignore[assignment]
    return cls(items[0], items[1])

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Gt validator requires a pandas DataFrame")

    if (
      self.col1 in data.columns
      and self.col2 in data.columns
      and np.any(data[self.col1].values <= data[self.col2].values)
    ):
      raise ValueError(f"{self.col1} must be > {self.col2}")
    return data


class Lt(Validator[pd.DataFrame]):
  """Validator that Col1 < Col2."""

  def __init__(self, col1: str, col2: str) -> None:
    self.col1 = col1
    self.col2 = col2

  def __class_getitem__(cls, items: tuple[str, str]) -> Lt:
    if get_origin(items) is Literal:
      items = get_args(items)  # type: ignore[assignment]
    return cls(items[0], items[1])

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Lt validator requires a pandas DataFrame")

    if (
      self.col1 in data.columns
      and self.col2 in data.columns
      and np.any(data[self.col1].values >= data[self.col2].values)
    ):
      raise ValueError(f"{self.col1} must be < {self.col2}")
    return data


class MonoUp(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for monotonically increasing values or index."""

  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, pd.Index) and not data.is_monotonic_increasing:
      raise ValueError("Index must be monotonically increasing")
    if isinstance(data, pd.Series) and not data.is_monotonic_increasing:
      raise ValueError("Values must be monotonically increasing")
    if isinstance(data, pd.DataFrame):
      for col in data.columns:
        if not data[col].is_monotonic_increasing:
          raise ValueError(f"Column '{col}' values must be monotonically increasing")
    return data


class MonoDown(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for monotonically decreasing values or index."""

  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, pd.Index) and not data.is_monotonic_decreasing:
      raise ValueError("Index must be monotonically decreasing")
    if isinstance(data, pd.Series) and not data.is_monotonic_decreasing:
      raise ValueError("Values must be monotonically decreasing")
    if isinstance(data, pd.DataFrame):
      for col in data.columns:
        if not data[col].is_monotonic_decreasing:
          raise ValueError(f"Column '{col}' values must be monotonically decreasing")
    return data


class Index(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for index properties.

  Can be used to apply validators to the index:
  - Index[Datetime] - Check index is DatetimeIndex
  - Index[MonoUp] - Check index is monotonically increasing
  - Index[Datetime, MonoUp] - Check both
  """

  def __init__(self, *validators: Validator[Any] | type[Validator[Any]]) -> None:
    self.validators = validators

  def __class_getitem__(
    cls, items: type[Validator[Any]] | tuple[type[Validator[Any]], ...]
  ) -> Index:
    # Handle single validator
    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)

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


class HasColumn(Validator[pd.DataFrame]):
  """Wrapper to apply validators to specific DataFrame columns."""

  def __init__(
    self, column: str, *validators: Validator[Any] | type[Validator[Any]]
  ) -> None:
    self.column = column
    self.validators = validators

  def __class_getitem__(cls, items: str | tuple[str, ...]) -> HasColumn:
    if get_origin(items) is Literal:
      args = get_args(items)
      if len(args) == 1:
        items = args[0]
      else:
        pass

    # Handle single column name
    if isinstance(items, str):
      return cls(items)

    # Handle tuple: (column, validators...)
    # Check if first item is Literal
    if isinstance(items, tuple) and len(items) > 0:
      first = items[0]
      if get_origin(first) is Literal:
        args = get_args(first)
        if args:
          items = (args[0], *items[1:])

    column = items[0]
    validators = items[1:] if len(items) > 1 else ()
    return cls(column, *validators)  # type: ignore[arg-type]

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
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
R = typing.TypeVar("R")


@overload
def validated(  # noqa: UP047
  func: Callable[P, R],
) -> Callable[P, R]: ...


@overload
def validated(
  *, skip_validation_by_default: bool = False, warn_only_by_default: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R | None]]: ...


def validated(  # noqa: UP047
  func: Callable[P, R] | None = None,
  *,
  skip_validation_by_default: bool = False,
  warn_only_by_default: bool = False,
) -> Callable[P, R | None] | Callable[[Callable[P, R]], Callable[P, R | None]]:
  """Decorator to validate function arguments based on Annotated types.

  The decorator automatically adds a `skip_validation` parameter to the function.
  When `skip_validation=False` (default), validation is performed. When
  `skip_validation=True`, validation is skipped for maximum performance.

  Args:
    func: The function to decorate.
    skip_validation_by_default: If True, `skip_validation` defaults to True.
    warn_only_by_default: If True, `warn_only` defaults to True. When `warn_only` is
      True, validation failures log an error and return None instead of raising.

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
    >>>
    >>> # Change default behavior
    >>> @validated(skip_validation_by_default=True)
    >>> def fast_process(data: Validated[pd.Series, Finite]):
    ...     return data.sum()
    >>>
    >>> # Validation skipped by default
    >>> result = fast_process(valid_data)
    >>>
    >>> # Enable validation explicitly
    >>> result = fast_process(valid_data, skip_validation=False)
  """

  def decorator(func: Callable[P, R]) -> Callable[P, R | None]:
    # Inspect function signature
    sig = inspect.signature(func)
    type_hints = typing.get_type_hints(func, include_extras=True)

    # Pre-compute validators for each argument
    arg_validators: dict[str, list[Validator[Any]]] = {}
    for name, _ in sig.parameters.items():
      if name in type_hints:
        hint = type_hints[name]

        # Handle Optional/Union types
        origin = get_origin(hint)
        if (
          origin is typing.Union
          or str(origin) == "typing.Union"
          or str(origin) == "<class 'types.UnionType'>"
        ):
          # Check args for Annotated
          for arg in get_args(hint):
            if get_origin(arg) is Annotated:
              hint = arg
              break

        if get_origin(hint) is Annotated:
          args = get_args(hint)
          # First arg is the type, rest are metadata (validators)
          validators = []
          for item in args[1:]:
            v = _instantiate_validator(item)
            if v:
              validators.append(v)
          if validators:
            arg_validators[name] = validators

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
      # Check for skip_validation in kwargs
      skip = kwargs.pop("skip_validation", skip_validation_by_default)
      if skip:
        return func(*args, **kwargs)

      # Check for warn_only in kwargs
      warn_only = kwargs.pop("warn_only", warn_only_by_default)

      # Bind arguments
      bound_args = sig.bind(*args, **kwargs)
      bound_args.apply_defaults()

      # Validate arguments
      try:
        for name, value in bound_args.arguments.items():
          if name in arg_validators:
            for v in arg_validators[name]:
              v.validate(value)
      except Exception as e:
        if warn_only:
          logger.error(f"Validation failed for {func.__name__}: {e}")
          return None
        raise e

      return func(*args, **kwargs)

    return wrapper

  if func is None:
    return decorator  # type: ignore

  return decorator(func)


def _instantiate_validator(item: Any) -> Validator[Any] | None:  # noqa: ANN401
  """Helper to instantiate a validator from a type or instance."""
  if isinstance(item, type) and issubclass(item, Validator):
    return item()
  if isinstance(item, Validator):
    return item
  return None
