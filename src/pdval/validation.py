"""Pandas validation using Annotated types and decorators (powered by pandera).

This module provides a decorator-based validation system for pandas DataFrames
and Series using Python's Annotated types, wrapping pandera for validation logic.
"""

from __future__ import annotations

# pyright: reportOperatorIssue=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportInvalidTypeForm=false
import functools
import inspect
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
  Annotated,
  Any,
  ParamSpec,
  get_args,
  get_origin,
  overload,
)

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError

# Validated alias for Annotated
Validated = Annotated


# Validator Classes


class Validator[T](ABC):
  """Base class for validators."""

  def get_checks(self) -> list[pa.Check]:
    """Return a list of pandera Checks."""
    return []

  def validate(self, data: T) -> T:
    """Validate data directly using pandera checks."""
    checks = self.get_checks()
    if not checks:
      return data

    if isinstance(data, pd.DataFrame):
      # For DataFrame, we need to know if checks are global or column-specific
      # Most simple validators (Finite, NonNaN) are element-wise or global
      # But pandera Checks on DataFrame are usually global unless specified
      # Let's assume simple validators apply to the whole dataframe
      schema = pa.DataFrameSchema(checks=checks, coerce=False)
      schema.validate(data)
    elif isinstance(data, pd.Series):
      schema = pa.SeriesSchema(checks=checks, coerce=False)
      schema.validate(data)

    return data


class Finite(Validator[pd.Series | pd.DataFrame]):
  """Validator for finite values (no Inf, no NaN)."""

  def get_checks(self) -> list[pa.Check]:
    # Use optimized numpy check
    return [pa.Check(lambda s: np.isfinite(s.values).all(), error="must be finite")]


class NonNaN(Validator[pd.Series | pd.DataFrame]):
  """Validator for non-NaN values."""

  def get_checks(self) -> list[pa.Check]:
    return [pa.Check(lambda s: pd.notna(s).all(), error="must not contain NaN")]


class NonNegative(Validator[pd.Series | pd.DataFrame]):
  """Validator for non-negative values (>= 0)."""

  def get_checks(self) -> list[pa.Check]:
    return [pa.Check.ge(0)]


class Positive(Validator[pd.Series | pd.DataFrame]):
  """Validator for positive values (> 0)."""

  def get_checks(self) -> list[pa.Check]:
    return [pa.Check.gt(0)]


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
    if isinstance(items, str):
      items = (items,)
    return cls(list(items))

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("HasColumns validator requires a pandas DataFrame")

    missing = [col for col in self.columns if col not in data.columns]
    if missing:
      # Raise SchemaError to match pandera behavior
      raise SchemaError(schema=None, data=data, message=f"Missing columns: {missing}")
    return data


class Ge(Validator[pd.DataFrame]):
  """Validator that Col1 >= Col2."""

  def __init__(self, col1: str, col2: str) -> None:
    self.col1 = col1
    self.col2 = col2

  def __class_getitem__(cls, items: tuple[str, str]) -> Ge:
    return cls(items[0], items[1])

  def get_checks(self) -> list[pa.Check]:
    return [
      pa.Check(
        lambda df: (
          np.all(df[self.col1].values >= df[self.col2].values)
          if isinstance(df, pd.DataFrame)
          and self.col1 in df.columns
          and self.col2 in df.columns
          else True
        ),
        name=f"{self.col1} >= {self.col2}",
        error=f"{self.col1} must be >= {self.col2}",
      )
    ]

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Ge validator requires a pandas DataFrame")
    return super().validate(data)


class Le(Validator[pd.DataFrame]):
  """Validator that Col1 <= Col2."""

  def __init__(self, col1: str, col2: str) -> None:
    self.col1 = col1
    self.col2 = col2

  def __class_getitem__(cls, items: tuple[str, str]) -> Le:
    return cls(items[0], items[1])

  def get_checks(self) -> list[pa.Check]:
    return [
      pa.Check(
        lambda df: (
          np.all(df[self.col1].values <= df[self.col2].values)
          if isinstance(df, pd.DataFrame)
          and self.col1 in df.columns
          and self.col2 in df.columns
          else True
        ),
        name=f"{self.col1} <= {self.col2}",
        error=f"{self.col1} must be <= {self.col2}",
      )
    ]

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Le validator requires a pandas DataFrame")
    return super().validate(data)


class Gt(Validator[pd.DataFrame]):
  """Validator that Col1 > Col2."""

  def __init__(self, col1: str, col2: str) -> None:
    self.col1 = col1
    self.col2 = col2

  def __class_getitem__(cls, items: tuple[str, str]) -> Gt:
    return cls(items[0], items[1])

  def get_checks(self) -> list[pa.Check]:
    return [
      pa.Check(
        lambda df: (
          np.all(df[self.col1].values > df[self.col2].values)
          if isinstance(df, pd.DataFrame)
          and self.col1 in df.columns
          and self.col2 in df.columns
          else True
        ),
        name=f"{self.col1} > {self.col2}",
        error=f"{self.col1} must be > {self.col2}",
      )
    ]

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Gt validator requires a pandas DataFrame")
    return super().validate(data)


class Lt(Validator[pd.DataFrame]):
  """Validator that Col1 < Col2."""

  def __init__(self, col1: str, col2: str) -> None:
    self.col1 = col1
    self.col2 = col2

  def __class_getitem__(cls, items: tuple[str, str]) -> Lt:
    return cls(items[0], items[1])

  def get_checks(self) -> list[pa.Check]:
    return [
      pa.Check(
        lambda df: (
          np.all(df[self.col1].values < df[self.col2].values)
          if isinstance(df, pd.DataFrame)
          and self.col1 in df.columns
          and self.col2 in df.columns
          else True
        ),
        name=f"{self.col1} < {self.col2}",
        error=f"{self.col1} must be < {self.col2}",
      )
    ]

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Lt validator requires a pandas DataFrame")
    return super().validate(data)


class MonoUp(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for monotonically increasing values or index."""

  def get_checks(self) -> list[pa.Check]:
    return [
      pa.Check(
        lambda x: x.is_monotonic_increasing
        if isinstance(x, (pd.Series, pd.Index))
        else x.apply(lambda col: col.is_monotonic_increasing).all(),
        error="must be monotonically increasing",
      )
    ]

  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    # MonoUp can be checked via pandera for Series/DataFrame, but Index?
    # Pandera supports Index checks.
    # But our base validate() only handles DataFrame/Series schemas.
    # So we should probably keep manual validation for Index, or enhance base validate.
    # For now, let's keep manual validation for robustness, or use super().validate() if data is not Index.
    if isinstance(data, pd.Index):
      if not data.is_monotonic_increasing:
        raise ValueError("Index must be monotonically increasing")
      return data
    
    # For Series/DataFrame, use pandera checks
    return super().validate(data)


class MonoDown(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for monotonically decreasing values or index."""

  def get_checks(self) -> list[pa.Check]:
    return [
      pa.Check(
        lambda x: x.is_monotonic_decreasing
        if isinstance(x, (pd.Series, pd.Index))
        else x.apply(lambda col: col.is_monotonic_decreasing).all(),
        error="must be monotonically decreasing",
      )
    ]

  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, pd.Index):
      if not data.is_monotonic_decreasing:
        raise ValueError("Index must be monotonically decreasing")
      return data
    return super().validate(data)


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
    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)

  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, (pd.Series, pd.DataFrame)):
      # Validate index
      # We can use _build_series_schema logic but applied to index
      # Or just use the validators directly on the index converted to Series
      s_index = data.index.to_series().reset_index(drop=True)
      for v_item in self.validators:
        v = _instantiate_validator(v_item)
        if v:
          if isinstance(v, Datetime):
            if not isinstance(data.index, pd.DatetimeIndex):
              raise SchemaError(
                schema=None, data=data, message="Index must be DatetimeIndex"
              )
          else:
            # Run check on index values
            v.validate(s_index)
    return data


class HasColumn(Validator[pd.DataFrame]):
  """Wrapper to apply validators to specific DataFrame columns."""

  def __init__(
    self, column: str, *validators: Validator[Any] | type[Validator[Any]]
  ) -> None:
    self.column = column
    self.validators = validators

  def __class_getitem__(cls, items: str | tuple[str, ...]) -> HasColumn:
    if isinstance(items, str):
      return cls(items)
    column = items[0]
    validators = items[1:] if len(items) > 1 else ()
    return cls(column, *validators)  # type: ignore[arg-type]

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
      if self.column not in data.columns:
        raise SchemaError(
          schema=None,
          data=data,
          message=f"Column '{self.column}' not found in DataFrame",
        )

      # Extract column as Series and validate
      series = data[self.column]
      for v_item in self.validators:
        v = _instantiate_validator(v_item)
        if v:
          v.validate(series)
    return data


@overload
def validated[P: ParamSpec, R](
  func: Callable[P, R],
) -> Callable[P, R]: ...

@overload
def validated[P: ParamSpec, R](
  *, skip_validation_by_default: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def validated[P: ParamSpec, R](
  func: Callable[P, R] | None = None,  # pyright: ignore[reportInvalidTypeForm]
  *,
  skip_validation_by_default: bool = False,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:  # pyright: ignore[reportInvalidTypeForm]
  """Decorator to validate function arguments based on Annotated types.

  The decorator automatically adds a `skip_validation` parameter to the function.
  When `skip_validation=False` (default), validation is performed. When
  `skip_validation=True`, validation is skipped for maximum performance.

  Args:
    func: The function to decorate.
    skip_validation_by_default: If True, `skip_validation` defaults to True.

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

  def decorator(func: Callable[P, R]) -> Callable[P, R]:
    # Inspect function signature
    sig = inspect.signature(func)
    type_hints = typing.get_type_hints(func, include_extras=True)

    # Pre-compute validators for each argument
    arg_validators: dict[str, list[Validator[Any]]] = {}
    for name, param in sig.parameters.items():
      if name in type_hints:
        hint = type_hints[name]
        
        # Handle Optional/Union types
        origin = get_origin(hint)
        if origin is typing.Union or str(origin) == "typing.Union" or str(origin) == "<class 'types.UnionType'>":
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
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # pyright: ignore[reportInvalidTypeForm]
      # Check for skip_validation in kwargs
      skip = kwargs.pop("skip_validation", skip_validation_by_default)
      if skip:
        return func(*args, **kwargs)

      # Bind arguments
      bound_args = sig.bind(*args, **kwargs)
      bound_args.apply_defaults()

      # Validate arguments
      for name, value in bound_args.arguments.items():
        if name in arg_validators:
          for v in arg_validators[name]:
            v.validate(value)

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
