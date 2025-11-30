"""Pandas validation using Annotated types and decorators (powered by pandera).

This module provides a decorator-based validation system for pandas DataFrames
and Series using Python's Annotated types, wrapping pandera for validation logic.
"""

from __future__ import annotations

# pyright: reportOperatorIssue=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportInvalidTypeForm=false
import functools
import inspect
import typing
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
import pandera.pandas as pa
from loguru import logger
from pandera.errors import SchemaError

# Validated alias for Annotated
Validated = Annotated


# Validator Classes


class Validator[T]:
  """Base class for validators."""

  def get_checks(self) -> list[pa.Check]:
    """Return a list of pandera Checks."""
    return []

  def validate(self, data: T) -> T:
    """Validate data directly using pandera checks."""
    checks = self.get_checks()
    if not checks:
      return data

    try:
      if isinstance(data, pd.DataFrame):
        # For DataFrame, we need to know if checks are global or column-specific
        # Most simple validators (Finite, NonNaN) are element-wise or global
        # But pandera Checks on DataFrame are usually global unless specified
        # Let's assume simple validators apply to the whole dataframe
        schema = pa.DataFrameSchema(checks=checks, coerce=False)
        schema.validate(data)
      elif isinstance(data, pd.Series):
        schema = pa.SeriesSchema(checks=checks, coerce=False, nullable=True)
        schema.validate(data)
    except SchemaError as e:
      # Convert SchemaError to ValueError to match master branch behavior
      raise ValueError(str(e)) from e

    return data


class Finite(Validator[pd.Series | pd.DataFrame]):
  """Validator for finite values (no Inf, no NaN)."""

  def get_checks(self) -> list[pa.Check]:
    # Use optimized numpy check
    return [
      pa.Check(
        lambda s: np.isfinite(s.values).all(), error="must be finite", ignore_na=False
      )
    ]


class NonEmpty(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-empty data."""

  def get_checks(self) -> list[pa.Check]:
    return [pa.Check(lambda data: not data.empty, error="Data must not be empty")]

  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, pd.Index):
      if data.empty:
        raise ValueError("Data must not be empty")
      return data
    return super().validate(data)


class Nullable(Validator[Any]):
  """Marker to allow NaN values (opt-out of default NonNaN)."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    return data


class MaybeEmpty(Validator[Any]):
  """Marker to allow empty data (opt-out of default NonEmpty)."""

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    return data


class NonNaN(Validator[pd.Series | pd.DataFrame]):
  """Validator for non-NaN values."""

  def get_checks(self) -> list[pa.Check]:
    return [
      pa.Check(
        lambda s: pd.notna(s).all(), error="must not contain NaN", ignore_na=False
      )
    ]


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

  def __class_getitem__(cls, item: str) -> NoTimeGaps:
    return cls(item)

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
      if not expected_range.difference(index).empty:
        raise ValueError(f"Time gaps detected with frequency '{self.freq}'")

    return data


class IsDtype(Validator[pd.Series | pd.DataFrame]):
  """Validator for specific dtype."""

  def __init__(self, dtype: str | type | np.dtype) -> None:
    self.dtype = np.dtype(dtype)

  def __class_getitem__(cls, item: str | type | np.dtype) -> IsDtype:
    return cls(item)

  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, pd.Series):
      if data.dtype != self.dtype:
        raise ValueError(f"Data must be of type {self.dtype}, got {data.dtype}")
    elif isinstance(data, pd.DataFrame):
      for col in data.columns:
        if data[col].dtype != self.dtype:
          msg = f"Column '{col}' must be of type {self.dtype}, got {data[col].dtype}"
          raise ValueError(msg)
    return data


class HasColumns(Validator[pd.DataFrame]):
  """Validator for presence of specific columns in DataFrame.

  Can also apply validators to the specified columns:
  HasColumns["a", "b", Finite, Positive]
  """

  def __init__(
    self, columns: list[str], validators: tuple[Validator[Any], ...] = ()
  ) -> None:
    self.columns = columns
    self.validators = validators

  def __class_getitem__(cls, items: Any) -> HasColumns:  # noqa: ANN401
    if get_origin(items) is Literal:
      items = get_args(items)

    if not isinstance(items, tuple):
      items = (items,)

    # Parse columns and validators
    columns: list[str] = []
    validators: list[Validator[Any]] = []

    for item in items:
      if get_origin(item) is Literal:
        args = get_args(item)
        for arg in args:
          if isinstance(arg, str):
            columns.append(arg)
        continue

      if isinstance(item, str):
        columns.append(item)
      else:
        v = _instantiate_validator(item)
        if v:
          validators.append(v)

    return cls(columns, tuple(validators))

  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("HasColumns validator requires a pandas DataFrame")

    missing = [col for col in self.columns if col not in data.columns]
    if missing:
      raise ValueError(f"Missing columns: {missing}")

    final_validators: list[Validator[Any]] = []
    is_nullable = False
    maybe_empty = False

    if self.validators:
      for v in self.validators:
        if isinstance(v, Nullable):
          is_nullable = True
        elif isinstance(v, MaybeEmpty):
          maybe_empty = True
        else:
          final_validators.append(v)

    if not is_nullable:
      final_validators.insert(0, NonNaN())
    if not maybe_empty:
      final_validators.insert(0, NonEmpty())

    if final_validators:
      for col in self.columns:
        column_data = data[col]
        for v in final_validators:
          column_data = v.validate(column_data)

    return data


class Ge(Validator[pd.Series | pd.DataFrame]):
  """Validator that data >= target (unary) or col1 >= col2 >= ... (n-ary)."""

  def __init__(self, *targets: str | float | int) -> None:
    self.targets = targets

  def __class_getitem__(cls, items: Any) -> Ge:  # noqa: ANN401
    if get_origin(items) is Literal:
      items = get_args(items)

    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)

  def get_checks(self) -> list[pa.Check]:
    checks = []
    if len(self.targets) == 1:
      target = self.targets[0]
      checks.append(
        pa.Check(
          lambda s: np.all(s.values >= target), error=f"Data must be >= {target}"
        )
      )
    else:
      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]
        # Ensure col1/col2 are strings for column access
        if isinstance(col1, str) and isinstance(col2, str):
          checks.append(
            pa.Check(
              lambda df, c1=col1, c2=col2: (
                np.all(df[c1].values >= df[c2].values)
                if isinstance(df, pd.DataFrame)
                and c1 in df.columns
                and c2 in df.columns
                else True
              ),
              name=f"{col1} >= {col2}",
              error=f"{col1} must be >= {col2}",
            )
          )
    return checks

  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if len(self.targets) > 1 and not isinstance(data, pd.DataFrame):
      raise TypeError("Ge validator requires a pandas DataFrame for column comparison")
    return super().validate(data)


class Le(Validator[pd.Series | pd.DataFrame]):
  """Validator that data <= target (unary) or col1 <= col2 <= ... (n-ary)."""

  def __init__(self, *targets: str | float | int) -> None:
    self.targets = targets

  def __class_getitem__(cls, items: Any) -> Le:  # noqa: ANN401
    if get_origin(items) is Literal:
      items = get_args(items)

    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)

  def get_checks(self) -> list[pa.Check]:
    checks = []
    if len(self.targets) == 1:
      target = self.targets[0]
      checks.append(
        pa.Check(
          lambda s: np.all(s.values <= target), error=f"Data must be <= {target}"
        )
      )
    else:
      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]
        if isinstance(col1, str) and isinstance(col2, str):
          checks.append(
            pa.Check(
              lambda df, c1=col1, c2=col2: (
                np.all(df[c1].values <= df[c2].values)
                if isinstance(df, pd.DataFrame)
                and c1 in df.columns
                and c2 in df.columns
                else True
              ),
              name=f"{col1} <= {col2}",
              error=f"{col1} must be <= {col2}",
            )
          )
    return checks

  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if len(self.targets) > 1 and not isinstance(data, pd.DataFrame):
      raise TypeError("Le validator requires a pandas DataFrame for column comparison")
    return super().validate(data)


class Gt(Validator[pd.Series | pd.DataFrame]):
  """Validator that data > target (unary) or col1 > col2 > ... (n-ary)."""

  def __init__(self, *targets: str | float | int) -> None:
    self.targets = targets

  def __class_getitem__(cls, items: Any) -> Gt:  # noqa: ANN401
    if get_origin(items) is Literal:
      items = get_args(items)

    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)

  def get_checks(self) -> list[pa.Check]:
    checks = []
    if len(self.targets) == 1:
      target = self.targets[0]
      checks.append(
        pa.Check(lambda s: np.all(s.values > target), error=f"Data must be > {target}")
      )
    else:
      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]
        if isinstance(col1, str) and isinstance(col2, str):
          checks.append(
            pa.Check(
              lambda df, c1=col1, c2=col2: (
                np.all(df[c1].values > df[c2].values)
                if isinstance(df, pd.DataFrame)
                and c1 in df.columns
                and c2 in df.columns
                else True
              ),
              name=f"{col1} > {col2}",
              error=f"{col1} must be > {col2}",
            )
          )
    return checks

  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if len(self.targets) > 1 and not isinstance(data, pd.DataFrame):
      raise TypeError("Gt validator requires a pandas DataFrame for column comparison")
    return super().validate(data)


class Lt(Validator[pd.Series | pd.DataFrame]):
  """Validator that data < target (unary) or col1 < col2 < ... (n-ary)."""

  def __init__(self, *targets: str | float | int) -> None:
    self.targets = targets

  def __class_getitem__(cls, items: Any) -> Lt:  # noqa: ANN401
    if get_origin(items) is Literal:
      items = get_args(items)

    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)

  def get_checks(self) -> list[pa.Check]:
    checks = []
    if len(self.targets) == 1:
      target = self.targets[0]
      checks.append(
        pa.Check(lambda s: np.all(s.values < target), error=f"Data must be < {target}")
      )
    else:
      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]
        if isinstance(col1, str) and isinstance(col2, str):
          checks.append(
            pa.Check(
              lambda df, c1=col1, c2=col2: (
                np.all(df[c1].values < df[c2].values)
                if isinstance(df, pd.DataFrame)
                and c1 in df.columns
                and c2 in df.columns
                else True
              ),
              name=f"{col1} < {col2}",
              error=f"{col1} must be < {col2}",
            )
          )
    return checks

  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if len(self.targets) > 1 and not isinstance(data, pd.DataFrame):
      raise TypeError("Lt validator requires a pandas DataFrame for column comparison")
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
    if isinstance(data, pd.Index):
      if not data.is_monotonic_increasing:
        raise ValueError("Index must be monotonically increasing")
      return data
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
  """Validator for index properties."""

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
      s_index = data.index.to_series().reset_index(drop=True)
      for v_item in self.validators:
        v = _instantiate_validator(v_item)
        if v:
          if isinstance(v, Datetime):
            if not isinstance(data.index, pd.DatetimeIndex):
              raise ValueError("Index must be DatetimeIndex")
          else:
            v.validate(s_index)
    return data


class HasColumn(Validator[pd.DataFrame]):
  """Wrapper to apply validators to specific DataFrame columns."""

  def __init__(
    self,
    column: str | Any,  # noqa: ANN401
    *validators: Validator[Any] | type[Validator[Any]],
  ) -> None:
    self.column = column
    self.validators = validators

  def __getitem__(self, item: str) -> HasColumn:
    """Support for templating: CustomVal["col"]."""
    return HasColumn(item, *self.validators)

  def __class_getitem__(cls, items: Any) -> HasColumn:  # noqa: ANN401
    if get_origin(items) is Literal:
      args = get_args(items)
      if len(args) == 1:
        items = args[0]
      else:
        pass

    # Handle single column name
    if isinstance(items, (str, typing.TypeVar)):
      return cls(items)

    # Handle tuple: (column, validators...)
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

      # Determine validators to apply
      final_validators: list[Validator[Any]] = []
      is_nullable = False
      maybe_empty = False

      for validator_item in self.validators:
        v = _instantiate_validator(validator_item)
        if v:
          if isinstance(v, Nullable):
            is_nullable = True
          elif isinstance(v, MaybeEmpty):
            maybe_empty = True
          else:
            final_validators.append(v)

      if not is_nullable:
        final_validators.insert(0, NonNaN())
      if not maybe_empty:
        final_validators.insert(0, NonEmpty())

      for validator in final_validators:
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
  *, skip_validation_by_default: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


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
  """Decorator to validate function arguments based on Annotated types."""

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
          for arg in get_args(hint):
            if get_origin(arg) is Annotated:
              hint = arg
              break

        if get_origin(hint) is Annotated:
          args = get_args(hint)
          # First arg is the type, rest are metadata (validators)
          validators = []
          is_nullable = False
          maybe_empty = False

          for item in args[1:]:
            v = _instantiate_validator(item)
            if v:
              if isinstance(v, Nullable):
                is_nullable = True
              elif isinstance(v, MaybeEmpty):
                maybe_empty = True
              else:
                validators.append(v)

          # Implicitly allow NaNs if checking specific columns
          has_column_validator = False
          for v in validators:
            if isinstance(v, (HasColumn, HasColumns)):
              has_column_validator = True
              break

          if has_column_validator:
            is_nullable = True
            maybe_empty = True

          # Add defaults if not opted out
          annotated_type = args[0]
          is_pandas = False
          try:
            if issubclass(annotated_type, (pd.Series, pd.DataFrame)):
              is_pandas = True
          except TypeError:
            origin = get_origin(annotated_type)
            if origin is not None:
              if isinstance(origin, type) and issubclass(
                origin, (pd.Series, pd.DataFrame)
              ):
                is_pandas = True
            elif (
              hasattr(annotated_type, "__module__")
              and "pandas" in annotated_type.__module__
            ):
              is_pandas = True

          if is_pandas:
            if not is_nullable:
              validators.insert(0, NonNaN())
            if not maybe_empty:
              validators.insert(0, NonEmpty())

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