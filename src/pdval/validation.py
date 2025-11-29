"""Pandas validation using Annotated types and decorators (powered by pandera).

This module provides a decorator-based validation system for pandas DataFrames
and Series using Python's Annotated types, wrapping pandera for validation logic.
"""

from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Callable
from typing import Annotated, Any, ParamSpec, TypeVar, get_args, get_origin

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError

# Validated alias for Annotated
Validated = Annotated

# Validator Classes


class Validator:
  """Base class for validators."""

  def get_checks(self) -> list[pa.Check]:
    """Return a list of pandera Checks."""
    return []

  def validate(self, data: Any) -> Any:  # noqa: ANN401
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
    else:
      # Fallback for non-pandas types (tests expect them to be ignored or handled)
      # But wait, some validators might work on other types?
      # The old validators often checked isinstance(data, (pd.Series, pd.DataFrame))
      # If passed int, they might just return it.
      # Let's check if we should return data or raise.
      # Old behavior: "if isinstance(...) ... else return data"
      pass
    
    return data


class Finite(Validator):
  """Validator for finite values (no Inf, no NaN)."""

  def get_checks(self) -> list[pa.Check]:
    return [pa.Check(lambda s: np.isfinite(s).all(), error="must be finite")]


class NonNaN(Validator):
  """Validator for non-NaN values."""

  def get_checks(self) -> list[pa.Check]:
    # pa.Check.notna() is not a thing, use pa.Check(lambda s: s.notna().all())
    # Or better, use built-in check if available.
    # Actually pa.Check.notna IS available in newer pandera versions
    # but maybe not the one installed?
    # Let's use robust lambda checks.
    return [pa.Check(lambda s: s.notna().all(), error="must not contain NaN")]


class NonNegative(Validator):
  """Validator for non-negative values (>= 0)."""

  def get_checks(self) -> list[pa.Check]:
    return [pa.Check.ge(0)]


class Positive(Validator):
  """Validator for positive values (> 0)."""

  def get_checks(self) -> list[pa.Check]:
    return [pa.Check.gt(0)]


class Datetime(Validator):
  """Validator for datetime index or values."""

  # Datetime validation is usually handled by schema dtype, but we can add a check
  def get_checks(self) -> list[pa.Check]:
    return []  # Handled by coerce/dtype in schema construction if possible


class HasColumns(Validator):
  """Validator for presence of specific columns in DataFrame."""

  def __init__(self, columns: list[str]) -> None:
    self.columns = columns

  def __class_getitem__(cls, items: str | tuple[str, ...]) -> HasColumns:
    if isinstance(items, str):
      items = (items,)
    return cls(list(items))

  # HasColumns is structural, handled at Schema level, not Check level
  # But we can simulate it with a check if needed, though Schema is better.
  # For now, we'll handle it in the schema construction logic.

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, pd.DataFrame):
      missing = [col for col in self.columns if col not in data.columns]
      if missing:
        # Raise SchemaError to match pandera behavior
        raise SchemaError(
            schema=None,
            data=data,
            message=f"Missing columns: {missing}"
        )
    return data


class Ge(Validator):
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
                df[self.col1] >= df[self.col2]
                if isinstance(df, pd.DataFrame)
                and self.col1 in df.columns
                and self.col2 in df.columns
                else True
            ),
            name=f"{self.col1} >= {self.col2}",
            error=f"{self.col1} must be >= {self.col2}",
        )
    ]


class Le(Validator):
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
                df[self.col1] <= df[self.col2]
                if isinstance(df, pd.DataFrame)
                and self.col1 in df.columns
                and self.col2 in df.columns
                else True
            ),
            name=f"{self.col1} <= {self.col2}",
            error=f"{self.col1} must be <= {self.col2}",
        )
    ]


class Gt(Validator):
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
                df[self.col1] > df[self.col2]
                if isinstance(df, pd.DataFrame)
                and self.col1 in df.columns
                and self.col2 in df.columns
                else True
            ),
            name=f"{self.col1} > {self.col2}",
            error=f"{self.col1} must be > {self.col2}",
        )
    ]


class Lt(Validator):
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
                df[self.col1] < df[self.col2]
                if isinstance(df, pd.DataFrame)
                and self.col1 in df.columns
                and self.col2 in df.columns
                else True
            ),
            name=f"{self.col1} < {self.col2}",
            error=f"{self.col1} must be < {self.col2}",
        )
    ]


class MonoUp(Validator):
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


class MonoDown(Validator):
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


class Index(Validator):
  """Validator for index properties."""

  def __init__(self, *validators: Validator | type[Validator]) -> None:
    self.validators = validators

  def __class_getitem__(
    cls, items: type[Validator] | tuple[type[Validator], ...]
  ) -> Index:
    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, (pd.Series, pd.DataFrame)):
      index = data.index
      # Validate index as if it were a Series (mostly) or check properties
      
      # Special handling for Datetime validator on Index
      for v_item in self.validators:
        v = _instantiate_validator(v_item)
        if v:
             if isinstance(v, Datetime):
                 if not isinstance(index, pd.DatetimeIndex):
                     raise SchemaError(
                         schema=None, data=data, message="Index must be DatetimeIndex"
                     )
             elif isinstance(v, (MonoUp, MonoDown)):
                 # These have get_checks which return pa.Check.monotonic...
                 # We can run these checks on the index converted to Series
                 s_index = index.to_series().reset_index(drop=True)
                 v.validate(s_index)
             else:
                 # Generic validation on index values
                 s_index = index.to_series().reset_index(drop=True)
                 v.validate(s_index)
    return data


class HasColumn(Validator):
  """Wrapper to apply validators to specific DataFrame columns."""

  def __init__(self, column: str, *validators: Validator | type[Validator]) -> None:
    self.column = column
    self.validators = validators

  def __class_getitem__(cls, items: str | tuple[str, ...]) -> HasColumn:
    if isinstance(items, str):
      return cls(items)
    column = items[0]
    validators = items[1:] if len(items) > 1 else ()
    return cls(column, *validators)  # type: ignore[arg-type]

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if isinstance(data, pd.DataFrame):
      if self.column not in data.columns:
         raise SchemaError(
            schema=None,
            data=data,
            message=f"Column '{self.column}' not found in DataFrame"
        )
      
      # Extract column as Series and validate
      series = data[self.column]
      for v_item in self.validators:
        v = _instantiate_validator(v_item)
        if v:
          v.validate(series)
    return data


P = ParamSpec("P")
R = TypeVar("R")


def _instantiate_validator(item: Any) -> Validator | None:  # noqa: ANN401
  """Helper to instantiate a validator from a type or instance."""
  if isinstance(item, type) and issubclass(item, Validator):
    return item()
  if isinstance(item, Validator):
    return item
  return None


def _validate_single_argument(
  name: str,
  value: Any,  # noqa: ANN401
  annotation: Any,  # noqa: ANN401
  bound_args: inspect.BoundArguments,
) -> None:
  """Validate a single argument using pandera."""
  origin = get_origin(annotation)

  # Handle Optional/Union
  if origin is typing.Union:
    args = get_args(annotation)
    if value is None and type(None) in args:
      return
    for arg in args:
      if get_origin(arg) is Annotated:
        annotation = arg
        origin = Annotated
        break

  if origin is not Annotated:
    return

  metadata = get_args(annotation)
  
  # Collect validators
  validators: list[Validator] = []
  for item in metadata[1:]:
    v = _instantiate_validator(item)
    if v:
      validators.append(v)

  if not validators:
    return

  # Construct Pandera Schema
  if isinstance(value, pd.DataFrame):
    _validate_dataframe(value, validators)
  elif isinstance(value, pd.Series):
    _validate_series(value, validators)
  elif isinstance(value, pd.Index):
    _validate_index(value, validators)


def _validate_dataframe(df: pd.DataFrame, validators: list[Validator]) -> None:
  """Validate DataFrame using pandera."""
  checks = []
  columns: dict[str, pa.Column] = {}
  index_checks = []
  index_dtype = None

  # 1. Collect global checks and column-specific checks
  for v in validators:
    if isinstance(v, HasColumns):
      for col in v.columns:
        if col not in columns:
          columns[col] = pa.Column(required=True)
    elif isinstance(v, HasColumn):
      # Column specific validation
      col_checks = []
      for sub_v_item in v.validators:
        sub_v = _instantiate_validator(sub_v_item)
        if sub_v:
          col_checks.extend(sub_v.get_checks())
      
      # Merge into existing column definition or create new
      if v.column in columns:
        # Append checks to existing column
        # (this is a bit tricky with pandera objects being immutable-ish)
        # We'll just re-create it or append to a list we track
        # Simpler: just add to a separate dict of checks and merge later?
        # For now, let's assume we can just overwrite/add.
        # Actually pa.Column checks arg is a list.
        pass 
      
      # Better approach: store checks in a dict first
      # But HasColumn might be used multiple times for same column?
      # Let's simplify:
      pass

  # Re-scanning to build schema properly
  
  # Global DataFrame checks (e.g. Ge, Le)
  for v in validators:
    if isinstance(v, (Ge, Le, Gt, Lt, Finite, NonNaN, NonNegative, Positive)):
       checks.extend(v.get_checks())
    elif isinstance(v, MonoUp):
       # For DataFrame, MonoUp checks ALL columns? Old behavior:
       # "for col in data.columns: if not monotonic... raise"
       # So we should add a check that iterates columns or check all columns
       # Pandera doesn't have a "all columns monotonic" check easily without iterating
       # We can add a global check:
       checks.append(
           pa.Check(
               lambda df: df.apply(lambda col: col.is_monotonic_increasing).all(),
               error="Values must be monotonically increasing",
           )
       )
    elif isinstance(v, MonoDown):
       checks.append(
           pa.Check(
               lambda df: df.apply(lambda col: col.is_monotonic_decreasing).all(),
               error="Values must be monotonically decreasing",
           )
       )

  # Column definitions
  # We need to handle HasColumn and HasColumns
  # And also implicit column checks if any?
  
  # Let's handle HasColumn explicitly
  column_checks_map: dict[str, list[pa.Check]] = {}
  required_columns = set()

  for v in validators:
    if isinstance(v, HasColumns):
      required_columns.update(v.columns)
    elif isinstance(v, HasColumn):
      required_columns.add(v.column)
      if v.column not in column_checks_map:
        column_checks_map[v.column] = []
      
      for sub_v_item in v.validators:
        sub_v = _instantiate_validator(sub_v_item)
        if sub_v:
          column_checks_map[v.column].extend(sub_v.get_checks())

  # Construct pa.Column objects
  for col_name in required_columns:
    columns[col_name] = pa.Column(
        checks=column_checks_map.get(col_name, []), required=True
    )

  # Index validation
  for v in validators:
    if isinstance(v, Index):
      for sub_v_item in v.validators:
        sub_v = _instantiate_validator(sub_v_item)
        if sub_v:
          if isinstance(sub_v, Datetime):
             index_dtype = "datetime64[ns]" # or generic datetime
          else:
             index_checks.extend(sub_v.get_checks())

  # Create Schema
  schema = pa.DataFrameSchema(
    columns=columns,
    checks=checks,
    index=(
        pa.Index(dtype=index_dtype, checks=index_checks)
        if (index_checks or index_dtype)
        else None
    ),
    coerce=False,  # Don't coerce by default, just validate
  )
  
  # Validate
  schema.validate(df)


def _validate_series(series: pd.Series, validators: list[Validator]) -> None:
  """Validate Series using pandera."""
  checks = []
  dtype = None
  
  for v in validators:
    if isinstance(v, Datetime):
        # For Series, Datetime usually means values are datetime?
        # Old code: "Index must be DatetimeIndex" check was in Datetime
        # validator but it checked data.index if it was Series/DataFrame.
        # Wait, the old Datetime validator checked the INDEX of the
        # series/df, not the values?
        # Let's re-read old code.
        # Old Datetime.validate:
        # if isinstance(data, pd.Index) ...
        # if isinstance(data, (pd.Series, pd.DataFrame)) and
        # not isinstance(data.index, pd.DatetimeIndex)
        # So it validated the INDEX.
        pass
    else:
        checks.extend(v.get_checks())

  # Special handling for Datetime which validates INDEX of Series
  # But pa.SeriesSchema validates values.
  # If we want to validate the index of the Series, we need pa.SeriesSchema(index=...)
  
  index_checks = []
  index_dtype = None
  
  for v in validators:
      if isinstance(v, Datetime):
          index_dtype = "datetime64[ns]"
      elif isinstance(v, Index):
           # Nested Index validator
           for sub_v_item in v.validators:
                sub_v = _instantiate_validator(sub_v_item)
                if sub_v:
                    if isinstance(sub_v, Datetime):
                        index_dtype = "datetime64[ns]"
                    else:
                        index_checks.extend(sub_v.get_checks())

  schema = pa.SeriesSchema(
      dtype=dtype,
      checks=checks,
      index=(
          pa.Index(dtype=index_dtype, checks=index_checks)
          if (index_checks or index_dtype)
          else None
      ),
  )
  
  schema.validate(series)


def _validate_index(index: pd.Index, validators: list[Validator]) -> None:
    """Validate Index using pandera."""
    # pa.Index schema is for a column-like index in a dataframe schema.
    # We can just use SeriesSchema on the index converted to series?
    # Or just check manually if pandera doesn't support standalone
    # Index validation easily.
    
    # Actually, we can use pa.SeriesSchema(index=...) and validate
    # a dummy series? Or just use the checks directly?
    
    # Let's try to use checks directly for simplicity if schema is too heavy
    checks = []
    
    for v in validators:
        if isinstance(v, Datetime):
            if not isinstance(index, pd.DatetimeIndex):
                 raise ValueError("Index must be DatetimeIndex")
        else:
            checks.extend(v.get_checks())
            
    # Run checks manually
    # Or wrap in SeriesSchema
    s_index = index.to_series()
    # Reset index so we validate values
    s_index.reset_index(drop=True, inplace=True) 
    
    # Wait, index.to_series() keeps the index as index?
    # If we want to validate the index values, we treat them as a Series.
    
    schema = pa.SeriesSchema(checks=checks)
    # We might need to handle the Datetime check separately or via dtype
    
    schema.validate(s_index)


def _validate_arguments(
  func: Callable[..., Any], bound_args: inspect.BoundArguments
) -> None:
  """Validate arguments based on Annotated types."""
  try:
    type_hints = typing.get_type_hints(func, include_extras=True)
  except Exception:
    return

  sig = inspect.signature(func)

  for name, value in bound_args.arguments.items():
    if name == "self":
      continue

    annotation = type_hints.get(name, sig.parameters[name].annotation)
    _validate_single_argument(name, value, annotation, bound_args)


def validated(func: Callable[P, R]) -> Callable[P, R]:  # noqa: UP047
  """Decorator to validate function arguments based on Annotated types.

  The decorator automatically adds a `skip_validation` parameter to the function.
  When `skip_validation=False` (default), validation is performed. When
  `skip_validation=True`, validation is skipped for maximum performance.

  Args:
    func: The function to decorate.

  Returns:
    The decorated function with automatic validation support.
  """

  @functools.wraps(func)
  def wrapper(
    *args: P.args, skip_validation: bool = False, **kwargs: P.kwargs
  ) -> R:
    if not skip_validation:
      # Need to bind args for validation
      sig = inspect.signature(func)
      bound_args = sig.bind(*args, **kwargs)
      bound_args.apply_defaults()
      _validate_arguments(func, bound_args)
      # Return with validated bound args
      return func(*bound_args.args, **bound_args.kwargs)

    # Fast path: no validation, just call directly
    return func(*args, **kwargs)

  return wrapper  # type: ignore[return-value]
