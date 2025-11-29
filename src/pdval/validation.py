"""Pandas validation using Annotated types and decorators (powered by pandera).

This module provides a decorator-based validation system for pandas DataFrames
and Series using Python's Annotated types, wrapping pandera for validation logic.
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
import pandera.pandas as pa
from pandera.errors import SchemaError

# Validated alias for Annotated
Validated = Annotated

# Validator Classes


class Validator(ABC):
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
    
    return data


class Finite(Validator):
  """Validator for finite values (no Inf, no NaN)."""

  def get_checks(self) -> list[pa.Check]:
    # Use optimized numpy check
    return [pa.Check(lambda s: np.isfinite(s.values).all(), error="must be finite")]


class NonNaN(Validator):
  """Validator for non-NaN values."""

  def get_checks(self) -> list[pa.Check]:
    return [pa.Check(lambda s: pd.notna(s).all(), error="must not contain NaN")]


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

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if not isinstance(data, pd.DataFrame):
      raise TypeError("HasColumns validator requires a pandas DataFrame")
      
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

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Ge validator requires a pandas DataFrame")
    return super().validate(data)


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

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Le validator requires a pandas DataFrame")
    return super().validate(data)


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

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Gt validator requires a pandas DataFrame")
    return super().validate(data)


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

  def validate(self, data: Any) -> Any:  # noqa: ANN401
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Lt validator requires a pandas DataFrame")
    return super().validate(data)


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

  # Index validation is handled in _build_dataframe_schema / _build_series_schema
  def validate(self, data: Any) -> Any:  # noqa: ANN401
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


def _build_dataframe_schema(validators: list[Validator]) -> pa.DataFrameSchema:
  """Build DataFrame schema from validators."""
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
        # If column exists, we should ideally merge checks.
        # But pa.Column is immutable-ish.
        # For simplicity in this implementation, we might overwrite or ignore duplicates.
        # A robust implementation would collect all checks per column first.
        pass 
      
  # Re-scanning to build schema properly
  
  # Global DataFrame checks (e.g. Ge, Le)
  for v in validators:
    if isinstance(v, (Ge, Le, Gt, Lt, Finite, NonNaN, NonNegative, Positive)):
       checks.extend(v.get_checks())
    elif isinstance(v, MonoUp):
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
  return pa.DataFrameSchema(
    columns=columns,
    checks=checks,
    index=(
        pa.Index(dtype=index_dtype, checks=index_checks)
        if (index_checks or index_dtype)
        else None
    ),
    coerce=False,
  )


def _build_series_schema(validators: list[Validator]) -> pa.SeriesSchema:
  """Build Series schema from validators."""
  checks = []
  dtype = None
  
  for v in validators:
    if not isinstance(v, (Index, Datetime)):
        checks.extend(v.get_checks())

  index_checks = []
  index_dtype = None
  
  for v in validators:
      if isinstance(v, Datetime):
          index_dtype = "datetime64[ns]"
      elif isinstance(v, Index):
           for sub_v_item in v.validators:
                sub_v = _instantiate_validator(sub_v_item)
                if sub_v:
                    if isinstance(sub_v, Datetime):
                        index_dtype = "datetime64[ns]"
                    else:
                        index_checks.extend(sub_v.get_checks())

  return pa.SeriesSchema(
      dtype=dtype,
      checks=checks,
      index=(
          pa.Index(dtype=index_dtype, checks=index_checks)
          if (index_checks or index_dtype)
          else None
      ),
  )


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
  # 1. Compute metadata ONCE at decoration time
  sig = inspect.signature(func)

  # Resolve type hints to handle string annotations
  try:
    type_hints = typing.get_type_hints(func, include_extras=True)
  except Exception:
    type_hints = {}

  # Pre-compute a mapping of arg_name -> pa.Schema
  arg_schemas: dict[str, pa.DataFrameSchema | pa.SeriesSchema] = {}

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
        v = _instantiate_validator(item)
        if v:
          validators.append(v)

      if validators:
        # Determine if we should build DataFrame or Series schema
        # We can look at the type hint (metadata[0])
        arg_type = metadata[0]
        if isinstance(arg_type, type) and issubclass(arg_type, pd.DataFrame):
            arg_schemas[name] = _build_dataframe_schema(validators)
        elif isinstance(arg_type, type) and issubclass(arg_type, pd.Series):
            arg_schemas[name] = _build_series_schema(validators)
        else:
            # Fallback: try to build both or decide at runtime?
            # If we don't know the type, we can't pre-build the exact schema type easily
            # unless we assume one.
            # But wait, Annotated[pd.DataFrame, ...] gives us the type!
            # If the user used Annotated[Any, ...], we might have trouble.
            # Let's assume standard usage.
            pass

  @functools.wraps(func)
  def wrapper(*args: P.args, skip_validation: bool = False, **kwargs: P.kwargs) -> R:
    if skip_validation:
      return func(*args, **kwargs)

    # 2. Bind args
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    # 3. Validate using pre-computed schemas
    for name, value in bound.arguments.items():
      if value is None:
        continue

      if name in arg_schemas:
        try:
            arg_schemas[name].validate(value)
        except Exception as e:
            raise e

    return func(*bound.args, **bound.kwargs)

  return wrapper  # type: ignore[return-value]
