from __future__ import annotations

import functools
import inspect
import typing
from typing import (
  TYPE_CHECKING,
  Annotated,
  Any,
  Literal,
  ParamSpec,
  get_args,
  get_origin,
  overload,
  override,
)

from loguru import logger
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from collections.abc import Callable

# Validated alias for Annotated
Validated = Annotated


# Validator Classes


class Validator[T]:
  """Base class for validators."""

  def validate(self, data: T) -> T:
    """Validate the data and return it (potentially modified)."""
    return data


class ValidatorMarker:
  """Base class for validator markers that don't perform validation.

  Markers like Nullable and MaybeEmpty are used to opt-out of default
  validation behaviors and don't need a validate method.
  """


class Finite(Validator[pd.Series | pd.DataFrame]):
  """Validator for finite values (no Inf, no NaN)."""

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, (pd.Series, pd.DataFrame)) and not np.all(
      np.isfinite(data.values)  # type: ignore[arg-type]
    ):
      raise ValueError("Data must be finite (no Inf, no NaN)")
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


class Nullable(ValidatorMarker):
  """Marker to allow NaN values (opt-out of default NonNaN)."""


class MaybeEmpty(ValidatorMarker):
  """Marker to allow empty data (opt-out of default NonEmpty)."""


class NonNaN(Validator[pd.Series | pd.DataFrame]):
  """Validator for non-NaN values."""

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(
      np.isnan(data.values)  # type: ignore[arg-type]
    ):
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


class Datetime(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for datetime index or values."""

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, pd.Index) and not isinstance(data, pd.DatetimeIndex):
      raise ValueError("Index must be DatetimeIndex")
    if isinstance(data, (pd.Series, pd.DataFrame)) and not isinstance(
      data.index,
      pd.DatetimeIndex,  # type: ignore[union-attr]
    ):
      raise ValueError("Index must be DatetimeIndex")
    return data


class UniqueIndex(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for unique index."""

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, pd.Index) and not data.is_unique:
      raise ValueError("Index must be unique")
    if isinstance(data, (pd.Series, pd.DataFrame)) and not data.index.is_unique:  # type: ignore[union-attr]
      raise ValueError("Index must be unique")
    return data


class NoTimeGaps(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for no time gaps in DatetimeIndex."""

  def __init__(self, freq: str) -> None:
    super().__init__()
    self.freq = freq

  def __class_getitem__(cls, item: str) -> NoTimeGaps:
    return cls(item)

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    index = None
    if isinstance(data, pd.Index):
      index = data
    elif isinstance(data, (pd.Series, pd.DataFrame)):
      index = data.index  # type: ignore[union-attr]

    if index is not None:
      if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("NoTimeGaps requires a DatetimeIndex")

      if index.empty:
        return data

      expected_range = pd.date_range(start=index.min(), end=index.max(), freq=self.freq)  # type: ignore[union-attr]
      if len(index) != len(expected_range):
        raise ValueError(f"Time gaps detected with frequency '{self.freq}'")

      # Check if all expected timestamps are present
      # Using difference is faster than checking equality of sets or lists
      if not expected_range.difference(index).empty:  # type: ignore[union-attr]
        raise ValueError(f"Time gaps detected with frequency '{self.freq}'")

    return data


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
      # Check all columns? Or just that the dataframe contains homogenous data?
      # Usually IsDtype on a DataFrame implies all columns match.
      for col in data.columns:  # type: ignore[var-annotated]
        if data[col].dtype != self.dtype:  # type: ignore[union-attr]
          msg = f"Column '{col}' must be of type {self.dtype}, got {data[col].dtype}"  # type: ignore[union-attr]
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


class Ge(Validator[pd.Series | pd.DataFrame]):
  """Validator that data >= target (unary) or col1 >= col2 >= ... (n-ary)."""

  def __init__(self, *targets: str | float | int) -> None:
    super().__init__()
    self.targets = targets

  def __class_getitem__(cls, items: object) -> Ge:
    if get_origin(items) is Literal:
      items = get_args(items)

    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)  # type: ignore[arg-type]

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if len(self.targets) == 1:
      # Unary comparison
      target = self.targets[0]
      if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(data.values < target):
        raise ValueError(f"Data must be >= {target}")
    else:
      # Column comparison
      if not isinstance(data, pd.DataFrame):
        raise TypeError("Column comparison requires a pandas DataFrame")

      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]

        if not isinstance(col1, str) or not isinstance(col2, str):
          raise TypeError("Column comparison requires string column names")

        if (
          col1 in data.columns
          and col2 in data.columns
          and np.any(data[col1].values < data[col2].values)
        ):
          raise ValueError(f"{col1} must be >= {col2}")

    return data


class Le(Validator[pd.Series | pd.DataFrame]):
  """Validator that data <= target (unary) or col1 <= col2 <= ... (n-ary)."""

  def __init__(self, *targets: str | float | int) -> None:
    super().__init__()
    self.targets = targets

  def __class_getitem__(cls, items: object) -> Le:
    if get_origin(items) is Literal:
      items = get_args(items)

    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)  # type: ignore[arg-type]

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if len(self.targets) == 1:
      # Unary comparison
      target = self.targets[0]
      if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(data.values > target):
        raise ValueError(f"Data must be <= {target}")
    else:
      # Column comparison
      if not isinstance(data, pd.DataFrame):
        raise TypeError("Column comparison requires a pandas DataFrame")

      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]

        if not isinstance(col1, str) or not isinstance(col2, str):
          raise TypeError("Column comparison requires string column names")

        if (
          col1 in data.columns
          and col2 in data.columns
          and np.any(data[col1].values > data[col2].values)
        ):
          raise ValueError(f"{col1} must be <= {col2}")

    return data


class Gt(Validator[pd.Series | pd.DataFrame]):
  """Validator that data > target (unary) or col1 > col2 > ... (n-ary)."""

  def __init__(self, *targets: str | float | int) -> None:
    super().__init__()
    self.targets = targets

  def __class_getitem__(cls, items: object) -> Gt:
    if get_origin(items) is Literal:
      items = get_args(items)

    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)  # type: ignore[arg-type]

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if len(self.targets) == 1:
      # Unary comparison
      target = self.targets[0]
      if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(data.values <= target):
        raise ValueError(f"Data must be > {target}")
    else:
      # Column comparison
      if not isinstance(data, pd.DataFrame):
        raise TypeError("Column comparison requires a pandas DataFrame")

      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]

        if not isinstance(col1, str) or not isinstance(col2, str):
          raise TypeError("Column comparison requires string column names")

        if (
          col1 in data.columns
          and col2 in data.columns
          and np.any(data[col1].values <= data[col2].values)
        ):
          raise ValueError(f"{col1} must be > {col2}")

    return data


class Lt(Validator[pd.Series | pd.DataFrame]):
  """Validator that data < target (unary) or col1 < col2 < ... (n-ary)."""

  def __init__(self, *targets: str | float | int) -> None:
    super().__init__()
    self.targets = targets

  def __class_getitem__(cls, items: object) -> Lt:
    if get_origin(items) is Literal:
      items = get_args(items)

    if not isinstance(items, tuple):
      items = (items,)
    return cls(*items)  # type: ignore[arg-type]

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if len(self.targets) == 1:
      # Unary comparison
      target = self.targets[0]
      if isinstance(data, (pd.Series, pd.DataFrame)) and np.any(data.values >= target):
        raise ValueError(f"Data must be < {target}")
    else:
      # Column comparison
      if not isinstance(data, pd.DataFrame):
        raise TypeError("Column comparison requires a pandas DataFrame")

      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]

        if not isinstance(col1, str) or not isinstance(col2, str):
          raise TypeError("Column comparison requires string column names")

        if (
          col1 in data.columns
          and col2 in data.columns
          and np.any(data[col1].values >= data[col2].values)
        ):
          raise ValueError(f"{col1} must be < {col2}")

    return data


class MonoUp(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for monotonically increasing values or index."""

  @override
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

  @override
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


P = ParamSpec("P")
R = typing.TypeVar("R")


@overload
def validated[**P, R](
  func: Callable[P, R],
) -> Callable[P, R]: ...


@overload
def validated[**P, R](
  *, skip_validation_by_default: bool = False, warn_only_by_default: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R | None]]: ...


def validated[**P, R](
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

  def decorator(
    func: Callable[P, R],
  ) -> Callable[P, R | None]:
    # Inspect function signature
    sig = inspect.signature(func)
    type_hints = typing.get_type_hints(func, include_extras=True)

    # Pre-compute validators for each argument
    arg_validators: dict[str, list[Validator[Any]]] = {}  # type: ignore[misc]
    for name in sig.parameters:
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

          # Check if any validator is HasColumn or HasColumns
          # If so, we implicitly allow NaNs in the container (disable default NonNaN)
          # because the user is likely focusing on specific columns.
          has_column_validator = False
          for v in validators:
            if isinstance(v, (HasColumn, HasColumns)):
              has_column_validator = True
              break

          if has_column_validator:
            is_nullable = True
            maybe_empty = True

          # Add defaults if not opted out
          # Only apply defaults if the type is pandas Series/DataFrame
          # We can check the first arg of Annotated (the type)
          annotated_type = args[0]
          is_pandas = False
          try:
            if issubclass(annotated_type, (pd.Series, pd.DataFrame)):
              is_pandas = True
          except TypeError:
            # annotated_type might be a generic alias or something else
            # rough check by name or module?
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
    return decorator

  return decorator(func)


def _instantiate_validator(
  item: object,
) -> Validator[Any] | ValidatorMarker | None:  # type: ignore[misc]
  """Helper to instantiate a validator from a type or instance."""
  if isinstance(item, type):
    try:
      if issubclass(item, (Validator, ValidatorMarker)):
        return item()  # type: ignore[return-value]
    except TypeError:
      pass
  if isinstance(item, (Validator, ValidatorMarker)):
    return item
  return None
