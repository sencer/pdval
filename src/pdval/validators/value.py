"""Value validators for Series and DataFrame data."""

from __future__ import annotations

from typing import Any, Literal, get_args, get_origin, override

import numpy as np
import pandas as pd

from pdval.base import Validator
from pdval.validators.comparison import Ge, Gt, Le, Lt


class Finite(Validator[pd.Series | pd.DataFrame]):
  """Validator for non-infinite values.

  Rejects infinite values (+Inf, -Inf), allows NaN.
  Works correctly with the Nullable marker.

  Example:
    Validated[pd.Series, Finite]           # No Inf, no NaN (NonNaN auto-applied)
    Validated[pd.Series, Finite, Nullable] # No Inf, allows NaN
  """

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, (pd.Series, pd.DataFrame)):
      numeric_data = (
        data.select_dtypes(include=[np.number])
        if isinstance(data, pd.DataFrame)
        else data
      )
      if len(numeric_data) > 0 and np.any(np.isinf(numeric_data.values)):  # type: ignore[arg-type]
        raise ValueError("Data must be finite (contains Inf)")
    return data


class StrictFinite(Validator[pd.Series | pd.DataFrame]):
  """Validator for strictly finite values (no Inf, no NaN).

  Checks for both NaN and infinite values. Always rejects NaN
  regardless of the Nullable marker.

  Use Finite instead if you want to allow NaN with the Nullable marker.
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


# =============================================================================
# Shape Validators
# =============================================================================

# typing.Any is used for 'any dimension' in Shape constraints
# pyright: reportMissingSuperCall=false, reportImplicitOverride=false
# pyright: reportIncompatibleMethodOverride=false


class _ShapeConstraint:
  """Base class for shape dimension constraints."""

  def check(self, value: int) -> bool:
    raise NotImplementedError

  def describe(self) -> str:
    raise NotImplementedError


class _GeDim(_ShapeConstraint):
  def __init__(self, n: int) -> None:
    self.n = n

  def check(self, value: int) -> bool:
    return value >= self.n

  def describe(self) -> str:
    return f">= {self.n}"


class _LeDim(_ShapeConstraint):
  def __init__(self, n: int) -> None:
    self.n = n

  def check(self, value: int) -> bool:
    return value <= self.n

  def describe(self) -> str:
    return f"<= {self.n}"


class _GtDim(_ShapeConstraint):
  def __init__(self, n: int) -> None:
    self.n = n

  def check(self, value: int) -> bool:
    return value > self.n

  def describe(self) -> str:
    return f"> {self.n}"


class _LtDim(_ShapeConstraint):
  def __init__(self, n: int) -> None:
    self.n = n

  def check(self, value: int) -> bool:
    return value < self.n

  def describe(self) -> str:
    return f"< {self.n}"


class _ExactDim(_ShapeConstraint):
  def __init__(self, n: int) -> None:
    self.n = n

  def check(self, value: int) -> bool:
    return value == self.n

  def describe(self) -> str:
    return f"== {self.n}"


class _AnyDimConstraint(_ShapeConstraint):
  def check(self, value: int) -> bool:  # noqa: ARG002
    return True

  def describe(self) -> str:
    return "any"


def _parse_dim_constraint(item: object) -> _ShapeConstraint:  # noqa: PLR0911
  """Parse a dimension constraint from Shape[] arguments.

  Accepts:
  - int: exact dimension match
  - Any (from typing): any dimension allowed
  - Ge[n], Le[n], Gt[n], Lt[n]: comparison constraints
  """
  # Check for typing.Any (use identity check)
  if item is Any:
    return _AnyDimConstraint()
  if isinstance(item, _ShapeConstraint):
    return item
  if isinstance(item, int):
    return _ExactDim(item)
  # Handle comparison validators (Ge[10], Lt[5], etc.)
  if isinstance(item, Ge) and len(item.targets) == 1:
    return _GeDim(int(item.targets[0]))
  if isinstance(item, Le) and len(item.targets) == 1:
    return _LeDim(int(item.targets[0]))
  if isinstance(item, Gt) and len(item.targets) == 1:
    return _GtDim(int(item.targets[0]))
  if isinstance(item, Lt) and len(item.targets) == 1:
    return _LtDim(int(item.targets[0]))
  raise TypeError(f"Invalid shape constraint: {item}")


class Shape(Validator[pd.Series | pd.DataFrame]):
  """Validator for DataFrame/Series dimensions.

  Supports exact values, constraints, or Any for flexible validation:
  - Shape[10, 5] - Exactly 10 rows, 5 columns
  - Shape[Ge[10], Any] - At least 10 rows, any columns
  - Shape[Any, Le[5]] - Any rows, at most 5 columns
  - Shape[Gt[0], Lt[100]] - More than 0 rows, less than 100 columns
  - Shape[100] - For Series: exactly 100 rows

  For Series, only the first dimension (rows) is checked.
  """

  def __init__(
    self,
    rows: _ShapeConstraint,
    cols: _ShapeConstraint | None = None,
  ) -> None:
    super().__init__()
    self.rows = rows
    self.cols = cols

  def __class_getitem__(
    cls,
    items: int
    | _ShapeConstraint
    | object  # typing.Any is checked via identity at runtime
    | tuple[int | _ShapeConstraint | object, ...],
  ) -> Shape:
    # Single item: rows only (for Series)
    if not isinstance(items, tuple):
      return cls(_parse_dim_constraint(items), None)

    if len(items) == 1:
      return cls(_parse_dim_constraint(items[0]), None)
    if len(items) == 2:  # noqa: PLR2004
      return cls(
        _parse_dim_constraint(items[0]),
        _parse_dim_constraint(items[1]),
      )

    raise TypeError("Shape requires 1 or 2 arguments: Shape[rows] or Shape[rows, cols]")

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, pd.Series):
      n_rows = len(data)
      if not self.rows.check(n_rows):
        raise ValueError(f"Series must have {self.rows.describe()} rows, got {n_rows}")
    elif isinstance(data, pd.DataFrame):
      n_rows = len(data)
      n_cols = len(data.columns)

      if not self.rows.check(n_rows):
        raise ValueError(
          f"DataFrame must have {self.rows.describe()} rows, got {n_rows}"
        )
      if self.cols is not None and not self.cols.check(n_cols):
        raise ValueError(
          f"DataFrame must have {self.cols.describe()} columns, got {n_cols}"
        )
    return data
