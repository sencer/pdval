"""Comparison validators for Series and DataFrame data."""

from __future__ import annotations

from typing import Literal, get_args, get_origin, override

import numpy as np
import pandas as pd

from pdval.base import Validator


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
