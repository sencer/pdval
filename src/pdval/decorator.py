"""The @validated decorator for automatic argument validation."""

from __future__ import annotations

import functools
import inspect
import typing
from typing import (
  TYPE_CHECKING,
  Annotated,
  Any,
  ParamSpec,
  get_args,
  get_origin,
  overload,
)

from loguru import logger
import pandas as pd

from pdval.base import Validator, ValidatorMarker
from pdval.validators.columns import HasColumn, HasColumns
from pdval.validators.markers import MaybeEmpty, Nullable
from pdval.validators.value import NonEmpty, NonNaN

if TYPE_CHECKING:
  from collections.abc import Callable

P = ParamSpec("P")
R = typing.TypeVar("R")


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
        for arg_name, value in bound_args.arguments.items():
          if arg_name in arg_validators:
            for v in arg_validators[arg_name]:
              v.validate(value)
      except Exception as e:
        if warn_only:
          logger.error(f"Validation failed for {func.__name__}: {e}")
          return None
        raise

      return func(*args, **kwargs)

    return wrapper

  if func is None:
    return decorator

  return decorator(func)
