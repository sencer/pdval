"""Legacy module - re-exports from new modular structure for backwards compatibility.

This module is deprecated. Import directly from pdval instead:
    from pdval import Validated, validated, Finite, ...
"""

# Re-export everything for backwards compatibility
from pdval.base import Validated, Validator, ValidatorMarker
from pdval.decorator import validated
from pdval.validators import (
  Datetime,
  Finite,
  Ge,
  Gt,
  HasColumn,
  HasColumns,
  Index,
  IsDtype,
  Le,
  Lt,
  MaxDiff,
  MaxGap,
  MaybeEmpty,
  MonoDown,
  MonoUp,
  NonEmpty,
  NonNaN,
  NonNegative,
  NoTimeGaps,
  Nullable,
  Positive,
  UniqueIndex,
)

__all__ = [
  "Datetime",
  "Finite",
  "Ge",
  "Gt",
  "HasColumn",
  "HasColumns",
  "Index",
  "IsDtype",
  "Le",
  "Lt",
  "MaxDiff",
  "MaxGap",
  "MaybeEmpty",
  "MonoDown",
  "MonoUp",
  "NoTimeGaps",
  "NonEmpty",
  "NonNaN",
  "NonNegative",
  "Nullable",
  "Positive",
  "UniqueIndex",
  "Validated",
  "Validator",
  "ValidatorMarker",
  "validated",
]
