"""Marker validators that opt-out of default validation behaviors."""

from __future__ import annotations

from pdval.base import ValidatorMarker


class Nullable(ValidatorMarker):
  """Marker to allow NaN values (opt-out of default NonNaN)."""


class MaybeEmpty(ValidatorMarker):
  """Marker to allow empty data (opt-out of default NonEmpty)."""
