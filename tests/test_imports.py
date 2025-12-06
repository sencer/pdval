"""Tests for package imports and public API."""

import pdval


class TestImports:
  """Test that all public APIs are importable."""

  def test_validators_importable(self):
    """Test all validators can be imported."""
    from pdval import (
      DateTimeIndexed,
      Finite,
      Ge,
      HasColumns,
      MonotonicIndex,
      NonNaN,
      NonNegative,
      Positive,
      Validator,
    )

    assert Finite is not None
    assert NonNaN is not None
    assert NonNegative is not None
    assert Positive is not None
    assert DateTimeIndexed is not None
    assert MonotonicIndex is not None
    assert HasColumns is not None
    assert Ge is not None
    assert Validator is not None

  def test_decorator_importable(self):
    """Test decorator can be imported."""
    from pdval import validated

    assert validated is not None

  def test_validated_type_importable(self):
    """Test Validated type alias can be imported."""
    from pdval import Validated

    assert Validated is not None

  def test_version_importable(self):
    """Test version can be imported."""
    from pdval import __version__

    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"

  def test_all_exports(self):
    """Test __all__ contains expected exports."""
    expected = {
      "DateTimeIndexed",
      "Finite",
      "Ge",
      "Gt",
      "HasColumns",
      "Le",
      "Lt",
      "MonoDown",
      "MonotonicIndex",
      "MonoUp",
      "NonNaN",
      "NonNegative",
      "HasColumn",
      "Positive",
      "Validated",
      "Validator",
      "__version__",
      "validated",
    }
    assert set(pdval.__all__) == expected

  def test_package_has_all_attrs(self):
    """Test package has all attributes listed in __all__."""
    for name in pdval.__all__:
      assert hasattr(pdval, name), f"Missing attribute: {name}"
