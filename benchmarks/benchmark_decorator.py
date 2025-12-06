"""Detailed benchmark of @validated decorator overhead."""

import timeit
import pandas as pd
import numpy as np
from typing import Any

from pdval import Finite, Index, Datetime, MonoUp, Positive, Validated, validated


# Setup test data
small_data = pd.Series(np.random.rand(100))
large_data = pd.Series(np.random.rand(10000))
datetime_data = pd.Series(
  np.random.rand(1000), index=pd.date_range("2024-01-01", periods=1000)
)


# Test functions
def plain(data: pd.Series) -> float:
  return data.sum()


@validated
def decorated_simple(data: Validated[pd.Series, Finite], validate: bool = True) -> float:
  return data.sum()


@validated
def decorated_multiple(
  data: Validated[pd.Series, Finite, Positive], validate: bool = True
) -> float:
  return data.sum()


@validated
def decorated_index(
  data: Validated[pd.Series, Index[Datetime, MonoUp], Finite],
  validate: bool = True,
) -> float:
  return data.sum()


def run_benchmark(func: Any, data: pd.Series, validate: bool | None, iterations: int):
  if validate is None:
    return timeit.timeit(lambda: func(data), number=iterations)
  else:
    return timeit.timeit(lambda: func(data, validate=validate), number=iterations)


def main():
  iterations = 10000
  print("=" * 70)
  print("@validated Decorator Performance Benchmark")
  print("=" * 70)
  print(f"Iterations per test: {iterations:,}\n")

  # Test 1: Small data (100 elements)
  print("Test 1: Small Series (100 elements)")
  print("-" * 70)

  t_plain = run_benchmark(plain, small_data, None, iterations)
  t_false = run_benchmark(decorated_simple, small_data, False, iterations)
  t_true = run_benchmark(decorated_simple, small_data, True, iterations)

  overhead_false = ((t_false - t_plain) / iterations) * 1_000_000
  overhead_true = ((t_true - t_plain) / iterations) * 1_000_000

  print(f"Plain function:              {t_plain:8.4f}s  (baseline)")
  print(f"Decorated (validate=False):  {t_false:8.4f}s  (+{overhead_false:6.2f}µs/call)")
  print(f"Decorated (validate=True):   {t_true:8.4f}s  (+{overhead_true:6.2f}µs/call)")
  print()

  # Test 2: Large data (10,000 elements)
  print("Test 2: Large Series (10,000 elements)")
  print("-" * 70)

  t_plain = run_benchmark(plain, large_data, None, iterations)
  t_false = run_benchmark(decorated_simple, large_data, False, iterations)
  t_true = run_benchmark(decorated_simple, large_data, True, iterations)

  overhead_false = ((t_false - t_plain) / iterations) * 1_000_000
  overhead_true = ((t_true - t_plain) / iterations) * 1_000_000

  print(f"Plain function:              {t_plain:8.4f}s  (baseline)")
  print(f"Decorated (validate=False):  {t_false:8.4f}s  (+{overhead_false:6.2f}µs/call)")
  print(f"Decorated (validate=True):   {t_true:8.4f}s  (+{overhead_true:6.2f}µs/call)")
  print()

  # Test 3: Multiple validators
  print("Test 3: Multiple Validators (Finite + Positive)")
  print("-" * 70)

  t_plain = run_benchmark(plain, small_data, None, iterations)
  t_false = run_benchmark(decorated_multiple, small_data, False, iterations)
  t_true = run_benchmark(decorated_multiple, small_data, True, iterations)

  overhead_false = ((t_false - t_plain) / iterations) * 1_000_000
  overhead_true = ((t_true - t_plain) / iterations) * 1_000_000

  print(f"Plain function:              {t_plain:8.4f}s  (baseline)")
  print(f"Decorated (validate=False):  {t_false:8.4f}s  (+{overhead_false:6.2f}µs/call)")
  print(f"Decorated (validate=True):   {t_true:8.4f}s  (+{overhead_true:6.2f}µs/call)")
  print()

  # Test 4: Index validators
  print("Test 4: Index Validators (Index[Datetime, MonoUp] + Finite)")
  print("-" * 70)

  t_plain = run_benchmark(plain, datetime_data, None, iterations)
  t_false = run_benchmark(decorated_index, datetime_data, False, iterations)
  t_true = run_benchmark(decorated_index, datetime_data, True, iterations)

  overhead_false = ((t_false - t_plain) / iterations) * 1_000_000
  overhead_true = ((t_true - t_plain) / iterations) * 1_000_000

  print(f"Plain function:              {t_plain:8.4f}s  (baseline)")
  print(f"Decorated (validate=False):  {t_false:8.4f}s  (+{overhead_false:6.2f}µs/call)")
  print(f"Decorated (validate=True):   {t_true:8.4f}s  (+{overhead_true:6.2f}µs/call)")
  print()

  # Summary
  print("=" * 70)
  print("Summary")
  print("=" * 70)
  print("• validate=False adds ~20-30µs overhead per call (argument binding)")
  print("• validate=True overhead scales with:")
  print("  - Number of validators")
  print("  - Data size")
  print("  - Validator complexity")
  print()
  print("Recommendation: Use validate=False in production hot paths after")
  print("                initial data validation during development/testing.")


if __name__ == "__main__":
  main()
