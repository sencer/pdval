# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportArgumentType=false
import timeit

import numpy as np
import pandas as pd

from pdval import Finite, Validated, validated

# Setup data
data = pd.Series(np.random.randn(1000))


# 1. Raw function
def raw_func(data: pd.Series) -> float:
  return data.sum()


# 2. Decorated function
@validated
def decorated_func(data: Validated[pd.Series, Finite]) -> float:
  return data.sum()


# 3. Decorated function with skip_validation=True
# (This is already handled by the decorator logic, we just call it with the arg)


def run_benchmarks() -> None:
  n = 10000

  t_raw = timeit.timeit(lambda: raw_func(data), number=n)
  t_dec = timeit.timeit(lambda: decorated_func(data), number=n)
  t_skip = timeit.timeit(lambda: decorated_func(data, skip_validation=True), number=n)

  print(f"Raw function: {t_raw:.4f}s")
  print(f"Decorated (validate=True): {t_dec:.4f}s")
  print(f"Decorated (skip_validation=True): {t_skip:.4f}s")

  overhead = (t_dec - t_raw) / t_raw * 100
  print(f"Overhead (validate=True): {overhead:.1f}%")

  overhead_skip = (t_skip - t_raw) / t_raw * 100
  print(f"Overhead (skip_validation=True): {overhead_skip:.1f}%")


if __name__ == "__main__":
  run_benchmarks()
