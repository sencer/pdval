# pyright: reportCallIssue=false
import numpy as np
import pandas as pd

from pdval import Finite, Validated, validated


def test_repro_validate_param() -> None:
  @validated
  def calculate_returns(
    prices: Validated[pd.Series, Finite], validate: bool = True
  ) -> pd.Series:
    return prices

  bad_prices = pd.Series([100.0, np.inf, 101.0])

  # Case 1: User passes validate=False, expecting validation to be skipped.
  # But decorator only looks for skip_validation.
  # Expected behavior (from user perspective): No error.
  # Actual behavior: Error (Finite validator raises).
  try:
    calculate_returns(bad_prices, validate=False)
    print("Case 1: Validation skipped (Unexpected if bug exists)")
  except ValueError as e:
    print(f"Case 1: Validation ran and raised error: {e}")

  # Case 2: User uses skip_validation=True (as per decorator implementation)
  try:
    calculate_returns(bad_prices, skip_validation=True)
    print("Case 2: Validation skipped (Expected)")
  except ValueError as e:
    print(f"Case 2: Validation ran and raised error: {e}")


if __name__ == "__main__":
  test_repro_validate_param()
