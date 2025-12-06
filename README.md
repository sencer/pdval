# pdval

![CI](https://github.com/sencer/pdval/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/sencer/pdval/branch/master/graph/badge.svg)](https://app.codecov.io/github/sencer/pdval)

**Pandas validation using Annotated types and decorators**

`pdval` is a lightweight Python library for validating pandas DataFrames and Series using Python's `Annotated` types and decorators. It provides a clean, type-safe way to express data validation constraints directly in function signatures.

## Features

- 🎯 **Type-safe validation** - Uses Python's `Annotated` types for inline constraints
- 🐼 **Pandas-focused** - Built specifically for pandas DataFrames and Series
- ⚡ **Decorator-based** - Simple `@validated` decorator for automatic validation
- 🔧 **Composable validators** - Chain multiple validators together
- 🎨 **Clean syntax** - Validation rules live in your type annotations
- 🚀 **Zero runtime overhead** - Optional validation can be disabled

## Installation

```bash
pip install pdval
```

Or with uv:
```bash
uv add pdval
```

> **Note**: If you prefer using [Pandera](https://pandera.readthedocs.io/) as the underlying validation engine (for more detailed error reporting and robustness), install the separate package:
> ```bash
> pip install pdval-pandera
> ```

## Quick Start

```python
import pandas as pd
from pdval import validated, Validated, Finite, NonNaN

@validated
def calculate_returns(
    prices: Validated[pd.Series, Finite, NonNaN],
) -> pd.Series:
    """Calculate percentage returns from prices."""
    return prices.pct_change()

# Valid data passes through
prices = pd.Series([100.0, 102.0, 101.0, 103.0])
returns = calculate_returns(prices)

# Invalid data raises ValueError
import numpy as np
bad_prices = pd.Series([100.0, np.inf, 101.0])
# Raises: ValueError: Data must be finite (contains Inf)
calculate_returns(bad_prices)
```

## Available Validators

### Value Validators (Series/Index)

- **`Finite`** - Ensures no Inf or NaN values
- **`NonNaN`** - Ensures no NaN values (allows Inf)
- **`NonNegative`** - Ensures all values >= 0
- **`Positive`** - Ensures all values > 0
- **`NonEmpty`** - Ensures data is not empty
- **`Unique`** - Ensures all values are unique
- **`MonoUp`** - Ensures values are monotonically increasing
- **`MonoDown`** - Ensures values are monotonically decreasing
- **`Datetime`** - Ensures data is a DatetimeIndex
- **`OneOf["a", "b", "c"]`** - Ensures values are in allowed set (categorical)


### Index Wrapper

The `Index[]` wrapper allows you to apply any Series/Index validator to the index of a Series or DataFrame:

- **`Index[Datetime]`** - Ensures index is a DatetimeIndex
- **`Index[MonoUp]`** - Ensures index is monotonically increasing
- **`Index[Unique]`** - Ensures index values are unique
- **`Index[Datetime, MonoUp, Unique]`** - Combine multiple validators

### DataFrame Column Validators

- **`HasColumns["col1", "col2"]`** - Ensures specified columns exist
- **`Ge["high", "low"]`** - Ensures one column >= another column
- **`Le["low", "high"]`** - Ensures one column <= another column
- **`Gt["high", "low"]`** - Ensures one column > another column
- **`Lt["low", "high"]`** - Ensures one column < another column

### Column-Specific Validators

- **`HasColumn["col"]`** - Check that DataFrame has column (with default NonNaN, NonEmpty)
- **`HasColumn["col", Validator, ...]`** - Check column exists and apply Series validators

### Gap Validators (Time Series)

- **`NoTimeGaps`** - Ensures no gaps in datetime values/index
- **`MaxGap[timedelta]`** - Ensures maximum gap between datetime values
- **`MaxDiff[value]`** - Ensures maximum difference between consecutive values

### Markers (Opt-out)

- **`Nullable`** - Opt out of default NonNaN check
- **`MaybeEmpty`** - Opt out of default NonEmpty check

## Examples

### Basic Series Validation

```python
from pdval import validated, Validated, Positive
import numpy as np
import pandas as pd

@validated
def calculate_log_returns(
    prices: Validated[pd.Series, Positive],
) -> pd.Series:
    """Calculate log returns - prices must be positive."""
    return np.log(prices / prices.shift(1))

prices = pd.Series([100.0, 102.0, 101.0, 103.0])
log_returns = calculate_log_returns(prices)
```

### DataFrame Column Validation

```python
from pdval import validated, Validated, HasColumns, Ge, NonNaN
import pandas as pd

@validated
def calculate_true_range(
    data: Validated[pd.DataFrame, HasColumns["high", "low", "close"], Ge["high", "low"], NonNaN],
) -> pd.Series:
    """Calculate True Range - requires OHLC data."""
    hl = data["high"] - data["low"]
    hc = abs(data["high"] - data["close"].shift(1))
    lc = abs(data["low"] - data["close"].shift(1))
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

# Valid OHLC data
ohlc = pd.DataFrame({
    "high": [102, 105, 104],
    "low": [100, 103, 101],
    "close": [101, 104, 102]
})
tr = calculate_true_range(ohlc)

# Missing column raises error
bad_data = pd.DataFrame({"high": [102], "close": [101]})
# Raises: ValueError: Missing columns: ['low']
calculate_true_range(bad_data)
```

### Time Series Validation with Index

```python
from pdval import validated, Validated, Index, Datetime, MonoUp, Finite
import pandas as pd

@validated
def resample_ohlc(
    data: Validated[pd.DataFrame, Index[Datetime, MonoUp], Finite],
    freq: str = "1D",
) -> pd.DataFrame:
    """Resample OHLC data to different frequency."""
    return data.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

# Valid time series
dates = pd.date_range("2024-01-01", periods=10, freq="1h")
data = pd.DataFrame({
    "open": range(100, 110),
    "high": range(101, 111),
    "low": range(99, 109),
    "close": range(100, 110)
}, index=dates)
daily = resample_ohlc(data)

# Non-datetime index raises error
bad_data = data.copy()
bad_data.index = range(len(bad_data))
# Raises: ValueError: Index must be DatetimeIndex
resample_ohlc(bad_data)
```

### Unique Values Validation

```python
from pdval import validated, Validated, Index, Unique
import pandas as pd

@validated
def process_unique_ids(
    data: Validated[pd.DataFrame, Index[Unique]],
) -> pd.DataFrame:
    """Process data with unique index values."""
    return data.sort_index()

# Valid unique index
df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
result = process_unique_ids(df)

# Duplicate index values raise error
bad_df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "x", "z"])
# Raises: ValueError: Values must be unique
process_unique_ids(bad_df)
```

### Categorical Values Validation

```python
from typing import Literal
from pdval import validated, Validated, OneOf, HasColumn
import pandas as pd

@validated
def process_orders(
    data: Validated[pd.DataFrame, HasColumn["status", OneOf["pending", "shipped", "delivered"]]],
) -> pd.DataFrame:
    """Process orders with validated status column."""
    return data[data["status"] != "pending"]

# Valid data
orders = pd.DataFrame({
    "order_id": [1, 2, 3],
    "status": ["pending", "shipped", "delivered"]
})
result = process_orders(orders)

# Invalid status raises error
bad_orders = pd.DataFrame({
    "order_id": [1, 2],
    "status": ["pending", "cancelled"]  # "cancelled" not in allowed values
})
# Raises: ValueError: Values must be one of {'pending', 'shipped', 'delivered'}, got invalid: {'cancelled'}
process_orders(bad_orders)

# Also works with Literal type syntax
@validated
def process_with_literal(
    data: Validated[pd.Series, OneOf[Literal["a", "b", "c"]]],
) -> pd.Series:
    return data
```

### Monotonic Value Validation

```python
from pdval import validated, Validated, MonoUp, MonoDown
import pandas as pd

@validated
def calculate_cumulative_returns(
    prices: Validated[pd.Series, MonoUp],
) -> pd.Series:
    """Calculate cumulative returns - prices must be monotonically increasing."""
    return (prices / prices.iloc[0]) - 1

@validated
def track_drawdown(
    equity: Validated[pd.Series, MonoDown],
) -> pd.Series:
    """Track drawdown - equity must be monotonically decreasing."""
    return (equity / equity.iloc[0]) - 1
```

### Column-Specific Validation with HasColumn

```python
from pdval import validated, Validated, HasColumn, Finite, Positive, MonoUp
import pandas as pd

@validated
def process_trading_data(
    data: Validated[
        pd.DataFrame,
        HasColumn["price", Finite, Positive],
        HasColumn["volume", Finite, Positive],
        HasColumn["timestamp", MonoUp],
    ],
) -> pd.DataFrame:
    """Process trading data with column-specific validation.

    - price: must exist, be finite and positive
    - volume: must exist, be finite and positive
    - timestamp: must exist and be monotonically increasing
    """
    return data.assign(
        notional=data["price"] * data["volume"]
    )

# Or just check column presence (with default NonNaN, NonEmpty):
@validated
def simple_check(
    data: Validated[pd.DataFrame, HasColumn["price"], HasColumn["volume"]],
) -> float:
    """Just check columns exist and have valid values."""
    return (data["price"] * data["volume"]).sum()
```

### Chaining Multiple Index Validators

```python
from pdval import validated, Validated, Index, Datetime, MonoUp, Unique, Finite, Positive
import pandas as pd

@validated
def calculate_volume_profile(
    volume: Validated[pd.Series, Index[Datetime, MonoUp, Unique], Finite, Positive],
) -> pd.Series:
    """Calculate volume profile - must be datetime-indexed, monotonic, unique, finite, positive."""
    return volume.groupby(volume.index.hour).sum()
```

### Optional Validation

Use `skip_validation` to disable validation for performance:

```python
# Validation enabled (default)
result = calculate_returns(prices)

# Validation disabled for performance
result = calculate_returns(prices, skip_validation=True)
```

### Custom Validators

Create your own validators by subclassing `Validator`:

```python
from pdval import Validator, validated, Validated
import pandas as pd

class InRange(Validator):
    """Validator for values within a specific range."""

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, data):
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if (data < self.min_val).any() or (data > self.max_val).any():
                raise ValueError(f"Data must be in range [{self.min_val}, {self.max_val}]")
        return data

@validated
def normalize_percentage(
    data: Validated[pd.Series, InRange(0, 100)],
) -> pd.Series:
    """Normalize percentage data to [0, 1] range."""
    return data / 100
```

## Type Checking

`pdval` includes a `py.typed` marker for full type checker support. Your IDE and type checkers (mypy, pyright, basedpyright) will understand the validation annotations.

### How Type Checkers Handle `Validated`

According to PEP 593, `Annotated[T, metadata]` (which `Validated` is an alias for) is treated as **equivalent to `T`** for type checking purposes. This means:

```python
@validated
def process(data: Validated[pd.Series, Finite]) -> float:
    return data.sum()

# Type checkers understand that pd.Series is compatible with Validated[pd.Series, ...]
series = pd.Series([1, 2, 3])
result = process(series)  # ✓ Type checker is happy!
```

The validation metadata is:
- **Preserved at runtime** - Used by the `@validated` decorator for validation
- **Ignored by type checkers** - `Validated[pd.Series, Finite]` is treated as `pd.Series`

This gives you the best of both worlds: clean type checking and runtime validation.

## Default Strictness

By default, `pdval` applies `NonNaN` and `NonEmpty` checks to all validated arguments. This can be opted out with markers:

```python
from pdval import validated, Validated, Nullable, MaybeEmpty
import pandas as pd

@validated
def process_with_defaults(data: Validated[pd.Series, None]) -> float:
    """NaN and empty data will raise ValueError by default."""
    return data.sum()

@validated
def process_nullable(data: Validated[pd.Series, Nullable]) -> float:
    """NaN values are allowed."""
    return data.sum()

@validated
def process_maybe_empty(data: Validated[pd.Series, MaybeEmpty]) -> int:
    """Empty data is allowed."""
    return len(data)
```

## Comparison with Pandera

While [Pandera](https://pandera.readthedocs.io/) is excellent for comprehensive schema validation, `pdval` offers a lighter-weight alternative focused on:

- **Inline validation** - Constraints live in function signatures
- **Decorator simplicity** - Single `@validated` decorator
- **Type annotation syntax** - Uses Python's native `Annotated` types
- **Minimal overhead** - Lightweight with no heavy dependencies

Use `pdval` when you want simple, inline validation. Use Pandera when you need comprehensive schema management, complex validation logic, or data contracts.

## Performance

`pdval` is designed to be lightweight with minimal overhead:

- Validation checks are only performed when `skip_validation=False` (default)
- No schema compilation or complex preprocessing
- Direct numpy/pandas operations for validation
- Optional validation can be disabled for production performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Why pdval?

**Problem:** When building data analysis pipelines with pandas, you often need to validate:
- Data has no NaN or Inf values
- DataFrames have required columns
- Values are in expected ranges
- Indices are properly formatted

**Traditional approach:** Add manual validation checks at the start of each function.

**With pdval:** Express validation constraints directly in type annotations using `Validated[Type, Validator, ...]` and get automatic validation with the `@validated` decorator.
