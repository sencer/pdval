# Benchmark Results

## Decorator Performance

Benchmark measuring the overhead of the `@validated` decorator with 10,000 iterations:

### Results Summary

| Scenario | validate=False Overhead | validate=True Overhead |
|----------|------------------------|----------------------|
| Small Series (100 elements) | **25.49µs/call** | 212.45µs/call |
| Large Series (10,000 elements) | **25.92µs/call** | 229.53µs/call |
| Multiple Validators (2) | **26.48µs/call** | 290.81µs/call |
| Index Validators (3) | **26.69µs/call** | 237.02µs/call |

### Key Findings

1. **`validate=False` overhead is consistent at ~25-27µs per call**, regardless of:
   - Data size
   - Number of validators
   - Validator complexity

2. **`validate=True` overhead scales with**:
   - Number of validators (more validators = more time)
   - Validator complexity (Index validators do more work)
   - Data size (slightly, since validation needs to check all elements)

3. **Overhead breakdown**:
   - The ~25µs with `validate=False` comes from Python's `inspect.signature().bind()` machinery
   - The validation itself adds ~200-300µs depending on complexity

### Recommendations

- **Development/Testing**: Use `validate=True` (default) to catch data issues early
- **Production hot paths**: Use `validate=False` after initial validation to minimize overhead
- **One-time operations**: The overhead is negligible - validation is worth it
- **Tight loops**: Consider validating once before the loop, then use `validate=False` inside

### Running the Benchmark

```bash
uv run python benchmarks/benchmark_decorator.py
```
