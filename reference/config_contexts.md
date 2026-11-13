# Debugging and safety configuration contexts

Ports of the configuration context managers of `jax`: each evaluates
`code` with the option set and restores the previous value.

## Usage

``` r
debug_nans(new_val = TRUE, code)

debug_infs(new_val = TRUE, code)

numpy_dtype_promotion(new_val = TRUE, code)

log_compiles(new_val = TRUE, code)

explain_cache_misses(new_val = TRUE, code)

check_tracer_leaks(new_val = TRUE, code)

checking_leaks(code)

no_tracing(code)

no_execution(code)

transfer_guard_host_to_device(new_val = TRUE, code)

transfer_guard_device_to_host(new_val = TRUE, code)

transfer_guard_device_to_device(new_val = TRUE, code)

transfer_guard(new_val, code)
```

## Arguments

- new_val:

  New value of the option.

- code:

  Expression evaluated with the option set.

## Details

- `debug_nans()` / `debug_infs()`: raise an error (class
  `jax_floating_point_error`) when a primitive or a jitted computation
  produces NaN / infinite values.

- `numpy_dtype_promotion("strict")`: forbid implicit promotion between
  different strongly typed dtypes.

- `log_compiles()`: report every XLA compilation.

- `explain_cache_misses()`: report why `jit` traces again.

- `check_tracer_leaks()`: leaked tracers raise errors (jaxR always
  detects use of a dead tracer).

- `no_tracing()` / `no_execution()`: raise an error on any jit tracing /
  execution.

- `transfer_guard()` and its directional variants: `"allow"`, `"log"`,
  `"disallow"`, `"log_explicit"` or `"disallow_explicit"` for implicit
  (conversions of R data that is not a scalar,
  [`as.array()`](https://rdrr.io/r/base/array.html), printing) and
  explicit
  ([`device_put()`](https://adamrsophie.github.io/jaxR/reference/device_put.md),
  [`device_get()`](https://adamrsophie.github.io/jaxR/reference/device_get.md))
  transfers.
