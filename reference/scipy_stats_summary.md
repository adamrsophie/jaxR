# Summary statistics

Ports of `jax.scipy.stats.mode`, `jax.scipy.stats.rankdata` and
`jax.scipy.stats.sem`. Axes are 0-based.

## Usage

``` r
scipy_stats_mode(a, axis = 0L, nan_policy = "propagate", keepdims = FALSE)

scipy_stats_rankdata(
  a,
  method = "average",
  axis = NULL,
  nan_policy = "propagate"
)

scipy_stats_sem(
  a,
  axis = 0L,
  ddof = 1L,
  nan_policy = "propagate",
  keepdims = FALSE
)
```

## Arguments

- a:

  Array.

- axis:

  Axis (0-based) or `NULL` for the flattened array.

- nan_policy:

  Only `"propagate"` (plus `"omit"` for `sem`), as in JAX.

- keepdims:

  Keep the reduced axis with size one.

- method:

  Ranking method: `"average"`, `"min"`, `"max"`, `"dense"` or
  `"ordinal"`.

- ddof:

  Delta degrees of freedom.

## Value

`scipy_stats_mode()` returns a list with `mode` and `count`;
`scipy_stats_rankdata()` and `scipy_stats_sem()` return arrays.
