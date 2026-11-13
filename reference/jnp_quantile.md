# Quantiles, percentiles and medians

Ports of `jnp.quantile`, `jnp.nanquantile`, `jnp.percentile`,
`jnp.nanpercentile`, `jnp.median` and `jnp.nanmedian`. The `nan`
variants ignore NaN entries; the others return NaN for slices that
contain one.

## Usage

``` r
jnp_quantile(
  a,
  q,
  axis = NULL,
  out = NULL,
  overwrite_input = FALSE,
  method = "linear",
  keepdims = FALSE,
  weights = NULL
)

jnp_nanquantile(
  a,
  q,
  axis = NULL,
  out = NULL,
  overwrite_input = FALSE,
  method = "linear",
  keepdims = FALSE,
  weights = NULL
)

jnp_percentile(
  a,
  q,
  axis = NULL,
  out = NULL,
  overwrite_input = FALSE,
  method = "linear",
  keepdims = FALSE,
  weights = NULL
)

jnp_nanpercentile(
  a,
  q,
  axis = NULL,
  out = NULL,
  overwrite_input = FALSE,
  method = "linear",
  keepdims = FALSE,
  weights = NULL
)

jnp_median(
  a,
  axis = NULL,
  out = NULL,
  overwrite_input = FALSE,
  keepdims = FALSE
)

jnp_nanmedian(
  a,
  axis = NULL,
  out = NULL,
  overwrite_input = FALSE,
  keepdims = FALSE
)
```

## Arguments

- a:

  Array.

- q:

  Quantile (in \[0, 1\]) or percentile (in \[0, 100\]); a scalar or 1-D
  array. A 1-D `q` adds a leading dimension to the result.

- axis:

  0-based axis or axes to reduce; `NULL` reduces all.

- out:

  Not supported (must be `NULL`).

- overwrite_input:

  Not supported (must be `FALSE`).

- method:

  One of `"linear"`, `"lower"`, `"higher"`, `"midpoint"`, `"nearest"`,
  `"inverted_cdf"`.

- keepdims:

  Keep reduced dimensions as size one.

- weights:

  Optional weights (only with `method = "inverted_cdf"`).

## Value

An array.
