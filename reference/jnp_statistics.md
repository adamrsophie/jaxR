# Statistical reductions

Ports of `jnp.minmax` (minimum and maximum in one call), `jnp.ptp`
(range), `jnp.count_nonzero` and `jnp.average` (weighted mean).

## Usage

``` r
jnp_minmax(
  a,
  axis = NULL,
  out = NULL,
  keepdims = FALSE,
  initial = NULL,
  where = NULL
)

jnp_ptp(a, axis = NULL, out = NULL, keepdims = FALSE)

jnp_count_nonzero(a, axis = NULL, keepdims = FALSE)

jnp_average(a, axis = NULL, weights = NULL, returned = FALSE, keepdims = FALSE)
```

## Arguments

- a:

  Array.

- axis:

  0-based axis or axes to reduce; `NULL` reduces all.

- out:

  Not supported (must be `NULL`).

- keepdims:

  Keep reduced dimensions as size one.

- initial:

  `minmax`: a scalar starting value for both reductions, or a list of
  two (for the minimum and the maximum).

- where:

  `minmax`: optional boolean mask (needs `initial`).

- weights:

  `average`: optional weights, of the shape of `a` or of the reduced
  axes.

- returned:

  `average`: also return the sum of the weights.

## Value

An array; `minmax` returns `list(min, max)` and `average` with
`returned = TRUE` returns `list(average, sum_of_weights)`.
