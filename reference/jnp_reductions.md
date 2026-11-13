# Reductions

Ports of `jnp.sum`, `jnp.prod`, `jnp.max`, `jnp.min`, `jnp.mean`,
`jnp.var`, `jnp.std`, `jnp.all`, `jnp.any`, `jnp.argmax` and
`jnp.argmin`. Axes are 0-based; `NULL` reduces over all of them.

## Usage

``` r
jnp_sum(
  a,
  axis = NULL,
  dtype = NULL,
  out = NULL,
  keepdims = FALSE,
  initial = NULL,
  where = NULL,
  promote_integers = TRUE
)

jnp_prod(
  a,
  axis = NULL,
  dtype = NULL,
  out = NULL,
  keepdims = FALSE,
  initial = NULL,
  where = NULL,
  promote_integers = TRUE
)

jnp_max(
  a,
  axis = NULL,
  out = NULL,
  keepdims = FALSE,
  initial = NULL,
  where = NULL
)

jnp_min(
  a,
  axis = NULL,
  out = NULL,
  keepdims = FALSE,
  initial = NULL,
  where = NULL
)

jnp_amax(
  a,
  axis = NULL,
  out = NULL,
  keepdims = FALSE,
  initial = NULL,
  where = NULL
)

jnp_amin(
  a,
  axis = NULL,
  out = NULL,
  keepdims = FALSE,
  initial = NULL,
  where = NULL
)

jnp_all(a, axis = NULL, out = NULL, keepdims = FALSE, where = NULL)

jnp_any(a, axis = NULL, out = NULL, keepdims = FALSE, where = NULL)

jnp_mean(
  a,
  axis = NULL,
  dtype = NULL,
  out = NULL,
  keepdims = FALSE,
  where = NULL
)

jnp_var(
  a,
  axis = NULL,
  dtype = NULL,
  out = NULL,
  ddof = 0L,
  keepdims = FALSE,
  where = NULL,
  mean = NULL,
  correction = NULL
)

jnp_std(
  a,
  axis = NULL,
  dtype = NULL,
  out = NULL,
  ddof = 0L,
  keepdims = FALSE,
  where = NULL,
  mean = NULL,
  correction = NULL
)

jnp_argmax(a, axis = NULL, out = NULL, keepdims = FALSE)

jnp_argmin(a, axis = NULL, out = NULL, keepdims = FALSE)
```

## Arguments

- a:

  Array.

- axis:

  Axis or axes.

- dtype:

  Optional result dtype (the accumulation dtype of `sum` and `prod`).

- out:

  Not supported (must be `NULL`).

- keepdims:

  Keep reduced dimensions as size one.

- initial:

  Optional scalar starting value of the reduction. `max` and `min` need
  it to reduce an empty array or to use `where`.

- where:

  Optional boolean mask of the elements to include.

- promote_integers:

  Whether integers narrower than the default integer are promoted to it
  (the NumPy rule).

- ddof:

  Delta degrees of freedom: the divisor is `N - ddof`.

- mean:

  Optional precomputed mean (with the reduced dimensions kept).

- correction:

  Array API name of `ddof` (give one of the two).

## Value

An array.
