# Cumulative sums and products

Ports of `jnp.cumsum`, `jnp.cumprod`, `jnp.nancumsum`, `jnp.nancumprod`,
`jnp.cumulative_sum` and `jnp.cumulative_prod`.

## Usage

``` r
jnp_cumsum(a, axis = NULL, dtype = NULL, out = NULL)

jnp_cumprod(a, axis = NULL, dtype = NULL, out = NULL)

jnp_nancumsum(a, axis = NULL, dtype = NULL, out = NULL)

jnp_nancumprod(a, axis = NULL, dtype = NULL, out = NULL)

jnp_cumulative_sum(x, axis = NULL, dtype = NULL, include_initial = FALSE)

jnp_cumulative_prod(x, axis = NULL, dtype = NULL, include_initial = FALSE)
```

## Arguments

- a, x:

  Array.

- axis:

  0-based axis; `NULL` flattens `a` (for `cumulative_*` it may be
  omitted only for one dimensional input).

- dtype:

  Optional dtype of the result.

- out:

  Unsupported (must be `NULL`).

- include_initial:

  Include the initial value (0 or 1) as first element.

## Value

An array.
