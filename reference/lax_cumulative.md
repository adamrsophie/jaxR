# Cumulative reductions

Ports of `jax.lax.cumsum`, `cumprod`, `cummax`, `cummin` and
`cumlogsumexp`. As in JAX the axis must be nonnegative (0-based).

## Usage

``` r
lax_cumsum(operand, axis = 0L, reverse = FALSE)

lax_cumprod(operand, axis = 0L, reverse = FALSE)

lax_cummax(operand, axis = 0L, reverse = FALSE)

lax_cummin(operand, axis = 0L, reverse = FALSE)

lax_cumlogsumexp(operand, axis = 0L, reverse = FALSE)
```

## Arguments

- operand:

  Array of a numeric dtype.

- axis:

  0-based axis.

- reverse:

  Accumulate from the end.

## Value

An array of the shape and dtype of `operand`.
