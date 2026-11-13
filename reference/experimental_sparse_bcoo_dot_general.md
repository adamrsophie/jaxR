# General dot product with BCOO operands

Port of `jax.experimental.sparse.bcoo_dot_general`: either operand may
be a BCOO array (both sparse gives a BCOO result, otherwise a dense
array).

## Usage

``` r
experimental_sparse_bcoo_dot_general(
  lhs,
  rhs,
  dimension_numbers,
  precision = NULL,
  preferred_element_type = NULL
)
```

## Arguments

- lhs, rhs:

  BCOO or dense arrays.

- dimension_numbers:

  As for
  [`lax_dot_general()`](https://adamrsophie.github.io/jaxR/reference/lax_dot_general.md)
  (0-based).

- precision:

  Ignored, as in JAX.

- preferred_element_type:

  Optional result dtype.

## Value

A BCOO or dense array.
