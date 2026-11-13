# Sampled dense dot product

Port of `jax.experimental.sparse.bcoo_dot_general_sampled`: the values
of `lax_dot_general(A, B, dimension_numbers)` at `indices`.

## Usage

``` r
experimental_sparse_bcoo_dot_general_sampled(A, B, indices, dimension_numbers)
```

## Arguments

- A, B:

  Dense arrays.

- indices:

  BCOO indices.

- dimension_numbers:

  As for
  [`lax_dot_general()`](https://adamrsophie.github.io/jaxR/reference/lax_dot_general.md).

## Value

The data of a BCOO array with the given indices.
