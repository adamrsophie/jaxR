# One-hot encoding

Port of `jax.nn.one_hot`. Each index in `x` (0-based) becomes a vector
of zeros of length `num_classes` with a one at the index; indices
outside `[0, num_classes)` give all zeros.

## Usage

``` r
nn_one_hot(x, num_classes, dtype = NULL, axis = -1L, out_sharding = NULL)
```

## Arguments

- x:

  Integer array of indices (other dtypes give a deprecation warning, as
  in JAX).

- num_classes:

  Number of classes (a concrete integer).

- dtype:

  Output dtype (default float).

- axis:

  0-based position of the new axis in the result, or the name of a
  mapped axis (`vmap(axis_name = )`), in which case `num_classes` must
  equal its size.

- out_sharding:

  Must be `NULL` (sharding is not supported).

## Value

An array with one more dimension than `x`.

## Examples

``` r
if (FALSE) { # \dontrun{
nn_one_hot(jnp_array(c(0L, 1L, 2L)), 3L)
} # }
```
