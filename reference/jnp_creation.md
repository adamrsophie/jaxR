# Arrays of zeros, ones, or a fill value

Arrays of zeros, ones, or a fill value

## Usage

``` r
jnp_zeros(shape, dtype = NULL)

jnp_ones(shape, dtype = NULL)

jnp_full(shape, fill_value, dtype = NULL)

jnp_zeros_like(a, dtype = NULL, shape = NULL)

jnp_ones_like(a, dtype = NULL, shape = NULL)

jnp_full_like(a, fill_value, dtype = NULL, shape = NULL)
```

## Arguments

- shape:

  Integer vector.

- dtype:

  Dtype (default float).

- fill_value:

  Scalar.

- a:

  Array whose shape and dtype are used.
