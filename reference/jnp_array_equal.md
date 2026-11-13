# Test arrays for equality

Ports of `jnp.array_equal` (same shape and elements) and
`jnp.array_equiv` (broadcastable and equal elements).

## Usage

``` r
jnp_array_equal(a1, a2, equal_nan = FALSE)

jnp_array_equiv(a1, a2)
```

## Arguments

- a1, a2:

  Arrays.

- equal_nan:

  Whether NaN values compare equal.

## Value

A boolean scalar array.
