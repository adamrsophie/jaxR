# Copy an array

Port of `jnp.copy`. Arrays are immutable, so the copy is the same value.

## Usage

``` r
jnp_copy(a, order = NULL)
```

## Arguments

- a:

  Array.

- order:

  Memory layout; only `NULL` or `"K"` is supported.

## Value

An array.
