# Evenly spaced values

Port of `jnp.arange` (half-open interval, 0-based like NumPy). The dtype
is the promoted type of the arguments; a whole valued R number counts as
an integer here, as R has no integer literal without the `L` suffix
(`jnp_arange(5)` is an integer array, `jnp_arange(0.5, 3)` a float one).
Pass `dtype` to choose.

## Usage

``` r
jnp_arange(start, stop = NULL, step = NULL, dtype = NULL)
```

## Arguments

- start, stop, step:

  Range specification; `jnp_arange(n)` gives `0..n-1`.

- dtype:

  Optional dtype.
