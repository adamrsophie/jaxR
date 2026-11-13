# Flatten a pytree of arrays into one vector

Port of `jax.flatten_util.ravel_pytree`.

## Usage

``` r
ravel_pytree(pytree)
```

## Arguments

- pytree:

  A pytree of arrays.

## Value

A list with `flat` (1-D array) and `unravel` (function).
