# Vandermonde matrix

Port of `jnp.vander`.

## Usage

``` r
jnp_vander(x, N = NULL, increasing = FALSE)
```

## Arguments

- x:

  One dimensional array.

- N:

  Number of columns (default `length(x)`).

- increasing:

  Powers increase from left to right.

## Value

An array of shape `(length(x), N)`.
