# Trapezoidal integration

Port of `jnp.trapezoid`.

## Usage

``` r
jnp_trapezoid(y, x = NULL, dx = 1, axis = -1L)
```

## Arguments

- y:

  Array of values.

- x:

  Optional sample points (default spacing `dx`).

- dx:

  Spacing when `x` is `NULL`.

- axis:

  Axis of integration (0-based).

## Value

An array.
