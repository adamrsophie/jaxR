# Integrate along an axis with the trapezoid rule

Port of `jax.scipy.integrate.trapezoid` (the same function as
`jnp.trapezoid`).

## Usage

``` r
scipy_integrate_trapezoid(y, x = NULL, dx = 1, axis = -1L)
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
