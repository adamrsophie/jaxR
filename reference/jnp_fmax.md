# Elementwise maximum and minimum ignoring NaN

Ports of `jnp.fmax` and `jnp.fmin`: when one value is NaN the other is
returned.

## Usage

``` r
jnp_fmax(x1, x2)

jnp_fmin(x1, x2)
```

## Arguments

- x1, x2:

  Arrays.

## Value

An array.
