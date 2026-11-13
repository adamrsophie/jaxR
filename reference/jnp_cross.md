# Cross product of 3-vectors

Port of `jnp.cross` (only vectors of length 3 are supported, as in JAX).

## Usage

``` r
jnp_cross(a, b, axisa = -1L, axisb = -1L, axisc = -1L, axis = NULL)
```

## Arguments

- a, b:

  Arrays whose vector axes have length 3.

- axisa, axisb, axisc:

  0-based vector axes of `a`, `b` and the result.

- axis:

  If given, overrides the three axes.

## Value

An array.
