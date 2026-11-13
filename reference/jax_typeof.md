# The abstract type of a value

Port of `jax.typeof`: the `ShapedArray` (shape, dtype and weak type) of
an array, tracer or R value. R numbers are weakly typed (`~float32[]`).

## Usage

``` r
jax_typeof(x)
```

## Arguments

- x:

  A value.

## Value

A `jax_aval`.
