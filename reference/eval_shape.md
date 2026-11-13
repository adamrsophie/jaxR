# Evaluate a function abstractly

Port of `jax.eval_shape`.

## Usage

``` r
eval_shape(fun, ...)
```

## Arguments

- fun:

  Function.

- ...:

  Arguments (arrays or
  [`ShapeDtypeStruct()`](https://adamrsophie.github.io/jaxR/reference/ShapeDtypeStruct.md)
  values).

## Value

A pytree of
[`ShapeDtypeStruct()`](https://adamrsophie.github.io/jaxR/reference/ShapeDtypeStruct.md)
values.
