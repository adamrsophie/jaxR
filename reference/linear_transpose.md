# Transpose a linear function

Port of `jax.linear_transpose`.

## Usage

``` r
linear_transpose(fun, ...)
```

## Arguments

- fun:

  A linear function.

- ...:

  Primal example arguments (only shapes and dtypes are used).

## Value

A function mapping output cotangents (with the structure, shapes and
dtypes of the output of `fun`) to a list of input cotangents.
