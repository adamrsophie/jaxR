# Lower a function to StableHLO text

Analogue of `jax.jit(f).lower(*args).as_text()` (generic op form).

## Usage

``` r
lower_text(fun, ..., static_argnums = integer(0))
```

## Arguments

- fun:

  Function.

- ...:

  Example arguments (arrays or
  [`ShapeDtypeStruct()`](https://adamrsophie.github.io/jaxR/reference/ShapeDtypeStruct.md)).

- static_argnums:

  0-based static argument positions.

## Value

A character scalar with the module text.
