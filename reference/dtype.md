# Canonical dtype name

Accepts dtype strings (`"float32"`, `"int32"`, `"bool"`, ...) and the
generic aliases `"float"`, `"int"`, `"uint"`, `"complex"` (which resolve
to the default types, 32-bit unless x64 is enabled), or a jax value
(returns its dtype).

## Usage

``` r
dtype(dtype)
```

## Arguments

- dtype:

  A dtype name or a jax value.

## Value

The canonical dtype string.
