# Symbolic zeros

Ports of `jax.custom_derivatives.SymbolicZero` (a zero tangent known
only by its abstract value; [`dim()`](https://rdrr.io/r/base/dim.html)
gives its shape and `$aval` its
[`ShapeDtypeStruct()`](https://adamrsophie.github.io/jaxR/reference/ShapeDtypeStruct.md))
and `jax.custom_derivatives.zero_from_primal` (zero tangents for the
leaves of a pytree, as arrays or as symbolic zeros).

## Usage

``` r
SymbolicZero(aval)

zero_from_primal(val, symbolic_zeros = FALSE)
```

## Arguments

- aval:

  A
  [`ShapeDtypeStruct()`](https://adamrsophie.github.io/jaxR/reference/ShapeDtypeStruct.md)
  (the tangent type).

- val:

  A pytree of arrays.

- symbolic_zeros:

  Return `SymbolicZero` objects instead of zero arrays.

## Value

`SymbolicZero()` returns an object of class `jax_SymbolicZero`;
`zero_from_primal()` a pytree with the structure of `val`.

## Examples

``` r
z <- SymbolicZero(ShapeDtypeStruct(c(2L, 3L), "float32"))
dim(z)
#> [1] 2 3
if (FALSE) { # \dontrun{
zero_from_primal(list(a = c(1, 2), b = 3L))
} # }
```
