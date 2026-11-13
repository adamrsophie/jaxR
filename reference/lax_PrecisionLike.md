# Canonical precision specification

Port of `jax.lax.PrecisionLike` (a union type in JAX) as the function
that canonicalizes such a value (`lax.canonicalize_precision`): `NULL`,
a precision name or a pair of names, a
[`lax_DotAlgorithm()`](https://adamrsophie.github.io/jaxR/reference/lax_DotAlgorithm.md)
or a preset of
[lax_DotAlgorithmPreset](https://adamrsophie.github.io/jaxR/reference/lax_enums.md)
(or its name).

## Usage

``` r
lax_PrecisionLike(precision = NULL)
```

## Arguments

- precision:

  A precision like value.

## Value

`NULL`, a pair of names from
[lax_Precision](https://adamrsophie.github.io/jaxR/reference/lax_enums.md),
a `jax_DotAlgorithm` or a `jax_DotAlgorithmPreset`.

## Examples

``` r
lax_PrecisionLike("highest")
#> [1] "HIGHEST" "HIGHEST"
lax_PrecisionLike(c("default", "high"))
#> [1] "DEFAULT" "HIGH"   
```
