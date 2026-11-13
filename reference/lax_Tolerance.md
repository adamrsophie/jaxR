# Tolerance of a unary function

Port of `jax.lax.Tolerance`: at most two of the tolerances may be set
(`atol` and `rtol`, or `atol` and `ulps`).

## Usage

``` r
lax_Tolerance(atol = 0, rtol = 0, ulps = 0L)
```

## Arguments

- atol, rtol:

  Nonnegative absolute and relative tolerances.

- ulps:

  Nonnegative tolerance in units in the last place.

## Value

An object of class `jax_Tolerance`.

## Examples

``` r
lax_Tolerance(atol = 1e-6)
#> $atol
#> [1] 1e-06
#> 
#> $rtol
#> [1] 0
#> 
#> $ulps
#> [1] 0
#> 
#> attr(,"class")
#> [1] "jax_Tolerance"
```
