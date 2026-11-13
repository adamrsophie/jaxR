# Evaluate a polynomial

Port of `jax.lax.polynomial`: Horner's rule for
`coeffs[[1]] + x * coeffs[[2]] + x^2 * coeffs[[3]] + ...` (coefficients
from the lowest degree to the highest). Derivatives with respect to `x`
are lower degree polynomials.

## Usage

``` r
lax_polynomial(x, coeffs, unroll = NULL)
```

## Arguments

- x:

  Array.

- coeffs:

  List (or vector) of coefficients; arrays must have the rank of `x` or
  be scalars.

- unroll:

  Number of terms unrolled before a scan is used (default 8).

## Value

An array.

## Examples

``` r
if (FALSE) { # \dontrun{
lax_polynomial(3, c(1, 5, 2))  # 34
} # }
```
