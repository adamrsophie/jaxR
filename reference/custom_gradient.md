# Define a function with a custom gradient

Port of `jax.custom_gradient`: `fun` returns `list(value, vjp)` where
`vjp(g)` maps the output cotangent to the input cotangents (a list with
one entry per argument, or the cotangent itself when `fun` has a single
argument). Built on
[`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md);
the values `vjp` closes over are saved as residuals.

## Usage

``` r
custom_gradient(fun)
```

## Arguments

- fun:

  A function returning `list(value, vjp)`.

## Value

A function returning `value`, differentiated with `vjp`.

## Examples

``` r
if (FALSE) { # \dontrun{
f <- custom_gradient(function(x) list(x^2, function(g) g * x))
grad(f)(3)  # 3, not 6
} # }
```
