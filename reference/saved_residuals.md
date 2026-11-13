# Residuals saved for reverse mode differentiation

Ports of `jax.ad_checkpoint.saved_residuals` and
`print_saved_residuals`: the values that reverse mode differentiation of
`f` at the given arguments stores between the forward and the backward
pass, with where each comes from (an argument, a constant or the output
of a primitive). Source locations are not recorded by jaxR and print as
`unknown`.

## Usage

``` r
saved_residuals(f, ...)

print_saved_residuals(f, ...)
```

## Arguments

- f:

  Function.

- ...:

  Arguments of `f`.

## Value

`saved_residuals()` returns a list of `list(aval, src)`;
`print_saved_residuals()` prints one line per residual and returns it
invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
print_saved_residuals(function(x) sin(x) * cos(x), 1)
} # }
```
